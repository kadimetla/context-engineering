"""CoALA Memory Consolidation — the "sleep cycle".

This module promotes scratchpad observations + recent episodic events into
durable SEMANTIC memory (the vector store) via MCP Sampling.

CoALA framing
-------------
- Working / Episodic memory are CHEAP, FAST, and TRANSIENT.
- Semantic memory is EXPENSIVE to embed/index but DURABLE.
- Consolidation = the moment an agent decides "this short-term observation
  is durable enough to commit to long-term knowledge."

Pedagogical simplifications (deliberate):

1. ADD-only. We do NOT implement Mem0's full ADD/UPDATE/DELETE/NOOP loop.
   Each consolidation cycle just appends new facts. Production agents need
   conflict resolution; that's homework.

2. Facts ride the existing Schematic shape, not a separate "fact" collection.
   They are tagged with `category="consolidated_fact"`, `model="MEMORY"`, and
   ID prefix `FACT-`, so dashboards can filter them out (or in) trivially.
   The cleaner alternative — a dedicated facts collection — would require
   editing all three vector adapters; that's out of scope for the class.

3. Sampling uses the same two-pass pattern as warn_explain_schematic in
   mcp_tools.py: try structured `result_type` first, fall back to plain-text
   JSON. One sampling idiom across the whole codebase.

The act of consolidating is itself recorded as an episodic OBSERVATION, so
students see that consolidation becomes its own memory.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.adapters import get_memory_store
from app.adapters.episodic_store import get_episodic_store
from app.adapters.scratchpad_store import get_scratchpad_store
from app.models.episodic import ConsolidationResult, EventKind
from app.models.schematic import Schematic, SchematicStatus

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURED-OUTPUT SCHEMA FOR THE LLM
# =============================================================================
# CoALA NOTE: Pydantic doubles as a contract. The LLM is asked to return a
# list of these objects, and FastMCP either validates them via result_type
# or we parse the JSON manually in the fallback path.


class ExtractedFact(BaseModel):
    fact: str = Field(description="A single durable, generalizable fact about WARNERCO schematics")
    supporting_ids: List[str] = Field(
        default_factory=list,
        description="IDs of source memories (scratchpad entries or episodic events) that support this fact",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="LLM confidence in this fact (0-1)"
    )


class ExtractedFacts(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list)


# =============================================================================
# CONSOLIDATION
# =============================================================================


def _build_extraction_prompt(
    scratchpad_lines: List[str],
    episodic_lines: List[str],
    max_facts: int,
) -> str:
    """Assemble the prompt fed to ctx.sample()."""
    sp_block = "\n".join(scratchpad_lines) if scratchpad_lines else "(none)"
    ep_block = "\n".join(episodic_lines) if episodic_lines else "(none)"
    return (
        "You are extracting durable, generalizable facts from an agent's recent\n"
        "session memory. The goal: identify knowledge worth promoting from short-term\n"
        "to long-term semantic memory.\n\n"
        f"Extract AT MOST {max_facts} facts. Skip mundane events. Each fact must be\n"
        "(a) generalizable beyond the immediate query, and\n"
        "(b) supported by the source memories below.\n\n"
        "=== Scratchpad (working memory) ===\n"
        f"{sp_block}\n\n"
        "=== Recent Episodic Events ===\n"
        f"{ep_block}\n\n"
        "Output a JSON object with key 'facts' whose value is a list of "
        "{fact, supporting_ids, confidence} objects."
    )


async def _sample_facts(ctx: Any, prompt: str) -> ExtractedFacts:
    """Two-pass sampling — structured first, plain-text JSON fallback.

    Mirrors the pattern in warn_explain_schematic (mcp_tools.py:4417-4466)
    so the class sees one sampling idiom across the codebase.
    """
    system_prompt = (
        "You are a careful knowledge-extraction agent. You distill durable facts "
        "from session memory. You output strict JSON only, no prose."
    )

    # Pass 1: structured output via result_type
    try:
        sampling_result = await ctx.sample(
            messages=prompt,
            system_prompt=system_prompt,
            result_type=ExtractedFacts,
            temperature=0.2,
            max_tokens=1024,
        )
        extracted = getattr(sampling_result, "result", None)
        if extracted is not None:
            return extracted
    except Exception as structured_err:
        logger.debug("Structured sampling failed, falling back: %s", structured_err)

    # Pass 2: plain-text JSON
    schema_hint = ExtractedFacts.model_json_schema()
    fallback_message = (
        f"{prompt}\n\n"
        "IMPORTANT: Respond ONLY with valid JSON (no markdown fences, no preamble) "
        f"matching this schema:\n{json.dumps(schema_hint)}"
    )

    sampling_result = await ctx.sample(
        messages=fallback_message,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=1024,
    )
    raw = sampling_result.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    return ExtractedFacts.model_validate_json(raw)


def _fact_to_schematic(fact: ExtractedFact, session_id: str) -> Schematic:
    """Wrap an extracted fact in the Schematic shape so it lands in the vector store.

    CoALA NOTE: deliberately reusing the Schematic shape avoids forking three
    adapters. Tags + category + id-prefix make these filterable.
    """
    fact_id = f"FACT-{uuid.uuid4().hex[:8].upper()}"
    today = datetime.now(timezone.utc).date().isoformat()
    return Schematic(
        id=fact_id,
        model="MEMORY",
        name="Consolidated Fact",
        component="semantic_memory",
        version="1.0.0",
        category="consolidated_fact",
        status=SchematicStatus.DRAFT,
        summary=fact.fact,
        url=f"memory://facts/{fact_id}",
        tags=["consolidated", "fact", f"session:{session_id}"],
        specifications={
            "provenance": {
                "source": "consolidation",
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trust_level": "agent_extracted",
            },
            "supporting_ids": fact.supporting_ids,
            "confidence": fact.confidence,
        },
        last_verified=today,
    )


async def consolidate_memory(
    ctx: Any,
    since_minutes: int = 60,
    max_facts: int = 5,
    session_id: Optional[str] = None,
) -> ConsolidationResult:
    """Run one consolidation cycle.

    Reads recent scratchpad + episodic memory, asks the LLM (via MCP Sampling)
    to extract durable facts, and writes them into the semantic vector store.

    Args:
        ctx: FastMCP Context (needed for ctx.sample()).
        since_minutes: How far back to look in episodic memory.
        max_facts: Cap on facts to extract per cycle.
        session_id: Optional session label for provenance/tagging.
    """
    started = datetime.now(timezone.utc)
    if session_id is None:
        session_id = f"consolidation-{uuid.uuid4().hex[:6]}"

    # 1) Pull source memories.
    sp_store = get_scratchpad_store()
    ep_store = get_episodic_store()

    sp_read = await sp_store.read()
    sp_lines: List[str] = []
    for entry in sp_read.entries[:30]:  # cap input — keep prompt small
        sp_lines.append(
            f"[{entry.id}] [{entry.predicate}] {entry.subject} -> {entry.object_}: {entry.content}"
        )

    ep_events = ep_store.since(minutes=since_minutes)
    ep_lines = [
        f"[{e.id}] ({e.kind.value}) imp={e.importance:.2f} {e.created_at}: {e.summary}"
        for e in ep_events[:30]
    ]

    if not sp_lines and not ep_lines:
        return ConsolidationResult(
            success=True,
            facts_added=0,
            fact_ids=[],
            elapsed_ms=(datetime.now(timezone.utc) - started).total_seconds() * 1000,
            message="Nothing to consolidate — scratchpad and recent episodic memory are empty.",
        )

    # 2) Ask the LLM.
    prompt = _build_extraction_prompt(sp_lines, ep_lines, max_facts)

    try:
        extracted = await _sample_facts(ctx, prompt)
    except Exception as e:
        return ConsolidationResult(
            success=False,
            facts_added=0,
            fact_ids=[],
            elapsed_ms=(datetime.now(timezone.utc) - started).total_seconds() * 1000,
            message=f"Sampling failed: {e}",
        )

    facts = extracted.facts[:max_facts]
    if not facts:
        return ConsolidationResult(
            success=True,
            facts_added=0,
            fact_ids=[],
            elapsed_ms=(datetime.now(timezone.utc) - started).total_seconds() * 1000,
            message="LLM returned no consolidatable facts.",
        )

    # 3) Write each fact as a synthetic Schematic into the vector store.
    memory = get_memory_store()
    written_ids: List[str] = []
    for f in facts:
        schematic = _fact_to_schematic(f, session_id=session_id)
        try:
            await memory.upsert_schematic(schematic)
            # Best-effort embed/index — JSON store no-ops, Chroma/Azure embed.
            try:
                await memory.embed_and_index(schematic.id)
            except Exception as embed_err:
                logger.debug("embed_and_index non-fatal: %s", embed_err)
            written_ids.append(schematic.id)
        except Exception as write_err:
            logger.warning("Failed to write consolidated fact %s: %s", schematic.id, write_err)

    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000

    # 4) Record the consolidation cycle itself as an episodic OBSERVATION.
    #    (This is the moment students see consolidation as its own memory.)
    try:
        await ep_store.log(
            session_id=session_id,
            kind=EventKind.OBSERVATION,
            summary=f"Consolidation promoted {len(written_ids)} facts to semantic memory",
            content=json.dumps({"fact_ids": written_ids, "elapsed_ms": elapsed_ms}),
            importance=0.5,
            provenance={"source": "consolidate_memory", "trust_level": "system"},
        )
    except Exception as ep_err:
        logger.debug("Episodic log of consolidation cycle failed (non-fatal): %s", ep_err)

    return ConsolidationResult(
        success=True,
        facts_added=len(written_ids),
        fact_ids=written_ids,
        elapsed_ms=elapsed_ms,
        message=f"Consolidated {len(written_ids)} fact(s) from {len(sp_lines)} scratchpad entries and {len(ep_lines)} recent episodes.",
    )
