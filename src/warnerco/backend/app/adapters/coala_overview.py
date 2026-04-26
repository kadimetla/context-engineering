"""CoALA Four-Tier Overview Helper.

Builds the JSON snapshot served by the `memory://coala-overview` resource.
This is the class anchor — a single response that shows all four CoALA
memory tiers, what backs each, and the current entry counts.

Reference: Sumers et al. (2024), "Cognitive Architectures for Language Agents"

Tiers:
  - working    — current decision cycle (LangGraph state + scratchpad)
  - episodic   — timestamped session events with importance + recency
  - semantic   — durable, generalizable facts (vector store)
  - procedural — versioned skills (MCP Prompts)
"""

from __future__ import annotations

from typing import Any, Dict


# Static catalog of MCP Prompts that comprise procedural memory.
# Kept in sync with the @mcp.prompt() definitions in mcp_tools.py.
PROCEDURAL_PROMPTS = [
    {
        "name": "diagnostic_prompt",
        "version": "1.0.0",
        "description": "Structured diagnostic analysis for one schematic.",
    },
    {
        "name": "comparison_prompt",
        "version": "1.0.0",
        "description": "Side-by-side comparison of two schematics.",
    },
    {
        "name": "search_strategy_prompt",
        "version": "1.0.0",
        "description": "Coaching prompt — how to search the schematic catalog effectively.",
    },
    {
        "name": "maintenance_report_prompt",
        "version": "1.0.0",
        "description": "Maintenance-report template for a robot model.",
    },
    {
        "name": "schematic_review_prompt",
        "version": "1.0.0",
        "description": "Technical-review checklist for a schematic.",
    },
]


async def build_coala_overview() -> Dict[str, Any]:
    """Build a live four-tier CoALA snapshot.

    Counts come from each tier's own stats() — semantic from MemoryStats,
    episodic from EpisodicStats, working from ScratchpadStats. Procedural
    is a static catalog of registered prompts.
    """
    # Lazy imports — avoid forcing every consumer to load all stores.
    from app.adapters import get_memory_store
    from app.adapters.episodic_store import get_episodic_store
    from app.adapters.scratchpad_store import get_scratchpad_store

    # ---- WORKING (scratchpad + GraphState) -----------------------------------
    try:
        sp = get_scratchpad_store()
        sp_stats = sp.stats()
        working_count = sp_stats.entry_count
    except Exception as e:
        sp_stats = None
        working_count = 0

    # ---- EPISODIC ------------------------------------------------------------
    try:
        ep = get_episodic_store()
        ep_stats = ep.stats()
        episodic_count = ep_stats.event_count
        episodic_sessions = ep_stats.session_count
    except Exception:
        ep_stats = None
        episodic_count = 0
        episodic_sessions = 0

    # ---- SEMANTIC (vector store via memory backend) --------------------------
    try:
        mem = get_memory_store()
        mem_stats = await mem.get_memory_stats()
        semantic_count = mem_stats.indexed_count
        backend_name = mem_stats.backend
    except Exception:
        mem_stats = None
        semantic_count = 0
        backend_name = "unknown"

    # ---- BUILD RESPONSE ------------------------------------------------------
    return {
        "framework": "CoALA (Sumers et al. 2024)",
        "description": (
            "Four canonical memory tiers in a Cognitive Architecture for Language Agents. "
            "WARNERCO Schematica implements each tier explicitly so a class can see them as "
            "four distinct things — backed by different stores, written by different paths, "
            "and read by different MCP primitives."
        ),
        "tiers": {
            "working": {
                "what": "Current decision cycle — observations and inferences for THIS session.",
                "backed_by": "LangGraph GraphState (in-memory) + SQLite scratchpad",
                "tools": [
                    "warn_scratchpad_write",
                    "warn_scratchpad_read",
                    "warn_scratchpad_clear",
                    "warn_scratchpad_stats",
                ],
                "resources": [],
                "prompts": [],
                "current_count": working_count,
                "details": (
                    sp_stats.model_dump()
                    if sp_stats is not None
                    else {"error": "scratchpad_store unavailable"}
                ),
            },
            "episodic": {
                "what": (
                    "Specific past events with timestamps. Recalled via "
                    "Park et al. recency × importance × relevance scoring."
                ),
                "backed_by": "SQLite events.db (CoALA Tier 2)",
                "tools": [
                    "warn_episodic_log",
                    "warn_episodic_recall",
                    "warn_episodic_recent",
                    "warn_episodic_stats",
                ],
                "resources": [],
                "prompts": [],
                "current_count": episodic_count,
                "session_count": episodic_sessions,
                "details": (
                    ep_stats.model_dump()
                    if ep_stats is not None
                    else {"error": "episodic_store unavailable"}
                ),
            },
            "semantic": {
                "what": (
                    "Durable, generalizable knowledge — schematics + LLM-extracted facts "
                    "promoted from working/episodic memory by the consolidation cycle."
                ),
                "backed_by": f"vector store ({backend_name})",
                "tools": [
                    "warn_semantic_search",
                    "warn_list_robots",
                    "warn_get_robot",
                    "warn_index_schematic",
                    "warn_consolidate_memory",
                ],
                "resources": ["memory://overview", "memory://architecture"],
                "prompts": [],
                "current_count": semantic_count,
                "details": (
                    mem_stats.model_dump()
                    if mem_stats is not None
                    else {"error": "memory_store unavailable"}
                ),
            },
            "procedural": {
                "what": (
                    "Versioned skills/workflows the agent can invoke. CoALA flags procedural "
                    "writes as the riskiest memory operation, which is why MCP Prompts are "
                    "USER-invoked, not model-invoked."
                ),
                "backed_by": "MCP Prompts registered in mcp_tools.py",
                "tools": [],
                "resources": ["memory://procedural-catalog"],
                "prompts": [p["name"] for p in PROCEDURAL_PROMPTS],
                "current_count": len(PROCEDURAL_PROMPTS),
                "details": {"catalog": PROCEDURAL_PROMPTS},
            },
        },
        "primitive_mapping": {
            "Tools": "read/write all four tiers",
            "Resources": "expose memory state for inspection (memory://*)",
            "Prompts": "procedural memory itself — versioned skills",
            "Sampling": "consolidation cycle (working/episodic -> semantic)",
            "Elicitation": "user confirmation for risky writes",
        },
    }
