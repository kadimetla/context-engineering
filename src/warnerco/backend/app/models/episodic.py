"""Episodic Memory models for WARNERCO Robotics Schematica.

CoALA Tier 2 — Episodic Memory.
==============================
Episodic memory records SPECIFIC PAST EVENTS with temporal indexing — what
the user asked at 14:32, what tool returned what, what the agent observed
between turns. Distinct from semantic memory (generalized facts) and from
working memory (current decision cycle).

Reference: Sumers et al. (2024), "Cognitive Architectures for Language Agents"
           Park et al. (2023), "Generative Agents" — recency × importance × relevance

Pedagogical notes for the class:
- Episodic memory is NOT raw conversation history. Each event has a SUMMARY
  (what should match a recall query) and a CONTENT (full payload for audit).
- IMPORTANCE is set at WRITE time (heuristic or LLM-scored). It never changes.
- RECENCY decays exponentially with elapsed time (half-life setting).
- RELEVANCE is computed at READ time against the current query.
- The retrieval signature exposes per-event score breakdowns so students can
  see WHY a particular memory surfaced.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# EVENT KINDS
# =============================================================================
# CoALA NOTE: episodic events are typed so a class can filter "show me only
# the user turns" or "show me the tool calls" without scanning all content.

class EventKind(str, Enum):
    """The kind of thing that happened in a session."""

    USER_TURN = "user_turn"            # The user said / asked something
    AGENT_RESPONSE = "agent_response"  # The agent replied
    TOOL_CALL = "tool_call"            # A tool was invoked (with args/result)
    OBSERVATION = "observation"        # Mid-pipeline note (e.g., "0 results")


VALID_EVENT_KINDS = {k.value for k in EventKind}


# =============================================================================
# DATA MODELS
# =============================================================================


class EpisodicEvent(BaseModel):
    """A single timestamped event in episodic memory.

    Every field has a CoALA-aligned purpose:

    - id:         unique row id, never reused
    - session_id: groups events into a "run" / conversation
    - kind:       what kind of event this is (see EventKind)
    - summary:    short text — THE recall payload, matched at read time
    - content:    full original payload (query, JSON tool result, etc.)
    - importance: 0.0-1.0 score set at write time (heuristic or LLM)
    - created_at: ISO-8601 UTC timestamp, drives recency decay
    - provenance: {source, trust_level, created_by} for memory-poisoning defense
    """

    id: str = Field(description="Unique event identifier")
    session_id: str = Field(description="Session/conversation grouping id")
    kind: EventKind = Field(description="Kind of event")
    summary: str = Field(description="Short text — the recall payload")
    content: str = Field(default="", description="Full original payload (JSON or text)")
    importance: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Park et al. importance score"
    )
    created_at: str = Field(description="ISO-8601 UTC timestamp")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance tags: source, trust_level, created_by",
    )


class EpisodicScoreBreakdown(BaseModel):
    """Per-event score breakdown — exposed so students see the recall math."""

    event_id: str
    recency: float = Field(description="0.5 ** (hours_since / half_life_hours)")
    importance: float = Field(description="Stored at write time")
    relevance: float = Field(description="Bag-of-words cosine vs. query")
    total: float = Field(description="α_r·recency + α_i·importance + α_l·relevance")


class EpisodicRecallResult(BaseModel):
    """Result of an episodic recall — events ordered by total score, with breakdown."""

    events: List[EpisodicEvent] = Field(default_factory=list)
    scores: List[EpisodicScoreBreakdown] = Field(default_factory=list)
    weights: Dict[str, float] = Field(
        default_factory=dict, description="The α weights used for this recall"
    )
    half_life_hours: float = Field(description="Recency half-life used")


class EpisodicStats(BaseModel):
    """Statistics about episodic memory."""

    event_count: int
    session_count: int
    by_kind: Dict[str, int] = Field(default_factory=dict)
    oldest: Optional[str] = None
    newest: Optional[str] = None
    db_path: str


class ConsolidationResult(BaseModel):
    """Result of a memory-consolidation cycle (CoALA "sleep cycle").

    Consolidation extracts durable facts from scratchpad + episodic memory
    via MCP Sampling and writes them as semantic-memory records.
    """

    success: bool
    facts_added: int = 0
    fact_ids: List[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    message: str = ""
