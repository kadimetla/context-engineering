"""Episodic Memory Store — CoALA Tier 2.

This module records timestamped session events and recalls them via
Park et al.'s scoring formula:

    total = α_recency · recency
          + α_importance · importance
          + α_relevance · relevance

Pedagogical intent
------------------
This file is the class anchor for "episodic memory." Read it top-to-bottom
and you should see, in order:

1. SQLite schema  — the simplest possible event log (one table, 8 columns).
2. log()          — write an event with optional LLM-scored importance.
3. recall()       — the Park et al. retrieval signature, with per-event
                    score breakdown so students can SEE why a memory surfaced.
4. recent()       — straight time-ordered listing (no scoring).
5. stats()        — aggregates for the CoALA overview resource.
6. clear()        — for class-demo resets.

Design simplifications (deliberate, flagged with `CoALA NOTE`):

- Relevance uses a **bag-of-words cosine**, not embeddings. Three lines of
  code, zero embedding spend per recall. Swap-in for embedding-based recall
  is a 3-line edit at `_relevance()`. Real production agents use the latter.
- No bi-temporal edges (Zep/Graphiti). Only `created_at`. Recency decay
  alone demonstrates the temporal idea adequately.
- No write-side conflict handling. Events are append-only. Demonstration
  trade — production agents need invalidation/dedup.

Reuses verbatim from scratchpad_store.py:
- thread-local SQLite connection + RLock pattern
- tiktoken cl100k_base encoder helper
- LLM client construction (Azure / OpenAI fallback)
- Singleton + reset
"""

import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config import settings
from app.models.episodic import (
    EpisodicEvent,
    EpisodicRecallResult,
    EpisodicScoreBreakdown,
    EpisodicStats,
    EventKind,
    VALID_EVENT_KINDS,
)


# =============================================================================
# UTILITIES
# =============================================================================
# CoALA NOTE: relevance uses bag-of-words cosine. To upgrade to embeddings,
# replace _tokenize and _cosine with calls to memory.semantic_search().

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Dict[str, int]:
    """Return lowercase term-frequency dict for bag-of-words cosine."""
    counts: Dict[str, int] = {}
    for tok in _TOKEN_RE.findall(text.lower()):
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Cosine similarity between two term-frequency dicts. Returns 0..1."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# =============================================================================
# EPISODIC STORE
# =============================================================================


class EpisodicStore:
    """SQLite-backed episodic event log with recency × importance × relevance recall.

    Thread-safety: same pattern as scratchpad_store — thread-local connections
    plus a write-lock for INSERT/DELETE serialization.
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path is None:
            self.db_path = settings.episodic_path
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path

        self._local = threading.local()
        self._write_lock = threading.RLock()

        self._init_db()

    # ------------------------------------------------------------------------
    # CONNECTION / SCHEMA
    # ------------------------------------------------------------------------

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        yield self._local.conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id           TEXT PRIMARY KEY,
                    session_id   TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    summary      TEXT NOT NULL,
                    content      TEXT NOT NULL DEFAULT '',
                    importance   REAL NOT NULL DEFAULT 0.3,
                    created_at   TEXT NOT NULL,
                    provenance   TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ep_created ON events(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ep_session ON events(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ep_kind ON events(kind)")
            conn.commit()

    @staticmethod
    def _generate_id() -> str:
        return f"ep-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------------
    # IMPORTANCE SCORING (write-time, optional LLM)
    # ------------------------------------------------------------------------
    # CoALA NOTE: Park et al. assigned importance via an LLM call. We do the
    # same when an LLM is configured, with a tiny prompt. If no LLM, callers
    # pass `importance` explicitly; otherwise we default to 0.3 (low).

    async def _score_importance(self, summary: str) -> float:
        """Call LLM to score importance 0.0-1.0. Falls back to 0.3 on error."""
        if not settings.has_llm_config:
            return 0.3

        try:
            from langchain_openai import AzureChatOpenAI, ChatOpenAI

            if settings.azure_openai_endpoint:
                llm = AzureChatOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                    azure_deployment=settings.azure_openai_deployment,
                    api_version=settings.azure_openai_api_version,
                    temperature=0,
                    max_tokens=8,
                )
            else:
                llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=8,
                )

            prompt = (
                "On a scale of 0.0 (mundane) to 1.0 (highly memorable / pivotal), "
                "rate the importance of this event for a robotics-engineering agent. "
                "Output ONLY the number, nothing else.\n\n"
                f"Event: {summary}"
            )
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()
            value = float(raw.split()[0])
            return max(0.0, min(1.0, value))
        except Exception as e:
            logger.debug("Importance scoring failed (non-fatal): %s", e)
            return 0.3

    # ------------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------------

    async def log(
        self,
        session_id: str,
        kind: EventKind | str,
        summary: str,
        content: str = "",
        importance: Optional[float] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> EpisodicEvent:
        """Append one event to episodic memory.

        If importance is None and an LLM is configured, calls _score_importance.
        Otherwise defaults to 0.3.
        """
        if isinstance(kind, str):
            if kind not in VALID_EVENT_KINDS:
                raise ValueError(
                    f"Invalid event kind: {kind!r}. Must be one of {sorted(VALID_EVENT_KINDS)}"
                )
            kind = EventKind(kind)

        if importance is None:
            importance = await self._score_importance(summary)
        importance = max(0.0, min(1.0, float(importance)))

        now = datetime.now(timezone.utc).isoformat()
        event_id = self._generate_id()
        provenance = provenance or {}
        provenance_json = json.dumps(provenance)

        with self._write_lock:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT INTO events
                       (id, session_id, kind, summary, content, importance, created_at, provenance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (event_id, session_id, kind.value, summary, content,
                     importance, now, provenance_json),
                )
                conn.commit()

        return EpisodicEvent(
            id=event_id,
            session_id=session_id,
            kind=kind,
            summary=summary,
            content=content,
            importance=importance,
            created_at=now,
            provenance=provenance,
        )

    # ------------------------------------------------------------------------
    # RECALL — Park et al. recency × importance × relevance
    # ------------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        session_id: Optional[str] = None,
    ) -> EpisodicRecallResult:
        """Retrieve the top-k most relevant events for `query`.

        The score breakdown is returned alongside the events so the class can
        see exactly WHY each memory surfaced.
        """
        # Resolve weights from settings if not overridden
        w_recency = (weights or {}).get("recency", settings.episodic_weight_recency)
        w_importance = (weights or {}).get("importance", settings.episodic_weight_importance)
        w_relevance = (weights or {}).get("relevance", settings.episodic_weight_relevance)
        half_life = settings.episodic_recency_half_life_hours

        # Pull all events (or filtered by session) — fine for a class app.
        # Production scale would page or pre-filter via a candidate stage.
        with self._get_connection() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY created_at DESC"
                ).fetchall()

        if not rows:
            return EpisodicRecallResult(
                events=[],
                scores=[],
                weights={
                    "recency": w_recency,
                    "importance": w_importance,
                    "relevance": w_relevance,
                },
                half_life_hours=half_life,
            )

        now = datetime.now(timezone.utc)
        query_terms = _tokenize(query)

        scored: List[Tuple[EpisodicEvent, EpisodicScoreBreakdown]] = []
        for row in rows:
            event = self._row_to_event(row)

            # Recency: 0.5 ** (hours_since / half_life)
            try:
                created = datetime.fromisoformat(event.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                hours_since = max(0.0, (now - created).total_seconds() / 3600.0)
                recency = 0.5 ** (hours_since / half_life) if half_life > 0 else 0.0
            except Exception:
                recency = 0.0

            # Relevance: bag-of-words cosine
            event_terms = _tokenize(f"{event.summary} {event.content}")
            relevance = _cosine(query_terms, event_terms)

            total = (
                w_recency * recency
                + w_importance * event.importance
                + w_relevance * relevance
            )

            breakdown = EpisodicScoreBreakdown(
                event_id=event.id,
                recency=round(recency, 4),
                importance=round(event.importance, 4),
                relevance=round(relevance, 4),
                total=round(total, 4),
            )
            scored.append((event, breakdown))

        scored.sort(key=lambda pair: pair[1].total, reverse=True)
        top = scored[:k]

        return EpisodicRecallResult(
            events=[e for e, _ in top],
            scores=[s for _, s in top],
            weights={
                "recency": w_recency,
                "importance": w_importance,
                "relevance": w_relevance,
            },
            half_life_hours=half_life,
        )

    # ------------------------------------------------------------------------
    # SIMPLE LISTING
    # ------------------------------------------------------------------------

    def recent(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[EpisodicEvent]:
        """Time-ordered listing — newest first. No scoring."""
        with self._get_connection() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def since(self, minutes: int) -> List[EpisodicEvent]:
        """Events created within the last `minutes` minutes."""
        cutoff = datetime.now(timezone.utc).timestamp() - minutes * 60
        # ISO timestamps sort lexicographically when in UTC, but to be safe
        # we filter in Python after a coarse SQL pull.
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT 500"
            ).fetchall()

        out: List[EpisodicEvent] = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["created_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts.timestamp() >= cutoff:
                    out.append(self._row_to_event(r))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------------
    # STATS / CLEAR
    # ------------------------------------------------------------------------

    def stats(self) -> EpisodicStats:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS event_count,
                    COUNT(DISTINCT session_id) AS session_count,
                    MIN(created_at) AS oldest,
                    MAX(created_at) AS newest
                FROM events
                """
            ).fetchone()
            kind_rows = conn.execute(
                "SELECT kind, COUNT(*) AS cnt FROM events GROUP BY kind"
            ).fetchall()

        return EpisodicStats(
            event_count=row["event_count"] or 0,
            session_count=row["session_count"] or 0,
            by_kind={r["kind"]: r["cnt"] for r in kind_rows},
            oldest=row["oldest"],
            newest=row["newest"],
            db_path=str(self.db_path),
        )

    def clear(self, session_id: Optional[str] = None) -> int:
        """Clear events. Returns count cleared. session_id=None clears all."""
        with self._write_lock:
            with self._get_connection() as conn:
                if session_id:
                    cur = conn.execute(
                        "DELETE FROM events WHERE session_id = ?", (session_id,)
                    )
                else:
                    cur = conn.execute("DELETE FROM events")
                conn.commit()
                return cur.rowcount

    # ------------------------------------------------------------------------
    # ROW MAPPING / CLEANUP
    # ------------------------------------------------------------------------

    def _row_to_event(self, row: sqlite3.Row) -> EpisodicEvent:
        try:
            provenance = json.loads(row["provenance"]) if row["provenance"] else {}
        except Exception:
            provenance = {}
        return EpisodicEvent(
            id=row["id"],
            session_id=row["session_id"],
            kind=EventKind(row["kind"]),
            summary=row["summary"],
            content=row["content"] or "",
            importance=row["importance"],
            created_at=row["created_at"],
            provenance=provenance,
        )

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# SINGLETON
# =============================================================================

_episodic_store: Optional[EpisodicStore] = None
_episodic_lock = threading.Lock()


def get_episodic_store() -> EpisodicStore:
    """Get the singleton episodic store. Thread-safe double-checked locking."""
    global _episodic_store
    if _episodic_store is None:
        with _episodic_lock:
            if _episodic_store is None:
                _episodic_store = EpisodicStore()
    return _episodic_store


def reset_episodic_store() -> None:
    """Reset the singleton (useful for tests)."""
    global _episodic_store
    with _episodic_lock:
        if _episodic_store is not None:
            _episodic_store.close()
        _episodic_store = EpisodicStore()
