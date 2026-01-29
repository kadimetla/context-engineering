"""SQLite-backed Scratchpad Memory Store for WARNERCO Robotics Schematica.

This module implements a persistent scratchpad memory that complements the
vector (Chroma) and graph (SQLite + NetworkX) memory layers.

Key Features:
- SQLite-backed persistent storage (entries survive server restarts)
- LLM-powered minimization on write (reduces token usage)
- LLM-powered enrichment on write (expands context, persisted to DB)
- Token tracking using tiktoken for accurate counting
- Thread-safe operations using thread-local SQLite connections
- No TTL, no write budget — entries persist until explicitly deleted

Usage:
    store = get_scratchpad_store()

    # Write with minimization + enrichment (both on ingest)
    entry = await store.write(
        subject="WRN-00006",
        predicate="observed",
        object_="thermal_system",
        content="WRN-00006 has thermal issues when running hydraulics",
        minimize=True,
        enrich=True,
    )

    # Read — returns cached enriched_content (no LLM call)
    entries = await store.read(subject="WRN-00006")

    # Get context for LangGraph injection
    context_lines, token_count = store.get_context_for_injection(token_budget=1500)
"""

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import tiktoken

logger = logging.getLogger(__name__)

from app.config import settings
from app.models.scratchpad import (
    ScratchpadEntry,
    ScratchpadStats,
    ScratchpadWriteResult,
    ScratchpadReadResult,
    ScratchpadClearResult,
    VALID_SCRATCHPAD_PREDICATES,
)


class ScratchpadStore:
    """SQLite-backed triplet store with LLM minimization and enrichment on ingest.

    This store provides persistent memory for observations, inferences, and
    contextual notes. It uses LLM calls on write to:
    - Minimize content (reduce token usage while preserving meaning)
    - Enrich content (expand context for richer retrieval)

    Both minimized and enriched versions are persisted to SQLite. Reads
    return the cached content without any LLM calls.

    Thread Safety:
        Uses thread-local SQLite connections (same pattern as graph_store.py).
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        """Initialize the scratchpad store.

        Args:
            db_path: Path to SQLite database. Defaults to settings.scratchpad_path.
        """
        if db_path is None:
            self.db_path = settings.scratchpad_path
        elif isinstance(db_path, str):
            self.db_path = Path(db_path)
        else:
            self.db_path = db_path

        self._local = threading.local()
        self._write_lock = threading.RLock()

        # Token counting — use cl100k_base encoding (GPT-4/GPT-3.5)
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

        # Initialize database
        self._init_db()

    # =========================================================================
    # DATABASE INFRASTRUCTURE
    # =========================================================================

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a thread-local database connection.

        Yields:
            SQLite connection for the current thread
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent reads
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        yield self._local.conn

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id               TEXT PRIMARY KEY,
                    subject          TEXT NOT NULL,
                    predicate        TEXT NOT NULL,
                    object_          TEXT NOT NULL,
                    content          TEXT NOT NULL,
                    original_content TEXT NOT NULL,
                    enriched_content TEXT,
                    original_tokens  INTEGER NOT NULL DEFAULT 0,
                    minimized_tokens INTEGER NOT NULL DEFAULT 0,
                    enriched_tokens  INTEGER NOT NULL DEFAULT 0,
                    created_at       TEXT NOT NULL,
                    metadata         TEXT
                )
            """)

            # Indexes for efficient lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_subject ON entries(subject)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_predicate ON entries(predicate)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_created_at ON entries(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_subject_predicate ON entries(subject, predicate)")

            conn.commit()

    # =========================================================================
    # TOKEN COUNTING
    # =========================================================================

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Falls back to word-based estimation if tiktoken unavailable.
        """
        if not text:
            return 0

        if self._encoding:
            return len(self._encoding.encode(text))

        # Fallback: rough estimate (words * 1.3)
        return int(len(text.split()) * 1.3)

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique entry ID."""
        return f"sp-{uuid.uuid4().hex[:12]}"

    # =========================================================================
    # LLM OPERATIONS (called on write/ingest only)
    # =========================================================================

    async def _minimize_content(self, content: str) -> Tuple[str, int, int]:
        """Use LLM to minimize content while preserving meaning.

        Returns (minimized_content, original_tokens, minimized_tokens).
        Falls back to truncation if LLM unavailable.
        """
        original_tokens = self._count_tokens(content)

        if settings.has_llm_config:
            try:
                from langchain_openai import AzureChatOpenAI, ChatOpenAI

                if settings.azure_openai_endpoint:
                    llm = AzureChatOpenAI(
                        azure_endpoint=settings.azure_openai_endpoint,
                        api_key=settings.azure_openai_api_key,
                        azure_deployment=settings.azure_openai_deployment,
                        api_version=settings.azure_openai_api_version,
                        temperature=0,
                        max_tokens=100,
                    )
                else:
                    llm = ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model="gpt-4o-mini",
                        temperature=0,
                        max_tokens=100,
                    )

                prompt = (
                    "Minimize this text to its essential meaning in as few words as possible.\n"
                    "Keep key entities, relationships, and facts. Remove filler words.\n"
                    "Output ONLY the minimized text, nothing else.\n\n"
                    f"Text: {content}"
                )

                response = await llm.ainvoke(prompt)
                minimized = response.content.strip()
                minimized_tokens = self._count_tokens(minimized)

                # Only use minimized version if it's actually shorter
                if minimized_tokens < original_tokens:
                    return minimized, original_tokens, minimized_tokens

            except Exception as e:
                logger.warning("Scratchpad minimization error (non-fatal): %s", e)

        # Fallback: truncation to ~75% of original
        if original_tokens > 50:
            words = content.split()
            target_words = int(len(words) * 0.75)
            truncated = " ".join(words[:target_words])
            truncated_tokens = self._count_tokens(truncated)
            return truncated, original_tokens, truncated_tokens

        return content, original_tokens, original_tokens

    async def _enrich_content(self, subject: str, predicate: str, object_: str, content: str) -> Tuple[Optional[str], int]:
        """Use LLM to expand/enrich entry content on ingest.

        Returns (enriched_content, enriched_tokens). Returns (None, 0) if LLM unavailable.
        """
        if not settings.has_llm_config:
            return None, 0

        try:
            from langchain_openai import AzureChatOpenAI, ChatOpenAI

            if settings.azure_openai_endpoint:
                llm = AzureChatOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                    azure_deployment=settings.azure_openai_deployment,
                    api_version=settings.azure_openai_api_version,
                    temperature=0.3,
                    max_tokens=200,
                )
            else:
                llm = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=200,
                )

            prompt = (
                "Expand this brief note into a more detailed explanation.\n"
                "Add relevant context, implications, and connections.\n"
                "Keep it concise but informative (2-3 sentences max).\n\n"
                f"Subject: {subject}\n"
                f"Relationship: {predicate}\n"
                f"Target: {object_}\n"
                f"Note: {content}"
            )

            response = await llm.ainvoke(prompt)
            enriched = response.content.strip()
            enriched_tokens = self._count_tokens(enriched)

            return enriched, enriched_tokens

        except Exception as e:
            logger.warning("Scratchpad enrichment error (non-fatal): %s", e)
            return None, 0

    # =========================================================================
    # WRITE (with minimize + enrich on ingest)
    # =========================================================================

    async def write(
        self,
        subject: str,
        predicate: str,
        object_: str,
        content: str,
        minimize: bool = True,
        enrich: bool = True,
        metadata: Optional[Dict] = None,
    ) -> ScratchpadWriteResult:
        """Store an observation with LLM minimization and enrichment on ingest.

        Both minimized and enriched versions are persisted to SQLite.
        Subsequent reads return cached content — no LLM calls needed.

        Args:
            subject: Entity being described (e.g., "WRN-00006")
            predicate: Cognitive operation type (observed, inferred, etc.)
            object_: Related entity or concept
            content: The text content to store
            minimize: Whether to use LLM to minimize content (default True)
            enrich: Whether to use LLM to enrich content (default True)
            metadata: Optional additional properties

        Returns:
            ScratchpadWriteResult with the created entry and token savings
        """
        # Validate required fields
        if not subject or not subject.strip():
            return ScratchpadWriteResult(
                success=False,
                entry=None,
                tokens_saved=0,
                message="subject is required and cannot be empty",
            )
        if not object_ or not object_.strip():
            return ScratchpadWriteResult(
                success=False,
                entry=None,
                tokens_saved=0,
                message="object_ is required and cannot be empty",
            )
        if not content and content != "":
            return ScratchpadWriteResult(
                success=False,
                entry=None,
                tokens_saved=0,
                message="content is required",
            )

        # Validate predicate
        if predicate not in VALID_SCRATCHPAD_PREDICATES:
            return ScratchpadWriteResult(
                success=False,
                entry=None,
                tokens_saved=0,
                message=f"Invalid predicate '{predicate}'. Must be one of: {', '.join(sorted(VALID_SCRATCHPAD_PREDICATES))}"
            )

        # Step 1: Minimize content
        if minimize:
            minimized_content, original_tokens, minimized_tokens = await self._minimize_content(content)
        else:
            original_tokens = self._count_tokens(content)
            minimized_content = content
            minimized_tokens = original_tokens

        # Step 2: Enrich content (on ingest)
        enriched_content = None
        enriched_tokens = 0
        if enrich:
            enriched_content, enriched_tokens = await self._enrich_content(
                subject, predicate, object_, content
            )

        # Step 3: Persist to SQLite
        now = datetime.now(timezone.utc).isoformat()
        entry_id = self._generate_id()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._write_lock:
            with self._get_connection() as conn:
                conn.execute(
                    """INSERT INTO entries
                       (id, subject, predicate, object_, content, original_content,
                        enriched_content, original_tokens, minimized_tokens,
                        enriched_tokens, created_at, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry_id, subject, predicate, object_,
                        minimized_content, content,
                        enriched_content,
                        original_tokens, minimized_tokens, enriched_tokens,
                        now, metadata_json,
                    ),
                )
                conn.commit()

        entry = ScratchpadEntry(
            id=entry_id,
            subject=subject,
            predicate=predicate,
            object_=object_,
            content=minimized_content,
            original_content=content if minimize and minimized_content != content else None,
            original_tokens=original_tokens,
            minimized_tokens=minimized_tokens,
            enriched_content=enriched_content,
            enriched_tokens=enriched_tokens,
            created_at=now,
            metadata=metadata,
        )

        tokens_saved = original_tokens - minimized_tokens
        msg = f"Stored entry (saved {tokens_saved} tokens)" if tokens_saved > 0 else "Stored entry"
        if enriched_content:
            msg += " [enriched]"

        return ScratchpadWriteResult(
            success=True,
            entry=entry,
            tokens_saved=tokens_saved,
            message=msg,
        )

    # =========================================================================
    # READ (no LLM calls — returns cached content)
    # =========================================================================

    async def read(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        enrich: bool = False,  # DEPRECATED: kept for backward compatibility, ignored
        query_context: Optional[str] = None,  # DEPRECATED: kept for backward compatibility, ignored
    ) -> ScratchpadReadResult:
        """Retrieve entries from persistent scratchpad memory.

        Entries are returned with their cached enriched_content (populated on write).
        No LLM calls are made during reads.

        Args:
            subject: Filter by subject entity (optional)
            predicate: Filter by predicate type (optional)
            enrich: DEPRECATED — enrichment now happens on write. Kept for backward compat.
            query_context: DEPRECATED — enrichment now happens on write. Kept for backward compat.

        Returns:
            ScratchpadReadResult with matching entries
        """
        conditions = []
        params: list = []

        if subject:
            conditions.append("subject = ?")
            params.append(subject)

        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM entries{where_clause} ORDER BY created_at DESC",
                params,
            )
            rows = cursor.fetchall()

        entries = [self._row_to_entry(row) for row in rows]

        return ScratchpadReadResult(
            entries=entries,
            total=len(entries),
        )

    # =========================================================================
    # CLEAR
    # =========================================================================

    def clear(
        self,
        subject: Optional[str] = None,
        older_than_minutes: Optional[int] = None,  # DEPRECATED: kept for backward compat, ignored
    ) -> ScratchpadClearResult:
        """Clear entries from persistent scratchpad memory.

        Args:
            subject: Clear only entries for this subject (optional).
                     If None, ALL entries are cleared.
            older_than_minutes: DEPRECATED — no TTL in persistent store. Kept for backward compat.

        Returns:
            ScratchpadClearResult with count of cleared entries
        """
        with self._write_lock:
            with self._get_connection() as conn:
                if subject:
                    cursor = conn.execute(
                        "DELETE FROM entries WHERE subject = ?", (subject,)
                    )
                else:
                    cursor = conn.execute("DELETE FROM entries")
                conn.commit()
                cleared = cursor.rowcount

        return ScratchpadClearResult(
            cleared_count=cleared,
            message=f"Cleared {cleared} entries",
        )

    # =========================================================================
    # STATS
    # =========================================================================

    def stats(self) -> ScratchpadStats:
        """Get scratchpad statistics from SQLite aggregation.

        Returns:
            ScratchpadStats with token usage, entry counts, and enrichment metrics
        """
        with self._get_connection() as conn:
            # Aggregate stats
            row = conn.execute("""
                SELECT
                    COUNT(*) as entry_count,
                    COALESCE(SUM(original_tokens), 0) as total_original,
                    COALESCE(SUM(minimized_tokens), 0) as total_minimized,
                    COALESCE(SUM(enriched_tokens), 0) as total_enriched,
                    COALESCE(SUM(CASE WHEN enriched_content IS NOT NULL THEN 1 ELSE 0 END), 0) as enriched_count,
                    COALESCE(SUM(CASE WHEN enriched_content IS NULL THEN 1 ELSE 0 END), 0) as unenriched_count,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM entries
            """).fetchone()

            # Predicate counts
            pred_rows = conn.execute(
                "SELECT predicate, COUNT(*) as cnt FROM entries GROUP BY predicate"
            ).fetchall()
            predicate_counts = {r["predicate"]: r["cnt"] for r in pred_rows}

        entry_count = row["entry_count"]
        total_original = row["total_original"]
        total_minimized = row["total_minimized"]
        tokens_saved = total_original - total_minimized
        savings_pct = (tokens_saved / total_original * 100) if total_original > 0 else 0.0

        return ScratchpadStats(
            entry_count=entry_count,
            total_original_tokens=total_original,
            total_minimized_tokens=total_minimized,
            total_enriched_tokens=row["total_enriched"],
            tokens_saved=tokens_saved,
            savings_percentage=round(savings_pct, 1),
            enriched_count=row["enriched_count"],
            unenriched_count=row["unenriched_count"],
            predicate_counts=predicate_counts,
            oldest_entry=row["oldest"],
            newest_entry=row["newest"],
            db_path=str(self.db_path),
        )

    # =========================================================================
    # CONTEXT INJECTION (for LangGraph pipeline)
    # =========================================================================

    def get_context_for_injection(
        self,
        token_budget: Optional[int] = None,
        query_context: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        """Get formatted context lines for LangGraph injection.

        Returns entries formatted as context lines, respecting token budget.
        Entries are sorted by recency (newest first). Prefers enriched_content
        over minimized content when available.

        Format: "[predicate] subject -> object_: content"

        Args:
            token_budget: Maximum tokens to use (default from settings)
            query_context: Optional query for relevance filtering (future use)

        Returns:
            Tuple of (context_lines, total_tokens)
        """
        budget = token_budget or settings.scratchpad_inject_budget

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object_, content, enriched_content FROM entries ORDER BY created_at DESC"
            ).fetchall()

        context_lines: list[str] = []
        total_tokens = 0

        for row in rows:
            # Prefer enriched content for injection
            text = row["enriched_content"] if row["enriched_content"] else row["content"]
            line = f"[{row['predicate']}] {row['subject']} -> {row['object_']}: {text}"
            line_tokens = self._count_tokens(line)

            if total_tokens + line_tokens <= budget:
                context_lines.append(line)
                total_tokens += line_tokens
            else:
                break

        return context_lines, total_tokens

    # =========================================================================
    # BACKFILL (fill enrichment gaps for entries written without LLM)
    # =========================================================================

    async def backfill_enrichments(self, limit: int = 50) -> int:
        """Backfill enrichment for entries that were stored without enrichment.

        Useful when LLM was unavailable during initial write. Processes
        oldest unenriched entries first.

        Args:
            limit: Maximum entries to backfill in one call

        Returns:
            Number of entries enriched
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE enriched_content IS NULL ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()

        enriched_count = 0
        for row in rows:
            enriched_content, enriched_tokens = await self._enrich_content(
                row["subject"], row["predicate"], row["object_"], row["original_content"]
            )
            if enriched_content:
                with self._write_lock:
                    with self._get_connection() as conn:
                        conn.execute(
                            "UPDATE entries SET enriched_content = ?, enriched_tokens = ? WHERE id = ?",
                            (enriched_content, enriched_tokens, row["id"]),
                        )
                        conn.commit()
                enriched_count += 1

        return enriched_count

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _row_to_entry(self, row: sqlite3.Row) -> ScratchpadEntry:
        """Convert a SQLite row to a ScratchpadEntry."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else None
        return ScratchpadEntry(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_=row["object_"],
            content=row["content"],
            original_content=row["original_content"],
            original_tokens=row["original_tokens"],
            minimized_tokens=row["minimized_tokens"],
            enriched_content=row["enriched_content"],
            enriched_tokens=row["enriched_tokens"],
            created_at=row["created_at"],
            metadata=metadata,
        )

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        """Clean up thread-local connection on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_scratchpad_store: Optional[ScratchpadStore] = None
_scratchpad_lock = threading.Lock()


def get_scratchpad_store() -> ScratchpadStore:
    """Get the singleton scratchpad store instance.

    Thread-safe singleton using double-checked locking.
    """
    global _scratchpad_store

    if _scratchpad_store is None:
        with _scratchpad_lock:
            if _scratchpad_store is None:
                _scratchpad_store = ScratchpadStore()

    return _scratchpad_store


def reset_scratchpad_store() -> None:
    """Reset the scratchpad store (useful for testing).

    Closes the existing store and creates a new empty instance.
    """
    global _scratchpad_store

    with _scratchpad_lock:
        if _scratchpad_store is not None:
            _scratchpad_store.close()
        _scratchpad_store = ScratchpadStore()
