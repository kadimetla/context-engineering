"""Unit tests for the persistent SQLite-backed Scratchpad Memory Store.

Tests:
- Write with and without LLM minimization (mocked)
- Write with and without LLM enrichment on ingest (mocked)
- Read returns cached content (no LLM calls)
- Clear operations by subject and clear-all
- Stats computation from SQLite aggregation
- Context injection formatting and budget
- Persistence across store instances
- Backfill enrichments
- Edge cases: empty content, special characters, concurrent access
"""

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from app.models.scratchpad import VALID_SCRATCHPAD_PREDICATES
from app.adapters.scratchpad_store import ScratchpadStore


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp SQLite database path."""
    return tmp_path / "test_scratchpad.db"


@pytest.fixture
def store(tmp_db):
    """Create a fresh scratchpad store with a temp DB."""
    return ScratchpadStore(db_path=tmp_db)


# =============================================================================
# TEST: WRITE WITHOUT MINIMIZATION
# =============================================================================


class TestWriteWithoutMinimization:
    """Tests for writing entries with minimize=False, enrich=False."""

    @pytest.mark.asyncio
    async def test_write_preserves_content(self, store):
        """Verify content is stored verbatim when minimize=False."""
        content = "WRN-00006 has thermal issues when running hydraulics at full load"
        result = await store.write(
            subject="WRN-00006",
            predicate="observed",
            object_="thermal_system",
            content=content,
            minimize=False,
            enrich=False,
        )

        assert result.success is True
        assert result.entry.content == content

    @pytest.mark.asyncio
    async def test_write_no_minimize_tokens_equal(self, store):
        """Verify original_tokens == minimized_tokens when minimize=False."""
        result = await store.write(
            subject="test",
            predicate="observed",
            object_="target",
            content="some content here",
            minimize=False,
            enrich=False,
        )

        assert result.entry.original_tokens == result.entry.minimized_tokens
        assert result.entry.original_tokens > 0

    @pytest.mark.asyncio
    async def test_write_no_minimize_no_original_content_field(self, store):
        """Verify original_content is None when minimize=False (content unchanged)."""
        result = await store.write(
            subject="test",
            predicate="observed",
            object_="target",
            content="test content",
            minimize=False,
            enrich=False,
        )

        assert result.entry.original_content is None

    @pytest.mark.asyncio
    async def test_write_tokens_saved_zero(self, store):
        """Verify tokens_saved is 0 when minimize=False."""
        result = await store.write(
            subject="test",
            predicate="observed",
            object_="target",
            content="test content",
            minimize=False,
            enrich=False,
        )

        assert result.tokens_saved == 0


# =============================================================================
# TEST: WRITE WITH MINIMIZATION (MOCKED LLM)
# =============================================================================


class TestWriteWithMinimization:
    """Tests for writing entries with LLM minimization (mocked)."""

    @pytest.mark.asyncio
    async def test_write_minimize_stores_minimized_content(self, store):
        """Verify minimized content is stored when LLM minimization succeeds."""
        original = "WRN-00006 has thermal issues when running hydraulics at full load capacity"
        minimized = "WRN-00006 thermal issues hydraulics full load"

        with patch.object(store, "_minimize_content", new_callable=AsyncMock) as mock_min:
            mock_min.return_value = (minimized, 14, 8)

            result = await store.write(
                subject="WRN-00006",
                predicate="observed",
                object_="thermal_system",
                content=original,
                minimize=True,
                enrich=False,
            )

            assert result.success is True
            assert result.entry.content == minimized
            assert result.entry.original_content == original
            assert result.entry.original_tokens == 14
            assert result.entry.minimized_tokens == 8

    @pytest.mark.asyncio
    async def test_write_minimize_reports_tokens_saved(self, store):
        """Verify tokens_saved is correctly calculated."""
        with patch.object(store, "_minimize_content", new_callable=AsyncMock) as mock_min:
            mock_min.return_value = ("short", 20, 5)

            result = await store.write(
                subject="test",
                predicate="observed",
                object_="target",
                content="a much longer piece of content that will be minimized",
                minimize=True,
                enrich=False,
            )

            assert result.tokens_saved == 15  # 20 - 5


# =============================================================================
# TEST: WRITE WITH ENRICHMENT ON INGEST
# =============================================================================


class TestWriteWithEnrichment:
    """Tests for writing entries with LLM enrichment on ingest (mocked)."""

    @pytest.mark.asyncio
    async def test_write_with_enrich_persists_enriched_content(self, store):
        """Verify enriched content is persisted to SQLite on write."""
        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("Expanded context with implications", 8)

            result = await store.write(
                subject="WRN-00006",
                predicate="observed",
                object_="thermal_system",
                content="thermal issues",
                minimize=False,
                enrich=True,
            )

            assert result.success is True
            assert result.entry.enriched_content == "Expanded context with implications"
            assert result.entry.enriched_tokens == 8
            mock_enrich.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_without_enrich_no_enriched_content(self, store):
        """Verify enriched_content is None when enrich=False."""
        result = await store.write(
            subject="test",
            predicate="observed",
            object_="target",
            content="test content",
            minimize=False,
            enrich=False,
        )

        assert result.entry.enriched_content is None
        assert result.entry.enriched_tokens == 0

    @pytest.mark.asyncio
    async def test_write_enrich_graceful_degradation(self, store):
        """Verify write succeeds even if enrichment fails."""
        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = (None, 0)  # LLM unavailable

            result = await store.write(
                subject="test",
                predicate="observed",
                object_="target",
                content="content",
                minimize=False,
                enrich=True,
            )

            assert result.success is True
            assert result.entry.enriched_content is None


# =============================================================================
# TEST: READ (returns cached content, no LLM calls)
# =============================================================================


class TestRead:
    """Tests for reading entries from SQLite."""

    @pytest.mark.asyncio
    async def test_read_returns_cached_content(self, store):
        """Verify read returns content without LLM calls."""
        await store.write("sub", "observed", "obj", "raw content", minimize=False, enrich=False)

        result = await store.read()

        assert result.total == 1
        assert result.entries[0].content == "raw content"

    @pytest.mark.asyncio
    async def test_read_sorts_newest_first(self, store):
        """Verify entries are sorted newest first."""
        await store.write("sub1", "observed", "obj", "first", minimize=False, enrich=False)
        await store.write("sub2", "observed", "obj", "second", minimize=False, enrich=False)

        result = await store.read()

        assert result.total == 2
        assert result.entries[0].subject == "sub2"
        assert result.entries[1].subject == "sub1"

    @pytest.mark.asyncio
    async def test_read_filter_by_subject(self, store):
        """Verify filtering by subject."""
        await store.write("WRN-001", "observed", "obj", "c1", minimize=False, enrich=False)
        await store.write("WRN-002", "observed", "obj", "c2", minimize=False, enrich=False)

        result = await store.read(subject="WRN-001")

        assert result.total == 1
        assert result.entries[0].subject == "WRN-001"

    @pytest.mark.asyncio
    async def test_read_filter_by_predicate(self, store):
        """Verify filtering by predicate."""
        await store.write("sub", "observed", "obj", "c1", minimize=False, enrich=False)
        await store.write("sub", "inferred", "obj", "c2", minimize=False, enrich=False)

        result = await store.read(predicate="inferred")

        assert result.total == 1
        assert result.entries[0].predicate == "inferred"

    @pytest.mark.asyncio
    async def test_read_returns_enriched_content_from_cache(self, store):
        """Verify read returns enriched_content persisted on write."""
        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("enriched text", 5)

            await store.write("sub", "observed", "obj", "brief", minimize=False, enrich=True)

        # Read without any mocks — should return cached enriched content
        result = await store.read()

        assert result.entries[0].enriched_content == "enriched text"


# =============================================================================
# TEST: CLEAR
# =============================================================================


class TestClear:
    """Tests for clearing entries."""

    @pytest.mark.asyncio
    async def test_clear_by_subject_removes_matching(self, store):
        """Verify clear by subject removes only matching entries."""
        await store.write("WRN-001", "observed", "obj", "c1", minimize=False, enrich=False)
        await store.write("WRN-002", "observed", "obj", "c2", minimize=False, enrich=False)
        await store.write("WRN-001", "inferred", "obj", "c3", minimize=False, enrich=False)

        result = store.clear(subject="WRN-001")

        assert result.cleared_count == 2
        stats = store.stats()
        assert stats.entry_count == 1

    @pytest.mark.asyncio
    async def test_clear_all(self, store):
        """Verify clear without subject clears all entries."""
        await store.write("WRN-001", "observed", "obj", "c1", minimize=False, enrich=False)
        await store.write("WRN-002", "observed", "obj", "c2", minimize=False, enrich=False)

        result = store.clear()

        assert result.cleared_count == 2
        stats = store.stats()
        assert stats.entry_count == 0

    @pytest.mark.asyncio
    async def test_clear_nonexistent_subject(self, store):
        """Verify clearing a non-existent subject returns 0."""
        await store.write("WRN-001", "observed", "obj", "c1", minimize=False, enrich=False)

        result = store.clear(subject="DOES-NOT-EXIST")
        assert result.cleared_count == 0


# =============================================================================
# TEST: STATS
# =============================================================================


class TestStats:
    """Tests for stats() from SQLite aggregation."""

    @pytest.mark.asyncio
    async def test_stats_token_counts(self, store):
        """Verify stats accurately sums token counts."""
        await store.write("s1", "observed", "o1", "hello world", minimize=False, enrich=False)
        await store.write("s2", "inferred", "o2", "another entry", minimize=False, enrich=False)

        stats = store.stats()

        assert stats.entry_count == 2
        assert stats.total_original_tokens > 0
        assert stats.total_minimized_tokens > 0
        assert stats.total_original_tokens == stats.total_minimized_tokens
        assert stats.tokens_saved == 0
        assert stats.savings_percentage == 0.0

    @pytest.mark.asyncio
    async def test_stats_enrichment_counts(self, store):
        """Verify enrichment counts are tracked."""
        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("enriched", 3)
            await store.write("s1", "observed", "o1", "c1", minimize=False, enrich=True)

        await store.write("s2", "inferred", "o2", "c2", minimize=False, enrich=False)

        stats = store.stats()

        assert stats.enriched_count == 1
        assert stats.unenriched_count == 1

    @pytest.mark.asyncio
    async def test_stats_predicate_distribution(self, store):
        """Verify stats tracks predicate distribution."""
        await store.write("s1", "observed", "o1", "c1", minimize=False, enrich=False)
        await store.write("s2", "observed", "o2", "c2", minimize=False, enrich=False)
        await store.write("s3", "inferred", "o3", "c3", minimize=False, enrich=False)

        stats = store.stats()

        assert stats.predicate_counts["observed"] == 2
        assert stats.predicate_counts["inferred"] == 1

    @pytest.mark.asyncio
    async def test_stats_oldest_newest_entries(self, store):
        """Verify stats tracks oldest and newest timestamps."""
        await store.write("s1", "observed", "o1", "first", minimize=False, enrich=False)
        await store.write("s2", "observed", "o2", "second", minimize=False, enrich=False)

        stats = store.stats()

        assert stats.oldest_entry is not None
        assert stats.newest_entry is not None
        assert stats.oldest_entry <= stats.newest_entry

    @pytest.mark.asyncio
    async def test_stats_empty_scratchpad(self, store):
        """Verify stats for empty scratchpad."""
        stats = store.stats()

        assert stats.entry_count == 0
        assert stats.total_original_tokens == 0
        assert stats.enriched_count == 0
        assert stats.oldest_entry is None
        assert stats.db_path is not None


# =============================================================================
# TEST: CONTEXT INJECTION
# =============================================================================


class TestContextInjection:
    """Tests for get_context_for_injection formatting and budget."""

    @pytest.mark.asyncio
    async def test_injection_format(self, store):
        """Verify context lines follow [predicate] subject -> object_: content format."""
        await store.write("WRN-001", "observed", "thermal", "has issues", minimize=False, enrich=False)

        lines, tokens = store.get_context_for_injection()

        assert len(lines) == 1
        assert lines[0] == "[observed] WRN-001 -> thermal: has issues"
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_injection_respects_budget(self, store):
        """Verify injection stops adding entries when budget is exceeded."""
        for i in range(20):
            await store.write(
                f"entry{i}", "observed", "obj",
                f"content number {i} with extra text to use tokens",
                minimize=False, enrich=False,
            )

        lines, tokens = store.get_context_for_injection(token_budget=50)

        assert tokens <= 50
        assert len(lines) < 20

    @pytest.mark.asyncio
    async def test_injection_newest_first(self, store):
        """Verify newest entries are injected first."""
        await store.write("oldest", "observed", "obj", "old", minimize=False, enrich=False)
        await store.write("newest", "observed", "obj", "new", minimize=False, enrich=False)

        lines, _ = store.get_context_for_injection()

        assert "newest" in lines[0]
        assert "oldest" in lines[1]

    @pytest.mark.asyncio
    async def test_injection_prefers_enriched_content(self, store):
        """Verify injection uses enriched_content when available."""
        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("enriched version of the note", 8)
            await store.write("sub", "observed", "obj", "brief", minimize=False, enrich=True)

        lines, _ = store.get_context_for_injection()

        assert "enriched version of the note" in lines[0]

    @pytest.mark.asyncio
    async def test_injection_empty_scratchpad(self, store):
        """Verify empty scratchpad returns empty list."""
        lines, tokens = store.get_context_for_injection()

        assert lines == []
        assert tokens == 0


# =============================================================================
# TEST: PERSISTENCE
# =============================================================================


class TestPersistence:
    """Tests for SQLite persistence across store instances."""

    @pytest.mark.asyncio
    async def test_entries_survive_new_store_instance(self, tmp_db):
        """Verify entries persist across store instances (simulates restart)."""
        store1 = ScratchpadStore(db_path=tmp_db)
        await store1.write("WRN-001", "observed", "obj", "persistent data", minimize=False, enrich=False)
        store1.close()

        store2 = ScratchpadStore(db_path=tmp_db)
        result = await store2.read()

        assert result.total == 1
        assert result.entries[0].content == "persistent data"
        store2.close()

    @pytest.mark.asyncio
    async def test_clear_persists(self, tmp_db):
        """Verify clear operation persists across store instances."""
        store1 = ScratchpadStore(db_path=tmp_db)
        await store1.write("sub", "observed", "obj", "data", minimize=False, enrich=False)
        store1.clear()
        store1.close()

        store2 = ScratchpadStore(db_path=tmp_db)
        stats = store2.stats()
        assert stats.entry_count == 0
        store2.close()


# =============================================================================
# TEST: BACKFILL
# =============================================================================


class TestBackfill:
    """Tests for backfill_enrichments method."""

    @pytest.mark.asyncio
    async def test_backfill_enriches_unenriched_entries(self, store):
        """Verify backfill fills enrichment gaps."""
        await store.write("sub", "observed", "obj", "content", minimize=False, enrich=False)

        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("backfilled enrichment", 5)
            count = await store.backfill_enrichments()

        assert count == 1

        result = await store.read()
        assert result.entries[0].enriched_content == "backfilled enrichment"

    @pytest.mark.asyncio
    async def test_backfill_skips_already_enriched(self, store):
        """Verify backfill skips entries that already have enrichment."""
        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("original enrichment", 5)
            await store.write("sub", "observed", "obj", "content", minimize=False, enrich=True)

        with patch.object(store, "_enrich_content", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = ("new enrichment", 5)
            count = await store.backfill_enrichments()

        assert count == 0  # Already enriched, nothing to backfill


# =============================================================================
# TEST: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_write_empty_content(self, store):
        """Verify writing empty content succeeds."""
        result = await store.write("sub", "observed", "obj", "", minimize=False, enrich=False)

        assert result.success is True
        assert result.entry.content == ""
        assert result.entry.original_tokens == 0

    @pytest.mark.asyncio
    async def test_write_special_characters(self, store):
        """Verify content with special characters is preserved."""
        content = "Temperature: 85\u00b0C, variance: \u00b15%, status: OK\u2713"
        result = await store.write("sub", "observed", "obj", content, minimize=False, enrich=False)

        assert result.success is True
        assert result.entry.content == content

    @pytest.mark.asyncio
    async def test_write_invalid_predicate_returns_failure(self, store):
        """Verify invalid predicate returns failure, not exception."""
        result = await store.write("sub", "INVALID", "obj", "content", minimize=False, enrich=False)

        assert result.success is False
        assert "Invalid predicate" in result.message

    @pytest.mark.asyncio
    async def test_write_empty_subject_returns_failure(self, store):
        """Verify empty subject returns failure."""
        result = await store.write("", "observed", "obj", "content", minimize=False, enrich=False)

        assert result.success is False
        assert "subject" in result.message

    @pytest.mark.asyncio
    async def test_write_whitespace_subject_returns_failure(self, store):
        """Verify whitespace-only subject returns failure."""
        result = await store.write("   ", "observed", "obj", "content", minimize=False, enrich=False)

        assert result.success is False
        assert "subject" in result.message

    @pytest.mark.asyncio
    async def test_write_empty_object_returns_failure(self, store):
        """Verify empty object_ returns failure."""
        result = await store.write("sub", "observed", "", "content", minimize=False, enrich=False)

        assert result.success is False
        assert "object_" in result.message

    @pytest.mark.asyncio
    async def test_write_whitespace_object_returns_failure(self, store):
        """Verify whitespace-only object_ returns failure."""
        result = await store.write("sub", "observed", "   ", "content", minimize=False, enrich=False)

        assert result.success is False
        assert "object_" in result.message

    @pytest.mark.asyncio
    async def test_write_very_long_content(self, store):
        """Verify very long content is handled."""
        content = "word " * 500
        result = await store.write("sub", "observed", "obj", content, minimize=False, enrich=False)

        assert result.success is True
        assert result.entry.original_tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_writes_no_data_loss(self, store):
        """Verify concurrent writes do not lose data."""
        import asyncio

        async def write_entry(i: int):
            return await store.write(
                f"subject{i}", "observed", "obj", f"content {i}",
                minimize=False, enrich=False,
            )

        results = await asyncio.gather(*[write_entry(i) for i in range(10)])

        assert all(r.success for r in results)
        stats = store.stats()
        assert stats.entry_count == 10

    @pytest.mark.asyncio
    async def test_no_token_limit_on_writes(self, store):
        """Verify there is no token budget limiting writes (persistent store)."""
        # Write many entries — none should be evicted
        for i in range(50):
            await store.write(f"sub{i}", "observed", "obj", f"content {i} " * 10, minimize=False, enrich=False)

        stats = store.stats()
        assert stats.entry_count == 50
