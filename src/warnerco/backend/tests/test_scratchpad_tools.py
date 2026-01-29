"""Unit tests for Scratchpad MCP tool functions.

Tests the four MCP tool functions defined in mcp_tools.py:
- warn_scratchpad_write
- warn_scratchpad_read
- warn_scratchpad_clear
- warn_scratchpad_stats

All tests mock the underlying ScratchpadStore to isolate tool logic.
Updated for persistent SQLite-backed scratchpad (no TTL, no budget).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.models.scratchpad import (
    ScratchpadEntry,
    ScratchpadStats,
    ScratchpadWriteResult,
    ScratchpadReadResult,
    ScratchpadClearResult,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_entry():
    """Create a sample ScratchpadEntry."""
    now = datetime.now(timezone.utc)
    return ScratchpadEntry(
        id="sp-test123",
        subject="WRN-00006",
        predicate="observed",
        object_="thermal_system",
        content="thermal issues hydraulics",
        original_content="WRN-00006 has thermal issues when running hydraulics",
        original_tokens=12,
        minimized_tokens=6,
        enriched_content="Expanded: WRN-00006 thermal degradation in hydraulic system under load.",
        enriched_tokens=14,
        created_at=now.isoformat(),
        metadata=None,
    )


@pytest.fixture
def mock_store(sample_entry):
    """Create a mock ScratchpadStore."""
    store = MagicMock()
    store.write = AsyncMock(return_value=ScratchpadWriteResult(
        success=True,
        entry=sample_entry,
        tokens_saved=6,
        message="Stored entry (saved 6 tokens) [enriched]",
    ))
    store.read = AsyncMock(return_value=ScratchpadReadResult(
        entries=[sample_entry],
        total=1,
    ))
    store.clear = MagicMock(return_value=ScratchpadClearResult(
        cleared_count=3,
        message="Cleared 3 entries",
    ))
    store.stats = MagicMock(return_value=ScratchpadStats(
        entry_count=5,
        total_original_tokens=200,
        total_minimized_tokens=140,
        total_enriched_tokens=250,
        tokens_saved=60,
        savings_percentage=30.0,
        enriched_count=4,
        unenriched_count=1,
        predicate_counts={"observed": 3, "inferred": 2},
        oldest_entry="2026-01-29T10:00:00+00:00",
        newest_entry="2026-01-29T10:30:00+00:00",
        db_path="/tmp/test/notes.db",
    ))
    return store


# =============================================================================
# TEST: warn_scratchpad_write
# =============================================================================


class TestWarnScratchpadWrite:
    """Tests for the warn_scratchpad_write MCP tool."""

    @pytest.mark.asyncio
    async def test_write_success(self, mock_store):
        """Verify successful write returns correct tool result."""
        from app.mcp_tools import warn_scratchpad_write as _warn_scratchpad_write_tool
        warn_scratchpad_write = _warn_scratchpad_write_tool.fn

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_write(
                subject="WRN-00006",
                predicate="observed",
                object_="thermal_system",
                content="thermal issues with hydraulics",
                minimize=True,
                enrich=True,
            )

        assert result.success is True
        assert result.entry_id == "sp-test123"
        assert result.tokens_saved == 6
        assert result.original_tokens == 12
        assert result.minimized_tokens == 6
        assert result.enriched is True
        assert result.enriched_tokens == 14

    @pytest.mark.asyncio
    async def test_write_failure(self, mock_store):
        """Verify failed write returns error result."""
        from app.mcp_tools import warn_scratchpad_write as _warn_scratchpad_write_tool
        warn_scratchpad_write = _warn_scratchpad_write_tool.fn

        mock_store.write = AsyncMock(return_value=ScratchpadWriteResult(
            success=False,
            entry=None,
            tokens_saved=0,
            message="Invalid predicate 'bad'",
        ))

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_write(
                subject="test",
                predicate="bad",
                object_="obj",
                content="content",
            )

        assert result.success is False
        assert result.entry_id is None
        assert result.enriched is False

    @pytest.mark.asyncio
    async def test_write_exception_handling(self):
        """Verify exceptions are caught and returned as error result."""
        from app.mcp_tools import warn_scratchpad_write as _warn_scratchpad_write_tool
        warn_scratchpad_write = _warn_scratchpad_write_tool.fn

        mock_store = MagicMock()
        mock_store.write = AsyncMock(side_effect=RuntimeError("Store crashed"))

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_write(
                subject="test",
                predicate="observed",
                object_="obj",
                content="content",
            )

        assert result.success is False
        assert "Error" in result.message


# =============================================================================
# TEST: warn_scratchpad_read
# =============================================================================


class TestWarnScratchpadRead:
    """Tests for the warn_scratchpad_read MCP tool."""

    @pytest.mark.asyncio
    async def test_read_returns_entries(self, mock_store):
        """Verify read returns formatted entries."""
        from app.mcp_tools import warn_scratchpad_read as _warn_scratchpad_read_tool
        warn_scratchpad_read = _warn_scratchpad_read_tool.fn

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_read(subject="WRN-00006")

        assert result.total == 1
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry["subject"] == "WRN-00006"
        assert entry["predicate"] == "observed"

    @pytest.mark.asyncio
    async def test_read_uses_enriched_content_when_available(self, mock_store, sample_entry):
        """Verify enriched_content is used for the primary content field."""
        from app.mcp_tools import warn_scratchpad_read as _warn_scratchpad_read_tool
        warn_scratchpad_read = _warn_scratchpad_read_tool.fn

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_read()

        # Content field should use enriched_content when available
        assert "Expanded" in result.entries[0]["content"]

    @pytest.mark.asyncio
    async def test_read_exception_returns_empty(self):
        """Verify exceptions return empty result."""
        from app.mcp_tools import warn_scratchpad_read as _warn_scratchpad_read_tool
        warn_scratchpad_read = _warn_scratchpad_read_tool.fn

        mock_store = MagicMock()
        mock_store.read = AsyncMock(side_effect=RuntimeError("Read failed"))

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_read()

        assert result.total == 0
        assert result.entries == []


# =============================================================================
# TEST: warn_scratchpad_clear
# =============================================================================


class TestWarnScratchpadClear:
    """Tests for the warn_scratchpad_clear MCP tool."""

    @pytest.mark.asyncio
    async def test_clear_returns_count(self, mock_store):
        """Verify clear returns correct cleared count."""
        from app.mcp_tools import warn_scratchpad_clear as _warn_scratchpad_clear_tool
        warn_scratchpad_clear = _warn_scratchpad_clear_tool.fn

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_clear(subject="WRN-00006")

        assert result.cleared_count == 3
        assert "3 entries" in result.message

    @pytest.mark.asyncio
    async def test_clear_exception_handling(self):
        """Verify exceptions are caught gracefully."""
        from app.mcp_tools import warn_scratchpad_clear as _warn_scratchpad_clear_tool
        warn_scratchpad_clear = _warn_scratchpad_clear_tool.fn

        mock_store = MagicMock()
        mock_store.clear = MagicMock(side_effect=RuntimeError("Clear failed"))

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_clear()

        assert result.cleared_count == 0
        assert "Error" in result.message


# =============================================================================
# TEST: warn_scratchpad_stats
# =============================================================================


class TestWarnScratchpadStats:
    """Tests for the warn_scratchpad_stats MCP tool."""

    @pytest.mark.asyncio
    async def test_stats_propagates_all_fields(self, mock_store):
        """Verify all stat fields are propagated correctly."""
        from app.mcp_tools import warn_scratchpad_stats as _warn_scratchpad_stats_tool
        warn_scratchpad_stats = _warn_scratchpad_stats_tool.fn

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_stats()

        assert result.entry_count == 5
        assert result.total_original_tokens == 200
        assert result.total_minimized_tokens == 140
        assert result.total_enriched_tokens == 250
        assert result.tokens_saved == 60
        assert result.savings_percentage == 30.0
        assert result.enriched_count == 4
        assert result.unenriched_count == 1
        assert result.predicate_counts == {"observed": 3, "inferred": 2}
        assert result.db_path == "/tmp/test/notes.db"

    @pytest.mark.asyncio
    async def test_stats_exception_returns_defaults(self):
        """Verify exceptions return default stats."""
        from app.mcp_tools import warn_scratchpad_stats as _warn_scratchpad_stats_tool
        warn_scratchpad_stats = _warn_scratchpad_stats_tool.fn

        mock_store = MagicMock()
        mock_store.stats = MagicMock(side_effect=RuntimeError("Stats failed"))

        with patch("app.adapters.scratchpad_store.get_scratchpad_store", return_value=mock_store):
            result = await warn_scratchpad_stats()

        assert result.entry_count == 0
        assert result.enriched_count == 0
        assert result.db_path == "unknown"
