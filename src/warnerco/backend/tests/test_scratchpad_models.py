"""Unit tests for Scratchpad Memory Pydantic models.

Tests ScratchpadEntry, ScratchpadStats, ScratchpadWriteResult,
ScratchpadReadResult, ScratchpadClearResult, and predicate vocabulary.

Updated for persistent SQLite-backed scratchpad (no TTL, no budget).
"""

import pytest
from datetime import datetime, timezone

from app.models.scratchpad import (
    ScratchpadEntry,
    ScratchpadStats,
    ScratchpadWriteResult,
    ScratchpadReadResult,
    ScratchpadClearResult,
    SCRATCHPAD_PREDICATES,
    VALID_SCRATCHPAD_PREDICATES,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_entry():
    """Create a sample ScratchpadEntry for reuse across tests."""
    now = datetime.now(timezone.utc)
    return ScratchpadEntry(
        id="sp-abc123def456",
        subject="WRN-00006",
        predicate="observed",
        object_="thermal_system",
        content="thermal issues with hydraulics",
        original_content="WRN-00006 has thermal issues when running hydraulics",
        original_tokens=12,
        minimized_tokens=8,
        enriched_content="WRN-00006 exhibits thermal degradation in hydraulic subsystem under load.",
        enriched_tokens=15,
        created_at=now.isoformat(),
        metadata={"source": "user"},
    )


@pytest.fixture
def sample_stats():
    """Create a sample ScratchpadStats for reuse."""
    return ScratchpadStats(
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
    )


# =============================================================================
# TEST: PREDICATE VOCABULARY
# =============================================================================


class TestPredicateVocabulary:
    """Tests for the predicate vocabulary constants."""

    def test_all_expected_predicates_exist(self):
        """Verify all seven predicates are defined."""
        expected = {
            "observed", "inferred", "relevant_to", "summarized_as",
            "contradicts", "supersedes", "depends_on",
        }
        assert VALID_SCRATCHPAD_PREDICATES == expected

    def test_predicate_dict_maps_to_self(self):
        """Verify SCRATCHPAD_PREDICATES maps each key to itself."""
        for key, value in SCRATCHPAD_PREDICATES.items():
            assert key == value

    def test_predicate_count(self):
        """Verify exactly 7 predicates are defined."""
        assert len(VALID_SCRATCHPAD_PREDICATES) == 7
        assert len(SCRATCHPAD_PREDICATES) == 7


# =============================================================================
# TEST: SCRATCHPAD ENTRY
# =============================================================================


class TestScratchpadEntry:
    """Tests for ScratchpadEntry model construction and validation."""

    def test_entry_construction_with_all_fields(self, sample_entry):
        """Verify entry can be constructed with all fields."""
        assert sample_entry.id == "sp-abc123def456"
        assert sample_entry.subject == "WRN-00006"
        assert sample_entry.predicate == "observed"
        assert sample_entry.object_ == "thermal_system"
        assert sample_entry.content == "thermal issues with hydraulics"
        assert sample_entry.original_content == "WRN-00006 has thermal issues when running hydraulics"
        assert sample_entry.original_tokens == 12
        assert sample_entry.minimized_tokens == 8
        assert sample_entry.enriched_content is not None
        assert sample_entry.enriched_tokens == 15
        assert sample_entry.metadata == {"source": "user"}

    def test_entry_defaults(self):
        """Verify default values for optional fields."""
        now = datetime.now(timezone.utc).isoformat()
        entry = ScratchpadEntry(
            id="sp-test",
            subject="test",
            predicate="observed",
            object_="target",
            content="test content",
            created_at=now,
        )
        assert entry.original_content is None
        assert entry.original_tokens == 0
        assert entry.minimized_tokens == 0
        assert entry.enriched_content is None
        assert entry.enriched_tokens == 0
        assert entry.metadata is None

    def test_entry_timestamp_is_iso_format(self, sample_entry):
        """Verify timestamp can be parsed as ISO format."""
        created = datetime.fromisoformat(sample_entry.created_at)
        assert isinstance(created, datetime)

    def test_entry_accepts_any_predicate_string(self):
        """Verify ScratchpadEntry model itself does not validate predicates.

        Predicate validation happens in the store write method, not the model.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry = ScratchpadEntry(
            id="sp-test",
            subject="test",
            predicate="any_string_is_fine",
            object_="target",
            content="test",
            created_at=now,
        )
        assert entry.predicate == "any_string_is_fine"

    def test_entry_serialization(self, sample_entry):
        """Verify entry can be serialized to dict."""
        data = sample_entry.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == "sp-abc123def456"
        assert data["subject"] == "WRN-00006"
        assert "object_" in data

    def test_entry_with_empty_content(self):
        """Verify entry can be created with empty content."""
        now = datetime.now(timezone.utc).isoformat()
        entry = ScratchpadEntry(
            id="sp-empty",
            subject="test",
            predicate="observed",
            object_="target",
            content="",
            created_at=now,
        )
        assert entry.content == ""

    def test_entry_with_unicode_content(self):
        """Verify entry handles unicode content."""
        now = datetime.now(timezone.utc).isoformat()
        entry = ScratchpadEntry(
            id="sp-unicode",
            subject="test",
            predicate="observed",
            object_="target",
            content="Thermal reading: 85\u00b0C with \u00b15% variance",
            created_at=now,
        )
        assert "\u00b0C" in entry.content
        assert "\u00b1" in entry.content

    def test_entry_without_enrichment(self):
        """Verify entry works without enrichment (enriched_content=None)."""
        now = datetime.now(timezone.utc).isoformat()
        entry = ScratchpadEntry(
            id="sp-no-enrich",
            subject="test",
            predicate="observed",
            object_="target",
            content="basic content",
            created_at=now,
        )
        assert entry.enriched_content is None
        assert entry.enriched_tokens == 0


# =============================================================================
# TEST: SCRATCHPAD STATS
# =============================================================================


class TestScratchpadStats:
    """Tests for ScratchpadStats model."""

    def test_stats_construction(self, sample_stats):
        """Verify stats can be constructed with all fields."""
        assert sample_stats.entry_count == 5
        assert sample_stats.total_original_tokens == 200
        assert sample_stats.total_minimized_tokens == 140
        assert sample_stats.total_enriched_tokens == 250
        assert sample_stats.tokens_saved == 60
        assert sample_stats.savings_percentage == 30.0
        assert sample_stats.enriched_count == 4
        assert sample_stats.unenriched_count == 1
        assert sample_stats.db_path == "/tmp/test/notes.db"

    def test_stats_savings_percentage_calculation(self):
        """Verify savings percentage is stored correctly."""
        stats = ScratchpadStats(
            entry_count=1,
            total_original_tokens=100,
            total_minimized_tokens=75,
            total_enriched_tokens=120,
            tokens_saved=25,
            savings_percentage=25.0,
            enriched_count=1,
            unenriched_count=0,
            db_path="/tmp/test.db",
        )
        assert stats.savings_percentage == 25.0

    def test_stats_zero_entries(self):
        """Verify stats handles zero entries."""
        stats = ScratchpadStats(
            entry_count=0,
            total_original_tokens=0,
            total_minimized_tokens=0,
            total_enriched_tokens=0,
            tokens_saved=0,
            savings_percentage=0.0,
            enriched_count=0,
            unenriched_count=0,
            db_path="/tmp/empty.db",
        )
        assert stats.entry_count == 0
        assert stats.savings_percentage == 0.0
        assert stats.oldest_entry is None
        assert stats.newest_entry is None

    def test_stats_predicate_counts(self, sample_stats):
        """Verify predicate counts are tracked."""
        assert sample_stats.predicate_counts["observed"] == 3
        assert sample_stats.predicate_counts["inferred"] == 2

    def test_stats_enrichment_counts(self, sample_stats):
        """Verify enrichment counts add up."""
        assert sample_stats.enriched_count + sample_stats.unenriched_count == sample_stats.entry_count

    def test_stats_tokens_saved_consistency(self, sample_stats):
        """Verify tokens_saved equals original minus minimized."""
        assert (
            sample_stats.tokens_saved
            == sample_stats.total_original_tokens - sample_stats.total_minimized_tokens
        )


# =============================================================================
# TEST: RESULT MODELS
# =============================================================================


class TestScratchpadWriteResult:
    """Tests for ScratchpadWriteResult model."""

    def test_successful_write_result(self, sample_entry):
        """Verify successful write result construction."""
        result = ScratchpadWriteResult(
            success=True,
            entry=sample_entry,
            tokens_saved=4,
            message="Stored entry (saved 4 tokens)",
        )
        assert result.success is True
        assert result.entry is not None
        assert result.tokens_saved == 4
        assert "saved 4 tokens" in result.message

    def test_failed_write_result(self):
        """Verify failed write result construction."""
        result = ScratchpadWriteResult(
            success=False,
            entry=None,
            tokens_saved=0,
            message="Invalid predicate 'bad'",
        )
        assert result.success is False
        assert result.entry is None
        assert result.tokens_saved == 0

    def test_write_result_defaults(self):
        """Verify default values for optional fields."""
        result = ScratchpadWriteResult(
            success=True,
            message="Stored entry",
        )
        assert result.entry is None
        assert result.tokens_saved == 0


class TestScratchpadReadResult:
    """Tests for ScratchpadReadResult model."""

    def test_read_result_with_entries(self, sample_entry):
        """Verify read result with entries."""
        result = ScratchpadReadResult(
            entries=[sample_entry],
            total=1,
        )
        assert len(result.entries) == 1
        assert result.total == 1

    def test_read_result_empty(self):
        """Verify empty read result."""
        result = ScratchpadReadResult(
            entries=[],
            total=0,
        )
        assert result.entries == []
        assert result.total == 0


class TestScratchpadClearResult:
    """Tests for ScratchpadClearResult model."""

    def test_clear_result(self):
        """Verify clear result construction."""
        result = ScratchpadClearResult(
            cleared_count=5,
            message="Cleared 5 entries",
        )
        assert result.cleared_count == 5
        assert "5 entries" in result.message

    def test_clear_result_zero(self):
        """Verify clear result with zero entries cleared."""
        result = ScratchpadClearResult(
            cleared_count=0,
            message="Cleared 0 entries",
        )
        assert result.cleared_count == 0
