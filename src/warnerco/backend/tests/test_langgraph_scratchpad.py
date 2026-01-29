"""Unit tests for the LangGraph inject_scratchpad node.

Tests the inject_scratchpad function from app/langgraph/flow.py (Node 3)
which retrieves scratchpad entries and injects them into the LangGraph state.
Also tests compress_context integration with scratchpad context.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.langgraph.flow import inject_scratchpad, compress_context, GraphState


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def base_state():
    """Create a base GraphState for testing."""
    return {
        "query": "What are the thermal issues with WRN-00006?",
        "filters": None,
        "top_k": 5,
        "intent": None,
        "graph_context": [],
        "scratchpad_context": [],
        "scratchpad_token_count": 0,
        "candidates": [],
        "compressed_context": "",
        "response": {},
        "error": None,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "timings": {},
    }


@pytest.fixture
def mock_scratchpad_store():
    """Create a mock scratchpad store with entries."""
    store = MagicMock()
    store.get_context_for_injection = MagicMock(return_value=(
        [
            "[observed] WRN-00006 -> thermal_system: thermal issues hydraulics",
            "[inferred] WRN-00006 -> cooling: may need cooling system check",
        ],
        25,  # token count
    ))
    return store


@pytest.fixture
def mock_empty_scratchpad():
    """Create a mock empty scratchpad store."""
    store = MagicMock()
    store.get_context_for_injection = MagicMock(return_value=([], 0))
    return store


# =============================================================================
# TEST: INJECT SCRATCHPAD NODE
# =============================================================================


class TestInjectScratchpad:
    """Tests for the inject_scratchpad LangGraph node."""

    @pytest.mark.asyncio
    async def test_inject_with_entries(self, base_state, mock_scratchpad_store):
        """Verify scratchpad entries are injected into state."""
        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=mock_scratchpad_store,
        ):
            result = await inject_scratchpad(base_state)

        assert len(result["scratchpad_context"]) == 2
        assert result["scratchpad_token_count"] == 25
        assert "[observed]" in result["scratchpad_context"][0]
        assert "WRN-00006" in result["scratchpad_context"][0]

    @pytest.mark.asyncio
    async def test_inject_with_empty_scratchpad(self, base_state, mock_empty_scratchpad):
        """Verify empty scratchpad produces empty context."""
        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=mock_empty_scratchpad,
        ):
            result = await inject_scratchpad(base_state)

        assert result["scratchpad_context"] == []
        assert result["scratchpad_token_count"] == 0

    @pytest.mark.asyncio
    async def test_inject_passes_query_context(self, base_state, mock_scratchpad_store):
        """Verify the query is passed to get_context_for_injection."""
        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=mock_scratchpad_store,
        ):
            await inject_scratchpad(base_state)

        # Verify query_context was passed
        call_kwargs = mock_scratchpad_store.get_context_for_injection.call_args
        assert call_kwargs.kwargs.get("query_context") == base_state["query"]

    @pytest.mark.asyncio
    async def test_inject_records_timing(self, base_state, mock_scratchpad_store):
        """Verify timing is recorded in state."""
        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=mock_scratchpad_store,
        ):
            result = await inject_scratchpad(base_state)

        assert "inject_scratchpad" in result["timings"]
        assert result["timings"]["inject_scratchpad"] >= 0

    @pytest.mark.asyncio
    async def test_inject_handles_store_error_gracefully(self, base_state):
        """Verify scratchpad errors do not break the pipeline."""
        mock_store = MagicMock()
        mock_store.get_context_for_injection = MagicMock(
            side_effect=RuntimeError("Store unavailable")
        )

        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=mock_store,
        ):
            result = await inject_scratchpad(base_state)

        # Should pass through cleanly with empty context
        assert result["scratchpad_context"] == []
        assert result["scratchpad_token_count"] == 0
        assert "inject_scratchpad" in result["timings"]

    @pytest.mark.asyncio
    async def test_inject_uses_settings_budget(self, base_state, mock_scratchpad_store):
        """Verify injection uses the configured inject budget from settings."""
        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=mock_scratchpad_store,
        ):
            with patch("app.langgraph.flow.settings") as mock_settings:
                mock_settings.scratchpad_inject_budget = 750
                await inject_scratchpad(base_state)

        call_kwargs = mock_scratchpad_store.get_context_for_injection.call_args
        assert call_kwargs.kwargs.get("token_budget") == 750


# =============================================================================
# TEST: ENTRY FORMATTING
# =============================================================================


class TestEntryFormatting:
    """Tests for scratchpad entry formatting in context lines."""

    @pytest.mark.asyncio
    async def test_format_predicate_subject_object_content(self, base_state):
        """Verify format: [predicate] subject -> object_: content."""
        store = MagicMock()
        store.get_context_for_injection = MagicMock(return_value=(
            ["[observed] WRN-001 -> thermal: has issues"],
            10,
        ))

        with patch(
            "app.adapters.scratchpad_store.get_scratchpad_store",
            return_value=store,
        ):
            result = await inject_scratchpad(base_state)

        line = result["scratchpad_context"][0]
        assert line.startswith("[observed]")
        assert "WRN-001" in line
        assert "->" in line
        assert "thermal" in line
        assert "has issues" in line


# =============================================================================
# TEST: COMPRESS CONTEXT WITH SCRATCHPAD
# =============================================================================


class TestCompressContextWithScratchpad:
    """Tests for compress_context node integration with scratchpad."""

    def test_compress_includes_scratchpad_header(self, base_state):
        """Verify compress_context includes Session Memory header when scratchpad has entries."""
        base_state["scratchpad_context"] = [
            "[observed] WRN-001 -> thermal: has issues",
        ]

        result = compress_context(base_state)

        assert "=== Session Memory (Scratchpad) ===" in result["compressed_context"]

    def test_compress_includes_scratchpad_entries(self, base_state):
        """Verify compress_context includes scratchpad entry text."""
        base_state["scratchpad_context"] = [
            "[observed] WRN-001 -> thermal: has issues",
            "[inferred] WRN-001 -> cooling: needs check",
        ]

        result = compress_context(base_state)

        assert "[observed] WRN-001 -> thermal: has issues" in result["compressed_context"]
        assert "[inferred] WRN-001 -> cooling: needs check" in result["compressed_context"]

    def test_compress_no_scratchpad_header_when_empty(self, base_state):
        """Verify no scratchpad header when scratchpad is empty."""
        base_state["scratchpad_context"] = []

        result = compress_context(base_state)

        assert "Session Memory (Scratchpad)" not in result["compressed_context"]

    def test_compress_scratchpad_appears_before_graph_context(self, base_state):
        """Verify scratchpad context appears before graph context."""
        base_state["scratchpad_context"] = [
            "[observed] WRN-001 -> thermal: has issues",
        ]
        base_state["graph_context"] = [
            "Entity: WRN-001 (robot)",
        ]

        result = compress_context(base_state)

        ctx = result["compressed_context"]
        scratchpad_pos = ctx.find("Session Memory (Scratchpad)")
        graph_pos = ctx.find("Knowledge Graph Context")
        assert scratchpad_pos < graph_pos
