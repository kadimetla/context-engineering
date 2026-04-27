"""Integration test for the 9-node LangGraph pipeline."""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# Ensure we're working from the backend directory
BACKEND_DIR = Path(__file__).parent.resolve()
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

# Wipe episodic db before any imports
EPISODIC_DB = BACKEND_DIR / "data" / "episodic" / "events.db"
if EPISODIC_DB.exists():
    EPISODIC_DB.unlink()
# Also wipe wal/shm artifacts
for ext in ("-wal", "-shm"):
    p = EPISODIC_DB.with_name(EPISODIC_DB.name + ext)
    if p.exists():
        p.unlink()


def report(name, passed, evidence):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {evidence}")
    return passed


async def test_1_intents():
    """Test 1: All four intents resolve correctly."""
    from app.langgraph import run_query

    cases = {
        "lookup": "get WRN-00001",  # has "wrn-" and "get " patterns
        "diagnostic": "what's the status of the hydraulic system",
        "analytics": "how many sensors are active",
        "search": "tell me about lidar",
    }

    results = {}
    for expected, query in cases.items():
        resp = await run_query(query, session_id=f"intent-test-{expected}")
        results[expected] = resp["intent"]

    passed = all(results[k] == k for k in cases)
    return report(
        "Test 1 (intents)",
        passed,
        f"got {results}",
    )


async def test_2_episodic_gating():
    """Test 2: Episodic recall is gated to ANALYTICS/DIAGNOSTIC."""
    from app.langgraph import run_query
    from app.adapters.episodic_store import get_episodic_store

    # Pre-populate episodic memory so recall would have something if not gated
    store = get_episodic_store()
    from app.models.episodic import EventKind
    await store.log(
        session_id="gating-prep",
        kind=EventKind.USER_TURN,
        summary="prior turn about sensors and status",
        content="seed",
        importance=0.5,
    )

    lookup_resp = await run_query("get WRN-00001", session_id="gating-lookup")
    search_resp = await run_query("tell me about lidar", session_id="gating-search")
    diag_resp = await run_query("status of sensors", session_id="gating-diag")
    ana_resp = await run_query("how many sensors", session_id="gating-ana")

    lookup_empty = lookup_resp["recalled_episodes"] == []
    search_empty = search_resp["recalled_episodes"] == []
    diag_called = isinstance(diag_resp["recalled_episodes"], list)
    ana_called = isinstance(ana_resp["recalled_episodes"], list)
    # We expect non-empty for diag/ana since we seeded one event
    diag_nonempty = len(diag_resp["recalled_episodes"]) > 0
    ana_nonempty = len(ana_resp["recalled_episodes"]) > 0

    passed = lookup_empty and search_empty and diag_called and ana_called and diag_nonempty and ana_nonempty
    return report(
        "Test 2 (gating)",
        passed,
        f"lookup empty={lookup_empty}, search empty={search_empty}, diag len={len(diag_resp['recalled_episodes'])}, ana len={len(ana_resp['recalled_episodes'])}",
    )


async def test_3_session_id():
    """Test 3: session_id round-trip."""
    from app.langgraph import run_query

    # Explicit session_id
    resp1 = await run_query("get WRN-00001", session_id="test-123")
    explicit_match = resp1["session_id"] == "test-123"

    # Auto-generated
    resp2 = await run_query("get WRN-00001")
    auto_match = resp2["session_id"].startswith("sess-") and len(resp2["session_id"]) == 5 + 8

    passed = explicit_match and auto_match
    return report(
        "Test 3 (session_id)",
        passed,
        f"explicit={resp1['session_id']!r}, auto={resp2['session_id']!r}",
    )


async def test_4_multi_turn():
    """Test 4: Multi-turn coherence with same session_id."""
    from app.adapters.episodic_store import reset_episodic_store, get_episodic_store
    from app.langgraph import run_query

    # Wipe DB physically and reset singleton
    if EPISODIC_DB.exists():
        EPISODIC_DB.unlink()
    for ext in ("-wal", "-shm"):
        p = EPISODIC_DB.with_name(EPISODIC_DB.name + ext)
        if p.exists():
            p.unlink()
    reset_episodic_store()

    sid = "multi-turn-1"
    # Three diagnostic queries
    r1 = await run_query("status of hydraulic sensors", session_id=sid)
    r2 = await run_query("status of motor sensors", session_id=sid)
    r3 = await run_query("status of lidar sensors", session_id=sid)

    n1 = len(r1["recalled_episodes"])
    n2 = len(r2["recalled_episodes"])
    n3 = len(r3["recalled_episodes"])

    # Verify events stored
    store = get_episodic_store()
    events = store.recent(session_id=sid, limit=20)
    user_turn_events = [e for e in events if e.kind.value == "user_turn" and e.session_id == sid]

    passed = n1 == 0 and n2 == 1 and n3 == 2 and len(user_turn_events) == 3
    return report(
        "Test 4 (multi-turn)",
        passed,
        f"recall counts=[{n1},{n2},{n3}], user_turn events for session={len(user_turn_events)}",
    )


async def test_5_importance_heuristic():
    """Test 5: Importance heuristic per intent and error path."""
    from app.adapters.episodic_store import reset_episodic_store, get_episodic_store
    from app.langgraph import run_query
    from app.adapters import get_memory_store

    # Reset
    if EPISODIC_DB.exists():
        EPISODIC_DB.unlink()
    for ext in ("-wal", "-shm"):
        p = EPISODIC_DB.with_name(EPISODIC_DB.name + ext)
        if p.exists():
            p.unlink()
    reset_episodic_store()

    # Diagnostic (no error) -> 0.6
    sid_diag = "imp-diag"
    await run_query("status of hydraulic sensors", session_id=sid_diag)

    # Analytics -> 0.4
    sid_ana = "imp-ana"
    await run_query("how many sensors are deployed", session_id=sid_ana)

    # Search -> 0.3
    sid_search = "imp-search"
    await run_query("tell me about lidar", session_id=sid_search)

    # Lookup -> 0.3
    sid_lookup = "imp-lookup"
    await run_query("get WRN-00001", session_id=sid_lookup)

    # Error path -> 0.8: monkeypatch semantic_search
    memory = get_memory_store()
    original = memory.semantic_search

    async def boom(*args, **kwargs):
        raise RuntimeError("forced failure for test")

    memory.semantic_search = boom
    sid_err = "imp-err"
    try:
        await run_query("tell me anything", session_id=sid_err)
    finally:
        memory.semantic_search = original

    # Inspect events
    store = get_episodic_store()

    def find_imp(sid):
        events = store.recent(session_id=sid, limit=5)
        ut = [e for e in events if e.kind.value == "user_turn"]
        return ut[0].importance if ut else None

    imp_diag = find_imp(sid_diag)
    imp_ana = find_imp(sid_ana)
    imp_search = find_imp(sid_search)
    imp_lookup = find_imp(sid_lookup)
    imp_err = find_imp(sid_err)

    passed = (
        imp_diag == 0.6
        and imp_ana == 0.4
        and imp_search == 0.3
        and imp_lookup == 0.3
        and imp_err == 0.8
    )
    return report(
        "Test 5 (importance)",
        passed,
        f"diag={imp_diag}, ana={imp_ana}, search={imp_search}, lookup={imp_lookup}, err={imp_err}",
    )


async def test_6_compress_block_order():
    """Test 6: compress_context places episodic block between scratchpad and graph."""
    from app.langgraph.flow import compress_context, GraphState, QueryIntent
    from datetime import datetime, timezone

    state: GraphState = {
        "query": "test",
        "filters": None,
        "top_k": 5,
        "session_id": "block-order-test",
        "intent": QueryIntent.DIAGNOSTIC,
        "graph_context": ["GraphLine1", "GraphLine2"],
        "scratchpad_context": ["ScratchLine1"],
        "scratchpad_token_count": 5,
        "recalled_episodes": ["EpisodeLine1", "EpisodeLine2"],
        "candidates": [],
        "compressed_context": "",
        "response": {},
        "error": None,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "timings": {},
    }

    state = compress_context(state)
    ctx = state["compressed_context"]

    has_episodic = "=== Session History (Episodic) ===" in ctx
    has_scratch = "=== Session Memory (Scratchpad) ===" in ctx
    has_graph = "=== Knowledge Graph Context ===" in ctx

    # Order check
    pos_scratch = ctx.find("=== Session Memory (Scratchpad) ===")
    pos_episodic = ctx.find("=== Session History (Episodic) ===")
    pos_graph = ctx.find("=== Knowledge Graph Context ===")

    order_ok = pos_scratch < pos_episodic < pos_graph

    passed = has_episodic and has_scratch and has_graph and order_ok
    return report(
        "Test 6 (block order)",
        passed,
        f"scratch@{pos_scratch}, episodic@{pos_episodic}, graph@{pos_graph}, order_ok={order_ok}",
    )


async def test_7_timing_telemetry():
    """Test 7: Timing telemetry has 8 keys (respond does not log)."""
    from app.langgraph import run_query

    resp = await run_query("status of hydraulic sensors", session_id="timing-test")
    timings = resp["timings"]

    expected_keys = {
        "parse_intent",
        "query_graph",
        "inject_scratchpad",
        "recall_episodes",
        "retrieve",
        "compress_context",
        "reason",
        "log_episode",
    }
    actual_keys = set(timings.keys())
    has_all = expected_keys.issubset(actual_keys)
    no_respond = "respond" not in actual_keys
    count_ok = len(actual_keys) == 8

    passed = has_all and no_respond and count_ok
    return report(
        "Test 7 (timings)",
        passed,
        f"keys={sorted(actual_keys)}, respond_present={'respond' in actual_keys}",
    )


async def test_8_non_fatal_failures():
    """Test 8: Pipeline survives broken episodic store."""
    from app.langgraph import run_query
    import app.langgraph.flow as flow_module
    import app.adapters.episodic_store as ep_module

    # Monkey-patch get_episodic_store to raise
    original = ep_module.get_episodic_store

    def boom():
        raise RuntimeError("episodic store unavailable")

    ep_module.get_episodic_store = boom

    try:
        resp = await run_query(
            "status of hydraulic sensors",
            session_id="broken-episodic",
        )
        completed = resp.get("success") is True or resp.get("intent") is not None
        recalled_empty = resp["recalled_episodes"] == []
    finally:
        ep_module.get_episodic_store = original

    passed = completed and recalled_empty
    return report(
        "Test 8 (non-fatal)",
        passed,
        f"completed={completed}, recalled={resp['recalled_episodes']}",
    )


async def test_9_fallback_path():
    """Test 9: LangGraph fallback path runs all 9 nodes sequentially."""
    from app.langgraph.flow import SchematicaGraph

    # Force fallback by using a fresh instance and skipping the build
    g = SchematicaGraph()
    g._graph = None  # explicit
    # Patch _build_graph to not actually build (simulate ImportError path)
    async def noop_build():
        g._graph = None
    g._build_graph = noop_build

    # Now run — should hit the else branch
    resp = await g.run("status of hydraulic sensors", session_id="fallback-test")
    timings = resp.get("timings", {})
    expected_keys = {
        "parse_intent",
        "query_graph",
        "inject_scratchpad",
        "recall_episodes",
        "retrieve",
        "compress_context",
        "reason",
        "log_episode",
    }
    has_all = expected_keys.issubset(set(timings.keys()))
    has_session = resp.get("session_id") == "fallback-test"

    passed = has_all and has_session
    return report(
        "Test 9 (fallback)",
        passed,
        f"timings_keys_count={len(timings)}, session={resp.get('session_id')}",
    )


async def main():
    results = []
    tests = [
        ("test_1_intents", test_1_intents),
        ("test_2_episodic_gating", test_2_episodic_gating),
        ("test_3_session_id", test_3_session_id),
        ("test_4_multi_turn", test_4_multi_turn),
        ("test_5_importance_heuristic", test_5_importance_heuristic),
        ("test_6_compress_block_order", test_6_compress_block_order),
        ("test_7_timing_telemetry", test_7_timing_telemetry),
        ("test_8_non_fatal_failures", test_8_non_fatal_failures),
        ("test_9_fallback_path", test_9_fallback_path),
    ]
    for name, fn in tests:
        try:
            ok = await fn()
            results.append((name, ok))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{total} passed")


if __name__ == "__main__":
    asyncio.run(main())
