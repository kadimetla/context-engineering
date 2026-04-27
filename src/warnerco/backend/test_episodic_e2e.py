"""End-to-end test of the EpisodicStore. Run from src/warnerco/backend.

Usage:
    uv run python test_episodic_e2e.py
"""

import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.adapters.episodic_store import (
    EpisodicStore,
    get_episodic_store,
    reset_episodic_store,
)
from app.config import settings
from app.models.episodic import EpisodicEvent, EventKind


PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []  # (area, status, detail)


def record(area: str, status: str, detail: str = "") -> None:
    results.append((area, status, detail))
    marker = "+" if status == PASS else "X"
    print(f"  [{marker}] {area}: {status}{(' - ' + detail) if detail else ''}")


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def cleanup_db() -> None:
    p = settings.episodic_path
    if p.exists():
        # close any cached singleton conn first
        try:
            reset_episodic_store()
        except Exception:
            pass
        try:
            p.unlink()
        except PermissionError:
            # SQLite may still hold a handle on Windows; retry once
            time.sleep(0.2)
            p.unlink()
    # Also drop WAL/shm
    for ext in ("-wal", "-shm"):
        side = p.with_name(p.name + ext)
        if side.exists():
            try:
                side.unlink()
            except Exception:
                pass


async def test_fresh_store_schema() -> None:
    section("1. Fresh store schema")
    cleanup_db()
    store = EpisodicStore()
    assert settings.episodic_path.exists(), "events.db not created"
    record("db file created", PASS, str(settings.episodic_path))

    # Inspect schema directly
    conn = sqlite3.connect(str(settings.episodic_path))
    try:
        cols = conn.execute("PRAGMA table_info(events)").fetchall()
        col_names = [c[1] for c in cols]
        expected = {"id", "session_id", "kind", "summary", "content",
                    "importance", "created_at", "provenance"}
        if set(col_names) == expected and len(col_names) == 8:
            record("8 columns w/ correct names", PASS, str(col_names))
        else:
            record("8 columns w/ correct names", FAIL,
                   f"got {col_names}, expected {expected}")

        idxs = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events' "
            "AND name NOT LIKE 'sqlite_autoindex%'"
        ).fetchall()
        idx_names = sorted(i[0] for i in idxs)
        expected_idx = ["idx_ep_created", "idx_ep_kind", "idx_ep_session"]
        if idx_names == expected_idx:
            record("3 indexes present", PASS, str(idx_names))
        else:
            record("3 indexes present", FAIL,
                   f"got {idx_names}, expected {expected_idx}")
    finally:
        conn.close()
    store.close()


async def test_log_explicit_importance() -> None:
    section("2. log() with explicit importance")
    cleanup_db()
    store = EpisodicStore()
    prov = {"source": "unit_test", "trust_level": "high", "extra": [1, 2, 3]}
    ev = await store.log(
        session_id="s1",
        kind=EventKind.USER_TURN,
        summary="hello world",
        content="full payload here",
        importance=0.85,
        provenance=prov,
    )

    if isinstance(ev, EpisodicEvent):
        record("returns EpisodicEvent", PASS)
    else:
        record("returns EpisodicEvent", FAIL, type(ev).__name__)

    if ev.session_id == "s1" and ev.summary == "hello world" and ev.kind == EventKind.USER_TURN:
        record("fields populated", PASS)
    else:
        record("fields populated", FAIL, str(ev))

    if abs(ev.importance - 0.85) < 1e-9:
        record("importance preserved", PASS)
    else:
        record("importance preserved", FAIL, str(ev.importance))

    # ISO timestamp must parse and be UTC
    try:
        ts = datetime.fromisoformat(ev.created_at)
        if ts.tzinfo is not None and ts.utcoffset() == (datetime.now(timezone.utc).utcoffset()):
            record("ISO timestamp is UTC", PASS, ev.created_at)
        else:
            record("ISO timestamp is UTC", FAIL, f"tz={ts.tzinfo}")
    except Exception as e:
        record("ISO timestamp is UTC", FAIL, str(e))

    # Round-trip provenance via raw SQLite
    conn = sqlite3.connect(str(settings.episodic_path))
    try:
        row = conn.execute("SELECT provenance FROM events WHERE id = ?", (ev.id,)).fetchone()
        loaded = json.loads(row[0])
        if loaded == prov:
            record("provenance JSON roundtrip", PASS)
        else:
            record("provenance JSON roundtrip", FAIL, f"got {loaded}")
    finally:
        conn.close()
    store.close()


async def test_log_default_importance_no_llm() -> None:
    section("3. log() with importance=None")
    cleanup_db()
    store = EpisodicStore()
    ev = await store.log("s1", EventKind.OBSERVATION, "no importance set")
    if settings.has_llm_config:
        # We hit a real LLM. Just ensure clamped 0..1 and record what came back.
        if 0.0 <= ev.importance <= 1.0:
            record("LLM-scored importance in [0,1]", PASS,
                   f"value={ev.importance} (LLM was called)")
        else:
            record("LLM-scored importance in [0,1]", FAIL, str(ev.importance))
    else:
        if abs(ev.importance - 0.3) < 1e-9:
            record("default 0.3 when no LLM", PASS)
        else:
            record("default 0.3 when no LLM", FAIL, str(ev.importance))
    store.close()


async def test_log_invalid_kind() -> None:
    section("4. log() with invalid kind raises ValueError")
    cleanup_db()
    store = EpisodicStore()
    try:
        await store.log("s1", "totally_bogus_kind", "x", importance=0.1)
        record("invalid kind raises ValueError", FAIL, "no exception")
    except ValueError as e:
        record("invalid kind raises ValueError", PASS, str(e)[:60])
    except Exception as e:
        record("invalid kind raises ValueError", FAIL,
               f"wrong type: {type(e).__name__}: {e}")
    store.close()


async def test_log_clamps_importance() -> None:
    section("5. log() clamps importance to [0,1]")
    cleanup_db()
    store = EpisodicStore()
    ev_hi = await store.log("s1", EventKind.OBSERVATION, "hi", importance=1.5)
    ev_lo = await store.log("s1", EventKind.OBSERVATION, "lo", importance=-0.2)
    if ev_hi.importance == 1.0:
        record("1.5 clamps to 1.0", PASS)
    else:
        record("1.5 clamps to 1.0", FAIL, str(ev_hi.importance))
    if ev_lo.importance == 0.0:
        record("-0.2 clamps to 0.0", PASS)
    else:
        record("-0.2 clamps to 0.0", FAIL, str(ev_lo.importance))
    store.close()


async def test_recall_empty() -> None:
    section("6. recall() on empty store")
    cleanup_db()
    store = EpisodicStore()
    res = await store.recall("anything")
    if res.events == [] and res.scores == []:
        record("empty events/scores", PASS)
    else:
        record("empty events/scores", FAIL, str(res))
    if (res.weights.get("recency") == settings.episodic_weight_recency
            and res.weights.get("importance") == settings.episodic_weight_importance
            and res.weights.get("relevance") == settings.episodic_weight_relevance):
        record("weights populated from settings", PASS, str(res.weights))
    else:
        record("weights populated from settings", FAIL, str(res.weights))
    if res.half_life_hours == settings.episodic_recency_half_life_hours:
        record("half_life populated", PASS)
    else:
        record("half_life populated", FAIL, str(res.half_life_hours))
    store.close()


async def test_recall_ranking() -> None:
    section("7. recall() ranking via Park formula")
    cleanup_db()
    store = EpisodicStore()
    query = "robot servo gear"

    # Old + low importance + low relevance: bottom
    e_old_low = await store.log(
        "s1", EventKind.OBSERVATION,
        summary="weather report",
        content="rain expected tomorrow",
        importance=0.1,
    )
    # Backdate 1000 hours
    backdated = (datetime.now(timezone.utc).replace(microsecond=0)
                 .fromtimestamp(time.time() - 1000 * 3600, tz=timezone.utc).isoformat())
    conn = sqlite3.connect(str(settings.episodic_path))
    try:
        conn.execute("UPDATE events SET created_at = ? WHERE id = ?",
                     (backdated, e_old_low.id))
        conn.commit()
    finally:
        conn.close()

    # Recent, mid importance, low relevance
    await store.log("s1", EventKind.OBSERVATION,
                    summary="cafeteria menu update",
                    content="pasta tuesdays",
                    importance=0.5)

    # Recent, mid importance, MEDIUM relevance (one term overlap)
    await store.log("s1", EventKind.OBSERVATION,
                    summary="robot arrived in shop",
                    content="standard intake",
                    importance=0.5)

    # Recent, mid importance, HIGH relevance (all terms overlap) — should win
    e_winner = await store.log(
        "s1", EventKind.OBSERVATION,
        summary="robot servo gear inspected and replaced",
        content="robot servo gear assembly checked thoroughly",
        importance=0.5,
    )

    # Recent, HIGH importance, low relevance
    await store.log("s1", EventKind.OBSERVATION,
                    summary="completely unrelated",
                    content="plumbing maintenance",
                    importance=0.95)

    res = await store.recall(query, k=5)
    if res.events and res.events[0].id == e_winner.id:
        record("top-1 is high-relevance recent event", PASS,
               f"score={res.scores[0].total}")
    else:
        record("top-1 is high-relevance recent event", FAIL,
               f"top={res.events[0].id if res.events else None} winner={e_winner.id}")

    # Verify last is the very old, low importance, irrelevant event
    if res.events and res.events[-1].id == e_old_low.id:
        record("bottom is old/low/irrelevant", PASS)
    else:
        record("bottom is old/low/irrelevant", FAIL,
               f"bottom={res.events[-1].id if res.events else None}")

    # Sanity: scores are sorted descending
    totals = [s.total for s in res.scores]
    if totals == sorted(totals, reverse=True):
        record("scores sorted descending", PASS, str([round(t, 3) for t in totals]))
    else:
        record("scores sorted descending", FAIL, str(totals))

    store.close()


async def test_recall_session_filter() -> None:
    section("8. recall(session_id=X) filtering")
    cleanup_db()
    store = EpisodicStore()
    await store.log("sA", EventKind.OBSERVATION, "alpha event", importance=0.5)
    await store.log("sA", EventKind.OBSERVATION, "alpha second", importance=0.5)
    await store.log("sB", EventKind.OBSERVATION, "beta event", importance=0.5)
    res = await store.recall("event", session_id="sA")
    if all(e.session_id == "sA" for e in res.events) and len(res.events) == 2:
        record("only sA events returned", PASS, f"count={len(res.events)}")
    else:
        record("only sA events returned", FAIL,
               f"sessions={[e.session_id for e in res.events]}")
    store.close()


async def test_recent() -> None:
    section("9. recent() ordering, session filter, limit")
    cleanup_db()
    store = EpisodicStore()
    # Log in sequence; force tiny ordering by sleeping 5ms between writes
    ids = []
    for i in range(6):
        ev = await store.log("s1" if i % 2 == 0 else "s2",
                             EventKind.OBSERVATION,
                             f"event #{i}", importance=0.4)
        ids.append((ev.id, ev.session_id))
        await asyncio.sleep(0.01)

    rec_all = store.recent(limit=10)
    if [e.summary for e in rec_all] == [f"event #{i}" for i in reversed(range(6))]:
        record("all newest-first", PASS)
    else:
        record("all newest-first", FAIL, [e.summary for e in rec_all])

    rec_lim = store.recent(limit=3)
    if len(rec_lim) == 3:
        record("limit honored", PASS)
    else:
        record("limit honored", FAIL, len(rec_lim))

    rec_s1 = store.recent(session_id="s1", limit=10)
    if all(e.session_id == "s1" for e in rec_s1) and len(rec_s1) == 3:
        record("session-filtered count + sessions", PASS)
    else:
        record("session-filtered count + sessions", FAIL,
               f"n={len(rec_s1)} sessions={[e.session_id for e in rec_s1]}")

    store.close()


async def test_since() -> None:
    section("10. since(minutes=N)")
    cleanup_db()
    store = EpisodicStore()
    # Recent
    await store.log("s1", EventKind.OBSERVATION, "fresh", importance=0.4)
    # Old: backdate to 2 hours ago
    e_old = await store.log("s1", EventKind.OBSERVATION, "old", importance=0.4)
    old_ts = datetime.fromtimestamp(time.time() - 2 * 3600, tz=timezone.utc).isoformat()
    conn = sqlite3.connect(str(settings.episodic_path))
    try:
        conn.execute("UPDATE events SET created_at = ? WHERE id = ?",
                     (old_ts, e_old.id))
        conn.commit()
    finally:
        conn.close()

    out = store.since(minutes=30)
    summaries = [e.summary for e in out]
    if "fresh" in summaries and "old" not in summaries and len(summaries) == 1:
        record("only events in last 30 min", PASS)
    else:
        record("only events in last 30 min", FAIL, summaries)
    store.close()


async def test_stats() -> None:
    section("11. stats()")
    cleanup_db()
    store = EpisodicStore()
    await store.log("s1", EventKind.USER_TURN, "u1", importance=0.4)
    await store.log("s1", EventKind.USER_TURN, "u2", importance=0.4)
    await store.log("s1", EventKind.AGENT_RESPONSE, "a1", importance=0.4)
    await store.log("s2", EventKind.TOOL_CALL, "t1", importance=0.4)

    s = store.stats()
    ok = True
    detail_parts = []
    if s.event_count != 4:
        ok = False
        detail_parts.append(f"event_count={s.event_count}!=4")
    if s.session_count != 2:
        ok = False
        detail_parts.append(f"session_count={s.session_count}!=2")
    expected_kinds = {"user_turn": 2, "agent_response": 1, "tool_call": 1}
    if s.by_kind != expected_kinds:
        ok = False
        detail_parts.append(f"by_kind={s.by_kind} != {expected_kinds}")
    if not s.oldest or not s.newest:
        ok = False
        detail_parts.append("oldest/newest empty")
    if str(settings.episodic_path) != s.db_path:
        ok = False
        detail_parts.append(f"db_path mismatch: {s.db_path}")
    if ok:
        record("stats correct", PASS,
               f"count={s.event_count} sessions={s.session_count} kinds={s.by_kind}")
    else:
        record("stats correct", FAIL, "; ".join(detail_parts))
    store.close()


async def test_clear() -> None:
    section("12. clear() session-scoped + global")
    cleanup_db()
    store = EpisodicStore()
    await store.log("sA", EventKind.OBSERVATION, "a1", importance=0.4)
    await store.log("sA", EventKind.OBSERVATION, "a2", importance=0.4)
    await store.log("sB", EventKind.OBSERVATION, "b1", importance=0.4)
    n = store.clear(session_id="sA")
    if n == 2:
        record("clear(session) returns count cleared", PASS)
    else:
        record("clear(session) returns count cleared", FAIL, f"n={n}")
    rest = store.recent(limit=10)
    if len(rest) == 1 and rest[0].session_id == "sB":
        record("only target session cleared", PASS)
    else:
        record("only target session cleared", FAIL, str([(e.session_id, e.summary) for e in rest]))

    n2 = store.clear()
    if n2 == 1:
        record("clear() all returns count", PASS)
    else:
        record("clear() all returns count", FAIL, f"n={n2}")

    if store.stats().event_count == 0:
        record("post-clear empty", PASS)
    else:
        record("post-clear empty", FAIL)
    store.close()


async def test_concurrency() -> None:
    section("13. Concurrency: 5 tasks x 10 logs each")
    cleanup_db()
    store = EpisodicStore()

    async def worker(wid: int) -> None:
        for i in range(10):
            await store.log(
                f"sess-{wid}",
                EventKind.OBSERVATION,
                f"worker-{wid}-event-{i}",
                importance=0.4,
            )

    await asyncio.gather(*(worker(w) for w in range(5)))
    s = store.stats()
    if s.event_count == 50:
        record("all 50 events persisted", PASS)
    else:
        record("all 50 events persisted", FAIL, f"count={s.event_count}")
    if s.session_count == 5:
        record("5 distinct sessions", PASS)
    else:
        record("5 distinct sessions", FAIL, f"sessions={s.session_count}")

    # Verify no row corruption — all rows readable + valid kinds
    bad = 0
    for ev in store.recent(limit=100):
        if not ev.summary.startswith("worker-"):
            bad += 1
        if ev.kind != EventKind.OBSERVATION:
            bad += 1
    if bad == 0:
        record("no row corruption", PASS)
    else:
        record("no row corruption", FAIL, f"{bad} bad rows")
    store.close()


async def test_singleton() -> None:
    section("14. Singleton get_episodic_store / reset_episodic_store")
    cleanup_db()
    a = get_episodic_store()
    b = get_episodic_store()
    if a is b:
        record("get_episodic_store returns same instance", PASS)
    else:
        record("get_episodic_store returns same instance", FAIL)

    reset_episodic_store()
    c = get_episodic_store()
    if c is not a:
        record("reset_episodic_store returns fresh instance", PASS)
    else:
        record("reset_episodic_store returns fresh instance", FAIL)


async def main() -> None:
    print("=" * 60)
    print("EpisodicStore E2E test suite")
    print("=" * 60)
    print(f"DB path: {settings.episodic_path}")
    print(f"has_llm_config: {settings.has_llm_config}")

    try:
        await test_fresh_store_schema()
        await test_log_explicit_importance()
        await test_log_default_importance_no_llm()
        await test_log_invalid_kind()
        await test_log_clamps_importance()
        await test_recall_empty()
        await test_recall_ranking()
        await test_recall_session_filter()
        await test_recent()
        await test_since()
        await test_stats()
        await test_clear()
        await test_concurrency()
        await test_singleton()
    finally:
        # final cleanup
        try:
            reset_episodic_store()
        except Exception:
            pass
        cleanup_db()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    p = sum(1 for _, s, _ in results if s == PASS)
    f = sum(1 for _, s, _ in results if s == FAIL)
    print(f"PASS: {p}    FAIL: {f}    TOTAL: {len(results)}")
    if f:
        print("\nFailures:")
        for area, status, detail in results:
            if status == FAIL:
                print(f"  - {area}: {detail}")
    sys.exit(0 if f == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
