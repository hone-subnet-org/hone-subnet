"""Pooled scoring over the shared ledger: admission, caps, window, scores, state."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from rlvr import protocol
from rlvr.config import Settings
from rlvr.neurons import decentralized
from rlvr.policy import RELEASE_POLICY
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.v3.api import LedgerPage, LedgerRound, RoundVerdict
from rlvr.v3.client import V3ProblemServerClient
from rlvr.v3.pool import (
    LedgerPool,
    parse_recorded_at,
    round_payments,
    validating_with_stake,
)
from rlvr.v3.verdicts import sign_round

TASK = "a" * 64
DAY = 86_400
HOTKEYS = [f"hk-{uid}" for uid in range(8)]


@pytest.fixture(autouse=True)
def offline_signatures(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)


def verdict(uid, passed, latency=None):
    return RoundVerdict(uid=uid, hotkey=f"hk-{uid}", passed=passed, response_latency_ms=latency)


def signed(validator, seq, verdicts, *, recorded_at="2026-10-05T12:00:00Z", challenge=None, signature=None):
    ordered = sorted(verdicts, key=lambda item: item.uid)
    challenge = challenge or f"{validator}-{seq}"
    return LedgerRound(
        validator_hotkey=validator, challenge_id=challenge, task_id=TASK, round_seq=seq, verdicts=ordered,
        recorded_at=recorded_at, signature=signature or sign_round(validator, challenge, TASK, seq, ordered),
    )


def page(*rounds, cursor="c1"):
    return LedgerPage(protocol_version=3, rounds=list(rounds), next_cursor=cursor)


def pool(**over):
    fields = {"own_hotkey": "me", "cap_per_validator": 3, "window_s": 4 * DAY, "speed_half_life_ms": 180_000, "speed_floor": 0.95}
    return LedgerPool(**(fields | over))


NOW = parse_recorded_at("2026-10-05T12:00:00Z")
EVERYONE = lambda hotkey: True  # noqa: E731


# --------------------------------------------------------------------------- #
# payments: the same speed factor the validator applies to its own rounds
# --------------------------------------------------------------------------- #
def test_round_payments_match_local_scoring_rules():
    payments = round_payments(
        [verdict(1, True, 1_000), verdict(2, True, 181_000), verdict(3, False, 50), verdict(4, True, None), verdict(5, False, None)],
        speed_half_life_ms=180_000, speed_floor=0.5,
    )
    assert payments[(1, "hk-1")] == 1.0  # the fastest pass
    assert payments[(2, "hk-2")] == pytest.approx(0.75)  # one half-life behind
    assert payments[(3, "hk-3")] == 0.0 and payments[(5, "hk-5")] == 0.0  # a miss is a miss
    assert payments[(4, "hk-4")] == 0.5  # a pass with no usable latency gets the floor
    assert round_payments([verdict(1, True, None)], speed_half_life_ms=180_000, speed_floor=0.5) == {(1, "hk-1"): 1.0}  # nobody to be behind


def test_recorded_at_accepts_the_servers_utc_forms():
    assert parse_recorded_at("2026-10-05T12:00:00Z") == parse_recorded_at("2026-10-05T12:00:00+00:00") == parse_recorded_at("2026-10-05T12:00:00")
    with pytest.raises(ValueError):
        parse_recorded_at("yesterday")


# --------------------------------------------------------------------------- #
# admission
# --------------------------------------------------------------------------- #
def test_only_verified_rounds_from_admitted_validators_enter_and_each_round_once():
    p = pool()
    good = signed("val-a", 1, [verdict(1, True, 10)])
    forged = signed("val-a", 2, [verdict(1, True, 10)], signature="0xdead")
    other_key = signed("val-b", 1, [verdict(1, True, 10)])
    mine = signed("me", 1, [verdict(1, True, 10)])
    taken, dropped = p.ingest(page(good, forged, other_key, mine, good), admitted=lambda hk: hk != "val-b", hotkeys=HOTKEYS, now=NOW)
    assert (taken, dropped) == (1, 2)  # forged and val-b dropped; mine and the repeat skipped silently
    assert p.pooled_observations(1, "hk-1") == [1.0]
    assert p.cursor == "c1"


def test_a_gap_in_a_validators_numbers_is_logged_and_the_round_still_counts(capsys):
    p = pool()
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)]), signed("val-a", 4, [verdict(1, False)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert "val-a skipped 2 round(s) before 4" in capsys.readouterr().out
    assert p.pooled_observations(1, "hk-1") == [1.0, 0.0] and p.last_seq["val-a"] == 4


def test_old_rounds_are_remembered_but_not_counted_and_the_window_prunes():
    p = pool(window_s=DAY)
    old = signed("val-a", 1, [verdict(1, True, 10)], recorded_at="2026-10-01T00:00:00Z")
    fresh = signed("val-a", 2, [verdict(1, True, 10)], recorded_at="2026-10-05T11:00:00Z")
    assert p.ingest(page(old, fresh), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (1, 0)
    assert ("val-a", 1) not in p.seen and p.last_seq["val-a"] == 2  # too old to matter, but the numbering is known
    assert p.pooled_observations(1, "hk-1") == [1.0]
    p.prune(NOW + DAY)  # the fresh one ages out
    assert p.pooled_observations(1, "hk-1") == [] and p.entries == {} and not p.active


def test_one_validator_contributes_at_most_the_cap_per_miner_and_every_round_counts_once():
    p = pool(cap_per_validator=3)
    rounds_a = [signed("val-a", seq, [verdict(1, True, 10)]) for seq in range(1, 6)]  # five rounds, all passes
    rounds_b = [signed("val-b", 1, [verdict(1, False)])]  # one round, a miss
    p.ingest(page(*rounds_a, *rounds_b), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.pooled_observations(1, "hk-1") == [1.0, 1.0, 1.0, 0.0]  # a is capped at 3, b counts once
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=4, now=NOW) == [0.0, 3.0 / 4]


def test_scores_are_exactly_the_local_rule_when_nothing_is_pooled():
    p = pool()
    local = {1: [(1.0, 1.0), (2.0, 0.0), (3.0, 1.0)], 2: [(1.0, 1.0)] * 6}
    assert p.scores(["hk-0", "hk-1", "hk-2"], local, min_samples=4, now=NOW) == [0.0, 2.0 / 4, 6.0 / 6]
    assert p.observation_count(["hk-0", "hk-1", "hk-2"], local, now=NOW) == 6


def test_our_own_window_is_capped_like_everyone_elses_once_rounds_are_pooled():
    p = pool(cap_per_validator=3)
    local = {1: [(float(i), 1.0) for i in range(10)]}  # ten local passes
    assert p.scores(["hk-0", "hk-1"], local, min_samples=4, now=NOW) == [0.0, 1.0]  # nothing pooled: the local rule, all ten
    p.ingest(page(signed("val-a", 1, [verdict(1, False)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    # pooled for uid 1: our newest three passes and val-a's one miss, four observations, equal footing
    assert p.scores(["hk-0", "hk-1"], local, min_samples=4, now=NOW) == [0.0, 3.0 / 4]
    assert p.observation_count(["hk-0", "hk-1"], local, now=NOW) == 4
    # a miner nobody else graded keeps its full local window even while others are pooled
    local[2] = [(float(i), 1.0) for i in range(10)]
    assert p.scores(["hk-0", "hk-1", "hk-2"], local, min_samples=4, now=NOW)[2] == 1.0


def test_a_re_registered_uid_inherits_nothing():
    p = pool()
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW) == [0.0, 1.0]
    assert p.scores(["hk-0", "hk-new"], {}, min_samples=1, now=NOW) == [0.0, 0.0]  # uid 1 re-registered: nothing inherited


def test_state_survives_a_restart(tmp_path):
    p = pool()
    p.ingest(page(signed("val-a", 3, [verdict(1, True, 10), verdict(2, False)]), cursor="after-3"), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    p.save(tmp_path / "pool.json")
    again = pool()
    assert again.load(tmp_path / "pool.json", now=NOW)
    assert again.cursor == "after-3" and again.last_seq == {"val-a": 3} and ("val-a", 3) in again.seen
    assert ("val-a", "val-a-3") in again.challenges
    assert again.scores(["hk-0", "hk-1", "hk-2"], {}, min_samples=1, now=NOW) == [0.0, 1.0, 0.0]
    for broken in ({"format": 99}, [], {"format": 1, "cursor": None, "last_seq": {}, "seen": [], "entries": {"val-a": [[1, "hk-1", [[1.0, 100.0, 1]]]]}}):
        (tmp_path / "pool.json").write_text(json.dumps(broken))
        fresh = pool()
        assert not fresh.load(tmp_path / "pool.json", now=NOW) and not fresh.active
    assert not pool().load(tmp_path / "missing.json", now=NOW)


def test_admission_needs_permit_trust_and_stake_share():
    metagraph = SimpleNamespace(
        hotkeys=["big", "small", "miner", "quiet"],
        validator_permit=[True, True, False, True],
        validator_trust=[0.5, 0.5, 0.0, 0.0],
        S=[900.0, 5.0, 95.0, 100.0],  # total 1100: big 82%, small 0.45%, quiet 9% but no trust
    )
    share = RELEASE_POLICY.v3_pool_min_stake_share
    assert validating_with_stake(metagraph, "big", min_share=share)
    assert not validating_with_stake(metagraph, "small", min_share=share)  # under 0.75%
    assert not validating_with_stake(metagraph, "miner", min_share=share)  # no permit
    assert not validating_with_stake(metagraph, "quiet", min_share=share)  # permit, no trust
    assert not validating_with_stake(metagraph, "unknown", min_share=share)


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def test_fetch_rounds_is_a_signed_get_and_reports_status(tmp_path):
    seen = []

    async def handler(request):
        seen.append((request.method, str(request.url), request.headers.get("Epistula-Signed-By")))
        if "since=after" in str(request.url):
            return httpx.Response(200, content=page(cursor=None).model_dump_json())
        if "limit=7" in str(request.url):
            return httpx.Response(404)
        if "limit=8" in str(request.url):
            return httpx.Response(200, content=b"{not json")
        return httpx.Response(200, content=page(signed("val-a", 1, [verdict(1, True, 10)])).model_dump_json())

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient("https://problems.invalid", "validator", http, retries=1)
            first = await client.fetch_rounds(since=None)
            caught_up = await client.fetch_rounds(since="after 1")
            absent = await client.fetch_rounds(since=None, limit=7)
            broken = await client.fetch_rounds(since=None, limit=8)
            return first, caught_up, absent, broken

    first, caught_up, absent, broken = asyncio.run(go())
    assert first.status == 200 and len(first.page.rounds) == 1 and first.page.next_cursor == "c1"
    assert caught_up.page.next_cursor is None and caught_up.page.rounds == []
    assert (absent.status, absent.page) == (404, None)
    assert (broken.status, broken.page) == (200, None)
    assert seen[0][0] == "GET" and seen[0][1].endswith("/v3/rounds?limit=200") and seen[0][2] == "validator"
    assert "since=after%201" in seen[1][1]


def test_sync_pool_is_bounded_a_404_is_a_no_op_and_state_is_saved(tmp_path, capsys):
    calls = []

    class Client:
        def __init__(self, pages):
            self.pages = list(pages)

        async def fetch_rounds(self, *, since, limit=200):
            calls.append(since)
            return self.pages.pop(0)

    metagraph = SimpleNamespace(
        hotkeys=["val-a", "hk-1", "me"], validator_permit=[True, False, True], validator_trust=[0.3, 0.0, 0.3], S=[50.0, 1.0, 50.0]
    )
    p = pool(cap_per_validator=50)
    path = tmp_path / "pool.json"
    # a 404: the ledger is not served yet; nothing changes and nothing is written
    asyncio.run(decentralized._sync_pool(p, Client([SimpleNamespace(status=404, page=None)]), metagraph, RELEASE_POLICY, path, now=NOW))
    assert not path.exists() and not p.active
    # many pages: at most v3_pool_pages_per_round are pulled per round, the cursor advances
    pages = [SimpleNamespace(status=200, page=page(signed("val-a", seq, [verdict(1, True, 10)]), cursor=f"c{seq}")) for seq in range(1, 20)]
    asyncio.run(decentralized._sync_pool(p, Client(pages), metagraph, RELEASE_POLICY, path, now=NOW))
    assert len(calls) == 1 + RELEASE_POLICY.v3_pool_pages_per_round and p.cursor == f"c{RELEASE_POLICY.v3_pool_pages_per_round}"
    assert path.exists() and p.active and "admitted" in capsys.readouterr().out
    # a failed page keeps what we have
    asyncio.run(decentralized._sync_pool(p, Client([SimpleNamespace(status=503, page=None)]), metagraph, RELEASE_POLICY, path, now=NOW))
    assert p.cursor == f"c{RELEASE_POLICY.v3_pool_pages_per_round}"


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def make_validator(hotkeys, calls):
    def set_weights(**kwargs):
        calls.append(kwargs)
        return True

    return SimpleNamespace(
        metagraph=SimpleNamespace(hotkeys=hotkeys, uids=list(range(len(hotkeys)))),
        subtensor=SimpleNamespace(set_weights=set_weights),
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="me")),
    )


def test_weights_come_from_the_pooled_window_once_rounds_are_admitted(monkeypatch, capsys):
    monkeypatch.setattr(decentralized.time, "time", lambda: NOW)  # the fixtures are dated; freeze the clock
    monkeypatch.setattr(decentralized, "_weight_result_status", lambda result: (True, "ok"))
    monkeypatch.setattr(decentralized, "_weight_failure_report", lambda result, validator: "")
    engine = EvalEngine(3, 57_600.0, 200, 4)
    engine.update({1: 1.0}, hotkeys={1: "hk-1"})  # one local pass for uid 1
    settings = Settings(_env_file=None)
    calls = []
    validator = make_validator(["hk-0", "hk-1", "hk-2"], calls)
    # local alone: one observation is below the gate of four
    assert decentralized._submit_local_weights(validator, engine, settings, pool=pool()) is None
    assert "local weight evidence 1/4" in capsys.readouterr().out and calls == []
    # three admitted rounds from another validator: uid 1 has four pooled observations, uid 2 three misses
    p = pool(cap_per_validator=50)
    p.ingest(page(*[signed("val-a", seq, [verdict(1, True, 10), verdict(2, False)]) for seq in (1, 2, 3)]), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert decentralized._submit_local_weights(validator, engine, settings, pool=p) is True
    weights = calls[0]["weights"]
    assert weights == pytest.approx([0.0, 1.0, 0.0])  # uid 1: 4 passes of 4; uid 2: 0 of 3
    assert "set local weights" in capsys.readouterr().out


def test_a_replayed_round_is_refused_even_after_a_restart_pushed_it_out_of_the_cap(tmp_path):
    p = pool(cap_per_validator=2)
    for seq in (1, 2, 3):
        p.ingest(page(signed("val-a", seq, [verdict(1, seq != 1)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.pooled_observations(1, "hk-1") == [1.0, 1.0]  # round 1 (a miss) was capped out
    p.save(tmp_path / "pool.json")
    again = pool(cap_per_validator=2)
    assert again.load(tmp_path / "pool.json", now=NOW)
    assert again.ingest(page(signed("val-a", 1, [verdict(1, False)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (0, 0)
    assert again.pooled_observations(1, "hk-1") == [1.0, 1.0]  # the replay changed nothing


def test_stale_entries_never_reach_the_scores_even_without_a_new_round():
    p = pool(window_s=DAY)
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW) == [0.0, 1.0]
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW + 2 * DAY) == [0.0, 0.0]  # pruned at scoring time
    assert not p.active and p.seen == {} and p.last_seq == {}


def test_a_round_from_the_future_is_refused():
    p = pool()
    ahead = signed("val-a", 1, [verdict(1, True, 10)], recorded_at="2099-01-01T00:00:00Z")
    assert p.ingest(page(ahead), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (0, 1)
    soon = signed("val-a", 2, [verdict(1, True, 10)], recorded_at="2026-10-05T12:05:00Z")  # five minutes ahead: clock skew
    assert p.ingest(page(soon), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (1, 0)


def test_only_miners_registered_as_named_get_a_bucket():
    p = pool()
    invented = [RoundVerdict(uid=1, hotkey="hk-1", passed=True, response_latency_ms=10)] + [
        RoundVerdict(uid=uid, hotkey=f"ghost-{uid}", passed=True, response_latency_ms=10) for uid in range(2, 1000)
    ]
    p.ingest(page(signed("val-a", 1, invented)), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert list(p.entries["val-a"]) == [(1, "hk-1")]  # 998 invented registrations left no trace


def test_an_expired_pool_scores_exactly_like_local(monkeypatch, capsys):
    monkeypatch.setattr(decentralized.time, "time", lambda: NOW + 10 * DAY)  # everything pooled has aged out
    monkeypatch.setattr(decentralized, "_weight_result_status", lambda result: (True, "ok"))
    monkeypatch.setattr(decentralized, "_weight_failure_report", lambda result, validator: "")
    engine = EvalEngine(10, 57_600.0, 200, 4)
    engine.update({0: 1.0}, hotkeys={0: "hk-0"})
    for _ in range(4):
        engine.update({9: 1.0}, hotkeys={9: "hk-9"})  # a uid that has since left the metagraph
    settings = Settings(_env_file=None)
    p = pool(cap_per_validator=50)
    p.ingest(page(signed("val-a", 1, [verdict(0, True, 10)])), admitted=EVERYONE, hotkeys=["hk-0"], now=NOW)
    local_calls, pooled_calls = [], []
    assert decentralized._submit_local_weights(make_validator(["hk-0"], local_calls), engine, settings, pool=None) is True
    assert decentralized._submit_local_weights(make_validator(["hk-0"], pooled_calls), engine, settings, pool=p) is True
    assert local_calls[0]["weights"] == pooled_calls[0]["weights"]  # the same gate, the same weights


def test_a_second_number_for_the_same_challenge_is_refused():
    p = pool()
    first = signed("val-a", 1, [verdict(1, True, 10)], challenge="chal-x")
    again = signed("val-a", 2, [verdict(1, False)], challenge="chal-x")
    assert p.ingest(page(first, again), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (1, 0)
    assert p.pooled_observations(1, "hk-1") == [1.0]


def test_a_validator_cannot_fill_memory_with_fresh_rounds():
    import rlvr.v3.pool as pool_module

    p = pool(cap_per_validator=2)
    flood = [signed("val-a", seq, [verdict(1, True, 10)]) for seq in range(1, pool_module.ROUNDS_PER_VALIDATOR + 10)]
    taken = dropped = 0
    for start in range(0, len(flood), 200):  # a page holds at most 200 rounds
        got, lost = p.ingest(page(*flood[start : start + 200]), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
        taken += got
        dropped += lost
    assert taken == pool_module.ROUNDS_PER_VALIDATOR and dropped == 9
    assert len(p.seen) == pool_module.ROUNDS_PER_VALIDATOR and len(p.pooled_observations(1, "hk-1")) == 2


def test_pruning_does_not_depend_on_arrival_order():
    p = pool(window_s=DAY)
    later = signed("val-a", 1, [verdict(1, True, 10)], recorded_at="2026-10-05T12:05:00Z")
    earlier = signed("val-a", 2, [verdict(1, True, 10)], recorded_at="2026-10-04T12:00:00Z")
    p.ingest(page(later, earlier), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW + 1) == [0.0, 1.0]  # the earlier one aged out, not hidden


def test_a_uid_whose_hotkey_changed_and_changed_back_starts_from_nothing():
    p = pool()
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=["hk-0", "hk-1"], now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW) == [0.0, 1.0]
    p.ingest(page(signed("val-a", 2, [RoundVerdict(uid=1, hotkey="hk-b", passed=False)])), admitted=EVERYONE, hotkeys=["hk-0", "hk-b"], now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW) == [0.0, 0.0]  # hk-1 is back at uid 1: nothing inherited


def test_a_full_replay_record_frees_up_as_it_ages_so_fresh_rounds_are_taken():
    import rlvr.v3.pool as pool_module

    p = pool(cap_per_validator=2, window_s=DAY)
    flood = [signed("val-a", seq, [verdict(1, True, 10)]) for seq in range(1, pool_module.ROUNDS_PER_VALIDATOR + 1)]
    for start in range(0, len(flood), 200):
        p.ingest(page(*flood[start : start + 200]), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    fresh = signed("val-a", 5000, [verdict(1, False)], recorded_at="2026-10-07T12:00:00Z")
    assert p.ingest(page(fresh), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW + 2 * DAY) == (1, 0)  # the old record aged out first
    assert p.pooled_observations(1, "hk-1") == [0.0]


def test_the_cap_keeps_the_newest_rounds_by_the_servers_clock_not_by_arrival():
    p = pool(cap_per_validator=1)
    newer = signed("val-a", 2, [verdict(1, True, 10)], recorded_at="2026-10-05T12:00:00Z")
    older = signed("val-a", 1, [verdict(1, False)], recorded_at="2026-10-05T11:59:00Z")
    p.ingest(page(newer, older), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)  # the older round arrives second
    assert p.pooled_observations(1, "hk-1") == [1.0]


def test_a_replaced_registration_does_not_trip_the_pooled_gate(monkeypatch, capsys):
    monkeypatch.setattr(decentralized.time, "time", lambda: NOW)
    monkeypatch.setattr(decentralized, "_weight_result_status", lambda result: (True, "ok"))
    monkeypatch.setattr(decentralized, "_weight_failure_report", lambda result, validator: "")
    engine = EvalEngine(10, 57_600.0, 200, 4)
    engine.update({0: 1.0}, hotkeys={0: "hk-0"})
    for _ in range(4):
        engine.update({9: 1.0}, hotkeys={9: "hk-9"})
    p = pool(cap_per_validator=50)
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=["hk-0", "hk-1"], now=NOW)
    settings = Settings(_env_file=None)
    local_calls, pooled_calls = [], []
    # uid 1 is now held by another hotkey: the only pooled bucket is gone, so this is a local submission
    assert decentralized._submit_local_weights(make_validator(["hk-0", "hk-new"], local_calls), engine, settings, pool=None) is True
    assert decentralized._submit_local_weights(make_validator(["hk-0", "hk-new"], pooled_calls), engine, settings, pool=p) is True
    assert local_calls[0]["weights"] == pooled_calls[0]["weights"]
