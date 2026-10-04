"""Pooled scoring over the shared ledger: admission, caps, window, scores, state."""

from __future__ import annotations

import asyncio
import json
import time
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
EVERYONE = lambda hotkey: True


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


def test_round_payments_equal_the_validators_own_payments_for_the_same_round():
    from rlvr.v3.grading import EvaluationResult
    from rlvr.v3.round import MinerEvaluation, RoundResult, compute_round_payments

    outcomes = [(1, True, 1_000), (2, True, 181_000), (3, False, 50), (4, True, 400_000), (5, False, 9)]
    local = compute_round_payments(
        RoundResult("completed", "", tuple(
            MinerEvaluation(uid, f"hk-{uid}", latency, EvaluationResult("passed" if ok else "failed", "" if ok else "x", (), None), 1)
            for uid, ok, latency in outcomes
        )),
        speed_half_life_ms=180_000, speed_floor=0.95,
    )
    pooled = round_payments([verdict(uid, ok, latency) for uid, ok, latency in outcomes], speed_half_life_ms=180_000, speed_floor=0.95)
    assert {uid: pooled[(uid, f"hk-{uid}")] for uid, _, _ in outcomes} == pytest.approx(local)


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


def test_a_gap_in_a_validators_numbers_still_counts():
    p = pool()
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)]), signed("val-a", 4, [verdict(1, False)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.pooled_observations(1, "hk-1") == [1.0, 0.0] and p.newest_seq("val-a") == 4


def test_old_rounds_are_remembered_but_not_counted_and_the_window_prunes():
    p = pool(window_s=DAY)
    old = signed("val-a", 1, [verdict(1, True, 10)], recorded_at="2026-10-01T00:00:00Z")
    fresh = signed("val-a", 2, [verdict(1, True, 10)], recorded_at="2026-10-05T11:00:00Z")
    assert p.ingest(page(old, fresh), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (1, 0)
    assert p.seen["val-a"] == {1, 2}  # too old to score, but remembered and numbered
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
    local = {1: [(NOW - 100 + i, 1.0) for i in range(10)]}  # ten recent local passes
    assert p.scores(["hk-0", "hk-1"], local, min_samples=4, now=NOW) == [0.0, 1.0]  # nothing pooled: the local rule, all ten
    p.ingest(page(signed("val-a", 1, [verdict(1, False)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    # pooled for uid 1: our newest three passes and val-a's one miss, four observations, equal footing
    assert p.scores(["hk-0", "hk-1"], local, min_samples=4, now=NOW) == [0.0, 3.0 / 4]
    assert p.observation_count(["hk-0", "hk-1"], local, now=NOW) == 4
    # a miner nobody else graded keeps its full local window even while others are pooled
    local[2] = [(NOW - 100 + i, 1.0) for i in range(10)]
    assert p.scores(["hk-0", "hk-1", "hk-2"], local, min_samples=4, now=NOW)[2] == 1.0


def test_a_re_registered_uid_inherits_nothing():
    p = pool()
    local = {1: [(NOW - 10, 1.0), (NOW - 5, 1.0)]}  # what the engine still holds for uid 1 at that moment
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], local, min_samples=1, now=NOW) == [0.0, 1.0]
    # uid 1 re-registered under a new hotkey: the pooled bucket is dropped; the engine resets its own
    # history on the same event, which is why the local entries here are the new registration's (none)
    assert p.scores(["hk-0", "hk-new"], {}, min_samples=1, now=NOW) == [0.0, 0.0]
    assert p.entries.get("val-a", {}) == {}


def test_state_survives_a_restart(tmp_path):
    p = pool()
    p.ingest(page(signed("val-a", 3, [verdict(1, True, 10), verdict(2, False)]), cursor="after-3"), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    p.save(tmp_path / "pool.json")
    again = pool()
    assert again.load(tmp_path / "pool.json", now=NOW)
    assert again.cursor == "after-3" and again.seen == {"val-a": {3}}
    assert again.scores(["hk-0", "hk-1", "hk-2"], {}, min_samples=1, now=NOW) == [0.0, 1.0, 0.0]
    valid = json.loads((tmp_path / "pool.json").read_text())
    bad_payment = {**valid, "entries": {"val-a": [[1, "hk-1", [[1.0, 100.0, 1]]]]}}
    assert json.loads((tmp_path / "pool.json").read_text()) == valid  # the fixture is the real shape, only the payment is wrong
    for broken in ({"format": 99}, [], bad_payment):
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
        if "since=absent" in str(request.url):
            return httpx.Response(404)
        if "since=broken" in str(request.url):
            return httpx.Response(200, content=b"{not json")
        return httpx.Response(200, content=page(signed("val-a", 1, [verdict(1, True, 10)])).model_dump_json())

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient("https://problems.invalid", "validator", http, retries=1)
            first = await client.fetch_rounds(since=None)
            caught_up = await client.fetch_rounds(since="after 1")
            absent = await client.fetch_rounds(since="absent")
            broken = await client.fetch_rounds(since="broken")
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

        async def fetch_rounds(self, *, since):
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
    # many pages: at most POOL_PAGES_PER_SYNC are pulled per round, the cursor advances
    pages = [SimpleNamespace(status=200, page=page(signed("val-a", seq, [verdict(1, True, 10)]), cursor=f"c{seq}")) for seq in range(1, 20)]
    asyncio.run(decentralized._sync_pool(p, Client(pages), metagraph, RELEASE_POLICY, path, now=NOW))
    assert len(calls) == 1 + decentralized.POOL_PAGES_PER_SYNC and p.cursor == f"c{decentralized.POOL_PAGES_PER_SYNC}"
    assert path.exists() and p.active and "admitted" in capsys.readouterr().out
    # a failed page keeps what we have
    asyncio.run(decentralized._sync_pool(p, Client([SimpleNamespace(status=503, page=None)]), metagraph, RELEASE_POLICY, path, now=NOW))
    assert p.cursor == f"c{decentralized.POOL_PAGES_PER_SYNC}"


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def make_validator(hotkeys, calls):
    """A validator whose metagraph lists the given miners plus val-a, a validating peer with stake."""

    def set_weights(**kwargs):
        calls.append(kwargs)
        return True

    everyone = [*hotkeys, "val-a"]
    return SimpleNamespace(
        metagraph=SimpleNamespace(
            hotkeys=everyone,
            uids=list(range(len(everyone))),
            validator_permit=[False] * len(hotkeys) + [True],
            validator_trust=[0.0] * len(hotkeys) + [0.4],
            S=[1.0] * len(hotkeys) + [100.0],
        ),
        subtensor=SimpleNamespace(set_weights=set_weights),
        wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="me")),
    )


def test_weights_come_from_the_pooled_window_once_rounds_are_admitted(monkeypatch, capsys):
    monkeypatch.setattr(decentralized.time, "time", lambda: NOW)  # the fixtures are dated; freeze the clock
    monkeypatch.setattr(decentralized, "_weight_result_status", lambda result: (True, "ok"))
    monkeypatch.setattr(decentralized, "_weight_failure_report", lambda result, validator: "")
    engine = EvalEngine(4, 57_600.0, 200, 4)
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
    assert weights == pytest.approx([0.0, 1.0, 0.0, 0.0])  # uid 1: 4 passes of 4; uid 2: 0 of 3; val-a itself: nothing
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


def test_an_aged_out_round_shown_again_with_a_fresh_time_is_refused(tmp_path):
    registered = ["val-a", "hk-1"]
    p = pool(cap_per_validator=50, window_s=DAY)
    p.ingest(page(signed("val-a", 100, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=registered, now=NOW)
    p.ingest(page(signed("val-a", 150, [verdict(1, True, 10)], recorded_at="2026-10-07T13:00:00Z")), admitted=EVERYONE, hotkeys=registered, now=NOW + 2 * DAY + 3600)
    p.settle(registered, NOW + 3 * DAY)
    assert p.seen == {"val-a": {100, 150}}  # out of the window, inside the slack: still remembered
    replayed = signed("val-a", 100, [verdict(1, True, 10)], recorded_at="2026-10-08T12:00:00Z")
    never_seen = signed("val-a", 7, [verdict(1, True, 10)], recorded_at="2026-10-08T12:00:00Z")
    late = signed("val-a", 148, [verdict(1, False)], recorded_at="2026-10-08T12:00:00Z")
    assert p.ingest(page(replayed, never_seen, late), admitted=EVERYONE, hotkeys=registered, now=NOW + 3 * DAY) == (1, 1)
    assert p.pooled_observations(1, "hk-1") == [1.0, 0.0]  # round 150, and the late round a few numbers behind it
    p.save(tmp_path / "pool.json")
    again = pool(cap_per_validator=50, window_s=DAY)
    assert again.load(tmp_path / "pool.json", now=NOW + 3 * DAY) and again.seen == {"val-a": {100, 148, 150}}  # the numbers survive a restart
    # much later: rounds 148 and 150 are out of the window, but their numbers are inside the slack and stay
    again.settle(registered, NOW + 30 * DAY)
    assert not again.active and again.seen == {"val-a": {100, 148, 150}}
    recycled = [signed("val-a", seq, [verdict(1, True, 10)], recorded_at="2026-11-04T12:00:00Z") for seq in (148, 150)]
    assert again.ingest(page(*recycled), admitted=EVERYONE, hotkeys=registered, now=NOW + 30 * DAY) == (0, 0)
    assert again.pooled_observations(1, "hk-1") == []


def test_a_round_first_seen_after_it_expired_cannot_come_back_redated():
    registered = ["val-a", "hk-1"]
    p = pool(cap_per_validator=50, window_s=DAY)
    expired = [signed("val-a", seq, [verdict(1, True, 10)], recorded_at="2026-09-01T00:00:00Z") for seq in (148, 150)]
    assert p.ingest(page(*expired), admitted=EVERYONE, hotkeys=registered, now=NOW) == (0, 0)
    assert p.seen == {"val-a": {148, 150}}
    redated = [signed("val-a", seq, [verdict(1, True, 10)], recorded_at="2026-10-05T11:00:00Z") for seq in (148, 150)]
    assert p.ingest(page(*redated), admitted=EVERYONE, hotkeys=registered, now=NOW) == (0, 0)
    assert p.pooled_observations(1, "hk-1") == [] and not p.active


def test_a_round_delivered_late_behind_dozens_of_newer_ones_still_counts():
    registered = ["val-a", "hk-1"]
    p = pool(cap_per_validator=50)
    newer = [signed("val-a", seq, [verdict(1, False)]) for seq in range(101, 141)]  # forty rounds in the six-hour window
    assert p.ingest(page(*newer), admitted=EVERYONE, hotkeys=registered, now=NOW) == (40, 0)
    late = signed("val-a", 100, [verdict(1, True, 10)], recorded_at="2026-10-05T12:01:00Z")
    assert p.ingest(page(late), admitted=EVERYONE, hotkeys=registered, now=NOW + 60) == (1, 0)
    assert sorted(p.pooled_observations(1, "hk-1")) == [0.0] * 40 + [1.0]


def test_stale_entries_never_reach_the_scores_even_without_a_new_round():
    p = pool(window_s=DAY)
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW) == [0.0, 1.0]
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW + 2 * DAY) == [0.0, 0.0]  # pruned at scoring time
    assert not p.active and p.seen == {"val-a": {1}}  # the number is kept: nothing to score, but never again


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


def test_the_numbers_remembered_per_validator_are_bounded_by_the_slack():
    import rlvr.v3.pool as pool_module

    registered = ["val-a", "hk-1"]
    p = pool(cap_per_validator=50)
    flood = [signed("val-a", seq, [verdict(1, True, 10)]) for seq in range(1, 1001)]
    for start in range(0, len(flood), 200):
        p.ingest(page(*flood[start : start + 200]), admitted=EVERYONE, hotkeys=registered, now=NOW)
    assert p.newest_seq("val-a") == 1000 and len(p.seen["val-a"]) == pool_module.SEQ_SLACK
    assert len(p.pooled_observations(1, "hk-1")) == 50  # the cap, not the flood
    # far below the newest: refused even though never seen; inside the slack: taken once
    stale = signed("val-a", 900, [verdict(1, False)], recorded_at="2026-10-05T12:01:00Z")
    unseen = signed("val-a", 1002, [verdict(1, False)], recorded_at="2026-10-05T12:01:00Z")
    assert p.ingest(page(stale, unseen, unseen), admitted=EVERYONE, hotkeys=registered, now=NOW + 60) == (1, 1)


def test_a_round_dated_before_the_epoch_is_refused_and_cannot_spoil_the_saved_state(tmp_path):
    registered = ["val-a", "hk-1"]
    p = pool(cap_per_validator=50)
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=registered, now=NOW)
    ancient = signed("val-a", 2, [verdict(1, True, 10)], recorded_at="1969-12-31T23:59:59Z")
    assert p.ingest(page(ancient), admitted=EVERYONE, hotkeys=registered, now=NOW) == (0, 1)
    assert p.seen == {"val-a": {1}}
    p.save(tmp_path / "pool.json")
    again = pool(cap_per_validator=50)
    assert again.load(tmp_path / "pool.json", now=NOW) and again.seen == {"val-a": {1}}  # the state is still good


def test_a_validator_cannot_fill_memory_with_fresh_rounds():
    import rlvr.v3.pool as pool_module

    p = pool(cap_per_validator=2)
    flood = [signed("val-a", seq, [verdict(1, True, 10)]) for seq in range(1, 1010)]
    taken = 0
    for start in range(0, len(flood), 200):  # a page holds at most 200 rounds
        got, lost = p.ingest(page(*flood[start : start + 200]), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
        taken += got
        assert lost == 0
    assert taken == 1009  # every round counts once...
    assert len(p.seen["val-a"]) == pool_module.SEQ_SLACK and len(p.pooled_observations(1, "hk-1")) == 2  # ...in bounded memory


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


def test_our_own_observations_age_out_of_a_pooled_window():
    p = pool(cap_per_validator=50, window_s=4 * DAY)
    local = {1: [(NOW - 5 * DAY, 1.0)] * 50}  # fifty passes, all five days old
    assert p.scores(["hk-0", "hk-1"], local, min_samples=4, now=NOW) == [0.0, 1.0]  # nothing pooled: the local rule as today
    p.ingest(page(*[signed("val-a", seq, [verdict(1, False)]) for seq in range(1, 51)]), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], local, min_samples=4, now=NOW) == [0.0, 0.0]  # pooled: our stale passes are outside the window


def test_a_validator_that_stopped_validating_loses_its_say_at_once():
    p = pool()
    p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW)
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW, admitted=EVERYONE) == [0.0, 1.0]
    assert p.scores(["hk-0", "hk-1"], {}, min_samples=1, now=NOW, admitted=lambda hk: False) == [0.0, 0.0]
    assert not p.active and p.seen == {"val-a": {1}}  # gone from the window, still refused as a replay
    assert p.ingest(page(signed("val-a", 1, [verdict(1, True, 10)])), admitted=EVERYONE, hotkeys=HOTKEYS, now=NOW) == (0, 0)


def test_sync_stops_when_the_server_does_not_move_the_cursor(tmp_path):
    calls = []

    class Client:
        def __init__(self, pages):
            self.pages = list(pages)

        async def fetch_rounds(self, *, since):
            calls.append(since)
            return self.pages.pop(0) if self.pages else SimpleNamespace(status=200, page=page(cursor=None))

    metagraph = SimpleNamespace(hotkeys=["val-a", "hk-1", "me"], validator_permit=[True, False, True], validator_trust=[0.3, 0.0, 0.3], S=[50.0, 1.0, 50.0])
    same = signed("val-a", 1, [verdict(1, True, 10)])
    # the same cursor back with the same rounds: one fetch, not five
    p = pool(cap_per_validator=50)
    p.cursor = "stuck"
    asyncio.run(decentralized._sync_pool(p, Client([SimpleNamespace(status=200, page=page(same, cursor="stuck"))] * 5), metagraph, RELEASE_POLICY, tmp_path / "pool.json", now=NOW))
    assert calls == ["stuck"]
    # a page of rounds we already hold still moves the cursor on; the same cursor back then stops it
    calls.clear()
    asyncio.run(decentralized._sync_pool(p, Client([SimpleNamespace(status=200, page=page(same, cursor="c9"))] * 5), metagraph, RELEASE_POLICY, tmp_path / "pool.json", now=NOW))
    assert calls == ["stuck", "c9"] and p.cursor == "c9"


def test_sync_pages_past_rounds_it_does_not_take_until_the_server_runs_out(tmp_path):
    calls = []
    expired = [signed("val-a", seq, [verdict(1, True, 10)], recorded_at="2026-09-01T00:00:00Z") for seq in (1, 2)]
    own = signed("me", 1, [verdict(1, True, 10)])
    fresh = signed("val-a", 3, [verdict(1, False)])
    pages = [page(*expired, cursor="c1"), page(own, cursor="c2"), page(fresh, cursor="c3"), page(cursor=None)]

    class Client:
        async def fetch_rounds(self, *, since):
            calls.append(since)
            return SimpleNamespace(status=200, page=pages.pop(0))

    metagraph = SimpleNamespace(hotkeys=["val-a", "hk-1", "me"], validator_permit=[True, False, True], validator_trust=[0.3, 0.0, 0.3], S=[50.0, 1.0, 50.0])
    p = pool(cap_per_validator=50)
    asyncio.run(decentralized._sync_pool(p, Client(), metagraph, RELEASE_POLICY, tmp_path / "pool.json", now=NOW))
    assert calls == [None, "c1", "c2", "c3"]  # a backlog of old rounds is paged through, not one page per round
    assert p.cursor == "c3" and p.pooled_observations(1, "hk-1") == [0.0]


def test_sync_respects_its_time_budget(tmp_path, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(decentralized.time, "monotonic", lambda: clock["t"])

    class Slow:
        def __init__(self):
            self.calls = 0

        async def fetch_rounds(self, *, since):
            self.calls += 1
            clock["t"] += decentralized.POOL_SYNC_BUDGET_S  # each page takes the whole budget
            return SimpleNamespace(status=200, page=page(signed("val-a", self.calls, [verdict(1, True, 10)]), cursor=f"c{self.calls}"))

    metagraph = SimpleNamespace(hotkeys=["val-a", "hk-1", "me"], validator_permit=[True, False, True], validator_trust=[0.3, 0.0, 0.3], S=[50.0, 1.0, 50.0])
    slow = Slow()
    asyncio.run(decentralized._sync_pool(pool(cap_per_validator=50), slow, metagraph, RELEASE_POLICY, tmp_path / "pool.json", now=NOW))
    assert slow.calls == 1  # the second page would start past the budget


def test_a_fetch_that_never_finishes_is_cut_at_the_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(decentralized, "POOL_SYNC_BUDGET_S", 0.2)

    class Trickle:
        async def fetch_rounds(self, *, since):
            await asyncio.sleep(60)  # a body arriving one byte at a time never trips the per-chunk timeout

    metagraph = SimpleNamespace(hotkeys=["val-a", "hk-1", "me"], validator_permit=[True, False, True], validator_trust=[0.3, 0.0, 0.3], S=[50.0, 1.0, 50.0])
    p = pool(cap_per_validator=50)
    started = time.monotonic()
    asyncio.run(decentralized._sync_pool(p, Trickle(), metagraph, RELEASE_POLICY, tmp_path / "pool.json", now=NOW))
    assert time.monotonic() - started < 5 and not p.active and p.cursor is None
    assert "exceeded the 0.2 s budget" in capsys.readouterr().out
