"""The validator, not the server, chooses which miners a task is offered to."""

import random

import pytest
from pydantic import ValidationError

from rlvr.v3.api import CANDIDATE_LIMIT, LeaseRequest, MinerCandidate
from rlvr.v3.selection import choose_candidates, eligible_miners, next_offer

SERVING = [(uid, f"hk-{uid}") for uid in range(10)]


def test_owner_and_permit_holders_are_never_offered():
    permits = [False] * 10
    permits[3] = permits[7] = True
    chosen = choose_candidates(SERVING, validator_permits=permits, rng=random.Random(1))
    assert {uid for uid, _ in chosen} == {1, 2, 4, 5, 6, 8, 9}


def test_permits_may_arrive_as_an_array():
    import numpy

    permits = numpy.array([False, False, False, True, False, False, False, True, False, False])
    chosen = choose_candidates(SERVING, validator_permits=permits, rng=random.Random(1))
    assert {uid for uid, _ in chosen} == {1, 2, 4, 5, 6, 8, 9}


def test_missing_permit_list_only_excludes_the_owner():
    chosen = choose_candidates(SERVING, validator_permits=None, rng=random.Random(1))
    assert {uid for uid, _ in chosen} == set(range(1, 10))


def test_order_is_random_and_the_list_is_capped():
    serving = [(uid, f"hk-{uid}") for uid in range(1, 400)]
    first = choose_candidates(serving, rng=random.Random(1))
    second = choose_candidates(serving, rng=random.Random(2))
    assert len(first) == len(second) == CANDIDATE_LIMIT
    assert first != second and set(first) != set(second)  # different subsets, different orders
    assert set(first) <= set(serving)


def test_default_source_of_randomness_is_not_the_module_random(monkeypatch):
    calls = []
    monkeypatch.setattr("random.shuffle", lambda *a, **k: calls.append("random"))
    chosen = choose_candidates(SERVING)
    assert calls == [] and set(chosen) == set(SERVING[1:])


def test_lease_request_names_its_candidates_and_bounds_them():
    request = LeaseRequest(request_id="r", candidates=[MinerCandidate(uid=5, hotkey="hk-5")])
    assert request.model_dump()["candidates"] == [{"uid": 5, "hotkey": "hk-5"}]
    with pytest.raises(ValidationError):
        LeaseRequest(request_id="r", candidates=[])
    with pytest.raises(ValidationError):
        LeaseRequest(request_id="r", candidates=[MinerCandidate(uid=i, hotkey=f"hk-{i}") for i in range(CANDIDATE_LIMIT + 1)])
    with pytest.raises(ValidationError, match="repeat"):
        LeaseRequest(request_id="r", candidates=[MinerCandidate(uid=5, hotkey="a"), MinerCandidate(uid=5, hotkey="b")])
    with pytest.raises(ValidationError):
        LeaseRequest(request_id="r")  # the field is required; an old validator cannot lease by accident


async def test_round_callback_offers_a_filtered_random_subset_and_dispatches_only_to_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rlvr.config import Settings
    from rlvr.neurons import decentralized
    from rlvr.v3.reasons import RoundReason, Stage
    from rlvr.v3.round import RoundResult

    settings = Settings(
        _env_file=None, problem_server_url="https://problems.invalid",
        validator_score_state_file=str(tmp_path / "scores.json"),
    )
    live = [SimpleNamespace(uid=uid, hotkey=f"hk-{uid}") for uid in range(4)]
    seen = {}

    class Validator:
        def __init__(self, *_args, **_kwargs):
            self.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
            self.subtensor = object()
            self.metagraph = SimpleNamespace(
                hotkeys=[f"hk-{uid}" for uid in range(4)],
                validator_permit=[False, False, True, False],  # uid 2 is a validator
                sync=lambda **_: None,
            )

        def setup_bittensor(self):
            pass

        def set_round_callback(self, callback):
            self.callback = callback

        def set_weight_setter(self, _setter):
            pass

        def defer_rounds_until(self, _deadline):
            pass

        async def run(self):
            for _ in range(4):
                await self.callback(self)

    offers = []

    async def evaluate(client, http, solvers, policy, *, cache_dir, work_dir, candidates):
        seen["solvers"] = [(s.uid, s.hotkey) for s in solvers]
        seen["candidates"] = list(candidates)
        offers.append(list(candidates))
        # lease fails, lease rejected by the validator (carries a challenge id),
        # round completes, then one more round
        if len(offers) == 2:
            return RoundResult("abandoned", "bad pool", (), reason_code=RoundReason.SLOT_POOL_MISMATCH,
                               stage=Stage.LEASE, challenge_id="c", task_id="a" * 64)
        if len(offers) == 3:
            return RoundResult("completed", "", (), challenge_id="c", task_id="a" * 64)
        return RoundResult("unavailable", "no task", (), retry_after_s=1)

    monkeypatch.setattr(decentralized, "ValidatorNeuron", Validator)
    monkeypatch.setattr(decentralized, "_apply_weights_rate_limit", lambda *_: None)
    monkeypatch.setattr(decentralized, "v3_round_policy", lambda *_a, **_k: object())
    monkeypatch.setattr(decentralized, "_solver_clients", lambda *_a, **_k: live)
    monkeypatch.setattr(decentralized, "evaluate_round", evaluate)

    await decentralized._run_decentralized_validator_async(settings)

    assert set(seen["candidates"]) == {(1, "hk-1"), (3, "hk-3")}  # not the owner, not the validator
    assert set(seen["solvers"]) == set(seen["candidates"])  # dispatch goes only to the offered miners
    assert offers[0] == offers[1] == offers[2]  # kept through a failed lease and a rejected one
    assert len(offers) == 4 and set(offers[3]) == set(offers[0])  # only a completed round earns a fresh draw


def test_next_offer_keeps_the_order_across_failed_leases():
    eligible = [(uid, f"hk-{uid}") for uid in range(1, 8)]
    first = next_offer(None, eligible, rng=random.Random(1))
    again = next_offer(first, eligible, rng=random.Random(99))
    assert again == first  # a rejected lease is retried with the same order
    # a miner that stopped serving drops out; new ones go to the end
    changed = [item for item in eligible if item[0] != 3] + [(20, "hk-20"), (21, "hk-21")]
    third = next_offer(first, changed, rng=random.Random(5))
    assert third[: len(first) - 1] == [item for item in first if item[0] != 3]
    assert set(third[len(first) - 1 :]) == {(20, "hk-20"), (21, "hk-21")}


def test_next_offer_without_a_previous_order_is_a_fresh_shuffle():
    eligible = [(uid, f"hk-{uid}") for uid in range(1, 50)]
    assert next_offer(None, eligible, rng=random.Random(1)) != next_offer(None, eligible, rng=random.Random(2))
    assert eligible_miners([(0, "owner"), (1, "a")]) == [(1, "a")]


async def test_round_callback_logs_and_bounds_a_lease_pause(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from rlvr.config import Settings
    from rlvr.neurons import decentralized
    from rlvr.neurons.decentralized import MAX_LEASE_DEFERRAL_S
    from rlvr.v3.reasons import RoundReason, Stage
    from rlvr.v3.round import RoundResult

    settings = Settings(_env_file=None, problem_server_url="https://problems.invalid",
                        validator_score_state_file=str(tmp_path / "scores.json"))
    deferrals = []

    class Validator:
        def __init__(self, *_args, **_kwargs):
            self.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
            self.subtensor = object()
            self.metagraph = SimpleNamespace(hotkeys=["owner", "hk-1"], sync=lambda **_: None)

        def setup_bittensor(self):
            pass

        def set_round_callback(self, callback):
            self.callback = callback

        def set_weight_setter(self, _setter):
            pass

        def defer_rounds_until(self, deadline):
            deferrals.append(deadline)

        async def run(self):
            await self.callback(self)

    async def evaluate(*_a, **_k):
        return RoundResult("unavailable", "service paused", (), retry_after_s=1_209_600,
                           reason_code=RoundReason.LEASE_UNAVAILABLE, stage=Stage.LEASE)

    import time

    monkeypatch.setattr(decentralized, "ValidatorNeuron", Validator)
    monkeypatch.setattr(decentralized, "_apply_weights_rate_limit", lambda *_: None)
    monkeypatch.setattr(decentralized, "v3_round_policy", lambda *_a, **_k: object())
    monkeypatch.setattr(decentralized, "_solver_clients", lambda *_a, **_k: [SimpleNamespace(uid=1, hotkey="hk-1")])
    monkeypatch.setattr(decentralized, "evaluate_round", evaluate)
    before = time.monotonic()
    await decentralized._run_decentralized_validator_async(settings)
    assert len(deferrals) == 1 and deferrals[0] - before <= MAX_LEASE_DEFERRAL_S + 5
    assert "no lease: service paused (next attempt in 900s)" in capsys.readouterr().out
