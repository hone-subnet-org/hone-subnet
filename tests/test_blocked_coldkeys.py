"""Blocked coldkeys: zeroed before ranking, on the local and pooled paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlvr.config import Settings
from rlvr.neurons import decentralized
from rlvr.policy import RELEASE_POLICY, ValidatorPolicy
from rlvr.scoring.eval_engine import EvalEngine, rank_weights

DECAY = RELEASE_POLICY.rank_decay
BAD = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OTHER_BAD = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
POLICY = ValidatorPolicy(blocked_coldkeys=(BAD, OTHER_BAD))
UNBLOCKED = ValidatorPolicy(blocked_coldkeys=())


def engine(scores):
    e = EvalEngine(len(scores), 1.0, 100, 30, rank_decay=DECAY)
    e.scores = np.asarray(scores, dtype=np.float64)
    e.histories = {uid: [(1.0, 0.0)] * 8 for uid in range(len(scores))}
    return e


def metagraph(coldkeys, *, permits=None, owner=None):
    n = len(coldkeys)
    return SimpleNamespace(
        hotkeys=[f"hk-{uid}" for uid in range(n)],
        coldkeys=list(coldkeys),
        uids=list(range(n)),
        validator_permit=permits if permits is not None else [False] * n,
        owner_hotkey=owner,
    )


def validator(graph, calls):
    def set_weights(**kwargs):
        calls.append(kwargs)
        return True

    return SimpleNamespace(metagraph=graph, subtensor=SimpleNamespace(set_weights=set_weights), wallet=object())


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum()


def stub_pool(scores):
    return SimpleNamespace(scores=lambda *args, **kwargs: list(scores))


def pooled(scores, blocked):
    return decentralized._pooled_weights(engine([0.0] * len(scores)), stub_pool(scores), [f"hk-{u}" for u in range(len(scores))], POLICY, 0.0, None, blocked)


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def test_the_release_lists_no_coldkeys():
    assert RELEASE_POLICY.blocked_coldkeys == ()
    assert "blocked_coldkeys=0" in RELEASE_POLICY.summary()


def test_the_list_is_part_of_the_policy_fingerprint():
    assert UNBLOCKED.fingerprint != POLICY.fingerprint != RELEASE_POLICY.fingerprint


@pytest.mark.parametrize(
    "coldkeys",
    [
        [BAD],  # a list, not a tuple
        (BAD, BAD),
        ("",),
        (BAD[:-1],),
        (BAD[:-1] + "0",),  # 0 is not base58
        ("6" + BAD[1:],),
    ],
)
def test_a_malformed_list_is_refused(coldkeys):
    with pytest.raises(ValueError):
        ValidatorPolicy(blocked_coldkeys=coldkeys)


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #
def test_every_uid_under_a_blocked_coldkey_is_masked():
    mask = decentralized._blocked_uids(metagraph(["ck-a", BAD, "ck-c", BAD, OTHER_BAD]), POLICY)
    assert mask.tolist() == [False, True, False, True, True]


def test_a_validator_permit_holder_and_the_owner_are_never_blocked(capsys):
    graph = metagraph([BAD, BAD, BAD], permits=[True, False, False], owner="hk-2")
    assert decentralized._blocked_uids(graph, POLICY).tolist() == [False, True, False]
    out = capsys.readouterr().out
    assert "uid 0: it holds a validator permit" in out and "uid 2: it is the subnet owner" in out


def test_array_permits_from_the_chain_are_read_per_uid():
    graph = metagraph([BAD, BAD], permits=np.array([False, True]))
    assert decentralized._blocked_uids(graph, POLICY).tolist() == [True, False]


@pytest.mark.parametrize("coldkeys", [None, ["ck-0"], ["ck-0", None], ["ck-0", ""]])
def test_unreadable_coldkeys_skip_the_weight_set(coldkeys, capsys):
    graph = metagraph(["ck-0", "ck-1"])
    if coldkeys is None:
        del graph.coldkeys
    else:
        graph.coldkeys = coldkeys
    calls = []
    assert decentralized._submit_local_weights(validator(graph, calls), engine([1.0, 0.5]), Settings(_env_file=None), policy=POLICY) is None
    assert not calls and "coldkeys unavailable" in capsys.readouterr().out


def test_logs_and_warnings_name_chain_uids_not_positions(capsys):
    graph = metagraph([BAD, "ck-1", BAD])
    graph.uids = [40, 41, 42]
    graph.validator_permit = [False, False, True]
    calls = []
    decentralized._submit_local_weights(validator(graph, calls), engine([0.9, 0.8, 0.0]), Settings(_env_file=None), policy=POLICY)
    out = capsys.readouterr().out
    assert f"{BAD}: uids [40]" in out and "not applied to uid 42" in out


def test_an_empty_list_needs_no_coldkeys():
    graph = metagraph(["ck-0", "ck-1"])
    del graph.coldkeys
    assert not decentralized._blocked_uids(graph, UNBLOCKED).any()


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
RAW = [0.95, 0.90, 0.85]
FIRST_BLOCKED = np.array([True, False, False])


def test_a_blocked_uid_gets_exactly_zero_locally():
    weights = engine(RAW).get_weights(n=3, blocked=FIRST_BLOCKED)
    assert weights[0] == 0.0 and weights.sum() == pytest.approx(1.0)


def test_a_blocked_uid_gets_exactly_zero_pooled():
    weights = pooled(RAW, FIRST_BLOCKED)
    assert weights[0] == 0.0 and weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("weights_of", [lambda s, b: engine(s).get_weights(n=len(s), blocked=b), pooled])
def test_the_others_are_paid_as_if_the_blocked_uid_were_absent(weights_of):
    weights = weights_of(RAW, FIRST_BLOCKED)
    assert weights.tolist() == pytest.approx([0.0, *normalized([1.0, DECAY])])
    absent = weights_of(RAW[1:], None)
    assert weights[1:] == pytest.approx(absent)


@pytest.mark.parametrize("weights_of", [lambda s, b: engine(s).get_weights(n=len(s), blocked=b), pooled])
def test_a_blocked_uid_in_the_middle_closes_the_gap(weights_of):
    raw = [0.9, 0.8, 0.7, 0.6]
    weights = weights_of(raw, np.array([False, True, False, False]))
    assert weights.tolist() == pytest.approx(normalized([1.0, 0.0, DECAY, DECAY**2]).tolist())


def test_no_mask_is_byte_identical_on_both_paths():
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.0, 1.0, 64)
    local = engine(raw)
    expected = rank_weights(raw, DECAY)
    expected = expected / expected.sum()
    assert np.array_equal(local.get_weights(n=64, blocked=None), local.get_weights(n=64))
    assert np.array_equal(local.get_weights(n=64), expected)
    assert np.array_equal(pooled(raw, None), expected)
    assert np.array_equal(pooled(raw, np.zeros(64, dtype=bool)), expected)


def test_a_mask_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError):
        engine(RAW).get_weights(n=3, blocked=np.array([True, False]))


# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
def test_submission_zeroes_blocked_uids_and_logs_what_they_lost(capsys):
    graph = metagraph([BAD, "ck-1", OTHER_BAD, "ck-3", BAD])
    calls = []
    ok = decentralized._submit_local_weights(validator(graph, calls), engine([0.95, 0.90, 0.85, 0.80, 0.0]), Settings(_env_file=None), policy=POLICY)
    assert ok is True
    sent = calls[0]["weights"]
    assert sent[0] == 0.0 and sent[2] == 0.0 and sent[4] == 0.0
    assert sent == pytest.approx(normalized([0.0, 1.0, 0.0, DECAY, 0.0]).tolist())
    unblocked = normalized(rank_weights(np.array([0.95, 0.90, 0.85, 0.80, 0.0]), DECAY))
    out = capsys.readouterr().out
    assert "zeroed 3 uid(s) under 2 coldkey(s)" in out
    assert f"would get {unblocked[0] + unblocked[2]:.2%} of weight" in out
    assert f"{BAD}: uids [0, 4]" in out and f"{OTHER_BAD}: uids [2]" in out


def test_submission_blocks_on_the_pooled_path_too(capsys):
    graph = metagraph([BAD, "ck-1", "ck-2"])
    graph.validator_trust, graph.S = [0.0] * 3, [1.0] * 3
    pool = SimpleNamespace(
        settle=lambda *args, **kwargs: None,
        active=True,
        observation_count=lambda *args, **kwargs: 10,
        scores=lambda *args, **kwargs: [0.95, 0.90, 0.85],
    )
    calls = []
    ok = decentralized._submit_local_weights(validator(graph, calls), engine([0.0] * 3), Settings(_env_file=None), policy=POLICY, pool=pool)
    assert ok is True
    assert calls[0]["weights"] == pytest.approx([0.0, *normalized([1.0, DECAY])])
    assert "zeroed 1 uid(s)" in capsys.readouterr().out


def test_blocking_every_miner_skips_the_weight_set(capsys):
    graph = metagraph([BAD, OTHER_BAD])
    calls = []
    ok = decentralized._submit_local_weights(validator(graph, calls), engine([0.9, 0.8]), Settings(_env_file=None), policy=POLICY)
    assert ok is None and not calls
    assert "all-zero; skipping" in capsys.readouterr().out


def test_nothing_is_logged_when_no_uid_is_blocked(capsys):
    calls = []
    decentralized._submit_local_weights(validator(metagraph(["ck-0", "ck-1"]), calls), engine([0.9, 0.8]), Settings(_env_file=None), policy=POLICY)
    assert calls and "blocked coldkeys" not in capsys.readouterr().out
