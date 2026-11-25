"""Focused tests for EvalEngine deregistration reset, non-responder scoring, and
weight-length alignment to a metagraph size n.

The baseline EvalEngine behaviors (rolling update, L1 normalization, nan-safety,
capacity growth, negative-score clamping) are covered in test_scoring.py; these
tests exercise the extensions added for dereg/silence/sizing without depending on
the rest of the scoring stack.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _load_scores,
    _save_scores,
    _submit_local_weights,
    _weight_observation_count,
)
from rlvr.policy import ValidatorPolicy
from rlvr.scoring.eval_engine import EvalEngine


# --------------------------------------------------------------------------- #
# Constructor contract (extended, not broken)
# --------------------------------------------------------------------------- #
def test_decay_default_on():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=2, min_samples=2)
    assert e.decay is True


def test_constructor_still_validates():
    with pytest.raises(ValueError):
        EvalEngine(
            num_uids=-1,
            window_seconds=100,
            max_samples=2,
            min_samples=2,
        )
    with pytest.raises(ValueError):
        EvalEngine(
            num_uids=2,
            window_seconds=0,
            max_samples=2,
            min_samples=2,
        )
    with pytest.raises(ValueError):
        EvalEngine(
            num_uids=2,
            window_seconds=100,
            max_samples=2,
            min_samples=3,
        )


class _FakeSubtensor:
    def __init__(self):
        self.calls: list[dict] = []

    def set_weights(self, **kwargs):
        self.calls.append(kwargs)
        return True, "accepted"

def test_weight_submission_waits_for_configured_completed_observations():
    engine = EvalEngine(
        num_uids=2,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
    )
    settings = Settings(
        _env_file=None,
        netuid=5,
        validator_min_weight_observations=4,
    )
    subtensor = _FakeSubtensor()
    validator = SimpleNamespace(
        metagraph=SimpleNamespace(
            hotkeys=["miner", "owner"],
            coldkeys=["ck-miner", "ck-owner"],
            # UIDs deliberately differ from positions.
            uids=[77, 251],
        ),
        subtensor=subtensor,
        wallet=object(),
    )

    for _ in range(3):
        engine.update({0: 1.0, 1: 0.0}, dispatched={0, 1})
    assert _weight_observation_count(engine) == 3
    assert _submit_local_weights(validator, engine, settings) is None
    assert subtensor.calls == []

    engine.update({0: 1.0, 1: 0.0}, dispatched={0, 1})
    assert _weight_observation_count(engine) == 4
    assert _submit_local_weights(validator, engine, settings) is True
    assert len(subtensor.calls) == 1
    assert subtensor.calls[0]["netuid"] == 5
    assert subtensor.calls[0]["uids"] == [77, 251]
    assert subtensor.calls[0]["weights"] == pytest.approx([1.0, 0.0])
    assert subtensor.calls[0]["wait_for_inclusion"] is True
    assert subtensor.calls[0]["wait_for_finalization"] is False
    assert subtensor.calls[0]["wait_for_revealed_execution"] is False
    assert subtensor.calls[0]["max_attempts"] == 1


def test_six_observation_override_blocks_five_problem_rehearsal():
    engine = EvalEngine(
        num_uids=2,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
    )
    settings = Settings(_env_file=None, netuid=5)
    policy = ValidatorPolicy(min_weight_observations=6)
    subtensor = _FakeSubtensor()
    validator = SimpleNamespace(
        metagraph=SimpleNamespace(hotkeys=["miner", "owner"], coldkeys=["ck-miner", "ck-owner"], uids=[0, 251]),
        subtensor=subtensor,
        wallet=object(),
    )

    for _ in range(5):
        engine.update({0: 1.0, 1: 0.0}, dispatched={0, 1})
    assert _submit_local_weights(validator, engine, settings, policy=policy) is None
    assert subtensor.calls == []

    engine.update({0: 1.0, 1: 0.0}, dispatched={0, 1})
    assert _submit_local_weights(validator, engine, settings, policy=policy) is True
    assert len(subtensor.calls) == 1


def test_weight_gate_survives_score_state_restart(tmp_path):
    path = tmp_path / "scores.json"
    source = EvalEngine(
        num_uids=1,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
    )
    source.set_hotkeys({0: "a"})
    for _ in range(4):
        source.update({0: 1.0}, dispatched={0})
    _save_scores(source, str(path))

    restored = EvalEngine(
        num_uids=0,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
    )
    _load_scores(restored, str(path))

    assert _weight_observation_count(restored) == 4


def test_weight_gate_stays_open_when_old_observations_age():
    now = [0.0]
    engine = EvalEngine(
        num_uids=1,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
        clock=lambda: now[0],
    )
    for _ in range(4):
        engine.update({0: 1.0}, dispatched={0})
    assert _weight_observation_count(engine) == 4

    now[0] = 200.0
    assert _weight_observation_count(engine) == 4

    engine.update({0: 1.0}, dispatched={0})
    assert _weight_observation_count(engine) == 5


# --------------------------------------------------------------------------- #
# Deregistration reset (HIGH): hotkey change clears inherited history
# --------------------------------------------------------------------------- #
def test_tracks_hotkeys_on_update():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: 1.0, 1: 0.5}, hotkeys={0: "hkA", 1: "hkB"})
    assert e.hotkeys == {0: "hkA", 1: "hkB"}


def test_update_without_hotkeys_preserves_old_contract():
    # update(rewards) with no hotkeys must behave exactly like before.
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2, decay=False)
    e.update({0: 1.0, 1: 0.0})
    assert e.scores[0] == pytest.approx(0.5)
    assert e.scores[1] == pytest.approx(0.0)
    assert e.scores[2] == pytest.approx(0.0)
    assert e.hotkeys == {}


def test_sync_resets_changed_hotkey():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1, decay=False)
    e.update({0: 1.0, 1: 1.0}, hotkeys={0: "hkA", 1: "hkB"})
    assert e.scores[0] == pytest.approx(1.0)
    # uid 0 deregistered: slot reused by a new miner (hkA -> hkZ). uid 1 unchanged.
    reset = e.sync({0: "hkZ", 1: "hkB"})
    assert reset == [0]
    assert e.scores[0] == pytest.approx(0.0)  # newcomer inherits nothing
    assert e.scores[1] == pytest.approx(1.0)  # still the same miner -> kept
    assert e.hotkeys[0] == "hkZ"


def test_sync_no_change_keeps_scores():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1, decay=False)
    e.update({0: 1.0, 1: 0.5}, hotkeys={0: "hkA", 1: "hkB"})
    reset = e.sync({0: "hkA", 1: "hkB"})
    assert reset == []
    assert e.scores[0] == pytest.approx(1.0)
    assert e.scores[1] == pytest.approx(0.5)


def test_sync_first_sight_does_not_reset():
    # A uid we have a score for but no tracked hotkey yet must NOT be wiped on
    # first observation (only an actual hotkey *change* triggers a reset).
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1, decay=False)
    e.scores[0] = 0.9
    reset = e.sync({0: "hkA"})
    assert reset == []
    assert e.scores[0] == pytest.approx(0.9)
    assert e.hotkeys[0] == "hkA"


def test_sync_detects_change_after_update_tracking():
    # Hotkeys recorded via update() are visible to a later sync().
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1, decay=False)
    e.update({0: 1.0}, hotkeys={0: "hkA"})
    reset = e.sync({0: "hkNEW"})
    assert reset == [0]
    assert e.scores[0] == pytest.approx(0.0)


def test_sync_with_n_resizes():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1, decay=False)
    e.update({0: 1.0}, hotkeys={0: "hkA"})
    e.sync({0: "hkA"}, n=5)
    assert e.scores.size == 5


def test_set_hotkeys_no_scoring_side_effect():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    e.scores[0] = 0.7
    e.set_hotkeys({0: "hkA", 1: "hkB"})
    assert e.scores[0] == pytest.approx(0.7)
    assert e.hotkeys == {0: "hkA", 1: "hkB"}


# --------------------------------------------------------------------------- #
# Non-responder handling: zeros enter only on scored problem opportunities
# --------------------------------------------------------------------------- #
def test_decay_pulls_nonresponders_toward_zero():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2, decay=True)
    e.update({0: 1.0, 1: 1.0})
    e.update({0: 1.0, 1: 1.0})
    assert e.scores[0] == pytest.approx(1.0)
    # uid 1 goes silent on a completed full-pool challenge.
    e.update({0: 1.0})
    assert e.scores[0] == pytest.approx(1.0)
    assert e.scores[1] == pytest.approx(0.5)
    assert e.scores[2] == pytest.approx(0.0)


def test_decay_disabled_freezes_nonresponders():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2, decay=False)
    e.update({0: 1.0, 1: 1.0})
    e.update({0: 1.0})
    assert e.scores[1] == pytest.approx(0.5)


def test_decay_does_not_touch_empty_rounds():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=2, min_samples=2, decay=True)
    e.update({0: 1.0})  # scores [0.5, 0]
    e.update({})  # empty round: no decay, no change
    assert e.scores[0] == pytest.approx(0.5)
    assert e.scores[1] == pytest.approx(0.0)


def test_nonserving_records_once_and_never_observed_uid_stays_empty():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=3, min_samples=3, decay=True)
    e.update({0: 1.0, 1: 1.0}, dispatched={0, 1})
    before_active = list(e.histories[0])

    # This represents the non-serving half of the same completed challenge.
    e.record_nonserving([0])

    assert e.histories[0] == before_active
    assert [reward for _, reward in e.histories[1]] == [1.0, 0.0]
    assert 2 not in e.histories


def test_decay_eventually_releases_emissions():
    # A one-time responder that disconnects should see its weight share shrink
    # round over round (toward 0) instead of draining emissions forever.
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=2, min_samples=2, decay=True)
    e.update({0: 1.0, 1: 1.0})
    w_before = e.get_weights()[1]
    for _ in range(2):
        e.update({0: 1.0})  # uid 1 silent every round
    w_after = e.get_weights()[1]
    assert w_after < w_before
    assert w_after == pytest.approx(0.0, abs=0.05)


def test_decay_with_grown_capacity_stays_finite():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=2, min_samples=2, decay=True)
    e.update({5: 1.0})  # grows to size 6; slots 0..4 decay from 0 -> stay 0
    assert e.scores.size == 6
    assert np.all(np.isfinite(e.scores))
    assert e.scores[5] == pytest.approx(0.5)
    assert np.all(e.scores[:5] == 0.0)


# --------------------------------------------------------------------------- #
# Weight length alignment to metagraph n (LOW)
# --------------------------------------------------------------------------- #
def test_get_weights_aligns_to_larger_n():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: 1.0, 1: 3.0})
    w = e.get_weights(n=5)  # metagraph larger than scores seen
    assert w.size == 5
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.25)
    assert w[1] == pytest.approx(0.75)
    assert np.all(w[2:] == 0.0)


def test_get_weights_truncates_to_smaller_n():
    e = EvalEngine(num_uids=4, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: 1.0, 3: 9.0})  # the high score is a dropped tail uid
    w = e.get_weights(n=2)
    assert w.size == 2
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(1.0)  # only uid 0 survives the truncation


def test_get_weights_n_none_uses_full_size():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: 1.0})
    assert e.get_weights().size == 3
    assert e.get_weights(n=None).size == 3


def test_get_weights_n_all_zero_safe():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    w = e.get_weights(n=4)
    assert w.size == 4
    assert np.all(w == 0.0)
    assert not np.any(np.isnan(w))


def test_get_weights_n_rejects_negative():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    with pytest.raises(ValueError):
        e.get_weights(n=-1)


def test_resize_grow_and_shrink():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: 1.0, 1: 2.0}, hotkeys={0: "a", 1: "b"})
    e.resize(4)
    assert e.scores.size == 4
    assert e.scores[1] == pytest.approx(2.0)
    e.resize(1)  # drop trailing uids
    assert e.scores.size == 1
    assert e.scores[0] == pytest.approx(1.0)
    assert 1 not in e.hotkeys  # forgotten on shrink


def test_resize_noop_when_equal():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: 1.0})
    e.resize(3)
    assert e.scores.size == 3
    assert e.scores[0] == pytest.approx(1.0)


def test_resize_rejects_negative():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    with pytest.raises(ValueError):
        e.resize(-1)


# --------------------------------------------------------------------------- #
# Integration: a realistic dereg + decay + sizing sequence
# --------------------------------------------------------------------------- #
def test_dereg_then_decay_then_resize_flow():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2, decay=True)
    # Round 1: all three respond and register hotkeys.
    e.update({0: 1.0, 1: 1.0, 2: 1.0}, hotkeys={0: "a", 1: "b", 2: "c"})
    # uid 1 deregisters; metagraph reports a new hotkey at that slot.
    e.sync({0: "a", 1: "NEW", 2: "c"})
    assert e.scores[1] == pytest.approx(0.0)  # reset on dereg
    # A second success fills uid 2's two-observation window.
    e.update({0: 1.0, 2: 1.0}, hotkeys={0: "a", 2: "c"})
    s2_before = e.scores[2]
    # Next completed challenge: only uid 0 responds, so uid 2 records zero.
    e.update({0: 1.0}, hotkeys={0: "a"})
    assert e.scores[2] < s2_before  # decayed toward 0
    # Metagraph grew to 6 uids; weights must align so set_weights lengths match.
    w = e.get_weights(n=6)
    assert w.size == 6
    assert w.sum() == pytest.approx(1.0)


def test_corrupt_score_state_is_not_partially_applied(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(
        '{"version":1,"scores":[0.9],"hotkeys":{"bad-uid":"hk"}}'
    )
    engine = EvalEngine(2, window_seconds=100, max_samples=2, min_samples=2)
    engine.scores[:] = [0.1, 0.2]
    engine.hotkeys = {0: "original"}

    _load_scores(engine, str(path))

    assert engine.scores == pytest.approx([0.1, 0.2])
    assert engine.hotkeys == {0: "original"}


def test_v1_score_state_migrates_as_a_full_window(tmp_path):
    path = tmp_path / "scores-v1.json"
    path.write_text(
        '{"version":1,"scores":[0.9,0.0],"hotkeys":{"0":"a","1":"b"}}'
    )
    engine = EvalEngine(
        2,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
        clock=lambda: 50.0,
    )

    _load_scores(engine, str(path))

    assert engine.scores == pytest.approx([0.9, 0.0])
    assert engine.histories == {
        0: [(50.0, 0.9), (50.0, 0.9), (50.0, 0.9), (50.0, 0.9)],
    }
    assert '"version": 2' in path.read_text()


def test_v2_score_state_roundtrips_histories(tmp_path):
    path = tmp_path / "scores-v2.json"
    source = EvalEngine(3, window_seconds=100, max_samples=3, min_samples=3)
    source.update(
        {0: 1.0, 1: 0.0},
        hotkeys={0: "a", 1: "b"},
        dispatched={0, 1},
    )
    source.update(
        {0: 0.5, 1: 1.0},
        hotkeys={0: "a", 1: "b"},
        dispatched={0, 1},
    )
    _save_scores(source, str(path))

    restored = EvalEngine(1, window_seconds=100, max_samples=3, min_samples=3)
    _load_scores(restored, str(path))

    assert restored.scores == pytest.approx(source.scores)
    assert restored.histories == source.histories
    assert restored.hotkeys == source.hotkeys


def test_v2_stale_derived_scores_are_recomputed(tmp_path, capsys):
    path = tmp_path / "scores-stale.json"
    path.write_text(
        '{"version":2,"num_uids":2,"window_seconds":100,'
        '"max_samples":6,"min_samples":3,'
        '"scores":[999,999],"histories":{"0":[[1,1],[2,0]]},'
        '"hotkeys":{"0":"a"}}'
    )
    engine = EvalEngine(
        2,
        window_seconds=100,
        max_samples=6,
        min_samples=3,
    )

    _load_scores(engine, str(path))

    assert engine.scores == pytest.approx([1 / 3, 0.0])
    assert "recomputed from histories" in capsys.readouterr().out


def test_v2_invalid_history_is_not_partially_applied(tmp_path):
    path = tmp_path / "scores-invalid-v2.json"
    path.write_text(
        '{"version":2,"num_uids":2,"window_seconds":100,'
        '"max_samples":6,"min_samples":3,'
        '"scores":[0,0],"histories":{"0":[[1,1],[2,-1]]},'
        '"hotkeys":{"0":"replacement"}}'
    )
    engine = EvalEngine(
        2,
        window_seconds=100,
        max_samples=6,
        min_samples=3,
    )
    engine.update({0: 1.0}, hotkeys={0: "original"}, dispatched={0})
    before_scores = engine.scores.copy()
    before_histories = {uid: list(values) for uid, values in engine.histories.items()}

    _load_scores(engine, str(path))

    assert engine.scores == pytest.approx(before_scores)
    assert engine.histories == before_histories
    assert engine.hotkeys == {0: "original"}
