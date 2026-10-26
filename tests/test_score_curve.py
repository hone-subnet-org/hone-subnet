"""Contract tests for the release score curve.

  * a miner's score is mean ** score_exponent * min(observations / 20, 1);
  * the ramp is linear below 20 observations and exactly 1 at or above it,
    and the exponent does not bend it;
  * an exponent of 1 reproduces the previous scoring exactly;
  * an exponent of 1.5 keeps the miner order and exact zeros while widening
    the better/worse weight ratio;
  * bad rewards stay safe, and saved histories restore without being cleared.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rlvr.neurons.decentralized import _load_scores, _save_scores
from rlvr.policy import LEGACY_SCORE_WINDOW_SECONDS, RELEASE_POLICY, ValidatorPolicy
from rlvr.scoring.eval_engine import EvalEngine, history_score

RAMP = 20


def _engine(num_uids: int = 4, exponent: float = RELEASE_POLICY.score_exponent):
    return EvalEngine(
        num_uids,
        LEGACY_SCORE_WINDOW_SECONDS,
        RELEASE_POLICY.score_window_max_samples,
        RELEASE_POLICY.score_window_min_samples,
        decay=RELEASE_POLICY.decay_nonresponders,
        score_exponent=exponent,
    )


def _observe(engine: EvalEngine, uid: int, rewards: list[float]) -> None:
    for reward in rewards:
        engine.update({uid: reward}, dispatched={uid})


def _rewards(passes: int, total: int) -> list[float]:
    return [1.0] * passes + [0.0] * (total - passes)


# --------------------------------------------------------------------------- #
# The ramp
# --------------------------------------------------------------------------- #
def test_ramp_is_twenty_observations_for_every_validator():
    assert RELEASE_POLICY.score_window_min_samples == RAMP


@pytest.mark.parametrize("observations", range(1, RAMP))
def test_ramp_is_linear_below_the_threshold(observations):
    engine = _engine()
    _observe(engine, 0, [1.0] * observations)

    assert engine.scores[0] == pytest.approx(observations / RAMP)


@pytest.mark.parametrize("observations", [RAMP, RAMP + 1, 100, 150])
def test_ramp_is_exactly_one_at_or_above_the_threshold(observations):
    engine = _engine()
    _observe(engine, 0, [1.0] * observations)

    assert engine.scores[0] == 1.0


@pytest.mark.parametrize("exponent", [1.0, 1.5, 3.0])
@pytest.mark.parametrize("mean", [0.25, 0.5, 0.8])
def test_exponent_shapes_quality_but_not_the_ramp(exponent, mean):
    full = history_score([mean] * RAMP, RAMP, exponent)
    assert full == pytest.approx(mean**exponent)

    for observations in (1, 5, 10, 19):
        thin = history_score([mean] * observations, RAMP, exponent)
        # The ramp factor is n / 20, never (n / 20) ** exponent.
        assert thin / full == pytest.approx(observations / RAMP)


# --------------------------------------------------------------------------- #
# The exponent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("min_samples", [4, RAMP])
def test_exponent_one_reproduces_the_previous_scoring(min_samples):
    rng = np.random.default_rng(7)
    for length in (1, 3, 4, 10, 19, 20, 57, 100):
        rewards = [float(x) for x in rng.random(length)]
        previous = sum(rewards) / max(len(rewards), min_samples)
        assert history_score(rewards, min_samples, 1.0) == pytest.approx(previous)


def test_exponent_one_reproduces_previous_mature_scores_exactly():
    engine = _engine(exponent=1.0)
    histories = [_rewards(passes, 100) for passes in (90, 60, 30, 0)]
    for uid, rewards in enumerate(histories):
        _observe(engine, uid, rewards)

    previous = np.array([sum(rewards) / len(rewards) for rewards in histories])
    assert np.array_equal(engine.scores, previous)
    assert np.array_equal(engine.get_weights(), previous / previous.sum())


def test_exponent_keeps_order_and_zeros_and_widens_the_gap():
    linear = _engine(exponent=1.0)
    curved = _engine(exponent=1.5)
    for engine in (linear, curved):
        for uid, passes in enumerate([90, 60, 30, 0]):
            _observe(engine, uid, _rewards(passes, 100))

    lw = linear.get_weights()
    cw = curved.get_weights()

    assert list(np.argsort(-cw)) == list(np.argsort(-lw)) == [0, 1, 2, 3]
    assert cw[3] == 0.0
    assert cw.sum() == pytest.approx(1.0)
    assert cw[0] / cw[2] > lw[0] / lw[2]
    assert cw[0] / cw[2] == pytest.approx(3.0**1.5)


def test_release_engine_uses_the_release_exponent():
    engine = _engine()
    _observe(engine, 0, _rewards(50, 100))

    assert engine.scores[0] == pytest.approx(0.5**1.5)


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_bad_rewards_count_as_zero(bad):
    engine = _engine()
    _observe(engine, 0, [bad] * RAMP)
    _observe(engine, 1, [1.0] * RAMP)

    assert engine.scores[0] == 0.0
    assert engine.get_weights() == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_negative_mean_is_clamped_before_the_exponent():
    # A negative base would make a fractional power complex.
    assert history_score([-0.5, -0.5], RAMP, 1.5) == 0.0
    assert history_score([], RAMP, 1.5) == 0.0


def test_a_huge_payment_overflows_to_zero_weight_without_raising():
    engine = _engine()
    _observe(engine, 0, [1e300] * RAMP)
    _observe(engine, 1, [1.0] * RAMP)

    assert engine.scores[0] == float("inf")
    assert engine.get_weights() == pytest.approx([0.0, 1.0, 0.0, 0.0])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_engine_rejects_an_unusable_exponent(bad):
    with pytest.raises(ValueError, match="score_exponent"):
        _engine(exponent=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_policy_rejects_an_unusable_exponent(bad):
    with pytest.raises(ValueError, match="score_exponent"):
        ValidatorPolicy(score_exponent=bad)


def test_policy_summary_names_the_exponent():
    assert "score_exponent=1.5" in RELEASE_POLICY.summary()


# --------------------------------------------------------------------------- #
# Saved histories
# --------------------------------------------------------------------------- #
def _previous_release_state() -> dict:
    """A score file as the previous release wrote it: ramp 4, no exponent."""
    histories = {
        "0": [[float(t), 1.0 if t % 10 else 0.0] for t in range(1, 101)],
        "1": [[float(t), 1.0] for t in range(1, 11)],
        "2": [[1.0, 0.0]],  # an all-miss history scores exactly zero
    }
    scores = [
        sum(r for _, r in values) / max(len(values), 4)
        for values in histories.values()
    ]
    return {
        "version": 2,
        "num_uids": 3,
        "window_seconds": LEGACY_SCORE_WINDOW_SECONDS,
        "max_samples": 100,
        "min_samples": 4,
        "scores": scores,
        "histories": histories,
        "hotkeys": {"0": "veteran", "1": "newcomer"},
    }


def test_previous_release_histories_restore_intact_and_rescore(tmp_path, capsys):
    path = tmp_path / "scores.json"
    state = _previous_release_state()
    path.write_text(json.dumps(state))
    engine = _engine(num_uids=0)

    _load_scores(engine, str(path))

    out = capsys.readouterr().out
    assert "could not restore" not in out
    assert "recomputed from histories" not in out
    assert engine.hotkeys == {0: "veteran", 1: "newcomer"}
    for uid, values in state["histories"].items():
        assert engine.histories[int(uid)] == [tuple(item) for item in values]
    assert engine.scores[0] == pytest.approx(0.9**1.5)  # mature: exponent only
    assert engine.scores[1] == pytest.approx(10 / RAMP)  # still ramping
    assert engine.scores[2] == 0.0


def test_saved_state_records_the_exponent_and_reloads_cleanly(tmp_path, capsys):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(_previous_release_state()))
    first = _engine(num_uids=0)
    _load_scores(first, str(path))
    assert _save_scores(first, str(path))
    capsys.readouterr()

    saved = json.loads(path.read_text())
    assert saved["score_exponent"] == RELEASE_POLICY.score_exponent
    assert saved["min_samples"] == RAMP

    second = _engine(num_uids=0)
    _load_scores(second, str(path))

    assert "recomputed from histories" not in capsys.readouterr().out
    assert second.histories == first.histories
    assert second.scores == pytest.approx(first.scores)


@pytest.mark.parametrize("bad", ["fast", None, [1.5], True, 0, -1.5, 1e400])
def test_a_malformed_saved_exponent_keeps_the_histories(tmp_path, capsys, bad):
    path = tmp_path / "scores.json"
    state = _previous_release_state()
    state["score_exponent"] = bad
    path.write_text(json.dumps(state))
    engine = _engine(num_uids=0)

    _load_scores(engine, str(path))

    out = capsys.readouterr().out
    assert "could not restore" not in out
    assert "recomputed from histories" in out
    assert len(engine.histories[0]) == 100
    assert math.isclose(engine.scores[0], 0.9**1.5)


def test_a_huge_saved_payment_restores_without_losing_histories(tmp_path, capsys):
    path = tmp_path / "scores.json"
    state = _previous_release_state()
    state["histories"]["1"][0][1] = 1e300
    path.write_text(json.dumps(state))
    engine = _engine(num_uids=0)

    _load_scores(engine, str(path))

    assert "could not restore" not in capsys.readouterr().out
    assert len(engine.histories) == 3
    assert engine.hotkeys == {0: "veteran", 1: "newcomer"}


def test_bad_derived_scores_and_all_zero_inputs_give_safe_weights():
    engine = _engine()
    assert not engine.get_weights().any()

    _observe(engine, 0, [0.0] * RAMP)
    assert not engine.get_weights().any()

    engine.scores[:] = [float("nan"), float("inf"), -1.0, 0.5]
    assert engine.get_weights() == pytest.approx([0.0, 0.0, 0.0, 1.0])
