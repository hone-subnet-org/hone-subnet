"""Contract tests for rank-based weights.

  * miners with a positive score are ranked best first; rank r (from 0) gets
    rank_decay ** r before normalization, so the shape never depends on the
    size of the score gaps;
  * ties go to the lower UID;
  * zero, negative, NaN and infinite scores get no weight and take no rank;
  * the 20-observation ramp still applies to the score used for ranking;
  * saved histories restore without being cleared.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rlvr.neurons.decentralized import _load_scores, _save_scores
from rlvr.policy import LEGACY_SCORE_WINDOW_SECONDS, RELEASE_POLICY, ValidatorPolicy
from rlvr.scoring.eval_engine import EvalEngine, rank_weights

DECAY = 0.995


def _engine(num_uids: int = 4, rank_decay: float | None = RELEASE_POLICY.rank_decay):
    return EvalEngine(
        num_uids,
        LEGACY_SCORE_WINDOW_SECONDS,
        RELEASE_POLICY.score_window_max_samples,
        RELEASE_POLICY.score_window_min_samples,
        decay=RELEASE_POLICY.decay_nonresponders,
        rank_decay=rank_decay,
    )


def _observe(engine: EvalEngine, uid: int, passes: int, total: int) -> None:
    for i in range(total):
        engine.update({uid: 1.0 if i < passes else 0.0}, dispatched={uid})


def _normalized(values: list[float]) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    return array / array.sum()


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #
def test_release_decay_is_0_995():
    assert RELEASE_POLICY.rank_decay == DECAY


def test_each_rank_gets_decay_times_the_rank_above():
    weights = rank_weights(np.array([0.9, 0.5, 0.7]), DECAY)

    assert weights == pytest.approx([1.0, DECAY**2, DECAY])


def test_shape_does_not_depend_on_score_gaps():
    close = rank_weights(np.array([0.91, 0.89, 0.90]), DECAY)
    far = rank_weights(np.array([1.0, 0.01, 0.5]), DECAY)

    assert np.array_equal(close, far)


def test_ties_go_to_the_lower_uid():
    weights = rank_weights(np.array([0.5, 0.8, 0.5, 0.8]), DECAY)

    assert weights == pytest.approx([DECAY**2, 1.0, DECAY**3, DECAY])


def test_decay_of_one_splits_evenly_among_positive_scores():
    weights = rank_weights(np.array([0.9, 0.0, 0.1]), 1.0)

    assert weights == pytest.approx([1.0, 0.0, 1.0])


def test_release_engine_weights_follow_rank():
    engine = _engine()
    for uid, passes in enumerate([90, 30, 60, 0]):
        _observe(engine, uid, passes, 100)

    weights = engine.get_weights()

    assert weights == pytest.approx(_normalized([1.0, DECAY**2, DECAY, 0.0]))
    assert weights.sum() == pytest.approx(1.0)


def test_top_share_with_one_hundred_scoring_miners():
    engine = _engine(num_uids=100)
    engine.scores[:] = np.linspace(1.0, 0.01, 100)

    weights = engine.get_weights()

    assert weights[0] == pytest.approx((1 - DECAY) / (1 - DECAY**100))
    assert weights[:10].sum() == pytest.approx((1 - DECAY**10) / (1 - DECAY**100))


def test_without_rank_decay_weights_stay_proportional():
    engine = _engine(rank_decay=None)
    for uid, passes in enumerate([90, 30, 60, 0]):
        _observe(engine, uid, passes, 100)

    assert engine.get_weights() == pytest.approx(_normalized([90, 30, 60, 0]))


# --------------------------------------------------------------------------- #
# Ramp
# --------------------------------------------------------------------------- #
def test_a_ramping_newcomer_ranks_below_a_steady_miner():
    engine = _engine(num_uids=2)
    _observe(engine, 0, 60, 100)  # mature, 60%
    _observe(engine, 1, 10, 10)  # perfect but only 10 observations: score 0.5

    assert engine.get_weights() == pytest.approx(_normalized([1.0, DECAY]))

    _observe(engine, 1, 10, 10)  # 20 observations: ramp complete

    assert engine.get_weights() == pytest.approx(_normalized([DECAY, 1.0]))


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
def test_unusable_scores_get_no_weight_and_take_no_rank():
    engine = _engine(num_uids=6)
    engine.scores[:] = [0.0, float("nan"), 0.4, -1.0, float("inf"), 0.2]

    weights = engine.get_weights()

    assert weights == pytest.approx(_normalized([0, 0, 1.0, 0, 0, DECAY]))


def test_all_zero_scores_give_all_zero_weights():
    engine = _engine()
    assert not engine.get_weights().any()

    _observe(engine, 0, 0, 20)
    assert not engine.get_weights().any()


def test_ranks_are_taken_after_aligning_to_the_metagraph():
    engine = _engine(num_uids=3)
    # uid 2 is beyond a 2-slot metagraph. Ranking it first would leave the
    # kept uids at ranks 0 and 2, giving [DECAY**2, 1] instead of [DECAY, 1].
    engine.scores[:] = [0.2, 0.9, 0.5]

    assert engine.get_weights(n=2) == pytest.approx(_normalized([DECAY, 1.0]))
    assert engine.get_weights(n=5) == pytest.approx(
        _normalized([DECAY**2, 1.0, DECAY, 0, 0])
    )


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_engine_rejects_an_unusable_decay(bad):
    with pytest.raises(ValueError, match="rank_decay"):
        _engine(rank_decay=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_policy_rejects_an_unusable_decay(bad):
    with pytest.raises(ValueError, match="rank_decay"):
        ValidatorPolicy(rank_decay=bad)


def test_policy_summary_names_the_decay():
    assert "rank_decay=0.995" in RELEASE_POLICY.summary()


# --------------------------------------------------------------------------- #
# Saved histories
# --------------------------------------------------------------------------- #
def test_previous_release_histories_restore_intact(tmp_path, capsys):
    histories = {
        "0": [[float(t), 1.0 if t % 10 else 0.0] for t in range(1, 101)],
        "1": [[float(t), 1.0] for t in range(1, 11)],
    }
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "num_uids": 2,
                "window_seconds": LEGACY_SCORE_WINDOW_SECONDS,
                "max_samples": 100,
                "min_samples": 4,  # the previous release's ramp
                "scores": [0.9, 1.0],
                "histories": histories,
                "hotkeys": {"0": "veteran", "1": "newcomer"},
            }
        )
    )
    engine = _engine(num_uids=0)

    _load_scores(engine, str(path))

    out = capsys.readouterr().out
    assert "could not restore" not in out
    assert "recomputed from histories" not in out
    for uid, values in histories.items():
        assert engine.histories[int(uid)] == [tuple(item) for item in values]
    assert engine.scores == pytest.approx([0.9, 10 / 20])  # newcomer re-ramped
    assert engine.get_weights() == pytest.approx(_normalized([1.0, DECAY]))


def test_saved_state_round_trips_to_the_same_weights(tmp_path, capsys):
    path = tmp_path / "scores.json"
    source = _engine()
    source.set_hotkeys({0: "a", 1: "b", 2: "c", 3: "d"})
    for uid, passes in enumerate([90, 30, 60, 0]):
        _observe(source, uid, passes, 100)
    assert _save_scores(source, str(path))

    restored = _engine(num_uids=0)
    _load_scores(restored, str(path))

    assert "recomputed from histories" not in capsys.readouterr().out
    assert restored.histories == source.histories
    assert np.array_equal(restored.get_weights(), source.get_weights())
