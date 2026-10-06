"""Contract tests for the count-only 100-problem scoring window.

  * a uid's score averages the latest 100 COMPLETED problems observed since its
    current registration -- a pure count window, no age component;
  * with fewer than 100 observations the denominator is the observed count, not
    the cap, floored at the four-problem startup gate;
  * the previous 16-hour expiry must no longer affect scores: an observation
    leaves a history only when a 101st observation pushes it out;
  * a hotkey change at a uid still clears the whole history (a new registration
    starts empty);
  * defaults are cap=100 with the startup gate unchanged at 4;
  * an existing persisted six-entry state loads intact -- no reset, no data
    loss -- and then keeps growing up to the cap.

`window_seconds` remains accepted by the constructor and written to the v2
state file for rollback compatibility, but it no longer prunes observations.
"""

from __future__ import annotations

import json

import pytest

from rlvr.config import Settings
from rlvr.neurons.decentralized import (
    _load_scores,
    _save_scores,
    _weight_observation_count,
)
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.policy import LEGACY_SCORE_WINDOW_SECONDS, RELEASE_POLICY

DEFAULTS = Settings(_env_file=None)
TARGET_SAMPLES = 100
STARTUP_GATE = 4


def _advancing_clock(start: float = 0.0, step: float = 1.0):
    """Distinct, strictly increasing timestamps so age effects stay visible."""
    state = {"t": float(start)}

    def clock() -> float:
        state["t"] += float(step)
        return state["t"]

    return clock


def _engine(num_uids: int = 1, clock=None, max_samples: int = TARGET_SAMPLES):
    """Build an engine exactly the way the validator entrypoint does."""
    extra = {} if clock is None else {"clock": clock}
    return EvalEngine(
        num_uids,
        LEGACY_SCORE_WINDOW_SECONDS,
        max_samples,
        RELEASE_POLICY.score_window_min_samples,
        decay=RELEASE_POLICY.decay_nonresponders,
        **extra,
    )


def _observe(engine: EvalEngine, count: int, reward: float = 1.0, uid: int = 0):
    for _ in range(count):
        engine.update({uid: reward}, dispatched={uid})


# --------------------------------------------------------------------------- #
# Count boundary: 99 / 100 / 101
# --------------------------------------------------------------------------- #
def test_one_below_the_cap_divides_by_the_observed_count():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, 1, reward=0.0)
    _observe(engine, TARGET_SAMPLES - 2, reward=1.0)

    assert len(engine.histories[0]) == TARGET_SAMPLES - 1
    assert engine.scores[0] == pytest.approx((TARGET_SAMPLES - 2) / (TARGET_SAMPLES - 1))


def test_the_observation_at_the_cap_is_retained_not_evicted():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, 1, reward=0.0)
    _observe(engine, TARGET_SAMPLES - 1, reward=1.0)

    assert len(engine.histories[0]) == TARGET_SAMPLES
    # The single miss is still inside the window and still costs exactly 1/100.
    assert engine.scores[0] == pytest.approx((TARGET_SAMPLES - 1) / TARGET_SAMPLES)


def test_the_observation_past_the_cap_evicts_only_the_oldest():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, 1, reward=0.0)
    _observe(engine, TARGET_SAMPLES, reward=1.0)

    history = engine.histories[0]
    assert len(history) == TARGET_SAMPLES
    # The original miss has aged out by COUNT, not by clock.
    assert all(reward == 1.0 for _, reward in history)
    assert engine.scores[0] == pytest.approx(1.0)


def test_window_stays_at_the_cap_under_sustained_load():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, TARGET_SAMPLES + 57, reward=1.0)

    assert len(engine.histories[0]) == TARGET_SAMPLES


# --------------------------------------------------------------------------- #
# No age expiry
# --------------------------------------------------------------------------- #
def test_ancient_observations_survive_a_much_later_record():
    now = [1.0]
    engine = _engine(clock=lambda: now[0])
    _observe(engine, STARTUP_GATE, reward=1.0)

    # Far beyond the retired 16-hour bound.
    now[0] = LEGACY_SCORE_WINDOW_SECONDS * 1000.0
    _observe(engine, 1, reward=1.0)

    assert len(engine.histories[0]) == STARTUP_GATE + 1
    assert engine.scores[0] == pytest.approx(1.0)


def test_weight_gate_stays_open_across_a_long_supply_pause():
    """Supersedes the old expiry-driven gate collapse in test_eval_engine.py."""
    now = [1.0]
    engine = _engine(clock=lambda: now[0])
    _observe(engine, STARTUP_GATE, reward=1.0)
    assert _weight_observation_count(engine) == STARTUP_GATE

    now[0] = LEGACY_SCORE_WINDOW_SECONDS * 10.0
    _observe(engine, 1, reward=1.0)

    assert _weight_observation_count(engine) == STARTUP_GATE + 1


def test_a_stale_miss_is_not_forgiven_by_the_passage_of_time():
    now = [1.0]
    engine = _engine(clock=lambda: now[0])
    _observe(engine, 1, reward=0.0)

    now[0] = LEGACY_SCORE_WINDOW_SECONDS * 100.0
    _observe(engine, 3, reward=1.0)

    # 0 + 1 + 1 + 1 over four observations; the old zero still counts.
    assert len(engine.histories[0]) == 4
    assert engine.scores[0] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Fewer than the cap: average over what exists, floored by the startup gate
# --------------------------------------------------------------------------- #
def test_fewer_than_the_cap_never_divides_by_the_cap():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, 10, reward=1.0)

    assert engine.scores[0] == pytest.approx(1.0)


def test_partial_history_averages_over_the_observed_count():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, 8, reward=1.0)
    _observe(engine, 2, reward=0.0)

    assert engine.scores[0] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "observations,expected",
    [(1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0), (5, 1.0)],
)
def test_four_problem_startup_gate_still_floors_thin_evidence(observations, expected):
    engine = _engine(clock=_advancing_clock())
    _observe(engine, observations, reward=1.0)

    assert engine.scores[0] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Registration change resets the whole window
# --------------------------------------------------------------------------- #
def test_hotkey_change_clears_a_full_history():
    engine = _engine(clock=_advancing_clock())
    engine.set_hotkeys({0: "incumbent"})
    _observe(engine, TARGET_SAMPLES, reward=1.0)
    assert len(engine.histories[0]) == TARGET_SAMPLES

    assert engine.sync({0: "replacement"}) == [0]

    assert 0 not in engine.histories
    assert engine.scores[0] == pytest.approx(0.0)


def test_new_registration_restarts_at_the_startup_gate_floor():
    engine = _engine(clock=_advancing_clock())
    engine.set_hotkeys({0: "incumbent"})
    _observe(engine, TARGET_SAMPLES, reward=1.0)
    engine.sync({0: "replacement"})

    _observe(engine, 1, reward=1.0)

    assert len(engine.histories[0]) == 1
    assert engine.scores[0] == pytest.approx(1.0 / STARTUP_GATE)


def test_persisted_history_is_still_reset_by_a_hotkey_change_after_reload(tmp_path):
    path = tmp_path / "scores.json"
    source = _engine(clock=_advancing_clock())
    source.set_hotkeys({0: "incumbent"})
    _observe(source, TARGET_SAMPLES, reward=1.0)
    _save_scores(source, str(path))

    restored = _engine(num_uids=0, clock=_advancing_clock())
    _load_scores(restored, str(path))
    assert len(restored.histories[0]) == TARGET_SAMPLES

    assert restored.sync({0: "replacement"}) == [0]
    assert 0 not in restored.histories
    assert restored.scores[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_default_score_window_cap_is_one_hundred():
    assert RELEASE_POLICY.score_window_max_samples == TARGET_SAMPLES


def test_default_startup_gates_stay_at_four():
    assert RELEASE_POLICY.score_window_min_samples == STARTUP_GATE
    assert RELEASE_POLICY.min_weight_observations == STARTUP_GATE


def test_default_engine_accumulates_a_full_window_of_observations():
    engine = _engine(clock=_advancing_clock())
    _observe(engine, TARGET_SAMPLES, reward=1.0)

    assert len(engine.histories[0]) == TARGET_SAMPLES


def test_internal_engine_supports_a_smaller_test_cap():
    engine = _engine(max_samples=50, clock=_advancing_clock())
    _observe(engine, 60, reward=1.0)

    assert len(engine.histories[0]) == 50


# --------------------------------------------------------------------------- #
# Persistence migration: existing six-entry state loads without reset
# --------------------------------------------------------------------------- #
def _six_entry_state(timestamps: list[float]) -> str:
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    return json.dumps(
        {
            "version": 2,
            "num_uids": 1,
            "window_seconds": 57_600.0,
            "max_samples": 6,
            "min_samples": 4,
            "scores": [sum(rewards) / 6],
            "histories": {
                "0": [[t, r] for t, r in zip(timestamps, rewards)]
            },
            "hotkeys": {"0": "incumbent"},
        }
    )


def test_existing_six_entry_state_loads_without_reset(tmp_path, capsys):
    path = tmp_path / "scores-legacy-window.json"
    path.write_text(_six_entry_state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    engine = _engine(num_uids=0, clock=_advancing_clock(start=10.0))

    _load_scores(engine, str(path))

    assert len(engine.histories[0]) == 6
    assert engine.scores[0] == pytest.approx(4.0 / 6.0)
    assert engine.hotkeys == {0: "incumbent"}
    out = capsys.readouterr().out
    assert "could not restore" not in out
    assert "recomputed from histories" not in out


def test_loaded_six_entry_state_keeps_growing_under_the_new_cap(tmp_path):
    path = tmp_path / "scores-legacy-growth.json"
    path.write_text(_six_entry_state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    engine = _engine(num_uids=0, clock=_advancing_clock(start=10.0))
    _load_scores(engine, str(path))

    _observe(engine, 3, reward=1.0)

    assert len(engine.histories[0]) == 9


def test_loaded_ancient_entries_are_not_expired_by_the_next_record(tmp_path):
    path = tmp_path / "scores-ancient.json"
    path.write_text(_six_entry_state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    engine = _engine(num_uids=0, clock=lambda: LEGACY_SCORE_WINDOW_SECONDS * 1000.0)
    _load_scores(engine, str(path))

    _observe(engine, 1, reward=1.0)

    assert len(engine.histories[0]) == 7
    assert engine.scores[0] == pytest.approx(5.0 / 7.0)


def test_a_full_window_of_state_roundtrips_through_persistence(tmp_path):
    path = tmp_path / "scores-full-window.json"
    source = _engine(clock=_advancing_clock())
    source.set_hotkeys({0: "incumbent"})
    _observe(source, TARGET_SAMPLES, reward=1.0)
    _save_scores(source, str(path))

    stored = json.loads(path.read_text())
    assert stored["max_samples"] == TARGET_SAMPLES
    assert len(stored["histories"]["0"]) == TARGET_SAMPLES

    restored = _engine(num_uids=0, clock=_advancing_clock())
    _load_scores(restored, str(path))

    assert restored.histories == source.histories
    assert restored.scores == pytest.approx(source.scores)


def test_a_previous_release_window_of_two_hundred_loads_as_its_newest_hundred(tmp_path):
    path = tmp_path / "scores-200.json"
    previous = _engine(clock=_advancing_clock(), max_samples=200)
    previous.set_hotkeys({0: "incumbent"})
    _observe(previous, 100, reward=0.0)  # the older half: all misses
    _observe(previous, 100, reward=1.0)  # the newer half: all passes
    _save_scores(previous, str(path))
    assert json.loads(path.read_text())["max_samples"] == 200

    current = _engine(num_uids=0, clock=_advancing_clock())
    _load_scores(current, str(path))

    assert len(current.histories[0]) == TARGET_SAMPLES == 100
    assert all(reward == 1.0 for _, reward in current.histories[0])  # only the newest hundred survive
    assert current.scores[0] == pytest.approx(1.0)


def test_reducing_the_cap_keeps_only_the_latest_observations(tmp_path):
    path = tmp_path / "scores-shrink.json"
    source = _engine(clock=_advancing_clock())
    source.set_hotkeys({0: "incumbent"})
    _observe(source, TARGET_SAMPLES - 1, reward=0.0)
    _observe(source, 1, reward=1.0)
    _save_scores(source, str(path))

    smaller = _engine(num_uids=0, clock=_advancing_clock(), max_samples=10)
    _load_scores(smaller, str(path))

    assert len(smaller.histories[0]) == 10
    assert smaller.scores[0] == pytest.approx(0.1)
