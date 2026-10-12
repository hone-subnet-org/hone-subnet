"""Tests for rlvr.scoring: Verifier, difficulty functions, EvalEngine.

A tiny FAKE executor returns scripted ExecutionResults so these tests do not
depend on the execution module or on running real code.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlvr.config import get_settings
from rlvr.policy import RELEASE_POLICY, ValidatorPolicy
from rlvr.scoring.difficulty import (
    classify_band,
    pass_rate,
)
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.scoring.verifier import Verifier
from rlvr.types import (
    DifficultyBand,
    ExecutionResult,
    MinerOutcome,
    Problem,
    SolutionResponse,
    Verification,
)
from rlvr.types import TestCase as Case  # aliased so pytest doesn't try to collect it


# --------------------------------------------------------------------------- #
# Fake executor
# --------------------------------------------------------------------------- #
class FakeExecutor:
    """Returns scripted results. `script` is a list-of-runs; each run is a list of
    (passed, error) tuples (one per test). When the runs list is exhausted the last
    run is repeated, so a single deterministic run can be supplied as [[...]].

    Mirrors the real executor's contract: a test that errored produced NO usable
    value (`value_ok` False); a test with no error was exercised and returned a
    serializable value (`value_ok` True), regardless of pass/fail."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def run_tests(self, code, entrypoint, tests, timeout_s):
        run = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        out = []
        for i, (passed, error) in enumerate(run):
            out.append(
                ExecutionResult(
                    test_index=i,
                    passed=passed,
                    value_ok=error is None,
                    actual_repr=None if error else "ok",
                    error=error,
                    error_kind=(
                        "compile_error"
                        if error and ("NameError" in error or "not defined" in error)
                        else ("runtime_error" if error else None)
                    ),
                    timed_out=False,
                    runtime_ms=1.0,
                )
            )
        return out


def make_problem(n_tests=3) -> Problem:
    return Problem(
        problem_id="p1",
        language="python",
        statement="add",
        entrypoint="f",
        tests=[Case(args=[i], expected=i) for i in range(n_tests)],
    )


def make_solution() -> SolutionResponse:
    return SolutionResponse(problem_id="p1", code="def f(x):\n    return x")


def passes(n):
    return [(True, None)] * n


def fails(n):
    return [(False, "AssertionError") for _ in range(n)]


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
def test_verify_all_pass_reward_one():
    s = get_settings()
    ex = FakeExecutor([passes(3)])  # one run repeated for determinism
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert isinstance(out, Verification)
    assert out.all_passed is True
    assert out.num_passed == 3
    assert out.num_tests == 3
    assert out.reward == 1.0
    assert out.compile_error is None
    assert out.flagged_flaky is False


def test_verify_some_fail_reward_zero_near_binary():
    s = get_settings()
    assert s.reward_partial_credit is False
    run = [(True, None), (True, None), (False, "AssertionError")]
    ex = FakeExecutor([run])
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert out.all_passed is False
    assert out.num_passed == 2
    assert out.reward == 0.0  # near-binary: not all -> 0


def test_verify_partial_credit_when_enabled():
    s = get_settings()
    s.reward_partial_credit = True
    run = [(True, None), (True, None), (False, "AssertionError")]
    ex = FakeExecutor([run])
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert out.num_passed == 2
    assert out.reward == pytest.approx(2 / 3)
    assert out.all_passed is False


def test_verify_compile_error_all_tests_errored():
    s = get_settings()
    err = "NameError: name 'f' is not defined"
    # every test failed with the same error -> compile failure
    run = [(False, err), (False, err), (False, err)]
    ex = FakeExecutor([run])
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert out.compile_error is not None
    assert "NameError" in out.compile_error
    assert out.reward == 0.0
    assert out.num_passed == 0
    assert out.all_passed is False


def test_verify_runtime_error_on_one_test_is_not_compile_error():
    s = get_settings()
    # some tests pass -> NOT a compile error even though one errored
    run = [(True, None), (False, "ZeroDivisionError"), (True, None)]
    ex = FakeExecutor([run])
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert out.compile_error is None
    assert out.num_passed == 2
    assert out.reward == 0.0


def test_verify_wrong_values_every_test_is_not_compile_error():
    """LOW fix: a candidate that RETURNS a (wrong but serializable) value on every
    test has value_ok=True everywhere, so it is scored as failing tests — NOT
    masked as a compile/define failure."""
    s = get_settings()
    # every test fails by VALUE comparison (no error, value_ok True) -> ran fine,
    # just wrong answers. The executor's signal says the entrypoint is callable.
    run = [(False, None), (False, None), (False, None)]
    ex = FakeExecutor([run])
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert out.compile_error is None  # not a compile error: code was exercised
    assert out.num_passed == 0
    assert out.all_passed is False
    assert out.reward == 0.0


def test_verify_undefined_entrypoint_is_compile_error():
    s = get_settings()
    err = "entrypoint 'f' is not defined by the candidate"
    run = [(False, err), (False, err), (False, err)]
    ex = FakeExecutor([run])  # repeated for both determinism runs
    v = Verifier(ex, s)
    out = v.verify(make_problem(3), make_solution())
    assert out.compile_error is not None
    assert "not defined" in out.compile_error
    assert out.reward == 0.0
    assert out.num_passed == 0
    assert out.flagged_flaky is False  # compile failure short-circuits flaky logic


def test_verify_runtime_crash_on_every_test_is_not_compile_error():
    settings = get_settings()
    run = [(False, "ValueError: bad input")] * 3
    out = Verifier(FakeExecutor([run]), settings).verify(
        make_problem(3), make_solution()
    )

    assert out.compile_error is None
    assert out.reward == 0.0


def test_verify_flaky_test_counts_as_failure_not_dropped():
    """EXPLOIT GUARD: a test that flips across runs must count as a FAILURE, never
    be silently excluded. A miner who makes one hidden test nondeterministic while
    passing the rest must NOT score 1.0."""
    s = get_settings()
    s.reward_partial_credit = False
    # test index 2 flips: passes in run0, fails in run1 -> flaky.
    run0 = [(True, None), (True, None), (True, None)]
    run1 = [(True, None), (True, None), (False, "Flaky")]
    ex = FakeExecutor([run0, run1])
    v = Verifier(ex, s, policy=ValidatorPolicy(verification_runs=2))
    out = v.verify(make_problem(3), make_solution())
    assert out.flagged_flaky is True
    # the flaky test is counted as failed, NOT excluded -> not all passed.
    assert out.num_passed == 2  # only the 2 deterministically-passing tests
    assert out.all_passed is False
    assert out.reward == 0.0  # the exploit must not yield reward 1.0


def test_verify_flaky_exploit_partial_credit_does_not_drop_test():
    """Even under partial credit, a flaky test counts against the fraction; it is
    never dropped to shrink the denominator."""
    s = get_settings()
    s.reward_partial_credit = True
    # idx2 flips -> flaky; the other two pass deterministically.
    run0 = [(True, None), (True, None), (True, None)]
    run1 = [(True, None), (True, None), (False, "Flaky")]
    ex = FakeExecutor([run0, run1])
    v = Verifier(ex, s, policy=ValidatorPolicy(verification_runs=2))
    out = v.verify(make_problem(3), make_solution())
    assert out.flagged_flaky is True
    assert out.num_passed == 2
    # denominator is the FULL suite (3), not the reduced (2) -> 2/3, not 1.0.
    assert out.reward == pytest.approx(2 / 3)
    assert out.all_passed is False


def test_verify_flaky_with_remaining_failure_zero_reward():
    s = get_settings()
    s.reward_partial_credit = False
    # idx2 flaky (counts as fail); idx1 stably fails -> not all passed.
    run0 = [(True, None), (False, "err"), (True, None)]
    run1 = [(True, None), (False, "err"), (False, "flip")]
    ex = FakeExecutor([run0, run1])
    v = Verifier(ex, s, policy=ValidatorPolicy(verification_runs=2))
    out = v.verify(make_problem(3), make_solution())
    assert out.flagged_flaky is True
    # idx0 passes deterministically; idx1 stably fails; idx2 flaky (fails).
    assert out.num_passed == 1
    assert out.all_passed is False
    assert out.reward == 0.0


def test_verify_correct_solution_still_scores_one():
    """A genuinely-correct solution (passes every test in EVERY run) still scores
    1.0 and is not flagged flaky."""
    s = get_settings()
    s.reward_partial_credit = False
    runs = [passes(3), passes(3), passes(3)]
    ex = FakeExecutor(runs)
    v = Verifier(ex, s, policy=ValidatorPolicy(verification_runs=3))
    out = v.verify(make_problem(3), make_solution())
    assert out.flagged_flaky is False
    assert out.num_passed == 3
    assert out.all_passed is True
    assert out.reward == 1.0
    assert out.compile_error is None


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #
def _outcome(uid, reward):
    sol = SolutionResponse(problem_id="p1", code="x")
    ver = Verification(
        problem_id="p1",
        num_tests=1,
        num_passed=1 if reward > 0 else 0,
        all_passed=reward > 0,
        reward=reward,
    )
    return MinerOutcome(uid=uid, solution=sol, verification=ver, reward=reward)


def test_pass_rate():
    outs = [_outcome(0, 1.0), _outcome(1, 0.0), _outcome(2, 1.0), _outcome(3, 0.0)]
    assert pass_rate(outs) == 0.5


def test_pass_rate_empty():
    assert pass_rate([]) == 0.0


def test_pass_rate_all_pass():
    assert pass_rate([_outcome(0, 1.0), _outcome(1, 1.0)]) == 1.0


def _partial_outcome(uid, reward, num_passed, num_tests):
    """An outcome where reward and all_passed can disagree (partial credit)."""
    sol = SolutionResponse(problem_id="p1", code="x")
    ver = Verification(
        problem_id="p1",
        num_tests=num_tests,
        num_passed=num_passed,
        all_passed=num_passed == num_tests,
        reward=reward,
    )
    return MinerOutcome(uid=uid, solution=sol, verification=ver, reward=reward)


def test_pass_rate_partial_credit_counts_only_full_passes():
    # Under reward_partial_credit a partial solve has reward>0 but all_passed=False.
    # p must reflect ONLY full passes, not the inflated reward>0 count.
    outs = [
        _partial_outcome(0, 1.0, 3, 3),  # full pass
        _partial_outcome(1, 2 / 3, 2, 3),  # partial: reward>0 but NOT all_passed
        _partial_outcome(2, 1 / 3, 1, 3),  # partial: reward>0 but NOT all_passed
        _partial_outcome(3, 0.0, 0, 3),  # full fail
    ]
    # reward>0 would (wrongly) give 3/4=0.75; all_passed gives 1/4=0.25.
    assert pass_rate(outs) == pytest.approx(0.25)


def test_classify_band_thresholds():
    low, high = 0.35, 0.65
    assert classify_band(0.9, low, high) == DifficultyBand.EASY
    assert classify_band(0.1, low, high) == DifficultyBand.HARD
    assert classify_band(0.5, low, high) == DifficultyBand.IN_BAND
    # boundaries: at low and at high are IN_BAND (strict comparisons outside)
    assert classify_band(low, low, high) == DifficultyBand.IN_BAND
    assert classify_band(high, low, high) == DifficultyBand.IN_BAND
    assert classify_band(high + 1e-9, low, high) == DifficultyBand.EASY
    assert classify_band(low - 1e-9, low, high) == DifficultyBand.HARD


# --------------------------------------------------------------------------- #
# EvalEngine
# --------------------------------------------------------------------------- #
def test_eval_engine_init_zeros():
    e = EvalEngine(num_uids=4, window_seconds=100, max_samples=10, min_samples=10)
    assert e.scores.shape == (4,)
    assert np.all(e.scores == 0.0)


def test_eval_engine_rolling_window_update():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2)
    e.update({0: 1.0, 1: 0.0})
    # A fixed denominator gives one success half credit until the window fills.
    assert e.scores[0] == pytest.approx(0.5)
    assert e.scores[1] == pytest.approx(0.0)
    assert e.scores[2] == pytest.approx(0.0)  # uid 2 not updated
    e.update({0: 1.0})
    assert e.scores[0] == pytest.approx(1.0)


def test_count_window_bounds_startup_without_expiring_old_results():
    now = [0.0]
    e = EvalEngine(
        num_uids=1,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
        clock=lambda: now[0],
    )
    e.update({0: 0.0}, dispatched={0})
    assert e.scores[0] == pytest.approx(0.0)
    now[0] = 1.0
    e.update({0: 1.0}, dispatched={0})
    assert e.scores[0] == pytest.approx(1 / 4)
    now[0] = 2.0
    e.update({0: 1.0}, dispatched={0})
    assert e.scores[0] == pytest.approx(1 / 2)
    now[0] = 100.5
    e.update({0: 1.0}, dispatched={0})
    assert e.scores[0] == pytest.approx(3 / 4)
    assert [reward for _, reward in e.histories[0]] == [0.0, 1.0, 1.0, 1.0]


def test_observations_do_not_expire_by_age():
    now = [0.0]
    e = EvalEngine(
        num_uids=1,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
        clock=lambda: now[0],
    )
    e.update({0: 1.0}, dispatched={0})
    assert e.scores[0] == pytest.approx(1 / 4)

    now[0] = 200.0
    assert e.get_weights()[0] == pytest.approx(1.0)
    assert len(e.histories[0]) == 1

    e.update({0: 1.0}, dispatched={0})
    assert e.scores[0] == pytest.approx(2 / 4)
    assert e.histories[0] == [(0.0, 1.0), (200.0, 1.0)]


def test_hybrid_window_keeps_only_six_latest_observations():
    e = EvalEngine(
        num_uids=1,
        window_seconds=100,
        max_samples=6,
        min_samples=4,
    )
    for reward in range(7):
        e.update({0: float(reward)}, dispatched={0})

    assert [reward for _, reward in e.histories[0]] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    assert e.scores[0] == pytest.approx(3.5)


def test_eval_engine_only_responders_updated():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=1, min_samples=1)
    e.update({1: 0.8})
    assert e.scores[0] == 0.0
    assert e.scores[1] == pytest.approx(0.8)  # one observation fills this window
    assert e.scores[2] == 0.0


def test_eval_engine_weights_l1_normalized():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=6, min_samples=4)
    e.update({0: 1.0, 1: 3.0})  # scores [1,3,0]
    w = e.get_weights()
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.25)
    assert w[1] == pytest.approx(0.75)
    assert w[2] == pytest.approx(0.0)


def test_eval_engine_weights_all_zero_safe():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=10, min_samples=10)
    w = e.get_weights()
    assert w.shape == (3,)
    assert np.all(w == 0.0)
    assert not np.any(np.isnan(w))


def test_eval_engine_nan_safe():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    e.update({0: float("nan"), 1: float("inf")})
    # non-finite rewards clamped to 0 on update
    assert np.all(np.isfinite(e.scores))
    # inject a non-finite score directly to exercise get_weights nan-safety
    e.scores[0] = float("nan")
    e.scores[1] = 2.0
    w = e.get_weights()
    assert not np.any(np.isnan(w))
    assert w[1] == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.0)


def test_eval_engine_grows_capacity():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    e.update({5: 1.0})  # uid beyond initial size
    assert e.scores.size == 6
    assert e.scores[5] == pytest.approx(1.0)
    w = e.get_weights()
    assert w.size == 6
    assert w[5] == pytest.approx(1.0)


def test_eval_engine_negative_scores_clamped_in_weights():
    e = EvalEngine(num_uids=2, window_seconds=100, max_samples=1, min_samples=1)
    e.scores[0] = -1.0
    e.scores[1] = 2.0
    w = e.get_weights()
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.0)
    assert w[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# compute_payments: complete-suite correctness gate + speed tiebreaker
# --------------------------------------------------------------------------- #
from rlvr.scoring.payment import compute_payments  # noqa: E402


def _outcome(
    uid: int,
    passed: bool,
    code: str = "def f(x):\n    return x\n",
    reward: float | None = None,
    latency_ms: float = 0.0,
) -> MinerOutcome:
    r = (1.0 if passed else 0.0) if reward is None else reward
    return MinerOutcome(
        uid=uid,
        hotkey=f"hk{uid}",
        solution=SolutionResponse(
            problem_id="p", code=code, latency_ms=latency_ms
        ),
        verification=Verification(
            problem_id="p", num_tests=4, num_passed=4 if passed else 0,
            all_passed=passed, reward=r,
        ),
        reward=r,
    )


def test_payment_sole_solver_earns_full():
    outs = [_outcome(0, True, code="def f(x):\n    return x\n")] + [
        _outcome(i, False, code=f"def f(x):\n    return {i}\n") for i in (1, 2, 3)
    ]
    pay = compute_payments(outs)
    assert pay[0] == pytest.approx(1.0)
    assert pay[1] == pay[2] == pay[3] == 0.0


def test_payment_all_pass_pays_full_at_equal_latency():
    outs = [_outcome(i, True, code=f"def f(x):\n    return x + {i} - {i}\n") for i in range(4)]
    pay = compute_payments(outs)
    assert all(pay[i] == pytest.approx(1.0) for i in range(4))


def test_payment_does_not_scale_with_rarity():
    outs = [
        _outcome(0, True, code="def f(x):\n    return x\n"),
        _outcome(1, True, code="def f(y):\n    return y + 0\n"),  # structurally distinct
        _outcome(2, False),
        _outcome(3, False),
    ]
    pay = compute_payments(outs)
    assert pay[0] == pytest.approx(1.0)
    assert pay[1] == pytest.approx(1.0)


def test_payment_identical_solutions_are_scored_independently_in_v1():
    outs = [
        _outcome(0, True, code="def f(x):\n    return x\n"),
        _outcome(1, True, code="# padded\ndef f(x):\n\n    return x\n"),
        _outcome(2, True, code="def f(x):\n    return 0 + x\n"),
        _outcome(3, False),
    ]
    pay = compute_payments(outs)
    assert pay[0] == pytest.approx(1.0)
    assert pay[1] == pytest.approx(1.0)
    assert pay[2] == pytest.approx(1.0)


def test_payment_is_flat_without_a_difficulty_setting():
    outs = [_outcome(0, True), _outcome(1, False)]
    pay = compute_payments(outs)
    assert pay[0] == pytest.approx(1.0)
    assert pay[1] == 0.0


def test_payment_requires_all_tests_even_with_partial_credit_reward():
    outs = [
        _outcome(0, False, reward=0.5, code="def f(x):\n    return x\n"),
        _outcome(1, False, reward=0.0),
    ]
    pay = compute_payments(outs)
    assert pay[0] == 0.0
    assert pay[1] == 0.0


def test_payment_single_responder_and_empty_pool():
    assert compute_payments([]) == {}
    pay = compute_payments([_outcome(7, True)])
    assert pay == {7: pytest.approx(1.0)}


def test_payment_speed_halves_for_each_relative_latency_half_life():
    outs = [
        _outcome(0, True, code="def f(x):\n    return x\n", latency_ms=100.0),
        _outcome(1, True, code="def f(x):\n    return x + 0\n", latency_ms=350.0),
        _outcome(2, True, code="def f(x):\n    return 0 + x\n", latency_ms=600.0),
    ]
    pay = compute_payments(
        outs,
        speed_half_life_ms=250.0,
        speed_floor=0.0,
    )
    assert pay[0] == pytest.approx(1.0)
    assert pay[1] == pytest.approx(0.5)
    assert pay[2] == pytest.approx(0.25)


def test_default_payment_speed_is_a_light_tiebreaker():
    outs = [
        _outcome(0, True, code="def f(x):\n    return x\n", latency_ms=100.0),
        _outcome(1, True, code="def f(x):\n    return x + 0\n", latency_ms=180_100.0),
    ]
    pay = compute_payments(outs)
    assert pay[0] == pytest.approx(1.0)
    assert pay[1] == pytest.approx(0.975)


def test_release_policy_pays_every_pass_in_full_whatever_its_latency():
    release = {
        "speed_half_life_ms": RELEASE_POLICY.payment_speed_half_life_ms,
        "speed_floor": RELEASE_POLICY.payment_speed_floor,
    }
    mixed = [
        _outcome(0, True, code="def f(x):\n    return x\n", latency_ms=100.0),
        _outcome(1, True, code="def f(x):\n    return x + 0\n", latency_ms=10**9),
        _outcome(2, True, code="def f(x):\n    return 0 + x\n", latency_ms=0.0),  # no usable latency
        _outcome(3, False, code="def f(x):\n    return 1\n", latency_ms=1.0),
    ]
    assert compute_payments(mixed, **release) == {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0}
    unusable = [_outcome(0, True, code="def f(x):\n    return x\n", latency_ms=float("nan"))]
    assert compute_payments(unusable, **release) == {0: 1.0}


def test_payment_speed_is_correctness_gated_and_respects_floor():
    outs = [
        _outcome(0, False, latency_ms=1.0),
        _outcome(1, True, code="def f(x):\n    return x\n", latency_ms=100.0),
        _outcome(2, True, code="def f(x):\n    return x + 0\n", latency_ms=10_100.0),
        _outcome(3, True, code="def f(x):\n    return 0 + x\n", latency_ms=0.0),
    ]
    pay = compute_payments(
        outs,
        speed_half_life_ms=250.0,
        speed_floor=0.001,
    )
    assert pay[0] == 0.0
    assert pay[1] == pytest.approx(1.0)
    assert pay[2] == pytest.approx(0.001, abs=1e-12)
    assert pay[3] == pytest.approx(0.001)


def test_payment_speed_skips_weighting_when_no_valid_latency_exists():
    outs = [
        _outcome(0, True, code="def f(x):\n    return x\n", latency_ms=0.0),
        _outcome(1, True, code="def f(x):\n    return x + 0\n", latency_ms=-1.0),
    ]
    pay = compute_payments(
        outs,
        speed_half_life_ms=250.0,
        speed_floor=0.001,
    )
    assert pay == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}


# --------------------------------------------------------------------------- #
# EvalEngine: subset-dispatch decay semantics
# --------------------------------------------------------------------------- #
def test_update_with_dispatched_freezes_unsampled():
    e = EvalEngine(num_uids=4, window_seconds=100, max_samples=2, min_samples=2)
    e.update({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    e.update({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    # Problem dispatched to {0, 1}; both responded (uid1 failed with 0.0).
    e.update({0: 1.0, 1: 0.0}, dispatched={0, 1})
    assert e.scores[0] == pytest.approx(1.0)
    assert e.scores[1] == pytest.approx(0.5)
    # uids 2,3 were NOT sampled -> frozen, not decayed.
    assert e.scores[2] == pytest.approx(1.0)
    assert e.scores[3] == pytest.approx(1.0)


def test_update_with_dispatched_decays_silent_sampled():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2)
    e.update({0: 1.0, 1: 1.0, 2: 1.0})
    e.update({0: 1.0, 1: 1.0, 2: 1.0})
    # uid1 was dispatched but produced no outcome at all (edge case).
    e.update({0: 1.0}, dispatched={0, 1})
    assert e.scores[1] == pytest.approx(0.5)
    assert e.scores[2] == pytest.approx(1.0)  # unsampled frozen


def test_update_without_dispatched_records_full_pool_silence():
    e = EvalEngine(num_uids=3, window_seconds=100, max_samples=2, min_samples=2)
    e.update({0: 1.0, 1: 1.0, 2: 1.0})
    e.update({0: 1.0, 1: 1.0, 2: 1.0})
    e.update({0: 1.0})  # full-pool semantics: absence == dead
    assert e.scores[1] == pytest.approx(0.5)
    assert e.scores[2] == pytest.approx(0.5)


def test_record_nonserving_advances_only_absent_histories():
    e = EvalEngine(num_uids=4, window_seconds=100, max_samples=2, min_samples=2)
    e.update({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    e.update({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    e.record_nonserving([0, 2])
    assert e.scores[0] == pytest.approx(1.0)
    assert e.scores[1] == pytest.approx(0.5)
    assert e.scores[2] == pytest.approx(1.0)
    assert e.scores[3] == pytest.approx(0.5)
