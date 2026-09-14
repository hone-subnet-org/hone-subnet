"""Contract for ``rlvr.v3.diagnostics`` — bounded, best-effort local records of
V3 round outcomes.

    round_records(result, validator_hotkey)  -> iterator of dict records; nothing
        for results without challenge/task ids (lease unavailable)
    EvaluationLog(path, *, max_file_bytes, backup_count)
        .record_round(result, validator_hotkey) -> bool, never raises

Records are one ASCII JSON line each, keyed by a deterministic ``record_id``
(sha256 over validator hotkey, challenge, task, record kind, miner hotkey) so
consumers deduplicate without a local index.  Files rotate by bytes; an
interrupted trailing write is truncated before the next append.  Any failure
(unwritable target, symlink, rotation error, serialization error) returns False
and leaves scoring untouched.  Phase 1 reports evaluation only: abandoned
rounds keep partial diagnostics but never apply score credit.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import string

import pytest

from rlvr.problemserver.client import LeaseCategory
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.v3 import diagnostics
from rlvr.v3.api import LeaseResponse
from rlvr.v3.client import V3LeaseOutcome
from rlvr.v3.diagnostics import EvaluationLog, round_records
from rlvr.v3.grading import CheckResult, EvaluationResult
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.round import (
    MinerEvaluation,
    RoundResult,
    apply_round_scores,
    compute_round_payments,
    evaluate_round,
)
from tests.test_v3_api import lease
from tests.test_v3_round import policy

VALIDATOR = "validator-hotkey"
TASK = "t" * 64
ROUND_KEYS = {
    "schema_version", "recorded_at_ms", "validator_hotkey", "challenge_id", "task_id",
    "round_status", "round_reason_code", "score_effect", "record_id", "record_type",
    "stage", "reason", "assigned_miners",
}
MINER_KEYS = {
    "schema_version", "recorded_at_ms", "validator_hotkey", "challenge_id", "task_id",
    "round_status", "round_reason_code", "score_effect", "record_id", "record_type",
    "uid", "miner_hotkey", "status", "reason_code", "dispatch_reason_code", "stage",
    "reason", "checks_passed", "checks_executed", "checks_total", "checks_skipped",
    "grading_duration_ms", "response_latency_ms",
}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def check(check_id, outcome, exit_code=0, kind="invocation"):
    return CheckResult(check_id, kind, outcome, None if outcome == "skipped" else exit_code)


PASSED = EvaluationResult(
    "passed", "", (check("c01", "passed"), check("c02", "passed", kind="inspection")), None,
    None, Stage.CHECK,
)
STOPPED = EvaluationResult(
    "failed", "candidate output did not match the verifier",
    (check("c01", "failed", 1), check("c02", "skipped", kind="inspection")), None,
    MinerReason.CHECK_FAILED, Stage.CHECK,
)
REJECTED = EvaluationResult(
    "rejected", "patch contains a NUL byte", (), None,
    MinerReason.PATCH_STATIC_REJECTED, Stage.PATCH,
)


def evaluation(uid, result, latency_ms=10, grading_ms=5):
    return MinerEvaluation(uid, f"hk-{uid}", latency_ms, result, grading_ms)


def registrations(*uids):
    return tuple((uid, f"hk-{uid}") for uid in uids)


def completed(challenge="chal-1", evaluations=(), assigned=None, **over):
    fields = {
        "challenge_id": challenge, "task_id": TASK, "checks_total": 2,
        "assigned_miners": tuple((item.uid, item.hotkey) for item in evaluations) if assigned is None else assigned,
        "diagnostic_evaluations": tuple(evaluations),
    }
    fields.update(over)
    return RoundResult("completed", "", tuple(evaluations), **fields)


def abandoned(reason, code, stage, *, partials=(), assigned=None, **over):
    fields = {
        "reason_code": code, "stage": stage, "challenge_id": "chal-1", "task_id": TASK,
        "assigned_miners": registrations(1, 2, 3) if assigned is None else assigned,
        "diagnostic_evaluations": tuple(partials),
    }
    fields.update(over)
    return RoundResult("abandoned", reason, (), **fields)


def records(result, hotkey=VALIDATOR):
    return list(round_records(result, hotkey))


def by_uid(items):
    return {item["uid"]: item for item in items if item["record_type"] == "miner_evaluation"}


def lines(path):
    return [line for line in path.read_bytes().split(b"\n") if line]


def parsed(path):
    return [json.loads(line) for line in lines(path)]


def all_files(path):
    return sorted(p for p in path.parent.iterdir() if p.name.startswith(path.name))


# --------------------------------------------------------------------------- #
# Record content
# --------------------------------------------------------------------------- #
def test_positional_construction_of_result_types_is_unchanged():
    result = RoundResult("completed", "", ())
    assert (result.reason_code, result.stage, result.challenge_id, result.task_id) == (None,) * 4
    assert (result.assigned_miners, result.diagnostic_evaluations, result.dispatch_failures) == ((),) * 3
    assert result.checks_total is None
    verdict = EvaluationResult("passed", "", (), None)
    assert (verdict.reason_code, verdict.stage) == (None, None)


def test_unavailable_round_emits_nothing_even_with_server_detail(tmp_path):
    result = RoundResult(
        "unavailable", "https://problems.invalid said 503", (), 30,
        reason_code=RoundReason.LEASE_UNAVAILABLE, stage=Stage.LEASE,
    )
    assert records(result) == []
    log = EvaluationLog(str(tmp_path / "d" / "log.jsonl"))
    assert log.record_round(result, VALIDATOR) is True
    assert not (tmp_path / "d" / "log.jsonl").exists()


def test_completed_round_records_every_assigned_miner_with_exact_schema():
    result = completed(
        evaluations=[evaluation(1, PASSED), evaluation(2, STOPPED, 20, 7), evaluation(3, REJECTED)],
        assigned=registrations(1, 2, 3, 4),
        dispatch_failures=((4, "hk-4", MinerReason.NOT_SERVING),),
    )
    items = records(result)
    assert [item["record_type"] for item in items] == ["round_outcome"] + ["miner_evaluation"] * 4
    outcome = items[0]
    assert set(outcome) == ROUND_KEYS
    assert (outcome["round_status"], outcome["round_reason_code"], outcome["score_effect"]) == (
        "completed", None, "not_reported")
    assert (outcome["stage"], outcome["reason"], outcome["assigned_miners"]) == (None, "", 4)
    assert all(set(item) == MINER_KEYS for item in items[1:])
    miners = by_uid(items)
    assert miners[1]["status"] == "passed" and miners[1]["reason_code"] is None
    assert (miners[1]["stage"], miners[1]["reason"]) == ("check", "")
    assert (miners[1]["checks_passed"], miners[1]["checks_executed"], miners[1]["checks_total"],
            miners[1]["checks_skipped"]) == (2, 2, 2, 0)
    assert (miners[1]["grading_duration_ms"], miners[1]["response_latency_ms"]) == (5, 10)
    assert (miners[2]["status"], miners[2]["reason_code"], miners[2]["stage"]) == (
        "failed", "check_failed", "check")
    assert miners[2]["reason"] == STOPPED.reason
    assert (miners[2]["checks_passed"], miners[2]["checks_executed"], miners[2]["checks_total"],
            miners[2]["checks_skipped"]) == (0, 1, 2, 1)
    assert (miners[2]["grading_duration_ms"], miners[2]["response_latency_ms"]) == (7, 20)
    assert (miners[3]["status"], miners[3]["reason_code"], miners[3]["stage"]) == (
        "rejected", "patch_static_rejected", "patch")
    assert (miners[3]["checks_executed"], miners[3]["checks_skipped"]) == (0, 2)
    assert miners[3]["dispatch_reason_code"] is None
    assert (miners[4]["status"], miners[4]["reason_code"], miners[4]["dispatch_reason_code"],
            miners[4]["stage"], miners[4]["reason"]) == (
        "not_evaluated", "not_serving", "not_serving", "dispatch", "")
    assert (miners[4]["grading_duration_ms"], miners[4]["response_latency_ms"]) == (None, None)
    assert all(item["miner_hotkey"] == f"hk-{item['uid']}" for item in items[1:])
    assert all(
        (item["validator_hotkey"], item["challenge_id"], item["task_id"], item["schema_version"])
        == (VALIDATOR, "chal-1", TASK, 1)
        for item in items
    )


@pytest.mark.parametrize("dispatch", [
    MinerReason.NOT_SERVING, MinerReason.DISPATCH_FAILED, MinerReason.RESPONSE_UNAVAILABLE,
])
def test_local_dispatch_code_is_kept_beside_the_server_artifact_outcome(dispatch):
    server_rejection = EvaluationResult(
        "rejected", "artifact_invalid", (), None, MinerReason.ARTIFACT_INVALID, Stage.COMMIT,
    )
    result = completed(
        evaluations=[evaluation(4, server_rejection, latency_ms=0)],
        dispatch_failures=((4, "hk-4", dispatch),),
    )
    miner = by_uid(records(result))[4]
    assert (miner["status"], miner["reason_code"], miner["stage"]) == (
        "rejected", "artifact_invalid", "commit")
    assert miner["dispatch_reason_code"] == dispatch.value


def test_completed_round_falls_back_to_evaluations_when_diagnostics_are_absent():
    result = RoundResult(
        "completed", "", (evaluation(1, PASSED),), challenge_id="chal-1", task_id=TASK,
        assigned_miners=registrations(1), checks_total=2,
    )
    assert by_uid(records(result))[1]["status"] == "passed"


def test_unknown_totals_before_the_manifest_leave_skipped_counts_unknown():
    result = abandoned("commit or verifier reveal failed", RoundReason.COMMIT_FAILED, Stage.COMMIT)
    items = records(result)
    assert len(items) == 4 and "checks_total" not in items[0]
    for miner in by_uid(items).values():
        assert (miner["status"], miner["checks_total"], miner["checks_skipped"]) == (
            "not_evaluated", None, None)
        assert (miner["checks_passed"], miner["checks_executed"]) == (0, 0)
        assert (miner["reason_code"], miner["dispatch_reason_code"], miner["stage"]) == (None,) * 3
    assert (items[0]["round_reason_code"], items[0]["stage"], items[0]["score_effect"]) == (
        "commit_failed", "commit", "unchanged")


def test_late_abandoned_round_keeps_partial_diagnostics_without_score_credit():
    result = abandoned(
        "validator failed during cleanup (RuntimeError)", RoundReason.CLEANUP_FAILED,
        Stage.CLEANUP, partials=[evaluation(1, PASSED)], checks_total=2,
    )
    assert result.evaluations == ()
    items = records(result)
    assert (items[0]["round_status"], items[0]["round_reason_code"], items[0]["stage"],
            items[0]["score_effect"]) == ("abandoned", "cleanup_failed", "cleanup", "unchanged")
    miners = by_uid(items)
    assert (miners[1]["status"], miners[1]["checks_passed"], miners[1]["checks_total"]) == ("passed", 2, 2)
    assert miners[2]["status"] == miners[3]["status"] == "not_evaluated"
    assert miners[2]["checks_skipped"] == 2

    engine = EvalEngine(6, 1, 200, 4)
    before = copy.deepcopy(engine.histories)
    assert compute_round_payments(result, speed_half_life_ms=1.0, speed_floor=0.5) == {}
    assert apply_round_scores(
        result, engine, active_hotkeys={uid: f"hk-{uid}" for uid in (1, 2, 3)},
        speed_half_life_ms=1.0, speed_floor=0.5,
    ) is False
    assert engine.histories == before == {}


def test_record_ids_are_deterministic_and_change_with_every_identity_key(monkeypatch):
    base = completed(evaluations=[evaluation(1, PASSED), evaluation(2, STOPPED)])
    monkeypatch.setattr(diagnostics.time, "time", lambda: 1_000.0)
    first = records(base)
    monkeypatch.setattr(diagnostics.time, "time", lambda: 2_000.0)
    second = records(base)
    assert [item["record_id"] for item in first] == [item["record_id"] for item in second]
    assert [item["recorded_at_ms"] for item in first] == [1_000_000] * 3
    assert [item["recorded_at_ms"] for item in second] == [2_000_000] * 3
    ids = [item["record_id"] for item in first]
    assert len(set(ids)) == 3 and all(len(i) == 64 and set(i) <= set(string.hexdigits) for i in ids)

    # Different evaluation content, same identity: same ids (consumers keep the latest).
    same_identity = completed(evaluations=[evaluation(1, STOPPED), evaluation(2, PASSED)])
    assert [item["record_id"] for item in records(same_identity)] == ids

    variants = {
        "validator": records(base, "other-validator"),
        "challenge": records(dataclasses.replace(base, challenge_id="chal-2")),
        "task": records(dataclasses.replace(base, task_id="u" * 64)),
    }
    for name, items in variants.items():
        assert not {item["record_id"] for item in items} & set(ids), name
    renamed = completed(
        evaluations=[evaluation(1, PASSED), evaluation(2, STOPPED)],
        assigned=((1, "hk-1"), (2, "hk-2-renamed")),
    )
    renamed_ids = [item["record_id"] for item in records(renamed)]
    assert renamed_ids[:2] == ids[:2] and renamed_ids[2] != ids[2]


def test_detail_collapses_whitespace_strips_controls_and_bounds_bytes():
    raw = "  a\x00b\n\tc  \x1bd  " + "é" * 600
    clean = diagnostics._detail(raw)
    assert clean.startswith("ab c d ")
    assert all(ch.isprintable() for ch in clean) and "\n" not in clean
    assert len(clean.encode("utf-8")) <= 512
    assert clean.encode("utf-8").decode("utf-8")  # never cut inside a multi-byte sequence
    assert diagnostics._detail("") == ""


def test_records_are_single_ascii_lines_without_control_bytes(tmp_path):
    noisy = EvaluationResult(
        "failed", "line one\nline two\ttab é \x07bell", (check("c01", "failed", 1),), None,
        MinerReason.CHECK_FAILED, Stage.CHECK,
    )
    log = EvaluationLog(str(tmp_path / "log.jsonl"))
    assert log.record_round(completed(evaluations=[evaluation(1, noisy)], checks_total=1), VALIDATOR)
    raw = (tmp_path / "log.jsonl").read_bytes()
    assert raw.isascii() and raw.endswith(b"\n")
    assert all(32 <= byte < 127 for byte in raw.replace(b"\n", b""))
    items = parsed(tmp_path / "log.jsonl")
    assert len(items) == 2 and items[1]["reason"] == "line one line two tab é bell"
    assert os.stat(tmp_path / "log.jsonl").st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------- #
# Sink configuration and bounds
# --------------------------------------------------------------------------- #
def test_empty_path_disables_the_sink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = EvaluationLog("")
    assert log.path is None
    assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), VALIDATOR) is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kwargs", [
    {"max_file_bytes": diagnostics.MAX_RECORD_BYTES - 1},
    {"max_file_bytes": 64 * 1024 * 1024 + 1},
    {"backup_count": -1},
    {"backup_count": 9},
])
def test_constructor_rejects_out_of_range_bounds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        EvaluationLog(str(tmp_path / "log.jsonl"), **kwargs)
    EvaluationLog(str(tmp_path / "log.jsonl"), max_file_bytes=diagnostics.MAX_RECORD_BYTES, backup_count=0)
    EvaluationLog(str(tmp_path / "log.jsonl"), max_file_bytes=64 * 1024 * 1024, backup_count=8)


def many_rounds(count, miners=4):
    return [
        completed(
            challenge=f"chal-{index:03}",
            evaluations=[evaluation(uid, PASSED if uid % 2 else STOPPED) for uid in range(1, miners + 1)],
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("backup_count", [0, 2])
def test_rotation_keeps_file_count_bytes_and_order_within_bounds(tmp_path, backup_count):
    path = tmp_path / "log.jsonl"
    cap = diagnostics.MAX_RECORD_BYTES
    log = EvaluationLog(str(path), max_file_bytes=cap, backup_count=backup_count)
    rounds = many_rounds(12)
    expected = [item["record_id"] for result in rounds for item in records(result)]
    for result in rounds:
        assert log.record_round(result, VALIDATOR) is True

    files = all_files(path)
    allowed = [path] + [tmp_path / f"log.jsonl.{index}" for index in range(1, backup_count + 1)]
    assert files and set(files) <= set(allowed)
    assert all(file.stat().st_size <= cap for file in files)
    assert sum(file.stat().st_size for file in files) <= cap * (backup_count + 1)
    oldest_first = [file for file in reversed(allowed) if file.exists()]
    kept = [item["record_id"] for file in oldest_first for item in parsed(file)]
    assert kept == expected[len(expected) - len(kept):]  # a contiguous, newest-last suffix
    assert kept[-1] == expected[-1] and len(kept) < len(expected)
    assert len(kept) > 1


def test_oversize_record_drops_reason_before_being_rejected(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    long_reason = EvaluationResult(
        "failed", "r" * 200, (check("c01", "failed", 1),), None, MinerReason.CHECK_FAILED, Stage.CHECK,
    )
    result = completed(evaluations=[evaluation(1, long_reason)], checks_total=1)
    full = [json.dumps(item, ensure_ascii=True, separators=(",", ":")) for item in records(result)]
    round_size, miner_size = (len(item) + 1 for item in full)
    assert miner_size > round_size + 150

    log = EvaluationLog(str(path))
    monkeypatch.setattr(diagnostics, "MAX_RECORD_BYTES", miner_size - 1)
    assert log.record_round(result, VALIDATOR) is True
    items = parsed(path)
    assert items[0]["reason"] == "" and items[1]["reason"] == "" and items[1]["reason_code"] == "check_failed"
    assert len(lines(path)[1]) + 1 <= miner_size - 1

    monkeypatch.setattr(diagnostics, "MAX_RECORD_BYTES", round_size - 1)
    before = path.read_bytes()
    assert log.record_round(result, VALIDATOR) is False
    assert path.read_bytes() == before


def test_unbounded_validator_hotkey_cannot_produce_an_oversize_line(tmp_path):
    path = tmp_path / "log.jsonl"
    log = EvaluationLog(str(path))
    assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), "v" * 5_000) is False
    assert not path.exists() or path.read_bytes() == b""


def test_interrupted_trailing_write_is_truncated_before_the_next_append(tmp_path):
    path = tmp_path / "log.jsonl"
    log = EvaluationLog(str(path))
    first = completed(challenge="chal-a", evaluations=[evaluation(1, PASSED)])
    assert log.record_round(first, VALIDATOR)
    intact = path.read_bytes()
    with path.open("ab") as output:
        output.write(b'{"record_type":"miner_evaluation","uid":9')
    second = completed(challenge="chal-b", evaluations=[evaluation(1, PASSED)])
    assert log.record_round(second, VALIDATOR) is True
    assert path.read_bytes().startswith(intact)
    items = parsed(path)
    assert [item["challenge_id"] for item in items] == ["chal-a", "chal-a", "chal-b", "chal-b"]
    assert b'"uid":9' not in path.read_bytes()


def test_partial_only_file_is_reset_and_unrepairable_tail_is_harmless(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"broken":')
    log = EvaluationLog(str(path))
    assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), VALIDATOR) is True
    assert [item["record_type"] for item in parsed(path)] == ["round_outcome", "miner_evaluation"]

    garbage = tmp_path / "garbage.jsonl"
    garbage.write_bytes(b"x" * (diagnostics.MAX_RECORD_BYTES + 1))
    log = EvaluationLog(str(garbage))
    assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), VALIDATOR) is False
    assert garbage.read_bytes() == b"x" * (diagnostics.MAX_RECORD_BYTES + 1)


# --------------------------------------------------------------------------- #
# Harmless failures
# --------------------------------------------------------------------------- #
def _directory(path):
    path.parent.mkdir()
    path.mkdir()


def _parent_is_file(path):
    path.parent.write_bytes(b"file")


def _fifo(path):
    path.parent.mkdir()
    os.mkfifo(path)


def _read_only_file(path):
    path.parent.mkdir()
    path.write_bytes(b"")
    path.chmod(0o400)


def _read_only_parent(path):
    path.parent.mkdir()
    path.parent.chmod(0o500)


NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")


@pytest.mark.parametrize("prepare", [
    pytest.param(_directory, id="directory"),
    pytest.param(_parent_is_file, id="parent-is-file"),
    pytest.param(_fifo, id="fifo"),
    pytest.param(_read_only_file, id="read-only-file", marks=NOT_ROOT),
    pytest.param(_read_only_parent, id="read-only-parent", marks=NOT_ROOT),
])
def test_invalid_targets_return_false_without_raising(tmp_path, prepare):
    path = tmp_path / "state" / "log.jsonl"
    prepare(path)
    log = EvaluationLog(str(path))
    try:
        assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), VALIDATOR) is False
    finally:
        if path.parent.is_dir():
            path.parent.chmod(0o700)
    if path.is_file():
        assert path.read_bytes() == b""
    if path.parent.is_file():
        assert path.parent.read_bytes() == b"file"


def test_symlinked_target_is_never_followed(tmp_path):
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(b"precious\n")
    link = tmp_path / "log.jsonl"
    link.symlink_to(victim)
    log = EvaluationLog(str(link))
    assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), VALIDATOR) is False
    assert victim.read_bytes() == b"precious\n"
    assert link.is_symlink()


def test_rotation_failure_returns_false_and_leaves_the_active_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    cap = diagnostics.MAX_RECORD_BYTES
    log = EvaluationLog(str(path), max_file_bytes=cap, backup_count=2)
    rounds = many_rounds(6, miners=2)
    assert log.record_round(rounds[0], VALIDATOR) is True
    before = path.read_bytes()

    def broken_replace(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(diagnostics.os, "replace", broken_replace)
    outcomes = [log.record_round(result, VALIDATOR) for result in rounds[1:]]
    assert False in outcomes
    assert all(outcome is True for outcome in outcomes[: outcomes.index(False)])
    assert path.read_bytes().startswith(before)
    for file in all_files(path):
        assert file.stat().st_size <= cap
        parsed(file)  # every surviving line is still complete JSON


def test_serialization_failure_returns_false_and_writes_nothing(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    log = EvaluationLog(str(path))
    result = completed(evaluations=[evaluation(1, PASSED)])
    good = records(result)

    def poisoned(*_args, **_kwargs):
        yield {**good[0], "reason": {"nested", "set"}}
        yield good[1]

    monkeypatch.setattr(diagnostics, "round_records", poisoned)
    assert log.record_round(result, VALIDATOR) is False
    assert not path.exists() or path.read_bytes() == b""


def test_malformed_result_values_return_false_instead_of_raising(tmp_path):
    path = tmp_path / "log.jsonl"
    log = EvaluationLog(str(path))
    plain_string_code = RoundResult(
        "abandoned", "x", (), reason_code="not-an-enum", challenge_id="chal-1", task_id=TASK,
    )
    assert log.record_round(plain_string_code, VALIDATOR) is False
    assert log.record_round(completed(evaluations=[evaluation(1, PASSED)]), VALIDATOR) is True
    assert len(parsed(path)) == 2


# --------------------------------------------------------------------------- #
# Round-level codes reachable without network or grading
# --------------------------------------------------------------------------- #
class LeaseOnlyClient:
    def __init__(self, outcome):
        self.outcome = outcome

    async def lease(self):
        return self.outcome

    async def commit(self, request):  # pragma: no cover - must never be reached here
        raise AssertionError("commit must not be reached")

    async def feedback(self, request):  # pragma: no cover
        raise AssertionError("feedback must not be reached")


def run_round(tmp_path, client, round_policy, solvers=()):
    return asyncio.run(evaluate_round(
        client, None, list(solvers), round_policy,
        cache_dir=tmp_path / "cache", work_dir=tmp_path / "work",
    ))


def test_transport_lease_failure_is_unavailable_and_unrecorded(tmp_path):
    outcome = V3LeaseOutcome(
        LeaseCategory.TRANSPORT, detail="https://problems.invalid unreachable", retry_after_s=7,
        transport_error="https://problems.invalid unreachable",
    )
    result = run_round(tmp_path, LeaseOnlyClient(outcome), policy(tmp_path))
    assert (result.status, result.reason_code, result.stage, result.retry_after_s) == (
        "unavailable", RoundReason.LEASE_UNAVAILABLE, Stage.LEASE, 7)
    assert result.challenge_id is None and result.evaluations == ()
    log = EvaluationLog(str(tmp_path / "log.jsonl"))
    assert log.record_round(result, VALIDATOR) is True
    assert not (tmp_path / "log.jsonl").exists()


@pytest.mark.parametrize("field, code", [
    ("execution_profile_id", RoundReason.UNSUPPORTED_PROFILE),
    ("verifier_policy", RoundReason.UNSUPPORTED_VERIFIER_POLICY),
])
def test_policy_mismatch_abandons_at_the_lease_stage_with_identity(tmp_path, field, code):
    leased = V3LeaseOutcome(LeaseCategory.LEASED, challenge=LeaseResponse(**lease()))
    mismatched = dataclasses.replace(policy(tmp_path), **{field: "something-else"})
    result = run_round(tmp_path, LeaseOnlyClient(leased), mismatched)
    assert (result.status, result.reason_code) == ("abandoned", code)
    assert result.evaluations == () and result.diagnostic_evaluations == ()
    assert (result.challenge_id, result.task_id) == ("chal-1", leased.challenge.task_id)
    assert result.assigned_miners == ((7, "hk-7"),)
    assert result.checks_total is None
    # Policy validation precedes workspace preparation and downloads.
    assert result.stage == Stage.LEASE


def test_cleanup_stage_fault_maps_to_cleanup_failed_and_leaks_no_message(tmp_path, monkeypatch):
    stale = tmp_path / "work" / "hone-v3-round-stale"
    stale.mkdir(parents=True)
    secret = "https://uploads.invalid/read?token=TOPSECRET"

    def broken(_path):
        raise RuntimeError(secret)

    monkeypatch.setattr("rlvr.v3.round._remove_tree", broken)
    leased = V3LeaseOutcome(LeaseCategory.LEASED, challenge=LeaseResponse(**lease()))
    result = run_round(tmp_path, LeaseOnlyClient(leased), policy(tmp_path))
    assert (result.status, result.reason_code, result.stage) == (
        "abandoned", RoundReason.CLEANUP_FAILED, Stage.CLEANUP)
    assert result.reason == "validator failed during cleanup (RuntimeError)"
    assert result.evaluations == () and result.dispatch_failures == ()

    path = tmp_path / "log.jsonl"
    assert EvaluationLog(str(path)).record_round(result, VALIDATOR) is True
    raw = path.read_bytes()
    assert b"https://" not in raw and b"TOPSECRET" not in raw and str(tmp_path).encode() not in raw
    items = parsed(path)
    assert (items[0]["round_reason_code"], items[0]["stage"], items[0]["score_effect"]) == (
        "cleanup_failed", "cleanup", "unchanged")
    miner = by_uid(items)[7]
    assert (miner["status"], miner["reason_code"], miner["stage"], miner["checks_total"],
            miner["checks_skipped"]) == ("not_evaluated", None, None, None, None)
