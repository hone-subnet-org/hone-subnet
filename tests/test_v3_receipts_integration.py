"""Receipts annotate a failed check without changing execution or scoring."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rlvr.v3 import grading
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.supervisor import SupervisorError
from tests.test_v3_grading import RUN, eval_repo, eval_term, result
from tests.test_v3_grading import (
    runner as runner,  # noqa: PLC0414 - expose the shared pytest fixture
)
from tests.test_v3_manifest import expect, inspection, invocation, repo_manifest
from tests.test_v3_receipts import decoded, make_receipt


def legacy_fields(verdict):
    return tuple(
        getattr(verdict, name)
        for name in (
            "status",
            "reason",
            "checks",
            "script_exit_code",
            "reason_code",
            "stage",
        )
    )


def test_only_first_failure_gets_a_receipt_in_original_check_order(tmp_path, runner):
    doc = repo_manifest(
        setup=None,
        checks=[
            invocation("first_pass"),
            inspection("z_failure"),
            inspection("a_unrun"),
        ],
    )
    runner.responses = [result(), result(stdout=b"wrong\xff\n")]
    verdict = eval_repo(tmp_path, runner, doc)
    assert (verdict.status, verdict.reason_code, verdict.stage) == (
        "failed",
        MinerReason.CHECK_FAILED,
        Stage.CHECK,
    )
    assert [(item.check_id, item.outcome) for item in verdict.checks] == [
        ("first_pass", "passed"),
        ("z_failure", "failed"),
        ("a_unrun", "skipped"),
    ]
    assert [request.name for request in runner.requests] == [
        f"{RUN}-first_pass",
        f"{RUN}-z_failure",
    ]
    receipt = verdict.receipt.to_record()
    assert (receipt["check_index"], receipt["checks_total"], receipt["check_id"]) == (
        2,
        3,
        "z_failure",
    )
    assert receipt["output_source"] == "checker" and receipt["stdin"] is None
    assert decoded(receipt["stdout"]["actual"]) == b"wrong\xff\n"
    assert "a_unrun" not in str(receipt)


def test_invocation_receipt_uses_resolved_cwd_and_existing_stdin(tmp_path, runner):
    doc = repo_manifest(
        setup=None, checks=[invocation(expect=expect(stderr="gold/err")), inspection()]
    )
    runner.responses = [result(stdout=b"bad", stderr=b"baderr", exit_code=1)]
    verdict = eval_repo(tmp_path, runner, doc, working_directory="sub")
    assert len(runner.requests) == 1
    receipt = verdict.receipt.to_record()
    assert receipt["cwd"] == "/work/sub"
    assert receipt["argv"] == list(runner.requests[0].argv)
    assert decoded(receipt["stdin"]) == runner.requests[0].stdin == b"x\n"
    assert decoded(receipt["stdout"]["expected"]) == b"x\n"
    assert decoded(receipt["stderr"]["expected"]) == b"x\n"
    assert receipt["mismatched"] == ["exit_code", "stdout", "stderr"]


@pytest.mark.parametrize("kind", ["invocation", "inspection"])
@pytest.mark.parametrize(
    "flags,code",
    [
        ({"timed_out": True}, MinerReason.TIMEOUT),
        ({"oom_killed": True}, MinerReason.MEMORY_LIMIT),
        ({"stdout_overflow": True}, MinerReason.OUTPUT_LIMIT),
        ({"stderr_overflow": True}, MinerReason.OUTPUT_LIMIT),
        (
            {"timed_out": True, "oom_killed": True, "stdout_overflow": True},
            MinerReason.TIMEOUT,
        ),
        ({"oom_killed": True, "stderr_overflow": True}, MinerReason.MEMORY_LIMIT),
    ],
)
def test_limit_receipt_preserves_fault_priority_and_does_not_read_gold(
    tmp_path, runner, monkeypatch, kind, flags, code
):
    check = invocation() if kind == "invocation" else inspection("c01")
    doc = repo_manifest(setup=None, checks=[check, inspection("c02")])
    runner.responses = [
        result(exit_code=127, stdout=b"partial", stderr=b"observed", **flags)
    ]
    original_read = Path.read_bytes
    gold_reads = []

    def guarded_read(path):
        if path.parent.name == "gold":
            gold_reads.append(path)
            raise OSError("gold must not be read on a limit path")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    verdict = eval_repo(tmp_path, runner, doc)
    assert (verdict.status, verdict.reason_code) == ("failed", code)
    assert [item.outcome for item in verdict.checks] == ["failed", "skipped"]
    assert len(runner.requests) == 1 and gold_reads == []
    receipt = verdict.receipt.to_record()
    assert receipt["limit"] == code.value and receipt["mismatched"] == []
    assert (
        receipt["stdout"]["expected"] is None and receipt["stderr"]["expected"] is None
    )
    assert receipt["output_source"] == (
        "checker" if kind == "inspection" else "candidate"
    )
    assert receipt["stdout"]["actual"]["capture_truncated"] is flags.get(
        "stdout_overflow", False
    )


@pytest.mark.parametrize("failure", [ValueError, OSError, RuntimeError])
@pytest.mark.parametrize("limit", [False, True])
def test_builder_exceptions_leave_all_grading_fields_and_execution_unchanged(
    tmp_path, runner, monkeypatch, failure, limit
):
    doc = repo_manifest(setup=None)
    response = result(stdout=b"wrong", **({"timed_out": True} if limit else {}))
    baseline_dir, failed_dir = tmp_path / "baseline", tmp_path / "builder_failure"
    baseline_dir.mkdir()
    failed_dir.mkdir()
    runner.responses = [response]
    baseline = eval_repo(baseline_dir, runner, doc)
    baseline_calls = len(runner.requests)
    calls = []

    def broken_builder(**kwargs):
        calls.append(kwargs)
        raise failure("receipt construction failed")

    monkeypatch.setattr(grading, "build_check_receipt", broken_builder, raising=False)
    runner.responses = [response]
    runner.requests.clear()
    verdict = eval_repo(failed_dir, runner, doc)
    assert len(calls) == 1, (
        "the builder failure must actually exercise the receipt exception boundary"
    )
    assert legacy_fields(verdict) == legacy_fields(baseline)
    assert verdict.receipt is None
    assert len(runner.requests) == baseline_calls == 1


@pytest.mark.parametrize(
    "response",
    [
        result(exit_code=126),
        result(exit_code=127),
        SupervisorError("container unavailable"),
        OSError("container unavailable"),
    ],
)
def test_infrastructure_failure_has_no_receipt(tmp_path, runner, monkeypatch, response):
    calls = []
    monkeypatch.setattr(
        grading,
        "build_check_receipt",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    runner.responses = [response]
    verdict = eval_repo(
        tmp_path, runner, repo_manifest(setup=None, checks=[inspection()])
    )
    assert (verdict.status, verdict.reason_code) == (
        "abandoned",
        RoundReason.VERIFIER_UNAVAILABLE,
    )
    assert verdict.receipt is None and calls == []
    assert len(runner.requests) == 1


@pytest.mark.parametrize(
    "case,code",
    [
        ("setup", MinerReason.SETUP_FAILED),
        ("script_limit", MinerReason.TIMEOUT),
        ("patch", MinerReason.PATCH_STATIC_REJECTED),
        ("missing_cwd", MinerReason.WORKING_DIRECTORY_INVALID),
        ("tree", MinerReason.RESULT_TREE_INVALID),
    ],
)
def test_out_of_scope_failures_keep_reasons_without_receipts(
    tmp_path, runner, monkeypatch, case, code
):
    calls = []
    monkeypatch.setattr(
        grading,
        "build_check_receipt",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    if case == "setup":
        runner.responses = [result(exit_code=1)]
        verdict = eval_repo(tmp_path, runner)
    elif case == "script_limit":
        runner.responses = [result(timed_out=True)]
        verdict = eval_term(tmp_path, runner)
    elif case == "patch":
        verdict = eval_repo(tmp_path, runner, patch=b"\x00")
    elif case == "missing_cwd":
        verdict = eval_repo(
            tmp_path,
            runner,
            repo_manifest(setup=None, checks=[invocation(cwd="absent")]),
        )
    else:

        def unsafe_tree(_request):
            (tmp_path / "ws" / "link").symlink_to("/etc/passwd")
            return result()

        runner.responses = [unsafe_tree]
        verdict = eval_repo(tmp_path, runner, repo_manifest(setup=None))
    assert verdict.reason_code == code
    assert verdict.receipt is None and calls == []


def test_passing_round_does_not_build_receipts(tmp_path, runner, monkeypatch):
    calls = []
    monkeypatch.setattr(
        grading,
        "build_check_receipt",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    verdict = eval_repo(tmp_path, runner, repo_manifest(setup=None))
    assert verdict.status == "passed" and verdict.receipt is None
    assert calls == [] and len(runner.requests) == 2


def test_terminal_inspection_mismatch_gets_checker_receipt(tmp_path, runner):
    runner.responses = [result(exit_code=9), result(stdout=b"bad")]
    verdict = eval_term(tmp_path, runner)
    assert verdict.status == "failed" and verdict.script_exit_code == 9
    assert len(runner.requests) == 2
    assert verdict.receipt.output_source == "checker"
    assert verdict.receipt.check_index == 1 and verdict.receipt.checks_total == 2


def test_result_compatibility_and_receipt_status_guard():
    old = EvaluationResult(
        "failed", "failed", (), None, MinerReason.CHECK_FAILED, Stage.CHECK
    )
    assert old.receipt is None
    receipt = make_receipt()
    assert replace(old, receipt=receipt).receipt is receipt
    for status in ("passed", "rejected", "abandoned"):
        with pytest.raises(ValueError):
            EvaluationResult(
                status,
                "" if status == "passed" else "failed",
                (),
                None,
                receipt=receipt,
            )
    with pytest.raises(ValueError):
        replace(old, receipt=object())
