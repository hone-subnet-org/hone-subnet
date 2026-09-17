"""A failed check display states the comparison without revealing its result."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rlvr.v3 import feedback, grading
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.manifest import Expectation, InspectionCheck, InvocationCheck
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.supervisor import ContainerRequest, SupervisorError
from tests.test_v3_grading import eval_repo, result
from tests.test_v3_grading import runner as runner  # noqa: PLC0414
from tests.test_v3_manifest import expect, invocation, repo_manifest


def display_inputs(
    *, argv=("/usr/bin/python3", "main.py"), cwd="/work", stdin=b"", stderr=None
):
    check = InvocationCheck(
        check_id="private_case_name",
        argv=argv,
        cwd=".",
        stdin="inputs/private.stdin" if stdin else None,
        timeout_s=10,
        max_stdout_bytes=65536,
        max_stderr_bytes=65536,
        expect=Expectation(
            0,
            "gold/private.stdout",
            "gold/private.stderr" if stderr is not None else None,
        ),
    )
    request = ContainerRequest(
        name="private-container",
        argv=argv,
        cwd=cwd,
        mounts=(),
        stdin=stdin,
        timeout_s=10,
        max_stdout_bytes=65536,
        max_stderr_bytes=65536,
        trusted=False,
    )
    return {
        "check": check,
        "request": request,
        "expected_stdout": b"red-fox\n",
        "expected_stderr": stderr,
    }


def render(**kwargs):
    return feedback.render_failed_check(**display_inputs(**kwargs))


def test_complete_slug_check_is_displayed_without_private_metadata():
    shown = render()
    assert shown == (
        'Command (argv): ["/usr/bin/python3","main.py"]\n'
        'Working directory: "/work"\n'
        'Stdin: ""\n'
        "Required exit code: 0\n"
        'Required stdout: "red-fox\\n"\n'
        "Required stderr: not checked"
    )
    assert "private" not in shown
    assert "slug(" not in shown


@pytest.mark.parametrize("python", ["/usr/bin/python3", "/usr/local/bin/python3"])
@pytest.mark.parametrize(
    "script", ["main.py", "./main.py", "pkg/run-test.py", "/work/main.py"]
)
def test_exact_supported_command_forms(python, script):
    shown = render(argv=(python, script), cwd="/work/project")
    assert shown is not None
    assert json.loads(shown.splitlines()[0].split(": ", 1)[1]) == [python, script]
    assert 'Working directory: "/work/project"' in shown


@pytest.mark.parametrize(
    "argv",
    [
        ("/usr/bin/python3",),
        ("/usr/bin/python3", "-c", "print('private')"),
        ("/usr/bin/python3", "-m", "pytest"),
        ("/usr/bin/python3", "-I", "main.py"),
        ("/usr/bin/python3", "main.py", "secret"),
        ("/usr/bin/python3", "-hidden.py"),
        ("/usr/bin/python3", "../hidden.py"),
        ("/usr/bin/python3", "pkg/../hidden.py"),
        ("/usr/bin/python3", "/work/../verify/hidden.py"),
        ("/usr/bin/python3", "/workspace/main.py"),
        ("/usr/bin/python3", "/verify/main.py"),
        ("/usr/bin/python3", "a\nsecret.py"),
        ("/usr/bin/python3", "a\tsecret.py"),
        ("/usr/bin/python3", "a\u202esecret.py"),
        ("/usr/bin/python3", "main.py;hidden.py"),
        ("/usr/bin/python3", "main.py "),
        ("/usr/bin/python3", "main.sh"),
        ("/usr/bin/python", "main.py"),
        ("/usr/bin/bash", "main.py"),
        ("/work/python3", "main.py"),
    ],
)
def test_opaque_commands_and_paths_are_omitted(argv):
    assert render(argv=argv) is None


@pytest.mark.parametrize(
    "cwd",
    [
        "/",
        "/workspace",
        "/work/../work",
        "/work//sub",
        "/work/sub/",
        "/work/a\n",
        "/work/a b",
    ],
)
def test_unsupported_working_directory_is_omitted(cwd):
    args = display_inputs()
    object.__setattr__(args["request"], "cwd", cwd)
    assert feedback.render_failed_check(**args) is None


def test_inspection_and_mismatched_execution_context_are_omitted():
    args = display_inputs()
    base = args["check"]
    inspected = InspectionCheck(
        base.check_id,
        base.argv,
        base.timeout_s,
        base.max_stdout_bytes,
        base.max_stderr_bytes,
        base.expect,
    )
    assert feedback.render_failed_check(**{**args, "check": inspected}) is None
    for request in (
        replace(args["request"], trusted=True),
        replace(args["request"], argv=("/usr/bin/python3", "other.py")),
        replace(args["request"], stdin=b"unaccounted input"),
    ):
        assert feedback.render_failed_check(**{**args, "request": request}) is None
    assert feedback.render_failed_check(**{**args, "expected_stderr": b""}) is None
    args = display_inputs(stderr=b"expected error")
    assert feedback.render_failed_check(**{**args, "expected_stderr": None}) is None


def test_checked_empty_stderr_is_different_from_unchecked():
    assert render(stderr=b"").endswith('Required stderr: ""')
    assert render().endswith("Required stderr: not checked")
    args = display_inputs(stderr=b"bad\n")
    args["check"] = replace(
        args["check"], expect=replace(args["check"].expect, exit_code=3)
    )
    shown = feedback.render_failed_check(**args)
    assert "Required exit code: 3" in shown
    assert shown.endswith('Required stderr: "bad\\n"')


def test_complete_bytes_survive_unicode_and_control_escaping():
    stdin = 'e\u0301\n"\\\x00'.encode()
    args = display_inputs(stdin=stdin, stderr=b"\t\r\n")
    args["expected_stdout"] = "\U0001f98a\n".encode()
    shown = feedback.render_failed_check(**args)
    assert shown is not None and shown.isascii() and len(shown.splitlines()) == 6
    fields = dict(line.split(": ", 1) for line in shown.splitlines())
    assert json.loads(fields["Stdin"]).encode() == stdin
    assert json.loads(fields["Required stdout"]).encode() == args["expected_stdout"]
    assert json.loads(fields["Required stderr"]).encode() == args["expected_stderr"]


@pytest.mark.parametrize("field", ["stdin", "expected_stdout", "expected_stderr"])
@pytest.mark.parametrize("value", [b"\xff", b"x" * 10000, b"\x00" * 600])
def test_binary_or_oversized_definition_is_omitted_whole(field, value):
    args = display_inputs(stdin=b"seed", stderr=b"")
    if field == "stdin":
        args["request"] = replace(args["request"], stdin=value)
    else:
        args[field] = value
    assert feedback.render_failed_check(**args) is None


def test_exact_escaped_size_boundary_never_returns_a_truncated_display():
    args = display_inputs()
    longest = None
    for size in range(1700, 2050):
        args["expected_stdout"] = b"x" * size
        shown = feedback.render_failed_check(**args)
        if shown is None:
            assert longest is not None
            assert (
                len(json.dumps(longest, ensure_ascii=True).encode("ascii"))
                == feedback.FAILED_CHECK_MAX_BYTES
            )
            break
        assert json.loads(shown.splitlines()[4].split(": ", 1)[1]) == "x" * size
        longest = shown
    else:
        pytest.fail("oversized display was not omitted")


def test_renderer_uses_existing_values_without_reading_files(monkeypatch):
    def forbidden(_path):
        raise AssertionError("renderer must not read files")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert render() is not None


def small_check(check_id="c01", **overrides):
    return invocation(check_id, argv=["/usr/bin/python3", "main.py"], **overrides)


def test_first_failed_comparison_only_no_actual_output_or_additional_runs(
    tmp_path, runner
):
    doc = repo_manifest(
        setup=None,
        checks=[small_check("passed"), small_check("failed"), small_check("unrun")],
    )
    runner.responses = [
        result(),
        result(stdout=b"SECRET_ACTUAL", stderr=b"SECRET_TRACEBACK", exit_code=1),
    ]
    verdict = eval_repo(tmp_path, runner, doc)
    assert (
        verdict.status == "failed" and verdict.reason_code == MinerReason.CHECK_FAILED
    )
    assert [x.outcome for x in verdict.checks] == ["passed", "failed", "skipped"]
    assert len(runner.requests) == 2
    assert verdict.failed_check is not None
    assert "SECRET" not in verdict.failed_check and "unrun" not in verdict.failed_check
    assert 'Required stdout: "x\\n"' in verdict.failed_check
    assert verdict.receipt is not None


def test_failure_display_uses_resolved_cwd_and_checked_stderr(tmp_path, runner):
    doc = repo_manifest(
        setup=None, checks=[small_check(expect=expect(stderr="gold/error"))]
    )
    runner.responses = [result(stdout=b"wrong")]
    verdict = eval_repo(tmp_path, runner, doc, working_directory="sub")
    assert 'Working directory: "/work/sub"' in verdict.failed_check
    assert 'Stdin: "x\\n"' in verdict.failed_check
    assert 'Required stderr: "x\\n"' in verdict.failed_check


@pytest.mark.parametrize(
    "response",
    [
        result(),
        result(timed_out=True),
        result(oom_killed=True),
        result(stdout_overflow=True),
        SupervisorError("offline"),
    ],
)
def test_noncomparison_paths_do_not_build_display_or_change_execution(
    tmp_path, runner, monkeypatch, response
):
    calls = []

    def forbidden(**_kwargs):
        calls.append(_kwargs)
        raise AssertionError("not a failed comparison")

    monkeypatch.setattr(grading, "render_failed_check", forbidden)
    doc = repo_manifest(setup=None, checks=[small_check()])
    runner.responses = [response]
    verdict = eval_repo(tmp_path, runner, doc)
    assert verdict.failed_check is None
    assert calls == []
    assert len(runner.requests) == 1
    if isinstance(response, SupervisorError):
        assert verdict.status == "abandoned"
    elif response.timed_out or response.oom_killed or response.stdout_overflow:
        assert verdict.status == "failed"
    else:
        assert verdict.status == "passed"


@pytest.mark.parametrize(
    "broken", [RuntimeError("display bug"), None, 5, {}, "", "x" * 10000]
)
def test_display_failures_never_change_grading_fields(
    tmp_path, runner, monkeypatch, broken
):
    doc = repo_manifest(setup=None, checks=[small_check(), small_check("unrun")])
    response = result(stdout=b"wrong")
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    runner.responses = [response]
    baseline = eval_repo(baseline_dir, runner, doc)

    def faulty(**_kwargs):
        if isinstance(broken, Exception):
            raise broken
        return broken

    monkeypatch.setattr(grading, "render_failed_check", faulty)
    failed_dir = tmp_path / "faulty"
    failed_dir.mkdir()
    runner.responses = [response]
    actual = eval_repo(failed_dir, runner, doc)
    assert replace(baseline, failed_check=None) == actual
    assert len(runner.requests) == 2


def test_evaluation_rejects_display_on_other_outcomes_and_oversized_text():
    for status, code, stage in [
        ("passed", None, Stage.CHECK),
        ("rejected", MinerReason.PATCH_REJECTED, Stage.PATCH),
        ("abandoned", RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK),
        ("failed", MinerReason.TIMEOUT, Stage.CHECK),
        ("failed", MinerReason.CHECK_FAILED, Stage.SETUP),
    ]:
        with pytest.raises(ValueError):
            EvaluationResult(
                status,
                "" if status == "passed" else "failed",
                (),
                None,
                code,
                stage,
                failed_check="check",
            )
    for text in ["", 1, "x" * 10000]:
        with pytest.raises(ValueError):
            EvaluationResult(
                "failed",
                "failed",
                (),
                None,
                MinerReason.CHECK_FAILED,
                Stage.CHECK,
                failed_check=text,
            )


def test_complete_size_includes_command_and_working_directory():
    assert render(argv=("/usr/bin/python3", "a" * 2000 + ".py")) is None
    assert render(cwd="/work/" + "a" * 2000) is None


def test_terminal_invocation_uses_the_existing_single_script_replay(tmp_path, runner):
    from tests.test_v3_grading import eval_term
    from tests.test_v3_manifest import term_manifest

    doc = term_manifest(checks=[small_check(), small_check("unrun")])
    runner.responses = [result(), result(stdout=b"ACTUAL_NOT_FOR_MINER")]
    verdict = eval_term(tmp_path, runner, doc)
    assert verdict.status == "failed"
    assert verdict.failed_check is not None
    assert "ACTUAL_NOT_FOR_MINER" not in verdict.failed_check
    assert len(runner.requests) == 2
    assert runner.requests[0].name.endswith("-script")
    assert runner.requests[1].name.endswith("-c01")
    assert [check.outcome for check in verdict.checks] == ["failed", "skipped"]


def test_display_validation_failure_cannot_abandon_grading(
    tmp_path, runner, monkeypatch
):
    doc = repo_manifest(setup=None, checks=[small_check()])
    runner.responses = [result(stdout=b"wrong")]

    def broken_validation(_value):
        raise ValueError("display validation failed")

    monkeypatch.setattr(grading, "is_bounded_display", broken_validation)
    verdict = eval_repo(tmp_path, runner, doc)
    assert verdict.status == "failed"
    assert verdict.reason_code == MinerReason.CHECK_FAILED
    assert verdict.failed_check is None
    assert len(runner.requests) == 1


def test_rendered_grading_failure_reaches_wire_without_candidate_output(
    tmp_path, runner
):
    from rlvr.v3.api import (
        FailureExplanation,
        MinerFailureNotice,
        serialize_failure_notice,
    )

    doc = repo_manifest(
        setup=None, checks=[small_check(expect=expect(stderr="gold/error"))]
    )
    runner.responses = [
        result(stdout=b"CAPTURED_STDOUT", stderr=b"CAPTURED_STDERR", exit_code=9)
    ]
    verdict = eval_repo(tmp_path, runner, doc)
    assert verdict.failed_check is not None
    encoded = serialize_failure_notice(MinerFailureNotice(
        protocol_version=3,
        message_type="failure_notice_v1",
        challenge_id="challenge",
        task_id="a" * 64,
        uid=7,
        hotkey="hk-7",
        failure=FailureExplanation(
            version=1, reason_code=verdict.reason_code, failed_check=verdict.failed_check,
        ),
    ))
    item = json.loads(encoded)
    assert item["failure"] == {
        "version": 1,
        "reason_code": "check_failed",
        "failed_check": verdict.failed_check,
    }
    assert len(runner.requests) == 1
    assert b"CAPTURED_STDOUT" not in encoded and b"CAPTURED_STDERR" not in encoded
    assert verdict.reason.encode() not in encoded
    assert "receipt" not in item["failure"]
