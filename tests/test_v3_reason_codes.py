"""Classification parity for ``rlvr.v3.reasons`` across every grading detection
site in ``rlvr.v3.grading`` and ``rlvr.v3.patch``.

Every verdict carries ``(status, reason_code, stage)``.  Status values are the
pre-existing classifications and must not move.  ``abandoned`` verdicts carry a
``RoundReason`` (validator fault), ``failed``/``rejected`` carry a ``MinerReason``,
``passed`` carries no code.  Enum values are stable lowercase strings that never
overlap, so a bare string identifies one code without knowing its enum.
"""

from __future__ import annotations

import json
import shutil
from typing import get_args

import pytest

from rlvr.v3 import patch as patch_module
from rlvr.v3.artifacts import ArtifactFailure
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.patch import (
    PatchLimits,
    PatchResult,
    PatchToolError,
    apply_patch_in_container,
)
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.supervisor import SupervisorError
from tests.test_v3_grading import (
    RUN,
    FakeRunner,
    _mod,
    eval_repo,
    eval_term,
    patch_limits,
    repo_identity,
    repo_setup,
    result,
    script_limits,
    term_identity,
    term_setup,
    tree_limits,
)
from tests.test_v3_grading import (
    runner as runner,  # noqa: PLC0414 - expose the shared pytest fixture
)
from tests.test_v3_manifest import (
    SETUP,
    expect,
    inspection,
    invocation,
    repo_manifest,
    term_manifest,
)
from tests.test_v3_patch import GIT, MODIFY, fake_git, make_workspace
from tests.test_v3_supervisor import policy

needs_git = pytest.mark.skipif(GIT is None, reason="git not installed")

DELETE_SUB = (
    b"diff --git a/sub/g.txt b/sub/g.txt\ndeleted file mode 100644\n"
    b"--- a/sub/g.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
)
ESCAPE = (
    b"diff --git a/../escape.txt b/../escape.txt\nnew file mode 100644\n"
    b"--- /dev/null\n+++ b/../escape.txt\n@@ -0,0 +1 @@\n+pwned\n"
)


def triple(verdict):
    return verdict.status, verdict.reason_code, verdict.stage


def assert_classified(verdict, status, code, stage):
    assert triple(verdict) == (status, code, stage)
    # Structural invariants that round.py relies on when it maps grading
    # verdicts onto round outcomes.
    assert (status == "abandoned") == isinstance(code, RoundReason)
    assert (status == "passed") == (code is None)
    if status in ("failed", "rejected"):
        assert isinstance(code, MinerReason)
    assert isinstance(stage, Stage)


# --------------------------------------------------------------------------- #
# Enum contract
# --------------------------------------------------------------------------- #
def test_codes_are_stable_disjoint_lowercase_strings():
    miner = {item.value for item in MinerReason}
    round_ = {item.value for item in RoundReason}
    stages = {item.value for item in Stage}
    assert not miner & round_
    for value in miner | round_ | stages:
        assert isinstance(value, str) and value and value == value.lower()
        assert value.replace("_", "a").isalnum() and value.isascii()
    assert MinerReason("not_serving") is MinerReason.NOT_SERVING
    assert json.dumps(MinerReason.PATCH_APPLY_FAILED) == '"patch_apply_failed"'
    assert json.dumps(Stage.WORKSPACE_MATERIALIZATION) == '"workspace_materialization"'
    assert MinerReason.PATCH_STATIC_REJECTED != MinerReason.PATCH_APPLY_FAILED
    assert {"not_serving", "dispatch_failed", "response_unavailable"} <= miner
    assert {"patch_static_rejected", "patch_apply_failed", "check_failed"} <= miner
    assert {"verifier_unavailable", "patch_tool_failed", "cleanup_failed", "validator_error"} <= round_
    assert {
        "workspace_download_failed", "workspace_extraction_failed",
        "workspace_materialization_failed",
    } <= round_


def test_closed_server_artifact_reasons_map_onto_miner_codes_by_equality():
    literals = get_args(ArtifactFailure.model_fields["reason"].annotation)
    assert set(literals) == {"slot_mismatch", "artifact_invalid", "trajectory_invalid"}
    for literal in literals:
        assert MinerReason(literal).value == literal
    with pytest.raises(ValueError):
        MinerReason("miner is not serving")  # free text never becomes a code


def test_result_types_take_codes_positionally_and_default_to_none():
    verdict = EvaluationResult("failed", "x", (), None, MinerReason.CHECK_FAILED, Stage.CHECK)
    assert triple(verdict) == ("failed", MinerReason.CHECK_FAILED, Stage.CHECK)
    assert triple(EvaluationResult("passed", "", (), None)) == ("passed", None, None)
    assert PatchResult("rejected", "why").reason_code is None
    assert PatchResult("applied", "").reason_code is None
    assert PatchResult("rejected", "why", MinerReason.PATCH_APPLY_FAILED).reason_code is MinerReason.PATCH_APPLY_FAILED


# --------------------------------------------------------------------------- #
# Repository grading sites
# --------------------------------------------------------------------------- #
def plant_symlink(tmp_path):
    def act(request):
        (tmp_path / "ws" / "evil").symlink_to("/etc/passwd")
        return result()
    return act


def bloat(tmp_path):
    def act(request):
        (tmp_path / "ws" / "big").write_bytes(b"x" * (2 << 20))
        return result()
    return act


REPO_CASES = {
    # id: (responses, doc-builder, patch, expected status, code, stage, check outcomes)
    "passed": ([result(), result(), result()], None, MODIFY, "passed", None, Stage.CHECK, ["passed", "passed"]),
    "check-stdout-mismatch": ([result(), result(stdout=b"wrong\n")], None, MODIFY, "failed",
                              MinerReason.CHECK_FAILED, Stage.CHECK, ["failed", "skipped"]),
    "check-exit-mismatch": ([result(), result(exit_code=1)], None, MODIFY, "failed",
                            MinerReason.CHECK_FAILED, Stage.CHECK, ["failed", "skipped"]),
    "candidate-exit-127-is-miner-fault": ([result(), result(exit_code=127, stdout=b"", stderr=b"")], None, MODIFY,
                                          "failed", MinerReason.CHECK_FAILED, Stage.CHECK, ["failed", "skipped"]),
    "candidate-timeout": ([result(), result(timed_out=True, exit_code=124)], None, MODIFY, "failed",
                          MinerReason.TIMEOUT, Stage.CHECK, ["failed", "skipped"]),
    "candidate-oom": ([result(), result(oom_killed=True, exit_code=137)], None, MODIFY, "failed",
                      MinerReason.MEMORY_LIMIT, Stage.CHECK, ["failed", "skipped"]),
    "candidate-stdout-overflow": ([result(), result(stdout_overflow=True)], None, MODIFY, "failed",
                                  MinerReason.OUTPUT_LIMIT, Stage.CHECK, ["failed", "skipped"]),
    "candidate-stderr-overflow": ([result(), result(stderr_overflow=True)], None, MODIFY, "failed",
                                  MinerReason.OUTPUT_LIMIT, Stage.CHECK, ["failed", "skipped"]),
    "inspection-mismatch": ([result(), result(), result(stdout=b"bad\n")], None, MODIFY, "failed",
                            MinerReason.CHECK_FAILED, Stage.CHECK, ["passed", "failed"]),
    "inspection-126": ([result(), result(), result(exit_code=126, stdout=b"", stderr=b"")], None, MODIFY,
                       "abandoned", RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK, ["passed", "failed"]),
    "inspection-127": ([result(), result(), result(exit_code=127, stdout=b"", stderr=b"")], None, MODIFY,
                       "abandoned", RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK, ["passed", "failed"]),
    "inspection-timeout": ([result(), result(), result(timed_out=True, exit_code=124)], None, MODIFY, "failed",
                           MinerReason.TIMEOUT, Stage.CHECK, ["passed", "failed"]),
    "inspection-oom": ([result(), result(), result(oom_killed=True, exit_code=137)], None, MODIFY, "failed",
                       MinerReason.MEMORY_LIMIT, Stage.CHECK, ["passed", "failed"]),
    "inspection-overflow": ([result(), result(), result(stderr_overflow=True)], None, MODIFY, "failed",
                            MinerReason.OUTPUT_LIMIT, Stage.CHECK, ["passed", "failed"]),
    "supervisor-error-mid-check": ([result(), SupervisorError("boom")], None, MODIFY, "abandoned",
                                   RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK, ["skipped", "skipped"]),
    "oserror-mid-check": ([result(), OSError("disk")], None, MODIFY, "abandoned",
                          RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK, ["skipped", "skipped"]),
    "setup-nonzero": ([result(exit_code=2)], None, MODIFY, "failed",
                      MinerReason.SETUP_FAILED, Stage.SETUP, ["skipped", "skipped"]),
    "setup-timeout": ([result(timed_out=True, exit_code=124)], None, MODIFY, "failed",
                      MinerReason.TIMEOUT, Stage.SETUP, ["skipped", "skipped"]),
    "setup-oom": ([result(oom_killed=True, exit_code=137)], None, MODIFY, "failed",
                  MinerReason.MEMORY_LIMIT, Stage.SETUP, ["skipped", "skipped"]),
    "setup-overflow": ([result(stdout_overflow=True)], None, MODIFY, "failed",
                       MinerReason.OUTPUT_LIMIT, Stage.SETUP, ["skipped", "skipped"]),
    "setup-supervisor-error": ([SupervisorError("offline")], None, MODIFY, "abandoned",
                               RoundReason.SETUP_UNAVAILABLE, Stage.SETUP, ["skipped", "skipped"]),
    "setup-oserror": ([OSError("no docker")], None, MODIFY, "abandoned",
                      RoundReason.SETUP_UNAVAILABLE, Stage.SETUP, ["skipped", "skipped"]),
    "tree-invalid-after-setup": ([bloat], None, MODIFY, "failed",
                                 MinerReason.RESULT_TREE_INVALID, Stage.RESULT_TREE, ["skipped", "skipped"]),
    "tree-invalid-after-invocation": ([result(), plant_symlink], None, MODIFY, "failed",
                                      MinerReason.RESULT_TREE_INVALID, Stage.RESULT_TREE, ["failed", "skipped"]),
    "missing-check-cwd": ([], lambda: {**repo_manifest(setup=None), "checks": [invocation(cwd="missing"), inspection()]},
                          MODIFY, "failed", MinerReason.WORKING_DIRECTORY_INVALID, Stage.CHECK, ["failed", "skipped"]),
    "patch-static-nul": ([], None, b"\x00binary", "rejected",
                         MinerReason.PATCH_STATIC_REJECTED, Stage.PATCH, []),
    "patch-static-binary": ([], None, b"diff --git a/b b/b\nGIT binary patch\n", "rejected",
                            MinerReason.PATCH_STATIC_REJECTED, Stage.PATCH, []),
    "patch-static-mode": ([], None, b"diff --git a/t b/t\nnew file mode 120000\n--- /dev/null\n+++ b/t\n@@ -0,0 +1 @@\n+x\n",
                          "rejected", MinerReason.PATCH_STATIC_REJECTED, Stage.PATCH, []),
    "patch-context-mismatch": ([], None, MODIFY.replace(b" line one\n", b" line one!\n"), "rejected",
                               MinerReason.PATCH_APPLY_FAILED, Stage.PATCH, []),
    "patch-unsafe-path": ([], None, ESCAPE, "rejected", MinerReason.PATCH_APPLY_FAILED, Stage.PATCH, []),
}


@needs_git
@pytest.mark.parametrize("case", REPO_CASES.keys())
def test_repository_detection_sites_keep_status_and_gain_codes(tmp_path, runner, case):
    responses, doc, patch, status, code, stage, outcomes = REPO_CASES[case]
    runner.responses = [item(tmp_path) if item in (bloat, plant_symlink) else item for item in responses]
    verdict = eval_repo(tmp_path, runner, doc() if doc else None, patch=patch)
    assert_classified(verdict, status, code, stage)
    assert [check.outcome for check in verdict.checks] == outcomes
    assert str(tmp_path) not in verdict.reason
    if status != "passed":
        assert 0 < len(verdict.reason) <= 200


@needs_git
@pytest.mark.parametrize("kind", ["missing", "symlink", "absent-workspace"])
def test_unavailable_working_directory_is_a_workspace_fault(tmp_path, runner, kind):
    workspace, verifier, manifest = repo_setup(tmp_path)
    if kind == "symlink":
        (workspace / "missing").symlink_to("sub", target_is_directory=True)
    if kind == "absent-workspace":
        workspace = tmp_path / "absent"
    verdict = _mod().evaluate_repository(
        workspace, MODIFY, repo_identity("missing" if kind != "absent-workspace" else "."), manifest,
        verifier, policy(), tree_limits(), patch_limits(), "/usr/bin/docker", RUN,
    )
    assert_classified(verdict, "abandoned", RoundReason.WORKSPACE_INVALID, Stage.WORKSPACE_MATERIALIZATION)
    assert [check.outcome for check in verdict.checks] == ["skipped", "skipped"]
    assert runner.requests == []


@needs_git
def test_patch_tool_fault_on_baseline_is_abandoned_with_patch_tool_code(tmp_path, runner):
    workspace, verifier, manifest = repo_setup(tmp_path)
    (workspace / ".git").mkdir()
    verdict = _mod().evaluate_repository(
        workspace, MODIFY, repo_identity(), manifest, verifier, policy(), tree_limits(),
        patch_limits(), "/usr/bin/docker", RUN,
    )
    assert_classified(verdict, "abandoned", RoundReason.PATCH_TOOL_FAILED, Stage.PATCH)
    assert verdict.checks == () and runner.requests == []


@needs_git
def test_patch_that_removes_the_working_directory_fails_at_patch_stage(tmp_path, runner):
    verdict = eval_repo(tmp_path, runner, patch=DELETE_SUB, working_directory="sub")
    assert_classified(verdict, "failed", MinerReason.WORKING_DIRECTORY_INVALID, Stage.PATCH)
    assert [check.outcome for check in verdict.checks] == ["skipped", "skipped"]
    assert runner.requests == []
    assert not (tmp_path / "ws" / "sub").exists()


@needs_git
@pytest.mark.parametrize("apply_cmd, apply_rc, status, code", [
    pytest.param("/bin/ln -s /etc/passwd link", 0, "rejected", MinerReason.RESULT_TREE_INVALID, id="post-apply-symlink"),
    pytest.param("", 1, "abandoned", RoundReason.PATCH_TOOL_FAILED, id="apply-fails-after-clean-check"),
], )
def test_host_git_post_check_outcomes(tmp_path, runner, monkeypatch, apply_cmd, apply_rc, status, code):
    from rlvr.v3.patch import apply_patch

    git = fake_git(tmp_path, version="2.0.0", apply_cmd=apply_cmd, apply_rc=apply_rc)
    monkeypatch.setattr(
        _mod(), "apply_patch_in_container",
        lambda workspace, patch, limits, **_kw: apply_patch(
            workspace, patch, limits, git_binary=str(git), expected_git_version="2.0.0"),
    )
    verdict = eval_repo(tmp_path, runner)
    assert_classified(verdict, status, code, Stage.PATCH)
    assert verdict.checks == () and runner.requests == []
    assert str(tmp_path) not in verdict.reason


# --------------------------------------------------------------------------- #
# Container patch tool sites (the production path)
# --------------------------------------------------------------------------- #
def container_apply(tmp_path, monkeypatch, *responses):
    fake = FakeRunner(*responses)
    monkeypatch.setattr(patch_module, "run_container", fake)
    workspace = make_workspace(tmp_path)
    outcome = apply_patch_in_container(
        workspace, MODIFY, PatchLimits(max_patch_bytes=1 << 20, git_timeout_s=5),
        supervisor_policy=policy(), docker_binary="/usr/bin/docker", run_prefix=RUN,
    )
    return outcome, fake


def test_container_static_rejection_never_starts_a_container(tmp_path, monkeypatch):
    fake = FakeRunner()
    monkeypatch.setattr(patch_module, "run_container", fake)
    outcome = apply_patch_in_container(
        make_workspace(tmp_path), b"\x00", PatchLimits(max_patch_bytes=1 << 20, git_timeout_s=5),
        supervisor_policy=policy(), docker_binary="/usr/bin/docker", run_prefix=RUN,
    )
    assert (outcome.status, outcome.reason_code) == ("rejected", MinerReason.PATCH_STATIC_REJECTED)
    assert fake.requests == []


def test_container_check_rejection_is_patch_apply_failed_with_first_stderr_line(tmp_path, monkeypatch):
    stderr = b"error: patch failed: f.txt:1\nerror: f.txt: patch does not apply\n"
    outcome, fake = container_apply(tmp_path, monkeypatch, result(exit_code=1, stdout=b"", stderr=stderr))
    assert (outcome.status, outcome.reason_code) == ("rejected", MinerReason.PATCH_APPLY_FAILED)
    assert outcome.reason == "error: patch failed: f.txt:1"
    assert [request.name for request in fake.requests] == [f"{RUN}-patch-check"]


def test_container_apply_failure_after_clean_check_is_patch_apply_failed(tmp_path, monkeypatch):
    outcome, fake = container_apply(tmp_path, monkeypatch, result(exit_code=0), result(exit_code=1))
    assert (outcome.status, outcome.reason_code) == ("rejected", MinerReason.PATCH_APPLY_FAILED)
    assert outcome.reason == "git apply failed"
    assert [request.name for request in fake.requests] == [f"{RUN}-patch-check", f"{RUN}-patch-apply"]


def test_container_post_apply_unsafe_tree_is_result_tree_invalid(tmp_path, monkeypatch):
    def plant(request):
        (tmp_path / "ws" / "link").symlink_to("/etc/passwd")
        return result(exit_code=0)

    outcome, _ = container_apply(tmp_path, monkeypatch, result(exit_code=0), plant)
    assert (outcome.status, outcome.reason_code) == ("rejected", MinerReason.RESULT_TREE_INVALID)
    assert str(tmp_path) not in outcome.reason


@pytest.mark.parametrize("first", [
    pytest.param(SupervisorError("no docker"), id="supervisor-error"),
    pytest.param(result(exit_code=127, stdout=b"", stderr=b""), id="git-missing-127"),
    pytest.param(result(exit_code=126, stdout=b"", stderr=b""), id="git-not-executable-126"),
    pytest.param(result(timed_out=True, exit_code=124), id="check-timeout"),
    pytest.param(result(stderr_overflow=True), id="check-overflow"),
])
def test_container_tool_faults_raise_patch_tool_error_not_a_miner_code(tmp_path, monkeypatch, first):
    with pytest.raises(PatchToolError):
        container_apply(tmp_path, monkeypatch, first)


# --------------------------------------------------------------------------- #
# Terminal grading sites
# --------------------------------------------------------------------------- #
def wipe_result_tree(tmp_path):
    def act(request):
        (tmp_path / "env" / "out" / "seed").unlink()
        (tmp_path / "env" / "out").rmdir()
        return result()
    return act


def escape_result_tree(tmp_path):
    def act(request):
        (tmp_path / "env" / "out" / "link").symlink_to("/etc")
        return result()
    return act


TERM_CASES = {
    "passed-with-nonzero-script": ([result(exit_code=3, stdout=b""), result(), result()], None, None,
                                   "passed", None, Stage.CHECK, ["passed", "passed"], 3),
    "script-timeout": ([result(timed_out=True, exit_code=124)], None, None, "failed",
                       MinerReason.TIMEOUT, Stage.SCRIPT, ["skipped", "skipped"], 124),
    "script-oom": ([result(oom_killed=True, exit_code=137)], None, None, "failed",
                   MinerReason.MEMORY_LIMIT, Stage.SCRIPT, ["skipped", "skipped"], 137),
    "script-stdout-overflow": ([result(stdout_overflow=True)], None, None, "failed",
                               MinerReason.OUTPUT_LIMIT, Stage.SCRIPT, ["skipped", "skipped"], 0),
    "script-supervisor-error": ([SupervisorError("boom")], None, None, "abandoned",
                                RoundReason.SCRIPT_UNAVAILABLE, Stage.SCRIPT, ["skipped", "skipped"], None),
    "script-oserror": ([OSError("no docker")], None, None, "abandoned",
                       RoundReason.SCRIPT_UNAVAILABLE, Stage.SCRIPT, ["skipped", "skipped"], None),
    "result-tree-wiped-by-script": ([wipe_result_tree], None, None, "failed",
                                    MinerReason.RESULT_TREE_INVALID, Stage.RESULT_TREE, ["skipped", "skipped"], 0),
    "result-tree-escapes-after-script": ([escape_result_tree], None, None, "failed",
                                         MinerReason.RESULT_TREE_INVALID, Stage.RESULT_TREE, ["skipped", "skipped"], 0),
    "inspection-mismatch": ([result(), result(stdout=b"bad\n")], None, None, "failed",
                            MinerReason.CHECK_FAILED, Stage.CHECK, ["failed", "skipped"], 0),
    "inspection-127": ([result(), result(exit_code=127, stdout=b"", stderr=b"")], None, None, "abandoned",
                       RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK, ["failed", "skipped"], 0),
    "script-rejected": ([], None, b"echo\x00", "rejected", MinerReason.SCRIPT_REJECTED, Stage.SCRIPT, [], None),
    "setup-nonzero": ([result(exit_code=1)], lambda: term_manifest(setup=SETUP), None, "failed",
                      MinerReason.SETUP_FAILED, Stage.SETUP, ["skipped", "skipped"], None),
    "setup-timeout": ([result(timed_out=True)], lambda: term_manifest(setup=SETUP), None, "failed",
                      MinerReason.TIMEOUT, Stage.SETUP, ["skipped", "skipped"], None),
    "setup-supervisor-error": ([SupervisorError("offline")], lambda: term_manifest(setup=SETUP), None, "abandoned",
                               RoundReason.SETUP_UNAVAILABLE, Stage.SETUP, ["skipped", "skipped"], None),
    "setup-removes-result-directory": ([wipe_result_tree], lambda: term_manifest(setup=SETUP), None, "failed",
                                       MinerReason.WORKING_DIRECTORY_INVALID, Stage.SETUP, ["skipped", "skipped"], None),
}


@pytest.mark.parametrize("case", TERM_CASES.keys())
def test_terminal_detection_sites_keep_status_and_gain_codes(tmp_path, runner, case):
    responses, doc, script, status, code, stage, outcomes, script_exit = TERM_CASES[case]
    runner.responses = [
        item(tmp_path) if item in (wipe_result_tree, escape_result_tree) else item for item in responses
    ]
    kwargs = {} if script is None else {"script": script}
    verdict = eval_term(tmp_path, runner, doc() if doc else None, **kwargs)
    assert_classified(verdict, status, code, stage)
    assert [check.outcome for check in verdict.checks] == outcomes
    assert verdict.script_exit_code == script_exit
    assert str(tmp_path) not in verdict.reason


def test_terminal_result_tree_missing_before_the_script_is_a_workspace_fault(tmp_path, runner):
    environment, verifier, manifest = term_setup(tmp_path)
    (environment / "out" / "seed").unlink()
    (environment / "out").rmdir()
    verdict = _mod().evaluate_terminal(
        environment, b"true\n", term_identity(), manifest, verifier, policy(), tree_limits(),
        script_limits(), "/usr/bin/docker", RUN,
    )
    assert_classified(verdict, "abandoned", RoundReason.WORKSPACE_INVALID, Stage.WORKSPACE_MATERIALIZATION)
    assert [check.outcome for check in verdict.checks] == ["skipped", "skipped"]
    assert runner.requests == []


# --------------------------------------------------------------------------- #
# Verifier authoring faults that reach the check loop
# --------------------------------------------------------------------------- #
@needs_git
def test_missing_checks_directory_for_an_inspection_is_verifier_unavailable(tmp_path, runner):
    doc = repo_manifest(setup=None, checks=[inspection("c02", argv=["/usr/bin/true"])])
    workspace, verifier, manifest = repo_setup(tmp_path, doc)
    shutil.rmtree(verifier / "checks")
    verdict = _mod().evaluate_repository(
        workspace, MODIFY, repo_identity(), manifest, verifier, policy(), tree_limits(),
        patch_limits(), "/usr/bin/docker", RUN,
    )
    assert_classified(verdict, "abandoned", RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK)
    assert [check.outcome for check in verdict.checks] == ["skipped"]
    assert runner.requests == []


@needs_git
def test_missing_gold_reference_at_run_time_is_verifier_unavailable(tmp_path, runner):
    doc = repo_manifest(setup=None)
    doc["checks"][0]["expect"] = expect(stdout="gold/late.stdout")
    workspace, verifier, manifest = repo_setup(tmp_path, doc)
    (verifier / "gold" / "late.stdout").unlink()
    runner.responses = [result()]
    verdict = _mod().evaluate_repository(
        workspace, MODIFY, repo_identity(), manifest, verifier, policy(), tree_limits(),
        patch_limits(), "/usr/bin/docker", RUN,
    )
    assert_classified(verdict, "abandoned", RoundReason.VERIFIER_UNAVAILABLE, Stage.CHECK)
    assert [check.outcome for check in verdict.checks] == ["skipped", "skipped"]
