"""Contract for ``rlvr.v3.grading`` — grading orchestration over the manifest / tree /
patch / script / supervisor primitives (no new framework).

    CheckResult(check_id, kind, outcome passed|failed|skipped, exit_code int|None)   frozen
    EvaluationResult(status passed|failed|rejected|abandoned, reason <=200 ("" iff passed),
                     checks tuple, script_exit_code int|None)   frozen
    evaluate_repository(workspace, patch, identity, manifest, verifier_dir, supervisor_policy,
        tree_limits, patch_limits, *, git_binary, expected_git_version, docker_binary, run_prefix)
    evaluate_terminal(environment, script, identity, manifest, verifier_dir, supervisor_policy,
        tree_limits, script_limits, *, docker_binary, run_prefix); script budget = SCRIPT_TIMEOUT_S/SCRIPT_OUTPUT_BYTES

Repository: apply_patch → optional setup ("<prefix>-setup", candidate, /work rw = workspace, cwd
/work/<working_directory>) → inspect_tree(workspace, tree_limits, normalize) → checks.  Terminal:
validate_script → script 0444 mounted ro at /submission → "<prefix>-script" runs /usr/bin/bash --noprofile
--norc /submission/script.sh at /work (environment rw); nonzero exit recorded, state still graded;
timeout/OOM/overflow fail → inspect_tree(environment/<result_tree_path>).
Checks in manifest order.  invocation: candidate "<prefix>-<id>", mounts EXACTLY /work rw + /verify ro
(verifier_dir/checks), cwd /work/<working_directory>/<check.cwd> (terminal: /work/<cwd>), stdin = inputs ref
bytes or b"", gold never mounted, byte-exact stdout/exit (stderr only when expected), tree re-inspected after
EVERY invocation.  inspection: trusted, mounts EXACTLY /result ro + /verify ro, cwd /result.  First failure
short-circuits, rest "skipped".  Supervisor/PatchTool/host fault → abandoned; patch/script structural
rejection → rejected; setup/check/tree/resource failure → failed.
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

import pytest

from tests.test_v3_manifest import (
    build_dir,
    expect,
    inspection,
    invocation,
    parse,
    repo_manifest,
    term_manifest,
)
from tests.test_v3_patch import MODIFY, git_version, make_workspace
from tests.test_v3_supervisor import policy

HEX = "0" * 64
RUN = "hone-r1"


def _mod():
    from rlvr.v3 import grading
    return grading


def _sup():
    from rlvr.v3 import supervisor
    return supervisor


def repo_identity(working_directory="."):
    from rlvr.v3.identity import RepositoryTaskIdentity
    return RepositoryTaskIdentity(task_kind="bug_fix", instruction="fix", primary_language="python",
                                  workspace_sha256=HEX, verifier_sha256=HEX, execution_profile_id="py",
                                  working_directory=working_directory, verifier_policy="p1", authoring_version="1")


def term_identity(result_tree_path="out"):
    from rlvr.v3.identity import TerminalScriptTaskIdentity
    return TerminalScriptTaskIdentity(instruction="do", environment_sha256=HEX, verifier_sha256=HEX,
                                      execution_profile_id="sh", result_tree_path=result_tree_path,
                                      verifier_policy="p1", authoring_version="1")


def result(exit_code=0, stdout=b"x\n", stderr=b"", **over):
    base = dict(exit_code=exit_code, oom_killed=False, timed_out=False, stdout=stdout, stderr=stderr, stdout_overflow=False)
    return _sup().ContainerResult(**{**base, "stderr_overflow": False, **over})


class FakeRunner:
    """Scripted run_container: records requests; entries are ContainerResult, exception, or callable(request)."""
    def __init__(self, *responses):
        self.responses, self.requests, self.policies, self.binaries = list(responses), [], [], []

    def __call__(self, request, policy, docker_binary):
        self.requests.append(request), self.policies.append(policy), self.binaries.append(docker_binary)
        item = self.responses.pop(0) if self.responses else result()
        if isinstance(item, BaseException):
            raise item
        return item(request) if callable(item) else item


@pytest.fixture
def runner(monkeypatch):
    fake = FakeRunner()
    monkeypatch.setattr(_mod(), "run_container", fake, raising=True)
    from rlvr.v3.patch import apply_patch
    monkeypatch.setattr(
        _mod(),
        "apply_patch_in_container",
        lambda workspace, patch, limits, **_kwargs: apply_patch(
            workspace,
            patch,
            limits,
            git_binary="git",
            expected_git_version=git_version(),
        ),
    )
    return fake


def tree_limits():
    from rlvr.v3.tree import TreeLimits
    return TreeLimits(max_entries=256, max_file_bytes=1 << 20, max_total_file_bytes=8 << 20, max_path_bytes=512)


def patch_limits():
    from rlvr.v3.patch import PatchLimits
    return PatchLimits(max_patch_bytes=1 << 20, git_timeout_s=10)


def script_limits():
    from rlvr.v3.script import ScriptLimits
    return ScriptLimits(max_script_bytes=1 << 20)


def repo_setup(tmp_path, doc=None, working_directory="."):
    doc = doc or repo_manifest()
    (tmp_path / "v").mkdir()
    return make_workspace(tmp_path), build_dir(tmp_path / "v", doc), parse(doc)


def eval_repo(tmp_path, runner, doc=None, *, patch=MODIFY, working_directory=".", workspace=None):
    ws, verifier, manifest = repo_setup(tmp_path, doc)
    return _mod().evaluate_repository(
        workspace or ws, patch, repo_identity(working_directory), manifest, verifier, policy(), tree_limits(),
        patch_limits(), docker_binary="/usr/bin/docker",
        run_prefix=RUN)


def term_setup(tmp_path, doc=None):
    doc = doc or term_manifest()
    env = tmp_path / "env"
    (env / "out").mkdir(parents=True)
    (env / "out" / "seed").write_bytes(b"s")
    (tmp_path / "v").mkdir()
    return env, build_dir(tmp_path / "v", doc), parse(doc)


def eval_term(tmp_path, runner, doc=None, *, script=b"#!/bin/bash\necho hi > /work/out/a\n"):
    env, verifier, manifest = term_setup(tmp_path, doc)
    return _mod().evaluate_terminal(
        env, script, term_identity(), manifest, verifier, policy(), tree_limits(), script_limits(),
        docker_binary="/usr/bin/docker", run_prefix=RUN)


def mounts(request): return {m.container: (m.host, m.read_only) for m in request.mounts}  # noqa: E704


def test_result_types_are_frozen_and_bounded():
    m = _mod()
    check = m.CheckResult(check_id="c01", kind="invocation", outcome="passed", exit_code=0)
    ok = m.EvaluationResult(status="passed", reason="", checks=(check,), script_exit_code=None)
    assert ok.checks[0].exit_code == 0 and ok.script_exit_code is None
    for bad in ({"status": "passed", "reason": "x"}, {"status": "failed", "reason": ""},
                {"status": "failed", "reason": "r" * 201}, {"status": "unknown", "reason": "r"}):
        with pytest.raises(ValueError):
            m.EvaluationResult(checks=(), script_exit_code=None, **bad)
    with pytest.raises(ValueError):
        m.CheckResult(check_id="c01", kind="invocation", outcome="maybe", exit_code=0)
    with pytest.raises(Exception):
        ok.status = "failed"  # frozen


def test_repository_happy_path_pins_every_container_request(tmp_path, runner):
    doc = repo_manifest()
    doc["checks"][0].update(cwd=".", expect=expect(stdout="gold/c01.stdout", stderr="gold/c01.stderr"))
    runner.responses = [result(), result(stderr=b"x\n"), result()]
    verdict = eval_repo(tmp_path, runner, doc, working_directory="sub")
    assert verdict.status == "passed" and verdict.reason == "" and verdict.script_exit_code is None
    assert [(c.check_id, c.kind, c.outcome, c.exit_code) for c in verdict.checks] == [
        ("c01", "invocation", "passed", 0), ("c02", "inspection", "passed", 0)]
    setup, c01, c02 = runner.requests
    ws, verifier = tmp_path / "ws", tmp_path / "v" / "verifier"
    assert (setup.name, setup.cwd, setup.trusted, setup.stdin) == (f"{RUN}-setup", "/work/sub", False, b"")
    assert setup.argv == ("/usr/bin/python3", "-m", "compileall", "-q", ".") and setup.timeout_s == 60
    assert (setup.max_stdout_bytes, setup.max_stderr_bytes) == (65536, 65536)
    assert mounts(setup) == {"/work": (ws, False)}
    assert (c01.name, c01.cwd, c01.trusted, c01.timeout_s) == (f"{RUN}-c01", "/work/sub", False, 10)
    assert c01.argv == ("/usr/bin/python3", "-I", "/work/greet.py") and c01.stdin == b"x\n"
    assert mounts(c01) == {"/work": (ws, False)}
    assert (c02.name, c02.cwd, c02.trusted) == (f"{RUN}-c02", "/result", True)
    assert mounts(c02) == {"/result": (ws, True), "/verify": (verifier / "checks", True)}
    assert all(p == policy() and b == "/usr/bin/docker" for p, b in zip(runner.policies, runner.binaries))
    assert (ws / "f.txt").read_bytes().count(b"TWO") == 1  # patch really applied


def test_no_inputs_or_gold_are_ever_mounted_and_stdin_defaults_to_empty(tmp_path, runner):
    doc = repo_manifest()
    doc["checks"][0]["stdin"] = None
    eval_repo(tmp_path, runner, doc)
    verifier = tmp_path / "v" / "verifier"
    for request in runner.requests:
        for m in request.mounts:
            assert m.host in (verifier / "checks", tmp_path / "ws")  # never inputs/ or gold/
    assert runner.requests[1].stdin == b""


def test_null_setup_skips_the_setup_container(tmp_path, runner):
    eval_repo(tmp_path, runner, repo_manifest(setup=None))
    assert [r.name for r in runner.requests] == [f"{RUN}-c01", f"{RUN}-c02"]


def test_tree_is_reinspected_after_every_invocation(tmp_path, runner, monkeypatch):
    import rlvr.v3.tree as tree

    seen = []
    real = tree.inspect_tree
    monkeypatch.setattr(_mod(), "inspect_tree", lambda root, **kw: seen.append((Path(root), kw)) or real(root, **kw),
                        raising=True)
    doc = repo_manifest(checks=[invocation("c01"), invocation("c03"), inspection("c02")])
    eval_repo(tmp_path, runner, doc)
    roots = [r for r, _ in seen]
    assert roots.count(tmp_path / "ws") >= 3  # after setup, after c01, after c03
    assert all(kw == dict(limits=tree_limits(), normalize_modes=True) for _, kw in seen)


def test_invocation_that_plants_a_symlink_fails_before_inspection_runs(tmp_path, runner):
    def plant(request):
        (tmp_path / "ws" / "evil").symlink_to("/etc/passwd")
        return result()

    runner.responses = [result(), plant]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and [r.name for r in runner.requests] == [f"{RUN}-setup", f"{RUN}-c01"]
    assert [(c.check_id, c.outcome) for c in verdict.checks] == [("c01", "failed"), ("c02", "skipped")]


@pytest.mark.parametrize("bad, field", [
    (result(stdout=b"wrong\n"), "stdout"), (result(exit_code=1), "exit"), (result(timed_out=True, exit_code=124), "timeout"),
    (result(oom_killed=True, exit_code=137), "oom"), (result(stdout_overflow=True), "overflow"),
], ids=["stdout", "exit", "timeout", "oom", "overflow"])
def test_check_mismatch_fails_and_skips_the_rest(tmp_path, runner, bad, field):
    runner.responses = [result(), bad]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and 0 < len(verdict.reason) <= 200
    assert [(c.check_id, c.outcome) for c in verdict.checks] == [("c01", "failed"), ("c02", "skipped")]
    assert verdict.checks[0].exit_code == bad.exit_code and verdict.checks[1].exit_code is None
    assert len(runner.requests) == 2


def test_stderr_is_compared_only_when_expected(tmp_path, runner):
    runner.responses = [result(), result(stderr=b"noise"), result()]
    assert eval_repo(tmp_path, runner).status == "passed"
    doc = repo_manifest()
    doc["checks"][0]["expect"] = expect(stderr="gold/c01.stderr")
    (tmp_path / "b").mkdir()
    runner.responses = [result(), result(stderr=b"noise"), result()]
    assert eval_repo(tmp_path / "b", runner, doc).status == "failed"


def test_setup_failure_is_failed_with_all_checks_skipped(tmp_path, runner):
    runner.responses = [result(exit_code=2)]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and [c.outcome for c in verdict.checks] == ["skipped", "skipped"]
    assert len(runner.requests) == 1


def test_tree_limit_violation_after_candidate_stage_is_failed(tmp_path, runner):
    def bloat(request):
        (tmp_path / "ws" / "big").write_bytes(b"x" * (2 << 20))
        return result()

    runner.responses = [bloat]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed" and [c.outcome for c in verdict.checks] == ["skipped", "skipped"]


def test_missing_invocation_cwd_fails_without_docker_creating_it(tmp_path, runner):
    doc = repo_manifest(setup=None)
    doc["checks"][0]["cwd"] = "missing"
    verdict = eval_repo(tmp_path, runner, doc)
    assert verdict.status == "failed"
    assert runner.requests == []
    assert not (tmp_path / "ws" / "missing").exists()


def test_patch_rejection_is_rejected_without_containers(tmp_path, runner):
    verdict = eval_repo(tmp_path, runner, patch=b"\x00binary")
    assert verdict.status == "rejected" and 0 < len(verdict.reason) <= 200 and verdict.checks == () and not runner.requests


def test_patch_tool_fault_is_abandoned(tmp_path, runner):
    verdict = eval_repo(tmp_path, runner, workspace=tmp_path / "absent-workspace")
    assert verdict.status == "abandoned" and runner.requests == []


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_repository_working_directory_must_exist_before_patch_or_docker(
    tmp_path, runner, kind
):
    workspace, verifier, manifest = repo_setup(tmp_path)
    if kind == "symlink":
        (workspace / "missing").symlink_to("sub", target_is_directory=True)
    verdict = _mod().evaluate_repository(
        workspace,
        MODIFY,
        repo_identity("missing"),
        manifest,
        verifier,
        policy(),
        tree_limits(),
        patch_limits(),
        docker_binary="/usr/bin/docker",
        run_prefix=RUN,
    )
    assert verdict.status == "abandoned"
    assert runner.requests == []
    assert b"TWO" not in (workspace / "f.txt").read_bytes()


def test_supervisor_fault_is_abandoned_and_check_count_preserved(tmp_path, runner):
    runner.responses = [result(), _sup().SupervisorError("boom")]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "abandoned" and 0 < len(verdict.reason) <= 200 and str(tmp_path) not in verdict.reason
    assert [(c.check_id, c.outcome) for c in verdict.checks] == [("c01", "skipped"), ("c02", "skipped")]


@pytest.mark.parametrize("exit_code", [126, 127])
def test_missing_trusted_inspection_executable_abandons_round(tmp_path, runner, exit_code):
    runner.responses = [result(), result(), result(exit_code=exit_code, stdout=b"", stderr=b"")]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "abandoned"
    assert [check.outcome for check in verdict.checks] == ["passed", "failed"]


@pytest.mark.parametrize(
    "failure",
    [
        result(timed_out=True, exit_code=124),
        result(oom_killed=True, exit_code=137),
        result(stdout_overflow=True),
        result(stderr_overflow=True),
    ],
)
def test_trusted_inspection_resource_failure_only_fails_candidate(tmp_path, runner, failure):
    runner.responses = [result(), result(), failure]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed"
    assert [check.outcome for check in verdict.checks] == ["passed", "failed"]


def test_candidate_exit_127_is_still_a_miner_failure(tmp_path, runner):
    runner.responses = [result(), result(exit_code=127, stdout=b"", stderr=b"")]
    verdict = eval_repo(tmp_path, runner)
    assert verdict.status == "failed"


def test_manifest_task_type_must_match_identity(tmp_path, runner):
    ws, verifier, manifest = term_setup(tmp_path)
    with pytest.raises(ValueError):
        _mod().evaluate_repository(make_workspace(tmp_path), MODIFY, repo_identity(), manifest, verifier, policy(),
                                   tree_limits(), patch_limits(),
                                   docker_binary="/usr/bin/docker", run_prefix=RUN)
    assert runner.requests == []


def test_terminal_happy_path_pins_script_container_and_result_root(tmp_path, runner):
    runner.responses = [result(exit_code=3, stdout=b""), result(), result()]
    verdict = eval_term(tmp_path, runner)
    assert verdict.status == "passed" and verdict.script_exit_code == 3  # nonzero recorded, state still graded
    script, c01, c02 = runner.requests
    env = tmp_path / "env"
    assert (script.name, script.cwd, script.trusted, script.stdin) == (f"{RUN}-script", "/work/out", False, b"")
    assert script.argv == ("/usr/bin/bash", "--noprofile", "--norc", "/submission/script.sh")
    m = _mod()
    assert (script.timeout_s, script.max_stdout_bytes, script.max_stderr_bytes) == (
        m.SCRIPT_TIMEOUT_S, m.SCRIPT_OUTPUT_BYTES, m.SCRIPT_OUTPUT_BYTES)
    assert 1 <= m.SCRIPT_TIMEOUT_S <= 3600 and 1 <= m.SCRIPT_OUTPUT_BYTES <= 8 << 20
    assert mounts(script)["/work"] == (env, False)
    host, ro = mounts(script)["/submission"]
    assert ro and host != env and not str(host).startswith(str(env))
    assert set(mounts(script)) == {"/work", "/submission"}
    assert (c01.name, c01.trusted) == (f"{RUN}-c01", True) and mounts(c01)["/result"] == (env / "out", True)
    assert set(mounts(c01)) == {"/result", "/verify"} and mounts(c02) == mounts(c01)


def test_script_file_is_written_read_only_with_exact_bytes(tmp_path, runner):
    seen = {}

    def capture(request):
        host = mounts(request)["/submission"][0]
        seen["bytes"] = (host / "script.sh").read_bytes()
        seen["mode"] = stat.S_IMODE((host / "script.sh").lstat().st_mode)
        return result()

    runner.responses = [capture]
    eval_term(tmp_path, runner, script="echo é\n".encode())
    assert seen == {"bytes": "echo é\n".encode(), "mode": 0o444}


@pytest.mark.parametrize("bad", [result(timed_out=True, exit_code=124), result(oom_killed=True, exit_code=137),
                                 result(stdout_overflow=True), result(stderr_overflow=True)],
                         ids=["timeout", "oom", "stdout-overflow", "stderr-overflow"])
def test_script_resource_failures_fail_before_checks(tmp_path, runner, bad):
    runner.responses = [bad]
    verdict = eval_term(tmp_path, runner)
    assert (verdict.status, verdict.script_exit_code, len(runner.requests)) == ("failed", bad.exit_code, 1)
    assert [c.outcome for c in verdict.checks] == ["skipped", "skipped"]


def test_invalid_script_is_rejected_without_containers(tmp_path, runner):
    verdict = eval_term(tmp_path, runner, script=b"echo\x00")
    assert verdict.status == "rejected" and runner.requests == [] and verdict.script_exit_code is None


def test_terminal_result_tree_is_inspected_under_result_tree_path(tmp_path, runner):
    def escape(request):
        (tmp_path / "env" / "out" / "link").symlink_to("/etc")
        return result()

    runner.responses = [escape]
    verdict = eval_term(tmp_path, runner)
    assert verdict.status == "failed" and len(runner.requests) == 1


def test_terminal_missing_result_tree_is_failed_not_abandoned(tmp_path, runner):
    def wipe(request):
        (tmp_path / "env" / "out" / "seed").unlink()
        (tmp_path / "env" / "out").rmdir()
        return result()

    runner.responses = [wipe]
    assert eval_term(tmp_path, runner).status == "failed"


def test_terminal_result_tree_must_exist_before_script_starts(tmp_path, runner):
    environment, verifier, manifest = term_setup(tmp_path)
    (environment / "out" / "seed").unlink()
    (environment / "out").rmdir()
    verdict = _mod().evaluate_terminal(
        environment,
        b"true\n",
        term_identity(),
        manifest,
        verifier,
        policy(),
        tree_limits(),
        script_limits(),
        docker_binary="/usr/bin/docker",
        run_prefix=RUN,
    )
    assert verdict.status == "abandoned"
    assert runner.requests == []


def test_terminal_supervisor_fault_is_abandoned(tmp_path, runner):
    runner.responses = [_sup().SupervisorError("boom")]
    verdict = eval_term(tmp_path, runner)
    assert (verdict.status, verdict.script_exit_code, [c.outcome for c in verdict.checks]) == (
        "abandoned", None, ["skipped", "skipped"])


@pytest.mark.parametrize("stage", ["script", "invocation", "setup"])
def test_terminal_rejects_replaced_result_ancestor_without_touching_host(
    tmp_path, runner, stage
):
    from tests.test_v3_manifest import SETUP

    checks = [invocation("invoke"), inspection()] if stage == "invocation" else [inspection()]
    doc = term_manifest(setup=SETUP if stage == "setup" else None, checks=checks)
    env, verifier, manifest = term_setup(tmp_path, doc)
    (env / "nested" / "out").mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "out").mkdir(parents=True)
    private = outside / "out" / "private"
    private.write_bytes(b"private fixture")
    private.chmod(0o600)

    def replace_ancestor(request):
        shutil.rmtree(env / "nested")
        (env / "nested").symlink_to(outside, target_is_directory=True)
        return result()

    runner.responses = ([result()] if stage == "invocation" else []) + [replace_ancestor]
    verdict = _mod().evaluate_terminal(
        env, b"true\n", term_identity("nested/out"), manifest, verifier,
        policy(), tree_limits(), script_limits(), "/usr/bin/docker", RUN,
    )
    assert verdict.status == "failed"
    assert len(runner.requests) == (2 if stage == "invocation" else 1)
    assert all(not request.trusted for request in runner.requests)
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert private.read_bytes() == b"private fixture"


def test_terminal_setup_runs_before_script_in_result_directory(tmp_path, runner):
    from tests.test_v3_manifest import SETUP

    def prepare(request):
        (tmp_path / "env" / "out" / "prepared").write_bytes(b"ready")
        return result()

    def script(request):
        assert (tmp_path / "env" / "out" / "prepared").read_bytes() == b"ready"
        return result()

    runner.responses = [prepare, script]
    verdict = eval_term(tmp_path, runner, term_manifest(setup=SETUP))
    assert verdict.status == "passed"
    setup = runner.requests[0]
    assert setup.name == f"{RUN}-setup" and setup.cwd == "/work/out"
    assert setup.argv == tuple(SETUP["argv"])
    assert not setup.trusted and mounts(setup) == {"/work": (tmp_path / "env", False)}
    assert runner.requests[1].name == f"{RUN}-script"


@pytest.mark.parametrize("failure,status", [
    (result(exit_code=1), "failed"),
    (result(timed_out=True), "failed"),
    (_sup().SupervisorError("offline"), "abandoned"),
])
def test_terminal_setup_failure_stops_before_script(tmp_path, runner, failure, status):
    from tests.test_v3_manifest import SETUP

    runner.responses = [failure]
    verdict = eval_term(tmp_path, runner, term_manifest(setup=SETUP))
    assert verdict.status == status and verdict.script_exit_code is None
    assert [check.outcome for check in verdict.checks] == ["skipped", "skipped"]
    assert [request.name for request in runner.requests] == [f"{RUN}-setup"]
