from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from .identity import RepositoryTaskIdentity, TerminalScriptTaskIdentity
from .manifest import (
    Check,
    InspectionCheck,
    InvocationCheck,
    VerifierManifest,
)
from .patch import PatchLimits, PatchToolError, apply_patch_in_container
from .script import ScriptLimits, validate_script
from .supervisor import (
    ContainerRequest,
    ContainerResult,
    Mount,
    SupervisorError,
    SupervisorPolicy,
    run_container,
)
from .tree import TreeError, TreeLimits, inspect_tree

SCRIPT_TIMEOUT_S = 300
SCRIPT_OUTPUT_BYTES = 1 << 20


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    kind: Literal["invocation", "inspection"]
    outcome: Literal["passed", "failed", "skipped"]
    exit_code: int | None

    def __post_init__(self) -> None:
        if self.kind not in ("invocation", "inspection"):
            raise ValueError("check kind is invalid")
        if self.outcome not in ("passed", "failed", "skipped"):
            raise ValueError("check outcome is invalid")
        if self.outcome == "skipped" and self.exit_code is not None:
            raise ValueError("a skipped check cannot have an exit code")


@dataclass(frozen=True)
class EvaluationResult:
    status: Literal["passed", "failed", "rejected", "abandoned"]
    reason: str
    checks: tuple[CheckResult, ...]
    script_exit_code: int | None

    def __post_init__(self) -> None:
        if self.status not in ("passed", "failed", "rejected", "abandoned"):
            raise ValueError("evaluation status is invalid")
        if self.status == "passed":
            if self.reason:
                raise ValueError("a passed evaluation cannot have a reason")
        elif not 0 < len(self.reason) <= 200:
            raise ValueError("a non-passing evaluation requires a bounded reason")
        if type(self.checks) is not tuple or any(
            type(item) is not CheckResult for item in self.checks
        ):
            raise ValueError("evaluation checks must be a tuple")


def _kind(check: Check) -> Literal["invocation", "inspection"]:
    return "invocation" if isinstance(check, InvocationCheck) else "inspection"


def _skipped(check: Check) -> CheckResult:
    return CheckResult(check.check_id, _kind(check), "skipped", None)


def _stop(
    status: Literal["failed", "abandoned"],
    reason: str,
    manifest: VerifierManifest,
    completed: list[CheckResult],
    *,
    script_exit_code: int | None,
) -> EvaluationResult:
    remaining = manifest.checks[len(completed) :]
    return EvaluationResult(
        status=status,
        reason=reason[:200],
        checks=tuple(completed) + tuple(_skipped(check) for check in remaining),
        script_exit_code=script_exit_code,
    )


def _container_failure(result: ContainerResult) -> str | None:
    if result.timed_out:
        return "container timed out"
    if result.oom_killed:
        return "container exceeded its memory limit"
    if result.stdout_overflow or result.stderr_overflow:
        return "container exceeded its output limit"
    return None


def _work_cwd(base: str, relative: str) -> str:
    return base if relative == "." else str(PurePosixPath(base, relative))


def _task_directory(root: Path, relative: str) -> Path | None:
    current = root
    segments = () if relative == "." else relative.split("/")
    try:
        for segment in segments:
            current = current / segment
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                return None
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None
        return current
    except OSError:
        return None


def _inspect_result_tree(workspace: Path, result_tree: Path, limits: TreeLimits) -> None:
    # Candidate processes have stopped, but may have replaced any ancestor of
    # a nested result tree. Check from the immutable mount root before chmod
    # or a trusted bind mount can follow those ancestors on the host.
    relative = result_tree.relative_to(workspace).as_posix()
    if _task_directory(workspace, relative) is None:
        raise TreeError("result tree path is unavailable or unsafe")
    inspect_tree(result_tree, limits=limits, normalize_modes=True)


def _request(
    *,
    name: str,
    argv: tuple[str, ...],
    cwd: str,
    mounts: tuple[Mount, ...],
    stdin: bytes,
    timeout_s: int,
    stdout_bytes: int,
    stderr_bytes: int,
    trusted: bool,
) -> ContainerRequest:
    return ContainerRequest(
        name=name,
        argv=argv,
        cwd=cwd,
        mounts=mounts,
        stdin=stdin,
        timeout_s=timeout_s,
        max_stdout_bytes=stdout_bytes,
        max_stderr_bytes=stderr_bytes,
        trusted=trusted,
    )


def _run_checks(
    *,
    workspace: Path,
    host_work_base: Path,
    result_tree: Path,
    work_base: str,
    manifest: VerifierManifest,
    verifier_dir: Path,
    supervisor_policy: SupervisorPolicy,
    tree_limits: TreeLimits,
    docker_binary: str,
    run_prefix: str,
    script_exit_code: int | None,
) -> EvaluationResult:
    completed: list[CheckResult] = []
    try:
        _inspect_result_tree(workspace, result_tree, tree_limits)
    except TreeError:
        return _stop(
            "failed",
            "candidate result tree is invalid",
            manifest,
            completed,
            script_exit_code=script_exit_code,
        )

    for check in manifest.checks:
        try:
            if isinstance(check, InvocationCheck):
                if _task_directory(host_work_base, check.cwd) is None:
                    completed.append(
                        CheckResult(check.check_id, "invocation", "failed", None)
                    )
                    return _stop(
                        "failed",
                        "candidate working directory is unavailable",
                        manifest,
                        completed,
                        script_exit_code=script_exit_code,
                    )
                stdin = (
                    b""
                    if check.stdin is None
                    else (verifier_dir / check.stdin).read_bytes()
                )
                request = _request(
                    name=f"{run_prefix}-{check.check_id}",
                    argv=check.argv,
                    cwd=_work_cwd(work_base, check.cwd),
                    mounts=(Mount(workspace, "/work", False),),
                    stdin=stdin,
                    timeout_s=check.timeout_s,
                    stdout_bytes=check.max_stdout_bytes,
                    stderr_bytes=check.max_stderr_bytes,
                    trusted=False,
                )
            else:
                request = _request(
                    name=f"{run_prefix}-{check.check_id}",
                    argv=check.argv,
                    cwd="/result",
                    mounts=(
                        Mount(result_tree, "/result", True),
                        Mount(verifier_dir / "checks", "/verify", True),
                    ),
                    stdin=b"",
                    timeout_s=check.timeout_s,
                    stdout_bytes=check.max_stdout_bytes,
                    stderr_bytes=check.max_stderr_bytes,
                    trusted=True,
                )
            container = run_container(request, supervisor_policy, docker_binary)
            failure = _container_failure(container)
            infrastructure = False
            if (
                failure is None
                and isinstance(check, InspectionCheck)
                and container.exit_code in (126, 127)
            ):
                failure = "verifier executable is unavailable"
                infrastructure = True
            if failure is not None:
                completed.append(
                    CheckResult(check.check_id, _kind(check), "failed", container.exit_code)
                )
                return _stop(
                    "abandoned" if infrastructure else "failed",
                    failure,
                    manifest,
                    completed,
                    script_exit_code=script_exit_code,
                )

            expected_stdout = (verifier_dir / check.expect.stdout).read_bytes()
            expected_stderr = (
                None
                if check.expect.stderr is None
                else (verifier_dir / check.expect.stderr).read_bytes()
            )
            passed = (
                container.exit_code == check.expect.exit_code
                and container.stdout == expected_stdout
                and (expected_stderr is None or container.stderr == expected_stderr)
            )
            completed.append(
                CheckResult(
                    check.check_id,
                    _kind(check),
                    "passed" if passed else "failed",
                    container.exit_code,
                )
            )
            if not passed:
                return _stop(
                    "failed",
                    "candidate output did not match the verifier",
                    manifest,
                    completed,
                    script_exit_code=script_exit_code,
                )
            if isinstance(check, InvocationCheck):
                try:
                    _inspect_result_tree(workspace, result_tree, tree_limits)
                except TreeError:
                    completed[-1] = CheckResult(
                        check.check_id, "invocation", "failed", container.exit_code
                    )
                    return _stop(
                        "failed",
                        "candidate result tree became invalid",
                        manifest,
                        completed,
                        script_exit_code=script_exit_code,
                    )
        except (OSError, SupervisorError, ValueError):
            return _stop(
                "abandoned",
                "validator could not run the verifier",
                manifest,
                completed,
                script_exit_code=script_exit_code,
            )

    return EvaluationResult(
        status="passed",
        reason="",
        checks=tuple(completed),
        script_exit_code=script_exit_code,
    )


def _run_setup(
    root: Path,
    work_base: str,
    manifest: VerifierManifest,
    supervisor_policy: SupervisorPolicy,
    docker_binary: str,
    run_prefix: str,
) -> EvaluationResult | None:
    if manifest.setup is not None:
        try:
            setup = manifest.setup
            result = run_container(
                _request(
                    name=f"{run_prefix}-setup",
                    argv=setup.argv,
                    cwd=work_base,
                    mounts=(Mount(root, "/work", False),),
                    stdin=b"",
                    timeout_s=setup.timeout_s,
                    stdout_bytes=setup.max_output_bytes,
                    stderr_bytes=setup.max_output_bytes,
                    trusted=False,
                ),
                supervisor_policy,
                docker_binary,
            )
            failure = _container_failure(result)
            if failure is None and result.exit_code != 0:
                failure = "task setup failed"
            if failure is not None:
                return _stop(
                    "failed", failure, manifest, [], script_exit_code=None
                )
        except (OSError, SupervisorError, ValueError):
            return _stop(
                "abandoned",
                "validator could not run task setup",
                manifest,
                [],
                script_exit_code=None,
            )
    return None


def evaluate_repository(
    workspace: str | os.PathLike[str],
    patch: bytes,
    identity: RepositoryTaskIdentity,
    manifest: VerifierManifest,
    verifier_dir: str | os.PathLike[str],
    supervisor_policy: SupervisorPolicy,
    tree_limits: TreeLimits,
    patch_limits: PatchLimits,
    docker_binary: str,
    run_prefix: str,
) -> EvaluationResult:
    root = Path(workspace)
    verifier = Path(verifier_dir)
    if type(identity) is not RepositoryTaskIdentity or manifest.task_type != "repository_patch_v1":
        raise ValueError("repository task contract does not match")
    if _task_directory(root, identity.working_directory) is None:
        return _stop(
            "abandoned",
            "repository working directory is unavailable",
            manifest,
            [],
            script_exit_code=None,
        )
    try:
        applied = apply_patch_in_container(
            root,
            patch,
            patch_limits,
            supervisor_policy=supervisor_policy,
            docker_binary=docker_binary,
            run_prefix=run_prefix,
        )
    except PatchToolError:
        return EvaluationResult("abandoned", "validator could not apply the patch", (), None)
    if applied.status == "rejected":
        return EvaluationResult("rejected", "patch was rejected", (), None)

    work_base = _work_cwd("/work", identity.working_directory)
    host_work_base = _task_directory(root, identity.working_directory)
    if host_work_base is None:
        return _stop(
            "failed",
            "patch removed the repository working directory",
            manifest,
            [],
            script_exit_code=None,
        )
    setup_failure = _run_setup(
        root, work_base, manifest, supervisor_policy, docker_binary, run_prefix
    )
    if setup_failure is not None:
        return setup_failure
    return _run_checks(
        workspace=root,
        host_work_base=host_work_base,
        result_tree=root,
        work_base=work_base,
        manifest=manifest,
        verifier_dir=verifier,
        supervisor_policy=supervisor_policy,
        tree_limits=tree_limits,
        docker_binary=docker_binary,
        run_prefix=run_prefix,
        script_exit_code=None,
    )


def evaluate_terminal(
    environment: str | os.PathLike[str],
    script: bytes,
    identity: TerminalScriptTaskIdentity,
    manifest: VerifierManifest,
    verifier_dir: str | os.PathLike[str],
    supervisor_policy: SupervisorPolicy,
    tree_limits: TreeLimits,
    script_limits: ScriptLimits,
    docker_binary: str,
    run_prefix: str,
    script_timeout_s: int = SCRIPT_TIMEOUT_S,
    script_max_output_bytes: int = SCRIPT_OUTPUT_BYTES,
) -> EvaluationResult:
    if type(identity) is not TerminalScriptTaskIdentity or manifest.task_type != "terminal_script_v1":
        raise ValueError("terminal task contract does not match")
    checked = validate_script(script, script_limits)
    if checked.status == "rejected":
        return EvaluationResult("rejected", "script was rejected", (), None)
    root = Path(environment)
    verifier = Path(verifier_dir)
    result_tree = _task_directory(root, identity.result_tree_path)
    if result_tree is None:
        return _stop(
            "abandoned",
            "terminal result tree is unavailable",
            manifest,
            [],
            script_exit_code=None,
        )

    setup_failure = _run_setup(
        root, _work_cwd("/work", identity.result_tree_path), manifest,
        supervisor_policy, docker_binary, run_prefix,
    )
    if setup_failure is not None:
        return setup_failure
    if _task_directory(root, identity.result_tree_path) is None:
        return _stop(
            "failed", "setup removed the terminal result directory", manifest, [],
            script_exit_code=None,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="hone-v3-submission-") as temporary:
            submission = Path(temporary)
            script_path = submission / "script.sh"
            script_path.write_bytes(script)
            script_path.chmod(0o444)
            result = run_container(
                _request(
                    name=f"{run_prefix}-script",
                    argv=(
                        "/usr/bin/bash",
                        "--noprofile",
                        "--norc",
                        "/submission/script.sh",
                    ),
                    cwd=_work_cwd("/work", identity.result_tree_path),
                    mounts=(
                        Mount(root, "/work", False),
                        Mount(submission, "/submission", True),
                    ),
                    stdin=b"",
                    timeout_s=script_timeout_s,
                    stdout_bytes=script_max_output_bytes,
                    stderr_bytes=script_max_output_bytes,
                    trusted=False,
                ),
                supervisor_policy,
                docker_binary,
            )
    except (OSError, SupervisorError, ValueError):
        return _stop(
            "abandoned",
            "validator could not run the script",
            manifest,
            [],
            script_exit_code=None,
        )

    failure = _container_failure(result)
    if failure is not None:
        return _stop(
            "failed",
            failure,
            manifest,
            [],
            script_exit_code=result.exit_code,
        )
    return _run_checks(
        workspace=root,
        host_work_base=root,
        result_tree=result_tree,
        work_base="/work",
        manifest=manifest,
        verifier_dir=verifier,
        supervisor_policy=supervisor_policy,
        tree_limits=tree_limits,
        docker_binary=docker_binary,
        run_prefix=run_prefix,
        script_exit_code=result.exit_code,
    )
