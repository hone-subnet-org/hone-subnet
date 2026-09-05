from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .tree import TreeError, inspect_tree
from .supervisor import ContainerRequest, Mount, SupervisorError, SupervisorPolicy, run_container

_MODE_LINE = re.compile(
    rb"^(?:old|new|new file|deleted file) mode ([0-9]+)$"
)
_ALLOWED_MODES = {b"100644", b"100755"}
_BINARY_MARKERS = (b"GIT binary patch", b"Binary files ", b"literal ", b"delta ")
_COPY_RENAME_MARKERS = (
    b"copy from ",
    b"copy to ",
    b"copy old ",
    b"copy new ",
    b"rename from ",
    b"rename to ",
    b"rename old ",
    b"rename new ",
    b"similarity index ",
    b"dissimilarity index ",
)
_REASON_LIMIT = 200


class PatchToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchLimits:
    max_patch_bytes: int
    git_timeout_s: int

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if type(value) is not int or value <= 0:
                raise ValueError("patch limits must be positive integers")


@dataclass(frozen=True)
class PatchResult:
    status: Literal["applied", "rejected"]
    reason: str

    def __post_init__(self) -> None:
        if self.status == "applied" and self.reason:
            raise ValueError("an applied patch cannot have a rejection reason")
        if self.status == "rejected" and not 0 < len(self.reason) <= _REASON_LIMIT:
            raise ValueError("a rejected patch requires a bounded reason")


def _rejected(reason: str) -> PatchResult:
    first_line = next((line.strip() for line in reason.splitlines() if line.strip()), "")
    return PatchResult("rejected", (first_line or "patch rejected")[:_REASON_LIMIT])


def _static_rejection(patch: bytes, limits: PatchLimits) -> PatchResult | None:
    if len(patch) > limits.max_patch_bytes:
        return _rejected("patch exceeds the policy byte limit")
    if b"\x00" in patch:
        return _rejected("patch contains a NUL byte")
    try:
        patch.decode("utf-8")
    except UnicodeDecodeError:
        return _rejected("patch is not valid UTF-8")
    for line in patch.splitlines():
        if any(line.startswith(marker) for marker in _BINARY_MARKERS):
            return _rejected("binary patches are not supported")
        if any(line.startswith(marker) for marker in _COPY_RENAME_MARKERS):
            return _rejected("copy and rename patches are not supported")
        match = _MODE_LINE.fullmatch(line)
        if match and match.group(1) not in _ALLOWED_MODES:
            return _rejected("patch contains an unsupported file mode")
    return None


def _walk_tree(workspace: Path, *, baseline: bool) -> str | None:
    try:
        inspect_tree(workspace, limits=None, normalize_modes=not baseline)
    except TreeError as exc:
        if baseline:
            raise PatchToolError("workspace could not be inspected") from None
        return f"patched workspace: {exc}"
    return None


def _run_git(
    command: list[str],
    *,
    workspace: Path,
    environment: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PatchToolError("git execution failed") from exc


def apply_patch(
    workspace: str | os.PathLike[str],
    patch: bytes,
    limits: PatchLimits,
    *,
    git_binary: str,
    expected_git_version: str,
) -> PatchResult:
    if not isinstance(patch, bytes):
        raise TypeError("patch must be bytes")
    rejection = _static_rejection(patch, limits)
    if rejection is not None:
        return rejection

    root = Path(workspace)
    _walk_tree(root, baseline=True)
    if not patch:
        return PatchResult("applied", "")
    executable = shutil.which(git_binary)
    if executable is None:
        raise PatchToolError("git executable is unavailable")
    executable = str(Path(executable).resolve())

    with tempfile.TemporaryDirectory(prefix="hone-v3-patch-") as scratch_name:
        scratch = Path(scratch_name)
        home = scratch / "home"
        home.mkdir()
        patch_path = scratch / "submission.diff"
        patch_path.write_bytes(patch)
        environment = {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CEILING_DIRECTORIES": str(root.resolve().parent),
            "HOME": str(home),
            "LC_ALL": "C",
            "PATH": str(Path(executable).parent),
        }

        version = _run_git(
            [executable, "--version"],
            workspace=root,
            environment=environment,
            timeout=limits.git_timeout_s,
        )
        if version.returncode != 0:
            raise PatchToolError("git version check failed")
        words = version.stdout.decode("ascii", "replace").strip().split()
        if len(words) < 3 or words[0:2] != ["git", "version"]:
            raise PatchToolError("git returned an invalid version")
        if words[2] != expected_git_version:
            raise PatchToolError("git version does not match release policy")

        check_result = _run_git(
            [executable, "apply", "--check", "-p1", str(patch_path)],
            workspace=root,
            environment=environment,
            timeout=limits.git_timeout_s,
        )
        if check_result.returncode != 0:
            return _rejected(check_result.stderr.decode("utf-8", "replace"))

        apply_result = _run_git(
            [executable, "apply", "-p1", str(patch_path)],
            workspace=root,
            environment=environment,
            timeout=limits.git_timeout_s,
        )
        if apply_result.returncode != 0:
            raise PatchToolError("git apply failed after a successful check")

    post_error = _walk_tree(root, baseline=False)
    if post_error is not None:
        return _rejected(post_error)
    return PatchResult("applied", "")


def apply_patch_in_container(
    workspace: str | os.PathLike[str],
    patch: bytes,
    limits: PatchLimits,
    *,
    supervisor_policy: SupervisorPolicy,
    docker_binary: str,
    run_prefix: str,
) -> PatchResult:
    rejection = _static_rejection(patch, limits)
    if rejection is not None:
        return rejection
    root = Path(workspace).resolve()
    _walk_tree(root, baseline=True)
    if not patch:
        return PatchResult("applied", "")

    with tempfile.TemporaryDirectory(prefix="hone-v3-patch-") as scratch_name:
        patch_path = Path(scratch_name) / "submission.diff"
        patch_path.write_bytes(patch)
        patch_path.chmod(0o444)

        def invoke(name: str, argv: tuple[str, ...]):
            try:
                return run_container(
                    ContainerRequest(
                        name=f"{run_prefix}-patch-{name}",
                        argv=argv,
                        cwd="/workspace",
                        mounts=(
                            Mount(root, "/workspace", False),
                            Mount(patch_path, "/submission.diff", True),
                        ),
                        stdin=b"",
                        timeout_s=limits.git_timeout_s,
                        max_stdout_bytes=65_536,
                        max_stderr_bytes=65_536,
                        trusted=False,
                    ),
                    supervisor_policy,
                    docker_binary,
                )
            except SupervisorError as exc:
                raise PatchToolError("git container failed") from exc

        checked = invoke(
            "check", ("/usr/bin/git", "apply", "--check", "-p1", "/submission.diff")
        )
        if checked.timed_out or checked.oom_killed or checked.stdout_overflow or checked.stderr_overflow:
            raise PatchToolError("git check exceeded its limits")
        if checked.exit_code in (126, 127):
            raise PatchToolError("git is unavailable in the sandbox")
        if checked.exit_code != 0:
            return _rejected(checked.stderr.decode("utf-8", "replace"))

        applied = invoke(
            "apply", ("/usr/bin/git", "apply", "-p1", "/submission.diff")
        )
        if (
            applied.exit_code != 0
            or applied.timed_out
            or applied.oom_killed
            or applied.stdout_overflow
            or applied.stderr_overflow
        ):
            return _rejected("git apply failed")

    post_error = _walk_tree(root, baseline=False)
    return _rejected(post_error) if post_error is not None else PatchResult("applied", "")
