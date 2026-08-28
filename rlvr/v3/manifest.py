from __future__ import annotations

import os
import posixpath
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

from .canonical import parse_strict_json
from .identity import validate_relative_path

_MAX_OUTPUT = 8 << 20
_MAX_MANIFEST = 1 << 20
_CHECK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_TASK_TYPES = {"repository_patch_v1", "terminal_script_v1"}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Expectation:
    exit_code: int
    stdout: str
    stderr: str | None


@dataclass(frozen=True)
class SetupStep:
    argv: tuple[str, ...]
    timeout_s: int
    max_output_bytes: int


@dataclass(frozen=True)
class InvocationCheck:
    check_id: str
    argv: tuple[str, ...]
    cwd: str
    stdin: str | None
    timeout_s: int
    max_stdout_bytes: int
    max_stderr_bytes: int
    expect: Expectation
    kind: ClassVar[Literal["invocation"]] = "invocation"


@dataclass(frozen=True)
class InspectionCheck:
    check_id: str
    argv: tuple[str, ...]
    timeout_s: int
    max_stdout_bytes: int
    max_stderr_bytes: int
    expect: Expectation
    kind: ClassVar[Literal["inspection"]] = "inspection"


Check: TypeAlias = InvocationCheck | InspectionCheck


@dataclass(frozen=True)
class VerifierManifest:
    manifest_version: int
    task_type: Literal["repository_patch_v1", "terminal_script_v1"]
    setup: SetupStep | None
    checks: tuple[Check, ...]


def _exact_keys(value: object, required: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != required:
        raise ManifestError("manifest fields do not match the schema")
    return value


def _bounded_int(value: object, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ManifestError("manifest integer is outside its allowed range")
    return value


def _absolute_executable(value: str, *, inspection: bool) -> None:
    if (
        not value.startswith("/")
        or value.startswith("//")
        or value == "/"
        or posixpath.normpath(value) != value
        or "\x00" in value
    ):
        raise ManifestError("manifest executable path is invalid")
    if inspection and any(
        value == root or value.startswith(root + "/")
        for root in ("/result", "/verify")
    ):
        raise ManifestError("inspection executable is not trusted")


def _argv(value: object, *, inspection: bool = False) -> tuple[str, ...]:
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise ManifestError("manifest argv is invalid")
    result: list[str] = []
    for item in value:
        if (
            type(item) is not str
            or not 1 <= len(item.encode("utf-8")) <= 4_096
            or "\x00" in item
        ):
            raise ManifestError("manifest argument is invalid")
        result.append(item)
    _absolute_executable(result[0], inspection=inspection)
    return tuple(result)


def _relative(value: object) -> str:
    if type(value) is not str:
        raise ManifestError("manifest path is invalid")
    try:
        return validate_relative_path(value)
    except ValueError:
        raise ManifestError("manifest path is invalid") from None


def _rooted(value: object, root: str, *, nullable: bool) -> str | None:
    if value is None and nullable:
        return None
    path = _relative(value)
    if path == root or not path.startswith(root + "/"):
        raise ManifestError("manifest reference has the wrong root")
    return path


def _expectation(value: object) -> Expectation:
    item = _exact_keys(value, {"exit_code", "stdout", "stderr"})
    return Expectation(
        exit_code=_bounded_int(item["exit_code"], 0, 255),
        stdout=_rooted(item["stdout"], "gold", nullable=False),
        stderr=_rooted(item["stderr"], "gold", nullable=True),
    )


def _setup(value: object) -> SetupStep | None:
    if value is None:
        return None
    item = _exact_keys(value, {"argv", "timeout_s", "max_output_bytes"})
    return SetupStep(
        argv=_argv(item["argv"]),
        timeout_s=_bounded_int(item["timeout_s"], 1, 300),
        max_output_bytes=_bounded_int(item["max_output_bytes"], 1, _MAX_OUTPUT),
    )


def _check(value: object) -> Check:
    if type(value) is not dict:
        raise ManifestError("manifest check is invalid")
    kind = value.get("kind")
    common = {
        "check_id",
        "kind",
        "argv",
        "timeout_s",
        "max_stdout_bytes",
        "max_stderr_bytes",
        "expect",
    }
    if kind == "invocation":
        item = _exact_keys(value, common | {"cwd", "stdin"})
    elif kind == "inspection":
        item = _exact_keys(value, common)
    else:
        raise ManifestError("manifest check kind is invalid")

    check_id = item["check_id"]
    if type(check_id) is not str or _CHECK_ID.fullmatch(check_id) is None:
        raise ManifestError("manifest check ID is invalid")
    shared = dict(
        check_id=check_id,
        argv=_argv(item["argv"], inspection=kind == "inspection"),
        timeout_s=_bounded_int(item["timeout_s"], 1, 300),
        max_stdout_bytes=_bounded_int(item["max_stdout_bytes"], 1, _MAX_OUTPUT),
        max_stderr_bytes=_bounded_int(item["max_stderr_bytes"], 1, _MAX_OUTPUT),
        expect=_expectation(item["expect"]),
    )
    if kind == "invocation":
        return InvocationCheck(
            **shared,
            cwd=_relative(item["cwd"]),
            stdin=_rooted(item["stdin"], "inputs", nullable=True),
        )
    return InspectionCheck(**shared)


def parse_manifest(raw: bytes, *, task_type: str) -> VerifierManifest:
    try:
        value = parse_strict_json(raw)
        root = _exact_keys(
            value, {"manifest_version", "task_type", "setup", "checks"}
        )
        if task_type not in _TASK_TYPES or root["task_type"] != task_type:
            raise ManifestError("manifest task type does not match")
        version = _bounded_int(root["manifest_version"], 1, 1)
        raw_checks = root["checks"]
        if type(raw_checks) is not list or not 1 <= len(raw_checks) <= 64:
            raise ManifestError("manifest check count is invalid")
        checks = tuple(_check(item) for item in raw_checks)
        if len({check.check_id for check in checks}) != len(checks):
            raise ManifestError("manifest check IDs are not unique")
        return VerifierManifest(
            manifest_version=version,
            task_type=task_type,
            setup=_setup(root["setup"]),
            checks=checks,
        )
    except ManifestError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise ManifestError("manifest is invalid") from None


def _regular_file(root: Path, relative: str) -> Path:
    current = root
    for segment in relative.split("/"):
        current = current / segment
        try:
            info = current.lstat()
        except OSError:
            raise ManifestError("manifest reference is unavailable") from None
        if stat.S_ISLNK(info.st_mode):
            raise ManifestError("manifest reference is a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ManifestError("manifest reference is not a regular file")
    return current


def load_manifest(
    verifier_dir: str | os.PathLike[str], *, task_type: str
) -> VerifierManifest:
    root = Path(verifier_dir)
    try:
        root_info = root.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise ManifestError("verifier directory is invalid")
        allowed = {"manifest.json", "checks", "inputs", "gold"}
        entries = list(root.iterdir())
        if any(entry.name not in allowed for entry in entries):
            raise ManifestError("verifier directory has an unexpected entry")
        for name in ("checks", "inputs", "gold"):
            entry = root / name
            if entry.exists() or entry.is_symlink():
                info = entry.lstat()
                if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    raise ManifestError("verifier directory layout is invalid")
        manifest_path = _regular_file(root, "manifest.json")
        if manifest_path.stat().st_size > _MAX_MANIFEST:
            raise ManifestError("manifest file is too large")
        manifest = parse_manifest(manifest_path.read_bytes(), task_type=task_type)
        for check in manifest.checks:
            if isinstance(check, InvocationCheck) and check.stdin is not None:
                _regular_file(root, check.stdin)
            if isinstance(check, InspectionCheck):
                for argument in check.argv[1:]:
                    if argument.startswith("/verify/"):
                        relative = _relative("checks/" + argument[len("/verify/") :])
                        _regular_file(root, relative)
            stdout = _regular_file(root, check.expect.stdout)
            if stdout.stat().st_size > check.max_stdout_bytes:
                raise ManifestError("expected stdout exceeds its check limit")
            if check.expect.stderr is not None:
                stderr = _regular_file(root, check.expect.stderr)
                if stderr.stat().st_size > check.max_stderr_bytes:
                    raise ManifestError("expected stderr exceeds its check limit")
        return manifest
    except ManifestError:
        raise
    except OSError:
        raise ManifestError("verifier manifest could not be loaded") from None
