"""Bounded public display of the first failed check, for optional miner feedback."""

from __future__ import annotations

import json
import posixpath

from .manifest import InvocationCheck
from .supervisor import ContainerRequest

FAILED_CHECK_MAX_BYTES = 2_048

_WORK = "/work"
_INTERPRETERS = ("/usr/bin/python3", "/usr/local/bin/python3")
_PATH_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-./"
)


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=True, separators=(",", ":"))


def escaped_bytes(text: str) -> int:
    """Size of the display as a JSON string, including its quotes and escapes."""
    return len(_quote(text).encode("ascii"))


def is_bounded_display(value: object) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= FAILED_CHECK_MAX_BYTES
        and escaped_bytes(value) <= FAILED_CHECK_MAX_BYTES
    )


def _path_characters(value: str) -> bool:
    return bool(value) and not set(value) - _PATH_CHARACTERS


def _no_parent(value: str) -> bool:
    return ".." not in value.split("/")


def _eligible_cwd(cwd: object) -> bool:
    return (
        type(cwd) is str
        and _path_characters(cwd)
        and _no_parent(cwd)
        and posixpath.normpath(cwd) == cwd
        and (cwd == _WORK or cwd.startswith(_WORK + "/"))
    )


def _eligible_script(script: object, cwd: str) -> bool:
    if (
        type(script) is not str
        or len(script) > FAILED_CHECK_MAX_BYTES
        or not _path_characters(script)
        or not _no_parent(script)
        or script.startswith("-")
        or not script.endswith(".py")
    ):
        return False
    resolved = posixpath.normpath(
        script if script.startswith("/") else posixpath.join(cwd, script)
    )
    return resolved.startswith(_WORK + "/") and len(resolved) > len(_WORK) + 1


def render_failed_check(
    *,
    check: InvocationCheck,
    request: ContainerRequest,
    expected_stdout: bytes,
    expected_stderr: bytes | None,
) -> str | None:
    """Render what the failing check required, or None when it cannot be shown.

    Pure and deterministic: it shows only the manifest's command, the working
    directory the validator used, and the expectation the candidate had to meet.
    It never sees or reveals a candidate's own output.
    """
    if type(check) is not InvocationCheck or type(request) is not ContainerRequest:
        return None
    if request.trusted is not False or request.argv != check.argv:
        return None
    if len(check.argv) != 2 or check.argv[0] not in _INTERPRETERS:
        return None
    if not _eligible_cwd(request.cwd) or not _eligible_script(check.argv[1], request.cwd):
        return None
    if type(expected_stdout) is not bytes or type(check.expect.exit_code) is not int:
        return None
    if (expected_stderr is None) is not (check.expect.stderr is None):
        return None
    if expected_stderr is not None and type(expected_stderr) is not bytes:
        return None
    stdin = request.stdin
    if type(stdin) is not bytes or (check.stdin is None and stdin != b""):
        return None
    if any(
        len(raw) > FAILED_CHECK_MAX_BYTES
        for raw in (stdin, expected_stdout, expected_stderr or b"")
    ):
        return None
    try:
        stdin_text = stdin.decode("utf-8")
        stdout_text = expected_stdout.decode("utf-8")
        stderr_text = None if expected_stderr is None else expected_stderr.decode("utf-8")
    except UnicodeDecodeError:
        return None

    display = "\n".join(
        (
            "Command (argv): "
            + json.dumps(list(check.argv), ensure_ascii=True, separators=(",", ":")),
            f"Working directory: {_quote(request.cwd)}",
            f"Stdin: {_quote(stdin_text)}",
            f"Required exit code: {check.expect.exit_code}",
            f"Required stdout: {_quote(stdout_text)}",
            "Required stderr: "
            + ("not checked" if stderr_text is None else _quote(stderr_text)),
        )
    )
    return display if escaped_bytes(display) <= FAILED_CHECK_MAX_BYTES else None
