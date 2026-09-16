"""Bounded, byte-exact receipts for the first failed verifier check."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Literal

from .manifest import InspectionCheck, InvocationCheck
from .supervisor import ContainerRequest, ContainerResult

RECEIPT_MAX_BYTES = 3_072
WINDOW_BYTES = 512
ARG_BYTES = 128
ARGV_BYTES = 512
CWD_BYTES = 256

Limit = Literal["timeout", "memory_limit", "output_limit"]

_LIMITS = ("timeout", "memory_limit", "output_limit")
_LOOKBEHIND = 128
_COMPARE_BLOCK = 4_096
# Budget reductions applied in order until the serialized receipt fits.
_FITTING_STEPS = (
    ("stderr", 256), ("stderr", 128), ("stderr", 64), ("stderr", 32),
    ("stdin", 256), ("stdin", 128), ("stdin", 64), ("stdin", 32), ("stdin", 0),
    ("stdout", 256), ("stdout", 128), ("stdout", 64), ("stdout", 32),
    ("argv", 256), ("argv", 128), ("argv", 64),
    ("stderr", 0), ("stdout", 0),
)


def serialize(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def first_difference(expected: bytes, actual: bytes) -> int | None:
    if expected == actual:
        return None
    shorter = min(len(expected), len(actual))
    for start in range(0, shorter, _COMPARE_BLOCK):
        end = min(start + _COMPARE_BLOCK, shorter)
        if expected[start:end] != actual[start:end]:
            return next(
                index for index in range(start, end) if expected[index] != actual[index]
            )
    return shorter


@dataclass(frozen=True)
class Excerpt:
    offset: int
    data: bytes
    captured_bytes: int
    capture_truncated: bool

    @property
    def total_bytes(self) -> int | None:
        return None if self.capture_truncated else self.captured_bytes

    @property
    def truncated(self) -> bool:
        return (
            self.capture_truncated
            or self.offset > 0
            or self.offset + len(self.data) < self.captured_bytes
        )

    def to_record(self) -> dict:
        return {
            "offset": self.offset,
            "data_b64": base64.b64encode(self.data).decode("ascii"),
            "captured_bytes": self.captured_bytes,
            "total_bytes": self.total_bytes,
            "capture_truncated": self.capture_truncated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class StreamReceipt:
    first_difference: int | None
    expected: Excerpt | None
    actual: Excerpt

    def to_record(self) -> dict:
        return {
            "first_difference": self.first_difference,
            "expected": None if self.expected is None else self.expected.to_record(),
            "actual": self.actual.to_record(),
        }


@dataclass(frozen=True)
class CheckReceipt:
    check_index: int
    checks_total: int
    check_id: str
    kind: Literal["invocation", "inspection"]
    argv: tuple[str, ...]
    argv_truncated: bool
    cwd: str
    cwd_truncated: bool
    limit: Limit | None
    mismatched: tuple[str, ...]
    exit_expected: int | None
    exit_actual: int
    stdin: Excerpt | None
    stdout: StreamReceipt
    stderr: StreamReceipt

    @property
    def output_source(self) -> Literal["candidate", "checker"]:
        return "candidate" if self.kind == "invocation" else "checker"

    def to_record(self) -> dict:
        return {
            "stage": "check",
            "check_index": self.check_index,
            "checks_total": self.checks_total,
            "check_id": self.check_id,
            "kind": self.kind,
            "output_source": self.output_source,
            "argv": list(self.argv),
            "argv_truncated": self.argv_truncated,
            "cwd": self.cwd,
            "cwd_truncated": self.cwd_truncated,
            "limit": self.limit,
            "mismatched": list(self.mismatched),
            "exit_code": {"expected": self.exit_expected, "actual": self.exit_actual},
            "stdin": None if self.stdin is None else self.stdin.to_record(),
            "stdout": self.stdout.to_record(),
            "stderr": self.stderr.to_record(),
        }


def _utf8_prefix(text: str, limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", "ignore"), True


def _bounded_argv(argv: tuple[str, ...], budget: int) -> tuple[tuple[str, ...], bool]:
    shown: list[str] = []
    truncated = False
    remaining = budget
    for argument in argv:
        if remaining <= 0:
            return tuple(shown), True
        limit = min(ARG_BYTES, remaining)
        prefix, shortened = _utf8_prefix(argument, limit)
        shown.append(prefix)
        truncated = truncated or shortened
        remaining -= len(prefix.encode("utf-8"))
    return tuple(shown), truncated


def _head(stream: bytes, budget: int) -> Excerpt:
    return Excerpt(0, stream[:budget], len(stream), False)


def _tail(stream: bytes, budget: int, capture_truncated: bool) -> Excerpt:
    offset = max(0, len(stream) - budget)
    return Excerpt(offset, stream[offset:], len(stream), capture_truncated)


def _window(stream: bytes, start: int, budget: int, capture_truncated: bool) -> Excerpt:
    offset = min(start, len(stream))
    return Excerpt(offset, stream[offset : offset + budget], len(stream), capture_truncated)


def _window_start(difference: int | None, budget: int) -> int:
    if difference is None:
        return 0
    if budget == 0:
        return difference
    return max(0, difference - min(_LOOKBEHIND, (budget - 1) // 2))


def _stream(
    expected: bytes | None,
    actual: bytes,
    difference: int | None,
    budget: int,
    capture_truncated: bool,
) -> StreamReceipt:
    if expected is None:
        return StreamReceipt(None, None, _tail(actual, budget, capture_truncated))
    start = _window_start(difference, budget)
    return StreamReceipt(
        difference,
        _window(expected, start, budget, False),
        _window(actual, start, budget, capture_truncated),
    )


def build_check_receipt(
    *,
    check: InvocationCheck | InspectionCheck,
    check_index: int,
    checks_total: int,
    request: ContainerRequest,
    container: ContainerResult,
    limit: Limit | None,
    expected_stdout: bytes | None,
    expected_stderr: bytes | None,
    max_bytes: int = RECEIPT_MAX_BYTES,
) -> CheckReceipt | None:
    if isinstance(check, InvocationCheck):
        kind: Literal["invocation", "inspection"] = "invocation"
    elif isinstance(check, InspectionCheck):
        kind = "inspection"
    else:
        raise ValueError("a receipt requires a verifier check")  # noqa: TRY004 - uniform input validation
    if type(check_index) is not int or type(checks_total) is not int:
        raise ValueError("check position must be an integer")
    if not 1 <= check_index <= checks_total:
        raise ValueError("check position is outside the manifest")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("receipt size bound must be a non-negative integer")

    if limit is None:
        if type(expected_stdout) is not bytes:
            raise ValueError("a mismatch receipt requires expected stdout")
        if expected_stderr is not None and type(expected_stderr) is not bytes:
            raise ValueError("expected stderr must be bytes when checked")
        stdout_difference = first_difference(expected_stdout, container.stdout)
        stderr_difference = (
            None
            if expected_stderr is None
            else first_difference(expected_stderr, container.stderr)
        )
        exit_expected: int | None = check.expect.exit_code
        mismatched = tuple(
            name
            for name, differs in (
                ("exit_code", container.exit_code != exit_expected),
                ("stdout", stdout_difference is not None),
                ("stderr", stderr_difference is not None),
            )
            if differs
        )
        if not mismatched:
            raise ValueError("a passing comparison has no failure receipt")
    else:
        if limit not in _LIMITS:
            raise ValueError("receipt limit is not recognized")
        if expected_stdout is not None or expected_stderr is not None:
            raise ValueError("a limit receipt does not compare output")
        if limit == "output_limit" and not (
            container.stdout_overflow or container.stderr_overflow
        ):
            raise ValueError("an output limit requires an overflowed capture")
        stdout_difference = stderr_difference = None
        exit_expected = None
        mismatched = ()

    cwd, cwd_truncated = _utf8_prefix(request.cwd, CWD_BYTES)
    stdin = request.stdin if kind == "invocation" else None
    budgets = {
        "stderr": WINDOW_BYTES,
        "stdin": WINDOW_BYTES,
        "stdout": WINDOW_BYTES,
        "argv": ARGV_BYTES,
    }

    def assemble() -> CheckReceipt:
        argv, argv_truncated = _bounded_argv(request.argv, budgets["argv"])
        return CheckReceipt(
            check_index=check_index,
            checks_total=checks_total,
            check_id=check.check_id,
            kind=kind,
            argv=argv,
            argv_truncated=argv_truncated,
            cwd=cwd,
            cwd_truncated=cwd_truncated,
            limit=limit,
            mismatched=mismatched,
            exit_expected=exit_expected,
            exit_actual=container.exit_code,
            stdin=None if stdin is None else _head(stdin, budgets["stdin"]),
            stdout=_stream(
                expected_stdout,
                container.stdout,
                stdout_difference,
                budgets["stdout"],
                container.stdout_overflow,
            ),
            stderr=_stream(
                expected_stderr,
                container.stderr,
                stderr_difference,
                budgets["stderr"],
                container.stderr_overflow,
            ),
        )

    candidate = assemble()
    if len(serialize(candidate.to_record())) <= max_bytes:
        return candidate
    for name, budget in _FITTING_STEPS:
        budgets[name] = budget
        candidate = assemble()
        if len(serialize(candidate.to_record())) <= max_bytes:
            return candidate
    return None
