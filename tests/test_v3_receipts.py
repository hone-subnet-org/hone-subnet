"""Byte fidelity and bounded serialization of selected-check receipts."""

from __future__ import annotations

import base64
import importlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from rlvr.v3.manifest import Expectation, InspectionCheck, InvocationCheck
from rlvr.v3.supervisor import ContainerRequest, ContainerResult


def receipt_module():
    return importlib.import_module("rlvr.v3.receipts")


def receipt_inputs(
    *,
    kind="invocation",
    expected_stdout=b"expected\n",
    stdout=b"actual\n",
    expected_stderr=None,
    stderr=b"",
    stdin=b"input\n",
    exit_code=0,
    limit=None,
    flags=None,
    argv=None,
    cwd=None,
):
    argv = argv or ("/usr/bin/python3", "-I", "/work/main.py")
    common = {
        "check_id": "selected",
        "argv": argv,
        "timeout_s": 10,
        "max_stdout_bytes": 8192,
        "max_stderr_bytes": 8192,
        "expect": Expectation(
            0, "gold/out", None if expected_stderr is None else "gold/err"
        ),
    }
    if kind == "invocation":
        check = InvocationCheck(cwd="sub", stdin="inputs/in", **common)
    else:
        check = InspectionCheck(**common)
    request = ContainerRequest(
        name="receipt-check",
        argv=argv,
        cwd=cwd or ("/result" if kind == "inspection" else "/work/project/sub"),
        mounts=(),
        stdin=stdin if kind == "invocation" else b"",
        timeout_s=10,
        max_stdout_bytes=8192,
        max_stderr_bytes=8192,
        trusted=kind == "inspection",
    )
    result = ContainerResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        **{
            "timed_out": False,
            "oom_killed": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            **(flags or {}),
        },
    )
    return {
        "check": check,
        "check_index": 1,
        "checks_total": 3,
        "request": request,
        "container": result,
        "limit": limit,
        "expected_stdout": expected_stdout,
        "expected_stderr": expected_stderr,
    }


def make_receipt(**kwargs):
    receipt = receipt_module().build_check_receipt(**receipt_inputs(**kwargs))
    assert receipt is not None
    return receipt


def decoded(excerpt):
    return base64.b64decode(excerpt["data_b64"], validate=True)


@pytest.mark.parametrize(
    "expected,actual,index",
    [
        (b"", b"", None),
        (b"same", b"same", None),
        (b"", b"x", 0),
        (b"x", b"", 0),
        (b"abc", b"abcd", 3),
        (b"abcd", b"abc", 3),
        (b"a\n", b"a\r\n", 1),
        (b"a\n", b"a \n", 1),
        (b"a\n", b"a", 1),
        (b"\xff", b"\xfe", 0),
        (b"a\x00b", b"a\x00c", 2),
    ],
)
def test_first_difference_is_a_byte_index(expected, actual, index):
    assert receipt_module().first_difference(expected, actual) == index


@pytest.mark.parametrize(
    "data,offset,captured,overflow,truncated,total",
    [
        (b"", 0, 0, False, False, 0),
        (b"\x00\xff\r\n", 0, 4, False, False, 4),
        (b"bc", 1, 3, False, True, 3),
        (b"ab", 0, 3, False, True, 3),
        (b"", 3, 3, False, True, 3),
        (b"ab", 0, 2, True, True, None),
        (b"", 0, 0, True, True, None),
    ],
)
def test_excerpt_distinguishes_omitted_window_from_incomplete_capture(
    data, offset, captured, overflow, truncated, total
):
    excerpt = receipt_module().Excerpt(offset, data, captured, overflow)
    record = excerpt.to_record()
    assert record == {
        "offset": offset,
        "data_b64": base64.b64encode(data).decode("ascii"),
        "captured_bytes": captured,
        "total_bytes": total,
        "capture_truncated": overflow,
        "truncated": truncated,
    }
    assert decoded(record) == data
    assert excerpt.total_bytes == total and excerpt.truncated == truncated
    with pytest.raises(FrozenInstanceError):
        excerpt.data = b"changed"


def test_serializer_counts_ascii_json_escaping_and_has_no_newline():
    value = {"text": "é\n\x00😀", "last": 1}
    encoded = receipt_module().serialize(value)
    assert encoded == json.dumps(
        value, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    assert encoded.isascii() and not encoded.endswith(b"\n")
    assert json.loads(encoded) == value


@pytest.mark.parametrize(
    "kind,source", [("invocation", "candidate"), ("inspection", "checker")]
)
def test_selected_check_metadata_and_bytes_are_preserved(kind, source):
    receipt = make_receipt(
        kind=kind,
        expected_stdout=b"\x00\xff\r\n",
        stdout=b"\x00\xff\n",
        expected_stderr=b"wanted",
        stderr=b"got",
        stdin=b"\xff\x00\r\n",
        exit_code=2,
    )
    record = receipt.to_record()
    assert list(record) == [
        "stage",
        "check_index",
        "checks_total",
        "check_id",
        "kind",
        "output_source",
        "argv",
        "argv_truncated",
        "cwd",
        "cwd_truncated",
        "limit",
        "mismatched",
        "exit_code",
        "stdin",
        "stdout",
        "stderr",
    ]
    assert (
        record["stage"],
        record["check_index"],
        record["checks_total"],
        record["check_id"],
    ) == ("check", 1, 3, "selected")
    assert (record["kind"], record["output_source"]) == (kind, source)
    assert record["mismatched"] == ["exit_code", "stdout", "stderr"]
    assert record["exit_code"] == {"expected": 0, "actual": 2}
    assert record["limit"] is None
    assert decoded(record["stdout"]["expected"]) == b"\x00\xff\r\n"
    assert decoded(record["stdout"]["actual"]) == b"\x00\xff\n"
    assert record["stdout"]["first_difference"] == 2
    assert record["cwd"] == ("/result" if kind == "inspection" else "/work/project/sub")
    if kind == "invocation":
        assert decoded(record["stdin"]) == b"\xff\x00\r\n"
    else:
        assert record["stdin"] is None
    assert not record["argv_truncated"] and not record["cwd_truncated"]
    with pytest.raises(FrozenInstanceError):
        receipt.check_id = "changed"


@pytest.mark.parametrize(
    "expected,actual", [(b"", b"x"), (b"x", b""), (b"abc", b"abcd"), (b"abcd", b"abc")]
)
def test_prefix_failure_exposes_the_extra_byte_and_shorter_eof(expected, actual):
    stream = make_receipt(expected_stdout=expected, stdout=actual).to_record()["stdout"]
    assert stream["first_difference"] == min(len(expected), len(actual))
    assert decoded(stream["expected"]) == expected
    assert decoded(stream["actual"]) == actual
    assert stream["expected"]["total_bytes"] == len(expected)
    assert stream["actual"]["total_bytes"] == len(actual)


@pytest.mark.parametrize(
    "expected_stderr,exit_code,mismatched",
    [
        (b"", 0, ["stderr"]),
        (None, 1, ["exit_code"]),
        (b"error", 1, ["exit_code"]),
    ],
)
def test_empty_stderr_expectation_is_checked_but_none_is_unchecked(
    expected_stderr, exit_code, mismatched
):
    record = make_receipt(
        expected_stdout=b"same",
        stdout=b"same",
        expected_stderr=expected_stderr,
        stderr=b"error",
        exit_code=exit_code,
    ).to_record()
    assert record["mismatched"] == mismatched
    assert record["stdout"]["first_difference"] is None
    if expected_stderr is None:
        assert record["stderr"]["expected"] is None
    else:
        assert decoded(record["stderr"]["expected"]) == expected_stderr
        assert record["stderr"]["first_difference"] == (
            0 if expected_stderr == b"" else None
        )


def test_exit_only_failure_shows_equal_stdout_and_unchecked_stderr_tail():
    stdout = b"head" + b"s" * 900
    stderr = b"discard" * 100 + b"tail"
    receipt = make_receipt(
        expected_stdout=stdout, stdout=stdout, stderr=stderr, exit_code=1, stdin=b""
    )
    record = receipt.to_record()
    assert record["mismatched"] == ["exit_code"]
    assert record["stdout"]["first_difference"] is None
    assert record["stdout"]["actual"]["offset"] == 0
    assert decoded(record["stdout"]["actual"]).startswith(b"head")
    stream = record["stderr"]
    assert stream["expected"] is None and stream["first_difference"] is None
    assert decoded(stream["actual"]) == stderr[stream["actual"]["offset"] :]
    assert decoded(stream["actual"]).endswith(b"tail")


@pytest.mark.parametrize(
    "limit,flags",
    [
        ("timeout", {"timed_out": True}),
        ("memory_limit", {"oom_killed": True}),
        ("output_limit", {"stdout_overflow": True}),
        (
            "timeout",
            {"timed_out": True, "stdout_overflow": True, "stderr_overflow": True},
        ),
    ],
)
@pytest.mark.parametrize("kind", ["invocation", "inspection"])
def test_limit_receipts_describe_observed_capture_without_comparing(limit, flags, kind):
    stdout, stderr = b"prefix" * 200, b"errors" * 180
    record = make_receipt(
        kind=kind,
        limit=limit,
        flags=flags,
        expected_stdout=None,
        stdout=stdout,
        stderr=stderr,
        exit_code=137,
    ).to_record()
    assert record["limit"] == limit and record["mismatched"] == []
    assert record["exit_code"] == {"expected": None, "actual": 137}
    for name, captured in (("stdout", stdout), ("stderr", stderr)):
        stream = record[name]
        assert stream["expected"] is None and stream["first_difference"] is None
        excerpt = stream["actual"]
        overflow = flags.get(f"{name}_overflow", False)
        assert excerpt["capture_truncated"] is overflow
        assert excerpt["captured_bytes"] == len(captured)
        assert excerpt["total_bytes"] == (None if overflow else len(captured))
        assert decoded(excerpt) == captured[excerpt["offset"] :]
        assert len(decoded(excerpt)) <= 512
        if overflow:
            assert excerpt["truncated"]


def test_utf8_argument_and_cwd_caps_mark_any_loss():
    argv = ("/usr/bin/python3", "é" * 100, "😀" * 80, "\x01" * 120, "z" * 120, "last")
    original_cwd = "/work/" + "é" * 200
    record = make_receipt(argv=argv, cwd=original_cwd).to_record()
    assert record["argv_truncated"] and record["cwd_truncated"]
    assert record["argv"][0] == argv[0]
    assert all(len(arg.encode("utf8")) <= 128 for arg in record["argv"])
    assert sum(len(arg.encode("utf8")) for arg in record["argv"]) <= 512
    assert all(
        original.startswith(shown)
        for original, shown in zip(argv, record["argv"], strict=False)
    )
    assert original_cwd.startswith(record["cwd"])
    assert len(record["cwd"].encode("utf8")) <= 256
    encoded = receipt_module().serialize(record)
    assert len(encoded) <= 3072 and json.loads(encoded) == record


def test_argument_budget_counts_only_the_utf8_prefix_that_is_retained():
    argv = (
        "/usr/bin/true",
        "x" * 127 + "é",
        "y" * 127 + "é",
        "z" * 127 + "é",
        "Q" * 120,
    )
    receipt = make_receipt(argv=argv)
    assert receipt.argv == ("/usr/bin/true", "x" * 127, "y" * 127, "z" * 127, "Q" * 118)
    assert sum(len(argument.encode("utf-8")) for argument in receipt.argv) == 512
    assert receipt.argv_truncated


def test_fitting_reduces_stderr_before_input_and_preserves_other_fields():
    module = receipt_module()
    inputs = receipt_inputs(
        expected_stdout=b"a", stdout=b"b", stderr=b"e" * 512, stdin=b"i" * 512
    )
    full = module.build_check_receipt(**inputs)
    assert full is not None
    assert len(full.stderr.actual.data) == len(full.stdin.data) == 512
    fitted = module.build_check_receipt(
        **inputs, max_bytes=len(module.serialize(full.to_record())) - 1
    )
    assert fitted is not None
    assert len(fitted.stderr.actual.data) == 256
    assert fitted.stdin == full.stdin and fitted.stdout == full.stdout
    assert fitted.argv == full.argv and fitted.mismatched == full.mismatched


def test_shrunken_windows_still_show_a_distant_mismatch():
    module = receipt_module()
    expected = b"a" * 300 + b"X" + b"a" * 1700
    actual = b"a" * 300 + b"Y" + b"a" * 1700
    inputs = receipt_inputs(expected_stdout=expected, stdout=actual, stdin=b"i" * 1000)
    small_windows = []
    for cap in (3072, 2500, 2200, 1900, 1600, 1400, 1200, 1000, 900):
        receipt = module.build_check_receipt(**inputs, max_bytes=cap)
        if receipt is None:
            continue
        assert len(module.serialize(receipt.to_record())) <= cap
        assert receipt.stdout.first_difference == 300
        for excerpt, original in (
            (receipt.stdout.expected, expected),
            (receipt.stdout.actual, actual),
        ):
            assert (
                excerpt.data
                == original[excerpt.offset : excerpt.offset + len(excerpt.data)]
            )
            if excerpt.data:
                assert excerpt.offset <= 300 < excerpt.offset + len(excerpt.data)
                assert excerpt.data[300 - excerpt.offset] == original[300]
                if len(excerpt.data) < 128:
                    small_windows.append(excerpt)
    assert small_windows, (
        "exercise the budgets where a fixed 128-byte lookbehind loses the mismatch"
    )


def test_zero_budget_keeps_mismatch_metadata_and_too_small_cap_omits_receipt():
    module = receipt_module()
    inputs = receipt_inputs(expected_stdout=b"a" * 500 + b"X", stdout=b"a" * 500 + b"Y")
    full = module.build_check_receipt(**inputs)
    assert full is not None

    # The specified final fitting candidate has metadata, but no data windows.
    def empty_comparison(excerpt):
        return replace(excerpt, offset=500, data=b"")

    minimal = replace(
        full,
        stdin=replace(full.stdin, data=b""),
        stdout=replace(
            full.stdout,
            expected=empty_comparison(full.stdout.expected),
            actual=empty_comparison(full.stdout.actual),
        ),
        stderr=replace(full.stderr, actual=replace(full.stderr.actual, data=b"")),
    )
    cap = len(module.serialize(minimal.to_record()))
    fitted = module.build_check_receipt(**inputs, max_bytes=cap)
    assert fitted is not None
    assert fitted.stdout.expected.data == fitted.stdout.actual.data == b""
    assert fitted.stdout.first_difference == 500
    assert fitted.stdout.actual.captured_bytes == 501
    assert fitted.stdout.actual.truncated and fitted.mismatched == ("stdout",)
    assert module.build_check_receipt(**inputs, max_bytes=1) is None
    assert module.build_check_receipt(**inputs, max_bytes=cap) == fitted


@pytest.mark.parametrize(
    "changes",
    [
        {"check_index": 0},
        {"check_index": 4, "checks_total": 3},
        {"expected_stdout": None},
        {"limit": "output_limit", "expected_stdout": None, "expected_stderr": None},
        {"limit": "timeout"},
    ],
)
def test_invalid_receipt_inputs_raise_value_error(changes):
    with pytest.raises(ValueError):
        receipt_module().build_check_receipt(**(receipt_inputs() | changes))


def test_passing_comparison_cannot_become_a_failure_receipt():
    with pytest.raises(ValueError):
        receipt_module().build_check_receipt(
            **receipt_inputs(expected_stdout=b"same", stdout=b"same")
        )
