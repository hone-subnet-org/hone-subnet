"""V3 ``bash_script_v1`` byte-level validation.

Contract for module ``rlvr.v3.script``:

    ScriptLimits(max_script_bytes)            frozen, required, positive
    ScriptResult(status, reason)              "accepted" with reason exactly ""
                                              or "rejected" with a non-empty
                                              reason of at most 200 characters
    validate_script(script: bytes, limits) -> ScriptResult
        size at most max_script_bytes, no NUL byte, strict UTF-8. Zero bytes
        is accepted. No other content rules: line endings, shebang lines,
        and Unicode normalization are not inspected here.
"""

from __future__ import annotations

import pytest

NFD_TEXT = "é".encode()


def _mod():
    from rlvr.v3 import script

    return script


def validate(data: bytes, max_bytes: int = 1 << 20):
    m = _mod()
    return m.validate_script(data, m.ScriptLimits(max_script_bytes=max_bytes))


def assert_rejected(result):
    assert result.status == "rejected"
    assert 0 < len(result.reason) <= 200


def test_empty_script_is_accepted():
    result = validate(b"")
    assert result.status == "accepted"
    assert result.reason == ""


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"#!/bin/bash\nset -e\nmkdir -p out\n", id="shebang-lf"),
        pytest.param(b"echo hi\r\necho there\r\n", id="crlf"),
        pytest.param("echo 'café'\n".encode(), id="utf8-nfc"),
        pytest.param(b"echo 'caf" + NFD_TEXT + b"'\n", id="utf8-nfd"),
        pytest.param(b"\t \n\n", id="whitespace-only"),
    ],
)
def test_plain_utf8_scripts_are_accepted(data):
    result = validate(data)
    assert result.status == "accepted" and result.reason == ""


def test_size_cap_counts_bytes_at_boundary():
    body = "echo café\n".encode()  # multibyte content
    assert validate(body, max_bytes=len(body)).status == "accepted"
    assert_rejected(validate(body, max_bytes=len(body) - 1))


def test_nul_byte_is_rejected():
    assert_rejected(validate(b"echo hi\x00\n"))


@pytest.mark.parametrize("data", [b"\xff\xfe", b"echo \xc3\n", b"\xed\xa0\x80"], ids=["invalid", "truncated-sequence", "surrogate"])
def test_invalid_utf8_is_rejected(data):
    assert_rejected(validate(data))


def test_result_invariants_are_enforced():
    m = _mod()
    with pytest.raises(ValueError):
        m.ScriptResult("accepted", "why")
    with pytest.raises(ValueError):
        m.ScriptResult("rejected", "")
    with pytest.raises(ValueError):
        m.ScriptResult("rejected", "x" * 201)


def test_limits_require_positive_integer():
    m = _mod()
    with pytest.raises(TypeError):
        m.ScriptLimits()
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            m.ScriptLimits(max_script_bytes=bad)


def test_non_bytes_input_is_a_type_error():
    m = _mod()
    with pytest.raises(TypeError):
        m.validate_script("echo hi\n", m.ScriptLimits(max_script_bytes=1024))
