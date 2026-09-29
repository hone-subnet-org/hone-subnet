"""V3 shared wire-model behavior and canonical JSON primitives.

Canonical form is the protocol's INTEGER-ONLY SUBSET of RFC 8785 (JCS).
RFC 8785 itself serializes finite IEEE-754 numbers; this protocol forbids
every non-integer number and every integer outside [-(2^53-1), 2^53-1], so
the only numbers that can appear are interoperable integers. Everything else
(key ordering by UTF-16 code units, string escaping, no whitespace) follows
RFC 8785 exactly.

Contracted interfaces:

    rlvr.v3.wire.WireModel
        pydantic base: extra="forbid", frozen, strict types; rejects floats,
        bool-as-int, str-as-int, out-of-range integers, and non-NFC strings at
        any nesting depth.
    rlvr.v3.canonical
        parse_strict_json(raw: bytes) -> object
            rejects invalid UTF-8, a BOM, trailing content, NaN/Infinity
            literals, non-integer numbers, out-of-range integers, duplicate
            keys at any depth, lone surrogate escapes, and non-NFC strings at
            any depth (values and keys).
        canonical_json_bytes(obj) -> bytes
            RFC 8785 serialization of the subset above; rejects floats,
            out-of-range integers, lone surrogates, non-NFC strings (values
            and keys), and non-JSON Python types such as bytes and tuple.
        CanonicalizationError(ValueError)

Imports are inside each test so a missing module reports per test.
"""

from __future__ import annotations

import json
import unicodedata

import pytest
from pydantic import ValidationError

SAFE_MAX = 2**53 - 1
NFC_E = "é"
NFD_E = "é"
assert unicodedata.normalize("NFC", NFD_E) == NFC_E and NFD_E != NFC_E


def _models():
    from rlvr.v3.wire import WireModel

    class Sample(WireModel):
        name: str
        count: int = 1

    class Nested(WireModel):
        items: dict[str, list[int]]
        labels: list[str]

    return Sample, Nested


def _canonical():
    from rlvr.v3 import canonical

    return canonical


# --------------------------------------------------------------------------- #
# WireModel behavior
# --------------------------------------------------------------------------- #
def test_wire_model_rejects_unknown_fields():
    Sample, _ = _models()
    with pytest.raises(ValidationError):
        Sample(name="a", extra_field=1)


def test_wire_model_is_frozen():
    Sample, _ = _models()
    instance = Sample(name="a")
    with pytest.raises(ValidationError):
        instance.name = "b"


@pytest.mark.parametrize("count", [1.0, True, "1"], ids=["float", "bool", "str"])
def test_wire_model_integer_fields_are_strict(count):
    Sample, _ = _models()
    with pytest.raises(ValidationError):
        Sample(name="a", count=count)


def test_wire_model_enforces_interoperable_integer_bounds():
    Sample, _ = _models()
    assert Sample(name="a", count=SAFE_MAX).count == SAFE_MAX
    assert Sample(name="a", count=-SAFE_MAX).count == -SAFE_MAX
    with pytest.raises(ValidationError):
        Sample(name="a", count=SAFE_MAX + 1)
    with pytest.raises(ValidationError):
        Sample(name="a", count=-SAFE_MAX - 1)


def test_wire_model_rejects_non_nfc_strings():
    Sample, _ = _models()
    assert Sample(name=NFC_E).name == NFC_E
    with pytest.raises(ValidationError):
        Sample(name=NFD_E)


def test_wire_model_validates_recursively():
    _, Nested = _models()
    good = Nested(items={"k": [1, SAFE_MAX]}, labels=[NFC_E])
    assert good.items == {"k": [1, SAFE_MAX]}

    with pytest.raises(ValidationError):
        Nested(items={"k": [1]}, labels=[NFD_E])  # non-NFC list element
    with pytest.raises(ValidationError):
        Nested(items={NFD_E: [1]}, labels=[])  # non-NFC dict key
    with pytest.raises(ValidationError):
        Nested(items={"k": [SAFE_MAX + 1]}, labels=[])  # nested out of range
    with pytest.raises(ValidationError):
        Nested(items={"k": [True]}, labels=[])  # nested bool-as-int


# --------------------------------------------------------------------------- #
# Strict parsing from raw JSON bytes
# --------------------------------------------------------------------------- #
def test_parse_rejects_duplicate_keys_at_any_depth():
    canonical = _canonical()
    for raw in (b'{"a": 1, "a": 2}', b'{"a": {"b": 1, "b": 2}}', b'[{"k": 1, "k": 1}]'):
        with pytest.raises(canonical.CanonicalizationError):
            canonical.parse_strict_json(raw)


@pytest.mark.parametrize(
    "raw",
    [b"1.0", b"1e3", b"-0.0", b'{"a": 0.5}', b"[1, 2.25]"],
    ids=["float", "exponent", "neg-zero-float", "nested", "in-list"],
)
def test_parse_rejects_non_integer_numbers(raw):
    canonical = _canonical()
    with pytest.raises(canonical.CanonicalizationError):
        canonical.parse_strict_json(raw)


@pytest.mark.parametrize(
    "raw",
    [b"NaN", b"Infinity", b"-Infinity", b'{"a": NaN}'],
    ids=["nan", "inf", "neg-inf", "nested-nan"],
)
def test_parse_rejects_non_finite_literals(raw):
    canonical = _canonical()
    with pytest.raises(canonical.CanonicalizationError):
        canonical.parse_strict_json(raw)


def test_parse_enforces_interoperable_integer_bounds():
    canonical = _canonical()
    assert canonical.parse_strict_json(str(SAFE_MAX).encode()) == SAFE_MAX
    assert canonical.parse_strict_json(str(-SAFE_MAX).encode()) == -SAFE_MAX
    for raw in (str(SAFE_MAX + 1), str(-SAFE_MAX - 1), '{"a": [%d]}' % (SAFE_MAX + 1)):
        with pytest.raises(canonical.CanonicalizationError):
            canonical.parse_strict_json(raw.encode())


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",  # invalid UTF-8
        b'\xef\xbb\xbf{"a": 1}',  # BOM
        b'{"a": 1} {"b": 2}',  # trailing document
        b'{"a": 1}x',  # trailing garbage
        b'"\\ud800"',  # lone high surrogate escape
        b'"\\udc00"',  # lone low surrogate escape
        b'{"a": "\\ud83d"}',  # lone surrogate nested
    ],
    ids=["bad-utf8", "bom", "trailing-json", "trailing-bytes", "lone-hi", "lone-lo", "lone-nested"],
)
def test_parse_rejects_malformed_encodings(raw):
    canonical = _canonical()
    with pytest.raises(canonical.CanonicalizationError):
        canonical.parse_strict_json(raw)


def test_parse_accepts_valid_surrogate_pair():
    canonical = _canonical()
    assert canonical.parse_strict_json(b'"\\ud83d\\ude00"') == "\U0001f600"


def test_parse_rejects_non_nfc_strings_at_any_depth():
    canonical = _canonical()
    bad_value = json.dumps({"a": NFD_E}).encode()
    bad_nested = json.dumps({"a": {"b": [NFD_E]}}).encode()
    bad_key = json.dumps({NFD_E: 1}).encode()
    for raw in (bad_value, bad_nested, bad_key):
        with pytest.raises(canonical.CanonicalizationError):
            canonical.parse_strict_json(raw)
    assert canonical.parse_strict_json(json.dumps({"a": NFC_E}).encode()) == {
        "a": NFC_E
    }


# --------------------------------------------------------------------------- #
# RFC 8785 serialization (integer-only subset)
# --------------------------------------------------------------------------- #
def test_canonical_bytes_have_no_whitespace_and_sorted_keys():
    canonical = _canonical()
    out = canonical.canonical_json_bytes({"b": 1, "a": [True, None, "x"]})
    assert out == b'{"a":[true,null,"x"],"b":1}'


def test_canonical_key_order_is_utf16_code_units():
    """RFC 8785 orders property names by UTF-16 code units, not code points.
    U+1F600 encodes as D83D DE00 and therefore sorts BEFORE U+FF00."""
    canonical = _canonical()
    out = canonical.canonical_json_bytes({"＀": 1, "\U0001f600": 2})
    assert out == '{"\U0001f600":2,"＀":1}'.encode("utf-8")


def test_canonical_string_escaping():
    """Control characters use the short escapes or lowercase \\u00xx; quote and
    backslash are escaped; '/' and non-ASCII are emitted literally."""
    canonical = _canonical()
    value = "\b\t\n\f\r\"\\/\x01" + NFC_E
    out = canonical.canonical_json_bytes({"s": value})
    assert out == b'{"s":"\\b\\t\\n\\f\\r\\"\\\\/\\u0001\xc3\xa9"}'


def test_semantically_equal_inputs_canonicalize_identically():
    canonical = _canonical()
    first = canonical.parse_strict_json(
        b'{ "z" : 1 , "a" : { "y" : [1, 2] , "x" : "v" } }'
    )
    second = canonical.parse_strict_json(b'{"a":{"x":"v","y":[1,2]},"z":1}')
    assert canonical.canonical_json_bytes(first) == canonical.canonical_json_bytes(
        second
    )
    assert canonical.canonical_json_bytes(first) == b'{"a":{"x":"v","y":[1,2]},"z":1}'


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        {"a": 0.5},
        [float("nan")],
        {"a": [SAFE_MAX + 1]},
        "\ud800",
        {"k": "\udc00"},
        {NFD_E: 1},
        {"k": NFD_E},
        b"bytes",
        (1, 2),
        {"k": (1, 2)},
    ],
    ids=[
        "float", "nested-float", "nan", "nested-out-of-range", "lone-surrogate",
        "nested-lone-surrogate", "non-nfc-key", "non-nfc-value", "bytes",
        "tuple", "nested-tuple",
    ],
)
def test_canonical_rejects_unrepresentable_values(value):
    canonical = _canonical()
    with pytest.raises(canonical.CanonicalizationError):
        canonical.canonical_json_bytes(value)


def test_canonical_of_model_includes_defaulted_fields():
    """Canonical bytes come from the validated model with every field present,
    so a defaulted field is serialized, never omitted."""
    Sample, _ = _models()
    canonical = _canonical()
    assert canonical.canonical_json_bytes(Sample(name="a")) == (
        b'{"count":1,"name":"a"}'
    )


# --------------------------------------------------------------------------- #
# Any-typed fields still enforce JSON-native wire values
# --------------------------------------------------------------------------- #
def _loose_models():
    from typing import Any

    from rlvr.v3.wire import WireModel

    class Sample(WireModel):
        name: str
        count: int = 1

    class Loose(WireModel):
        value: Any

    class Outer(WireModel):
        inner: Sample

    return Sample, Loose, Outer


def _unsupported_values():
    import datetime
    import decimal

    from pydantic import BaseModel

    class NotWire(BaseModel):
        x: int = 1

    return {
        "bytes": b"raw",
        "tuple": (1, 2),
        "set": {1, 2},
        "decimal": decimal.Decimal("1"),
        "datetime": datetime.datetime(2026, 1, 1),
        "non-wire-model": NotWire(),
    }


@pytest.mark.parametrize("kind", ["bytes", "tuple", "set", "decimal", "datetime", "non-wire-model"])
def test_any_field_rejects_non_json_native_values(kind):
    _, Loose, _ = _loose_models()
    with pytest.raises(ValidationError):
        Loose(value=_unsupported_values()[kind])


def test_any_field_still_enforces_nfc_and_bounds_recursively():
    _, Loose, _ = _loose_models()
    assert Loose(value={"k": [1, "x", None, True]}).value == {"k": [1, "x", None, True]}
    with pytest.raises(ValidationError):
        Loose(value={"k": [NFD_E]})
    with pytest.raises(ValidationError):
        Loose(value=[[SAFE_MAX + 1]])


def test_nested_wire_model_is_accepted_as_instance_and_as_mapping():
    Sample, Loose, Outer = _loose_models()
    assert Outer(inner=Sample(name="a")).inner.name == "a"
    assert Outer(inner={"name": "a"}).inner.count == 1
    assert Loose(value=Sample(name="a")).value.name == "a"


def test_model_construct_cannot_smuggle_invalid_values_into_a_parent():
    """model_construct skips validation; a parent must revalidate nested
    instances rather than trust them."""
    Sample, Loose, Outer = _loose_models()
    smuggled = Sample.model_construct(name=NFD_E, count=SAFE_MAX + 1)
    with pytest.raises(ValidationError):
        Outer(inner=smuggled)
    with pytest.raises(ValidationError):
        Loose(value=smuggled)


def test_model_construct_cannot_smuggle_invalid_values_into_canonical_bytes():
    Sample, _, _ = _loose_models()
    canonical = _canonical()
    smuggled = Sample.model_construct(name="a", count=SAFE_MAX + 1)
    with pytest.raises(canonical.CanonicalizationError):
        canonical.canonical_json_bytes(smuggled)


# --------------------------------------------------------------------------- #
# Nesting depth is bounded everywhere, with typed failures
# --------------------------------------------------------------------------- #
def test_max_nesting_depth_is_64():
    canonical = _canonical()
    assert canonical.MAX_NESTING_DEPTH == 64


def test_parser_accepts_depth_64_and_rejects_65():
    canonical = _canonical()
    assert canonical.parse_strict_json(b"[" * 64 + b"]" * 64) is not None
    with pytest.raises(canonical.CanonicalizationError):
        canonical.parse_strict_json(b"[" * 65 + b"]" * 65)


def test_parser_turns_pathological_depth_into_a_typed_error():
    canonical = _canonical()
    with pytest.raises(canonical.CanonicalizationError):
        canonical.parse_strict_json(b"[" * 100_000)


def test_serializer_rejects_depth_65():
    canonical = _canonical()
    value: object = []
    for _ in range(64):
        value = [value]
    with pytest.raises(canonical.CanonicalizationError):
        canonical.canonical_json_bytes(value)


def test_wire_model_rejects_depth_65_in_any_field():
    _, Loose, _ = _loose_models()
    value: object = []
    for _ in range(64):
        value = [value]
    with pytest.raises(ValidationError):
        Loose(value=value)


# --------------------------------------------------------------------------- #
# Further RFC 8785 conformance and invalid-JSON forms
# --------------------------------------------------------------------------- #
def test_del_and_line_separator_are_emitted_literally():
    canonical = _canonical()
    out = canonical.canonical_json_bytes({"s": "\x7f\u2028"})
    assert out == b'{"s":"\x7f\xe2\x80\xa8"}'


def test_negative_zero_integer_is_zero():
    canonical = _canonical()
    assert canonical.parse_strict_json(b"-0") == 0
    assert canonical.canonical_json_bytes(canonical.parse_strict_json(b"-0")) == b"0"


def test_empty_containers():
    canonical = _canonical()
    assert canonical.parse_strict_json(b"[]") == []
    assert canonical.parse_strict_json(b"{}") == {}
    assert canonical.canonical_json_bytes({"a": [], "b": {}}) == b'{"a":[],"b":{}}'


@pytest.mark.parametrize(
    "raw",
    [b"00", b"01", b"-01", b"", b"   ", b"\n"],
    ids=["double-zero", "leading-zero", "neg-leading-zero", "empty", "spaces", "newline"],
)
def test_parse_rejects_invalid_json_forms(raw):
    canonical = _canonical()
    with pytest.raises(canonical.CanonicalizationError):
        canonical.parse_strict_json(raw)


# --------------------------------------------------------------------------- #
# Exact integer literals. pydantic accepts True for Literal[1] even in strict
# mode, so protocol version fields need a shared exact-int literal type.
#
# Exact integer literal contract:
#     rlvr.v3.wire.exact_int_literal(*values: int) -> type
#         an int Literal that additionally rejects bool and int subclasses
#     rlvr.v3.wire.SchemaVersion == exact_int_literal(1)
# --------------------------------------------------------------------------- #
def _versioned():
    from rlvr.v3.wire import WireModel, exact_int_literal

    class Versioned(WireModel):
        version: exact_int_literal(3) = 3

    class Multi(WireModel):
        version: exact_int_literal(1, 2)

    return Versioned, Multi


def test_exact_int_literal_accepts_exact_int_and_default():
    Versioned, Multi = _versioned()
    assert Versioned().version == 3
    assert Versioned(version=3).version == 3
    assert Versioned().model_dump(mode="json") == {"version": 3}
    assert Multi(version=1).version == 1
    assert Multi(version=2).version == 2


@pytest.mark.parametrize(
    "value",
    [True, False, "3", 3.0, 4, 0, -3, None, [3]],
    ids=["true", "false", "str", "float", "other-int", "zero", "neg", "none", "list"],
)
def test_exact_int_literal_rejects_non_exact_values(value):
    Versioned, _ = _versioned()
    with pytest.raises(ValidationError):
        Versioned(version=value)


def test_exact_int_literal_rejects_bool_for_every_allowed_value():
    _, Multi = _versioned()
    for value in (True, False):
        with pytest.raises(ValidationError):
            Multi(version=value)


def test_exact_int_literal_rejects_int_subclass():
    import enum

    class Version(enum.IntEnum):
        THREE = 3

    Versioned, _ = _versioned()
    with pytest.raises(ValidationError):
        Versioned(version=Version.THREE)


def test_exact_int_literal_json_schema_is_integer_const():
    Versioned, _ = _versioned()
    prop = Versioned.model_json_schema()["properties"]["version"]
    assert prop["type"] == "integer"
    assert prop.get("const") == 3 or prop.get("enum") == [3]


def test_shared_schema_version_alias():
    from rlvr.v3.wire import SchemaVersion, WireModel

    class Sample(WireModel):
        schema_version: SchemaVersion = 1

    assert Sample().schema_version == 1
    for value in (True, "1", 1.0, 2):
        with pytest.raises(ValidationError):
            Sample(schema_version=value)
