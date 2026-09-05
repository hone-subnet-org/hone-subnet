from __future__ import annotations

import json
import unicodedata
from typing import Any, NoReturn

from pydantic import BaseModel

SAFE_INTEGER_MAX = 2**53 - 1
MAX_NESTING_DEPTH = 64


class CanonicalizationError(ValueError):
    pass


def _reject_number(value: str) -> NoReturn:
    raise CanonicalizationError(f"non-integer number is not allowed: {value}")


def _parse_integer(value: str) -> int:
    parsed = int(value)
    if not -SAFE_INTEGER_MAX <= parsed <= SAFE_INTEGER_MAX:
        raise CanonicalizationError("integer is outside the interoperable range")
    return parsed


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalizationError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def validate_protocol_string(value: str) -> None:
    if unicodedata.normalize("NFC", value) != value:
        raise CanonicalizationError("strings must be Unicode NFC")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise CanonicalizationError("strings must not contain surrogate code points")


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        validate_protocol_string(value)
        return
    if isinstance(value, int):
        if not -SAFE_INTEGER_MAX <= value <= SAFE_INTEGER_MAX:
            raise CanonicalizationError("integer is outside the interoperable range")
        return
    if isinstance(value, float):
        raise CanonicalizationError("floating-point values are not allowed")
    if isinstance(value, list):
        if depth >= MAX_NESTING_DEPTH:
            raise CanonicalizationError("maximum JSON nesting depth exceeded")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if depth >= MAX_NESTING_DEPTH:
            raise CanonicalizationError("maximum JSON nesting depth exceeded")
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("object keys must be strings")
            validate_protocol_string(key)
            _validate_json_value(item, depth=depth + 1)
        return
    raise CanonicalizationError(f"unsupported JSON value: {type(value).__name__}")


def parse_strict_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        if text.startswith("\ufeff"):
            raise CanonicalizationError("a UTF-8 BOM is not allowed")
        value = json.loads(
            text,
            parse_int=_parse_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
            object_pairs_hook=_object_from_pairs,
        )
        _validate_json_value(value)
        return value
    except CanonicalizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CanonicalizationError("invalid canonical JSON input") from exc


def _serialize(value: Any, *, depth: int = 0) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        if depth >= MAX_NESTING_DEPTH:
            raise CanonicalizationError("maximum JSON nesting depth exceeded")
        return "[" + ",".join(
            _serialize(item, depth=depth + 1) for item in value
        ) + "]"
    if isinstance(value, dict):
        if depth >= MAX_NESTING_DEPTH:
            raise CanonicalizationError("maximum JSON nesting depth exceeded")
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return "{" + ",".join(
            _serialize(key, depth=depth + 1)
            + ":"
            + _serialize(value[key], depth=depth + 1)
            for key in keys
        ) + "}"
    raise CanonicalizationError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        # Python mode preserves non-JSON values so validation rejects them.
        value = value.model_dump(mode="python")
    _validate_json_value(value)
    try:
        return _serialize(value).encode("utf-8")
    except CanonicalizationError:
        raise
    except (UnicodeEncodeError, RecursionError, TypeError, ValueError) as exc:
        raise CanonicalizationError("value cannot be serialized") from exc
