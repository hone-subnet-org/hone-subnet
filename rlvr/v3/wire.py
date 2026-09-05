from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from .canonical import (
    MAX_NESTING_DEPTH,
    SAFE_INTEGER_MAX,
    validate_protocol_string,
)


def _require_exact_integer(value: object) -> object:
    if type(value) is not int:
        raise ValueError("value must be an integer")
    return value


def exact_int_literal(*values: int) -> type:
    if not values or any(type(value) is not int for value in values):
        raise TypeError("exact_int_literal requires integer values")
    literal = Literal.__getitem__(values)
    return Annotated[literal, BeforeValidator(_require_exact_integer)]


SchemaVersion: TypeAlias = exact_int_literal(1)
HexDigest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedIdentifier: TypeAlias = Annotated[str, Field(min_length=1, max_length=128)]
BoundedURL: TypeAlias = Annotated[str, Field(min_length=1, max_length=8_192)]
BoundedSize: TypeAlias = Annotated[int, Field(ge=0, le=SAFE_INTEGER_MAX)]
PositiveBoundedSize: TypeAlias = Annotated[
    int, Field(gt=0, le=SAFE_INTEGER_MAX)
]
Timestamp: TypeAlias = Annotated[int, Field(ge=0, le=SAFE_INTEGER_MAX)]
UID: TypeAlias = Annotated[int, Field(ge=0, le=1_023)]


def _validate_value(value: Any, *, depth: int = 0) -> None:
    if isinstance(value, str):
        validate_protocol_string(value)
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        if not -SAFE_INTEGER_MAX <= value <= SAFE_INTEGER_MAX:
            raise ValueError("integer is outside the interoperable range")
        return
    if isinstance(value, float):
        raise ValueError("floating-point values are not allowed")
    if isinstance(value, WireModel):
        _validate_value(value.model_dump(mode="python"), depth=depth)
        return
    if isinstance(value, dict):
        if depth >= MAX_NESTING_DEPTH:
            raise ValueError("maximum wire nesting depth exceeded")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("wire object keys must be strings")
            _validate_value(key, depth=depth + 1)
            _validate_value(item, depth=depth + 1)
        return
    if isinstance(value, list):
        if depth >= MAX_NESTING_DEPTH:
            raise ValueError("maximum wire nesting depth exceeded")
        for item in value:
            _validate_value(item, depth=depth + 1)
        return
    raise ValueError(f"unsupported wire value: {type(value).__name__}")


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_wire_values(cls, value: Any) -> Any:
        _validate_value(value)
        return value
