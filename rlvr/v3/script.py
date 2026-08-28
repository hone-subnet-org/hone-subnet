from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ScriptLimits:
    max_script_bytes: int

    def __post_init__(self) -> None:
        if type(self.max_script_bytes) is not int or self.max_script_bytes <= 0:
            raise ValueError("script byte limit must be a positive integer")


@dataclass(frozen=True)
class ScriptResult:
    status: Literal["accepted", "rejected"]
    reason: str

    def __post_init__(self) -> None:
        if self.status == "accepted" and self.reason:
            raise ValueError("an accepted script cannot have a rejection reason")
        if self.status == "rejected" and not 0 < len(self.reason) <= 200:
            raise ValueError("a rejected script requires a bounded reason")


def validate_script(script: bytes, limits: ScriptLimits) -> ScriptResult:
    if not isinstance(script, bytes):
        raise TypeError("script must be bytes")
    if len(script) > limits.max_script_bytes:
        return ScriptResult("rejected", "script exceeds the policy byte limit")
    if b"\x00" in script:
        return ScriptResult("rejected", "script contains a NUL byte")
    try:
        script.decode("utf-8")
    except UnicodeDecodeError:
        return ScriptResult("rejected", "script is not valid UTF-8")
    return ScriptResult("accepted", "")
