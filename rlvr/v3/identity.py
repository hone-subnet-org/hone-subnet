from __future__ import annotations

import hashlib
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, Field

from .canonical import canonical_json_bytes, validate_protocol_string
from .wire import HexDigest, SchemaVersion, WireModel

TASK_ID_DOMAIN = b"hone-v3-task-id\x00"

ProfileToken: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]


def validate_relative_path(value: str) -> str:
    validate_protocol_string(value)
    if not 1 <= len(value) <= 1_024:
        raise ValueError("relative path length is outside the allowed range")
    if "\x00" in value:
        raise ValueError("relative paths must not contain NUL bytes")
    if "\\" in value:
        raise ValueError("relative paths must use POSIX separators")
    if value.startswith("/"):
        raise ValueError("relative paths must not be absolute")
    if value == ".":
        return value
    segments = value.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError("relative path is not normalized")
    return value


RelativePath: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=1_024),
    AfterValidator(validate_relative_path),
]


class RepositoryTaskIdentity(WireModel):
    schema_version: SchemaVersion = 1
    task_type: Literal["repository_patch_v1"] = "repository_patch_v1"
    task_kind: Literal["bug_fix", "feature"]
    instruction: Annotated[str, Field(min_length=1, max_length=500_000)]
    primary_language: Annotated[str, Field(min_length=1, max_length=64)]
    workspace_sha256: HexDigest
    verifier_sha256: HexDigest
    execution_profile_id: ProfileToken
    working_directory: RelativePath
    patch_format: Literal["unified_diff_v1"] = "unified_diff_v1"
    verifier_policy: ProfileToken
    authoring_version: Annotated[str, Field(min_length=1, max_length=128)]


class TerminalScriptTaskIdentity(WireModel):
    schema_version: SchemaVersion = 1
    task_type: Literal["terminal_script_v1"] = "terminal_script_v1"
    instruction: Annotated[str, Field(min_length=1, max_length=500_000)]
    environment_sha256: HexDigest
    verifier_sha256: HexDigest
    execution_profile_id: ProfileToken
    result_tree_path: RelativePath
    verifier_policy: ProfileToken
    authoring_version: Annotated[str, Field(min_length=1, max_length=128)]


TaskIdentity: TypeAlias = Annotated[
    RepositoryTaskIdentity | TerminalScriptTaskIdentity,
    Field(discriminator="task_type"),
]


def compute_task_id(
    identity: RepositoryTaskIdentity | TerminalScriptTaskIdentity,
) -> str:
    if type(identity) not in (RepositoryTaskIdentity, TerminalScriptTaskIdentity):
        raise TypeError("identity must be a V3 task identity")
    validated = type(identity).model_validate(identity.model_dump(mode="python"))
    return hashlib.sha256(TASK_ID_DOMAIN + canonical_json_bytes(validated)).hexdigest()
