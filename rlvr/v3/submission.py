from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from .artifacts import ArtifactGrant
from . import download


@dataclass(frozen=True)
class SubmissionLimits:
    max_patch_bytes: int
    max_script_bytes: int

    def __post_init__(self) -> None:
        for value in (self.max_patch_bytes, self.max_script_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("submission limits must be positive integers")


@dataclass(frozen=True)
class FetchedSubmission:
    path: Path
    artifact_role: Literal["patch", "script"]
    artifact_format: Literal["unified_diff_v1", "bash_script_v1"]
    size_bytes: int
    sha256: str


async def fetch_submission(
    http: httpx.AsyncClient,
    grant: ArtifactGrant,
    destination: str | os.PathLike[str],
    limits: SubmissionLimits,
    *,
    allowed_origins: frozenset[str],
) -> FetchedSubmission:
    artifact_role: Literal["patch", "script"] = (
        "patch" if grant.format == "unified_diff_v1" else "script"
    )
    cap = limits.max_patch_bytes if artifact_role == "patch" else limits.max_script_bytes
    path = await download.download_object(
        http,
        grant.read_url,
        sha256=grant.sha256,
        size_bytes=grant.size_bytes,
        max_bytes=cap,
        destination=destination,
        allowed_origins=allowed_origins,
    )
    try:
        path.chmod(0o444)
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        raise download.ArtifactDownloadError("submission could not be secured") from None
    return FetchedSubmission(
        path=path,
        artifact_role=artifact_role,
        artifact_format=grant.format,
        size_bytes=grant.size_bytes,
        sha256=grant.sha256,
    )
