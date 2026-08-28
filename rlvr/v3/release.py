from __future__ import annotations

import os
import shutil

from ..policy import ValidatorPolicy
from .archive import ArchiveLimits
from .patch import PatchLimits
from .round import RoundPolicy
from .script import ScriptLimits
from .submission import SubmissionLimits
from .supervisor import SupervisorPolicy
from .tree import TreeLimits


def round_policy(policy: ValidatorPolicy, *, dispatch_concurrency: int) -> RoundPolicy:
    candidate_uid = os.getuid()
    candidate_gid = os.getgid()
    if candidate_uid == 0 or candidate_gid == 0:
        raise ValueError("V3 validator must run as an unprivileged service user")
    docker_binary = shutil.which("docker")
    if docker_binary is None:
        raise ValueError("Docker is required for V3 grading")
    workspace_archive = ArchiveLimits(
        max_compressed_bytes=policy.v3_workspace_compressed_bytes,
        max_expanded_bytes=policy.v3_workspace_bytes,
        max_file_bytes=policy.v3_max_file_bytes,
        max_entries=200_000,
        max_path_bytes=4_096,
        max_zstd_window_bytes=128 * 1024**2,
    )
    verifier_archive = ArchiveLimits(
        max_compressed_bytes=policy.v3_verifier_compressed_bytes,
        max_expanded_bytes=policy.v3_verifier_expanded_bytes,
        max_file_bytes=512 * 1024**2,
        max_entries=20_000,
        max_path_bytes=4_096,
        max_zstd_window_bytes=128 * 1024**2,
    )
    return RoundPolicy(
        workspace_archive=workspace_archive,
        verifier_archive=verifier_archive,
        submissions=SubmissionLimits(
            max_patch_bytes=policy.v3_patch_bytes,
            max_script_bytes=policy.v3_script_bytes,
        ),
        tree=TreeLimits(
            max_entries=200_000,
            max_file_bytes=policy.v3_max_file_bytes,
            max_total_file_bytes=policy.v3_workspace_bytes,
            max_path_bytes=4_096,
        ),
        patch=PatchLimits(max_patch_bytes=policy.v3_patch_bytes, git_timeout_s=30),
        script=ScriptLimits(max_script_bytes=policy.v3_script_bytes),
        supervisor=SupervisorPolicy(
            image=policy.v3_image,
            candidate_uid=candidate_uid,
            candidate_gid=candidate_gid,
            memory_bytes=policy.v3_memory_bytes,
            cpus=policy.v3_cpus,
            pids_limit=policy.v3_pids_limit,
            tmpfs_bytes=policy.v3_tmpfs_bytes,
            max_file_bytes=policy.v3_max_file_bytes,
            watchdog_slack_s=5,
        ),
        docker_binary=str(os.path.realpath(docker_binary)),
        artifact_origins=frozenset(policy.v3_artifact_origins),
        execution_profile_id=policy.v3_execution_profile_id,
        verifier_policy="command-gold-digest-v1",
        dispatch_concurrency=dispatch_concurrency,
    )
