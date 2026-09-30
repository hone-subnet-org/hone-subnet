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

GRADING_CONCURRENCY_MAX = 16
HOST_MEMORY_RESERVE_BYTES = 4 * 1024**3  # the validator, Docker and the OS
HOST_DISK_RESERVE_BYTES = 2 * 1000**3  # the round's verifier and submissions; the workspace check runs per grading


def default_grading_concurrency(
    policy: ValidatorPolicy, *, cpus: int, memory_bytes: int, free_disk_bytes: int
) -> int:
    """How many gradings this host can run at once: one per sandbox CPU
    allowance, and one per sandbox memory limit and per workspace of disk
    after a reserve for the validator itself."""

    by_cpu = cpus // policy.v3_cpus
    by_memory = (memory_bytes - HOST_MEMORY_RESERVE_BYTES) // policy.v3_memory_bytes
    by_disk = (free_disk_bytes - HOST_DISK_RESERVE_BYTES) // policy.v3_workspace_bytes
    return max(1, min(GRADING_CONCURRENCY_MAX, by_cpu, by_memory, by_disk))


def round_policy(
    policy: ValidatorPolicy, *, dispatch_concurrency: int, grading_concurrency: int = 1
) -> RoundPolicy:
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
            max_workspace_bytes=policy.v3_workspace_bytes,
        ),
        docker_binary=str(os.path.realpath(docker_binary)),
        artifact_origins=frozenset(policy.v3_artifact_origins),
        execution_profile_id=policy.v3_execution_profile_id,
        verifier_policy="command-gold-digest-v1",
        dispatch_concurrency=dispatch_concurrency,
        grading_concurrency=grading_concurrency,
        miners_per_task=policy.v3_miners_per_task,
        commit_quorum=policy.v3_commit_quorum,
        min_lease_s=policy.v3_min_lease_s,
    )
