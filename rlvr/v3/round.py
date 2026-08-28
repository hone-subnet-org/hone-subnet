from __future__ import annotations

import asyncio
import hashlib
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from .api import (
    ChallengeCommitRequest,
    ChallengeFeedbackRequest,
    FeedbackVerdict,
    MinerSubmission,
    MinerTaskRequest,
    MinerTaskResponse,
    derive_miner_request_id,
    validate_commit_reveal,
)
from .artifacts import ArtifactGrant
from .archive import ArchiveLimits, extract_archive
from .client import V3ProblemServerClient
from .download import download_artifact
from .grading import EvaluationResult, evaluate_repository, evaluate_terminal
from .identity import RepositoryTaskIdentity, TerminalScriptTaskIdentity
from .manifest import load_manifest
from .patch import PatchLimits
from .script import ScriptLimits
from .submission import SubmissionLimits, fetch_submission
from .supervisor import SupervisorPolicy
from .tree import TreeLimits
from .workspace import cached_workspace_dir, ensure_cached_workspace, materialize_workspace


class V3Solver(Protocol):
    uid: int
    hotkey: str

    async def solve_v3(
        self, task: MinerTaskRequest
    ) -> tuple[MinerSubmission, MinerTaskResponse | None]: ...


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("cleanup target must be a real directory")
    target = path.absolute()
    if target.parent == target or not (
        target.name.startswith("hone-v3-round-")
        or re.fullmatch(r"[0-9a-f]{64}", target.name)
        or re.fullmatch(r"miner-[0-9]+", target.name)
    ):
        raise ValueError("cleanup target is outside the managed workspace")
    chmod = shutil.which("chmod")
    rm = shutil.which("rm")
    if chmod is None or rm is None:
        raise RuntimeError("grading workspace cleanup tools are unavailable")
    try:
        subprocess.run(
            [chmod, "-R", "u+rwX", "--", str(target)],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [rm, "-rf", "--", str(target)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("grading workspace cleanup failed") from None


@dataclass(frozen=True)
class RoundPolicy:
    workspace_archive: ArchiveLimits
    verifier_archive: ArchiveLimits
    submissions: SubmissionLimits
    tree: TreeLimits
    patch: PatchLimits
    script: ScriptLimits
    supervisor: SupervisorPolicy
    docker_binary: str
    artifact_origins: frozenset[str]
    execution_profile_id: str
    verifier_policy: str
    dispatch_concurrency: int

    def __post_init__(self) -> None:
        if type(self.dispatch_concurrency) is not int or self.dispatch_concurrency < 1:
            raise ValueError("dispatch concurrency must be positive")


@dataclass(frozen=True)
class MinerEvaluation:
    uid: int
    hotkey: str
    latency_ms: int
    result: EvaluationResult
    grading_duration_ms: int = 0


@dataclass(frozen=True)
class RoundResult:
    status: Literal["completed", "unavailable", "abandoned"]
    reason: str
    evaluations: tuple[MinerEvaluation, ...]
    retry_after_s: int | None = None


def _failed_submission(challenge_id: str, uid: int, hotkey: str, error: str) -> MinerSubmission:
    return MinerSubmission(
        uid=uid,
        hotkey=hotkey,
        request_id=derive_miner_request_id(challenge_id, uid, hotkey),
        response_body="",
        response_headers={},
        error=error[:4_096] or "miner unavailable",
        latency_ms=0,
    )


def _grant_matches_signed_response(
    grant: ArtifactGrant, response: MinerTaskResponse | None
) -> bool:
    return bool(
        response is not None
        and grant.upload_id == response.submission.upload_id
        and grant.sha256 == response.submission.sha256
        and grant.size_bytes == response.submission.size_bytes
        and grant.format == response.submission.artifact_format
    )


async def _send_diagnostic_feedback(
    client: V3ProblemServerClient,
    challenge_id: str,
    task_id: str,
    grants: list[ArtifactGrant],
    evaluations: list[MinerEvaluation],
) -> bool:
    if not grants:
        return True
    try:
        by_registration = {(item.uid, item.hotkey): item for item in evaluations}
        request = ChallengeFeedbackRequest(
            protocol_version=3,
            challenge_id=challenge_id,
            task_id=task_id,
            verdicts=[
                FeedbackVerdict(
                    uid=grant.uid,
                    hotkey=grant.hotkey,
                    passed=(by_registration[(grant.uid, grant.hotkey)].result.status == "passed"),
                    grading_duration_ms=by_registration[
                        (grant.uid, grant.hotkey)
                    ].grading_duration_ms,
                )
                for grant in grants
            ],
        )
        return await client.feedback(request)
    except Exception:  # noqa: BLE001 - feedback is diagnostic only
        return False


def compute_round_payments(
    result: RoundResult,
    *,
    speed_half_life_ms: float,
    speed_floor: float,
) -> dict[int, float]:
    if result.status != "completed":
        return {}
    passed = [item for item in result.evaluations if item.result.status == "passed"]
    fastest = min((item.latency_ms for item in passed if item.latency_ms > 0), default=None)
    floor = min(1.0, max(0.0, float(speed_floor)))
    half_life = float(speed_half_life_ms)
    payments: dict[int, float] = {}
    for item in result.evaluations:
        if item.result.status != "passed":
            payments[item.uid] = 0.0
            continue
        if fastest is None or not math.isfinite(half_life) or half_life <= 0:
            payments[item.uid] = 1.0
            continue
        delay = max(0, item.latency_ms - fastest)
        payments[item.uid] = floor + (1.0 - floor) * (2.0 ** (-delay / half_life))
    return payments


def apply_round_scores(
    result: RoundResult,
    engine,
    *,
    active_hotkeys: dict[int, str],
    speed_half_life_ms: float,
    speed_floor: float,
) -> bool:
    if result.status != "completed":
        return False
    eligible = tuple(
        item
        for item in result.evaluations
        if active_hotkeys.get(item.uid) == item.hotkey
    )
    filtered = RoundResult("completed", result.reason, eligible)
    payments = compute_round_payments(
        filtered,
        speed_half_life_ms=speed_half_life_ms,
        speed_floor=speed_floor,
    )
    dispatched = {item.uid for item in eligible}
    observed_at = engine.update(
        payments,
        hotkeys={item.uid: active_hotkeys[item.uid] for item in eligible},
        dispatched=dispatched,
    )
    engine.record_nonserving(set(active_hotkeys), observed_at=observed_at)
    return True


async def evaluate_round(
    client: V3ProblemServerClient,
    http: httpx.AsyncClient,
    solvers: list[V3Solver],
    policy: RoundPolicy,
    *,
    cache_dir: str | Path,
    work_dir: str | Path,
) -> RoundResult:
    outcome = await client.lease()
    lease = outcome.challenge
    if lease is None:
        reason = outcome.detail or outcome.category.value
        return RoundResult("unavailable", reason[:200], (), outcome.retry_after_s)

    stage = "workspace download"
    try:
        if lease.identity.execution_profile_id != policy.execution_profile_id:
            return RoundResult("abandoned", "unsupported execution profile", ())
        if lease.identity.verifier_policy != policy.verifier_policy:
            return RoundResult("abandoned", "unsupported verifier policy", ())
        cache = Path(cache_dir)
        root = Path(work_dir)
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for entry in root.iterdir():
            if (
                entry.name.startswith("hone-v3-round-")
                and entry.is_dir()
                and not entry.is_symlink()
            ):
                _remove_tree(entry)
        target_cache = cached_workspace_dir(cache, lease.workspace)
        for entry in cache.iterdir():
            if entry != target_cache:
                if entry.is_dir() and not entry.is_symlink():
                    _remove_tree(entry)
                elif entry.name.startswith(".workspace-"):
                    entry.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="hone-v3-round-", dir=root, ignore_cleanup_errors=True
        ) as temporary:
            round_dir = Path(temporary)
            scratch = round_dir / "scratch"
            scratch.mkdir()
            workspace_archive = round_dir / "workspace.tar.zst"
            if not target_cache.exists():
                workspace_space = (
                    lease.workspace.compressed_size_bytes
                    + 2 * lease.workspace.expanded_size_bytes
                )
                if shutil.disk_usage(root).free < workspace_space:
                    return RoundResult("abandoned", "insufficient workspace storage", ())
                workspace_archive = await download_artifact(
                    http,
                    lease.workspace_url,
                    lease.workspace,
                    workspace_archive,
                    policy.workspace_archive,
                    allowed_origins=policy.artifact_origins,
                )
            cached = ensure_cached_workspace(
                cache,
                workspace_archive,
                lease.workspace,
                policy.workspace_archive,
                scratch_dir=scratch,
            )

            by_registration = {(solver.uid, solver.hotkey): solver for solver in solvers}
            semaphore = asyncio.Semaphore(policy.dispatch_concurrency)

            async def dispatch(slots):
                registration = (slots.submission.uid, slots.submission.hotkey)
                solver = by_registration.get(registration)
                if solver is None:
                    return (
                        _failed_submission(
                            lease.challenge_id, *registration, "miner is not serving"
                        ),
                        None,
                    )
                task = MinerTaskRequest(
                    protocol_version=3,
                    challenge_id=lease.challenge_id,
                    task_id=lease.task_id,
                    identity=lease.identity,
                    workspace=lease.workspace,
                    workspace_url=lease.workspace_url,
                    expires_at=lease.expires_at,
                    slots=slots,
                )
                try:
                    async with semaphore:
                        submission, parsed = await solver.solve_v3(task)
                    return submission, parsed
                except Exception:  # noqa: BLE001
                    return (
                        _failed_submission(
                            lease.challenge_id, *registration, "miner dispatch failed"
                        ),
                        None,
                    )

            stage = "miner dispatch"
            dispatched = await asyncio.gather(
                *(dispatch(slots) for slots in lease.slot_pool)
            )
            submissions = [item[0] for item in dispatched]
            signed_responses = {
                (submission.uid, submission.hotkey): parsed
                for submission, parsed in dispatched
                if parsed is not None
            }
            if len(signed_responses) < lease.commit_min_signed_responses:
                return RoundResult("abandoned", "signed response quorum was not met", ())
            commit = ChallengeCommitRequest(
                protocol_version=3,
                challenge_id=lease.challenge_id,
                submissions=submissions,
            )
            stage = "commit"
            revealed = await client.commit(commit)
            if revealed is None:
                return RoundResult("abandoned", "commit or verifier reveal failed", ())
            try:
                validate_commit_reveal(lease, commit, revealed)
            except (TypeError, ValueError):
                return RoundResult("abandoned", "commit response did not match the lease", ())
            if revealed.grading_expires_at <= int(time.time()):
                return RoundResult("abandoned", "grading window expired", ())

            stage = "verifier download"
            verifier_space = (
                revealed.verifier.compressed_size_bytes
                + 2 * revealed.verifier.expanded_size_bytes
                + policy.tree.max_total_file_bytes
            )
            if shutil.disk_usage(round_dir).free < verifier_space:
                return RoundResult("abandoned", "insufficient verifier storage", ())
            verifier_archive = await download_artifact(
                http,
                revealed.verifier_url,
                revealed.verifier,
                round_dir / "verifier.tar.zst",
                policy.verifier_archive,
                allowed_origins=policy.artifact_origins,
            )
            verifier_dir = round_dir / "verifier"
            stage = "verifier extraction"
            extract_archive(
                verifier_archive,
                revealed.verifier,
                verifier_dir,
                policy.verifier_archive,
                scratch_dir=scratch,
            )
            stage = "verifier manifest"
            manifest = load_manifest(
                verifier_dir, task_type=lease.identity.task_type
            )

            evaluations: list[MinerEvaluation] = []
            submissions_by_registration = {
                (item.uid, item.hotkey): item for item in submissions
            }
            for failure in revealed.artifact_failures:
                latency = submissions_by_registration[(failure.uid, failure.hotkey)].latency_ms
                evaluations.append(
                    MinerEvaluation(
                        failure.uid,
                        failure.hotkey,
                        latency,
                        EvaluationResult("rejected", failure.reason, (), None),
                    )
                )
            for index, grant in enumerate(revealed.submission_grants):
                expected_format = (
                    "unified_diff_v1"
                    if lease.identity.task_type == "repository_patch_v1"
                    else "bash_script_v1"
                )
                if grant.format != expected_format:
                    return RoundResult("abandoned", "submission format did not match the task", ())
                signed = signed_responses.get((grant.uid, grant.hotkey))
                if not _grant_matches_signed_response(grant, signed):
                    return RoundResult("abandoned", "submission grant did not match the signed response", ())
                try:
                    stage = "submission download"
                    submission_path = await fetch_submission(
                        http,
                        grant,
                        round_dir / f"submission-{index}",
                        policy.submissions,
                        allowed_origins=policy.artifact_origins,
                    )
                    submission_bytes = submission_path.path.read_bytes()
                except Exception:  # noqa: BLE001 - post-commit storage is infrastructure
                    return RoundResult("abandoned", "submission download failed", ())
                stage = "workspace materialization"
                if revealed.grading_expires_at <= int(time.time()):
                    return RoundResult("abandoned", "grading window expired", ())
                if shutil.disk_usage(round_dir).free < policy.tree.max_total_file_bytes:
                    return RoundResult("abandoned", "insufficient grading storage", ())
                grading_started = time.monotonic()
                miner_workspace: Path | None = None
                challenge_tag = hashlib.sha256(
                    lease.challenge_id.encode("utf-8")
                ).hexdigest()[:12]
                run_prefix = f"v3-{challenge_tag}-{grant.uid}"
                try:
                    miner_workspace = materialize_workspace(
                        cached, round_dir / f"miner-{grant.uid}"
                    )
                    stage = "candidate grading"
                    if isinstance(lease.identity, RepositoryTaskIdentity):
                        result = await asyncio.to_thread(
                            evaluate_repository,
                            miner_workspace,
                            submission_bytes,
                            lease.identity,
                            manifest,
                            verifier_dir,
                            policy.supervisor,
                            policy.tree,
                            policy.patch,
                            policy.docker_binary,
                            run_prefix,
                        )
                    elif isinstance(lease.identity, TerminalScriptTaskIdentity):
                        result = await asyncio.to_thread(
                            evaluate_terminal,
                            miner_workspace,
                            submission_bytes,
                            lease.identity,
                            manifest,
                            verifier_dir,
                            policy.supervisor,
                            policy.tree,
                            policy.script,
                            policy.docker_binary,
                            run_prefix,
                        )
                    else:  # pragma: no cover - discriminated identity is closed
                        return RoundResult("abandoned", "unsupported task identity", ())
                    grading_duration_ms = min(
                        2**53 - 1,
                        int((time.monotonic() - grading_started) * 1000),
                    )
                finally:
                    if miner_workspace is not None:
                        _remove_tree(miner_workspace)
                if result.status == "abandoned":
                    return RoundResult("abandoned", result.reason, ())
                evaluations.append(
                    MinerEvaluation(
                        grant.uid,
                        grant.hotkey,
                        submissions_by_registration[(grant.uid, grant.hotkey)].latency_ms,
                        result,
                        grading_duration_ms,
                    )
                )
            evaluations.sort(key=lambda item: item.uid)
            feedback_accepted = await _send_diagnostic_feedback(
                client,
                lease.challenge_id,
                lease.task_id,
                revealed.submission_grants,
                evaluations,
            )
            if not feedback_accepted:
                print("[validator] WARN: V3 diagnostic feedback was not accepted")
            return RoundResult("completed", "", tuple(evaluations))
    except Exception as error:  # noqa: BLE001 - infrastructure faults abandon atomically
        return RoundResult(
            "abandoned",
            f"validator failed during {stage} ({type(error).__name__})",
            (),
        )
