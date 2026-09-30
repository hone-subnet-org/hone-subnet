from __future__ import annotations

import asyncio
import hashlib
import math
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from .api import (
    FEEDBACK_LATENCY_MAX_MS,
    ChallengeCommitRequest,
    ChallengeFeedbackRequest,
    FeedbackVerdict,
    MinerSubmission,
    MinerTaskRequest,
    MinerTaskResponse,
    derive_miner_request_id,
    validate_commit_reveal,
)
from .archive import ArchiveLimits, extract_archive
from .artifacts import ArtifactGrant, MinerArtifactRef
from .client import V3ProblemServerClient
from .download import download_artifact
from .grading import EvaluationResult, evaluate_repository, evaluate_terminal
from .identity import RepositoryTaskIdentity, TerminalScriptTaskIdentity
from .manifest import load_manifest
from .patch import PatchLimits
from .reasons import MinerReason, RoundReason, Stage
from .script import ScriptLimits
from .submission import SubmissionLimits, fetch_submission
from .supervisor import SupervisorPolicy
from .tree import TreeLimits
from .workspace import (
    cached_workspace_dir,
    ensure_cached_workspace,
    materialize_workspace,
)


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
    grading_concurrency: int = 1
    miners_per_task: int = 32
    # Signed responses a lease must gather before commit. The lease carries
    # the value too and must agree, so a quorum failure is a fact about the
    # miners and never a knob the server could turn to force a reroll.
    commit_quorum: int = 4
    # The least time a lease may leave miners at dispatch. Shorter leases are
    # refused without a fresh draw.
    min_lease_s: int = 600

    def __post_init__(self) -> None:
        if type(self.dispatch_concurrency) is not int or self.dispatch_concurrency < 1:
            raise ValueError("dispatch concurrency must be positive")
        if type(self.grading_concurrency) is not int or self.grading_concurrency < 1:
            raise ValueError("grading concurrency must be positive")
        if type(self.miners_per_task) is not int or not 1 <= self.miners_per_task <= 1_024:
            raise ValueError("miners per task must be between 1 and 1024")
        if type(self.commit_quorum) is not int or not 1 <= self.commit_quorum <= self.miners_per_task:
            raise ValueError("commit quorum must be between 1 and miners per task")
        if type(self.min_lease_s) is not int or self.min_lease_s < 1:
            raise ValueError("minimum lease window must be positive")


class _Abandon(Exception):
    """One miner's grading hit a validator-side fault that ends the round.

    Carries that miner's evaluation when grading had already produced one,
    so diagnostics still record it.
    """

    def __init__(
        self,
        reason: str,
        code: RoundReason,
        stage: Stage,
        evaluation: MinerEvaluation | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.stage = stage
        self.evaluation = evaluation


@dataclass(frozen=True)
class MinerEvaluation:
    uid: int
    hotkey: str
    latency_ms: int
    result: EvaluationResult
    grading_duration_ms: int = 0
    # From the miner's signed, slot-matched response; None when the server
    # rejected the submission at commit and no trusted trajectory exists.
    trajectory: MinerArtifactRef | None = None


@dataclass(frozen=True)
class RoundResult:
    status: Literal["completed", "unavailable", "abandoned"]
    reason: str
    evaluations: tuple[MinerEvaluation, ...]
    retry_after_s: int | None = None
    reason_code: RoundReason | None = None
    stage: Stage | None = None
    challenge_id: str | None = None
    task_id: str | None = None
    assigned_miners: tuple[tuple[int, str], ...] = ()
    diagnostic_evaluations: tuple[MinerEvaluation, ...] = ()
    dispatch_failures: tuple[tuple[int, str, MinerReason], ...] = ()
    checks_total: int | None = None


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


def _verdict(grant: ArtifactGrant, evaluation: MinerEvaluation) -> FeedbackVerdict:
    passed = evaluation.result.status == "passed"
    reason = evaluation.result.reason_code
    latency = evaluation.latency_ms
    # Anything the server would refuse is sent as null, so one odd value can
    # never sink the whole call.
    if type(latency) is not int or not 0 <= latency <= FEEDBACK_LATENCY_MAX_MS:
        latency = None
    return FeedbackVerdict(
        uid=grant.uid,
        hotkey=grant.hotkey,
        passed=passed,
        grading_duration_ms=evaluation.grading_duration_ms,
        response_latency_ms=latency,
        reason_code=reason if not passed and isinstance(reason, MinerReason) else None,
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
            verdicts=[_verdict(grant, by_registration[(grant.uid, grant.hotkey)]) for grant in grants],
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
    candidates: Sequence[tuple[int, str]],
) -> RoundResult:
    # The validator offers these miners, in its own random order. The server
    # must issue slots for exactly the first N of them, N being its setting.
    offered = list(candidates)
    if not offered:
        return RoundResult(
            "unavailable", "no eligible miners to offer", (),
            reason_code=RoundReason.LEASE_UNAVAILABLE, stage=Stage.LEASE,
        )
    outcome = await client.lease(offered)
    lease = outcome.challenge
    if lease is None:
        reason = outcome.detail or outcome.category.value
        return RoundResult(
            "unavailable", reason[:200], (), outcome.retry_after_s,
            reason_code=RoundReason.LEASE_UNAVAILABLE, stage=Stage.LEASE,
        )

    stage = Stage.LEASE
    evaluations: list[MinerEvaluation] = []
    dispatch_failures: dict[tuple[int, str], MinerReason] = {}
    checks_total: int | None = None

    def finish(
        status: Literal["completed", "abandoned"],
        reason: str = "",
        code: RoundReason | None = None,
        failure_stage: Stage | None = None,
    ) -> RoundResult:
        observed = tuple(sorted(evaluations, key=lambda item: item.uid))
        return RoundResult(
            status, reason, observed if status == "completed" else (),
            reason_code=code,
            stage=failure_stage or stage,
            challenge_id=lease.challenge_id,
            task_id=lease.task_id,
            assigned_miners=tuple(
                (slots.submission.uid, slots.submission.hotkey) for slots in lease.slot_pool
            ),
            diagnostic_evaluations=observed,
            dispatch_failures=tuple(
                (uid, hotkey, failure) for (uid, hotkey), failure in sorted(dispatch_failures.items())
            ),
            checks_total=checks_total,
        )

    try:
        pool = {(slots.submission.uid, slots.submission.hotkey) for slots in lease.slot_pool}
        expected = min(policy.miners_per_task, len(offered))
        if len(pool) != expected or pool != set(offered[:expected]):
            # Not exactly the first N offered, N fixed by release policy: the
            # server added, skipped, substituted, widened or narrowed, any of
            # which would let it steer selection.
            return finish("abandoned", "lease pool is not the first miners the validator offered", RoundReason.SLOT_POOL_MISMATCH)
        if lease.commit_min_signed_responses != policy.commit_quorum:
            return finish("abandoned", "lease quorum differs from release policy", RoundReason.LEASE_QUORUM_MISMATCH)
        if lease.expires_at - time.time() < policy.min_lease_s:
            return finish("abandoned", "lease leaves miners less time than release policy requires", RoundReason.LEASE_TOO_SHORT)
        if lease.identity.execution_profile_id != policy.execution_profile_id:
            return finish("abandoned", "unsupported execution profile", RoundReason.UNSUPPORTED_PROFILE)
        if lease.identity.verifier_policy != policy.verifier_policy:
            return finish("abandoned", "unsupported verifier policy", RoundReason.UNSUPPORTED_VERIFIER_POLICY)
        stage = Stage.WORKSPACE_MATERIALIZATION
        # Absolute, whatever the caller passed: the sandbox only mounts absolute
        # paths, and before Python 3.12 tempfile keeps a relative dir relative.
        cache = Path(cache_dir).absolute()
        root = Path(work_dir).absolute()
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        stage = Stage.CLEANUP
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
        stage = Stage.WORKSPACE_MATERIALIZATION
        with tempfile.TemporaryDirectory(
            prefix="hone-v3-round-", dir=root, ignore_cleanup_errors=True
        ) as temporary:
            round_dir = Path(temporary)
            scratch = round_dir / "scratch"
            scratch.mkdir()
            workspace_archive = round_dir / "workspace.tar.zst"
            stage = Stage.WORKSPACE_DOWNLOAD
            if not target_cache.exists():
                workspace_space = (
                    lease.workspace.compressed_size_bytes
                    + 2 * lease.workspace.expanded_size_bytes
                )
                if shutil.disk_usage(root).free < workspace_space:
                    return finish("abandoned", "insufficient workspace storage", RoundReason.INSUFFICIENT_STORAGE)
                workspace_archive = await download_artifact(
                    http,
                    lease.workspace_url,
                    lease.workspace,
                    workspace_archive,
                    policy.workspace_archive,
                    allowed_origins=policy.artifact_origins,
                )
            stage = Stage.WORKSPACE_EXTRACTION
            cached = ensure_cached_workspace(
                cache,
                workspace_archive,
                lease.workspace,
                policy.workspace_archive,
                scratch_dir=scratch,
            )

            by_registration = {(solver.uid, solver.hotkey): solver for solver in solvers}
            # Every offered miner is contacted at once: a miner that waited for
            # a dispatch slot would get less than the released minimum window.
            semaphore = asyncio.Semaphore(max(policy.dispatch_concurrency, len(lease.slot_pool)))

            async def dispatch(slots):
                registration = (slots.submission.uid, slots.submission.hotkey)
                solver = by_registration.get(registration)
                if solver is None:
                    dispatch_failures[registration] = MinerReason.NOT_SERVING
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
                    if parsed is None:
                        dispatch_failures[registration] = MinerReason.RESPONSE_UNAVAILABLE
                    return submission, parsed
                except Exception:  # noqa: BLE001
                    dispatch_failures[registration] = MinerReason.DISPATCH_FAILED
                    return (
                        _failed_submission(
                            lease.challenge_id, *registration, "miner dispatch failed"
                        ),
                        None,
                    )

            stage = Stage.DISPATCH
            if lease.expires_at - time.time() < policy.min_lease_s:
                # Checked again here: downloading and extracting the workspace
                # took time, and miners must still get the released minimum.
                return finish("abandoned", "lease leaves miners less time than release policy requires", RoundReason.LEASE_TOO_SHORT)
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
                return finish("abandoned", "signed response quorum was not met", RoundReason.QUORUM_NOT_MET)
            commit = ChallengeCommitRequest(
                protocol_version=3,
                challenge_id=lease.challenge_id,
                submissions=submissions,
            )
            stage = Stage.COMMIT
            revealed = await client.commit(commit)
            if revealed is None:
                return finish("abandoned", "commit or verifier reveal failed", RoundReason.COMMIT_FAILED)
            try:
                validate_commit_reveal(lease, commit, revealed)
            except (TypeError, ValueError):
                return finish("abandoned", "commit response did not match the lease", RoundReason.REVEAL_MISMATCH)
            if revealed.grading_expires_at <= int(time.time()):
                return finish("abandoned", "grading window expired", RoundReason.GRADING_EXPIRED)

            stage = Stage.VERIFIER_DOWNLOAD
            verifier_space = (
                revealed.verifier.compressed_size_bytes
                + 2 * revealed.verifier.expanded_size_bytes
                + policy.tree.max_total_file_bytes * policy.grading_concurrency
            )
            if shutil.disk_usage(round_dir).free < verifier_space:
                return finish("abandoned", "insufficient verifier storage", RoundReason.INSUFFICIENT_STORAGE)
            verifier_archive = await download_artifact(
                http,
                revealed.verifier_url,
                revealed.verifier,
                round_dir / "verifier.tar.zst",
                policy.verifier_archive,
                allowed_origins=policy.artifact_origins,
            )
            verifier_dir = round_dir / "verifier"
            stage = Stage.VERIFIER_EXTRACTION
            extract_archive(
                verifier_archive,
                revealed.verifier,
                verifier_dir,
                policy.verifier_archive,
                scratch_dir=scratch,
            )
            stage = Stage.VERIFIER_MANIFEST
            manifest = load_manifest(
                verifier_dir, task_type=lease.identity.task_type
            )
            checks_total = len(manifest.checks)

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
                        EvaluationResult(
                            "rejected", failure.reason, (), None,
                            MinerReason(failure.reason), Stage.COMMIT,
                        ),
                    )
                )
            stage = Stage.COMMIT
            expected_format = (
                "unified_diff_v1"
                if lease.identity.task_type == "repository_patch_v1"
                else "bash_script_v1"
            )
            for grant in revealed.submission_grants:
                if grant.format != expected_format:
                    return finish("abandoned", "submission format did not match the task", RoundReason.SUBMISSION_FORMAT_MISMATCH)
                if not _grant_matches_signed_response(grant, signed_responses.get((grant.uid, grant.hotkey))):
                    return finish("abandoned", "submission grant did not match the signed response", RoundReason.SUBMISSION_GRANT_MISMATCH)

            challenge_tag = hashlib.sha256(
                lease.challenge_id.encode("utf-8")
            ).hexdigest()[:12]
            grading_slots = asyncio.Semaphore(policy.grading_concurrency)
            round_dead = asyncio.Event()  # set on the first fault; no new grading starts after it

            async def grade(index: int, grant: ArtifactGrant) -> MinerEvaluation | None:
                async with grading_slots:
                    if round_dead.is_set():
                        return None
                    try:
                        return await grade_one(index, grant)
                    except BaseException:
                        round_dead.set()
                        raise

            async def grade_one(index: int, grant: ArtifactGrant) -> MinerEvaluation:
                registration = (grant.uid, grant.hotkey)
                try:
                    submission_path = await fetch_submission(
                        http,
                        grant,
                        round_dir / f"submission-{index}",
                        policy.submissions,
                        allowed_origins=policy.artifact_origins,
                    )
                    submission_bytes = submission_path.path.read_bytes()
                except Exception:  # noqa: BLE001 - post-commit storage is infrastructure
                    raise _Abandon("submission download failed", RoundReason.SUBMISSION_DOWNLOAD_FAILED, Stage.SUBMISSION_DOWNLOAD) from None
                if revealed.grading_expires_at <= int(time.time()):
                    raise _Abandon("grading window expired", RoundReason.GRADING_EXPIRED, Stage.WORKSPACE_MATERIALIZATION)
                if shutil.disk_usage(round_dir).free < policy.tree.max_total_file_bytes:
                    raise _Abandon("insufficient grading storage", RoundReason.INSUFFICIENT_STORAGE, Stage.WORKSPACE_MATERIALIZATION)
                grading_started = time.monotonic()
                run_prefix = f"v3-{challenge_tag}-{grant.uid}"
                evaluation: MinerEvaluation | None = None
                try:
                    miner_workspace = await asyncio.to_thread(
                        materialize_workspace, cached, round_dir / f"miner-{grant.uid}"
                    )
                except Exception as error:  # noqa: BLE001 - workspace copies are infrastructure
                    raise _Abandon(
                        f"validator failed during workspace materialization ({type(error).__name__})",
                        RoundReason.WORKSPACE_MATERIALIZATION_FAILED, Stage.WORKSPACE_MATERIALIZATION,
                    ) from None
                try:
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
                        raise _Abandon("unsupported task identity", RoundReason.UNSUPPORTED_TASK, Stage.GRADING)
                    grading_duration_ms = min(
                        2**53 - 1,
                        int((time.monotonic() - grading_started) * 1000),
                    )
                    evaluation = MinerEvaluation(
                        grant.uid, grant.hotkey,
                        submissions_by_registration[registration].latency_ms,
                        result, grading_duration_ms,
                        trajectory=signed_responses[registration].trajectory,
                    )
                finally:
                    try:
                        await asyncio.to_thread(_remove_tree, miner_workspace)
                    except Exception as error:  # noqa: BLE001 - a leftover workspace ends the round
                        raise _Abandon(
                            f"validator failed during cleanup ({type(error).__name__})",
                            RoundReason.CLEANUP_FAILED, Stage.CLEANUP, evaluation,
                        ) from None
                if result.status == "abandoned":
                    raise _Abandon(
                        result.reason,
                        result.reason_code if isinstance(result.reason_code, RoundReason) else RoundReason.VALIDATOR_ERROR,
                        result.stage, evaluation,
                    )
                return evaluation

            stage = Stage.GRADING
            outcomes = await asyncio.gather(
                *(grade(index, grant) for index, grant in enumerate(revealed.submission_grants)),
                return_exceptions=True,
            )
            # Every grading task has finished and cleaned up. Graded miners are
            # recorded for diagnostics even when the round is abandoned, and
            # the first grading fault in grant order decides. Grant format and
            # signature mismatches were checked for every grant before any
            # grading started, so they take precedence over grading faults.
            evaluations.extend(
                outcome.evaluation if isinstance(outcome, _Abandon) else outcome
                for outcome in outcomes
                if isinstance(outcome, MinerEvaluation)
                or (isinstance(outcome, _Abandon) and outcome.evaluation is not None)
            )
            for outcome in outcomes:
                if isinstance(outcome, _Abandon):
                    return finish("abandoned", outcome.reason, outcome.code, outcome.stage)
                if isinstance(outcome, BaseException):
                    raise outcome
            stage = Stage.CLEANUP
        stage = Stage.GRADING
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
        return finish("completed")
    except Exception as error:  # noqa: BLE001 - infrastructure faults abandon atomically
        return finish(
            "abandoned",
            f"validator failed during {stage.value.replace('_', ' ')} ({type(error).__name__})",
            {
                Stage.WORKSPACE_DOWNLOAD: RoundReason.WORKSPACE_DOWNLOAD_FAILED,
                Stage.WORKSPACE_EXTRACTION: RoundReason.WORKSPACE_EXTRACTION_FAILED,
                Stage.WORKSPACE_MATERIALIZATION: RoundReason.WORKSPACE_MATERIALIZATION_FAILED,
                Stage.VERIFIER_DOWNLOAD: RoundReason.VERIFIER_DOWNLOAD_FAILED,
                Stage.VERIFIER_EXTRACTION: RoundReason.VERIFIER_INVALID,
                Stage.VERIFIER_MANIFEST: RoundReason.VERIFIER_INVALID,
                Stage.CLEANUP: RoundReason.CLEANUP_FAILED,
            }.get(stage, RoundReason.VALIDATOR_ERROR),
        )
