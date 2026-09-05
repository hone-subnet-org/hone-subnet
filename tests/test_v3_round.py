from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat

import httpx
import pytest

from rlvr.scoring.eval_engine import EvalEngine
from rlvr.v3.api import (
    EPISTULA_HEADERS,
    ChallengeFeedbackRequest,
    ChallengeFeedbackResponse,
    CommitRevealResponse,
    LeaseResponse,
    MinerSubmission,
    MinerTaskResponse,
    derive_miner_request_id,
)
from rlvr.v3.archive import ArchiveLimits
from rlvr.v3.artifacts import ArtifactGrant, ArtifactRef
from rlvr.v3.client import V3ProblemServerClient
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.identity import compute_task_id
from rlvr.v3.patch import PatchLimits
from rlvr.v3.round import (
    MinerEvaluation,
    RoundPolicy,
    RoundResult,
    _remove_tree,
    _send_diagnostic_feedback,
    apply_round_scores,
    compute_round_payments,
    evaluate_round,
)
from rlvr.v3.script import ScriptLimits
from rlvr.v3.submission import SubmissionLimits
from rlvr.v3.supervisor import SupervisorPolicy
from rlvr.v3.tree import TreeLimits
from tests.test_v3_api import lease, slot_set
from tests.test_v3_archive import compress, make_tar


@pytest.mark.parametrize("locked_mode", [0o000, 0o555])
def test_remove_tree_handles_locked_directories_without_following_symlinks(
    tmp_path, locked_mode
):
    victim = tmp_path / "victim"
    victim.write_bytes(b"secret")
    victim.chmod(0o600)
    root = tmp_path / "hone-v3-round-test"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (locked / "file").write_bytes(b"x")
    os.symlink(victim, locked / "link")
    locked.chmod(locked_mode)

    _remove_tree(root)

    assert not root.exists()
    assert victim.read_bytes() == b"secret"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o600


def test_remove_tree_handles_deep_directories(tmp_path):
    root = tmp_path / "hone-v3-round-deep"
    root.mkdir()
    original = os.open(".", os.O_RDONLY)
    try:
        os.chdir(root)
        for _ in range(1_500):
            os.mkdir("a")
            os.chdir("a")
    finally:
        os.fchdir(original)
        os.close(original)

    _remove_tree(root)
    assert not root.exists()


def test_feedback_is_skipped_without_grants_and_failures_are_diagnostic():
    class Client:
        calls = 0

        async def feedback(self, request):
            self.calls += 1
            raise RuntimeError("offline")

    client = Client()
    assert asyncio.run(
        _send_diagnostic_feedback(client, "challenge", "a" * 64, [], [])
    )
    assert client.calls == 0

    evaluation = MinerEvaluation(
        7,
        "hk-7",
        1,
        EvaluationResult("passed", "", (), None),
        2,
    )
    assert not asyncio.run(
        _send_diagnostic_feedback(
            client,
            "challenge",
            "a" * 64,
            [
                ArtifactGrant(
                    uid=7,
                    hotkey="hk-7",
                    upload_id="upload",
                    sha256="b" * 64,
                    size_bytes=1,
                    format="unified_diff_v1",
                    read_url="https://uploads.invalid/read",
                )
            ],
            [evaluation],
        )
    )


class Solver:
    def __init__(self, uid, hotkey, content):
        self.uid = uid
        self.hotkey = hotkey
        self.content = content

    async def solve_v3(self, task):
        response = MinerTaskResponse(
            protocol_version=3,
            challenge_id=task.challenge_id,
            task_id=task.task_id,
            response_type="repository_patch_v1",
            submission={
                "artifact_role": "patch",
                "artifact_format": "unified_diff_v1",
                "upload_id": task.slots.submission.upload_id,
                "sha256": hashlib.sha256(self.content).hexdigest(),
                "size_bytes": len(self.content),
            },
            trajectory={
                "artifact_role": "trajectory",
                "artifact_format": "trajectory_v1",
                "upload_id": task.slots.trajectory.upload_id,
                "sha256": hashlib.sha256(b"trajectory").hexdigest(),
                "size_bytes": len(b"trajectory"),
            },
        )
        body = response.model_dump_json()
        return MinerSubmission(
            uid=self.uid,
            hotkey=self.hotkey,
            request_id=derive_miner_request_id(task.challenge_id, self.uid, self.hotkey),
            response_body=body,
            response_headers={name: "value" for name in EPISTULA_HEADERS},
            error="",
            latency_ms=10,
        ), response


class Stream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        yield self.content


def streamed(content):
    return httpx.Response(200, stream=Stream(content))


def artifact_ref(role, tar_bytes, compressed):
    return ArtifactRef(
        artifact_role=role,
        artifact_format="tar_zst_v1",
        sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_size_bytes=len(compressed),
        expanded_size_bytes=len(tar_bytes),
    )


def policy(tmp_path):
    limits = ArchiveLimits(1 << 20, 1 << 20, 1 << 20, 100, 1024, 8 << 20)
    return RoundPolicy(
        workspace_archive=limits,
        verifier_archive=limits,
        submissions=SubmissionLimits(1024, 1024),
        tree=TreeLimits(100, 1 << 20, 1 << 20, 1024),
        patch=PatchLimits(1024, 5),
        script=ScriptLimits(1024),
        supervisor=SupervisorPolicy(
            image="registry.invalid/profile@sha256:" + "a" * 64,
            candidate_uid=65534,
            candidate_gid=65534,
            memory_bytes=256 << 20,
            cpus=1,
            pids_limit=64,
            tmpfs_bytes=64 << 20,
            max_file_bytes=1 << 20,
            watchdog_slack_s=2,
        ),
        docker_binary=str(tmp_path / "docker"),
        artifact_origins=frozenset({"https://uploads.invalid:443"}),
        execution_profile_id="repo-polyglot-v1",
        verifier_policy="command-gold-digest-v1",
        dispatch_concurrency=4,
    )


@pytest.mark.parametrize("submission_download_fails", [False, True])
@pytest.mark.parametrize("checker_times_out", [False, True])
def test_complete_synthetic_round_has_pass_fail_malformed_and_no_response(
    tmp_path, monkeypatch, submission_download_fails, checker_times_out
):
    workspace_tar = make_tar([("repo", "dir", b"", 0o755), ("repo/a.txt", "file", b"a", 0o644)])
    workspace_blob = compress(workspace_tar)
    manifest = {
        "manifest_version": 1,
        "task_type": "repository_patch_v1",
        "setup": None,
        "checks": [{
            "check_id": "check",
            "kind": "inspection",
            "argv": ["/usr/bin/true"],
            "timeout_s": 1,
            "max_stdout_bytes": 100,
            "max_stderr_bytes": 100,
            "expect": {"exit_code": 0, "stdout": "gold/out", "stderr": None},
        }],
    }
    verifier_tar = make_tar([
        ("manifest.json", "file", json.dumps(manifest).encode(), 0o644),
        ("checks", "dir", b"", 0o755),
        ("gold", "dir", b"", 0o755),
        ("gold/out", "file", b"", 0o644),
    ])
    verifier_blob = compress(verifier_tar)
    contents = {1: b"pass", 2: b"fail", 3: b"malformed"}
    workspace_ref = artifact_ref("workspace", workspace_tar, workspace_blob)
    verifier_ref = artifact_ref("verifier", verifier_tar, verifier_blob)
    base = lease()
    task_identity = base["identity"].model_copy(
        update={
            "workspace_sha256": workspace_ref.sha256,
            "verifier_sha256": verifier_ref.sha256,
        }
    )
    task_id = compute_task_id(task_identity)
    slots = []
    for uid in range(1, 5):
        current = slot_set(uid=uid, hotkey=f"hk-{uid}", prefix=f"u{uid}")
        slots.append(current.model_validate({
            "submission": current.submission.model_dump() | {"task_id": task_id},
            "trajectory": current.trajectory.model_dump() | {"task_id": task_id},
        }))
    lease_fields = lease(
        identity=task_identity,
        task_id=task_id,
        slot_pool=slots,
        commit_min_signed_responses=3,
        workspace=workspace_ref,
        verifier=verifier_ref,
        expires_at=2**53 - 1,
    )
    leased = LeaseResponse(**lease_fields)
    feedback_requests = []

    async def handler(request):
        path = request.url.path
        if request.method == "POST" and path == "/v3/challenges/lease":
            return httpx.Response(200, content=leased.model_dump_json())
        if request.method == "POST" and path == "/v3/challenges/commit":
            grants = [
                {
                    "uid": uid,
                    "hotkey": f"hk-{uid}",
                    "upload_id": f"u{uid}-submission",
                    "sha256": hashlib.sha256(contents[uid]).hexdigest(),
                    "size_bytes": len(contents[uid]),
                    "format": "unified_diff_v1",
                    "read_url": f"https://uploads.invalid/submission-{uid}",
                }
                for uid in contents
            ]
            response = CommitRevealResponse(
                protocol_version=3,
                challenge_id=leased.challenge_id,
                task_id=leased.task_id,
                verifier=verifier_ref,
                verifier_policy="command-gold-digest-v1",
                verifier_url="https://uploads.invalid/verifier",
                grading_expires_at=2**53 - 1,
                submission_grants=grants,
                artifact_failures=[{"uid": 4, "hotkey": "hk-4", "reason": "artifact_invalid"}],
            )
            return httpx.Response(200, content=response.model_dump_json())
        if request.method == "POST" and path == "/v3/challenges/feedback":
            feedback_requests.append(
                ChallengeFeedbackRequest.model_validate_json(await request.aread())
            )
            response = ChallengeFeedbackResponse(
                protocol_version=3,
                challenge_id=leased.challenge_id,
                task_id=leased.task_id,
            )
            return httpx.Response(200, content=response.model_dump_json())
        if path == "/workspace":
            return streamed(workspace_blob)
        if path == "/verifier":
            return streamed(verifier_blob)
        if path.startswith("/submission-"):
            return streamed(contents[int(path.rsplit("-", 1)[1])])
        return httpx.Response(404)

    def fake_grade(workspace, patch, *args, **kwargs):
        if patch == b"pass":
            return EvaluationResult("passed", "", (), None)
        if patch == b"fail":
            if checker_times_out:
                from rlvr.v3.grading import _run_checks
                from rlvr.v3.supervisor import ContainerResult

                monkeypatch.setattr(
                    "rlvr.v3.grading.run_container",
                    lambda *_args: ContainerResult(124, False, True, b"", b"", False, False),
                )
                _, manifest, verifier, supervisor, tree, _, docker, prefix = args
                return _run_checks(
                    workspace=workspace, host_work_base=workspace, result_tree=workspace,
                    work_base="/work", manifest=manifest, verifier_dir=verifier,
                    supervisor_policy=supervisor, tree_limits=tree, docker_binary=docker,
                    run_prefix=prefix, script_exit_code=None,
                )
            return EvaluationResult("failed", "tests failed", (), None)
        return EvaluationResult("rejected", "patch was rejected", (), None)

    monkeypatch.setattr("rlvr.v3.round.evaluate_repository", fake_grade)
    if submission_download_fails:
        async def fail_download(*_args, **_kwargs):
            raise httpx.ConnectError("storage unavailable")

        monkeypatch.setattr("rlvr.v3.round.fetch_submission", fail_download)
    stale_cache = tmp_path / "cache" / ("f" * 64)
    stale_cache.mkdir(parents=True)
    (stale_cache / "old").write_bytes(b"old")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = V3ProblemServerClient("https://problems.invalid", "validator", http, retries=1)
            return await evaluate_round(
                client,
                http,
                [Solver(uid, f"hk-{uid}", contents[uid]) for uid in contents],
                policy(tmp_path),
                cache_dir=tmp_path / "cache",
                work_dir=tmp_path / "work",
            )

    result = asyncio.run(go())
    if submission_download_fails:
        assert result.status == "abandoned"
        assert result.reason == "submission download failed"
        assert feedback_requests == []
        return
    assert result.reason == ""
    assert result.status == "completed"
    assert [item.result.status for item in result.evaluations] == [
        "passed", "failed", "rejected", "rejected"
    ]
    assert not stale_cache.exists()
    assert len(feedback_requests) == 1
    assert {item.uid for item in feedback_requests[0].verdicts} == {1, 2, 3}
    assert {item.uid: item.passed for item in feedback_requests[0].verdicts} == {
        1: True,
        2: False,
        3: False,
    }
    assert all(
        type(item.grading_duration_ms) is int and item.grading_duration_ms >= 0
        for item in feedback_requests[0].verdicts
    )
    payments = compute_round_payments(
        result, speed_half_life_ms=180_000, speed_floor=0.95
    )
    assert payments[1] == 1.0
    assert all(payments[uid] == 0.0 for uid in (2, 3, 4))
    engine = EvalEngine(6, 1, 200, 4)
    assert apply_round_scores(
        result,
        engine,
        active_hotkeys={uid: f"hk-{uid}" for uid in (1, 2, 3, 4, 5)},
        speed_half_life_ms=180_000,
        speed_floor=0.95,
    )
    assert engine.histories[1][-1][1] == 1.0
    assert all(engine.histories[uid][-1][1] == 0.0 for uid in (2, 3, 4))


def test_scoring_ignores_stale_or_unknown_registrations():
    engine = EvalEngine(4, 1, 200, 4)
    engine.set_hotkeys({1: "current-1", 2: "current-2"})
    result = RoundResult(
        "completed",
        "",
        (
            MinerEvaluation(1, "stale-1", 10, EvaluationResult("passed", "", (), None)),
            MinerEvaluation(9, "unknown", 10, EvaluationResult("passed", "", (), None)),
        ),
    )
    assert apply_round_scores(
        result,
        engine,
        active_hotkeys={1: "current-1", 2: "current-2"},
        speed_half_life_ms=180_000,
        speed_floor=0.95,
    )
    assert engine.hotkeys[1] == "current-1"
    assert not engine.histories
