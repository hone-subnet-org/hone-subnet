from __future__ import annotations

import json
import subprocess

import httpx
import pytest

from rlvr.neurons.demo_miner import (
    DemoMiner,
    DemoMinerSettings,
    BedrockClient,
    ModelCompletion,
    build_model_messages,
    extract_submission,
)
from rlvr.v3.api import MinerTaskRequest, MinerTaskResponse
from rlvr.v3.trajectory import parse_trajectory
from tests.test_v3_api import lease, slot_set


class WalletLike:
    class Hotkey:
        def __init__(self, address: str):
            self.ss58_address = address

    def __init__(self, address: str):
        self.hotkey = self.Hotkey(address)


def settings(**updates) -> DemoMinerSettings:
    return DemoMinerSettings(_env_file=None, bedrock_api_key="key", **updates)


def task(hotkey="miner") -> MinerTaskRequest:
    value = lease()
    return MinerTaskRequest(
        protocol_version=3,
        challenge_id=value["challenge_id"],
        task_id=value["task_id"],
        identity=value["identity"],
        workspace=value["workspace"],
        workspace_url=value["workspace_url"],
        expires_at=2**53 - 1,
        slots=slot_set(hotkey=hotkey),
    )


def test_prompt_and_submission_are_v3_shapes():
    messages = build_model_messages(task())
    assert "Git unified diff" in messages[0]["content"]
    assert messages[1]["content"] == "Fix it"
    assert extract_submission("```diff\n--- a/a\n+++ b/a\n```") == b"--- a/a\n+++ b/a\n"
    assert extract_submission("```diff\n context line \n```") == b" context line \n"
    assert extract_submission("```diff\n+value\r\n```") == b"+value\r\n"


def test_extracted_repository_patch_remains_git_applicable(tmp_path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    patch = extract_submission(
        "```diff\n"
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```"
    )
    completed = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=tmp_path,
        input=patch,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


async def test_provider_response_records_exact_bodies_and_five_alternatives():
    token = {
        "token": "x",
        "bytes": [120],
        "token_id": 7,
        "logprob": -0.1,
        "top_logprobs": [
            {"token": value, "bytes": [ord(value)], "token_id": index, "logprob": -index}
            for index, value in enumerate("abcde")
        ],
    }

    async def handler(request):
        body = json.loads(request.content)
        assert body["logprobs"] is True and body["top_logprobs"] == 5
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "x", "reasoning_content": "r"}, "logprobs": {"content": [token]}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BedrockClient(
        settings(bedrock_base_url="https://provider.invalid"), http=http
    )
    result = await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert result.generated_bytes == b"x"
    assert len(result.tokens[0]["top_logprobs"]) == 5
    assert json.loads(result.request_body)["model"] == "moonshotai.kimi-k2.5"


async def test_demo_uploads_submission_and_genuine_canonical_trajectory(monkeypatch):
    completion = ModelCompletion(
        request_body=b"{}",
        response_body=b'{"ok":true}',
        output="```diff\n\n```",
        reasoning="No change is needed.",
        generated_bytes=b"x",
        tokens=[{
            "token_bytes_b64": "eA==",
            "token_id": None,
            "logprob": 0,
            "top_logprobs": [
                {"token_bytes_b64": "eA==", "token_id": None, "logprob": 0}
            ] * 5,
        }],
    )

    class Provider:
        async def complete(self, messages, *, timeout_s):
            return completion

        async def aclose(self):
            return None

    captured = {}

    async def upload(_http, request, submission, trajectory, *, allowed_origins):
        captured.update(submission=submission, trajectory=trajectory, origins=allowed_origins)
        return MinerTaskResponse(
            protocol_version=3,
            challenge_id=request.challenge_id,
            task_id=request.task_id,
            response_type=request.identity.task_type,
            submission={
                "artifact_role": "patch",
                "artifact_format": "unified_diff_v1",
                "upload_id": request.slots.submission.upload_id,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "size_bytes": 0,
            },
            trajectory={
                "artifact_role": "trajectory",
                "artifact_format": "trajectory_v1",
                "upload_id": request.slots.trajectory.upload_id,
                "sha256": "1" * 64,
                "size_bytes": len(trajectory),
            },
        )

    monkeypatch.setattr("rlvr.neurons.demo_miner.upload_miner_result", upload)
    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    response = await miner.solve(task(), 30)
    trajectory = parse_trajectory(captured["trajectory"])
    assert response.response_type == "repository_patch_v1"
    assert captured["submission"] == b""
    assert trajectory.miner_hotkey == "miner"
    assert trajectory.events[-1].submission_sha256 == trajectory.submission_sha256


async def test_bedrock_client_requires_https():
    client = BedrockClient(settings(bedrock_base_url="http://provider.invalid"))
    try:
        with pytest.raises(RuntimeError, match="HTTPS"):
            _ = client.completion_url
    finally:
        await client.aclose()
