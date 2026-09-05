from __future__ import annotations

import base64
import json
import subprocess
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from rlvr.neurons.demo_miner import (
    BedrockClient,
    DemoMiner,
    DemoMinerSettings,
    ModelCompletion,
    build_model_messages,
    extract_submission,
)
from rlvr.v3.api import MinerTaskRequest, MinerTaskResponse
from rlvr.v3.miner_workspace import WorkspaceReader
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


async def test_demo_uploads_submission_and_genuine_canonical_trajectory(monkeypatch, tmp_path):
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
        calls = 0

        async def complete(self, messages, *, timeout_s):
            self.calls += 1
            if self.calls == 1:
                assert "Task working directory" in messages[-1]["content"]
                return replace(
                    completion,
                    output='```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```',
                    request_body=json.dumps({"messages": messages}).encode(),
                )
            assert "existing repository contents" in messages[-1]["content"]
            return replace(completion, request_body=json.dumps({"messages": messages}).encode())

        async def aclose(self):
            return None

    captured = {}

    (tmp_path / "a.txt").write_text("existing repository contents")

    @asynccontextmanager
    async def workspace(*_args):
        yield WorkspaceReader(tmp_path)

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
    monkeypatch.setattr("rlvr.neurons.demo_miner.open_miner_workspace", workspace)
    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    response = await miner.solve(task(), 30)
    trajectory = parse_trajectory(captured["trajectory"])
    assert response.response_type == "repository_patch_v1"
    assert captured["submission"] == b""
    assert trajectory.miner_hotkey == "miner"
    assert trajectory.events[-1].submission_sha256 == trajectory.submission_sha256
    turns = [event for event in trajectory.events if event.event_type == "model_turn"]
    assert len(turns) == 2
    assert b"existing repository contents" in base64.b64decode(turns[-1].request_body_b64)
    assert [event.sequence for event in trajectory.events] == list(range(len(trajectory.events)))
    reads = [event for event in trajectory.events if event.event_type == "tool_result"]
    assert b"existing repository contents" in base64.b64decode(reads[0].output_body_b64)


async def test_bedrock_client_requires_https():
    client = BedrockClient(settings(bedrock_base_url="http://provider.invalid"))
    try:
        with pytest.raises(RuntimeError, match="HTTPS"):
            _ = client.completion_url
    finally:
        await client.aclose()


def completion_for(output):
    encoded = base64.b64encode(output.encode()).decode()
    return ModelCompletion(
        request_body=b"{}", response_body=b"{}", output=output, reasoning="Inspect the file.",
        generated_bytes=output.encode(),
        tokens=[{
            "token_bytes_b64": encoded, "token_id": None, "logprob": 0,
            "top_logprobs": [{"token_bytes_b64": encoded, "token_id": None, "logprob": 0}] * 5,
        }],
    )


@pytest.mark.parametrize("keep_calling", [False, True])
async def test_workspace_tool_budget_allows_one_final_turn(tmp_path, keep_calling):
    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s):
            self.calls += 1
            if self.calls == 2:
                assert "Tool budget exhausted" in messages[-1]["content"]
                if not keep_calling:
                    return completion_for("```diff\n\n```")
            return completion_for('```workspace\n{"tool":"list_files","path":".","offset":0}\n```')

    import time

    provider = Provider()
    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), provider)
    if keep_calling:
        with pytest.raises(ValueError, match="tool call limit"):
            await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    else:
        completion, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
        assert extract_submission(completion.output) == b""
    assert provider.calls == 2


async def test_workspace_reads_share_one_model_deadline(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", SimpleNamespace(monotonic=lambda: clock[0]))
    budgets = []

    class Provider:
        async def complete(self, messages, *, timeout_s):
            budgets.append(timeout_s)
            clock[0] += 25
            if len(budgets) == 1:
                return completion_for('```workspace\n{"tool":"read_file","path":"missing","offset":0}\n```')
            assert "error" in messages[-1]["content"]
            return completion_for("```diff\n\n```")

    miner = DemoMiner(settings(), Provider())
    await miner._generate(task(), WorkspaceReader(tmp_path), 200)
    assert budgets == [100, 75]
    with pytest.raises(TimeoutError):
        await miner._generate(task(), WorkspaceReader(tmp_path), 100)
    assert len(budgets) == 2
