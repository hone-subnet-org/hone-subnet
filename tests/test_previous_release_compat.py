"""The previous release's miner and validator keep working with this one.

The previous release's code is taken from git and run in its own process, so
its caps, its receiver and its sender are the real ones, not a model of them.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from rlvr import protocol
from rlvr.neurons import demo_miner
from rlvr.neurons.feedback_sender import send_failure_notices
from rlvr.v3.round import RoundResult
from tests.test_demo_miner_feedback import make_receiver
from tests.test_failure_sender import DISPLAY, evaluation, solver

REPO_ROOT = Path(__file__).resolve().parents[1]
# The last commit of the previous release, before this one changed the notice.
PREVIOUS_RELEASE = "d5786a9"

OLD_PROCESS = r'''
import asyncio, base64, json, sys, time
from pathlib import Path

from rlvr import protocol
from rlvr.neurons import demo_miner, feedback_sender
from rlvr.v3.api import serialize_failure_notice
from rlvr.v3.reasons import MinerReason, Stage
from tests.test_demo_miner_feedback import make_receiver

for module in (protocol, demo_miner, feedback_sender):
    assert Path(module.__file__).is_relative_to(Path.cwd()), module.__file__
assert Path(make_receiver.__code__.co_filename).is_relative_to(Path.cwd())
protocol._HAVE_CRYPTO = False
printed = []
demo_miner.print_feedback = printed.append
miner = make_receiver()

for line in sys.stdin:
    message = json.loads(line)
    if message["op"] == "receive":
        body = base64.b64decode(message["body"])
        doc = json.loads(body)
        miner.served_tasks.add(
            "validator", doc["challenge_id"], doc["task_id"], doc["uid"], doc["hotkey"], time.monotonic()
        )
        status, _ = asyncio.run(miner.handle_failure_notice(message["headers"], body))
        reply = {"status": status, "printed": printed[:]}
        printed.clear()
    elif message["op"] == "schemas":
        from rlvr.v3 import api
        reply = {name: getattr(api, name).model_json_schema() for name in message["models"]}
    else:  # what the previous release's validator sends for this outcome
        notice = feedback_sender.build_notice(
            message["challenge"], "a" * 64, 7, "miner", message["status"],
            MinerReason(message["code"]) if message["code"] else None,
            Stage.CHECK, message["display"], include_details=True,
        )
        body = serialize_failure_notice(notice)
        headers = protocol.sign_message("validator", body, signed_for="miner")
        reply = {"headers": headers, "body": base64.b64encode(body).decode()}
    print(json.dumps(reply), flush=True)
'''


@pytest.fixture(scope="module")
def previous_release(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    known = subprocess.run(
        ["git", "cat-file", "-e", f"{PREVIOUS_RELEASE}^{{commit}}"],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    assert known.returncode == 0, f"fetch history: commit {PREVIOUS_RELEASE} is not in this clone"
    tree = tmp_path_factory.mktemp("previous-release")
    archive = subprocess.run(
        ["git", "archive", PREVIOUS_RELEASE, "rlvr", "tests"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    (tree / "old_process.py").write_text(OLD_PROCESS, encoding="utf-8")
    stderr_log = tree / "stderr.log"
    with stderr_log.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [sys.executable, "old_process.py"],
            cwd=tree,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )

    def ask(message):
        assert process.poll() is None, stderr_log.read_text(encoding="utf-8")
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        reply = []
        reader = threading.Thread(target=lambda: reply.append(process.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout=30)
        assert reply and reply[0], "the previous release's process gave no answer:\n" + stderr_log.read_text(encoding="utf-8")
        return json.loads(reply[0])

    try:
        yield ask
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(autouse=True)
def offline_signatures(monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)


_challenges = iter(range(1, 1_000))


def round_result(item):
    """One completed round for one miner, under a challenge id not seen before."""
    return RoundResult(
        "completed", "", (item,), challenge_id=f"challenge-{next(_challenges)}",
        task_id="a" * 64, assigned_miners=((item.uid, item.hotkey),),
    )


async def send_to_previous_release(previous_release, result):
    """Run this release's sender against the previous release's receiver."""
    exchanges = []

    async def handler(request):
        body = await request.aread()
        headers = {key.decode(): value.decode() for key, value in request.headers.raw}
        reply = previous_release({"op": "receive", "headers": headers, "body": base64.b64encode(body).decode()})
        exchanges.append((json.loads(body)["failure"]["failed_check"], reply["status"], reply["printed"]))
        return httpx.Response(reply["status"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        delivered = await send_failure_notices(
            round_result(result), [solver(http, 7, "miner")], wallet="validator", http=http, include_details=True
        )
    return delivered, exchanges


@pytest.mark.parametrize("display_size", [4_000, 10_000], ids=["over-the-old-display-cap", "over-the-old-body-cap"])
async def test_previous_miner_still_gets_the_reason_for_a_big_display(previous_release, display_size):
    display = "Inline script:\n" + "x" * display_size
    delivered, exchanges = await send_to_previous_release(previous_release, evaluation(7, "miner", display=display))
    assert delivered == 1
    assert [(shown is not None, status) for shown, status, _ in exchanges] == [(True, 400 if display_size < 8_000 else 413), (False, 200)]
    printed = exchanges[-1][2]
    assert len(printed) == 1 and printed[0][0].startswith("[demo-miner] feedback: check_failed")
    assert all("x" * 50 not in line for line in printed[0])


async def test_previous_miner_accepts_a_small_display_first_time(previous_release):
    delivered, exchanges = await send_to_previous_release(previous_release, evaluation(7, "miner", display=DISPLAY))
    assert delivered == 1
    assert [(status, len(printed)) for _, status, printed in exchanges] == [(200, 1)]
    assert any(DISPLAY.split("\n")[0] in line for line in exchanges[0][2][0])


async def test_previous_miner_ignores_a_pass_and_still_takes_the_next_failure(previous_release):
    delivered, exchanges = await send_to_previous_release(
        previous_release, evaluation(7, "miner", status="passed", display=None)
    )
    assert (delivered, [(status, printed) for _, status, printed in exchanges]) == (0, [(400, [])])
    delivered, exchanges = await send_to_previous_release(previous_release, evaluation(7, "miner", display=None))
    assert delivered == 1
    assert exchanges[0][1] == 200 and exchanges[0][2][0][0].startswith("[demo-miner] feedback: check_failed")


@pytest.mark.parametrize(
    "status,code,display",
    [("failed", "check_failed", DISPLAY), ("failed", "timeout", None), ("rejected", "patch_rejected", None), ("failed", None, None)],
)
async def test_this_miner_accepts_what_the_previous_validator_sends(previous_release, monkeypatch, status, code, display):
    printed = []
    monkeypatch.setattr(demo_miner, "print_feedback", printed.append)
    miner = make_receiver()
    reply = previous_release({
        "op": "send", "challenge": f"old-{next(_challenges)}", "status": status, "code": code, "display": display,
    })
    body = base64.b64decode(reply["body"])
    doc = json.loads(body)
    miner.served_tasks.add("validator", doc["challenge_id"], doc["task_id"], doc["uid"], doc["hotkey"], time.monotonic())
    assert await miner.handle_failure_notice(reply["headers"], body) == (200, {"accepted": True})
    assert len(printed) == 1
    expected = code or "evaluation_failed"
    assert printed[0][0].startswith(f"[demo-miner] feedback: {expected}")
    assert (len(printed[0]) > 1) is (display is not None)


TASK_WIRE_MODELS = [
    "MinerCandidate",
    "LeaseRequest",
    "LeaseResponse",
    "MinerTaskRequest",
    "MinerTaskResponse",
    "MinerSubmission",
    "ChallengeCommitRequest",
    "CommitRevealResponse",
]


def test_task_wire_models_are_unchanged_since_the_previous_release(previous_release):
    from rlvr.v3 import api

    old = previous_release({"op": "schemas", "models": TASK_WIRE_MODELS + ["MinerFailureNotice"]})
    for name in TASK_WIRE_MODELS:
        new = getattr(api, name).model_json_schema()
        if name == "LeaseRequest":
            # The lease request grew one optional field, the pool size; a server
            # that does not know it treats the request as the previous release's.
            assert new["properties"].pop("miners_per_task")["default"] is None
            assert "miners_per_task" not in new.get("required", [])
        assert new == old[name], name
    # The notice is identical except that the reason code grew by exactly "passed".
    new = api.MinerFailureNotice.model_json_schema()
    old_notice = old["MinerFailureNotice"]
    new_reason = new["$defs"]["FailureExplanation"]["properties"].pop("reason_code")
    old_reason = old_notice["$defs"]["FailureExplanation"]["properties"].pop("reason_code")
    new["$defs"]["FailureExplanation"].pop("description", None)  # a docstring, not wire
    assert new == old_notice
    assert literals(new_reason) == literals(old_reason) | {"passed"}


def literals(schema):
    """Every literal string a schema fragment allows, whatever shape pydantic gave it."""
    found = set()
    if isinstance(schema, dict):
        if "const" in schema:
            found.add(schema["const"])
        found.update(schema.get("enum", []))
        for value in schema.values():
            found |= literals(value)
    elif isinstance(schema, list):
        for value in schema:
            found |= literals(value)
    return found
