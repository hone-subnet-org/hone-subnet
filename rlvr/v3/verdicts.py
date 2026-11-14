"""A validator's round of verdicts, signed so other validators can pool it.

The signed message is a domain tag plus the canonical JSON of the round: the
challenge, the task, the validator's own round sequence number, and for each
miner its uid, hotkey, verdict and response latency, sorted by uid. A round
is signed whole, so it is pooled whole or not at all, and the sequence number
makes a withheld round visible as a gap.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..protocol import sign_detached, verify_detached

ROUND_DOMAIN = b"hone-v3-round-v1:"


def _signed_verdict(item: Any) -> dict[str, Any]:
    uid, hotkey, passed, latency = item.uid, item.hotkey, item.passed, item.response_latency_ms
    if type(uid) is not int or type(hotkey) is not str or type(passed) is not bool:
        raise TypeError("a verdict is signed over an int uid, a str hotkey and a bool")
    if latency is not None and type(latency) is not int:
        raise TypeError("a verdict latency is an int or None")
    return {"hotkey": hotkey, "passed": passed, "response_latency_ms": latency, "uid": uid}


def round_message(challenge_id: str, task_id: str, round_seq: int, verdicts: Iterable[Any]) -> bytes:
    """The exact bytes a validator signs for one round."""
    if type(challenge_id) is not str or type(task_id) is not str or type(round_seq) is not int:
        raise TypeError("a round is signed over str identifiers and an int sequence number")
    signed = sorted((_signed_verdict(item) for item in verdicts), key=lambda item: item["uid"])
    if len({item["uid"] for item in signed}) != len(signed):
        raise ValueError("a round names each uid once")
    body = {"challenge_id": challenge_id, "round_seq": round_seq, "task_id": task_id, "verdicts": signed}
    return ROUND_DOMAIN + json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_round(signer: Any, challenge_id: str, task_id: str, round_seq: int, verdicts: Iterable[Any]) -> str:
    return sign_detached(signer, round_message(challenge_id, task_id, round_seq, verdicts))


def verify_round(
    validator_hotkey: str, signature: str, challenge_id: str, task_id: str, round_seq: int, verdicts: Iterable[Any]
) -> bool:
    try:
        message = round_message(challenge_id, task_id, round_seq, verdicts)
    except (TypeError, ValueError):
        return False
    return verify_detached(validator_hotkey, message, signature)
