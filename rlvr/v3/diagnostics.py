"""Bounded, best-effort local records of V3 evaluation outcomes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .round import RoundResult

MAX_RECORD_BYTES = 4_096
MAX_FILE_BYTES = 8 * 1024 * 1024
BACKUP_COUNT = 3


def _detail(value: str) -> str:
    clean = " ".join(value.split())
    clean = "".join(character for character in clean if character.isprintable())
    return clean.encode("utf-8", "replace")[:512].decode("utf-8", "ignore")


def _code(value) -> str | None:
    return None if value is None else value.value


def round_records(result: RoundResult, validator_hotkey: str):
    if result.challenge_id is None or result.task_id is None:
        return
    common = {
        "schema_version": 1,
        "recorded_at_ms": int(time.time() * 1_000),
        "validator_hotkey": validator_hotkey,
        "challenge_id": result.challenge_id,
        "task_id": result.task_id,
        "round_status": result.status,
        "round_reason_code": _code(result.reason_code),
        "score_effect": "unchanged" if result.status == "abandoned" else "not_reported",
    }

    def record(kind, **fields):
        identity = [validator_hotkey, result.challenge_id, result.task_id, kind]
        if kind == "miner_evaluation":
            identity.append(fields["miner_hotkey"])
        encoded = json.dumps(
            identity, ensure_ascii=False, separators=(",", ":")
        ).encode()
        return {
            **common,
            "record_id": hashlib.sha256(encoded).hexdigest(),
            "record_type": kind,
            **fields,
        }

    yield record(
        "round_outcome",
        stage=_code(result.stage),
        reason=_detail(result.reason),
        assigned_miners=len(result.assigned_miners),
    )
    evaluations = {
        (item.uid, item.hotkey): item
        for item in result.diagnostic_evaluations or result.evaluations
    }
    dispatch = {(uid, hotkey): code for uid, hotkey, code in result.dispatch_failures}
    for uid, hotkey in result.assigned_miners:
        item = evaluations.get((uid, hotkey))
        evaluation = None if item is None else item.result
        checks = () if evaluation is None else evaluation.checks
        executed = sum(check.outcome != "skipped" for check in checks)
        total = result.checks_total
        dispatch_code = dispatch.get((uid, hotkey))
        yield record(
            "miner_evaluation",
            uid=uid,
            miner_hotkey=hotkey,
            status="not_evaluated" if evaluation is None else evaluation.status,
            reason_code=_code(
                dispatch_code if evaluation is None else evaluation.reason_code
            ),
            dispatch_reason_code=_code(dispatch_code),
            stage=("dispatch" if dispatch_code else None)
            if evaluation is None
            else _code(evaluation.stage),
            reason="" if evaluation is None else _detail(evaluation.reason),
            checks_passed=sum(check.outcome == "passed" for check in checks),
            checks_executed=executed,
            checks_total=total,
            checks_skipped=None if total is None else total - executed,
            grading_duration_ms=None if item is None else item.grading_duration_ms,
            response_latency_ms=None if item is None else item.latency_ms,
        )


class EvaluationLog:
    """One writer per validator; consumers deduplicate using stable record IDs."""

    def __init__(
        self,
        path: str,
        *,
        max_file_bytes: int = MAX_FILE_BYTES,
        backup_count: int = BACKUP_COUNT,
    ):
        if (
            type(max_file_bytes) is not int
            or not MAX_RECORD_BYTES <= max_file_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("diagnostic file size is outside its allowed range")
        if type(backup_count) is not int or not 0 <= backup_count <= 8:
            raise ValueError("diagnostic backup count is outside its allowed range")
        self.path = Path(path) if path else None
        self.max_file_bytes = max_file_bytes
        self.backup_count = backup_count

    def _file(self, index: int) -> Path:
        assert self.path is not None
        return self.path if index == 0 else Path(f"{self.path}.{index}")

    def _open(self, path: Path, flags: int):
        descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("diagnostic path must be a regular file")
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _rotate(self) -> None:
        if self.backup_count:
            self._file(self.backup_count).unlink(missing_ok=True)
            for index in range(self.backup_count - 1, -1, -1):
                try:
                    os.replace(self._file(index), self._file(index + 1))
                except FileNotFoundError:
                    pass
        else:
            self._file(0).unlink(missing_ok=True)

    def _append(self, raw: bytes) -> None:
        path = self._file(0)
        descriptor = self._open(path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(descriptor, "r+b") as output:
            size = output.seek(0, os.SEEK_END)
            if size:
                output.seek(max(0, size - MAX_RECORD_BYTES))
                tail = output.read(MAX_RECORD_BYTES)
                if not tail.endswith(b"\n"):
                    last_line = tail.rfind(b"\n")
                    if last_line < 0 and size > MAX_RECORD_BYTES:
                        raise OSError("diagnostic tail could not be repaired")
                    size = size - len(tail) + last_line + 1
                    output.truncate(size)
            if size + len(raw) <= self.max_file_bytes:
                output.seek(size)
                output.write(raw)
                output.flush()
                return
        self._rotate()
        self._append(raw)

    def record_round(self, result: RoundResult, validator_hotkey: str) -> bool:
        if self.path is None or result.challenge_id is None:
            return True
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            for record in round_records(result, validator_hotkey):
                raw = (
                    json.dumps(
                        record, ensure_ascii=True, separators=(",", ":")
                    ).encode()
                    + b"\n"
                )
                if len(raw) > MAX_RECORD_BYTES:
                    record["reason"] = ""
                    raw = (
                        json.dumps(
                            record, ensure_ascii=True, separators=(",", ":")
                        ).encode()
                        + b"\n"
                    )
                if len(raw) > MAX_RECORD_BYTES:
                    raise ValueError("diagnostic record exceeds its limit")
                self._append(raw)
            return True
        except Exception:  # noqa: BLE001 - diagnostics must never change scoring
            return False
