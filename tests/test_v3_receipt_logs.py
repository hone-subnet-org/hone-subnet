"""Optional receipt records must not displace required evaluation records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rlvr.scoring.eval_engine import EvalEngine
from rlvr.v3 import diagnostics
from rlvr.v3.diagnostics import EvaluationLog, round_records
from rlvr.v3.grading import CheckResult, EvaluationResult
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.round import MinerEvaluation, RoundResult, apply_round_scores
from tests.test_v3_receipts import make_receipt


def receipt_round(*, abandoned=False, rich=False):
    receipt = make_receipt(
        **(
            {
                "expected_stdout": b"a" * 900,
                "stdout": b"b" * 900,
                "stdin": b"i" * 900,
                "stderr": b"e" * 900,
            }
            if rich
            else {}
        )
    )
    receipt = replace(receipt, checks_total=1)
    failed = EvaluationResult(
        "failed",
        "candidate output did not match the verifier",
        (CheckResult("selected", "invocation", "failed", 0),),
        None,
        MinerReason.CHECK_FAILED,
        Stage.CHECK,
        receipt=receipt,
    )
    passed = EvaluationResult(
        "passed",
        "",
        (CheckResult("selected", "invocation", "passed", 0),),
        None,
        stage=Stage.CHECK,
    )
    evaluations = (
        MinerEvaluation(1, "hk-1", 10, failed, 5),
        MinerEvaluation(2, "hk-2", 20, failed, 6),
        MinerEvaluation(3, "hk-3", 30, passed, 7),
    )
    return RoundResult(
        "abandoned" if abandoned else "completed",
        "cleanup failed" if abandoned else "",
        () if abandoned else evaluations,
        reason_code=RoundReason.CLEANUP_FAILED if abandoned else None,
        stage=Stage.CLEANUP if abandoned else Stage.GRADING,
        challenge_id="challenge-1",
        task_id="a" * 64,
        assigned_miners=((1, "hk-1"), (2, "hk-2"), (3, "hk-3")),
        diagnostic_evaluations=evaluations,
        checks_total=1,
    )


def log_records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def test_receipts_are_separate_records_with_distinct_stable_miner_ids(monkeypatch):
    result = receipt_round()
    monkeypatch.setattr(diagnostics.time, "time", lambda: 1000)
    records = list(round_records(result, "validator"))
    assert [record["record_type"] for record in records] == [
        "round_outcome",
        "miner_evaluation",
        "failure_receipt",
        "miner_evaluation",
        "failure_receipt",
        "miner_evaluation",
    ]
    receipts = [
        record for record in records if record["record_type"] == "failure_receipt"
    ]
    assert [record["uid"] for record in receipts] == [1, 2]
    assert len({record["record_id"] for record in records}) == len(records)
    for record in receipts:
        assert record["schema_version"] == 1
        assert (
            record["validator_hotkey"],
            record["challenge_id"],
            record["task_id"],
        ) == ("validator", "challenge-1", "a" * 64)
        assert record["miner_hotkey"] == f"hk-{record['uid']}"
        assert (
            record["round_status"] == "completed"
            and record["score_effect"] == "not_reported"
        )
        assert (
            record["receipt"]
            == result.evaluations[record["uid"] - 1].result.receipt.to_record()
        )
    assert all(
        "receipt" not in record
        for record in records
        if record["record_type"] != "failure_receipt"
    )
    monkeypatch.setattr(diagnostics.time, "time", lambda: 2000)
    replay = list(round_records(result, "validator"))
    assert [record["record_id"] for record in replay] == [
        record["record_id"] for record in records
    ]
    assert replay[0]["recorded_at_ms"] != records[0]["recorded_at_ms"]


@pytest.mark.parametrize("identity", ["validator", "challenge", "task", "miner"])
def test_receipt_id_changes_with_each_identity_component(identity):
    result = receipt_round()
    before = next(
        record["record_id"]
        for record in round_records(result, "validator")
        if record["record_type"] == "failure_receipt" and record["uid"] == 1
    )
    validator = "validator"
    if identity == "validator":
        validator = "other-validator"
    elif identity == "challenge":
        result = replace(result, challenge_id="other-challenge")
    elif identity == "task":
        result = replace(result, task_id="b" * 64)
    else:
        evaluations = (
            replace(result.evaluations[0], hotkey="replacement"),
            *result.evaluations[1:],
        )
        result = replace(
            result,
            evaluations=evaluations,
            diagnostic_evaluations=evaluations,
            assigned_miners=((1, "replacement"), (2, "hk-2"), (3, "hk-3")),
        )
    after = next(
        record["record_id"]
        for record in round_records(result, validator)
        if record["record_type"] == "failure_receipt" and record["uid"] == 1
    )
    assert after != before


def test_receipts_do_not_modify_existing_required_record_fields(monkeypatch):
    monkeypatch.setattr(diagnostics.time, "time", lambda: 1000)
    result = receipt_round()
    plain_evaluations = tuple(
        replace(item, result=replace(item.result, receipt=None))
        for item in result.evaluations
    )
    plain = replace(
        result, evaluations=plain_evaluations, diagnostic_evaluations=plain_evaluations
    )
    required = [
        record
        for record in round_records(result, "validator")
        if record["record_type"] != "failure_receipt"
    ]
    assert required == list(round_records(plain, "validator"))


def test_abandoned_round_keeps_receipts_as_diagnostics_without_scoring(tmp_path):
    result = receipt_round(abandoned=True)
    engine = EvalEngine(5, 1, 200, 4)
    engine.update({3: 1.0}, hotkeys={3: "hk-3"}, dispatched={3})
    before = repr(engine.histories)
    assert not apply_round_scores(
        result,
        engine,
        active_hotkeys={1: "hk-1", 2: "hk-2", 3: "hk-3"},
        speed_half_life_ms=180_000,
        speed_floor=0.95,
    )
    assert result.evaluations == () and repr(engine.histories) == before
    path = tmp_path / "evaluations.jsonl"
    assert EvaluationLog(str(path)).record_round(result, "validator")
    receipts = [
        record
        for record in log_records(path)
        if record["record_type"] == "failure_receipt"
    ]
    assert len(receipts) == 2
    assert all(
        record["round_status"] == "abandoned" and record["score_effect"] == "unchanged"
        for record in receipts
    )
    assert all(record["round_reason_code"] == "cleanup_failed" for record in receipts)
    assert repr(engine.histories) == before


@pytest.mark.parametrize("fault", ["conversion", "serialization", "oversize"])
def test_bad_receipt_does_not_prevent_later_required_or_optional_records(
    tmp_path, monkeypatch, fault
):
    result = receipt_round()
    first = result.evaluations[0]
    changed = replace(
        first,
        result=replace(
            first.result, receipt=replace(first.result.receipt, check_id="bad")
        ),
    )
    evaluations = (changed, *result.evaluations[1:])
    result = replace(
        result, evaluations=evaluations, diagnostic_evaluations=evaluations
    )
    receipt_type = type(changed.result.receipt)
    original = receipt_type.to_record

    def faulty_record(receipt):
        if receipt.check_id != "bad":
            return original(receipt)
        if fault == "conversion":
            raise RuntimeError("optional receipt conversion failed")
        record = original(receipt)
        record["argv"] = (
            {"unserializable"} if fault == "serialization" else ["x" * 5000]
        )
        return record

    monkeypatch.setattr(receipt_type, "to_record", faulty_record)
    path = tmp_path / "evaluations.jsonl"
    assert EvaluationLog(str(path)).record_round(result, "validator") is False
    records = log_records(path)
    assert [
        record["uid"]
        for record in records
        if record["record_type"] == "miner_evaluation"
    ] == [1, 2, 3]
    assert [
        record["uid"]
        for record in records
        if record["record_type"] == "failure_receipt"
    ] == [2]
    assert records[0]["record_type"] == "round_outcome"
    assert all(
        len(line) <= 4096 for line in path.read_bytes().splitlines(keepends=True)
    )


def test_whole_line_cap_includes_identity_envelope_and_newline(tmp_path):
    result = receipt_round(rich=True)
    validator = "v" * 1800
    proposed = list(round_records(result, validator))
    encode = lambda record: (
        json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    assert all(
        len(encode(record)) <= 4096
        for record in proposed
        if record["record_type"] != "failure_receipt"
    )
    assert any(
        len(encode(record)) > 4096
        for record in proposed
        if record["record_type"] == "failure_receipt"
    )
    path = tmp_path / "evaluations.jsonl"
    assert EvaluationLog(str(path)).record_round(result, validator) is False
    records = log_records(path)
    assert [
        record["uid"]
        for record in records
        if record["record_type"] == "miner_evaluation"
    ] == [1, 2, 3]
    assert all(
        len(line) <= 4096 for line in path.read_bytes().splitlines(keepends=True)
    )


def test_receipt_lines_obey_rotation_and_repair_an_interrupted_write(tmp_path):
    path = tmp_path / "evaluations.jsonl"
    sink = EvaluationLog(str(path), max_file_bytes=4096, backup_count=2)
    result = receipt_round(rich=True)
    assert sink.record_round(result, "validator")
    with path.open("ab") as output:
        output.write(b'{"record_type":"failure_receipt","broken":')
    for index in range(1, 6):
        assert sink.record_round(
            replace(result, challenge_id=f"challenge-{index + 1}"), "validator"
        )
    files = list(tmp_path.iterdir())
    assert 1 <= len(files) <= 3
    records = []
    for file in files:
        assert file.stat().st_size <= 4096
        assert file.read_bytes().endswith(b"\n")
        records.extend(log_records(file))
        assert b'"broken"' not in file.read_bytes()
    assert any(record["record_type"] == "failure_receipt" for record in records)
    assert any(
        record["record_type"] == "miner_evaluation"
        and record["uid"] == 3
        and record["challenge_id"] == "challenge-6"
        for record in records
    )
