from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rlvr.config import Settings
from rlvr.neurons import decentralized
from rlvr.v3.diagnostics import EvaluationLog
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.reasons import RoundReason, Stage
from rlvr.v3.round import MinerEvaluation, RoundResult


def test_empty_environment_setting_disables_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("VALIDATOR_DIAGNOSTICS_FILE", "")
    settings = Settings(_env_file=None)
    assert settings.validator_diagnostics_file == ""
    sink = EvaluationLog(settings.validator_diagnostics_file)
    assert sink.path is None
    assert sink.record_round(
        RoundResult("completed", "", (), challenge_id="challenge", task_id="a" * 64),
        "validator",
    )


@pytest.mark.parametrize("status", ["completed", "abandoned", "unavailable"])
@pytest.mark.parametrize("logging_failure", [None, "write", "exception"])
async def test_diagnostics_follow_persistence_and_cannot_interrupt_callback(
    tmp_path, monkeypatch, status, logging_failure
):
    score_file = tmp_path / "scores.json"
    diagnostic_file = tmp_path / "v3_evaluations.jsonl"
    settings = Settings(
        _env_file=None,
        problem_server_url="https://problems.invalid",
        validator_score_state_file=str(score_file),
    )
    candidate = MinerEvaluation(
        1, "miner", 10, EvaluationResult("passed", "", (), None)
    )
    leased = status != "unavailable"
    outcome = RoundResult(
        status,
        "" if status == "completed" else "unavailable",
        (candidate,) if status == "completed" else (),
        reason_code=None if status == "completed" else RoundReason.VALIDATOR_ERROR,
        stage=Stage.GRADING,
        challenge_id="challenge" if leased else None,
        task_id="a" * 64 if leased else None,
        assigned_miners=((1, "miner"),) if leased else (),
        diagnostic_evaluations=(candidate,) if status == "abandoned" else (),
        checks_total=0 if leased else None,
    )
    returned = []
    log_calls = []

    class Validator:
        def __init__(self, *_args, **_kwargs):
            self.wallet = SimpleNamespace(
                hotkey=SimpleNamespace(ss58_address="validator")
            )
            self.subtensor = object()
            self.metagraph = SimpleNamespace(
                hotkeys=["owner", "miner"], sync=lambda **_: None
            )

        def setup_bittensor(self):
            pass

        def set_round_callback(self, callback):
            self.callback = callback

        def set_weight_setter(self, _setter):
            pass

        async def run(self):
            returned.append(await self.callback(self))

    class Log(EvaluationLog):
        def record_round(self, result, hotkey):
            assert score_file.exists()
            state = json.loads(score_file.read_text())
            assert bool(state["histories"]) == (status == "completed")
            log_calls.append((result, hotkey))
            if logging_failure == "exception":
                raise RuntimeError("diagnostic formatting failure")
            return super().record_round(result, hotkey)

    async def evaluate(*_args, **_kwargs):
        return outcome

    if logging_failure == "write":
        diagnostic_file.mkdir()
    monkeypatch.setattr(decentralized, "ValidatorNeuron", Validator)
    monkeypatch.setattr(decentralized, "EvaluationLog", Log)
    monkeypatch.setattr(decentralized, "_apply_weights_rate_limit", lambda *_: None)
    monkeypatch.setattr(
        decentralized, "v3_round_policy", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        decentralized,
        "_solver_clients",
        lambda *_args, **_kwargs: [SimpleNamespace(uid=1, hotkey="miner")],
    )
    monkeypatch.setattr(decentralized, "evaluate_round", evaluate)

    await decentralized._run_decentralized_validator_async(settings)

    assert log_calls == [(outcome, "validator")]
    assert len(returned) == 1
    assert returned[0][1] == (0.25 if status == "completed" else 0)
    if logging_failure is None:
        if not leased:
            assert not diagnostic_file.exists()
        else:
            records = [
                json.loads(line) for line in diagnostic_file.read_text().splitlines()
            ]
            assert len(records) == 2
            assert records[1]["score_effect"] == (
                "not_reported" if status == "completed" else "unchanged"
            )
