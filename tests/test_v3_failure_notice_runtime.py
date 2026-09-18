from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rlvr.config import Settings
from rlvr.neurons import decentralized, feedback_sender
from rlvr.scoring.eval_engine import EvalEngine
from rlvr.v3.grading import EvaluationResult
from rlvr.v3.reasons import MinerReason, RoundReason, Stage
from rlvr.v3.round import MinerEvaluation, RoundResult


def _engine():
    engine = EvalEngine(2, 3600, 10, 4, clock=lambda: 1000)
    engine.set_hotkeys({0: "owner", 1: "miner"})
    engine.update({1: 1.0})
    return engine


def test_score_save_success_means_replaced_persisted_state(tmp_path):
    path = tmp_path / "nested" / "scores.json"
    engine = _engine()

    assert decentralized._save_scores(engine, str(path)) is True

    state = json.loads(path.read_text())
    assert state["scores"] == engine.scores.tolist()
    assert state["histories"]["1"] == [[1000.0, 1.0]]
    assert state["hotkeys"]["1"] == "miner"
    assert not path.with_name("scores.json.tmp").exists()


def test_score_save_empty_path_is_unsaved(monkeypatch):
    def unexpected_write(*_args, **_kwargs):
        pytest.fail("disabled persistence must not write")

    monkeypatch.setattr(decentralized.os, "makedirs", unexpected_write)
    assert decentralized._save_scores(_engine(), "") is False


@pytest.mark.parametrize("operation", ["makedirs", "fsync", "replace"])
def test_score_save_failure_retains_previous_state(tmp_path, monkeypatch, operation):
    path = tmp_path / "scores.json"
    previous = b'{"previous":true}\n'
    path.write_bytes(previous)

    def fail(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(decentralized.os, operation, fail)
    assert decentralized._save_scores(_engine(), str(path)) is False
    assert path.read_bytes() == previous


@pytest.fixture
def run_callback(tmp_path, monkeypatch):
    async def run(
        *,
        status="completed",
        persistence=True,
        notices=True,
        details=True,
        diagnostic_result=True,
        sender_error=False,
        real_sender=False,
    ):
        path = tmp_path / "scores.json"
        settings = Settings(
            _env_file=None,
            problem_server_url="https://problems.invalid",
            validator_score_state_file=str(path),
            validator_failure_notices=notices,
            validator_failed_check_details=details,
        )
        passed = MinerEvaluation(
            1, "miner", 10, EvaluationResult("passed", "", (), None)
        )
        failed = MinerEvaluation(
            2,
            "failed-miner",
            10,
            EvaluationResult(
                "failed",
                "mismatch",
                (),
                None,
                reason_code=MinerReason.CHECK_FAILED,
                stage=Stage.CHECK,
                failed_check='Required stdout: "expected"',
            ),
        )
        leased = status != "unavailable"
        outcome = RoundResult(
            status,
            "" if status == "completed" else "unavailable",
            (passed, failed) if status == "completed" else (),
            reason_code=None if status == "completed" else RoundReason.VALIDATOR_ERROR,
            stage=Stage.GRADING,
            challenge_id="challenge" if leased else None,
            task_id="a" * 64 if leased else None,
            assigned_miners=((1, "miner"), (2, "failed-miner")) if leased else (),
            diagnostic_evaluations=(passed, failed) if status == "abandoned" else (),
            checks_total=1 if leased else None,
        )
        events, returned, sends = [], [], []
        solvers = [
            SimpleNamespace(uid=1, hotkey="miner", url="https://miner.invalid"),
            SimpleNamespace(uid=2, hotkey="failed-miner", url="https://failed.invalid"),
        ]

        class Validator:
            def __init__(self, *_args, **_kwargs):
                self.wallet = SimpleNamespace(
                    hotkey=SimpleNamespace(ss58_address="validator")
                )
                self.subtensor = object()
                self.metagraph = SimpleNamespace(
                    hotkeys=["owner", "miner", "failed-miner"], sync=lambda **_: None
                )

            def setup_bittensor(self):
                pass

            def set_round_callback(self, callback):
                self.callback = callback

            def set_weight_setter(self, _setter):
                pass

            async def run(self):
                returned.append(await self.callback(self))

        class Diagnostics:
            def __init__(self, _path):
                pass

            def record_round(self, result, hotkey):
                assert result is outcome
                assert hotkey == "validator"
                assert events == ["saved" if persistence else "unsaved"]
                events.append("diagnostics")
                if diagnostic_result == "exception":
                    raise OSError("diagnostic unavailable")
                return diagnostic_result

        original_save = decentralized._save_scores

        def save(engine, filename):
            saved = original_save(engine, filename) if persistence else False
            assert saved is persistence
            events.append("saved" if saved else "unsaved")
            return saved

        async def evaluate(*_args, **_kwargs):
            return outcome

        async def send(result, clients, **kwargs):
            assert events == ["saved", "diagnostics"]
            assert result is outcome
            assert clients is solvers
            assert json.loads(path.read_text())["scores"][1] == 0.25
            sends.append(kwargs)
            events.append("notice")
            if sender_error:
                raise RuntimeError("delivery unavailable")
            return 1

        monkeypatch.setattr(decentralized, "ValidatorNeuron", Validator)
        monkeypatch.setattr(decentralized, "EvaluationLog", Diagnostics)
        monkeypatch.setattr(decentralized, "_apply_weights_rate_limit", lambda *_: None)
        monkeypatch.setattr(
            decentralized, "v3_round_policy", lambda *_args, **_kwargs: object()
        )
        monkeypatch.setattr(
            decentralized, "_solver_clients", lambda *_args, **_kwargs: solvers
        )
        monkeypatch.setattr(decentralized, "evaluate_round", evaluate)
        monkeypatch.setattr(decentralized, "_save_scores", save)
        monkeypatch.setattr(
            decentralized,
            "send_failure_notices",
            feedback_sender.send_failure_notices if real_sender else send,
        )
        await decentralized._run_decentralized_validator_async(settings)
        assert len(returned) == 1
        return returned[0], events, sends, path

    return run


@pytest.mark.parametrize("diagnostic_result", [True, False, "exception"])
@pytest.mark.parametrize("sender_error", [False, True])
async def test_notices_follow_score_persistence_and_diagnostic_attempt(
    run_callback, diagnostic_result, sender_error
):
    scores, events, sends, path = await run_callback(
        diagnostic_result=diagnostic_result, sender_error=sender_error
    )
    assert events == ["saved", "diagnostics", "notice"]
    assert sends[0]["include_details"] is True
    assert scores == {0: 0.0, 1: 0.25, 2: 0.0}
    assert json.loads(path.read_text())["scores"] == list(scores.values())


@pytest.mark.parametrize("persistence,notices", [(False, True), (True, False)])
async def test_disabled_or_unsaved_notices_do_not_change_scores(
    run_callback, persistence, notices
):
    scores, events, sends, _path = await run_callback(
        persistence=persistence, notices=notices
    )
    assert events == ["saved" if persistence else "unsaved", "diagnostics"]
    assert sends == []
    assert scores == {0: 0.0, 1: 0.25, 2: 0.0}


async def test_reason_only_setting_reaches_sender(run_callback):
    _scores, _events, sends, _path = await run_callback(details=False)
    assert sends[0]["include_details"] is False


@pytest.mark.parametrize("status", ["abandoned", "unavailable"])
async def test_incomplete_round_never_signs_or_transmits_notices(
    run_callback, monkeypatch, status
):
    attempts = []

    def sign(*_args, **_kwargs):
        attempts.append("sign")
        raise RuntimeError("no signing allowed")

    def stream(*_args, **_kwargs):
        attempts.append("request")
        raise RuntimeError("no request allowed")

    monkeypatch.setattr(feedback_sender, "sign_message", sign)
    monkeypatch.setattr(decentralized.httpx.AsyncClient, "stream", stream)
    scores, events, _sends, path = await run_callback(status=status, real_sender=True)
    assert attempts == []
    assert events == ["saved", "diagnostics"]
    assert scores == {0: 0.0, 1: 0.0, 2: 0.0}
    assert json.loads(path.read_text())["histories"] == {}
