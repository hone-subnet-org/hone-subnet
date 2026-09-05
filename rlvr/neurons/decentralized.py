"""V3 decentralized validator runtime."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from ..config import Settings, get_settings
from ..policy import (
    LEGACY_SCORE_WINDOW_SECONDS,
    RELEASE_POLICY,
    ValidatorPolicy,
)
from ..problemserver.client import (
    next_lease_not_before,
    require_secure_problem_url,
)
from ..scoring.eval_engine import EvalEngine
from ..v3.client import V3ProblemServerClient
from ..v3.release import round_policy as v3_round_policy
from ..v3.round import apply_round_scores, evaluate_round
from .live import SendGate, _solver_clients
from .validator import ValidatorNeuron

_WEIGHTS_RATE_LIMIT_MARGIN = 20
_MAX_WEIGHT_DETAIL_CHARS = 200
_MAX_WEIGHT_FIELD_CHARS = 80


def _weight_result_status(result: object) -> tuple[bool, str]:
    """Normalize legacy tuple/bool and modern Bittensor extrinsic results."""
    success = getattr(result, "success", None)
    if success is not None:
        return bool(success), str(getattr(result, "message", "") or "")
    if isinstance(result, tuple):
        ok = bool(result[0]) if result else False
        message = str(result[1]) if len(result) > 1 else ""
        return ok, message
    return bool(result), ""


def _weight_failure_detail(result: object) -> str:
    """Return bounded single-line SDK ``error``/``data`` without raising."""

    def render_field(name: str) -> str:
        try:
            value = getattr(result, name)
        except Exception:  # noqa: BLE001 - diagnostics must never break weights
            return ""
        try:
            if value is None or value is False:
                return ""
            if isinstance(value, str):
                rendered = value
            else:
                if hasattr(value, "__len__") and len(value) == 0:
                    return ""
                if isinstance(value, (int, float)) and value == 0:
                    return ""
                rendered = repr(value)
        except Exception:  # noqa: BLE001 - hostile SDK values are non-fatal
            return ""
        rendered = " ".join(rendered.split())
        rendered = "".join(ch for ch in rendered if ch.isprintable())
        if not rendered:
            return ""
        if len(rendered) > _MAX_WEIGHT_FIELD_CHARS:
            rendered = rendered[: _MAX_WEIGHT_FIELD_CHARS - 3] + "..."
        return f"{name}={rendered}"

    parts = [part for name in ("error", "data") if (part := render_field(name))]
    return " ".join(parts)[:_MAX_WEIGHT_DETAIL_CHARS]


def _weights_rate_limited(
    blocks_elapsed: Optional[int], chain_limit: Optional[int]
) -> bool:
    """Mirror the chain's strict admission boundary for diagnostics."""
    return bool(
        blocks_elapsed is not None
        and blocks_elapsed >= 0
        and chain_limit is not None
        and chain_limit > 0
        and blocks_elapsed < chain_limit
    )


def _weight_failure_report(result: object, validator: ValidatorNeuron) -> str:
    """Explain a failed, otherwise silent weight submission without raising."""
    try:
        ok, message = _weight_result_status(result)
    except Exception:  # noqa: BLE001 - diagnostics must not mask submission
        ok, message = False, ""
    if ok or message:
        return ""

    detail = _weight_failure_detail(result)
    if detail:
        return detail

    try:
        current = int(validator.subtensor.get_current_block())
    except Exception:  # noqa: BLE001 - retain the context that is available
        current = None
    try:
        uid = validator.uid
        if isinstance(uid, bool) or not isinstance(uid, int) or uid < 0:
            raise ValueError("invalid validator uid")
        last_update = int(validator.metagraph.last_update[uid])
    except Exception:  # noqa: BLE001 - optional diagnostic context
        last_update = None
    elapsed = (
        current - int(last_update)
        if current is not None and last_update is not None
        else None
    )
    chain_limit = getattr(validator, "chain_weights_rate_limit", None)
    effective = getattr(validator, "weights_interval_blocks", None)
    report = (
        f"current_block={current if current is not None else 'unknown'} "
        f"last_update_block={last_update if last_update is not None else 'unknown'} "
        f"blocks_elapsed={elapsed if elapsed is not None else 'unknown'} "
        f"chain_limit={chain_limit if chain_limit is not None else 'unknown'} "
        f"effective_interval={effective if effective is not None else 'unknown'}"
    )
    if elapsed is not None and chain_limit is not None:
        report += f" rate_limited={_weights_rate_limited(elapsed, chain_limit)}"
    return report


def _read_weights_rate_limit(
    validator: ValidatorNeuron, netuid: int
) -> Optional[int]:
    """Read a trustworthy positive chain limit, or fall back with a warning."""
    try:
        reader = getattr(validator.subtensor, "weights_rate_limit")
        value = reader(netuid)
    except Exception as error:  # noqa: BLE001 - startup must retain safe fallback
        print(
            "[validator] WARN: could not read chain weights rate limit; "
            f"using configured interval ({error})"
        )
        return None
    if type(value) is not int or value <= 0:
        print(
            "[validator] WARN: chain returned an invalid weights rate limit; "
            "using configured interval"
        )
        return None
    return value


def effective_weights_interval(
    configured: int, chain_limit: Optional[int]
) -> int:
    """Apply the chain-derived rate-limit floor plus its fixed safety margin."""
    configured_interval = int(configured)
    if type(chain_limit) is not int or chain_limit <= 0:
        return configured_interval
    return max(
        configured_interval,
        chain_limit + _WEIGHTS_RATE_LIMIT_MARGIN,
    )


def _apply_weights_rate_limit(
    validator: ValidatorNeuron,
    settings: Settings,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> None:
    """Install the startup chain-derived cadence and retain its source value."""
    chain_limit = _read_weights_rate_limit(validator, settings.netuid)
    validator.chain_weights_rate_limit = chain_limit
    validator.weights_interval_blocks = effective_weights_interval(
        policy.weights_interval_blocks,
        chain_limit,
    )


def _validator_http_limits(settings: Settings) -> httpx.Limits:
    """Size the connection pool for a true single-wave full-pool dispatch."""
    dispatch = max(1, int(settings.validator_dispatch_concurrency))
    return httpx.Limits(
        # Reserve a few slots for the problem server and metagraph turnover
        # while every configured miner-dispatch permit is in use.
        max_connections=dispatch + 8,
        max_keepalive_connections=min(dispatch, 64),
    )


def _weight_observation_count(engine: EvalEngine) -> int:
    """Largest authoritative per-uid history available for weight evidence."""
    return max((len(history) for history in engine.histories.values()), default=0)


def _submit_local_weights(
    validator: ValidatorNeuron,
    engine: EvalEngine,
    settings: Settings,
    *,
    log_response_repr: bool = False,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> Optional[bool]:
    """Submit normalized local weights only after enough completed evidence."""
    observations = _weight_observation_count(engine)
    required = policy.min_weight_observations
    if observations < required:
        print(
            "[validator] local weight evidence "
            f"{observations}/{required}; skipping submission"
        )
        return None

    weights = engine.get_weights(n=len(validator.metagraph.hotkeys))

    if float(sum(weights)) <= 0.0:
        print("[validator] local miner weights are all-zero; skipping")
        return None
    uids = [int(uid) for uid in validator.metagraph.uids]
    result = validator.subtensor.set_weights(
        wallet=validator.wallet,
        netuid=settings.netuid,
        uids=uids,
        weights=[float(weight) for weight in weights],
        wait_for_inclusion=True,
        wait_for_finalization=False,
        wait_for_revealed_execution=False,
        max_attempts=1,
    )
    ok, message = _weight_result_status(result)
    failure_detail = _weight_failure_report(result, validator)
    top = max(range(len(weights)), key=lambda idx: weights[idx])
    response_detail = ""
    if log_response_repr:
        response_repr = repr(result)
        if len(response_repr) > 500:
            response_repr = response_repr[:497] + "..."
        response_detail = f" response={response_repr}"
    print(
        "[validator] set local weights "
        f"response_type={type(result).__name__}{response_detail} "
        f"ok={ok} msg={message!r} "
        f"(top uid={uids[top]} w={float(weights[top]):.4f})"
        f"{f' failure_detail={failure_detail}' if failure_detail else ''}"
    )
    return ok


def _load_scores(engine: EvalEngine, path: str) -> None:
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)

        version = state.get("version")
        if version not in (1, 2):
            raise ValueError("unsupported score-state version")

        if version == 1:
            scores = np.asarray(state.get("scores", []), dtype=np.float64)
            if (
                scores.ndim != 1
                or not np.all(np.isfinite(scores))
                or np.any(scores < 0.0)
            ):
                raise ValueError("scores must be a finite non-negative vector")
            num_uids = int(scores.size)
        else:
            num_uids = state.get("num_uids")
            if (
                isinstance(num_uids, bool)
                or not isinstance(num_uids, int)
                or num_uids < 0
            ):
                raise ValueError("num_uids must be a non-negative integer")

        hotkeys = {
            int(uid): str(hotkey) for uid, hotkey in state.get("hotkeys", {}).items()
        }
        if any(
            uid < 0 or uid >= num_uids or not hotkey
            for uid, hotkey in hotkeys.items()
        ):
            raise ValueError("hotkey map must reference non-empty in-range UID slots")

        if version == 1:
            if any(scores[uid] > 0.0 and uid not in hotkeys for uid in range(num_uids)):
                raise ValueError("every positive score must retain its owning hotkey")
            # A scalar EMA cannot reconstruct its observations. Filling the new
            # window with that scalar preserves both the current score and the
            # intended one-slot-per-observation sensitivity during migration.
            migrated_at = engine._timestamp()
            histories = {
                uid: [(migrated_at, float(scores[uid]))] * engine.min_samples
                for uid in hotkeys
                if scores[uid] > 0.0
            }
        else:
            stored_window_seconds = state.get("window_seconds")
            if (
                isinstance(stored_window_seconds, bool)
                or not isinstance(stored_window_seconds, (int, float))
                or not np.isfinite(stored_window_seconds)
                or stored_window_seconds <= 0.0
            ):
                raise ValueError("window_seconds must be finite and positive")
            stored_max = state.get("max_samples")
            stored_min = state.get("min_samples")
            if (
                isinstance(stored_max, bool)
                or not isinstance(stored_max, int)
                or stored_max < 1
                or isinstance(stored_min, bool)
                or not isinstance(stored_min, int)
                or not 1 <= stored_min <= stored_max
            ):
                raise ValueError("stored sample bounds are invalid")
            raw_histories = state.get("histories")
            if not isinstance(raw_histories, dict):
                raise ValueError("histories must be a UID-keyed mapping")
            histories: dict[int, list[tuple[float, float]]] = {}
            for raw_uid, raw_values in raw_histories.items():
                uid = int(raw_uid)
                if uid in histories or uid < 0 or uid >= num_uids:
                    raise ValueError("history UID must be unique and in range")
                if (
                    not isinstance(raw_values, list)
                    or not raw_values
                    or len(raw_values) > stored_max
                ):
                    raise ValueError(
                        "each history must contain 1..max_samples observations"
                    )
                values: list[tuple[float, float]] = []
                for item in raw_values:
                    if not isinstance(item, list) or len(item) != 2:
                        raise ValueError(
                            "history observations must be [timestamp, payment]"
                        )
                    raw_timestamp, raw_payment = item
                    if isinstance(raw_timestamp, bool) or isinstance(
                        raw_payment, bool
                    ):
                        raise ValueError("history values must be numeric")
                    timestamp = float(raw_timestamp)
                    payment = float(raw_payment)
                    if (
                        not np.isfinite(timestamp)
                        or timestamp < 0.0
                        or not np.isfinite(payment)
                        or payment < 0.0
                    ):
                        raise ValueError(
                            "history values must be finite and non-negative"
                        )
                    values.append((timestamp, payment))
                histories[uid] = values

            # Histories are authoritative. Stored scores are for operators, so
            # malformed or stale derived values are repaired rather than
            # discarding valid histories and suppressing weights for hours.
            stored_scores = state.get("scores")
            scores_match = False
            try:
                scores = np.asarray(stored_scores, dtype=np.float64)
                if (
                    scores.ndim == 1
                    and scores.size == num_uids
                    and np.all(np.isfinite(scores))
                ):
                    expected = np.zeros(num_uids, dtype=np.float64)
                    for uid, values in histories.items():
                        expected[uid] = sum(
                            payment for _, payment in values
                        ) / max(len(values), stored_min)
                    scores_match = bool(
                        np.allclose(scores, expected, rtol=0.0, atol=1e-9)
                    )
            except (TypeError, ValueError):
                pass
            if not scores_match:
                print(
                    "[validator] WARN: persisted derived scores were stale; "
                    "recomputed from histories"
                )

            # A configured cap change intentionally keeps only the latest
            # completed-problem observations.
            histories = {
                uid: values[-engine.max_samples :]
                for uid, values in histories.items()
            }
            if any(
                sum(payment for _, payment in values) > 0.0
                and uid not in hotkeys
                for uid, values in histories.items()
            ):
                raise ValueError(
                    "every positive history must retain its owning hotkey"
                )

        # Apply only after the entire file validates. A partially parsed state
        # must not restore histories without the hotkeys needed to detect reuse.
        engine._restore(num_uids, histories, hotkeys)
        suffix = " (migrated v1)" if version == 1 else ""
        print(f"[validator] restored local scores for {num_uids} UIDs{suffix}")
        # Make legacy migration one-shot even if the validator crashes before
        # its first round callback has a chance to persist normal score state.
        if version == 1:
            _save_scores(engine, path)
    except FileNotFoundError:
        return
    except Exception as e:  # noqa: BLE001 - corrupt state starts safely at zero
        print(f"[validator] WARN: could not restore local scores ({e})")


def _save_scores(engine: EvalEngine, path: str) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": 2,
                    "num_uids": int(engine.scores.size),
                    "window_seconds": engine.window_seconds,
                    "max_samples": engine.max_samples,
                    "min_samples": engine.min_samples,
                    "scores": [float(score) for score in engine.scores],
                    "histories": {
                        str(uid): list(history)
                        for uid, history in sorted(engine.histories.items())
                    },
                    "hotkeys": {str(uid): hk for uid, hk in engine.hotkeys.items()},
                },
                fh,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as e:
        print(f"[validator] WARN: could not persist local scores ({e})")


async def _run_decentralized_validator_async(settings: Settings) -> None:
    require_secure_problem_url(
        settings.problem_server_url,
        settings.problem_server_allow_insecure_http,
    )
    policy = RELEASE_POLICY
    validator = ValidatorNeuron(settings, policy=policy)
    validator.setup_bittensor()
    _apply_weights_rate_limit(validator, settings, policy)
    engine = EvalEngine(
        len(validator.metagraph.hotkeys),
        LEGACY_SCORE_WINDOW_SECONDS,
        policy.score_window_max_samples,
        policy.score_window_min_samples,
        decay=policy.decay_nonresponders,
    )
    _load_scores(engine, settings.validator_score_state_file)
    grading_policy = v3_round_policy(
        policy, dispatch_concurrency=settings.validator_dispatch_concurrency
    )
    state_dir = Path(settings.validator_score_state_file).parent

    async with httpx.AsyncClient(limits=_validator_http_limits(settings)) as http:
        client = V3ProblemServerClient(
            settings.problem_server_url,
            validator.wallet,
            http,
            allow_insecure_http=settings.problem_server_allow_insecure_http,
            timeout_s=settings.problem_server_request_timeout_s,
            max_response_bytes=policy.v3_problem_response_read_bytes,
        )
        dispatch_policy_logged = False
        send_gate = SendGate(settings.validator_send_concurrency)

        async def round_callback(v: ValidatorNeuron) -> dict[int, float]:
            nonlocal dispatch_policy_logged
            await asyncio.to_thread(v.metagraph.sync, subtensor=v.subtensor)
            engine.resize(len(v.metagraph.hotkeys))
            engine.sync({uid: hk for uid, hk in enumerate(v.metagraph.hotkeys)})
            live_solvers = _solver_clients(v, v.wallet, settings, http, gate=send_gate)
            if not live_solvers:
                return {}
            if not dispatch_policy_logged:
                print(
                    "[validator] V3 dispatch uses the server-assigned slot pool; "
                    f"serving miners={len(live_solvers)}"
                )
                dispatch_policy_logged = True
            result = await evaluate_round(
                client,
                http,
                live_solvers,
                grading_policy,
                cache_dir=state_dir / "v3-workspace-cache",
                work_dir=state_dir / "v3-rounds",
            )
            completed = int(result.status == "completed")
            if result.status == "completed":
                apply_round_scores(
                    result,
                    engine,
                    active_hotkeys={solver.uid: solver.hotkey for solver in live_solvers},
                    speed_half_life_ms=policy.payment_speed_half_life_ms,
                    speed_floor=policy.payment_speed_floor,
                )
            elif result.status == "unavailable" and result.retry_after_s is not None:
                v.defer_rounds_until(next_lease_not_before(result.retry_after_s))
            elif result.status == "abandoned":
                print(f"[validator] WARN: V3 round abandoned ({result.reason})")
            _save_scores(engine, settings.validator_score_state_file)
            print(f"[validator] locally evaluated {completed} challenges")
            return {uid: float(score) for uid, score in enumerate(engine.scores)}

        weight_response_logged = False

        def weight_setter(v: ValidatorNeuron) -> None:
            nonlocal weight_response_logged
            result = _submit_local_weights(
                v,
                engine,
                settings,
                log_response_repr=not weight_response_logged,
            )
            if result is not None:
                weight_response_logged = True

        validator.set_round_callback(round_callback)
        validator.set_weight_setter(weight_setter)
        await validator.run()


def run_decentralized_validator(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if not settings.problem_server_url:
        raise SystemExit("V3 decentralized evaluation requires PROBLEM_SERVER_URL")
    asyncio.run(_run_decentralized_validator_async(settings))
