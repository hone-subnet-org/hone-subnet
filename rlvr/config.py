"""Validator configuration.

Problem-source transport, sandbox verification, scoring, and chain settings
belong here. The optional demo miner keeps its model-provider settings in
``rlvr.neurons.demo_miner``.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .policy import RELEASE_POLICY, RELEASE_POLICY_ENV_KEYS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Wall-clock a responder gets per problem. The dispatch path adds a small
    # wire grace period and enforces a true total-duration deadline around the
    # full response; HTTPX's own timeout is only a per-phase inactivity bound.
    # Too-short deadlines zero out honest miners on exactly the in-band-hard
    # problems the curriculum targets.
    solve_deadline_s: float = Field(default=300.0, gt=0.0, le=3600.0)
    # --- Difficulty classification ---
    band_low: float = Field(default=0.35, ge=0.0, le=1.0)
    band_high: float = Field(default=0.65, ge=0.0, le=1.0)
    # Minimum number of responses required before trusting a band placement.
    difficulty_samples_m: int = Field(default=8, ge=1, le=1024)

    # --- Verification / sandbox ---
    # Candidate code is adversarial. Docker is the fail-closed production
    # default; the subprocess backend must be selected explicitly for local
    # development because it cannot block filesystem reads or direct network
    # access on Linux.
    executor: str = Field(
        default="docker",
        pattern="^(subprocess|docker)$",
        description="'subprocess' | 'docker'",
    )
    per_test_timeout_s: float = Field(default=5.0, gt=0.0, le=300.0)
    docker_image: str = Field(
        default="python:3.12-slim",
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:@-]*$",
    )
    docker_memory: str = Field(
        default="256m", pattern=r"^[1-9][0-9]*[kKmMgG]$"
    )
    docker_cpus: float = Field(default=1.0, gt=0.0, le=256.0)
    docker_pids_limit: int = Field(default=128, ge=16, le=4096)
    # Controls reward labels written to exported datasets, not validator weights.
    reward_partial_credit: bool = False

    # --- Private problem-source client ---
    problem_server_url: str = ""
    # HTTPS authenticates problem-server responses; Epistula authenticates
    # validator requests. Plain HTTP is local-test only.
    problem_server_allow_insecure_http: bool = False
    # Exact miner response bodies are committed and archived. Keep this
    # separate from the larger cap for problem-server challenge/reveal
    # responses so full-pool commits remain predictably bounded.
    miner_max_response_bytes: int = Field(default=128_000, gt=0, le=1_000_000)
    problem_server_request_timeout_s: float = Field(default=60.0, gt=0.0)
    problem_server_metagraph_sync_s: float = Field(default=60.0, gt=0.0)
    # HTTP fan-out width when querying miners. I/O-bound: full-pool dispatch
    # needs one in-flight request per serving miner or slow solvers serialize
    # into deadline-length waves.
    validator_dispatch_concurrency: int = Field(default=256, ge=1, le=1024)
    # Requests waiting for one of these short-lived send-start slots remain
    # unsigned. The slot is released after the request body reaches the HTTP
    # transport, so miner solve time does not serialize the fan-out.
    validator_send_concurrency: int = Field(default=32, ge=1, le=1024)
    # Concurrent sandbox verifications (each spawns per-test subprocesses or
    # Docker containers). Bounded separately from the HTTP fan-out so a
    # full-pool dispatch cannot thrash the executor.
    validator_verify_concurrency: int = Field(default=16, ge=1, le=256)
    validator_score_state_file: str = "data/validator_scores.json"
    validator_diagnostics_file: str | None = None

    # --- Dataset export ---
    dataset_dir: str = "data/rollouts"

    # --- Bittensor (live testnet) ---
    netuid: int = Field(default=0, ge=0)
    subtensor_network: str = "test"
    subtensor_chain_endpoint: str = ""
    wallet_name: str = "default"
    wallet_hotkey: str = "default"

    @model_validator(mode="after")
    def validate_ranges(self) -> "Settings":
        if self.band_low > self.band_high:
            raise ValueError("BAND_LOW must be <= BAND_HIGH")
        return self


def get_settings() -> Settings:
    """Load settings from environment / .env."""
    return Settings()


def ignored_release_policy_keys(env_file: str = ".env") -> list[str]:
    """Return obsolete policy overrides present in the environment or env file.

    Values are never read or logged. The keys are reported so an upgraded
    operator can remove stale entries without startup depending on that edit.
    """
    found = {key for key in RELEASE_POLICY_ENV_KEYS if key in os.environ}
    try:
        for raw_line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key.removeprefix("export ").strip()
            if key in RELEASE_POLICY_ENV_KEYS:
                found.add(key)
    except (OSError, UnicodeError):
        pass
    return sorted(found)


def release_policy_summary() -> str:
    """Safe, non-secret summary of consensus-relevant validator policy."""
    return RELEASE_POLICY.summary()


_SAFE_EFFECTIVE_SETTINGS = {
    "docker_cpus": "DOCKER_CPUS",
    "docker_memory": "DOCKER_MEMORY",
    "docker_pids_limit": "DOCKER_PIDS_LIMIT",
    "executor": "EXECUTOR",
    "miner_max_response_bytes": "MINER_MAX_RESPONSE_BYTES",
    "problem_server_request_timeout_s": "PROBLEM_SERVER_REQUEST_TIMEOUT_S",
    "validator_dispatch_concurrency": "VALIDATOR_DISPATCH_CONCURRENCY",
    "validator_send_concurrency": "VALIDATOR_SEND_CONCURRENCY",
    "validator_verify_concurrency": "VALIDATOR_VERIFY_CONCURRENCY",
}


def nondefault_settings_summary(settings: Settings) -> str:
    """Return an allowlisted summary of non-default machine settings."""
    changed = []
    for field_name, label in _SAFE_EFFECTIVE_SETTINGS.items():
        default = Settings.model_fields[field_name].default
        value = getattr(settings, field_name)
        if value != default:
            changed.append(f"{label}={value}")
    return " ".join(changed)
