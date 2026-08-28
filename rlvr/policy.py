"""Versioned validator policy shipped with each release.

These values affect validator scoring or scheduling and are intentionally not
loaded from operator environment files. Tests may inject another frozen policy
into internal components; the production entrypoint always uses
``RELEASE_POLICY``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ValidatorPolicy:
    version: str = "2026-08-10"
    challenges_per_round: int = 1
    verification_runs: int = 1
    dispatch_fraction: float = 0.5
    payment_speed_half_life_ms: float = 180_000.0
    payment_speed_floor: float = 0.95
    owner_burn_share: float = 0.0
    score_window_max_samples: int = 200
    score_window_min_samples: int = 4
    min_weight_observations: int = 4
    decay_nonresponders: bool = True
    round_interval_blocks: int = 38
    weights_interval_blocks: int = 180
    rust_image: str = (
        "ghcr.io/hone-subnet-org/hone-rust-sandbox@sha256:"
        "bc05667bf1dacdbeb1491f15aacfeacc99ef3607f3b65274d62a509a003625fa"
    )
    rustc_version: str = "1.89.0"
    rust_edition: str = "2021"
    rustc_flags: tuple[str, ...] = ("-C", "opt-level=2")
    rust_judge_version: str = "rust-exact-token-v1"
    rust_compile_timeout_s: float = 60.0
    rust_artifact_max_bytes: int = 10_000_000
    rust_memory: str = "2g"
    rust_tmpfs_bytes: int = 256 * 1024 * 1024
    problem_response_read_bytes: int = 2 * 1024 * 1024
    v3_artifact_origins: tuple[str, ...] = (
        "https://rlvr-agentic-v3-private-pilot-artifactbucket-upncfmevdb38.s3.us-east-1.amazonaws.com:443",
    )
    v3_execution_profile_id: str = "repo-polyglot-v1"
    v3_image: str = (
        "public.ecr.aws/t3h1r6x1/hone-subnet/polyglot-sandbox@sha256:"
        "87f7ea823a2ffde124040db0f271e59110e666afecfc68106c4b07fddb4eee08"
    )
    v3_memory_bytes: int = 4 * 1024**3
    v3_cpus: int = 2
    v3_pids_limit: int = 256
    v3_tmpfs_bytes: int = 1024**3
    v3_workspace_compressed_bytes: int = 2 * 1024**3
    v3_workspace_bytes: int = 10 * 1024**3
    v3_verifier_compressed_bytes: int = 512 * 1024**2
    v3_verifier_expanded_bytes: int = 2 * 1024**3
    v3_max_file_bytes: int = 2 * 1024**3
    v3_patch_bytes: int = 1024**2
    v3_script_bytes: int = 1024**2
    v3_problem_response_read_bytes: int = 32 * 1024**2

    def __post_init__(self) -> None:
        if not 0.0 < self.dispatch_fraction <= 1.0:
            raise ValueError("dispatch_fraction must be in (0, 1]")
        if not 0.0 <= self.payment_speed_floor <= 1.0:
            raise ValueError("payment_speed_floor must be in [0, 1]")
        if self.owner_burn_share != 0.0:
            raise ValueError("owner_burn_share must be zero")
        if self.payment_speed_half_life_ms <= 0.0:
            raise ValueError("payment_speed_half_life_ms must be positive")
        if not 1 <= self.score_window_min_samples <= self.score_window_max_samples:
            raise ValueError("score sample bounds are invalid")
        if not 1 <= self.min_weight_observations <= self.score_window_max_samples:
            raise ValueError("min_weight_observations is outside the score window")
        if self.challenges_per_round < 1:
            raise ValueError("challenges_per_round must be positive")
        if self.verification_runs < 1:
            raise ValueError("verification_runs must be positive")
        if self.round_interval_blocks < 0 or self.weights_interval_blocks < 0:
            raise ValueError("block intervals must be non-negative")
        if "@sha256:" not in self.rust_image:
            raise ValueError("rust_image must be digest-pinned")
        if self.rust_edition not in {"2021", "2024"}:
            raise ValueError("unsupported Rust edition")
        if self.rust_compile_timeout_s <= 0 or self.rust_artifact_max_bytes <= 0:
            raise ValueError("Rust compile limits must be positive")
        if self.rust_tmpfs_bytes <= 0 or self.problem_response_read_bytes <= 0:
            raise ValueError("Rust resource limits must be positive")
        if not self.v3_artifact_origins or any(
            not origin.startswith("https://") for origin in self.v3_artifact_origins
        ):
            raise ValueError("V3 artifact origins must be pinned HTTPS origins")
        if "@sha256:" not in self.v3_image:
            raise ValueError("V3 image must be digest-pinned")
        for value in (
            self.v3_memory_bytes,
            self.v3_cpus,
            self.v3_pids_limit,
            self.v3_tmpfs_bytes,
            self.v3_workspace_compressed_bytes,
            self.v3_workspace_bytes,
            self.v3_verifier_compressed_bytes,
            self.v3_verifier_expanded_bytes,
            self.v3_max_file_bytes,
            self.v3_patch_bytes,
            self.v3_script_bytes,
            self.v3_problem_response_read_bytes,
        ):
            if type(value) is not int or value <= 0:
                raise ValueError("V3 resource limits must be positive integers")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def summary(self) -> str:
        return (
            f"version={self.version} hash={self.fingerprint} "
            f"protocol=3 execution_profile={self.v3_execution_profile_id} "
            f"score_samples={self.score_window_max_samples} "
            f"startup_samples={self.score_window_min_samples} "
            f"round_blocks={self.round_interval_blocks} "
            f"weight_blocks={self.weights_interval_blocks} "
            f"speed_floor={self.payment_speed_floor:g} "
            f"speed_half_life_ms={self.payment_speed_half_life_ms:g} "
            f"owner_burn={self.owner_burn_share:g}"
        )


RELEASE_POLICY = ValidatorPolicy()

# Version-2 score files retain this inactive value for rollback compatibility.
LEGACY_SCORE_WINDOW_SECONDS = 57_600.0

RELEASE_POLICY_ENV_KEYS = frozenset(
    {
        "DISPATCH_SUBSET_FRACTION",
        "DISPATCH_SUBSET_K",
        "DETERMINISM_RUNS",
        "PAYMENT_SPEED_FLOOR",
        "PAYMENT_SPEED_HALF_LIFE_MS",
        "PROBLEM_MAX_RESPONSE_BYTES",
        "ROUND_INTERVAL_BLOCKS",
        "SCORE_DECAY_NONRESPONDERS",
        "SCORE_WINDOW_MAX_SAMPLES",
        "SCORE_WINDOW_MIN_SAMPLES",
        "SCORE_WINDOW_SECONDS",
        "VALIDATOR_CHALLENGES_PER_ROUND",
        "VALIDATOR_MIN_WEIGHT_OBSERVATIONS",
        "WEIGHTS_INTERVAL_BLOCKS",
    }
)
