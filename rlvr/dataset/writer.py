"""RolloutWriter: build solver prompts, convert ChallengeResults to Rollouts,
and append them to sharded JSONL files.

The on-disk format is one JSON object per line (JSONL). Files are sharded so a
single shard never grows unbounded: once a shard reaches `shard_size` records,
the writer rolls over to the next shard. Every write APPENDS (never truncates),
so a process restart resumes the current shard.
"""

from __future__ import annotations

import json
import os

from rlvr.types import (
    ChallengeResult,
    MinerOutcome,
    Problem,
    Rollout,
)

# Records per JSONL shard before rolling over to a new file.
DEFAULT_SHARD_SIZE = 1000
DATASET_MAX_BYTES = 2 * 1024**3
SHARD_PREFIX = "rollouts"
# The problem store: statement + hidden tests + measured difficulty, one record
# per evaluated problem-turn. It is stored only by the validator because it
# contains evaluation cases, but it never contains an implementation.
PROBLEM_PREFIX = "problems"


class RolloutWriter:
    """Persists Rollouts as appended, sharded JSONL."""

    def __init__(
        self,
        dataset_dir: str,
        shard_size: int = DEFAULT_SHARD_SIZE,
        max_bytes: int = DATASET_MAX_BYTES,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.shard_size = max(1, int(shard_size))
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        os.makedirs(self.dataset_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Prompt construction — EXACTLY what the solver sees.
    # ------------------------------------------------------------------ #
    def build_prompt(self, problem: Problem) -> str:
        """Render all public task content; hidden cases are never added."""
        statement = problem.statement.strip()
        if not problem.public_examples:
            return statement
        examples = [case.model_dump(mode="json") for case in problem.public_examples]
        return (
            statement
            + "\n\nPublic examples (args, kwargs, expected):\n"
            + json.dumps(examples, ensure_ascii=False, sort_keys=True, indent=2)
        )

    # ------------------------------------------------------------------ #
    # ChallengeResult -> Rollouts
    # ------------------------------------------------------------------ #
    def to_rollouts(
        self,
        problem: Problem,
        result: ChallengeResult,
    ) -> list[Rollout]:
        """One Rollout per MinerOutcome. The completion is the miner's raw
        response; the reward and pass counts come from its Verification. The
        pool pass-rate / band come from the ChallengeResult (difficulty signal
        captured at farming time)."""
        prompt = self.build_prompt(problem)
        rollouts: list[Rollout] = []
        for outcome in result.outcomes:
            rollouts.append(
                self._rollout_from_outcome(problem, result, outcome, prompt)
            )
        return rollouts

    @staticmethod
    def _rollout_from_outcome(
        problem: Problem,
        result: ChallengeResult,
        outcome: MinerOutcome,
        prompt: str,
    ) -> Rollout:
        v = outcome.verification
        completion = outcome.solution.raw_response or outcome.solution.code
        return Rollout(
            problem_id=problem.problem_id,
            prompt=prompt,
            # Validators record the authored user prompt and completion;
            # provider-side system prompts are not part of the response.
            system_prompt="",
            prompt_variant=problem.prompt_variant,
            completion=completion,
            reward=outcome.reward,
            all_passed=v.all_passed,
            num_passed=v.num_passed,
            num_tests=v.num_tests,
            pool_pass_rate=result.pass_rate,
            band=result.band,
            miner_uid=outcome.uid,
            miner_hotkey=outcome.hotkey,
            created_at=problem.created_at,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def write(self, rollouts: list[Rollout]) -> None:
        """Append rollouts to JSONL, rolling over to a new shard when the active
        shard is full."""
        for r in rollouts:
            try:
                shard_path = self._active_shard_path()
                line = json.dumps(self._serialize(r), ensure_ascii=False)
                self._append_line(shard_path, line)
                self._prune_to_cap()
            except OSError as error:
                self._warn_write_error(error)
                break

    def write_problem(self, problem: Problem) -> None:
        """Append one locally evaluated problem and its revealed tests.

        Revealed tests stay local to the validator.
        """
        try:
            shard_path = self._active_shard_path(PROBLEM_PREFIX)
            line = json.dumps(problem.model_dump(mode="json"), ensure_ascii=False)
            self._append_line(shard_path, line)
            self._prune_to_cap()
        except OSError as error:
            self._warn_write_error(error)

    @staticmethod
    def _warn_write_error(error: OSError) -> None:
        detail = str(error).replace("\n", " ")[:160]
        print(f"[validator] WARN: local rollout export failed ({detail})")

    @staticmethod
    def _append_line(path: str, line: str) -> None:
        """Append one UTF-8 JSON line without merging into a torn prior write."""
        encoded = (line + "\n").encode("utf-8")
        with open(path, "ab+") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.seek(0, os.SEEK_END)
                    fh.write(b"\n")
            fh.seek(0, os.SEEK_END)
            fh.write(encoded)

    @staticmethod
    def _serialize(rollout: Rollout) -> dict:
        # mode="json" turns the DifficultyBand enum into its string value.
        return rollout.model_dump(mode="json")

    def _shard_path(self, index: int, prefix: str = SHARD_PREFIX) -> str:
        return os.path.join(self.dataset_dir, f"{prefix}-{index:05d}.jsonl")

    def _shard_count(self, prefix: str = SHARD_PREFIX) -> int:
        indices = self._shard_indices(prefix)
        return max(indices, default=-1) + 1

    def _shard_indices(self, prefix: str) -> list[int]:
        start = f"{prefix}-"
        end = ".jsonl"
        indices = []
        for name in os.listdir(self.dataset_dir):
            if not name.startswith(start) or not name.endswith(end):
                continue
            value = name[len(start) : -len(end)]
            if value.isdigit():
                indices.append(int(value))
        return indices

    def _prune_to_cap(self) -> None:
        shards = []
        protected = set()
        for prefix in (SHARD_PREFIX, PROBLEM_PREFIX):
            indices = self._shard_indices(prefix)
            if indices:
                protected.add((prefix, max(indices)))
            for index in indices:
                path = self._shard_path(index, prefix)
                stat = os.stat(path)
                shards.append(
                    (stat.st_mtime_ns, index, prefix, path, stat.st_size)
                )

        total = sum(size for _, _, _, _, size in shards)
        for _, index, prefix, path, size in sorted(shards):
            if total <= self.max_bytes:
                break
            if (prefix, index) in protected:
                continue
            os.unlink(path)
            total -= size

    def _active_shard_path(self, prefix: str = SHARD_PREFIX) -> str:
        """Return the path of the shard to append to next, rolling over once the
        current shard has `shard_size` lines."""
        idx = max(0, self._shard_count(prefix) - 1)
        path = self._shard_path(idx, prefix)
        if not os.path.exists(path):
            return path
        if self._line_count(path) >= self.shard_size:
            return self._shard_path(idx + 1, prefix)
        return path

    @staticmethod
    def _line_count(path: str) -> int:
        n = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n
