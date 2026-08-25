"""Rollout dataset export is bounded on disk by a fixed 2 GiB cap."""

from __future__ import annotations

import os

from rlvr.config import Settings
from rlvr.dataset.writer import RolloutWriter
from rlvr.types import Problem, TestCase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _problem(size: int, idx: int = 0) -> Problem:
    return Problem(
        problem_id=f"p-{idx}",
        language="python",
        statement="x" * size,
        entrypoint="solve",
        tests=[TestCase(args=[1], kwargs={}, expected=1)],
    )


def _shards(path: str, prefix: str) -> list[str]:
    return sorted(f for f in os.listdir(path) if f.startswith(prefix + "-"))


def _total_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def test_cap_is_a_fixed_code_constant_used_by_default(tmp_path):
    from rlvr.dataset.writer import DATASET_MAX_BYTES

    assert DATASET_MAX_BYTES == 2 * 1024**3
    writer = RolloutWriter(str(tmp_path))
    assert writer.max_bytes == DATASET_MAX_BYTES


def test_no_operator_knob_exists_for_the_dataset_cap():
    for field in Settings.model_fields:
        assert "dataset_max" not in field and "dataset_export" not in field
    env_example = open(os.path.join(REPO_ROOT, ".env.example"), encoding="utf-8").read()
    assert "DATASET_MAX_BYTES" not in env_example
    assert "DATASET_EXPORT_ENABLED" not in env_example


def test_retention_prunes_oldest_shards_to_stay_under_cap(tmp_path):
    writer = RolloutWriter(str(tmp_path), shard_size=1, max_bytes=2500)

    for i in range(10):
        writer.write_problem(_problem(400, i))

    assert _total_bytes(str(tmp_path)) <= 2500
    remaining = _shards(str(tmp_path), "problems")
    assert remaining, "the newest shard must survive"
    assert remaining[-1] == "problems-00009.jsonl"
    assert "problems-00000.jsonl" not in remaining


def test_newest_shard_survives_even_when_cap_is_tiny(tmp_path):
    writer = RolloutWriter(str(tmp_path), shard_size=1, max_bytes=10)

    writer.write_problem(_problem(400, 0))
    writer.write_problem(_problem(400, 1))

    assert _shards(str(tmp_path), "problems") == ["problems-00001.jsonl"]


def test_pruned_indices_are_never_reused(tmp_path):
    writer = RolloutWriter(str(tmp_path), shard_size=1, max_bytes=1000)

    for i in range(5):
        writer.write_problem(_problem(400, i))

    names = _shards(str(tmp_path), "problems")
    assert names[-1] == "problems-00004.jsonl"
    assert all(int(n.split("-")[1][:5]) >= 3 for n in names)


def test_disk_errors_never_propagate_from_writes(tmp_path, monkeypatch):
    writer = RolloutWriter(str(tmp_path))

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(RolloutWriter, "_append_line", staticmethod(boom))

    writer.write_problem(_problem(10))
    writer.write([])
