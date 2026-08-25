import os

from rlvr.dataset.writer import RolloutWriter


def test_retention_prunes_by_age_across_prefixes(tmp_path):
    paths = {
        "rollouts-00000.jsonl": (400, 1),
        "rollouts-00001.jsonl": (400, 5),
        "problems-00000.jsonl": (400, 4),
        "problems-00001.jsonl": (400, 6),
    }
    for name, (size, modified) in paths.items():
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        os.utime(path, ns=(modified, modified))

    writer = RolloutWriter(str(tmp_path), max_bytes=1200)
    writer._prune_to_cap()

    assert not (tmp_path / "rollouts-00000.jsonl").exists()
    assert (tmp_path / "problems-00000.jsonl").exists()
