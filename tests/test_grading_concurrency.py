"""Grading concurrency is sized from the host unless the operator sets it."""

from __future__ import annotations

from types import SimpleNamespace

from rlvr.config import Settings
from rlvr.neurons import decentralized
from rlvr.policy import RELEASE_POLICY
from rlvr.v3.release import GRADING_CONCURRENCY_MAX, default_grading_concurrency

GIB = 1024**3
GB = 1000**3


def sized(cpus, memory_gib, free_disk_gb):
    return default_grading_concurrency(
        RELEASE_POLICY, cpus=cpus, memory_bytes=memory_gib * GIB, free_disk_bytes=free_disk_gb * GB
    )


def test_the_documented_minimum_host_grades_two_at_once():
    assert sized(cpus=4, memory_gib=12, free_disk_gb=25) == 2


def test_each_resource_bounds_it_and_nothing_goes_below_one():
    assert sized(cpus=16, memory_gib=256, free_disk_gb=500) == 8  # two CPUs per grading sandbox
    assert sized(cpus=1, memory_gib=64, free_disk_gb=500) == 1  # never zero
    assert sized(cpus=16, memory_gib=16, free_disk_gb=500) == 3  # (16 - 4) / 4 memory
    assert sized(cpus=16, memory_gib=64, free_disk_gb=35) == 3  # (35 - 2) / 10.7 disk
    assert sized(cpus=2, memory_gib=2, free_disk_gb=1) == 1  # never zero, never negative


def test_a_large_host_is_capped():
    assert sized(cpus=64, memory_gib=256, free_disk_gb=2_000) == GRADING_CONCURRENCY_MAX == 16


def test_settings_default_to_unset_and_env_overrides(monkeypatch):
    monkeypatch.delenv("VALIDATOR_GRADING_CONCURRENCY", raising=False)
    assert Settings(_env_file=None).validator_grading_concurrency is None
    monkeypatch.setenv("VALIDATOR_GRADING_CONCURRENCY", "5")
    assert Settings(_env_file=None).validator_grading_concurrency == 5


def test_startup_uses_the_setting_or_measures_the_host(monkeypatch, tmp_path, capsys):
    explicit = Settings(_env_file=None, validator_grading_concurrency=7)
    assert decentralized._grading_concurrency(explicit, RELEASE_POLICY, tmp_path / "data") == 7
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(decentralized.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        decentralized.os, "sysconf", lambda name: {"SC_PHYS_PAGES": 8 * GIB // 4096, "SC_PAGE_SIZE": 4096}[name]
    )
    probed = []

    def disk_usage(path):
        probed.append(path)
        return SimpleNamespace(free=200 * GB)

    monkeypatch.setattr(decentralized.shutil, "disk_usage", disk_usage)
    unset = Settings(_env_file=None)
    unset.validator_grading_concurrency = None
    missing = tmp_path / "data" / "deeper"
    assert decentralized._grading_concurrency(unset, RELEASE_POLICY, missing) == 1  # (8 - 4) / 4 memory
    out = capsys.readouterr().out
    assert "grading concurrency 1 for this host" in out and "VALIDATOR_GRADING_CONCURRENCY" in out
    assert probed == [tmp_path]  # the nearest existing ancestor is measured
    assert not (tmp_path / "data").exists()  # measuring creates nothing
