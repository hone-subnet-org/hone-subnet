from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.benchmark_v3_supervisor import PROGRAMS


ROOT = Path(__file__).resolve().parents[1]


def test_polyglot_profile_has_a_digest_pinned_base_and_all_toolchains():
    dockerfile = (ROOT / "docker/polyglot-sandbox/Dockerfile").read_text()
    first = dockerfile.splitlines()[0]
    assert first.startswith("FROM ubuntu@sha256:")
    assert len(first.rsplit("@sha256:", 1)[1]) == 64
    for package in (
        "python3",
        "nodejs",
        "node-typescript",
        "rustc",
        "cargo",
        "gcc",
        "g++",
        "openjdk-21-jdk-headless",
        "golang-go",
        "git",
        "bash",
        "coreutils",
    ):
        assert package in dockerfile
    assert "--no-install-recommends" in dockerfile
    assert "rm -rf /var/lib/apt/lists" in dockerfile
    assert 'repo-polyglot-v1' in dockerfile
    assert "USER 65534:65534" in dockerfile
    assert "WORKDIR /tmp" in dockerfile


def test_benchmark_covers_every_profile_language():
    assert set(PROGRAMS) == {
        "python",
        "javascript",
        "typescript",
        "rust",
        "c",
        "cpp",
        "java",
        "go",
    }
    for filename, source, command in PROGRAMS.values():
        assert filename and source
        assert command.startswith("/usr/bin/")


def test_benchmark_help_exposes_only_benchmark_resource_controls():
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/benchmark_v3_supervisor.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    for flag in (
        "--image",
        "--repeats",
        "--concurrency",
        "--memory-gib",
        "--cpus",
        "--pids",
        "--tmpfs-mib",
    ):
        assert flag in result.stdout


def test_polyglot_build_script_requires_an_explicit_push():
    script = ROOT / "scripts/build_polyglot_sandbox.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0
    text = script.read_text()
    assert "--push" in text
    assert "RepoDigests" in text
    assert "docker push" in text
    assert "@sha256:" in text
