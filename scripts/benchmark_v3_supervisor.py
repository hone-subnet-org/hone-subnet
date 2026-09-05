#!/usr/bin/env python
"""Benchmark the V3 container supervisor with synthetic polyglot programs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from rlvr.v3.supervisor import ContainerRequest, Mount, SupervisorPolicy, run_container


PROGRAMS = {
    "python": ("main.py", "print('ok')\n", "/usr/bin/python3 /work/main.py"),
    "javascript": ("main.js", "console.log('ok');\n", "/usr/bin/node /work/main.js"),
    "typescript": (
        "main.ts",
        "const value: string = 'ok'; console.log(value);\n",
        "/usr/bin/tsc --outDir /work/out /work/main.ts && /usr/bin/node /work/out/main.js",
    ),
    "rust": (
        "main.rs",
        "fn main() { println!(\"ok\"); }\n",
        "/usr/bin/rustc /work/main.rs -o /work/program && /work/program",
    ),
    "c": (
        "main.c",
        "#include <stdio.h>\nint main(void) { puts(\"ok\"); return 0; }\n",
        "/usr/bin/gcc /work/main.c -o /work/program && /work/program",
    ),
    "cpp": (
        "main.cpp",
        "#include <iostream>\nint main() { std::cout << \"ok\\n\"; }\n",
        "/usr/bin/g++ /work/main.cpp -o /work/program && /work/program",
    ),
    "java": (
        "Main.java",
        "public class Main { public static void main(String[] a) { System.out.println(\"ok\"); } }\n",
        "/usr/bin/javac -d /work/out /work/Main.java && /usr/bin/java -cp /work/out Main",
    ),
    "go": (
        "main.go",
        'package main\nimport "fmt"\nfunc main() { fmt.Println("ok") }\n',
        "/usr/bin/go run /work/main.go",
    ),
}


def _policy(image: str, args: argparse.Namespace) -> SupervisorPolicy:
    return SupervisorPolicy(
        image=image,
        candidate_uid=os.getuid(),
        candidate_gid=os.getgid(),
        trusted_uid=os.getuid(),
        trusted_gid=os.getgid(),
        memory_bytes=args.memory_gib << 30,
        cpus=args.cpus,
        pids_limit=args.pids,
        tmpfs_bytes=args.tmpfs_mib << 20,
        max_file_bytes=1 << 30,
        watchdog_slack_s=5,
    )


def _one(
    language: str,
    index: int,
    root: Path,
    policy: SupervisorPolicy,
    docker: str,
    timeout_s: int,
) -> tuple[float, bool]:
    filename, source, command = PROGRAMS[language]
    work = root / f"{language}-{index}"
    work.mkdir()
    (work / filename).write_text(source, encoding="utf-8")
    started = time.monotonic()
    result = run_container(
        ContainerRequest(
            name=f"hone-v3-bench-{language}-{index}",
            argv=("/usr/bin/bash", "--noprofile", "--norc", "-c", command),
            cwd="/work",
            mounts=(Mount(work, "/work", False),),
            stdin=b"",
            timeout_s=timeout_s,
            max_stdout_bytes=65_536,
            max_stderr_bytes=65_536,
            trusted=False,
        ),
        policy,
        docker,
    )
    elapsed = time.monotonic() - started
    passed = (
        result.exit_code == 0
        and result.stdout == b"ok\n"
        and not result.stderr
        and not result.oom_killed
        and not result.timed_out
        and not result.stdout_overflow
        and not result.stderr_overflow
    )
    return elapsed, passed


def _summary(times: list[float], failures: int) -> dict[str, float | int]:
    ordered = sorted(times)
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
    return {
        "runs": len(times),
        "failures": failures,
        "wall_p50_s": statistics.median(times),
        "wall_p95_s": p95,
        "wall_max_s": max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--memory-gib", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--pids", type=int, default=256)
    parser.add_argument("--tmpfs-mib", type=int, default=1024)
    args = parser.parse_args()
    for name in (
        "repeats",
        "concurrency",
        "timeout_s",
        "memory_gib",
        "cpus",
        "pids",
        "tmpfs_mib",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    docker = shutil.which("docker")
    if docker is None:
        parser.error("docker is unavailable")
    policy = _policy(args.image, args)
    output: dict[str, object] = {
        "image": args.image,
        "policy": {
            "memory_gib": args.memory_gib,
            "cpus": args.cpus,
            "pids": args.pids,
            "tmpfs_mib": args.tmpfs_mib,
            "timeout_s": args.timeout_s,
        },
        "languages": {},
    }
    failures = 0
    with tempfile.TemporaryDirectory(prefix="hone-v3-benchmark-") as temporary:
        root = Path(temporary)
        for language in PROGRAMS:
            times: list[float] = []
            language_failures = 0
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [
                    pool.submit(
                        _one,
                        language,
                        index,
                        root,
                        policy,
                        docker,
                        args.timeout_s,
                    )
                    for index in range(args.repeats)
                ]
                for future in as_completed(futures):
                    elapsed, passed = future.result()
                    times.append(elapsed)
                    language_failures += not passed
            output["languages"][language] = _summary(times, language_failures)
            failures += language_failures
    print(json.dumps(output, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
