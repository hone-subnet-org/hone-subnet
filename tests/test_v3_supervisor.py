"""Contract for ``rlvr.v3.supervisor`` — host Docker runner + result core.

    SupervisorError(RuntimeError)   launch / inspect / cleanup faults; messages
        never contain host paths, secrets, container names or URLs.
    SupervisorPolicy(image, candidate_uid, candidate_gid, memory_bytes, cpus,
                     pids_limit, tmpfs_bytes, max_file_bytes, watchdog_slack_s,
                     trusted_uid=65534, trusted_gid=65534)   frozen
        image must be ``name@sha256:<64 hex>`` — an audited profile image.  The
        runner cannot clear ENV baked into an image; the policy only pins a
        digest and makes NO claim to scrub image ENV.  All uids/gids >= 1
        (nothing runs as root); other fields positive ints.
    Mount(host: Path, container: str, read_only: bool)   frozen; host absolute
        and existing; container absolute, normalized, not "/"
    ContainerRequest(name, argv, cwd, mounts, stdin, timeout_s,
                     max_stdout_bytes, max_stderr_bytes, trusted)   frozen
        name ^[a-z0-9][a-z0-9_-]{0,63}$; argv non-empty tuple, absolute
        argv[0]; cwd absolute; stdin bytes; positive int timeout and caps;
        trusted selects trusted_uid:gid, otherwise candidate_uid:gid.
    ContainerResult(exit_code, oom_killed, timed_out, stdout, stderr,
                    stdout_overflow, stderr_overflow)   frozen
    run_container(request, policy, docker_binary) -> ContainerResult
        <docker> run --name <name> -i <hardening> <image@digest>
            /usr/bin/timeout --signal=TERM --kill-after=1 <timeout_s> <argv...>
        i.e. GNU timeout runs INSIDE the container; the host only keeps a
        watchdog of timeout_s + watchdog_slack_s on the docker CLI.  After the
        CLI returns (or the watchdog fires) the runner runs ``docker inspect``
        and then ``docker rm -f <name>`` exactly once on every path.
        exit_code / oom_killed come from inspect; timed_out iff inspected exit
        is 124, or 137 with OOMKilled=false.  The CLI rc is never a verdict:
        125/126/127 is a launch failure ONLY when inspect cannot return a
        stopped container (candidate exit codes collide).  stdin is attached;
        stdout/stderr are captured up to their caps and flagged *_overflow.
        The docker CLI receives ONLY an allowlisted host environment (PATH,
        HOME, LANG, LC_ALL) and the container ONLY an explicit --env allowlist.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

DIGEST = "sha256:" + "ab" * 32
IMAGE = "ghcr.io/hone/profile-python@" + DIGEST
HARDENING = {"--network=none", "--ipc=none", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
             "--log-driver=none"}
ARGV = ("/usr/bin/python3", "-I", "/work/greet.py")
ENV_ALLOW = {"PATH", "HOME", "LANG", "LC_ALL"}


def _mod():
    from rlvr.v3 import supervisor

    return supervisor


def policy(**over):
    base = dict(image=IMAGE, candidate_uid=65534, candidate_gid=65534, memory_bytes=256 << 20, cpus=1,
                pids_limit=128, tmpfs_bytes=64 << 20, max_file_bytes=16 << 20, watchdog_slack_s=5)
    return _mod().SupervisorPolicy(**{**base, **over})


def mount(tmp_path, name="work", container="/work", read_only=False):
    host = tmp_path / name
    host.mkdir(exist_ok=True)
    return _mod().Mount(host=host, container=container, read_only=read_only)


def request(tmp_path, **over):
    base = dict(name="hone-c01", argv=ARGV, cwd="/work", stdin=b"", timeout_s=5, max_stdout_bytes=1024,
                max_stderr_bytes=1024, trusted=False,
                mounts=(mount(tmp_path), mount(tmp_path, "verify", "/verify", True)))
    return _mod().ContainerRequest(**{**base, **over})


def fake_docker(root: Path, *, run_cmd="", run_rc=0, inspect_out="exited 0 false", inspect_rc=0, rm_rc=0,
                read_stdin=True) -> Path:
    """Stand-in docker CLI: appends every argv (including the inner command after IMAGE) to root/log,
    dumps its environment to root/env, copies stdin to root/stdin, then behaves as configured.
    inspect_rc != 0 models 'no such container' (a true launch failure).  Absolute tool paths only."""
    log, script = root / "log", root / "docker"
    stdin_cmd = f'/bin/cat > "{root / "stdin"}";' if read_stdin else ""  # read_stdin=False: candidate ignores stdin
    script.write_text(
        "#!/bin/sh\n"
        f'/usr/bin/printf "%s\\n" "$*" >> "{log}"\n'
        'case "$1" in\n'
        f'  run) /usr/bin/env > "{root / "env"}"; {stdin_cmd} {run_cmd}\n'
        f"       exit {run_rc};;\n"
        f'  inspect) /usr/bin/printf "%s\\n" "{inspect_out}"; exit {inspect_rc};;\n'
        f"  rm) exit {rm_rc};;\n"
        "esac\nexit 2\n"
    )
    script.chmod(0o755)
    return script


def calls(root: Path) -> list[list[str]]:
    return [line.split() for line in (root / "log").read_text().splitlines()]


def run(tmp_path, docker, req=None, pol=None):
    return _mod().run_container(req or request(tmp_path), pol or policy(), docker_binary=str(docker))


def fails(tmp_path, docker, req=None, pol=None):
    with pytest.raises(_mod().SupervisorError) as info:
        run(tmp_path, docker, req, pol)
    return info.value


# ---------------------------------------------------------------- command line


def test_run_command_pins_hardening_layout_and_inner_timeout(tmp_path):
    docker = fake_docker(tmp_path, run_cmd="/usr/bin/printf out; /usr/bin/printf err >&2")
    result = run(tmp_path, docker, request(tmp_path, stdin=b"hello", timeout_s=7))
    run_line, inspect_line, rm_line = calls(tmp_path)
    assert run_line[:3] == ["run", "--name", "hone-c01"] and "-i" in run_line
    assert HARDENING <= set(run_line)
    assert {f"--memory={256 << 20}", f"--memory-swap={256 << 20}", "--cpus=1", "--pids-limit=128",
            "--user=65534:65534", f"--ulimit=fsize={16 << 20}:{16 << 20}",
            f"--tmpfs=/tmp:rw,exec,nosuid,nodev,size={64 << 20}"} <= set(run_line)
    assert run_line[run_line.index("--workdir") + 1] == "/work"
    assert "--rm" not in run_line and "--privileged" not in run_line
    mounts = [run_line[i + 1] for i, a in enumerate(run_line) if a == "--mount"]
    assert mounts == [f"type=bind,source={tmp_path / 'work'},target=/work",
                      f"type=bind,source={tmp_path / 'verify'},target=/verify,readonly"]
    inner = run_line[run_line.index(IMAGE) + 1:]
    assert inner == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=1", "7", *ARGV]
    assert inspect_line[0] == "inspect" and "--format" in inspect_line and inspect_line[-1] == "hone-c01"
    assert rm_line == ["rm", "-f", "hone-c01"]
    assert (tmp_path / "stdin").read_bytes() == b"hello"
    assert (result.stdout, result.stderr, result.exit_code, result.timed_out) == (b"out", b"err", 0, False)


def test_host_docker_cli_is_not_wrapped_in_timeout(tmp_path, monkeypatch):
    m = _mod()
    seen, real_popen = [], m.subprocess.Popen

    def spy(cmd, *a, **kw):
        seen.append(list(cmd))
        return real_popen(cmd, *a, **kw)

    monkeypatch.setattr(m.subprocess, "Popen", spy)
    docker = fake_docker(tmp_path)
    run(tmp_path, docker)
    assert seen[0][:2] == [str(docker), "run"]
    assert all("timeout" not in c[0] for c in seen)


def test_environment_allowlists_for_host_cli_and_container(tmp_path, monkeypatch):
    monkeypatch.setenv("HONE_SECRET_TOKEN", "s3cr3t-value")
    monkeypatch.setenv("DOCKER_HOST", "tcp://10.0.0.9:2375")
    docker = fake_docker(tmp_path)
    run(tmp_path, docker)
    env = dict(line.split("=", 1) for line in (tmp_path / "env").read_text().splitlines() if "=" in line)
    assert set(env) <= ENV_ALLOW | {"PWD", "SHLVL", "_", "OLDPWD"}  # sh adds the latter itself
    run_line = calls(tmp_path)[0]
    assert "--env-file" not in run_line
    container_env = [run_line[i + 1] for i, a in enumerate(run_line) if a in ("--env", "-e")]
    assert container_env and all(item.split("=", 1)[0] in ENV_ALLOW for item in container_env)
    assert "s3cr3t-value" not in " ".join(run_line) and "tcp://" not in " ".join(run_line)


def test_trusted_and_candidate_users_come_from_policy_and_are_never_root(tmp_path):
    docker = fake_docker(tmp_path)
    run(tmp_path, docker, request(tmp_path, trusted=True))
    run(tmp_path, docker, request(tmp_path, trusted=True), policy(trusted_uid=2000, trusted_gid=2001))
    run(tmp_path, docker, request(tmp_path, trusted=False), policy(candidate_uid=1000, candidate_gid=1001))
    users = [next(a for a in line if a.startswith("--user=")) for line in calls(tmp_path) if line[0] == "run"]
    assert users == ["--user=65534:65534", "--user=2000:2001", "--user=1000:1001"]


# ---------------------------------------------------------------- capture


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_capture_is_bounded_and_marks_overflow(tmp_path, stream):
    redirect = "" if stream == "stdout" else ">&2"
    docker = fake_docker(tmp_path, run_cmd=f"/usr/bin/head -c 5000 /dev/zero | /usr/bin/tr '\\0' 'x' {redirect}")
    result = run(tmp_path, docker, request(tmp_path, max_stdout_bytes=100, max_stderr_bytes=100))
    other = "stderr" if stream == "stdout" else "stdout"
    assert len(getattr(result, stream)) == 100 and getattr(result, f"{stream}_overflow") is True
    assert getattr(result, other) == b"" and getattr(result, f"{other}_overflow") is False


def test_exact_cap_is_not_overflow(tmp_path):
    docker = fake_docker(tmp_path, run_cmd="/usr/bin/head -c 100 /dev/zero")
    result = run(tmp_path, docker, request(tmp_path, max_stdout_bytes=100))
    assert len(result.stdout) == 100 and result.stdout_overflow is False


# ---------------------------------------------------------------- classification


@pytest.mark.parametrize("run_rc, inspect_out, expected", [
    (0, "exited 3 false", dict(exit_code=3, oom_killed=False, timed_out=False)),
    (1, "exited 0 false", dict(exit_code=0, oom_killed=False, timed_out=False)),
    (0, "exited 124 false", dict(exit_code=124, oom_killed=False, timed_out=True)),
    (137, "exited 137 true", dict(exit_code=137, oom_killed=True, timed_out=False)),
    (137, "exited 137 false", dict(exit_code=137, oom_killed=False, timed_out=True)),
    (125, "exited 125 false", dict(exit_code=125, oom_killed=False, timed_out=False)),
    (127, "exited 3 false", dict(exit_code=3, oom_killed=False, timed_out=False)),
], ids=["inspect-beats-rc", "rc-1-inspect-0", "124-timeout", "137-oom", "137-kill-after",
        "candidate-125-is-not-launch-failure", "cli-127-valid-inspect-wins"])
def test_classification_uses_inspect_only(tmp_path, run_rc, inspect_out, expected):
    m = _mod()
    docker = fake_docker(tmp_path, run_rc=run_rc, inspect_out=inspect_out)
    result = run(tmp_path, docker)
    assert isinstance(result, m.ContainerResult)
    assert {k: getattr(result, k) for k in expected} == expected
    assert [c[0] for c in calls(tmp_path)] == ["run", "inspect", "rm"]
    with pytest.raises(Exception):
        result.exit_code = 0


def test_host_watchdog_fires_at_deadline_plus_slack_then_inspects_and_cleans_up(tmp_path):
    docker = fake_docker(tmp_path, run_cmd="/bin/sleep 30", inspect_out="exited 137 false")
    started = time.monotonic()
    result = run(tmp_path, docker, request(tmp_path, timeout_s=1), policy(watchdog_slack_s=1))
    assert time.monotonic() - started < 10
    assert result.timed_out is True and result.oom_killed is False and result.exit_code == 137
    assert [c[0] for c in calls(tmp_path)] == ["run", "kill", "inspect", "rm"]


def test_large_stdin_ignored_by_candidate_does_not_stall_the_host(tmp_path):
    """Regression: feeding stdin must be covered by the host watchdog.  A candidate that never reads
    a 1 MiB stdin must not block run_container beyond timeout_s + watchdog_slack_s."""
    docker = fake_docker(tmp_path, run_cmd="/bin/sleep 30", inspect_out="exited 137 false", read_stdin=False)
    req, pol, outcome = request(tmp_path, stdin=b"x" * (1 << 20), timeout_s=1), policy(watchdog_slack_s=1), []
    m = _mod()

    def target():
        try:
            outcome.append(m.run_container(req, pol, docker_binary=str(docker)))
        except m.SupervisorError as exc:
            outcome.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "run_container stalled on stdin write past the host watchdog"
    assert isinstance(outcome[0], (m.ContainerResult, m.SupervisorError))
    assert [c[0] for c in calls(tmp_path)] == ["run", "kill", "inspect", "rm"]


# ---------------------------------------------------------------- faults and cleanup


def _assert_clean(exc: BaseException, *forbidden: str):
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(repr(exc) + str(exc))
        exc = exc.__cause__ or exc.__context__
    text = " ".join(chain)
    for item in forbidden:
        assert item not in text


@pytest.mark.parametrize("kwargs", [
    dict(run_rc=125, inspect_rc=1), dict(run_rc=126, inspect_rc=1), dict(run_rc=127, inspect_rc=1),
    dict(run_rc=0, inspect_rc=1), dict(inspect_out="bogus"), dict(inspect_out="running 0 false"),
    dict(inspect_out="exited 256 false"), dict(rm_rc=1),
], ids=["launch-125", "launch-126", "launch-127", "inspect-missing", "inspect-garbage", "inspect-still-running",
        "inspect-exit-out-of-range", "rm-failure"])
def test_faults_are_supervisor_errors_with_exactly_one_rm(tmp_path, kwargs, monkeypatch):
    monkeypatch.setenv("HONE_SECRET_TOKEN", "s3cr3t-value")
    docker = fake_docker(tmp_path, **kwargs)
    exc = fails(tmp_path, docker)
    assert [c[0] for c in calls(tmp_path)] == ["run", "inspect", "rm"]
    _assert_clean(exc, str(tmp_path), "s3cr3t-value", "hone-c01", "tcp://")


def test_missing_docker_binary_is_a_supervisor_error_without_paths(tmp_path):
    exc = fails(tmp_path, tmp_path / "absent-docker")
    _assert_clean(exc, str(tmp_path))


# ---------------------------------------------------------------- validation


def test_policy_validation_and_defaults():
    pol = policy()
    assert (pol.trusted_uid, pol.trusted_gid) == (65534, 65534)
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", pol.image)
    with pytest.raises(Exception):
        pol.cpus = 2
    for over in (dict(image="ghcr.io/hone/profile-python:latest"), dict(image="ghcr.io/hone/profile-python@sha256:abc"),
                 dict(image=IMAGE.upper()), dict(candidate_uid=0), dict(candidate_gid=0), dict(trusted_uid=0),
                 dict(trusted_gid=-1), dict(memory_bytes=0), dict(cpus=0), dict(pids_limit=-1), dict(tmpfs_bytes=True),
                 dict(max_file_bytes=0),
                 dict(watchdog_slack_s="5")):
        with pytest.raises(ValueError):
            policy(**over)


def test_request_validation(tmp_path):
    for over in (dict(name=""), dict(name="Bad Name"), dict(name="a" * 65), dict(argv=()), dict(argv=("python3",)),
                 dict(argv=list(ARGV)), dict(cwd="work"), dict(cwd="/work/../x"), dict(stdin="text"),
                 dict(timeout_s=0), dict(max_stdout_bytes=0), dict(max_stderr_bytes=-1), dict(trusted=1),
                 dict(mounts=[mount(tmp_path)])):
        with pytest.raises(ValueError):
            request(tmp_path, **over)


def test_mount_validation(tmp_path):
    m = _mod()
    host = tmp_path / "work"
    host.mkdir()
    for args in (dict(host=host, container="work"), dict(host=host, container="/"), dict(host=host, container="/work/"),
                 dict(host=tmp_path / "absent", container="/work"), dict(host=Path("work"), container="/work"),
                 dict(host=str(host), container="/work")):
        with pytest.raises(ValueError):
            m.Mount(read_only=False, **args)
