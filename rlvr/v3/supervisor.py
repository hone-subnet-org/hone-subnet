from __future__ import annotations

import os
import posixpath
import re
import signal
import stat
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

_IMAGE = re.compile(r"^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_HOST_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}


class SupervisorError(RuntimeError):
    pass


def _positive_int(value: object, *, minimum: int = 1) -> bool:
    return type(value) is int and value >= minimum


def _absolute_container_path(value: object, *, allow_root: bool = False) -> bool:
    return (
        type(value) is str
        and value.startswith("/")
        and not value.startswith("//")
        and (allow_root or value != "/")
        and posixpath.normpath(value) == value
        and "\x00" not in value
    )


@dataclass(frozen=True)
class SupervisorPolicy:
    image: str
    candidate_uid: int
    candidate_gid: int
    memory_bytes: int
    cpus: int
    pids_limit: int
    tmpfs_bytes: int
    max_file_bytes: int
    watchdog_slack_s: int
    trusted_uid: int = 65_534
    trusted_gid: int = 65_534

    def __post_init__(self) -> None:
        if type(self.image) is not str or _IMAGE.fullmatch(self.image) is None:
            raise ValueError("supervisor image must be digest-pinned")
        for value in (
            self.candidate_uid,
            self.candidate_gid,
            self.trusted_uid,
            self.trusted_gid,
        ):
            if not _positive_int(value):
                raise ValueError("supervisor users must be unprivileged")
        for value in (
            self.memory_bytes,
            self.cpus,
            self.pids_limit,
            self.tmpfs_bytes,
            self.max_file_bytes,
            self.watchdog_slack_s,
        ):
            if not _positive_int(value):
                raise ValueError("supervisor limits must be positive integers")


@dataclass(frozen=True)
class Mount:
    host: Path
    container: str
    read_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.host, Path) or not self.host.is_absolute() or "," in str(self.host):
            raise ValueError("mount source must be an absolute safe path")
        try:
            info = self.host.lstat()
        except OSError:
            raise ValueError("mount source is unavailable") from None
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise ValueError("mount source is invalid")
        if not _absolute_container_path(self.container) or "," in self.container:
            raise ValueError("mount target is invalid")
        if type(self.read_only) is not bool:
            raise ValueError("mount mode must be boolean")


@dataclass(frozen=True)
class ContainerRequest:
    name: str
    argv: tuple[str, ...]
    cwd: str
    mounts: tuple[Mount, ...]
    stdin: bytes
    timeout_s: int
    max_stdout_bytes: int
    max_stderr_bytes: int
    trusted: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NAME.fullmatch(self.name) is None:
            raise ValueError("container name is invalid")
        if (
            type(self.argv) is not tuple
            or not self.argv
            or any(type(item) is not str or not item or "\x00" in item for item in self.argv)
            or not _absolute_container_path(self.argv[0])
        ):
            raise ValueError("container argv is invalid")
        if not _absolute_container_path(self.cwd, allow_root=True):
            raise ValueError("container working directory is invalid")
        if type(self.mounts) is not tuple or any(type(item) is not Mount for item in self.mounts):
            raise ValueError("container mounts are invalid")
        targets = [item.container for item in self.mounts]
        if len(set(targets)) != len(targets):
            raise ValueError("container mount targets must be unique")
        if type(self.stdin) is not bytes:
            raise ValueError("container stdin must be bytes")
        if not _positive_int(self.timeout_s) or self.timeout_s > 3_600:
            raise ValueError("container timeout is invalid")
        if not _positive_int(self.max_stdout_bytes) or not _positive_int(self.max_stderr_bytes):
            raise ValueError("container output limits must be positive")
        if type(self.trusted) is not bool:
            raise ValueError("container trust mode must be boolean")


@dataclass(frozen=True)
class ContainerResult:
    exit_code: int
    oom_killed: bool
    timed_out: bool
    stdout: bytes
    stderr: bytes
    stdout_overflow: bool
    stderr_overflow: bool


def _drain(pipe, limit: int, output: bytearray, overflow: list[bool]) -> None:
    try:
        while True:
            chunk = pipe.read(8_192)
            if not chunk:
                break
            remaining = limit - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow[0] = True
    finally:
        pipe.close()


def _run_process(
    command: list[str], stdin: bytes, stdout_limit: int, stderr_limit: int, timeout: int
) -> tuple[int, bytes, bytes, bool, bool, bool]:
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_HOST_ENV,
            start_new_session=True,
        )
    except (OSError, ValueError):
        raise SupervisorError("container launch failed") from None

    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = [False]
    stderr_overflow = [False]
    assert child.stdout is not None and child.stderr is not None
    readers = [
        threading.Thread(
            target=_drain,
            args=(child.stdout, stdout_limit, stdout, stdout_overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=(child.stderr, stderr_limit, stderr, stderr_overflow),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    def feed() -> None:
        try:
            if child.stdin is not None:
                child.stdin.write(stdin)
                child.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            if child.stdin is not None:
                try:
                    child.stdin.close()
                except OSError:
                    pass

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()

    watchdog_expired = False
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        watchdog_expired = True
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except OSError:
            try:
                child.kill()
            except OSError:
                pass
        child.wait()
    for reader in readers:
        reader.join(timeout=2)
    feeder.join(timeout=1)
    return (
        int(child.returncode),
        bytes(stdout),
        bytes(stderr),
        stdout_overflow[0],
        stderr_overflow[0],
        watchdog_expired,
    )


def _control(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            env=_HOST_ENV,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        raise SupervisorError("container control operation failed") from None


def run_container(
    request: ContainerRequest,
    policy: SupervisorPolicy,
    docker_binary: str,
) -> ContainerResult:
    if type(request) is not ContainerRequest or type(policy) is not SupervisorPolicy:
        raise TypeError("run_container requires validated inputs")
    docker = Path(docker_binary)
    if not docker.is_absolute() or not docker.is_file() or not os.access(docker, os.X_OK):
        raise SupervisorError("container runtime is unavailable")
    for mount in request.mounts:
        try:
            info = mount.host.lstat()
        except OSError:
            raise SupervisorError("container mount is unavailable") from None
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise SupervisorError("container mount is invalid")

    uid = policy.trusted_uid if request.trusted else policy.candidate_uid
    gid = policy.trusted_gid if request.trusted else policy.candidate_gid
    command = [
        str(docker),
        "run",
        "--name",
        request.name,
        "-i",
        "--network=none",
        "--ipc=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--log-driver=none",
        f"--memory={policy.memory_bytes}",
        f"--memory-swap={policy.memory_bytes}",
        f"--cpus={policy.cpus}",
        f"--pids-limit={policy.pids_limit}",
        f"--ulimit=fsize={policy.max_file_bytes}:{policy.max_file_bytes}",
        f"--tmpfs=/tmp:rw,exec,nosuid,nodev,size={policy.tmpfs_bytes}",
        f"--user={uid}:{gid}",
        "--workdir",
        request.cwd,
        "--env",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "--env",
        "HOME=/tmp",
        "--env",
        "LANG=C.UTF-8",
        "--env",
        "LC_ALL=C.UTF-8",
    ]
    for mount in request.mounts:
        value = f"type=bind,source={mount.host},target={mount.container}"
        if mount.read_only:
            value += ",readonly"
        command.extend(("--mount", value))
    command.extend(
        (
            policy.image,
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=1",
            str(request.timeout_s),
            *request.argv,
        )
    )

    run_started = False
    pending_error: SupervisorError | None = None
    result: ContainerResult | None = None
    try:
        run_started = True
        _, stdout, stderr, stdout_overflow, stderr_overflow, watchdog = _run_process(
            command,
            request.stdin,
            request.max_stdout_bytes,
            request.max_stderr_bytes,
            request.timeout_s + policy.watchdog_slack_s,
        )
        if watchdog:
            _control([str(docker), "kill", request.name])
        inspected = _control(
            [
                str(docker),
                "inspect",
                "--format",
                "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}",
                request.name,
            ]
        )
        parts = inspected.stdout.strip().split()
        if inspected.returncode != 0 or len(parts) != 3:
            raise SupervisorError("container state is unavailable")
        if parts[0] != "exited":
            raise SupervisorError("container did not stop")
        try:
            exit_code = int(parts[1])
        except ValueError:
            raise SupervisorError("container state is invalid") from None
        if not 0 <= exit_code <= 255 or parts[2] not in ("true", "false"):
            raise SupervisorError("container state is invalid")
        oom_killed = parts[2] == "true"
        result = ContainerResult(
            exit_code=exit_code,
            oom_killed=oom_killed,
            timed_out=watchdog or (exit_code in (124, 137) and not oom_killed),
            stdout=stdout,
            stderr=stderr,
            stdout_overflow=stdout_overflow,
            stderr_overflow=stderr_overflow,
        )
    except SupervisorError as exc:
        pending_error = exc
    finally:
        if run_started:
            cleanup = _control([str(docker), "rm", "-f", request.name])
            if cleanup.returncode != 0:
                pending_error = SupervisorError("container cleanup failed")
    if pending_error is not None:
        raise pending_error from None
    assert result is not None
    return result
