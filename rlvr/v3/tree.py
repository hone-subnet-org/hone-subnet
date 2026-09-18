from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .identity import validate_relative_path


class TreeError(ValueError):
    pass


@dataclass(frozen=True)
class TreeLimits:
    max_entries: int
    max_file_bytes: int
    max_total_file_bytes: int
    max_path_bytes: int

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if type(value) is not int or value <= 0:
                raise ValueError("tree limits must be positive integers")


@dataclass(frozen=True)
class TreeReport:
    entries: tuple[str, ...]
    file_bytes: int


def inspect_tree(
    root: str | os.PathLike[str],
    *,
    limits: TreeLimits | None,
    normalize_modes: bool,
) -> TreeReport:
    base = Path(root)
    inspected: list[tuple[str, Path, bool, int]] = []
    total_file_bytes = 0
    try:
        root_info = base.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise TreeError("result tree root is not a directory")

        def walk_error(_error: OSError) -> None:
            raise TreeError("result tree could not be inspected")

        for current, directories, files in os.walk(
            base, followlinks=False, onerror=walk_error
        ):
            directories.sort()
            files.sort()
            for name in directories + files:
                path = Path(current, name)
                relative = path.relative_to(base).as_posix()
                try:
                    validate_relative_path(relative)
                except ValueError:
                    raise TreeError("result tree contains an unsafe path") from None
                if ".git" in relative.split("/"):
                    raise TreeError("result tree contains .git metadata")
                if limits is not None and len(relative.encode("utf-8")) > limits.max_path_bytes:
                    raise TreeError("result tree path exceeds the policy limit")

                info = path.lstat()
                is_directory = stat.S_ISDIR(info.st_mode)
                if not is_directory and (
                    not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                ):
                    raise TreeError("result tree contains an unsafe entry")
                inspected.append((relative, path, is_directory, info.st_mode))
                if limits is not None and len(inspected) > limits.max_entries:
                    raise TreeError("result tree contains too many entries")
                if not is_directory:
                    if limits is not None and info.st_size > limits.max_file_bytes:
                        raise TreeError("result tree file exceeds the policy limit")
                    total_file_bytes += info.st_size
                    if (
                        limits is not None
                        and total_file_bytes > limits.max_total_file_bytes
                    ):
                        raise TreeError("result tree exceeds the total byte limit")

        if normalize_modes:
            os.chmod(base, 0o755)
            for _, path, is_directory, original_mode in inspected:
                mode = 0o755 if is_directory or original_mode & stat.S_IXUSR else 0o644
                os.chmod(path, mode)
        return TreeReport(
            entries=tuple(sorted(item[0] for item in inspected)),
            file_bytes=total_file_bytes,
        )
    except TreeError:
        raise
    except OSError:
        raise TreeError("result tree could not be inspected") from None
