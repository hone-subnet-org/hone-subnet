"""Reserved file names a candidate may not add to the tree it is graded in:
modules and packages named after the standard library or common test tools,
Python start-up files, and .pth files. Files the workspace already ships are
graded as they are.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

# Python 3.12's standard library (sys.stdlib_module_names, without the
# ``test`` package, which no test runner imports), the packages the grading
# image installs, the common test tools and what they import, and the files
# Python runs at start-up.
# Fixed here rather than read from the running interpreter so every validator
# refuses exactly the same trees.
SHADOWED_NAMES = frozenset(
    {
    "__future__", "_abc", "_aix_support", "_ast", "_asyncio", "_bisect", "_blake2",
    "_bz2", "_codecs", "_codecs_cn", "_codecs_hk", "_codecs_iso2022", "_codecs_jp",
    "_codecs_kr", "_codecs_tw", "_collections", "_collections_abc", "_compat_pickle",
    "_compression", "_contextvars", "_crypt", "_csv", "_ctypes", "_curses",
    "_curses_panel", "_datetime", "_dbm", "_decimal", "_elementtree",
    "_frozen_importlib", "_frozen_importlib_external", "_functools", "_gdbm",
    "_hashlib", "_heapq", "_imp", "_io", "_json", "_locale", "_lsprof", "_lzma",
    "_markupbase", "_md5", "_msi", "_multibytecodec", "_multiprocessing", "_opcode",
    "_operator", "_osx_support", "_overlapped", "_pickle", "_posixshmem",
    "_posixsubprocess", "_py_abc", "_pydatetime", "_pydecimal", "_pyio", "_pylong",
    "_queue", "_random", "_scproxy", "_sha1", "_sha2", "_sha3", "_signal",
    "_sitebuiltins", "_socket", "_sqlite3", "_sre", "_ssl", "_stat", "_statistics",
    "_string", "_strptime", "_struct", "_symtable", "_thread", "_threading_local",
    "_tkinter", "_tokenize", "_tracemalloc", "_typing", "_uuid", "_warnings",
    "_weakref", "_weakrefset", "_winapi", "_zoneinfo", "abc", "aifc", "antigravity",
    "argparse", "array", "ast", "asyncio", "atexit", "audioop", "base64", "bdb",
    "binascii", "bisect", "builtins", "bz2", "cProfile", "calendar", "cgi", "cgitb",
    "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime", "dbm",
    "decimal", "difflib", "dis", "doctest", "email", "encodings", "ensurepip", "enum",
    "errno", "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "genericpath", "getopt", "getpass", "gettext", "glob",
    "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http", "idlelib",
    "imaplib", "imghdr", "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder", "msilib",
    "msvcrt", "multiprocessing", "netrc", "nis", "nntplib", "nt", "ntpath",
    "nturl2path", "numbers", "opcode", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty",
    "pwd", "py_compile", "pyclbr", "pydoc", "pydoc_data", "pyexpat", "queue", "quopri",
    "random", "re", "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3", "sre_compile",
    "sre_constants", "sre_parse", "ssl", "stat", "statistics", "string", "stringprep",
    "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "textwrap", "this",
    "threading", "time", "timeit", "tkinter", "token", "tokenize", "tomllib", "trace",
    "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
    "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
        "_distutils_hack", "_pytest", "attr", "attrs", "colorama", "coverage", "distutils",
        "exceptiongroup", "hypothesis", "iniconfig", "mock", "nose", "nose2", "packaging",
        "pip", "pkg_resources", "pluggy", "py", "pygments", "pytest", "setuptools", "six",
        "tomli", "typing_extensions", "wheel",
        "sitecustomize", "usercustomize",
    }
)

_REASON_LIMIT = 200


def _raise(error: OSError) -> None:
    raise error


def tree_files(root: str | os.PathLike[str]) -> frozenset[str]:
    """Every file under root as a relative POSIX path, and every link, to
    anything or nothing, with a trailing slash. Links are not followed. Any
    directory that cannot be read raises OSError, so a scan is never partial."""
    base = Path(root)
    found: set[str] = set()
    for current, directories, files in os.walk(base, followlinks=False, onerror=_raise):
        for name in files + directories:
            path = Path(current, name)
            relative = path.relative_to(base).as_posix()
            if path.is_symlink():
                found.add(relative + "/")
            elif name in files:
                found.add(relative)
    return frozenset(found)


def _module_name(filename: str) -> str | None:
    """The name Python would import this file under, if it is a module."""
    for suffix in (".py", ".pyc"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    if filename.endswith(".so"):
        return filename.split(".", 1)[0]
    return None


def _shadows(path: str) -> bool:
    """Whether a file at this path uses a reserved name."""
    parent, _, name = path.rpartition("/")
    module = _module_name(name)
    if name.endswith(".pth"):
        return True
    if module == "__init__":
        return parent.rpartition("/")[2] in SHADOWED_NAMES
    return module in SHADOWED_NAMES


def shadowing_files(before: Iterable[str], after: Iterable[str]) -> list[str]:
    """Files and links added between two snapshots that use a reserved name,
    anywhere in the tree."""
    found = []
    for path in sorted(set(after) - set(before)):
        if path.endswith("/"):  # a link: it may resolve to a file or a package
            target = path[:-1]
            if target.rpartition("/")[2] in SHADOWED_NAMES or _shadows(target):
                found.append(path)
        elif _shadows(path):
            found.append(path)
    return found


def shadowing_reason(found: list[str]) -> str:
    """At most 200 characters, the limit on a verdict's reason."""
    shown = ", ".join(found[:3]) + (f" and {len(found) - 3} more" if len(found) > 3 else "")
    reason = f"adds files with reserved Python module names: {shown}"
    return reason if len(reason) <= _REASON_LIMIT else reason[: _REASON_LIMIT - 3] + "..."
