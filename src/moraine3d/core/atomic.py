"""Atomic file replacement: write to a sibling temp name, then rename.

The rename is what makes a file visible, all-or-nothing — a concurrent
reader sees the old bytes or the new, never a partial file. The temp
name is ``<name>.tmp<pid>-<n>``: the pid keeps concurrent processes off
each other's file, the process-wide counter keeps concurrent threads of
one process off each other's (the server evaluates on the loop and in
worker threads at once, and two writers of one content-addressed record
sharing a temp file would leave one of them with a vanished file), and
the shape is part of the cache filename grammar
(``cache.store._NAME_RE`` recognizes stale leftovers) and is read back
by ``in_flight_to`` — change them together.

Policy-free: a failure at either step raises ``ReplaceError`` naming
the step and the temp file, and leaves that file in place. What the
failure means (a benign same-bytes race, an unwritable cache dir, a
locked export) and how the leftover is removed (the cache deletion
choke point, or a direct unlink beside an export) differ per caller,
so both stay with the caller.
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path


class ReplaceError(Exception):
    """``atomic_replace`` failed at ``step`` (``"write"`` or ``"replace"``);
    ``tmp`` is the temp file the caller now owns, ``cause`` the OSError."""

    def __init__(self, step: str, tmp: Path, cause: OSError):
        super().__init__(f"{step} failed: {cause}")
        self.step = step
        self.tmp = tmp
        self.cause = cause


_counter = itertools.count()


_TMP_SUFFIX = re.compile(r"\.tmp\d+(?:-\d+)?\Z")


def tmp_name(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp{os.getpid()}-{next(_counter)}")


def in_flight_to(name: str) -> str | None:
    """The name a temp name is on its way to, or None for a name that
    is no temp name: ``tmp_name`` read back, the earlier counter-less
    form included."""
    m = _TMP_SUFFIX.search(name)
    return name[: m.start()] if m and m.start() else None


def atomic_replace(path: Path, data: bytes) -> None:
    tmp = tmp_name(path)
    try:
        tmp.write_bytes(data)
    except OSError as e:
        raise ReplaceError("write", tmp, e) from e
    try:
        os.replace(tmp, path)
    except OSError as e:
        raise ReplaceError("replace", tmp, e) from e
