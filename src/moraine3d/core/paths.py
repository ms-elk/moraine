"""Machine-level paths.

The evaluation cache is tool-managed, not user-managed: it lives in the
OS cache directory, keyed content-addressed (DESIGN 4.5), shared across
all projects, and safe to delete at any time. ``MORAINE_CACHE_DIR``
overrides the location (tests, CI hermeticity).
"""

from __future__ import annotations

import re

import os
from pathlib import Path

from platformdirs import user_cache_path

from ..errors import MoraineError

# Cache Directory Tagging standard (https://bford.info/cachedir/): the
# tag marks the dir as Moraine's deletable cache, and tag-honoring
# backup tools skip it. Stamped here at the moment of creation — the
# only instant "Moraine created this directory and it is empty" is
# fresh evidence — and by cache.store._ensure_tag on first write into a dir
# holding nothing but Moraine's own files (§4.5 deletion safety).
# The signature line is fixed by the spec. The "created by Moraine"
# line is what the deletion gate matches (cache.store._tag_state): the bare
# signature only proves the dir is *a* cache — another tool's tag
# (uv, cargo) must not open Moraine's deletion gate.
# _TAG_MORAINE_PREFIX is a FROZEN on-disk format: the gate compares it
# byte-exact against tags already stamped on user machines, so any
# rewording orphans every existing cache (gate refused, never
# re-stamped). A test pins the exact bytes; change them only together
# with a migration that still accepts the old prefix.
# Portable-identifier grammar: names that become filenames and engine
# object names (output names, param ids, face-group names, material
# names) and the cache filename fragments composed from them (cache/store.py;
# the server's /cache/ route composes from cache's fragments). One
# source on purpose: widening it for the asset loader without the
# cache's fragments would classify freshly written records as foreign
# content — blocking the deletion gates and 404ing them in the viewer.
PORTABLE_NAME_FRAGMENT = r"[A-Za-z][A-Za-z0-9_-]*"
# The whole-string form, compiled once: material names, rename targets,
# and every other identity that becomes a filename or engine object name.
PORTABLE_NAME_RE = re.compile(rf"^{PORTABLE_NAME_FRAGMENT}$")

TAG_NAME = "CACHEDIR.TAG"
_TAG_SIGNATURE = b"Signature: 8a477f597d28d172789f06886806bc55"
_TAG_MORAINE_PREFIX = _TAG_SIGNATURE + (
    b"\n# This file is a cache directory tag created by Moraine.\n"
)
_TAG_BODY = _TAG_MORAINE_PREFIX + (
    b"# For information about cache directory tags see https://bford.info/cachedir/\n"
)


# The ephemeral preview bucket (DESIGN 4.5): a flat subdirectory of the
# cache dir holding preview-overlay evaluations. Same filename grammar
# inside, outside the persistent cap and LRU (store.scan ignores
# directories), purged by the server at start and stop. The name is part
# of the deletion-safety surface: cache.store recognizes exactly this
# directory as Moraine infrastructure, and the deletion choke point
# accepts its direct children.
PREVIEW_DIR_NAME = "preview"


def is_redirect(p: Path | str) -> bool:
    """True when ``p`` is any filesystem redirection — a symlink or a
    Windows NTFS directory junction. Every place that asks "is this the
    real preview bucket?" — the write that creates it, the deletion
    surface, the serving route — must ask through here: ``is_symlink()``
    alone is False for junctions (``os.path.isjunction`` is the separate
    probe), and junctions need no privilege to create, so a junction
    named after the bucket would otherwise carry a write or the purge
    into an arbitrary target tree."""
    return os.path.islink(p) or os.path.isjunction(p)


def preview_dir() -> Path:
    """The ephemeral preview bucket, created on demand inside the cache
    dir. No own CACHEDIR.TAG: the parent's tag already marks the whole
    tree as Moraine's deletable cache. A symlink or junction wearing the
    bucket's name is refused, never written through: ``mkdir`` fails on
    it and ``is_dir()`` would follow it."""
    d = cache_dir() / PREVIEW_DIR_NAME
    try:
        d.mkdir()
    except OSError as e:
        if is_redirect(d):
            raise MoraineError(
                "cache_unwritable",
                f"preview bucket '{d}' is a symlink or junction, which "
                "Moraine never writes through; remove the link (the next "
                "preview recreates the bucket), or set MORAINE_CACHE_DIR "
                "to a writable directory",
            ) from e
        try:
            ok = d.is_dir()
        except OSError:
            ok = False
        if not ok:
            raise MoraineError(
                "cache_unwritable",
                f"preview bucket '{d}' is not writable "
                f"({e.__class__.__name__}); set MORAINE_CACHE_DIR to a "
                "writable directory",
            ) from e
    return d


def cache_dir() -> Path:
    override = os.environ.get("MORAINE_CACHE_DIR")
    d = Path(override) if override else user_cache_path("moraine", appauthor=False)
    try:
        d.mkdir(parents=True)
    except OSError as e:
        # Any failure mode surfaces as one actionable error — including a
        # stat denied inside the exists-check (is_dir itself can raise).
        try:
            ok = d.is_dir()
        except OSError:
            ok = False
        if not ok:
            raise MoraineError(
                "cache_unwritable",
                f"cache dir '{d}' is not writable ({e.__class__.__name__}); "
                "set MORAINE_CACHE_DIR to a writable directory",
            ) from e
    else:
        # Freshly created by Moraine: provably empty and ours, so stamp
        # now. Load-bearing: this creation-time stamp is the ONLY
        # intra-process re-stamp for a deleted-and-recreated dir once
        # cache's judged-dir memo holds — _ensure_tag never revalidates,
        # and its staleness argument names this line as the heal, so it
        # must stay at mkdir time. A failed stamp here is retried by
        # the write path only for a dir this process has not yet
        # judged. Exclusive create: never replace a tag that appeared
        # in the mkdir-to-write window.
        try:
            with open(d / TAG_NAME, "xb") as f:
                f.write(_TAG_BODY)
        except OSError:
            pass
    return d
