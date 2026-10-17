"""What a project's ``build/`` still holds, and what reads it (DESIGN
4.5, 4.8).

A build writes the IR of its closure and leaves what earlier builds
wrote: the variant of an override that has since moved, the IR of a
sub-asset once built on its own. A reader that opens ``build/`` by path
takes those for current, so the build of a project's root removes them.

Two questions decide a removal, and they have different answers.
*Is the file the build's?* is asked of the file alone: its name is the
writer's, and its first bytes are an IR's. *Is it still needed?* cannot
be asked of the build that is cleaning, because a ``build/`` has more
than one writer: a project that references a sibling under overrides
writes the variant beside the sibling, and with it the variants of the
sibling's own references. So the question is asked of the workspace.
Every project's root IR is a root, a reference leads to the IR it
names, and what no root reaches is removed.

A file written within the last minute is left for a later build. A
``build/`` is written by other processes too, and a build writes its
closure file by file: a variant can land beside this project before
the root that names it lands beside its own, and in that moment no
root reaches it.

The same walk answers who reads a project: the other projects whose
roots reach one of its assets.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core import atomic
from ..core import asset as asset_mod
from ..core import sources as sources_mod
from ..core import workspace as workspace_mod
from ..core.paths import is_redirect
from ..core.asset import BUILD_DIR, ASSET_EXT, SUBASSETS_DIR
from ..errors import MoraineError
from ..ops import asset_ref as ref_op

# How a canonical IR begins (DESIGN 4.1): the root's opening brace,
# then the stamp on a line of its own. Frozen with the writer.
HEADER = b'{\n"build_version": "'
_REFERENCE = '"op": "asset_ref"'
# How long a file is left after it was written (seconds): longer than
# any build takes to write its closure, short enough that the next
# build of an editing round removes what the last one left.
FRESH_S = 60.0


def _fresh(now: float, mtime: float) -> bool:
    """Whether a file was written too lately to judge. The window is
    symmetric, so a time stamp from a clock ahead of this one is not
    fresh for ever, and strict, so a window of zero protects nothing."""
    return abs(now - mtime) < FRESH_S


def _key(path: Path) -> str:
    """A path as two spellings of one file compare equal: resolved, and
    folded in case. The supported filesystems fold case, and one of
    them resolves a path without restoring the case on disk, so a file
    written through another spelling of its name is the same file."""
    return str(path.resolve()).casefold()


def is_ir(path: Path) -> bool:
    """Whether a file begins as the build writes an IR."""
    try:
        with open(path, "rb") as f:
            return f.read(len(HEADER)) == HEADER
    except OSError:
        return False


def _homes(project: Path) -> list[Path]:
    """The directories a build writes a project's IR into."""
    build = project / BUILD_DIR
    return [build, build / SUBASSETS_DIR]


def _named_by_the_writer(path: Path) -> bool:
    """The name grammar of the two directories: ``<name>.mora``, a
    variant ``<name>.<24 hex>.mora``."""
    name = path.name
    return name.endswith(ASSET_EXT) and name != ASSET_EXT


def unlink_stale_ir(path: Path, project: Path) -> bool:
    """The one place an IR is deleted (DESIGN 4.5): ``path`` is removed
    only as a direct child of one of the project's two build
    directories, neither reached through a link, under a name the
    writer gives and with an IR's first bytes. Every condition is
    judged here, at the unlink, whatever the caller judged before. A
    file that is gone or held open is skipped."""
    homes = _homes(project)
    if path.parent not in homes:
        return False
    if is_redirect(homes[0]) or is_redirect(path.parent) or is_redirect(path):
        return False
    if not path.is_file() or not _named_by_the_writer(path) or not is_ir(path):
        return False
    try:
        os.unlink(path)
    except OSError:
        return False
    return True


# -- the walk --------------------------------------------------------------------


def _references(path: Path) -> list[Path] | None:
    """The IRs the references of one IR name, or None for a file that
    cannot be read as the build writes it."""
    if not is_ir(path):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    out: list[Path] = []
    for line in text.split("\n"):
        if _REFERENCE not in line:
            continue
        try:
            node = json.loads(line.rstrip(","))
            if node.get("op") != "asset_ref":
                continue
            spec = ref_op.ref_params(node["params"])
            out.append(ref_op.resolve_ref(path, spec))
        except (ValueError, KeyError, AttributeError, MoraineError):
            return None
    return out


@dataclass
class Reach:
    """What the roots of a workspace reach, by project."""

    # Project name -> the IRs its root reaches, the root included.
    reached: dict[str, set[Path]] = field(default_factory=dict)
    # Projects that hold a root asset and no root IR.
    not_built: list[str] = field(default_factory=list)
    # The files on the way that could not be read as a build writes
    # them, by the project whose root led there. A walk that could not
    # read one knows less than the disk holds, and removes nothing.
    unread: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.unread

    @property
    def kept(self) -> set[Path]:
        out: set[Path] = set()
        for paths in self.reached.values():
            out |= paths
        return out


def _projects(project: Path) -> list[Path]:
    """The projects whose roots are asked: the workspace's, or the one
    project outside a workspace. A workspace that cannot be listed is
    asked for this project alone, and ``walk`` hears of it as a file
    it could not read."""
    root = workspace_mod.enclosing_root(project)
    if root == project:
        return [project]
    found = [child.resolve() for child in workspace_mod.project_dirs(root)]
    return found if project in found else [project, *found]


def root_ir(project: Path) -> Path | None:
    """Where the IR of a project's root asset is written, or None for
    a directory without one root asset. The root is the recipe's to
    name, so the IR of a root asset since renamed is no root."""
    found = sources_mod.asset_files(project)
    if len(found) != 1:
        return None
    stem = found[0].name[: -len(sources_mod.RECIPE_EXT)]
    return project / BUILD_DIR / f"{stem}{ASSET_EXT}"


def walk(project: Path) -> Reach:
    """What every root of the workspace ``project`` belongs to reaches
    on disk."""
    reach = Reach()
    try:
        homes = _projects(project)
    except OSError:
        homes = [project]
        reach.unread[project.name] = [workspace_mod.enclosing_root(project)]
    for home in homes:
        root = root_ir(home)
        if root is None:
            continue
        if not root.is_file():
            reach.not_built.append(home.name)
            continue
        seen: set[Path] = set()
        pending = [root.resolve()]
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                continue  # a reference to an IR not written: nothing to keep
            named = _references(path)
            if named is None:
                reach.unread.setdefault(home.name, []).append(path)
                continue
            pending.extend(p.resolve() for p in named)
        reach.reached[home.name] = seen
    return reach


# -- the sweep -------------------------------------------------------------------


@dataclass
class Swept:
    # The IR of the project no root reaches, and of it what was removed.
    stale: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    unrecognized: list[Path] = field(default_factory=list)
    # The files that kept the walk from knowing what is needed: while
    # there is one, nothing is removed.
    blocked: list[Path] = field(default_factory=list)
    # The project's top-level asset files where they are not one: a
    # project without one root has no root to be read from, and
    # nothing of it is removed.
    rootless: list[Path] = field(default_factory=list)

    @property
    def readers(self) -> dict | None:
        """The report's ``consumers`` (DESIGN 6), or None where no
        other project reads this one and none is left to ask."""
        if not self.consumers and not self.not_built and not self.unread:
            return None
        out: dict = {"projects": list(self.consumers)}
        if self.not_built:
            out["not_built"] = list(self.not_built)
        if self.unread:
            out["unread"] = list(self.unread)
        unasked = " and ".join(repr(k) for k in ("not_built", "unread") if k in out)
        if not self.consumers:
            # No project is known to reference this one, so there is
            # none to check: what is left to say is how to find out.
            out["fix"] = (
                f"validate the workspace root once: the projects under "
                f"{unasked} have no IR on disk that says whether they "
                f"reference this one"
            )
            return out
        out["fix"] = (
            "a change here reaches the projects that reference this one, "
            "and this project's own checks do not run theirs; validate "
            "the workspace root to check them"
            + ("" if not unasked else
               f" ({unasked} name the projects with no IR on disk a "
               f"reference could be read from)")
        )
        return out
    # The projects whose roots reach an asset of this one, and the
    # projects that could not be asked: those not built, and those
    # whose IR could not be read.
    consumers: list[str] = field(default_factory=list)
    not_built: list[str] = field(default_factory=list)
    unread: list[str] = field(default_factory=list)


def sweep(project: Path, written: set[Path], *, remove: bool) -> Swept:
    """Remove from the project's ``build/`` the IR no root of the
    workspace reaches, and name who reads the project. ``written`` is
    what the build in hand made, kept whatever the disk says.
    ``remove`` False only looks."""
    project = project.resolve()
    reach = walk(project)
    out = Swept(
        not_built=[n for n in reach.not_built if n != project.name],
        unread=sorted(n for n in reach.unread if n != project.name),
        blocked=sorted(p for paths in reach.unread.values() for p in paths),
    )
    if root_ir(project) is None:
        out.rootless = sources_mod.asset_files(project)
    for name, paths in sorted(reach.reached.items()):
        if name == project.name:
            continue
        if any(_belongs(p, project) for p in paths):
            out.consumers.append(name)
    keep = {_key(p) for p in reach.kept} | {_key(p) for p in written}
    now = time.time()
    homes = _homes(project)
    for home in homes:
        if not home.is_dir() or is_redirect(home) or is_redirect(homes[0]):
            continue
        for path in sorted(home.iterdir()):
            if path.is_dir() and not is_redirect(path):
                if path.parent == homes[0] and path.name == SUBASSETS_DIR:
                    continue
                out.unrecognized.append(path)
                continue
            if (atomic.in_flight_to(path.name) or "").endswith(ASSET_EXT):
                continue  # a writer's own temp, its to rename or remove
            if not _named_by_the_writer(path) or not is_ir(path):
                out.unrecognized.append(path)
                continue
            if _key(path) in keep or not reach.complete or out.rootless:
                continue
            out.stale.append(path)
            if not remove or _written_lately(path, now):
                continue
            if unlink_stale_ir(path, project):
                out.removed.append(path)
    return out


def _written_lately(path: Path, now: float) -> bool:
    try:
        return _fresh(now, path.stat().st_mtime)
    except OSError:
        return True  # gone or unreadable: not this build's to judge


def _belongs(ir_path: Path, project: Path) -> bool:
    try:
        return asset_mod.project_root(ir_path) == project
    except (MoraineError, ValueError):
        return False
