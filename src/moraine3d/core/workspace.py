"""Workspace detection and project discovery (DESIGN 4.2).

A directory with a top-level asset file is a project; a directory
holding ``.moraine/`` and no top-level asset file is a workspace; a
directory holding both is an invalid layout, refused with the fix
named. Projects live directly under the workspace root: discovery
examines only the root's immediate child directories, skips
dot-directories and the reserved layout names, identifies projects by
directory name, and orders them lexicographically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import MoraineError
from . import asset as asset_mod
from . import sources as sources_mod
from .asset import MORAINE_DIR, RESERVED_DIRS


@dataclass(frozen=True)
class Workspace:
    root: Path
    projects: tuple[str, ...]  # child dir names, lexicographic

    def project_dir(self, name: str) -> Path:
        return self.root / name

    def resolve_asset(self, name: str) -> Path:
        """Resolve a roster project's asset, validating membership first:
        every consumer of an externally supplied project name goes
        through this, so no client-chosen string ever reaches the
        filesystem as a path (DESIGN 7.2 target restriction). The
        resolved asset file must lie inside the resolved workspace root
        (DESIGN 4.6): a link-spelled sibling is legal while its target
        stays internal; one pointing outside is a refusal."""
        if name not in self.projects:
            valid = ", ".join(self.projects)
            raise MoraineError(
                "unknown_project",
                f"no project '{name}' in this workspace; one of: {valid}",
                projects=list(self.projects),
            )
        asset = sources_mod.resolve(self.project_dir(name))
        if not asset.resolve().is_relative_to(self.root.resolve()):
            raise MoraineError(
                "project_outside_workspace",
                f"project '{name}' resolves outside the workspace (a "
                f"link under the workspace root points elsewhere); "
                f"references and serving stay workspace-internal - move "
                f"the target into the workspace, or vendor the asset "
                f"under subassets/",
                project=name,
            )
        return asset


def is_project(path: Path) -> bool:
    """True iff the directory holds a top-level asset file (DESIGN
    4.2). The IR under ``build/`` never classifies anything."""
    return bool(sources_mod.asset_files(path))


def layout_conflict(path: Path) -> bool:
    """True iff the directory holds both ``.moraine/`` and a top-level
    asset file — the invalid layout (DESIGN 4.2)."""
    return (path / MORAINE_DIR).is_dir() and is_project(path)


def ensure_layout(path: Path) -> None:
    """Refuse the workspace/project layout conflict (DESIGN 4.2): a
    directory holding both ``.moraine/`` and a top-level asset file
    cannot be classified — under either reading one stray file silently
    flips discovery, serve, and record framing, so it is an error at
    every classification site, with the fix named."""
    if layout_conflict(path):
        raise MoraineError(
            "workspace_asset_conflict",
            f"'{Path(path).resolve().name}' holds both {MORAINE_DIR}/ and a top-level asset "
            f"file (*{sources_mod.RECIPE_EXT}), so it cannot be told apart "
            "as a workspace root or a project; move the asset into a "
            f"child project directory, or remove {MORAINE_DIR}/ to keep "
            "a standalone project",
        )


def ensure_project_name(path: Path) -> None:
    """Refuse a directory that holds a top-level asset file under a
    reserved layout name (DESIGN 4.2): the sweep skips the name, so the
    asset would evaluate when addressed and be missing when swept."""
    # The name as spelled is the sweep's identity, and the one resolved
    # is what a path such as "." names.
    spelled = [Path(path).name, Path(path).resolve().name]
    name = next((n for n in spelled if n.lower() in RESERVED_DIRS), None)
    if name is not None and is_project(path):
        raise MoraineError(
            "reserved_project_name",
            f"'{name}' holds a top-level asset file and is a reserved "
            f"layout name ({', '.join(sorted(RESERVED_DIRS))}), so it is "
            "never a project; rename the directory",
            directory=name, reserved=sorted(RESERVED_DIRS),
        )


def is_workspace(path: Path) -> bool:
    """True iff the directory holds ``.moraine/`` and is not a project.

    A directory with both ``.moraine/`` and a top-level asset file is
    a diagnosed layout conflict (DESIGN 4.2), never a classification.
    """
    ensure_layout(path)
    return (path / MORAINE_DIR).is_dir()


def project_dirs(root: Path) -> list[Path]:
    """The root's immediate children that are projects, by name
    (DESIGN 4.2): dot-directories and the reserved layout names are
    skipped. Raises ``OSError`` for a root that cannot be listed."""
    return sorted(
        (
            child for child in root.iterdir()
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name.lower() not in RESERVED_DIRS
            and is_project(child)
        ),
        key=lambda child: child.name,
    )


def discover(root: Path) -> Workspace:
    """Sweep the root's immediate children for projects, deterministically.

    An unlistable root is a terse error, not a traceback: a root can be
    stat-able (so workspace detection succeeds) yet deny listing, and
    ``iterdir`` raises where ``glob`` silently returns nothing.
    """
    ensure_layout(root)
    try:
        for child in sorted(root.iterdir(), key=lambda child: child.name):
            if child.is_dir():
                ensure_project_name(child)
        names = [child.name for child in project_dirs(root)]
    except OSError as e:
        raise MoraineError(
            "workspace_unreadable",
            f"cannot list workspace '{root}' ({e.__class__.__name__}: {e}); "
            "check directory permissions",
        ) from e
    return Workspace(root=root, projects=tuple(names))


def unswept_assets(ws: Workspace) -> list[str]:
    """Asset files under the workspace that no sweep reaches (DESIGN
    4.2): assets below the root's project level — a project directory
    nested one level too deep, an asset under a plain child directory —
    as root-relative POSIX paths, sorted. Roster projects are not
    descended (their ``subassets/`` are theirs), nor dot-directories
    anywhere or the reserved layout names at the root; a non-roster
    directory holding a top-level asset lists its assets and is not
    descended (its own sub-assets belong to it). Links (symlinks,
    junctions) are not followed and no directory is entered twice, so
    a link back into the workspace cannot manufacture rows. The sweep
    report names these under ``not_swept``, so a nested project
    vanishing from the roster never reads as swept green.
    """
    found: list[str] = []
    roster = set(ws.projects)
    seen: set[Path] = set()

    def walk(d: Path, depth: int) -> None:
        try:
            real = d.resolve()
            if real in seen:
                return
            seen.add(real)
            children = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        assets = [
            c for c in children if c.is_file() and sources_mod.is_asset_file(c)
        ]
        if depth > 0 and assets:
            found.extend(c.relative_to(ws.root).as_posix() for c in assets)
            return
        for c in children:
            if not c.is_dir() or c.name.startswith("."):
                continue
            if c.is_symlink() or c.is_junction():
                continue
            if depth == 0 and (c.name in roster or c.name.lower() in RESERVED_DIRS):
                continue
            if c.name.lower() in (asset_mod.LIB_DIR, asset_mod.BUILD_DIR):
                continue  # definitions and IR: no asset file lives there
            walk(c, depth + 1)

    walk(ws.root, 0)
    return sorted(found)


def enclosing_root(project_dir: Path) -> Path:
    """The root a project's reports and records are framed from: its
    workspace root when it is a workspace member, else the project
    directory itself (DESIGN 4.2) — resolved. One derivation for every
    consumer (the server's watch root, the asset content hash's relative
    paths, the session log's ``root``), so a revision is named the same
    way by every writer.
    """
    root = Path(project_dir)
    ensure_layout(root)
    ensure_project_name(root)
    if is_workspace(root.parent):
        root = root.parent
    return root.resolve()
