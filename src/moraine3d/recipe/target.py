"""A command's target, opened (DESIGN 6): every evaluating command
builds first, and an IR handed over by path is evaluated as it is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..core import asset as asset_mod
from ..core import sources as sources_mod
from ..core import workspace as workspace_mod
from . import build as build_mod
from . import version


@dataclass
class Opened:
    asset: asset_mod.Asset  # the target's IR
    # The IRs the build made, by the path each is written at; empty for
    # an IR handed over by path, whose references open from disk.
    irs: dict[Path, asset_mod.Asset]
    # The files whose bytes are the revision (DESIGN 6): the recipe
    # files a build read, or the IR file handed over.
    sources: frozenset[Path]
    root: Path  # the root reports and records are framed from
    build: build_mod.BuildResult | None = None

    @property
    def found(self) -> dict[Path, bytes | None]:
        """What the build found at the recipe files it named, by
        resolved path; nothing for an IR handed over by path."""
        return self.build.contents if self.build is not None else {}

    def files(self, reached: Iterable[Path] = ()) -> frozenset[Path]:
        """The files whose bytes are the revision (DESIGN 6): the
        recipe files the build read, or the IR handed over with the IRs
        its evaluation ``reached``."""
        if self.build is not None:
            return self.sources
        return self.sources | frozenset(reached)

    def revision_hash(self, reached: Iterable[Path] = (), *,
                      root: Path | None = None, also: Iterable[Path] = (),
                      missing_ok: bool = False) -> str:
        """The name of that revision, over what the build found.
        ``also`` joins files a caller frames the revision with, and a
        file the build did not look at is read here."""
        return asset_mod.content_hash(
            self.files(reached) | frozenset(also), root or self.root,
            missing_ok=missing_ok, read=self.found,
        )

    @property
    def inputs(self) -> dict:
        """What an evaluation of this target takes beside its IR, by
        keyword: the IRs of its build, the file a report names in the
        place of an IR, and the stamp an IR opened from disk must
        carry."""
        return {
            "irs": self.irs,
            "source_of": sources_mod.recipe_of,
            "stamp": version.build_version(),
        }

    @property
    def watch(self) -> frozenset[Path]:
        """The files whose edits change this target (DESIGN 4.6): every
        recipe file the build named, a requirements file not yet
        written included."""
        return self.build.named if self.build is not None else self.sources


def open_ir(path: Path) -> asset_mod.Asset:
    """An IR from disk, refused when another build wrote it (DESIGN
    4.1)."""
    sc = asset_mod.load(path)
    asset_mod.check_stamp(
        sc, version.build_version(), Path(path).name,
        source=lambda: sources_mod.recipe_of(sc.path),
    )
    return sc


def open_target(target: str | Path, *, offered: dict | None = None,
                write: bool = True, strict_offers: bool = True,
                ranged_offers: bool = True) -> Opened:
    """Open a command's target: build a recipe and its closure, or
    read an IR as given."""
    target = Path(target)
    if asset_mod.is_ir_file(target):
        sc = open_ir(target)
        root = workspace_mod.enclosing_root(asset_mod.project_root(sc.path))
        return Opened(asset=sc, irs={}, sources=frozenset({sc.path}), root=root)
    result = build_mod.build(
        target, offered=offered, write=write, strict_offers=strict_offers,
        ranged_offers=ranged_offers,
    )
    return Opened(
        asset=result.target.asset,
        irs={path: built.asset for path, built in result.assets.items()},
        sources=result.sources, root=result.root, build=result,
    )
