"""Recipe files on disk (DESIGN 4.2): which file is an asset, which
asset is a project's root, and the asset file an IR was built from.

Everything here is about the files a recipe is written in. The IR's own
layout under ``build/`` is ``core.asset``'s, and the runtime past the
build reads that alone.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import AssetError
from .asset import (
    BUILD_DIR, LIB_DIR, ASSET_EXT, SUBASSETS_DIR, VARIANT_STEM,
    ir_home,
)

# An asset file is a recipe that is not a requirements file; the suffix
# tells the two apart wherever they sit.
RECIPE_EXT = ".star"
REQUIREMENTS_EXT = ".requirements.star"


# Extensions compare case-folded: the supported platforms' filesystems
# are case-insensitive by default, so ``Yard.STAR`` is an asset file
# wherever ``yard.star`` is.


def is_requirements_file(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(REQUIREMENTS_EXT)


def is_asset_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(RECIPE_EXT) and not name.endswith(REQUIREMENTS_EXT)


def requirements_file(asset_file: Path) -> Path:
    """The requirements file that belongs to an asset file (DESIGN 5.2):
    ``<name>.requirements.star`` beside ``<name>.star``."""
    stem = asset_file.name[: -len(RECIPE_EXT)]
    return asset_file.with_name(stem + REQUIREMENTS_EXT)


def asset_files(path: Path) -> list[Path]:
    """The asset files at a directory's top level (DESIGN 4.2): every
    ``*.star`` that is not a requirements file, sorted."""
    return sorted(
        p for p in Path(path).glob(f"*{RECIPE_EXT}")
        if p.is_file() and is_asset_file(p)
    )


def root_asset(project_dir: Path) -> Path:
    """The project's root asset: its one top-level asset file (DESIGN
    4.2). A second one is refused, naming where definitions and parts
    live."""
    found = asset_files(project_dir)
    if len(found) == 1:
        return found[0]
    if not found:
        raise AssetError(
            "asset_not_found",
            f"no asset file in '{project_dir}': a project holds one "
            f"top-level <name>{RECIPE_EXT}; create it or pass its path",
        )
    raise AssetError(
        "ambiguous_asset",
        f"'{project_dir}' holds {len(found)} top-level asset files "
        f"({', '.join(p.name for p in found)}); a project has one root "
        f"asset - move definitions under {LIB_DIR}/ and parts "
        f"under {SUBASSETS_DIR}/",
        assets=[p.name for p in found],
    )


def resolve(path_arg: str | Path) -> Path:
    """Resolve a CLI argument to a command's target (DESIGN 6): a
    project directory's root asset, or the file given, a recipe or an
    IR handed over as it is."""
    p = Path(path_arg)
    if p.is_file():
        return p
    if p.is_dir():
        return root_asset(p)
    raise AssetError("asset_not_found", f"no such file or directory: '{p}'")


def recipe_of(ir_path: Path) -> Path | None:
    """The asset file an IR under a ``build/`` was built from; None for
    an IR outside one, which names no recipe."""
    path = Path(ir_path).resolve()
    home = ir_home(path)
    if home.name.lower() != BUILD_DIR:
        return None
    sub = path.parent.name.lower() == SUBASSETS_DIR
    stem = path.name[: -len(ASSET_EXT)]
    if VARIANT_STEM.search(stem):
        stem = stem.rsplit(".", 1)[0]
    where = home.parent / SUBASSETS_DIR if sub else home.parent
    return where / f"{stem}{RECIPE_EXT}"
