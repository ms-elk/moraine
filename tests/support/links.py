"""Links on disk, for the containment tests (DESIGN 4.5, 4.6, 4.8)."""

from __future__ import annotations

from pathlib import Path


def link_dir(src: Path, dst: Path) -> None:
    """Directory junction (Windows, no admin) or symlink (POSIX)."""
    try:
        import _winapi

        _winapi.CreateJunction(str(src), str(dst))
    except ImportError:
        dst.symlink_to(src, target_is_directory=True)
