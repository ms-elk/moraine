"""moraine — AI-native procedural modeling tool for real-time 3D assets.

Nothing importable from this package is a public API (DESIGN 3): use
the ``moraine`` command. Modules move between releases without notice.
"""

from importlib import metadata as _im


def dist_version(name: str = "moraine3d") -> str:
    try:
        return _im.version(name)
    except _im.PackageNotFoundError:  # pragma: no cover - dev installs
        return "unknown"


def __getattr__(name: str) -> str:
    # Lazy so plain imports never pay a metadata scan; pyproject stays
    # the single source of the version (no hand-kept copy to drift).
    if name == "__version__":
        return dist_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
