"""Machine-readable errors (DESIGN P2: terse, structured)."""

from __future__ import annotations


class MoraineError(Exception):
    """Base error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str, **detail):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.detail:
            d["detail"] = self.detail
        return d


class AssetError(MoraineError):
    pass


class OpError(MoraineError):
    pass


class BuildError(MoraineError):
    """A recipe the build refuses (DESIGN 4.8). ``detail`` carries the
    recipe ``site`` and, where one edit settles it, the ``fix``."""
