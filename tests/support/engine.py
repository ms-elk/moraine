"""The engine as an entry point calls it (DESIGN 3).

An evaluation takes from its entry point the file a report names in an
IR's place and the stamp an IR opened from disk must carry. These hand
in what the CLI hands in, so a test reads the reports a user reads.
"""

from __future__ import annotations

from moraine3d.core.sources import recipe_of
from moraine3d.engine import evaluator, export
from moraine3d.recipe import version


def Evaluation(sc, *args, **kw) -> evaluator.Evaluation:
    kw.setdefault("source_of", recipe_of)
    kw.setdefault("stamp", version.build_version())
    return evaluator.Evaluation(sc, *args, **kw)


def evaluate_records(sc, only: str | None = None) -> dict:
    return Evaluation(sc).records(only)


def evaluate_asset(sc, only: str | None = None) -> dict:
    return Evaluation(sc).results(only)


def export_asset(sc, *args, **kw) -> dict:
    if kw.get("evaluation") is None:
        kw["evaluation"] = Evaluation(sc)
    return export.export_asset(sc, *args, **kw)
