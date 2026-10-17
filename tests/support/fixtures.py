"""IR fixtures made evaluable as they are (DESIGN 10).

The round trip compares what a fixture evaluates to with what the
build of its flat recipe evaluates to. The fixture side never runs a
recipe: a reference that carries overrides opens the variant its
overrides define, and here that variant is made by patching the knobs
the fixture records as reaching one node param, which by that record
equals a rebuild (DESIGN 7.2).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.ops import asset_ref as ref_op


def realize(fixture: Path, dest: Path) -> Path:
    """Copy a fixture project to ``dest`` and write the variant IR of
    every overridden reference in it. Returns the root IR."""
    shutil.copytree(fixture, dest)
    [root] = sorted(p for p in dest.glob("*.mora") if p.is_file())
    pending = [root, *sorted((dest / "subassets").glob("*.mora"))]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        for node in doc["nodes"]:
            overrides = node["params"].get("overrides") if node["op"] == "asset_ref" else None
            if not overrides:
                continue
            source = dest / node["params"]["asset"]
            sc = asset_mod.load(source)
            params_mod.apply_overrides(sc, overrides)
            stem = source.name[: -len(".mora")]
            variant = source.with_name(
                f"{stem}.{ref_op.override_identity(overrides)}.mora"
            )
            with open(variant, "w", encoding="utf-8", newline="\n") as f:
                f.write(asset_mod.dumps(sc))
            pending.append(variant)
    return root
