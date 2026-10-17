"""Asset-level aggregate over per-output stats: the one derivation the
``stats`` report and the viewer HUD both read (DESIGN 6, 7.2).

Per-value ``stats()`` (core.mesh) describes one output; this rolls the
set up into the asset numbers a human budgets against.
"""

from __future__ import annotations

from ..core.mesh import NEUTRAL_MATERIAL_NAME


def asset_rollup(stats_by_name: dict[str, dict], node_count: int) -> dict:
    """Asset-level aggregate over per-output stats (token economy, DESIGN 8.2).

    ``verts``/``tris`` are rendered totals (instance counts multiplied
    in) — the asset-budget number — where the per-output entries report
    the prototype. ``parts`` counts exportable parts by the DESIGN 4.1
    budget vocabulary: mesh outputs, an instanced output as one part, an
    image output as none. Image outputs carry no geometry and are
    excluded from the geometry aggregates. ``materials`` lists the
    material identities the asset ships (every mesh output,
    reference-borne identities included — the budget's scope, DESIGN
    4.1) by name, from cached stats alone — no value loads (DESIGN 8.2
    tiering); an unmaterialed part ships the neutral default, so it
    counts as one. Content identity (the budget gate and the uniqueness
    gate) is validate's job — the two counts agree whenever the
    uniqueness gate is green.
    """
    meshes = {n: s for n, s in stats_by_name.items() if s.get("kind") != "image"}
    roll: dict = {
        "outputs": len(stats_by_name),
        "parts": len(meshes),
        "nodes": node_count,
        "instances": sum(s["instances"] for s in meshes.values() if "instances" in s),
        "verts": sum(s["verts"] * s.get("instances", 1) for s in meshes.values()),
        "tris": sum(s["tris"] * s.get("instances", 1) for s in meshes.values()),
    }
    if meshes:
        by_name: dict[str, list[str]] = {}
        for n, s in meshes.items():
            by_name.setdefault(s.get("material") or NEUTRAL_MATERIAL_NAME, []).append(n)
        roll["materials"] = {
            "count": len(by_name),
            "identities": [
                {"name": k, "outputs": v} for k, v in sorted(by_name.items())
            ],
        }
        mins = [s["bounds"]["min"] for s in meshes.values()]
        maxs = [s["bounds"]["max"] for s in meshes.values()]
        # A null component (non-finite geometry, JSON-safe per mesh.stats)
        # makes that axis of the union null too — never a number that
        # pretends the overflowed output is not there.
        lo = [_axis_extreme(v, min) for v in zip(*mins)]
        hi = [_axis_extreme(v, max) for v in zip(*maxs)]
        roll["bounds"] = {
            "min": lo,
            "max": hi,
            "size": [
                None if a is None or b is None else round(b - a, 6)
                for a, b in zip(lo, hi)
            ],
        }
    return roll


def _axis_extreme(values, pick):
    return None if any(v is None for v in values) else pick(values)
