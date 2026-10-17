"""Deterministic geometry validation (verify tier 1 — DESIGN 8.3).

Zero-token checks the engine can settle without stats or pixels. Every
failing check states the shortest path to a fix (DESIGN 8.2). Profile
validation (container conformance etc.) lives with the export profiles
(DESIGN 9); these are the profile-independent invariants.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..core import asserts as vocab
from ..core.mesh import (
    DEGEN_HEIGHT_TOL,
    NEUTRAL_MATERIAL_NAME,
    Image,
    Instances,
    Value,
    degenerate_tri_count,
    proto_of,
)
from . import bounds, solids
from .asserts import HONESTY_KEYS, check_asserts, warnings as assert_warnings
from .asserts.common import unchecked_instanced

# Real-world scale sanity range for a single asset, meters.
_SCALE_MIN, _SCALE_MAX = 0.001, 1000.0

# Ground tolerance, meters (DESIGN 4.1 ground checks): asset-level
# ``grounded`` passes when the asset's true minimum y is at most this
# above 0; per-output ``not_sunk`` fails below -this.
GROUND_TOL = 1e-3


def _check(name: str, ok: bool, message: str | None = None, **detail) -> dict:
    c: dict = {"check": name, "ok": bool(ok)}
    if not ok and message:
        c["fix"] = message
    if detail:
        c["detail"] = detail
    return c


def _min_y(value: Value) -> float:
    """True minimum y of a geometry value, measured on actual vertices.

    For Instances the conservative AABB-corner bounds() can dip below the
    real surface (a compound-rotated curved prototype reads ~40 mm low at
    8°+8°), so both ground checks measure vertices. Only min y is needed,
    so it is the y row of each instance matrix against the prototype —
    cheap at a tier that already loads the mesh. stats() bounds stay the
    fast conservative AABB; the ground checks are the authority on
    ground contact.
    """
    if isinstance(value, Instances):
        v = value.prototype.vertices.astype(np.float64)
        xf = value.transforms
        ys = v @ xf[:, 1, :3].T + xf[:, 1, 3]  # (n, k) transformed y only
        return float(ys.min())
    lo, _hi = value.bounds()
    return float(lo[1])


def _check_not_sunk(value: Value, output: str, policy: dict | None) -> dict:
    """Below-ground check (DESIGN 4.1): the output's true minimum y no
    lower than GROUND_TOL below the ground plane, or waived.

    The sinking direction of the ground split: sinking is an output
    property — the report names the culprit and the fix is one upward
    translate — while resting on the ground is an asset property
    (``_check_asset_grounded``). The waiver documents geometry that
    intentionally extends below ground (a foundation, a terrain slab).
    """
    if policy is not None and policy.get("not_sunk") is False:
        return {"check": "not_sunk", "ok": True, "waived": True}
    min_y = _min_y(value)
    if min_y < -GROUND_TOL:
        return _check(
            "not_sunk", False,
            f"sinks {-min_y:.6g} m below the ground plane (min y = "
            f"{min_y:.6g}); translate +{-min_y:.6g} in y, or waive with "
            f'mo.waive("{output}", "not_sunk") if it '
            f"intentionally extends below ground",
            min_y=round(min_y, 6),
        )
    return _check("not_sunk", True)


def _check_asset_grounded(
    values: dict[str, Value],
    per_checks: dict[str, list[dict]],
    asset_checks: dict | None,
) -> dict | None:
    """Asset-level ground contact (DESIGN 4.1 asset_checks): the true
    minimum y across all geometry outputs reaches the ground plane —
    within GROUND_TOL of 0, from above or below.

    The floating direction of the ground split: red means the whole
    asset hovers, and the fix is one downward translate of the asset.
    A sunk output already reaches the plane, so it satisfies this check
    and is named by its own ``not_sunk`` entry — one culprit, one red.
    The waiver documents support outside the asset (a hanging fixture,
    a parts-bin sub-asset). None = no geometry outputs to measure.
    """
    eligible = [
        v for name, v in values.items()
        if not isinstance(v, Image) and _finite_ok(per_checks[name])
    ]
    if not eligible:
        return None
    if asset_checks is not None and asset_checks.get("grounded") is False:
        return {"check": "grounded", "ok": True, "waived": True}
    min_y = min(_min_y(v) for v in eligible)
    if min_y > GROUND_TOL:
        return _check(
            "grounded", False,
            f"the asset floats {min_y:.6g} m above the ground plane "
            f"(lowest output min y = {min_y:.6g}); translate the asset "
            f"-{min_y:.6g} in y, or waive with "
            f'mo.waive_asset("grounded") if its support is '
            f"outside this asset",
            min_y=round(min_y, 6),
        )
    return _check("grounded", True)


def validate_value(value: Value) -> list[dict]:
    checks: list[dict] = []
    if isinstance(value, Image):
        # Texture buffers: geometry checks do not apply; finiteness does.
        finite = bool(np.isfinite(value.pixels).all())
        checks.append(_check("finite", finite, "image has NaN/inf pixels; check op params"))
        return checks
    proto = proto_of(value)
    v = proto.vertices.astype(np.float64)
    f = proto.faces.astype(np.int64)

    # Finiteness covers everything the value carries: vertices, normals,
    # UVs, extra attribute slots, instance transforms, and bound-material
    # factors and texture pixels — NaN in any of them would otherwise slip
    # through every threshold comparison downstream (never-silent, P6).
    bad = [] if np.isfinite(v).all() else ["vertices"]
    if proto.normals is not None and not np.isfinite(proto.normals).all():
        bad.append("normals")
    if proto.uv is not None and not np.isfinite(proto.uv).all():
        bad.append("uv")
    for slot in sorted(proto.attributes or {}):
        arr = proto.attributes[slot]
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            bad.append(f"attribute '{slot}'")
    if isinstance(value, Instances) and not np.isfinite(value.transforms).all():
        bad.append("instance transforms")
    if proto.material is not None:
        factors = [
            *proto.material.base_color_factor,
            proto.material.metallic_factor,
            proto.material.roughness_factor,
            *proto.material.emissive_factor,
            proto.material.normal_scale,
        ]
        if not np.isfinite(factors).all():
            bad.append("material factors")
        for slot in sorted(proto.material.textures):
            if not np.isfinite(proto.material.textures[slot].pixels).all():
                bad.append(f"material texture '{slot}'")
    checks.append(
        _check(
            "finite", not bad,
            f"NaN/inf in {', '.join(bad)}; check op params" if bad else None,
        )
    )
    if bad:
        return checks  # everything downstream would be noise

    degen = degenerate_tri_count(v, f)
    checks.append(
        _check(
            "no_degenerate_tris", degen == 0,
            f"{degen} triangle(s) thinner than {DEGEN_HEIGHT_TOL:g} m; usually "
            f"exactly-coplanar boolean operands, duplicate points, or a "
            f"collapsed scale",
            count=degen,
        )
    )

    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    tm.merge_vertices()
    checks.append(
        _check(
            "watertight", bool(tm.is_watertight),
            "surface has open boundary edges; extrude/close it before boolean or export",
        )
    )

    if isinstance(value, Instances):
        lo, hi = value.bounds()
    else:
        lo, hi = v.min(axis=0), v.max(axis=0)
    dim = float((hi - lo).max())
    checks.append(
        _check(
            "scale_sane", _SCALE_MIN <= dim <= _SCALE_MAX,
            f"largest dimension is {dim:.6g} m - outside [{_SCALE_MIN}, {_SCALE_MAX}]; "
            f"moraine units are meters (DESIGN 4.3), check size params",
            largest_dim_m=round(dim, 6),
        )
    )

    if isinstance(value, Instances):
        checks.append(_check("instancing_intact", True, instances=len(value.transforms)))
    return checks


def _finite_ok(checks: list[dict]) -> bool:
    return all(c["ok"] for c in checks if c["check"] == "finite")


def _opaque(material) -> bool:
    """Can this material hide what is behind it? (Conservative.)

    The neutral default is opaque. A base-color alpha factor below 1
    makes an output ineligible as an enclosure container — false
    negatives are fine here, false "invisible" verdicts are not. Textures
    carry no alpha (a base-color texture is 3-channel by contract), so
    the factor is the whole transparency signal.
    """
    if material is None:
        return True
    return bool(material.base_color_factor[3] >= 1.0)


# The material channels a verdict reads for more than finiteness and
# identity (DESIGN 7.2): a base color's alpha decides what encloses
# (``not_enclosed``). A check that comes to read another channel names
# it here, since the viewer's accept validates nothing for a pick of
# the others alone.
VERDICT_CHANNELS = frozenset({"base_color"})


def _append_material_name_checks(
    values: dict[str, Value],
    per_checks: dict[str, list[dict]],
    origins: dict[str, dict] | None,
) -> None:
    """Material-name uniqueness: the ``material_name_unique`` check.

    A material name is export identity (DESIGN 4.1/9.1): one name bound
    to two different contents is a GLB writer hard error, so plain
    ``validate`` gates it — green must never outrun ``export``. Every
    output carrying a colliding name fails; the fix routes to ``asset_ref``
    ``material_rename`` when the output's material arrives through a
    reference (``origins[name]["refs"]`` names the ref nodes — the
    consumer's only handle on a name authored elsewhere), else to the
    local rename or content merge. The fix names every colliding side
    with its route and offers either: which side deviates from the
    design is the author's call, and steering at the majority side would
    invert a deliberate variant. Never waivable: the export failure it
    mirrors is not waivable either.
    """
    by_name: dict[str, dict[str, list[str]]] = {}  # name -> content hash -> outputs
    for name, value in values.items():
        if isinstance(value, Image):
            continue
        mat = proto_of(value).material
        if mat is not None:
            by_name.setdefault(mat.name, {}).setdefault(mat.content_hash(), []).append(name)
    for mat_name, by_hash in by_name.items():
        if len(by_hash) < 2:
            continue
        involved = sorted(n for outs in by_hash.values() for n in outs)
        refs_of = {
            n: ((origins or {}).get(n) or {}).get("refs") or [] for n in involved
        }
        sides = ", ".join(
            f"'{n}' (via asset_ref {refs_of[n]})" if refs_of[n] else f"'{n}' (authored here)"
            for n in involved
        )
        routes = []
        if any(refs_of.values()):
            routes.append(
                f"set \"material_rename\": {{\"{mat_name}\": \"<new name>\"}} on the "
                f"asset_ref node of the reference-borne side"
            )
        if not all(refs_of.values()):
            routes.append("rename the material node of the locally authored side")
        fix = (
            f"material name '{mat_name}' is bound to different params across "
            f"outputs {sides}; material names are export identity - give the "
            f"side that deviates from the design its own identity "
            f"({' / '.join(routes)}), or make the params identical"
        )
        for name in involved:
            per_checks[name].append(
                _check(
                    "material_name_unique", False, fix,
                    material=mat_name, outputs=[n for n in involved if n != name],
                )
            )


def check_budgets(
    values: dict[str, Value],
    budgets: dict[str, int],
) -> list[dict]:
    """Asset-declared budgets (DESIGN 4.1/9.1), gating like asserts.

    ``materials``: distinct material identities the asset ships —
    content identities (name + params + textures) over every mesh
    output, identities arriving through a reference included (a budget
    bounds the exported asset, DESIGN 4.1; shared identities count
    once, so deliberate kit-wide sharing is never penalized), plus the
    neutral default once when any output binds no material (it ships
    as a real glTF material). Each identity reports the outputs
    binding it and whether it is structurally its own identity
    (transparent or emissive materials cannot share with opaque ones),
    so the fix can tell "share these" from "this one cannot merge".
    """
    unknown = set(budgets) - {"materials"}
    if unknown:  # the loader rejects these; a direct caller must not get a silent pass
        raise ValueError(f"unknown budget(s) {sorted(unknown)}")
    limit = budgets.get("materials")
    if limit is None:
        return []
    by_hash: dict[str, dict] = {}
    for name, value in values.items():
        if isinstance(value, Image):
            continue
        mat = proto_of(value).material
        key, mat_name, transparent, emissive = (
            ("neutral", NEUTRAL_MATERIAL_NAME, False, False) if mat is None
            else (mat.content_hash(), mat.name, not _opaque(mat), mat.is_emissive())
        )
        ident = by_hash.setdefault(key, {
            "name": mat_name, "outputs": [],
            "transparent": transparent, "emissive": emissive,
        })
        ident["outputs"].append(name)
    for ident in by_hash.values():
        ident["outputs"].sort()
    identities = sorted(by_hash.values(), key=lambda i: (i["name"], i["outputs"]))
    count = len(identities)
    entry: dict = {
        "budget": "materials", "ok": count <= limit, "count": count, "limit": limit,
        "identities": identities,
    }
    if count > limit:
        structural = [i["name"] for i in identities if i["transparent"] or i["emissive"]]
        fix = (
            f"{count} distinct material identities exceed the declared budget "
            f"of {limit}; share one material name+params across parts where "
            f"the look allows, or raise \"budgets\": {{\"materials\": {count}}} "
            f"in the asset if the budget was wrong"
        )
        if structural:
            fix += (
                f" (transparent/emissive identities cannot merge with opaque "
                f"ones: {structural})"
            )
        entry["fix"] = fix
    return [entry]


# Relative volume slack for "intersection == inner": the manifold round-trip
# stores float32 vertices, so exact equality is not available.
_ENCLOSE_VOL_RTOL = 1e-4


def _encloses(container, inner, container_vol: float | None = None) -> bool:
    """True iff manifold ``inner`` lies entirely inside ``container``.

    The container must be strictly larger: coincident duplicates would
    otherwise "enclose" each other and both flag. (A coincident duplicate
    is its own kind of defect — z-fighting — but not this check's.)
    ``container_vol`` lets a per-instance caller hoist the loop-invariant
    container volume.
    """
    vol = solids.manifold_volume(inner)
    if container_vol is None:
        container_vol = solids.manifold_volume(container)
    if container_vol <= vol * (1.0 + _ENCLOSE_VOL_RTOL):
        return False
    inter = container ^ inner
    if inter.is_empty():
        return False
    return solids.manifold_volume(inter) >= vol * (1.0 - _ENCLOSE_VOL_RTOL)


def _append_enclosure_checks(
    values: dict[str, Value],
    per_checks: dict[str, list[dict]],
    checks_policy: dict[str, dict] | None,
    proto_manifold=None,
    copy_bounds=None,
) -> None:
    """Full-enclosure detection: the ``not_enclosed`` check.

    An output completely inside another opaque output contributes no
    visible pixels while passing every structural check (DESIGN §4.1).
    Containment is exact, via the manifold3d kernel: enclosed iff
    volume(inner ∩ container) equals volume(inner) (within float32
    round-trip slack). Candidate
    containers are single-Mesh outputs that are finite, watertight, and
    opaque (`_opaque`); Instances are tested as inner candidates — the
    output fails only when *every* instance is enclosed — never as
    containers (per-instance enclosure has no single answer). An inner
    surface that is not watertight cannot be built as a solid and is
    skipped: the check under-reports, it never false-fails.

    Gating, waivable per output like ``not_sunk``
    (``"checks": {"<output>": {"not_enclosed": false}}`` documents an
    intentional interior part). The check entry appears only on failure,
    waiver, or an unresolved copy — absence means "no opaque container
    encloses it". An instanced output is judged over all of its copies
    together, so one fully swallowed copy passes; when some copy's box
    lies inside an opaque container's box and the output did not fail,
    a green entry names the output under ``unchecked.instanced_copies``
    (the asserts convention — the report says what it did not
    quantify, never a silent gap; per-copy enclosure is not tested).
    """
    # Exact-vertex AABBs for the cheap prefilter; manifolds built lazily —
    # most validates never run a single CSG intersection.
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, v in values.items():
        if isinstance(v, Image) or not _finite_ok(per_checks[name]):
            continue
        boxes[name] = bounds.world_bounds(v)
    manifolds: dict[str, object | None] = {}

    def manifold_of(name: str):
        # Prefer the caller's shared memo (validate_outputs shares one
        # prototype-solid build between this check and check_asserts —
        # each build re-welds the mesh, ~seconds on large outputs).
        if proto_manifold is not None:
            return proto_manifold(name)
        if name not in manifolds:
            mesh = proto_of(values[name])
            manifolds[name] = solids.to_manifold(mesh.vertices, mesh.faces)
        return manifolds[name]

    def enclosed_by(name: str, cname: str) -> bool:
        container = manifold_of(cname)
        if isinstance(container, solids.NoSolid):
            return False
        inner = manifold_of(name)
        if isinstance(inner, solids.NoSolid):
            return False
        value = values[name]
        if isinstance(value, Instances):
            vol_container = solids.manifold_volume(container)
            return all(
                _encloses(
                    container,
                    inner.transform(np.ascontiguousarray(t[:3, :4])),
                    vol_container,
                )
                for t in value.transforms
            )
        return _encloses(container, inner)

    containers = {
        cname: box for cname, box in boxes.items()
        if not isinstance(values[cname], Instances) and _opaque(values[cname].material)
    }
    for name in boxes:
        policy = (checks_policy or {}).get(name)
        if policy is not None and policy.get("not_enclosed") is False:
            per_checks[name].append({"check": "not_enclosed", "ok": True, "waived": True})
            continue
        lo, hi = boxes[name]
        failed = False
        for cname, (clo, chi) in containers.items():
            if cname == name:
                continue
            if not (np.all(lo >= clo) and np.all(hi <= chi)):
                continue
            if enclosed_by(name, cname):
                per_checks[name].append(
                    _check(
                        "not_enclosed", False,
                        f"completely enclosed in opaque output '{cname}' - "
                        f"contributes no visible pixels; reposition it, use it "
                        f"as a boolean cutter, or waive with "
                        f'mo.waive("{name}", "not_enclosed") '
                        f"if intentional",
                        container=cname,
                    )
                )
                failed = True
                break
        if failed or not isinstance(values[name], Instances):
            continue
        # The union verdict says nothing about one copy sitting inside a
        # container on its own: name the output as unquantified when a
        # copy's exact box lies inside some container's box.
        if not containers:
            continue
        klo, khi = (
            copy_bounds(name) if copy_bounds is not None
            else bounds.copy_world_bounds(values[name])
        )
        for clo, chi in containers.values():
            if bool(np.any(np.all(klo >= clo, axis=1) & np.all(khi <= chi, axis=1))):
                entry = _check("not_enclosed", True)
                unchecked_instanced(entry, "instanced_copies", [name])
                per_checks[name].append(entry)
                break


def validate_outputs(
    values: dict[str, Value],
    checks_policy: dict[str, dict] | None = None,
    warnings: list[dict] | None = None,
    asserts: list[dict] | None = None,
    *,
    asset_checks: dict | None = None,
    budgets: dict[str, int] | None = None,
    origins: dict[str, dict] | None = None,
    ref_embeds: dict | None = None,
    groups: dict[str, list[str]] | None = None,
    params: list[dict] | None = None,
    sites: dict[str, dict] | None = None,
) -> dict:
    """Tier-1 report over named outputs.

    ``checks_policy`` is the asset's root "checks" object (per-output
    waivers); ``asset_checks`` the root "asset_checks" object
    (asset-level waivers), consumed by the asset-level ``grounded``
    check and reported as the top-level ``asset_checks`` list.
    ``warnings`` are advisory graph-level findings (attribute
    drops); they ride in the report but never affect ``ok``. ``asserts``
    are the asset's root assertions; they gate ``ok`` like checks, and
    the built-in entries (vocab.BUILTIN_ENTRIES, the overlap guard)
    join them on every run — overlap is checked by default, not by
    authorship.
    ``budgets`` are the asset's root budgets (DESIGN 9.1): reported under
    ``budgets`` and gating like asserts; absent = not checked.
    ``origins`` (``Evaluation.material_origins()``) says per output
    which asset_ref nodes its material identity arrives through — the
    uniqueness fix routing (None = no routing). ``ref_embeds``
    (``Evaluation.embedded_via_refs()``) extends the guard's cede with
    embedded declarations arriving through references. ``groups`` is
    the asset's root ``groups`` object, the assert-side name vocabulary
    (``check_asserts``). ``params`` are the declared knobs' range
    entries (``core.params.range_entries``, DESIGN 6): reported under
    ``params`` and gating like budgets; absent = no knob carries a range.
    ``sites`` (``Evaluation.output_sites()``) is each output's recipe
    position, carried on its entry.
    """
    per_checks: dict[str, list[dict]] = {}
    for name, value in values.items():
        checks = validate_value(value)
        # Skip the ground checks on images (no geometry) and on
        # non-finite geometry: NaN bounds compare False against every
        # threshold and would silently pass the check.
        if not isinstance(value, Image) and _finite_ok(checks):
            checks.append(
                _check_not_sunk(value, name, (checks_policy or {}).get(name))
            )
        per_checks[name] = checks
    # One prototype-solid build and one per-copy bounds pass per output,
    # shared by the enclosure check and the asserts below — a build
    # re-welds the whole mesh, a bounds pass touches every copy.
    proto_manifolds: dict[str, object | None] = {}
    copy_boxes: dict[str, tuple] = {}

    def proto_manifold(name: str):
        if name not in proto_manifolds:
            mesh = proto_of(values[name])
            proto_manifolds[name] = solids.to_manifold(mesh.vertices, mesh.faces)
        return proto_manifolds[name]

    def copy_bounds(name: str):
        if name not in copy_boxes:
            copy_boxes[name] = bounds.copy_world_bounds(values[name])
        return copy_boxes[name]

    _append_enclosure_checks(values, per_checks, checks_policy, proto_manifold, copy_bounds)
    _append_material_name_checks(values, per_checks, origins)
    out: dict[str, dict] = {}
    all_ok = True
    for name, checks in per_checks.items():
        ok = all(c["ok"] for c in checks)
        all_ok = all_ok and ok
        out[name] = {"ok": ok}
        if sites and name in sites:
            out[name].update(sites[name])
        out[name]["checks"] = checks
    report: dict = {"ok": all_ok, "outputs": out}
    asset_entry = _check_asset_grounded(values, per_checks, asset_checks)
    if asset_entry is not None:
        report["asset_checks"] = [asset_entry]
        report["ok"] = report["ok"] and asset_entry["ok"]
    guarded = vocab.builtin_entries() + list(asserts or [])
    entries = check_asserts(
        values, guarded, proto_manifold, ref_embeds=ref_embeds, groups=groups,
        copy_bounds=copy_bounds,
    )
    report["asserts"] = entries
    report["ok"] = report["ok"] and all(e["ok"] for e in entries)
    if budgets:
        budget_entries = check_budgets(values, budgets)
        report["budgets"] = budget_entries
        report["ok"] = report["ok"] and all(e["ok"] for e in budget_entries)
    if params:
        report["params"] = [dict(e) for e in params]
        report["ok"] = report["ok"] and all(e["ok"] for e in params)
    warnings = list(warnings or []) + assert_warnings(entries)
    if warnings:
        report["warnings"] = warnings
    return report


def _actionable(entry: dict) -> bool:
    """A green entry the reader still has to read: it carries one of the
    honesty keys (``HONESTY_KEYS``)."""
    return any(k in entry for k in HONESTY_KEYS)


def append_ref_asserts(report: dict, bundles: list[dict]) -> dict:
    """Extend a validate report with sub-asset assert entries (DESIGN
    4.6): each bundle (``Evaluation.ref_asserts()``) runs the built-in
    overlap guard and its own asserts in the sub-asset's namespace, once
    per (asset, override identity) — the guard covers the kit's outputs
    as the consumer's overrides shaped them, consumed or not, with the
    kit's own embeds (and those of the kits it references) ceded.
    Entries are stamped with ``asset`` — and ``overrides`` when present,
    since two identities of one file share assert ids — and gate ``ok``
    like root asserts: asserts are never waivable, and consuming an asset
    through ``asset_ref`` must not launder its asserts away.

    Legibility at consumer scale: a bundle's green entries that carry
    nothing to act on fold into one ``ref_assets`` row per (asset,
    identity) — ``asserts_total`` / ``asserts_ok`` — while failing
    entries and green ones naming an ``unchecked`` scope, a ``ceded``
    pair set, or a ``stronger`` mode stay as entries in ``asserts``. A referenced kit
    repeated under many identities costs one row per identity, not one
    line per assert.

    A bundle's knob range entries (``params``, DESIGN 6) gate the same
    way: a referenced asset's knob outside its declared range is a
    failing, stamped entry in the consumer's ``params``, and its green
    knobs fold into the row as ``params_total`` / ``params_ok``.
    """
    for b in bundles:
        guarded = vocab.builtin_entries() + list(b["asserts"])
        entries = check_asserts(
            b["values"], guarded, ref_embeds=b.get("ref_embeds"), groups=b.get("groups")
        )
        stamp = {"asset": b["asset"]}
        if "overrides" in b:
            stamp["overrides"] = b["overrides"]
        kept = []
        for e in entries:
            e.update(stamp)
            if not e["ok"] or _actionable(e):
                kept.append(e)
        report.setdefault("asserts", []).extend(kept)
        notes = assert_warnings(entries)
        if notes:
            report.setdefault("warnings", []).extend(notes)
        row = dict(stamp)
        row["asserts_total"] = len(entries)
        row["asserts_ok"] = sum(1 for e in entries if e["ok"])
        knobs = b.get("params") or []
        if knobs:
            row["params_total"] = len(knobs)
            row["params_ok"] = sum(1 for e in knobs if e["ok"])
            red = [{**e, **stamp} for e in knobs if not e["ok"]]
            if red:
                report.setdefault("params", []).extend(red)
        report.setdefault("ref_assets", []).append(row)
        report["ok"] = (
            report["ok"] and all(e["ok"] for e in entries)
            and all(e["ok"] for e in knobs)
        )
    return report


def add_by_site(report: dict) -> dict:
    """The ``by_site`` rollup (DESIGN 6): failing entries counted by
    the recipe line that made them, for the lines two or more share,
    the most first. One wrong expression in a loop then reads as one
    row with a count. Where the entries of a line were reached through
    more than one call, the row counts them by the calls on the way: a
    line every caller fails at is the line to fix, and a line one
    caller fails at leads to that call. Beside the entries, never in
    their place."""
    rows: dict[str, dict] = {}

    def count(entry: dict) -> None:
        site = entry.get("site")
        if site is None:
            return
        row = rows.setdefault(site, {"count": 0, "via": {}})
        row["count"] += 1
        chain = tuple(entry.get("via") or ())
        row["via"][chain] = row["via"].get(chain, 0) + 1

    for out in report["outputs"].values():
        if not out["ok"]:
            count(out)
    for section in ("asserts", "params"):
        for entry in report.get(section) or []:
            if not entry["ok"]:
                count(entry)
    shared = []
    for site, row in sorted(rows.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        if row["count"] < 2:
            continue
        out: dict = {"site": site, "count": row["count"]}
        if len(row["via"]) > 1:
            out["via"] = [
                {"via": list(chain), "count": n}
                for chain, n in sorted(row["via"].items(), key=lambda kv: (-kv[1], kv[0]))
            ]
        shared.append(out)
    if shared:
        report["by_site"] = shared
    return report


def _tally(tally: dict, key: str, value) -> None:
    """Count one green entry's honesty key (DESIGN 6): the entry, and
    under each name of the key what the entry holds there."""
    row = tally.setdefault(key, {"entries": 0})
    row["entries"] += 1
    if isinstance(value, dict):
        for name, held in value.items():
            n = len(held) if isinstance(held, (list, dict)) else 1
            row[name] = row.get(name, 0) + n
    else:
        row[str(value)] = row.get(str(value), 0) + 1


def failures_only(report: dict) -> dict:
    """Compress a report to its failures (token economy, DESIGN 8.2).

    ``outputs_ok`` counts outputs whose gating ``ok`` is true — the same
    ``ok`` shown per output, so the numbers never contradict the
    entries. An output is kept only if it has checks with ``ok: false``
    — including advisory ``level: "warn"`` entries from profile checks,
    which carry fixes and do not gate (such an output is kept *and*
    counted ok) — and then keeps only those checks. ``ok`` and
    ``warnings`` pass through unchanged, so gating and exit codes are
    identical to the full report's.

    The honesty machinery survives the compression as counts: a green
    entry that carries an honesty key (``HONESTY_KEYS``) is counted
    under that key, with what it holds counted by name, whether the
    report is green or red. The rows are the full report's, and a
    failing entry keeps its own keys whole. The per-identity
    ``ref_assets`` counts, the ``mates`` provenance rows, the
    ``by_site`` rollup, and the ``consumers`` of the project pass
    through. A report that says less still says that something was not
    checked, and how much.
    """
    kept: dict[str, dict] = {}
    ok_count = 0
    rollups: dict[str, dict] = {}

    def roll_up(entry: dict) -> None:
        for key in HONESTY_KEYS:
            if key in entry:
                _tally(rollups, key, entry[key])

    for name, entry in report["outputs"].items():
        ok_count += bool(entry["ok"])
        bad = [c for c in entry["checks"] if not c["ok"]]
        if bad:
            kept[name] = {"ok": entry["ok"], "checks": bad}
        for c in entry["checks"]:
            if c["ok"]:
                roll_up(c)
    out: dict = {
        "ok": report["ok"],
        "outputs_total": len(report["outputs"]),
        "outputs_ok": ok_count,
    }
    if kept:
        out["outputs"] = kept
    # The gating lists: counts + failures.
    for section in ("asset_checks", "asserts", "budgets", "params"):
        if section in report:
            failed = [e for e in report[section] if not e["ok"]]
            out[f"{section}_total"] = len(report[section])
            out[f"{section}_ok"] = len(report[section]) - len(failed)
            if failed:
                out[section] = failed
    # The knobs that hold an accepted value, by name (DESIGN 6): a
    # default the agent edits moves nothing while one stands, and the
    # green entry that says so is among the ones counted away.
    accepted = [
        e["param"] for e in report.get("params") or []
        if e.get("from") == "accepted" and "asset" not in e
    ]
    if accepted:
        out["params_accepted"] = accepted
    for e in report.get("asserts") or []:
        if e["ok"]:
            roll_up(e)
    for key in HONESTY_KEYS:
        if key in rollups:
            out[key] = rollups[key]
    for section in ("ref_assets", "mates", "by_site", "consumers"):
        if report.get(section):
            out[section] = report[section]
    if report.get("warnings"):
        out["warnings"] = report["warnings"]
    return out
