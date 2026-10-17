"""Deterministic validation checks (verify tier 1)."""

import json

import numpy as np

from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Material
from moraine3d.engine import evaluator
from moraine3d.engine.evaluator import OpContext
from moraine3d.ops import array, box, grid, scatter, transform, unwrap
from moraine3d.verify.validate import failures_only, validate_outputs, validate_value
from support.engine import Evaluation


def _by_name(checks):
    return {c["check"]: c for c in checks}


def _grounded_box():
    # Unit box lifted to rest on the ground plane.
    return transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 0.5, 0.0]}, None
    )


def test_closed_box_passes():
    report = validate_outputs({"main": _grounded_box()})
    assert report["ok"] is True
    assert report["outputs"]["main"]["ok"] is True


def test_open_grid_fails_watertight_with_fix():
    checks = _by_name(validate_value(grid.evaluate([], {}, None)))
    assert checks["watertight"]["ok"] is False
    assert "extrude" in checks["watertight"]["fix"]


def test_out_of_scale_flagged():
    m = box.evaluate([], {"size": [5000.0, 1.0, 1.0]}, None)
    checks = _by_name(validate_value(m))
    assert checks["scale_sane"]["ok"] is False
    assert "meters" in checks["scale_sane"]["fix"]


def test_instances_report_count():
    inst = array.evaluate(
        [box.evaluate([], {}, None)], {"count": 5, "step_translate": [2.0, 0.0, 0.0]}, None
    )
    checks = _by_name(validate_value(inst))
    assert checks["instancing_intact"]["detail"]["instances"] == 5
    assert checks["watertight"]["ok"] is True  # prototype is checked


def _one_triangle(height: float):
    from moraine3d.core.mesh import Mesh

    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]], dtype=np.float32)
    f = np.array([[0, 1, 2]], dtype=np.int32)
    return Mesh(vertices=v, faces=f)


def test_near_degenerate_sliver_fails_locally():
    """A sliver thinner than 1 nm fails no_degenerate_tris in its producing
    asset, even with finite area — the bar is relative (height over the
    longest edge), so it cannot pass locally and collapse under an
    asset_ref transform downstream (a latent coplanarity defect an
    absolute-area threshold lets through; DESIGN §4.1)."""
    checks = _by_name(validate_value(_one_triangle(1e-10)))
    c = checks["no_degenerate_tris"]
    assert c["ok"] is False
    assert c["detail"]["count"] == 1
    assert "thinner" in c["fix"] and "coplanar" in c["fix"]


def test_near_degenerate_bar_leaves_thin_but_honest_triangles():
    checks = _by_name(validate_value(_one_triangle(1e-7)))
    assert checks["no_degenerate_tris"]["ok"] is True


# -- ground checks (DESIGN 4.1) ------------------------------------


def test_not_sunk_penetration_fails_with_fix():
    """DESIGN 4.1: sinking is an output property — the report names the
    culprit and the fix is one upward translate; the sunk output already
    reaches the ground plane, so asset-level grounded stays green (one
    culprit, one red)."""
    report = validate_outputs({"main": box.evaluate([], {}, None)})  # centered: sinks 0.5
    checks = _by_name(report["outputs"]["main"]["checks"])
    g = checks["not_sunk"]
    assert g["ok"] is False
    assert "below the ground plane" in g["fix"]
    assert "translate +0.5" in g["fix"]
    assert 'mo.waive("main", "not_sunk")' in g["fix"]  # the recipe's waiver names the output
    assert g["detail"]["min_y"] == -0.5
    assert report["asset_checks"] == [{"check": "grounded", "ok": True}]
    assert report["ok"] is False


def test_elevated_output_passes_without_declaration():
    """DESIGN 4.1: an elevated output passes with no declaration at all —
    the asset grounds through its base, and what holds the part up is
    stated as an assert, not a waiver."""
    seat = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 2.0, 0.0]}, None
    )
    report = validate_outputs({"base": _grounded_box(), "seat": seat})
    assert report["ok"] is True
    assert _by_name(report["outputs"]["seat"]["checks"])["not_sunk"]["ok"] is True
    assert report["asset_checks"] == [{"check": "grounded", "ok": True}]


def test_asset_grounded_floating_asset_fails_with_fix():
    """DESIGN 4.1 asset_checks: resting on the ground is an asset
    property — red means the whole asset hovers, and the fix is one
    downward translate of the asset."""
    floating = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 2.0, 0.0]}, None
    )
    report = validate_outputs({"lamp": floating})
    assert _by_name(report["outputs"]["lamp"]["checks"])["not_sunk"]["ok"] is True
    [g] = report["asset_checks"]
    assert g["ok"] is False
    assert "floats" in g["fix"] and "translate the asset -1.5" in g["fix"]
    assert 'mo.waive_asset("grounded")' in g["fix"]  # the recipe's waiver
    assert g["detail"]["min_y"] == 1.5
    assert report["ok"] is False


def test_asset_grounded_waived_passes_and_is_visible():
    """DESIGN 4.1 asset_checks: the grounded waiver documents support
    outside the asset and stays visible in the report."""
    floating = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 2.0, 0.0]}, None
    )
    report = validate_outputs({"lamp": floating}, asset_checks={"grounded": False})
    [g] = report["asset_checks"]
    assert g["ok"] is True
    assert g["waived"] is True
    assert report["ok"] is True


def test_not_sunk_waived_passes_and_is_visible():
    """DESIGN 4.1: the not_sunk waiver documents geometry that
    intentionally extends below ground (a foundation, a terrain slab)."""
    report = validate_outputs(
        {"pad": box.evaluate([], {}, None)},  # centered: extends below
        checks_policy={"pad": {"not_sunk": False}},
    )
    g = _by_name(report["outputs"]["pad"]["checks"])["not_sunk"]
    assert g["ok"] is True
    assert g["waived"] is True
    assert report["ok"] is True


def test_asset_grounded_covers_instanced_outputs():
    inst = array.evaluate(
        [_grounded_box()], {"count": 3, "step_translate": [1.5, 0.0, 0.0]}, None
    )
    report = validate_outputs({"row": inst})
    assert report["asset_checks"] == [{"check": "grounded", "ok": True}]


def test_ground_checks_measure_real_vertices_for_rotated_instances():
    # A compound-rotated curved prototype: the conservative AABB-corner
    # bounds dip ~40 mm below the true surface, so both ground checks must
    # judge the actual transformed vertices, not bounds().
    import numpy as np

    from moraine3d.ops import cylinder

    proto = cylinder.evaluate([], {"radius": 0.5, "height": 2.0, "segments": 64}, None)
    lifted = transform.evaluate([proto], {"translate": [0.0, 1.0, 0.0]}, None)
    inst = array.evaluate([lifted], {"count": 1}, None)
    tilted = transform.evaluate([inst], {"rotate_deg": [8.0, 0.0, 8.0]}, None)

    # True minimum from the actual vertices; conservative AABB reads lower.
    m = tilted.transforms[0]
    v = np.c_[lifted.vertices.astype(np.float64), np.ones(len(lifted.vertices))] @ m.T
    true_min = float(v[:, 1].min())
    assert float(tilted.bounds()[0][1]) < true_min - 0.01  # the AABB gap is real

    seated = transform.evaluate([tilted], {"translate": [0.0, -true_min, 0.0]}, None)
    report = validate_outputs({"t": seated})
    # Asset grounded would false-fail on AABB bounds (reads grounded when
    # the asset floats, but here the vertex minimum is exactly 0).
    assert report["asset_checks"] == [{"check": "grounded", "ok": True}]
    assert _by_name(report["outputs"]["t"]["checks"])["not_sunk"]["ok"] is True

    sunk = transform.evaluate([seated], {"translate": [0.0, -0.01, 0.0]}, None)
    g = _by_name(validate_outputs({"t": sunk})["outputs"]["t"]["checks"])["not_sunk"]
    assert g["ok"] is False
    import pytest

    assert g["detail"]["min_y"] == pytest.approx(-0.01, abs=1e-5)


def test_ground_checks_skipped_on_nonfinite_mesh():
    import numpy as np

    from moraine3d.core.mesh import Mesh

    bad = Mesh(
        vertices=np.array([[0.0, np.nan, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32),
        faces=np.array([[0, 1, 2]], np.int32),
    )
    report = validate_outputs({"m": bad})
    names = [c["check"] for c in report["outputs"]["m"]["checks"]]
    assert "not_sunk" not in names  # NaN bounds must not silently pass it
    assert "asset_checks" not in report  # no measurable geometry output
    assert report["ok"] is False


# -- asset "checks" root key -------------------------------------------------


def _asset_with_checks(checks_json: str):
    from moraine3d.core import asset as asset_mod

    return asset_mod.loads(
        '{"build_version": "fixture", '
        '"nodes": [{"id": "b1", "op": "box", "params": {}}], '
        '"outputs": {"main": "b1"}, '
        f'"checks": {checks_json}}}'
    )


def test_asset_checks_parse_and_round_trip():
    from moraine3d.core import asset as asset_mod

    sc = _asset_with_checks('{"main": {"not_sunk": false}}')
    assert sc.checks == {"main": {"not_sunk": False}}
    text = asset_mod.dumps(sc)
    # One entry per line, like the root lists (DESIGN 4.1).
    assert '"checks": {\n"main": {"not_sunk": false}\n}' in text
    assert asset_mod.loads(text).checks == sc.checks


def test_asset_checks_rejections_are_actionable():
    import pytest

    from moraine3d.errors import AssetError

    with pytest.raises(AssetError) as e:
        _asset_with_checks('{"nope": {"not_sunk": false}}')
    assert e.value.code == "checks_unknown_output"
    with pytest.raises(AssetError) as e:
        _asset_with_checks('{"main": {"floaty": false}}')
    assert e.value.code == "unknown_check"
    with pytest.raises(AssetError) as e:
        _asset_with_checks('{"main": {"not_sunk": true}}')
    assert e.value.code == "bad_check_value"  # waivers only: false is the one value


def test_per_output_grounded_refused_with_the_split_explained():
    """DESIGN 4.1: per-output 'grounded' no longer exists — the loader's
    error carries the redesign (the error channel is the strongest
    carrier for a rule whose moment is a load)."""
    import pytest

    from moraine3d.errors import AssetError

    with pytest.raises(AssetError) as e:
        _asset_with_checks('{"main": {"grounded": false}}')
    assert e.value.code == "unknown_check"
    assert "not_sunk" in str(e.value) and "asset_checks" in str(e.value)


def _asset_with_asset_checks(asset_checks_json: str):
    from moraine3d.core import asset as asset_mod

    return asset_mod.loads(
        '{"build_version": "fixture", '
        '"nodes": [{"id": "b1", "op": "box", "params": {}}], '
        '"outputs": {"main": "b1"}, '
        f'"asset_checks": {asset_checks_json}}}'
    )


def test_asset_checks_root_key_parses_and_round_trips_inline():
    """DESIGN 4.1: asset_checks grows per check kind, never per output,
    so it serializes as one inline object (the budgets rule)."""
    from moraine3d.core import asset as asset_mod

    sc = _asset_with_asset_checks('{"grounded": false}')
    assert sc.asset_checks == {"grounded": False}
    text = asset_mod.dumps(sc)
    assert '\n"asset_checks": {"grounded": false}' in text
    assert asset_mod.loads(text).asset_checks == sc.asset_checks


def test_asset_checks_root_key_rejections_are_actionable():
    import pytest

    from moraine3d.errors import AssetError

    with pytest.raises(AssetError) as e:
        _asset_with_asset_checks('{"not_sunk": false}')  # an output check
    assert e.value.code == "unknown_check"
    with pytest.raises(AssetError) as e:
        _asset_with_asset_checks('{"grounded": true}')
    assert e.value.code == "bad_check_value"


def test_ground_tolerance_accepts_millimeter_seat():
    nearly = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 0.5005, 0.0]}, None
    )
    report = validate_outputs({"m": nearly})
    assert report["asset_checks"] == [{"check": "grounded", "ok": True}]
    barely_sunk = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 0.4995, 0.0]}, None
    )
    checks = _by_name(validate_outputs({"m": barely_sunk})["outputs"]["m"]["checks"])
    assert checks["not_sunk"]["ok"] is True  # 0.5 mm within GROUND_TOL


# -- not_enclosed (full-enclosure detection, DESIGN §4.1) ---------------------


def _box(size, translate=None):
    m = box.evaluate([], {"size": size}, None)
    if translate is not None:
        m = transform.evaluate([m], {"translate": translate}, None)
    return m


def test_enclosed_output_fails_with_container_named():
    report = validate_outputs({"outer": _box([1.0, 1.0, 1.0]), "inner": _box([0.5, 0.5, 0.5])})
    checks = _by_name(report["outputs"]["inner"]["checks"])
    ne = checks["not_enclosed"]
    assert ne["ok"] is False
    assert ne["detail"]["container"] == "outer"
    assert "no visible pixels" in ne["fix"]
    assert 'mo.waive("inner", "not_enclosed")' in ne["fix"]  # the recipe's waiver
    assert report["outputs"]["inner"]["ok"] is False  # gating
    # The container itself is fine, and passing outputs carry no entry.
    assert "not_enclosed" not in _by_name(report["outputs"]["outer"]["checks"])


def test_enclosed_waiver_passes_and_is_visible():
    report = validate_outputs(
        {"outer": _box([1.0, 1.0, 1.0]), "inner": _box([0.5, 0.5, 0.5])},
        checks_policy={"inner": {"not_enclosed": False}},
    )
    ne = _by_name(report["outputs"]["inner"]["checks"])["not_enclosed"]
    assert ne["ok"] is True
    assert ne["waived"] is True


def test_transparent_container_does_not_enclose():
    from moraine3d.ops import material

    outer = material.evaluate(
        [_box([1.0, 1.0, 1.0])],
        {"name": "glass", "base_color": [1.0, 1.0, 1.0, 0.5]},
        None,
    )
    report = validate_outputs({"outer": outer, "inner": _box([0.5, 0.5, 0.5])})
    assert "not_enclosed" not in _by_name(report["outputs"]["inner"]["checks"])


def test_overlapping_output_is_not_enclosed():
    # Same size, offset: AABBs overlap but neither contains the other.
    report = validate_outputs(
        {"a": _box([1.0, 1.0, 1.0]), "b": _box([1.0, 1.0, 1.0], [0.4, 0.0, 0.0])}
    )
    assert "not_enclosed" not in _by_name(report["outputs"]["a"]["checks"])
    assert "not_enclosed" not in _by_name(report["outputs"]["b"]["checks"])


def test_enclosed_instances_fail_as_inner():
    inst = array.evaluate(
        [_box([0.2, 0.2, 0.2])], {"count": 2, "step_translate": [0.3, 0.0, 0.0]}, None
    )
    report = validate_outputs({"outer": _box([3.0, 3.0, 3.0]), "bolts": inst})
    ne = _by_name(report["outputs"]["bolts"]["checks"])["not_enclosed"]
    assert ne["ok"] is False
    assert ne["detail"]["container"] == "outer"


def test_instances_are_never_containers():
    inst = array.evaluate([_box([3.0, 3.0, 3.0])], {"count": 1}, None)
    report = validate_outputs({"outer": inst, "inner": _box([0.5, 0.5, 0.5])})
    assert "not_enclosed" not in _by_name(report["outputs"]["inner"]["checks"])


def test_not_enclosed_waiver_parses_in_asset_checks():
    sc = _asset_with_checks('{"main": {"not_enclosed": false}}')
    assert sc.checks == {"main": {"not_enclosed": False}}


def test_open_inner_surface_is_skipped_not_false_failed():
    # A non-watertight inner (open grid) cannot be built as a solid; the
    # check must under-report (skip), never false-fail.
    open_inner = grid.evaluate([], {}, None)  # unit plane at y=0, inside the box AABB
    report = validate_outputs({"outer": _box([3.0, 3.0, 3.0]), "sheet": open_inner})
    assert "not_enclosed" not in _by_name(report["outputs"]["sheet"]["checks"])


# -- failures_only (token economy, DESIGN 8.2) --------------------------------


def test_failures_only_collapses_green_report():
    apart = transform.evaluate([_grounded_box()], {"translate": [2.0, 0.0, 0.0]}, None)
    report = validate_outputs({"a": _grounded_box(), "b": apart})
    slim = failures_only(report)
    assert slim == {
        "ok": True, "outputs_total": 2, "outputs_ok": 2,
        "asset_checks_total": 1, "asset_checks_ok": 1,
        "asserts_total": 1, "asserts_ok": 1,  # the built-in overlap guard
    }


def test_failures_only_keeps_only_failing_checks():
    report = validate_outputs(
        {"good": _grounded_box(), "sunk": box.evaluate([], {}, None)}
    )
    slim = failures_only(report)
    assert slim["ok"] is False
    assert slim["outputs_total"] == 2
    assert slim["outputs_ok"] == 1
    assert list(slim["outputs"]) == ["sunk"]
    kept = slim["outputs"]["sunk"]["checks"]
    assert [c["check"] for c in kept] == ["not_sunk"]  # passing checks dropped
    assert "fix" in kept[0]  # the actionable part survives


def test_failures_only_keeps_failing_asset_checks():
    """DESIGN 8.2 / 4.1: the asset_checks list is a gating section —
    failures_only carries its counts and keeps only failing entries,
    like asserts and budgets."""
    floating = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.0, 2.0, 0.0]}, None
    )
    slim = failures_only(validate_outputs({"lamp": floating}))
    assert slim["ok"] is False
    assert slim["asset_checks_total"] == 1
    assert slim["asset_checks_ok"] == 0
    [g] = slim["asset_checks"]
    assert g["check"] == "grounded" and "fix" in g


def test_failures_only_keeps_advisory_warn_checks():
    # An output can be ok overall while carrying a non-gating level:"warn"
    # check (export profile advisories) — those carry fixes and must survive.
    report = validate_outputs({"m": _grounded_box()})
    report["outputs"]["m"]["checks"].append(
        {"check": "uv_present", "ok": False, "level": "warn", "fix": "unwrap before texturing"}
    )
    slim = failures_only(report)
    assert slim["ok"] is True
    assert slim["outputs_ok"] == 1  # advisory-only output still counts as ok
    assert slim["outputs"]["m"]["ok"] is True
    assert [c["check"] for c in slim["outputs"]["m"]["checks"]] == ["uv_present"]


def test_failures_only_passes_warnings_through():
    report = validate_outputs({"m": _grounded_box()}, warnings=[{"warning": "attr_drop"}])
    slim = failures_only(report)
    assert slim["warnings"] == [{"warning": "attr_drop"}]


# -- shading warnings (flat_shading, DESIGN 4.1 never-silent policy) -----------


def _shading_asset(mid_nodes: list[dict], material_input: str, source: str = "cylinder"):
    """cylinder ∪ box → [mid nodes] → material, as a loaded asset (a
    curved operand, so the boolean output has shading to restore)."""
    nodes = [
        {"id": "a", "op": source, "params": {}},
        {
            "id": "b",
            "op": "transform",
            "inputs": ["a"],
            "params": {"translate": [0.4, 0.0, 0.0]},
        },
        {"id": "u1", "op": "boolean", "inputs": ["a", "b"], "params": {}},
        *mid_nodes,
        {
            "id": "m1",
            "op": "material",
            "inputs": [material_input],
            "params": {"name": "steel"},
        },
    ]
    return asset_mod.loads(
        json.dumps(
            {"build_version": "fixture", "nodes": nodes, "outputs": {"main": "m1"}}
        )
    )


def test_validate_warns_flat_shading_boolean_to_material(tmp_path, monkeypatch):
    """A boolean output reaching material with its flat shading intact
    warns (flat_shading): boolean output is flat-shaded, so the bound
    surface loses curved shading."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = _shading_asset([], "u1")
    ev = Evaluation(sc)
    ev.records()
    warnings = ev.shading_warnings()
    assert len(warnings) == 1
    w = warnings[0]
    assert w["warning"] == "flat_shading"
    assert w["node"] == "m1" and w["booleans"] == ["u1"]
    assert "smooth_normals" in w["note"]


def test_no_flat_shading_warn_with_smooth_normals_between(tmp_path, monkeypatch):
    """smooth_normals rebuilds shading (its semantics), so a chain
    boolean → smooth_normals → material carries no flat_shading warn."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sn = {"id": "sn1", "op": "smooth_normals", "inputs": ["u1"], "params": {}}
    sc = _shading_asset([sn], "sn1")
    ev = Evaluation(sc)
    ev.records()
    assert ev.shading_warnings() == []


def test_flat_shading_warn_never_gates_ok(tmp_path, monkeypatch):
    """Warnings ride the validate report and never gate ok (DESIGN §4.1:
    the shading warn is advisory, not a check)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = _shading_asset([], "u1")
    ev = Evaluation(sc)
    values = {n: r.value() for n, r in ev.records().items()}
    # Ground contact is irrelevant here: not_sunk waived for the output.
    report = validate_outputs(
        values, checks_policy={"main": {"not_sunk": False}},
        warnings=ev.shading_warnings(),
    )
    assert report["warnings"] and report["ok"] is True


def test_flat_shading_warn_crosses_asset_ref(tmp_path, monkeypatch):
    """The flat_shading walk crosses asset_ref boundaries, and a source
    from another asset is labeled asset:id — node ids are per-file, so
    a bare foreign id would name an unfindable edit site."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    (proj / "subassets" / "kit.mora").write_text(
        json.dumps(
            {
                "build_version": "fixture",
                "nodes": [
                    {"id": "a", "op": "cylinder", "params": {}},
                    {
                        "id": "b",
                        "op": "transform",
                        "inputs": ["a"],
                        "params": {"translate": [0.4, 0.0, 0.0]},
                    },
                    {"id": "u1", "op": "boolean", "inputs": ["a", "b"], "params": {}},
                ],
                "outputs": {"part": "u1"},
            }
        ),
        encoding="utf-8",
    )
    (proj / "root.mora").write_text(
        json.dumps(
            {
                "build_version": "fixture",
                "nodes": [
                    {
                        "id": "rf01",
                        "op": "asset_ref",
                        "params": {"asset": "subassets/kit.mora", "output": "part"},
                    },
                    {
                        "id": "m1",
                        "op": "material",
                        "inputs": ["rf01"],
                        "params": {"name": "steel"},
                    },
                ],
                "outputs": {"main": "m1"},
            }
        ),
        encoding="utf-8",
    )
    ev = Evaluation(asset_mod.load(proj / "root.mora"))
    ev.records()
    warnings = ev.shading_warnings()
    assert len(warnings) == 1
    assert warnings[0]["node"] == "m1"
    assert warnings[0]["booleans"] == ["subassets/kit.mora:u1"]


def test_no_flat_shading_warn_for_reference_only_inputs(tmp_path, monkeypatch):
    """The flat_shading walk follows only surface-carrying inputs
    (CARRIED_INPUTS): scatter's target is sampled, never emitted, so a
    boolean target must not be blamed for the prototype's shading. The
    curvature gate is forced open so the walk alone decides."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(evaluator.Evaluation, "_curved", lambda self, key: True)
    sc = asset_mod.loads(
        json.dumps(
            {
                "build_version": "fixture",
                "nodes": [
                    {"id": "a", "op": "box", "params": {}},
                    {
                        "id": "b",
                        "op": "transform",
                        "inputs": ["a"],
                        "params": {"translate": [0.4, 0.0, 0.0]},
                    },
                    {"id": "u1", "op": "boolean", "inputs": ["a", "b"], "params": {}},
                    {"id": "p1", "op": "sphere", "params": {"radius": 0.05}},
                    {
                        "id": "sc1",
                        "op": "scatter",
                        "inputs": ["u1", "p1"],
                        "params": {"count": 4, "seed": 1},
                    },
                    {
                        "id": "m1",
                        "op": "material",
                        "inputs": ["sc1"],
                        "params": {"name": "steel"},
                    },
                ],
                "outputs": {"main": "m1"},
            }
        )
    )
    ev = Evaluation(sc)
    ev.records()
    assert ev.shading_warnings() == []


def test_no_flat_shading_warn_through_bevel(tmp_path, monkeypatch):
    """A shading-replacing op ends the causal chain (SHADING_POLICY):
    bevel's output shading is independent of its input, so an upstream
    boolean must not be blamed — inserting smooth_normals after that
    boolean could not change the material's surface."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sn = {"id": "bv1", "op": "bevel", "inputs": ["u1"], "params": {"width": 0.02}}
    # Planar input (bevel is convex-only); the gate is forced open so the
    # SHADING_POLICY stop, not the curvature test, is what silences it.
    monkeypatch.setattr(evaluator.Evaluation, "_curved", lambda self, key: True)
    sc = _shading_asset([sn], "bv1", source="box")
    ev = Evaluation(sc)
    ev.records()
    assert ev.shading_warnings() == []


# -- finite coverage beyond vertices ------------------------------------------


def _finite_after(mutate):
    m = unwrap.evaluate([_grounded_box()], {}, None)
    mutate(m)
    return _by_name(validate_value(m))["finite"]


def test_finite_covers_normals_uv_attributes():
    c = _finite_after(lambda m: m.normals.__setitem__((0, 0), np.nan))
    assert c["ok"] is False and "normals" in c["fix"]

    c = _finite_after(lambda m: m.uv.__setitem__((0, 0), np.inf))
    assert c["ok"] is False and "uv" in c["fix"]

    def poison_attr(m):
        m.attributes = {"COLOR_0": np.full((len(m.vertices), 4), np.nan, np.float32)}

    c = _finite_after(poison_attr)
    assert c["ok"] is False and "COLOR_0" in c["fix"]


def test_finite_covers_material_factors():
    def poison_material(m):
        m.material = Material(name="bad", base_color_factor=(np.nan, 1.0, 1.0, 1.0))

    c = _finite_after(poison_material)
    assert c["ok"] is False and "material factors" in c["fix"]


# -- asset "budgets" root key ------------------------------------------------


def _asset_with_budgets(budgets_json: str):
    from moraine3d.core import asset as asset_mod

    return asset_mod.loads(
        '{"build_version": "fixture", '
        '"nodes": [{"id": "b1", "op": "box", "params": {}}], '
        '"outputs": {"main": "b1"}, '
        f'"budgets": {budgets_json}}}'
    )


def test_asset_budgets_parse_and_round_trip():
    """DESIGN 4.1: root ``budgets`` is asset content — parsed, kept, and
    serialized in canonical position (after asserts, before provenance)."""
    from moraine3d.core import asset as asset_mod

    sc = _asset_with_budgets('{"materials": 3}')
    assert sc.budgets == {"materials": 3}
    text = asset_mod.dumps(sc)
    assert '"budgets": {"materials": 3}' in text
    assert text.index('"outputs"') < text.index('"budgets"')
    assert asset_mod.loads(text).budgets == sc.budgets
    assert _asset_with_budgets("{}").budgets == {}
    assert asset_mod.ROOT_KEY_ORDER.index("budgets") == \
        asset_mod.ROOT_KEY_ORDER.index("asserts") + 1


def test_asset_budgets_rejections_are_actionable():
    import pytest

    from moraine3d.errors import AssetError

    for text, code in (
        ("[]", "bad_budgets"),
        ('{"triangles": 5}', "unknown_budget"),
        ('{"materials": 0}', "bad_budget_value"),
        ('{"materials": 2.5}', "bad_budget_value"),
        ('{"materials": true}', "bad_budget_value"),
    ):
        with pytest.raises(AssetError) as e:
            _asset_with_budgets(text)
        assert e.value.code == code


def test_failures_only_carries_budgets():
    """DESIGN 8.2: a failing budget survives compression with its fix and
    identities; a passing one collapses to the counts."""
    from moraine3d.verify.validate import failures_only

    report = {
        "ok": False,
        "outputs": {"a": {"ok": True, "checks": [{"check": "finite", "ok": True}]}},
        "budgets": [
            {"budget": "materials", "ok": False, "count": 5, "limit": 4,
             "identities": [], "fix": "share or raise"},
        ],
    }
    out = failures_only(report)
    assert out["budgets_total"] == 1 and out["budgets_ok"] == 0
    assert out["budgets"] == report["budgets"]
    report["budgets"][0].update(ok=True, count=4)
    del report["budgets"][0]["fix"]
    out = failures_only(report)
    assert out["budgets_total"] == 1 and out["budgets_ok"] == 1
    assert "budgets" not in out


# -- solid construction reasons (asserts on unbuildable surfaces) --------------


def test_unbuildable_solid_reason_names_the_kernel_status():
    """An output the kernel cannot build as a solid is reported with the
    manifold3d status and a matching fix (no_overlap semantics): a box
    with one flipped face passes `watertight` yet is NotManifold — the
    reason must not send the author to a green check alone, and a
    non-finite mesh is sent to the finite check."""
    from moraine3d.core.mesh import Mesh
    from moraine3d.ops import box as box_op

    b = box_op.evaluate([], {}, None)
    faces = b.faces.copy()
    faces[0] = faces[0, ::-1]
    flipped = Mesh(vertices=b.vertices, faces=faces, normals=b.normals)
    other = transform.evaluate([box_op.evaluate([], {}, None)], {"translate": [0.5, 0.0, 0.0]}, None)
    checks = _by_name(validate_value(flipped))
    assert checks["watertight"]["ok"] is True
    report = validate_outputs(
        {"a": flipped, "b": other},
        asserts=[{"id": "sep", "kind": "no_overlap", "outputs": ["a", "b"],
                  "owner": "agent"}],
    )
    entry = report["asserts"][0]
    assert entry["ok"] is False
    assert "cannot be built as a solid (NotManifold)" in entry["fix"]
    assert "winding" in entry["fix"] and "watertight" in entry["fix"]
    verts = b.vertices.copy()
    verts[0, 0] = np.nan
    with np.errstate(invalid="ignore"):  # the weld's NaN cast is the defect under test
        report = validate_outputs(
            {"a": Mesh(vertices=verts, faces=b.faces, normals=b.normals), "b": other},
            asserts=[{"id": "sep", "kind": "no_overlap", "outputs": ["a", "b"],
                  "owner": "agent"}],
        )
    fix = report["asserts"][0]["fix"]
    assert "(NonFiniteVertex)" in fix and "finite check" in fix


# -- distance kernel: degenerate second segment -------------------------------


def test_seg_seg_dist_handles_a_point_second_segment():
    """The segment-segment feature is exact when the second segment is a
    point (the distance-kind kernel's degenerate-segment contract): it
    equals the point-to-first-segment distance, symmetric with the
    point-first case, never the endpoint-only overestimate."""
    from moraine3d.verify.distance import seg_seg_dist

    rng = np.random.default_rng(11)
    n = 2000
    p1 = rng.uniform(-1.0, 1.0, (n, 3))
    q1 = p1 + rng.uniform(-1.0, 1.0, (n, 3))
    p2 = rng.uniform(-1.0, 1.0, (n, 3))
    got = seg_seg_dist(p1, q1, p2, p2)
    d = q1 - p1
    t = np.clip(np.einsum("ij,ij->i", p2 - p1, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    brute = np.linalg.norm(p1 + t[:, None] * d - p2, axis=1)
    assert np.allclose(got, brute, atol=1e-12)
    assert np.allclose(got, seg_seg_dist(p2, p2, p1, q1), atol=1e-12)
    # Both degenerate: the point-point distance.
    assert np.allclose(seg_seg_dist(p1, p1, p2, p2), np.linalg.norm(p1 - p2, axis=1), atol=1e-12)


# -- self paths: bounded memo -------------------------------------------------


def test_self_pair_memos_release_copies_after_their_last_pair():
    """The opt-in self paths hold each copy's world geometry only until
    its last pair (DESIGN 5.2 'self' over a dense scatter must not
    materialize every copy at once): the release schedule covers both
    pair columns, and the clearance / overlap verdicts are unchanged and
    deterministic."""
    from moraine3d.verify.distance import self_clearance_pairs, world_tri_source
    from moraine3d.verify.pairs import last_uses
    from moraine3d.verify.depth import copy_pair_depths
    from moraine3d.verify.solids import to_manifold

    ii = np.array([0, 0, 1, 2, 2])
    jj = np.array([1, 3, 2, 3, 4])
    last = last_uses(ii, jj, 6)
    assert last.tolist() == [1, 2, 4, 3, 4, -1]
    proto = box.evaluate([], {"size": [0.3, 0.3, 0.3]}, None)
    target = grid.evaluate([], {"size": [2.0, 2.0]}, None)
    inst = scatter.evaluate([target, proto], {"count": 40, "seed": 5}, OpContext("sc01"))
    src, why = world_tri_source(inst, "field")
    assert why is None
    runs = [self_clearance_pairs(src, 0.05) for _ in range(2)]
    assert runs[0] == runs[1]
    total, failing, known = runs[0]
    assert total == 40 * 39 // 2 and len(failing) > 0
    # Every priced pair is handed on as the closest-pair search's seed.
    assert all(known[(i, j)] == d for i, j, d in failing)
    # Brute-force per pair over copy views agrees with the memoized loop.
    from moraine3d.verify.distance import copy_bounds, copy_source, surface_distance

    brute = []
    for i in range(40):
        for j in range(i + 1, 40):
            d = surface_distance(
                copy_source(src, i), copy_bounds(src, i),
                copy_source(src, j), copy_bounds(src, j), 0.05,
            )
            if d is not None and d < 0.05:
                brute.append((i, j, d))
    assert [(i, j) for i, j, _ in failing] == [(i, j) for i, j, _ in brute]
    assert np.allclose([d for *_, d in failing], [d for *_, d in brute], atol=1e-12)
    solid = to_manifold(proto.vertices, proto.faces)
    o1 = copy_pair_depths(inst, "field", solid)
    o2 = copy_pair_depths(inst, "field", solid)
    assert o1 == o2 and o1[0] == 40 * 39 // 2 and len(o1[1]) > 0


def test_flat_shading_gated_on_curved_surface(tmp_path, monkeypatch):
    """flat_shading fires only where smooth_normals would change shading
    (DESIGN 4.1): an all-planar boolean output (box minus box) shades
    identically flat or smooth, so it reaches material without a
    warning; a curved operand's output warns. The gate is the op's own
    criterion (`would_smooth` at the default threshold)."""
    from moraine3d.ops import cylinder, smooth_normals

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = _shading_asset([], "u1", source="box")
    ev = Evaluation(sc)
    ev.records()
    assert ev.shading_warnings() == []
    sc = _shading_asset([], "u1")
    ev = Evaluation(sc)
    ev.records()
    (w,) = ev.shading_warnings()
    assert w["booleans"] == ["u1"] and "curved" in w["note"]
    assert not smooth_normals.would_smooth(box.evaluate([], {}, None))
    assert smooth_normals.would_smooth(cylinder.evaluate([], {}, None))


def test_enclosed_copy_is_named_unchecked_not_passed_silently():
    """not_enclosed judges an instanced output over all its copies
    (DESIGN 4.1): one copy swallowed by an opaque container passes, and
    the green entry names the output under unchecked.instanced_copies
    (the asserts convention); with no copy box inside any container box
    nothing is named, and --failures-only keeps the count."""
    hull = transform.evaluate(
        [box.evaluate([], {"size": [3.0, 3.0, 3.0]}, None)], {"translate": [0.0, 1.5, 0.0]}, None
    )
    bit = transform.evaluate(
        [box.evaluate([], {"size": [0.2, 0.2, 0.2]}, None)], {"translate": [0.0, 1.0, 0.0]}, None
    )
    swallowed = array.evaluate([bit], {"count": 2, "step_translate": [5.0, -0.9, 0.0]}, None)
    report = validate_outputs({"hull": hull, "bits": swallowed})
    entry = _by_name(report["outputs"]["bits"]["checks"])["not_enclosed"]
    assert entry == {
        "check": "not_enclosed", "ok": True, "unchecked": {"instanced_copies": ["bits"]},
    }
    assert failures_only(report)["unchecked"] == {"entries": 1, "instanced_copies": 1}

    clear = array.evaluate([bit], {"count": 2, "step_translate": [5.0, -0.9, 0.0]}, None)
    clear = transform.evaluate([clear], {"translate": [5.0, 0.0, 0.0]}, None)
    report = validate_outputs({"hull": hull, "bits": clear})
    assert "not_enclosed" not in _by_name(report["outputs"]["bits"]["checks"])
    assert "unchecked" not in failures_only(report)


def test_failures_only_rolls_up_green_unchecked_asserts():
    """DESIGN 6: `--failures-only` keeps the honesty machinery as
    counts. A green contact entry over an instanced output without
    `every` is counted under `unchecked`, with the outputs it did not
    quantify over counted by name; the row is the full report's."""
    ground = _grounded_box()
    copies = array.evaluate(
        [transform.evaluate([box.evaluate([], {"size": [0.2, 0.2, 0.2]}, None)],
                            {"translate": [0.0, 1.1, 0.0]}, None)],
        {"count": 2, "step_translate": [0.3, 0.0, 0.0]}, None,
    )
    report = validate_outputs(
        {"ground": ground, "bits": copies},
        asserts=[{"id": "rest", "kind": "contact", "outputs": ["ground", "bits"],
                  "owner": "agent"}],
    )
    entry = next(e for e in report["asserts"] if e["assert"] == "rest")
    assert entry["ok"] is True and entry["unchecked"] == {"instanced_copies": ["bits"]}
    slim = failures_only(report)
    assert "asserts" not in slim
    assert slim["unchecked"] == {"entries": 1, "instanced_copies": 1}


def test_contact_every_entry_carries_the_union_distance():
    """Every contact entry reports `distance` (DESIGN 5.2), the quantified
    one included: the union's minimum is the per-copy minimum, each copy
    measured exactly."""
    ground = _grounded_box()
    copies = array.evaluate(
        [transform.evaluate([box.evaluate([], {"size": [0.2, 0.2, 0.2]}, None)],
                            {"translate": [0.0, 1.1, 0.0]}, None)],
        {"count": 2, "step_translate": [0.3, 0.0, 0.0]}, None,
    )
    report = validate_outputs(
        {"ground": ground, "bits": copies},
        asserts=[{"id": "rest", "kind": "contact", "outputs": ["ground", "bits"],
                  "every": ["bits"], "owner": "agent"}],
    )
    entry = next(e for e in report["asserts"] if e["assert"] == "rest")
    assert entry["ok"] is True and entry["every"] == {"bits": 2}
    assert entry["distance"] <= entry["every_max"][1] <= 1e-5


def test_failures_only_rolls_up_every_honesty_key():
    """The honesty keys are one roster (DESIGN 6): a green guard entry
    that ceded pairs to a declared embed is counted under `ceded` the
    way `unchecked` and `stronger` are — the compressed report still
    says that something was not checked, and how much."""
    lower = _grounded_box()
    upper = transform.evaluate([_grounded_box()], {"translate": [0.0, 0.995, 0.0]}, None)
    report = validate_outputs(
        {"lower": lower, "upper": upper},
        asserts=[{"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
                  "min": 0.004, "max": 0.006, "owner": "agent"}],
    )
    guard = next(e for e in report["asserts"] if e["assert"] == "no_overlap")
    assert guard["ok"] is True and guard["ceded"] == {"pairs": [["lower", "upper"]]}
    slim = failures_only(report)
    assert slim["ok"] is True and "asserts" not in slim
    assert slim["ceded"] == {"entries": 1, "pairs": 1}


def test_would_smooth_ignores_sliver_normals():
    """The curvature gate leaves sliver faces out (DESIGN 4.1: an
    all-planar output is not reported): a float32 sliver's normal is off
    by ulp/height, which crosses the dihedral floor on rotated geometry
    at asset scale although the surface is planar."""
    from moraine3d.ops import boolean, smooth_normals

    rot = {"rotate_deg": [37.0, 23.0, 11.0]}
    outer = transform.evaluate([box.evaluate([], {"size": [10.0, 10.0, 10.0]}, None)], rot, None)
    # A cutter leaving a 1 mm-thick slab: its long faces are 10 m x 1 mm
    # (aspect 1e-4), sliver by shape.
    cutter = transform.evaluate(
        [box.evaluate([], {"size": [12.0, 12.0, 9.999]}, None)],
        {"translate": [0.0, 0.0, -1.0005], **rot}, None,
    )
    slab = boolean.evaluate([outer, cutter], {"mode": "difference"}, None)
    assert not smooth_normals.would_smooth(slab)
