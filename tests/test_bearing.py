"""contact's bearing form, ``min_area`` (DESIGN 5.2): the face area two
subjects share must reach the stated square meters, so an edge or a
corner contact, distance 0 like a face, no longer passes for a seat."""

import json
import math

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError
from moraine3d.verify.bearing import shared_area
from moraine3d.verify.validate import validate_outputs
from support.engine import evaluate_records


def _run(asset_json: str) -> dict:
    sc = asset_mod.loads(asset_json)
    recs = evaluate_records(sc)
    return validate_outputs(
        {n: r.value() for n, r in recs.items()},
        checks_policy=sc.checks, asserts=sc.asserts, groups=sc.groups,
    )


def _entry(report: dict, aid: str) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == aid)


def _stack(upper_at, entries, upper_size=(1.0, 1.0, 1.0), extra=()):
    """A unit box on the ground and an ``upper_size`` box centered at
    ``upper_at``: (0, 1.5, 0) is flush on the lower's top face."""
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "g1", "op": "transform", "inputs": ["b1"],
             "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "b2", "op": "box", "params": {"size": list(upper_size)}},
            {"id": "g2", "op": "transform", "inputs": ["b2"],
             "params": {"translate": list(upper_at)}},
            *extra,
        ],
        "outputs": {"lower": "g1", "upper": "g2"},
        "asserts": entries,
    })


def _bearing(**keys) -> dict:
    return {"id": "seat", "kind": "contact", "outputs": ["lower", "upper"],
            "owner": "agent", "min_area": 0.5, **keys}


def test_shared_area_is_the_coplanar_overlap():
    """``shared_area`` is the overlap of two face sets in a shared plane
    (DESIGN 5.2): a face on a face reads the overlap, an edge reads
    zero, a gap past ``tol`` reads zero, and a seat turned in the plane
    reads the exact polygon."""
    def box(size, center):
        sx, sy, sz = [s / 2 for s in size]
        v = np.array(
            [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], float
        ) + center
        faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                 (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
        return v[np.array(faces)]

    lower = box((1, 1, 1), (0, 0.5, 0))
    assert shared_area(lower, box((1, 1, 1), (0, 1.5, 0)), 1e-5) == pytest.approx(1.0)
    assert shared_area(lower, box((1, 1, 1), (0.5, 1.5, 0.5)), 1e-5) == pytest.approx(0.25)
    assert shared_area(lower, box((1, 1, 1), (1.0, 1.5, 0)), 1e-5) == 0.0  # an edge
    assert shared_area(lower, box((1, 1, 1), (0, 1.5 + 1e-4, 0)), 1e-5) == 0.0
    assert shared_area(lower, box((1, 1, 1), (0, 1.5 + 5e-6, 0)), 1e-5) == pytest.approx(1.0)
    th = math.radians(45)
    rot = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    turned = box((1, 1, 1), (0, 0, 0)) @ rot.T + [0, 1.5, 0]
    assert shared_area(lower, turned, 1e-5) == pytest.approx(2 * (math.sqrt(2) - 1))
    # A side face bears like a top face: the measure has no up.
    assert shared_area(lower, box((0.5, 2, 1), (0.75, 1.0, 0)), 1e-5) == pytest.approx(1.0)
    assert shared_area(lower, np.empty((0, 3, 3)), 1e-5) == 0.0
    # A 1 mm seat on a 20 m slab keeps its area: the sliver floor is
    # the seat's own, not the slab's.
    slab = box((20, 0.2, 20), (0, 0.1, 0))
    seat = box((0.001, 0.001, 0.001), (3.0, 0.2005, -7.0))
    assert shared_area(slab, seat, 1e-5) == pytest.approx(1e-6, rel=1e-6)
    assert shared_area(slab, box((0.001, 0.001, 0.001), (10.0005, 0.2005, 0.0)), 1e-5) == 0.0


def test_face_contact_passes_and_reports_the_area(tmp_path, monkeypatch):
    """A flush seat passes the bearing form and the entry carries the
    shared ``area`` beside ``distance`` (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run(_stack((0.0, 1.5, 0.0), [_bearing()]))
    e = _entry(report, "seat")
    assert e["ok"] is True and report["ok"] is True
    assert e["min_area"] == 0.5
    assert e["area"] == pytest.approx(1.0)
    assert e["distance"] <= 1e-5
    # Half the face still clears min_area 0.5 (inclusive).
    report = _run(_stack((0.5, 1.5, 0.0), [_bearing()]))
    e = _entry(report, "seat")
    assert e["ok"] is True and e["area"] == pytest.approx(0.5)


def test_edge_contact_fails_the_bearing_form_alone(tmp_path, monkeypatch):
    """An edge contact is distance 0 and passes a plain contact; under
    ``min_area`` it shares no face and fails with the area and the
    owner's advice (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    plain = {k: v for k, v in _bearing().items() if k != "min_area"}
    report = _run(_stack((1.0, 1.5, 0.0), [plain]))
    assert _entry(report, "seat")["ok"] is True
    report = _run(_stack((1.0, 1.5, 0.0), [_bearing()]))
    e = _entry(report, "seat")
    assert e["ok"] is False and report["ok"] is False
    assert e["area"] == 0.0 and e["distance"] <= 1e-5
    assert "share 0 m^2" in e["fix"] and "min_area 0.5" in e["fix"]
    assert "lower min_area" in e["fix"]  # agent-owned: the weakening is offered
    report = _run(_stack((1.0, 1.5, 0.0), [_bearing(owner="human")]))
    e = _entry(report, "seat")
    assert "lower min_area" not in e["fix"] and "not the min_area" in e["fix"]


def test_gap_fails_distance_and_area_together(tmp_path, monkeypatch):
    """A part above its seat fails the distance and shares no area; both
    sentences join, each with its number."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run(_stack((0.0, 1.6, 0.0), [_bearing()]))
    e = _entry(report, "seat")
    assert e["ok"] is False
    assert "closest approach" in e["fix"] and "share 0 m^2" in e["fix"]
    assert e["area"] == 0.0 and abs(e["distance"] - 0.1) < 1e-4


def test_area_floor_is_on_the_side_length(tmp_path, monkeypatch):
    """The storage-noise floor applies to the side of the area (DESIGN
    5.2): a seat 1e-6 m short on a side passes with ``noise_floor``, a
    seat short by more fails."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    near = _bearing(min_area=1.0)
    report = _run(_stack((0.0, 1.5, 0.0), [near], upper_size=(1.0 - 5e-7, 1.0, 1.0)))
    e = _entry(report, "seat")
    assert e["ok"] is True and e.get("noise_floor") == 1e-6
    report = _run(_stack((0.0, 1.5, 0.0), [near], upper_size=(0.999, 1.0, 1.0)))
    e = _entry(report, "seat")
    assert e["ok"] is False and "noise_floor" not in e


def test_every_judges_each_copys_bearing(tmp_path, monkeypatch):
    """Under ``every`` each copy must share ``min_area`` with the other
    output; the copy sharing least rides the entry as ``area_min`` and
    the bare copies are named (DESIGN 5.2) — the stair whose steps
    touched their stringers along an edge, caught."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))

    def stair(stringer_x: float, entries):
        # Three steps rising by their own height, bearing on a stringer
        # beside them when its inner face sits at x = 0.5 (the steps'
        # +x face); at x = 1.0 the stringer only grazes their edges.
        return json.dumps({
            "build_version": "fixture",
            "nodes": [
                {"id": "s1", "op": "box", "params": {"size": [1.0, 0.2, 0.3]}},
                {"id": "s2", "op": "transform", "inputs": ["s1"],
                 "params": {"translate": [0.0, 0.1, 0.0]}},
                {"id": "a1", "op": "array", "inputs": ["s2"],
                 "params": {"count": 3, "step_translate": [0.0, 0.2, -0.3]}},
                {"id": "r1", "op": "box", "params": {"size": [0.1, 1.0, 1.2]}},
                {"id": "r2", "op": "transform", "inputs": ["r1"],
                 "params": {"translate": [stringer_x + 0.05, 0.5, -0.3]}},
            ],
            "outputs": {"steps": "a1", "stringer": "r2"},
            "asserts": entries,
        })

    entry = {"id": "bear", "kind": "contact", "outputs": ["stringer", "steps"],
             "every": ["steps"], "min_area": 0.05, "owner": "human"}
    report = _run(stair(0.5, [entry]))
    e = _entry(report, "bear")
    assert e["ok"] is True
    assert e["every"] == {"steps": 3}
    assert e["area_min"][0].startswith("steps[") and e["area_min"][1] == pytest.approx(0.06)
    assert e["area"] == pytest.approx(0.18)
    # Move the stringer's inner face 1 mm off the steps: within tol of
    # nothing, so every copy is bare and the distance fails too.
    bare = _run(stair(0.5 + 1e-3, [entry]))
    e = _entry(bare, "bear")
    assert e["ok"] is False
    assert e["detail"]["bare_copies"] == [["steps[0]", 0.0], ["steps[1]", 0.0], ["steps[2]", 0.0]]
    assert "share" in e["fix"] and "closest approach" not in e["fix"]
    assert e["area_min"] == ["steps[0]", 0.0]


def test_every_names_the_bare_copies(tmp_path, monkeypatch):
    """A copy that touches along an edge while its neighbors seat on a
    face is named with its area under ``detail.bare_copies``."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "g1", "op": "transform", "inputs": ["b1"],
             "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "a1", "op": "array", "inputs": ["g1"],
             "params": {"count": 3, "step_translate": [1.0, 0.0, 0.0]}},
            # A slab under copies 0 and 1 whose far edge ends exactly at
            # copy 2's near edge: copy 2 touches it along a line only.
            {"id": "p1", "op": "box", "params": {"size": [2.0, 0.2, 1.0]}},
            {"id": "p2", "op": "transform", "inputs": ["p1"],
             "params": {"translate": [0.5, -0.1, 0.0]}},
        ],
        "outputs": {"row": "a1", "slab": "p2"},
        "asserts": [{"id": "c", "kind": "contact", "outputs": ["row", "slab"],
                     "every": ["row"], "min_area": 0.9, "owner": "agent"}],
    })
    report = _run(asset)
    e = _entry(report, "c")
    assert e["ok"] is False
    assert e["detail"]["bare_copies"] == [["row[2]", 0.0]]
    assert e["area_min"] == ["row[2]", 0.0]
    assert e["area"] == pytest.approx(2.0)
    assert "'row[2]' shares 0 m^2" in e["fix"]


def test_min_area_parses_as_a_positive_area(tmp_path):
    """``min_area`` is a finite number > 0; a written 0 is refused as
    the plain form spelled twice, and the key is contact's alone."""
    def asset(entry):
        return _stack((0.0, 1.5, 0.0), [entry])

    asset_mod.loads(asset(_bearing(min_area=1)))  # an int is a number
    for bad in (0, -1.0, "wide"):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(asset(_bearing(min_area=bad)))
        assert exc.value.code == "bad_assert" and "min_area" in str(exc.value)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(asset({"id": "x", "kind": "clearance", "outputs": ["lower", "upper"],
                               "min_area": 0.1, "owner": "agent", "min": 0.001}))
    assert "min_area" in str(exc.value) and "contact" in str(exc.value)


def test_group_union_bears_as_one(tmp_path, monkeypatch):
    """A group name stands for its members' union: the shared area is
    summed over the member faces that bear (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = json.loads(_stack((0.0, 1.5, 0.0), []))
    asset["nodes"] += [
        {"id": "b3", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
        {"id": "g3", "op": "transform", "inputs": ["b3"],
         "params": {"translate": [1.0, 0.5, 0.0]}},
    ]
    asset["outputs"]["lower2"] = "g3"
    asset["groups"] = {"base": ["lower", "lower2"]}
    asset["nodes"][3]["params"]["translate"] = [0.5, 1.5, 0.0]  # upper spans both
    asset["asserts"] = [{"id": "seat", "kind": "contact", "outputs": ["base", "upper"],
                         "min_area": 0.9, "owner": "agent"}]
    report = _run(json.dumps(asset))
    e = _entry(report, "seat")
    assert e["ok"] is True and e["area"] == pytest.approx(1.0)
    assert e["members"] == {"base": ["lower", "lower2"]}


def test_min_area_builds_from_the_recipe(tmp_path):
    """The recipe spelling reaches the IR (DESIGN 4.8): ``min_area`` is
    a float key written after ``tol``, an integer widened, and a
    requirements file may take it whole from a knob."""
    from support.recipes import build_recipe, ir

    asset = (
        'mo.output("slab", mo.box(id = "slab", size = [2.0, 0.1, 2.0]))\n'
        'mo.output("lid", mo.box(id = "lid", size = [1, 1, 1]))\n'
        'mo.assert_contact(id = "seat", outputs = ["lid", "slab"], tol = 0.001, min_area = 1)\n'
    )
    doc = ir(build_recipe(tmp_path / "a", asset))
    [entry] = doc["asserts"]
    assert entry["min_area"] == 1.0 and isinstance(entry["min_area"], float)
    assert list(entry)[:6] == ["id", "kind", "outputs", "tol", "min_area", "owner"]
    req = (
        'mo.assert_contact(id = "seat", outputs = ["lid", "slab"], '
        'min_area = mo.param_value("bearing"))\n'
    )
    asset2 = (
        'b = mo.param("bearing", 0.5, min = 0.1, max = 1.0)\n'
        'mo.output("slab", mo.box(id = "slab", size = [2.0, 0.1, 2.0]))\n'
        'mo.output("lid", mo.box(id = "lid", size = [1, 1, 1]))\n'
    )
    doc = ir(build_recipe(tmp_path / "b", asset2, files={"asset.requirements.star": req}))
    [entry] = doc["asserts"]
    assert entry["min_area"] == 0.5 and entry["from_param"] == {"min_area": "bearing"}
