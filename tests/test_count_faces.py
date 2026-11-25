"""``count`` over a face set (DESIGN 5.2): the number of patches in one
tagged face set, counted once on the prototype — a number of features
stated in one entry."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import SRC_ROOT
from moraine3d.app.cli import main
from moraine3d.core import mating
from moraine3d.ops import array, bake_instances, boolean, box, cylinder, tag, transform
from moraine3d.recipe.build import build
from moraine3d.recipe.version import CONTRACT, build_version
from moraine3d.verify.validate import validate_outputs
from support.recipes import refusal, write_files

WALLS = {
    "normal_y": {"gt": -0.1, "lt": 0.1},
    "center_x": {"gt": -0.45, "lt": 0.45},
    "center_z": {"gt": -0.45, "lt": 0.45},
}
TOP = {"normal_y": {"gt": 0.9}}


def _plate(holes: int):
    """A square plate with ``holes`` through holes, their walls tagged."""
    plate = box.evaluate([], {"size": [1.0, 0.1, 1.0]}, None)
    bore = cylinder.evaluate([], {"radius": 0.05, "height": 0.3, "segments": 8}, None)
    centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)][:holes]
    cutters = [
        transform.evaluate([bore], {"translate": [x, 0.0, z]}, None) for x, z in centers
    ]
    cut = boolean.evaluate([plate, *cutters], {"mode": "difference"}, None)
    return tag.evaluate([cut], {"name": "holes", "select": WALLS}, None)


def _crate(translate=(0.0, 0.0, 0.0)):
    """A unit box with its top face tagged."""
    b = transform.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)],
        {"translate": [float(t) for t in translate]}, None,
    )
    return tag.evaluate([b], {"name": "top", "select": TOP}, None)


def _count(**keys) -> dict:
    return {"id": "n", "kind": "count", "faces": ["part", "holes"], "owner": "agent", **keys}


def _entry(value, entry: dict) -> dict:
    rep = validate_outputs({"part": value}, asserts=[entry])
    return next(e for e in rep["asserts"] if e["assert"] == entry["id"])


# -- the kernel ----------------------------------------------------------------


def test_a_patch_is_the_faces_joined_through_shared_edges():
    """A patch is the faces joined through shared edges with vertices
    at one position taken as one vertex; two patches that meet at a
    point alone count two, and a face without area joins nothing and is
    not counted (DESIGN 5.2)."""
    a, b, c, d = [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
    quad = np.array([[a, b, c], [b, d, c]], dtype=np.float64)
    assert mating.patch_count(quad) == 1
    # Two triangles that share the one vertex `b`.
    bow = np.array([[a, b, c], [b, [2, 0, 0], [2, 1, 0]]], dtype=np.float64)
    assert mating.patch_count(bow) == 2
    # Three faces on one edge are one patch.
    fin = np.array([[a, b, c], [a, b, [0, -1, 0]], [a, b, [0, 0, 1]]], dtype=np.float64)
    assert mating.patch_count(fin) == 1
    # A sliver without area between two faces joins neither.
    apart = np.array([[a, b, c], [b, [2, 0, 0], b], [[2, 0, 0], [3, 0, 0], [2, 1, 0]]],
                     dtype=np.float64)
    assert mating.patch_count(apart) == 2
    assert mating.patch_count(np.array([[a, b, b]], dtype=np.float64)) == 0
    assert mating.patch_count(np.zeros((0, 3, 3))) == 0


# -- the measurement -----------------------------------------------------------


def test_the_holes_of_a_plate_are_counted_by_their_walls():
    """`"faces": [output, tag]` is the number of patches in one tagged
    face set: a number of features in one entry, where a feature that
    goes missing fails the entry that states the number (DESIGN 5.2)."""
    e = _entry(_plate(4), _count(value=4))
    assert e["ok"] is True and e["measured"] == 4
    assert (e["faces"], e["value"]) == (["part", "holes"], 4) and "tol" not in e
    miss = _entry(_plate(3), _count(value=4))
    assert miss["ok"] is False and miss["measured"] == 3
    assert "face set 'part.holes' has 3 patches against value 4 (-1 off)" in miss["fix"]
    assert "state the intended count" in miss["fix"]
    human = _entry(_plate(3), _count(value=4, owner="human"))
    assert "not the number" in human["fix"]
    band = _entry(_plate(4), _count(min=3, max=5))
    assert band["ok"] is True and (band["min"], band["max"]) == (3, 5)


def test_an_instanced_set_is_counted_once_on_the_prototype():
    """Tags ride the prototype, so an instanced output's set is counted
    once, on the prototype: the number is one copy's and the same on
    every copy. An output baked into one mesh carries the set on every
    copy, and its patches are counted across the whole mesh (DESIGN
    5.2)."""
    row = array.evaluate([_plate(4)], {"count": 2, "step_translate": [3.0, 0.0, 0.0]}, None)
    assert _entry(row, _count(value=4))["measured"] == 4
    baked = bake_instances.evaluate([row], {}, None)
    assert _entry(baked, _count(value=8))["measured"] == 8


@pytest.mark.parametrize("step,patches", [
    ([3.0, 0.0, 0.0], 2),  # apart
    ([1.0, 0.0, 0.0], 1),  # the tops share an edge
    ([1.0, 0.5, 0.0], 2),  # the boxes touch, the tops do not
])
def test_a_baked_set_joins_only_through_its_own_faces(step, patches):
    """Its patches are counted across the whole mesh: two copies' sets
    are one patch where tagged faces share an edge, and stay two where
    the copies touch outside the set (DESIGN 5.2)."""
    row = array.evaluate([_crate()], {"count": 2, "step_translate": step}, None)
    baked = bake_instances.evaluate([row], {}, None)
    e = _entry(baked, _count(faces=["part", "top"], value=patches))
    assert e["ok"] is True and e["measured"] == patches
    assert _entry(row, _count(faces=["part", "top"], value=1))["measured"] == 1


@pytest.mark.parametrize("segments", [8, 12, 48, 144])
def test_the_flange_example_counts_its_holes_at_any_segment_count(
    segments, tmp_path, monkeypatch, capsys,
):
    """`examples/flange_pipe`: the tag selects the walls of the bottom
    flange's bolt holes and never the rim, whatever the segment count,
    so eight holes read 8 and a hole that goes missing there reads 7
    and fails the entry that says eight (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    source = SRC_ROOT / "docs" / "examples" / "flange_pipe"
    recipe = (source / "flange_pipe.star").read_text(encoding="utf-8")
    assert "SEGMENTS = 48\n" in recipe and "    [0.1475, 0.0], " in recipe
    recipe = recipe.replace("SEGMENTS = 48\n", f"SEGMENTS = {segments}\n")
    requirements = (source / "flange_pipe.requirements.star").read_text(encoding="utf-8")
    for holes, text in ((8, recipe), (7, recipe.replace("    [0.1475, 0.0], ", "    "))):
        project = write_files(tmp_path / str(holes) / "flange_pipe", {
            "flange_pipe.star": text, "flange_pipe.requirements.star": requirements,
        })
        code, entries = _validate(project, capsys)
        entry = entries["bolt_hole_count"]
        assert entry["measured"] == holes and entry["ok"] is (holes == 8)
        assert code == (0 if holes == 8 else 1)


# -- what cannot be measured ---------------------------------------------------


def test_a_set_with_no_face_that_has_area_fails_with_the_reason():
    """A set that holds no face with area is unevaluatable and the
    entry fails with the reason, so the face form never measures 0 and
    no entry passes on a tag that selected nothing (DESIGN 5.2)."""
    crate = _crate()
    none = tag.evaluate(
        [crate], {"name": "none", "select": {"center_y": {"gt": 100.0}}, "expect": "any"}, None,
    )
    for keys in ({"value": 0}, {"max": 3}):
        e = _entry(none, _count(faces=["part", "none"], **keys))
        assert e["ok"] is False and "selects 0 faces" in e["fix"] and "measured" not in e
    # The top face's vertices drawn to one point: the tag still holds
    # its faces, and none has area.
    v = crate.vertices.copy()
    v[np.unique(crate.faces[crate.face_groups["top"]])] = [0.0, 0.5, 0.0]
    flat = replace(crate, vertices=v)
    for keys in ({"value": 0}, {"max": 3}):
        e = _entry(flat, _count(faces=["part", "top"], **keys))
        assert e["ok"] is False and "measured" not in e
        assert "holds no face with area" in e["fix"]
    unknown = _entry(crate, _count(faces=["part", "lid"], value=1))
    assert unknown["ok"] is False and "available tags: ['top']" in unknown["fix"]


def test_a_non_finite_output_is_not_counted():
    """An unevaluatable subject fails with the reason (DESIGN 5.2): a
    vertex that is not finite leaves one triangle of the tagged quad
    without an area to read, and the entry does not count the other as
    the whole set."""
    crate = _crate()
    top = crate.faces[crate.face_groups["top"]]
    lone = np.setdiff1d(top[0], top[1])
    v = crate.vertices.copy()
    v[lone] = np.nan
    e = _entry(replace(crate, vertices=v), _count(faces=["part", "top"], value=1))
    assert e["ok"] is False and "measured" not in e
    assert "non-finite vertices" in e["fix"]


LAMPS = '''n = mo.param("lamps", 2, min = 0, max = 4, label = "Lamps")
mo.output("deck", mo.transform(mo.box(id = "deck_box", size = [2, 0.1, 2]), id = "deck", translate = [0, 0.05, 0]))
lamp = mo.transform(mo.box(id = "lamp_box", size = [0.1, 1, 0.1]), id = "lamp", translate = [-0.6, 0.6, 0])
lamp = mo.tag(lamp, id = "lamp_top", name = "top", select = {"normal_y": {"gt": 0.9}})
mo.output("lamps", mo.array(lamp, id = "lamps", count = n, step_translate = [0.4, 0, 0]))
'''
REQUIREMENTS = '''mo.assert_count(id = "lamp_count", outputs = ["lamps"], value = mo.param_value("lamps"))
mo.assert_count(id = "lamp_tops", faces = ["lamps", "top"], value = 1)
'''


def _validate(project, capsys, *argv) -> tuple[int, dict]:
    code = main(["validate", str(project), "--full", *argv])
    report = json.loads(capsys.readouterr().out)
    return code, {a["assert"]: a for a in report["asserts"]}


def test_an_output_without_copies_has_no_face_set_to_count(tmp_path, monkeypatch, capsys):
    """`count` over an output alone measures one that is declared and
    has no copies, as 0, optional or not. Over a face set it has
    nothing to read there, and is judged as every other kind is (DESIGN
    5.2): it fails over a required output and judges nothing over an
    optional one."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "pier", {
        "pier.star": LAMPS, "pier.requirements.star": REQUIREMENTS,
    })
    code, entries = _validate(project, capsys)
    assert code == 0 and entries["lamp_tops"]["measured"] == 1
    code, entries = _validate(project, capsys, "--param", "lamps=0")
    assert code == 1
    assert entries["lamp_count"]["ok"] is True and entries["lamp_count"]["measured"] == 0
    tops = entries["lamp_tops"]
    assert tops["ok"] is False and tops["absent"] == ["lamps"]
    assert "no copies" in tops["fix"] and "measured" not in tops
    optional = write_files(tmp_path / "b" / "pier", {
        "pier.star": LAMPS,
        "pier.requirements.star": 'mo.optional(["lamps"], "lamps")\n' + REQUIREMENTS,
    })
    code, entries = _validate(optional, capsys, "--param", "lamps=0")
    assert code == 0
    assert entries["lamp_count"]["measured"] == 0 and "unchecked" not in entries["lamp_count"]
    tops = entries["lamp_tops"]
    assert tops["ok"] is True and tops["unchecked"] == {"optional": ["lamps"]}
    assert "measured" not in tops


# -- the entry's shape ---------------------------------------------------------

ASSET = '''crate = mo.tag(mo.box(id = "crate_box", size = [1, 1, 1]), id = "crate", name = "top", select = {"normal_y": {"gt": 0.9}})
mo.output("crates", mo.array(crate, id = "crates", count = 2, step_translate = [3, 0, 0]))
mo.output("lid", mo.box(id = "lid", size = [1, 1, 1]))
mo.group("all", ["crates", "lid"])
'''


@pytest.mark.parametrize("entry,code,words", [
    ('faces = ["crates", "top", 0], value = 1', "bad_assert_faces", "takes no copy index"),
    ('faces = ["crates", "top"], every = True, value = 1', "bad_assert", "'every' is only for"),
    ('faces = ["crates", "top"], outputs = ["crates"], value = 1', "bad_assert_outputs", "not both"),
    ("value = 1", "bad_assert_outputs", "not neither"),
    ('faces = ["all", "top"], value = 1', "bad_assert_outputs", "one entry per member"),
    ('faces = ["crates"], value = 1', "bad_assert_faces", "[output, tag]"),
    ('faces = ["crates", "top"], value = 1, tol = 1', "bad_assert", "no tolerance on a count"),
    ('faces = ["crates", "top"], value = 1.5', "bad_assert", "integer"),
])
def test_the_face_form_names_one_set_and_no_copy(tmp_path, entry, code, words):
    """`"faces": [output, tag]`: a copy index and `every` are refused as
    naming nothing the count depends on, and a group has no faces
    (DESIGN 5.2)."""
    err = refusal(tmp_path, ASSET, files={
        "asset.requirements.star": f'mo.assert_count(id = "n", {entry})\n',
    })
    assert err.code == code and words in err.message


@pytest.mark.real_stamp
def test_an_ir_of_the_build_before_the_face_form_is_refused(tmp_path, monkeypatch, capsys):
    """DESIGN 4.4: `build_version` covers the IR's shape, so an IR that
    carries the stamp of the contract before `count` took a face set is
    stale, and the loader refuses it (`version_mismatch`)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET.replace("mo.group", "# mo.group"),
        "yard.requirements.star": 'mo.assert_count(id = "n", faces = ["crates", "top"], value = 1)\n',
    })
    ir = build(project / "yard.star").target.ir_path
    assert main(["stats", str(ir)]) == 0
    capsys.readouterr()
    now = build_version()
    assert CONTRACT >= 11 and now.startswith(f"{CONTRACT}+")
    before = "10" + now[len(str(CONTRACT)):]
    text = ir.read_text(encoding="utf-8")
    with open(ir, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace(now, before))
    assert main(["stats", str(ir)]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "version_mismatch"
    assert (err["detail"]["found"], err["detail"]["expected"]) == (before, now)
