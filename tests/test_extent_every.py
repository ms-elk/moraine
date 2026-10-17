"""``extent``'s ``every`` form (DESIGN 5.2): each copy of the subject
measured on its own, where the output form measures the union of the
copies — the per-copy statement (every leg reaches the ground) the
union cannot make.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core.asserts.common import NOISE_FLOOR
from moraine3d.core.mesh import Instances
from moraine3d.errors import AssetError
from moraine3d.ops import array as array_op
from moraine3d.ops import box, tag as tag_op, transform
from moraine3d.verify.validate import validate_outputs


def _row(lifts: list[float], tagged: bool = False) -> Instances:
    """Unit boxes on the ground along x, copy i lifted by ``lifts[i]``."""
    proto = box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)
    lifted = transform.evaluate([proto], {"translate": [0.0, 0.5, 0.0]}, None)
    if tagged:  # after the transform: only the evaluator carries groups across ops
        lifted = tag_op.evaluate([lifted], {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, None)
    row = array_op.evaluate(
        [lifted], {"count": len(lifts), "step_translate": [2.0, 0.0, 0.0]}, None
    )
    xf = np.array(row.transforms, dtype=np.float64)
    for i, dy in enumerate(lifts):
        xf[i, 1, 3] += dy
    return Instances(prototype=row.prototype, transforms=xf)


def _entry(report: dict, aid: str) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == aid)


def _extent(aid: str, **keys) -> dict:
    return {"id": aid, "kind": "extent", "owner": "agent", **keys}


def test_every_reads_each_copy_where_the_union_reads_the_lowest():
    """The output form measures the union of the copies, so a floating
    copy passes `min_y`; `every` measures each copy and names the one
    that floats, reporting the counts as `every` and the span as
    `measured_range` in place of `measured` (DESIGN 5.2)."""
    rep = validate_outputs({"t": _row([0.0, 0.0, 0.4])}, asserts=[
        _extent("union", outputs=["t"], measure="min_y", value=0),
        _extent("each", outputs=["t"], every=True, measure="min_y", value=0),
    ])
    union = _entry(rep, "union")
    assert union["ok"] is True and union["measured"] == pytest.approx(0.0, abs=1e-9)
    each = _entry(rep, "each")
    assert each["ok"] is False
    assert each["every"] == {"t": 3}
    assert each["measured_range"] == pytest.approx([0.0, 0.4], abs=1e-6)
    assert "measured" not in each
    assert "1 copy of output 't' miss min_y: 't[2]' measures 0.4 m against value 0 m" in each["fix"]
    assert "state the intended dimension" in each["fix"]
    assert each["detail"]["failing_copies"] == [["t[2]", pytest.approx(0.4, abs=1e-6)]]


def test_every_green_carries_the_span_and_a_plain_mesh_is_one_copy():
    """A green `every` entry still reports the counts and the span; a
    plain mesh counts one copy — vacuous and legal, as contact counts it
    (DESIGN 5.2)."""
    plain = transform.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)], {"translate": [0.0, 0.5, 0.0]}, None
    )
    rep = validate_outputs({"t": _row([0.0, 0.0, 0.0]), "p": plain}, asserts=[
        _extent("each", outputs=["t"], every=True, measure="min_y", value=0),
        _extent("one", outputs=["p"], every=True, measure="size_y", value=1),
    ])
    each = _entry(rep, "each")
    assert each["ok"] is True and each["every"] == {"t": 3}
    assert each["measured_range"] == pytest.approx([0.0, 0.0], abs=1e-9)
    one = _entry(rep, "one")
    assert one["ok"] is True and one["every"] == {"p": 1}
    assert one["measured_range"] == pytest.approx([1.0, 1.0], abs=1e-6)


def test_every_face_form_reads_the_face_set_on_every_copy():
    """The face form under `every` drops the copy index and resolves the
    tagged face set on each copy (DESIGN 5.2): the cap that sits high
    is named, the others pass."""
    rep = validate_outputs({"row": _row([0.0, 0.3, 0.0], tagged=True)}, asserts=[
        _extent("caps", faces=["row", "top"], every=True, measure="max_y", value=1.0),
        _extent("gone", faces=["row", "lip"], every=True, measure="max_y", value=1.0),
    ])
    caps = _entry(rep, "caps")
    assert caps["ok"] is False and caps["every"] == {"row": 3}
    assert caps["measured_range"] == pytest.approx([1.0, 1.3], abs=1e-6)
    assert caps["detail"]["failing_copies"] == [["row[1]", pytest.approx(1.3, abs=1e-6)]]
    assert "face set 'row.top'" in caps["fix"]
    gone = _entry(rep, "gone")
    assert gone["ok"] is False and "available tags: ['top']" in gone["fix"]
    assert "measured_range" not in gone


def test_every_over_a_group_counts_per_member():
    """`every` on a group quantifies every member's copies, the counts
    reported per member (DESIGN 5.2)."""
    plain = transform.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)], {"translate": [9.0, 0.7, 0.0]}, None
    )
    rep = validate_outputs(
        {"t": _row([0.0, 0.0, 0.0]), "p": plain},
        asserts=[_extent("piece", outputs=["piece"], every=True, measure="min_y", value=0)],
        groups={"piece": ["t", "p"]},
    )
    e = _entry(rep, "piece")
    assert e["members"] == {"piece": ["t", "p"]}
    assert e["every"] == {"t": 3, "p": 1}
    assert e["ok"] is False
    assert e["detail"]["failing_copies"] == [["p", pytest.approx(0.2, abs=1e-6)]]


def _asset(entries: list[dict]) -> str:
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "g1", "op": "tag", "inputs": ["b1"],
             "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
            {"id": "a1", "op": "array", "inputs": ["g1"],
             "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}},
        ],
        "outputs": {"row": "a1", "lid": "g1"},
        "asserts": entries,
    })


def test_every_parse_shape_and_round_trip():
    """`every` is JSON true or absent; under it the face form drops its
    copy index; on a kind without it the refusal names contact and
    extent; the key serializes in its fixed place (DESIGN 5.2)."""
    ok = [
        _extent("bounds", outputs=["row"], every=True, measure="min_y", value=0),
        _extent("faces", faces=["row", "top"], every=True, measure="max_y", value=1),
    ]
    sc = asset_mod.loads(_asset(ok))
    assert asset_mod.loads(asset_mod.dumps(sc)).asserts == sc.asserts
    line = [ln for ln in asset_mod.dumps(sc).splitlines() if '"bounds"' in ln][0]
    assert line.index('"measure"') < line.index('"every"') < line.index('"value"')
    for entry, code, words in (
        (_extent("x", outputs=["row"], every="yes", measure="min_y", value=0),
         "bad_assert", "'every' must be JSON true"),
        (_extent("x", faces=["row", "top", 1], every=True, measure="max_y", value=1),
         "bad_assert_faces", "drop the copy index, or name one copy without \"every\""),
        ({"id": "x", "kind": "count", "outputs": ["row"], "every": True, "value": 3,
          "owner": "agent"}, "bad_assert", "and extent (each copy measured on its own)"),
    ):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_asset([entry]))
        assert exc.value.code == code and words in str(exc.value)


def test_every_fix_lists_a_bounded_number_of_copies():
    """The fix sentence names a bounded number of failing copies and
    points at `detail` for the rest, which lists them all (token
    economy, DESIGN 8.2)."""
    rep = validate_outputs({"t": _row([0.5] * 14)}, asserts=[
        _extent("each", outputs=["t"], every=True, measure="min_y", value=0),
    ])
    e = _entry(rep, "each")
    assert e["ok"] is False and "14 copies of output 't' miss min_y" in e["fix"]
    assert "and 4 more (see detail)" in e["fix"]
    assert len(e["detail"]["failing_copies"]) == 14


def test_every_takes_the_storage_noise_floor_per_copy():
    """A copy missing its bound by less than the storage-noise floor
    passes and the entry says so (`noise_floor`), as every length
    comparison does (DESIGN 5.2)."""
    rep = validate_outputs({"t": _row([0.0, 4e-7, 0.0])}, asserts=[
        _extent("each", outputs=["t"], every=True, measure="min_y", max=0),
    ])
    e = _entry(rep, "each")
    assert e["ok"] is True and e["noise_floor"] == NOISE_FLOOR


def test_both_face_forms_refuse_a_defective_output_the_same_way():
    """The whole-output refusals (image, empty, non-finite) fire first on
    every form of extent, so one defect has one sentence and a NaN
    outside the tagged face set cannot pass the union face form while
    failing the `every` form (DESIGN 5.2: an unevaluatable subject fails
    with the reason)."""
    from moraine3d.core.mesh import Mesh

    tagged = tag_op.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)],
        {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, None,
    )
    verts = np.array(tagged.vertices, dtype=np.float32)
    bottom = np.flatnonzero(verts[:, 1] < 0)[0]  # a vertex no top face uses
    verts[bottom, 0] = np.nan
    broken = Mesh(vertices=verts, faces=tagged.faces, normals=tagged.normals,
                  face_groups=tagged.face_groups)
    rep = validate_outputs({"p": broken}, asserts=[
        _extent("one", faces=["p", "top"], measure="max_y", value=0.5),
        _extent("each", faces=["p", "top"], every=True, measure="max_y", value=0.5),
    ])
    for aid in ("one", "each"):
        e = _entry(rep, aid)
        assert e["ok"] is False
        assert e["fix"] == "output 'p' has non-finite vertices; fix the finite check first"
