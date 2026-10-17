"""``mated``'s ``every`` form (DESIGN 5.2): each copy of the outputs it
names judged as an indexed entry would judge it — two named outputs
paired copy by copy, one named output against the other face as
written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core.mesh import Instances
from moraine3d.errors import AssetError
from moraine3d.ops import array as array_op
from moraine3d.ops import box, tag as tag_op, transform
from moraine3d.verify.validate import validate_outputs

_TOP = {"normal_y": {"gt": 0.9}}
_BOTTOM = {"normal_y": {"lt": -0.9}}


def _part(size, center, name: str, select: dict):
    proto = box.evaluate([], {"size": size}, None)
    placed = transform.evaluate([proto], {"translate": center}, None)
    return tag_op.evaluate([placed], {"name": name, "select": select}, None)


def _row(part, count: int, shifts: dict[int, float] | None = None) -> Instances:
    """``count`` copies 2 m apart along x, copy i moved ``shifts[i]``
    further along x."""
    row = array_op.evaluate(
        [part], {"count": count, "step_translate": [2.0, 0.0, 0.0]}, None
    )
    xf = np.array(row.transforms, dtype=np.float64)
    for i, dx in (shifts or {}).items():
        xf[i, 0, 3] += dx
    return Instances(prototype=row.prototype, transforms=xf)


def _grove(count: int = 3, crowns: int | None = None, shifts=None) -> dict:
    """Trunks with a `seat` on top and crowns with a `socket`
    underneath, crown i standing on trunk i."""
    trunk = _part([0.4, 1.0, 0.4], [0.0, 0.5, 0.0], "seat", _TOP)
    crown = _part([0.4, 1.0, 0.4], [0.0, 1.5, 0.0], "socket", _BOTTOM)
    return {
        "trunk": _row(trunk, count),
        "crown": _row(crown, count if crowns is None else crowns, shifts),
    }


def _entry(report: dict, aid: str) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == aid)


def _mated(aid: str, faces, **keys) -> dict:
    return {"id": aid, "kind": "mated", "faces": faces, "owner": "agent", **keys}


_JOINT = [["trunk", "seat"], ["crown", "socket"]]


def test_every_pairs_two_named_outputs_copy_by_copy():
    """Two named outputs pair by copy index, and the entry reports the
    copy counts and each measure's maximum over the pairs (DESIGN 5.2)."""
    rep = validate_outputs(_grove(), asserts=[
        _mated("joints", _JOINT, every=["trunk", "crown"]),
    ])
    joints = _entry(rep, "joints")
    assert joints["ok"] is True
    assert joints["every"] == {"trunk": 3, "crown": 3}
    assert joints["detail"]["pairs"] == 3
    assert joints["detail"]["distance_max"] == pytest.approx(0.0, abs=1e-9)
    assert joints["detail"]["origin"] == ["centroid", "centroid"]


def test_every_names_the_copy_that_is_not_mated():
    """A failing pair rides the fix as its two face labels and
    `detail.failing_pairs` with its own numbers (DESIGN 5.2)."""
    rep = validate_outputs(_grove(shifts={2: 0.3}), asserts=[
        _mated("joints", _JOINT, every=["trunk", "crown"]),
    ])
    joints = _entry(rep, "joints")
    assert joints["ok"] is False
    assert joints["fix"].startswith(
        "1 of 3 copy pairs are not mated: trunk.seat[2] and crown.socket[2]: "
        "mating origins are 0.3 m apart"
    )
    [failing] = joints["detail"]["failing_pairs"]
    assert failing["faces"] == ["trunk.seat[2]", "crown.socket[2]"]
    assert failing["distance"] == pytest.approx(0.3, abs=1e-6)
    assert joints["detail"]["distance_max"] == pytest.approx(0.3, abs=1e-6)


@pytest.mark.parametrize("shifts", [None, {1: 0.3}, {0: 0.2, 2: 0.5}])
def test_every_is_the_indexed_entries_in_one(shifts):
    """The `every` form states what one indexed entry per copy states:
    the same verdict, and as its maximum the same numbers (DESIGN 5.2)."""
    indexed = [
        _mated(f"joint_{i}", [["trunk", "seat", i], ["crown", "socket", i]])
        for i in range(3)
    ]
    rep = validate_outputs(_grove(shifts=shifts), asserts=[
        *indexed, _mated("joints", _JOINT, every=["trunk", "crown"]),
    ])
    each = [_entry(rep, f"joint_{i}") for i in range(3)]
    joints = _entry(rep, "joints")
    assert joints["ok"] is all(e["ok"] for e in each)
    for key in ("angle_deg", "distance"):
        assert joints["detail"][f"{key}_max"] == max(e["detail"][key] for e in each)
    assert [f["faces"][1] for f in joints["detail"].get("failing_pairs", [])] == [
        f"crown.socket[{i}]" for i, e in enumerate(each) if not e["ok"]
    ]


def test_every_refuses_outputs_whose_counts_differ():
    """Pairing by index needs the two counts to agree; a mismatch fails
    naming both, a plain output counting one (DESIGN 5.2)."""
    rep = validate_outputs(_grove(count=3, crowns=2), asserts=[
        _mated("joints", _JOINT, every=["trunk", "crown"]),
    ])
    joints = _entry(rep, "joints")
    assert joints["ok"] is False
    assert joints["every"] == {"trunk": 3, "crown": 2}
    assert "'trunk' has 3 while 'crown' has 2" in joints["fix"]

    rep = validate_outputs(_deck(), asserts=[
        _mated("feet", _FOOTING, every=["deck", "posts"], mode="contained"),
    ])
    feet = _entry(rep, "feet")
    assert feet["ok"] is False
    assert feet["every"] == {"deck": 1, "posts": 3}
    assert "'deck' has 1 while 'posts' has 3" in feet["fix"]


def _deck(shifts=None) -> dict:
    """Three posts standing on one deck, 2 m apart from x = 0."""
    deck = _part([6.0, 0.2, 1.0], [2.0, 0.1, 0.0], "top", _TOP)
    post = _part([0.4, 1.0, 0.4], [0.0, 0.7, 0.0], "foot", _BOTTOM)
    return {"deck": deck, "posts": _row(post, 3, shifts)}


_FOOTING = [["deck", "top"], ["posts", "foot"]]


def test_every_puts_one_named_output_against_the_other_face():
    """One named output is judged copy by copy against the other face as
    written — every part butting one anchor face (DESIGN 5.2)."""
    rep = validate_outputs(_deck(), asserts=[
        _mated("feet", _FOOTING, every=["posts"], mode="contained"),
    ])
    feet = _entry(rep, "feet")
    assert feet["ok"] is True
    assert feet["every"] == {"posts": 3}
    assert feet["detail"]["pairs"] == 3
    assert "stronger" not in feet  # two of the three feet sit off the center

    rep = validate_outputs(_deck(shifts={2: 1.0}), asserts=[
        _mated("feet", _FOOTING, every=["posts"], mode="contained"),
    ])
    feet = _entry(rep, "feet")
    assert feet["ok"] is False
    assert [f["faces"] for f in feet["detail"]["failing_pairs"]] == [
        ["deck.top", "posts.foot[2]"]
    ]


def test_an_unnamed_instanced_face_is_still_never_guessed():
    """A face without a copy index on an instanced output is judged on
    every copy only when `every` names the output; otherwise the entry
    fails with the reason and both fixes (DESIGN 5.2)."""
    rep = validate_outputs(_deck(), asserts=[
        _mated("feet", _FOOTING, mode="contained"),
    ])
    feet = _entry(rep, "feet")
    assert feet["ok"] is False
    assert "add the copy index <0..2>" in feet["fix"]
    assert 'or write "every": ["posts"]' in feet["fix"]


def test_the_every_fix_is_offered_for_a_missing_index_alone():
    """A reason names the shortest path to its own fix (DESIGN 8.2): a
    tag the output does not carry is not repaired by `every`."""
    rep = validate_outputs(_deck(), asserts=[
        _mated("feet", [["deck", "top"], ["posts", "fooot"]], mode="contained"),
    ])
    feet = _entry(rep, "feet")
    assert feet["ok"] is False
    assert "has no face group 'fooot'" in feet["fix"]
    assert '"every"' not in feet["fix"]


def test_a_plain_output_named_under_every_is_one_copy():
    """A plain mesh counts one copy — vacuous and legal, as contact and
    extent count it (DESIGN 5.2)."""
    values = _deck()
    values["post"] = _part([0.4, 1.0, 0.4], [0.0, 0.7, 0.0], "foot", _BOTTOM)
    rep = validate_outputs(values, asserts=[
        _mated("foot", [["deck", "top"], ["post", "foot"]], every=["post"],
               mode="contained"),
    ])
    foot = _entry(rep, "foot")
    assert foot["ok"] is True
    assert foot["every"] == {"post": 1}
    assert foot["detail"]["pairs"] == 1


def _asset(entry: dict) -> str:
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "t1", "op": "tag", "inputs": ["b1"],
             "params": {"name": "seat", "select": _TOP}},
            {"id": "a1", "op": "array", "inputs": ["t1"],
             "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}},
            {"id": "b2", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "t2", "op": "tag", "inputs": ["b2"],
             "params": {"name": "socket", "select": _BOTTOM}},
            {"id": "a2", "op": "array", "inputs": ["t2"],
             "params": {"count": 3, "step_translate": [2.0, 0.0, 0.0]}},
        ],
        "outputs": {"trunk": "a1", "crown": "a2", "stump": "t1"},
        "asserts": [dict(entry, kind="mated", owner="agent")],
    })


def test_every_parse_rules():
    """`every` is a list of names drawn from the entry's faces, and a
    face it names carries no copy index (DESIGN 5.2)."""
    text = _asset({"id": "j", "faces": _JOINT, "every": ["trunk", "crown"]})
    assert asset_mod.dumps(asset_mod.loads(text)).count('"every": ["trunk", "crown"]') == 1
    asset_mod.loads(_asset(
        {"id": "j", "faces": [["trunk", "seat", 0], ["crown", "socket"]], "every": ["crown"]}
    ))
    for entry, code, needle in [
        ({"faces": _JOINT, "every": True}, "bad_assert", "drawn from its 'faces'"),
        ({"faces": _JOINT, "every": []}, "bad_assert", "non-empty list"),
        ({"faces": _JOINT, "every": ["stump"]}, "bad_assert", "['trunk', 'crown']"),
        ({"faces": _JOINT, "every": ["trunk", "trunk"]}, "bad_assert", "distinct names"),
        ({"faces": [["trunk", "seat", 0], ["crown", "socket"]], "every": ["trunk"]},
         "bad_assert_faces", "drop the copy index"),
    ]:
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_asset({"id": "j", **entry}))
        assert exc.value.code == code, entry
        assert needle in exc.value.message, entry
