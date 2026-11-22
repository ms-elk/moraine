"""A face spec's copy index on ``mated`` and ``extent`` (DESIGN 5.2,
ownership): the index is a line of the entry, so where the output
lacks the copy, or has copies and the face names none, a human-owned
entry is pointed at the geometry or at the human, never at an edit of
its own index."""

from __future__ import annotations

import pytest

from moraine3d.ops import array as array_op
from moraine3d.ops import box, tag as tag_op, transform
from moraine3d.verify.validate import validate_outputs

_TOP = {"normal_y": {"gt": 0.9}}
_BOTTOM = {"normal_y": {"lt": -0.9}}


def _part(center, name: str, select: dict):
    proto = box.evaluate([], {"size": [0.4, 1.0, 0.4]}, None)
    placed = transform.evaluate([proto], {"translate": center}, None)
    return tag_op.evaluate([placed], {"name": name, "select": select}, None)


def _values() -> dict:
    """A plain plinth with a ``seat`` on top, and three posts in a row,
    each with a ``foot`` underneath, post 0 standing on the plinth."""
    post = _part([0.0, 1.5, 0.0], "foot", _BOTTOM)
    return {
        "plinth": _part([0.0, 0.5, 0.0], "seat", _TOP),
        "posts": array_op.evaluate(
            [post], {"count": 3, "step_translate": [2.0, 0.0, 0.0]}, None
        ),
    }


def _fix(entry: dict) -> str:
    rep = validate_outputs(_values(), asserts=[entry])
    e = next(e for e in rep["asserts"] if e["assert"] == entry["id"])
    assert e["ok"] is False
    return e["fix"]


def _mated(faces, owner: str) -> dict:
    return {"id": "m", "kind": "mated", "faces": faces, "owner": owner}


def _extent(face, owner: str) -> dict:
    return {"id": "x", "kind": "extent", "faces": face, "measure": "min_y",
            "value": 1.0, "owner": owner}


@pytest.mark.parametrize("entry", [
    lambda owner: _mated([["plinth", "seat", 0], ["posts", "foot", 0]], owner),
    lambda owner: _extent(["plinth", "seat", 0], owner),
])
def test_an_index_on_a_plain_mesh_follows_the_owner(entry):
    """The agent may drop its own index; a human-owned entry names a
    copy the requirement meant, so the fix is the output built with
    that copy (DESIGN 5.2)."""
    agent = _fix(entry("agent"))
    assert "plain mesh and has no copy 0" in agent and "drop the copy index" in agent
    human = _fix(entry("human"))
    assert "build 'plinth' as an array or scatter that has the copy" in human
    assert "drop" not in human and "not the copy index" in human


@pytest.mark.parametrize("entry", [
    lambda owner: _mated([["plinth", "seat"], ["posts", "foot", 5]], owner),
    lambda owner: _extent(["posts", "foot", 5], owner),
])
def test_an_index_out_of_range_follows_the_owner(entry):
    """A copy the output does not have: the agent may name another, a
    human-owned entry is met by building the copy (DESIGN 5.2)."""
    agent = _fix(entry("agent"))
    assert "copy index 5 is out of range (0..2)" in agent
    assert "name a copy the output has" in agent
    human = _fix(entry("human"))
    assert "copy index 5 is out of range (0..2)" in human and "build the copy" in human
    assert "name a copy" not in human and "not the copy index" in human


@pytest.mark.parametrize("entry, every", [
    (lambda owner: _mated([["plinth", "seat"], ["posts", "foot"]], owner), True),
    (lambda owner: _extent(["posts", "foot"], owner), False),
])
def test_a_face_without_its_index_follows_the_owner(entry, every):
    """An output arrayed under an entry that names no copy: the agent
    adds the index (or ``every`` on mated); which copies a human's
    requirement means is the human's to say, so the entry points at
    the geometry or at them (DESIGN 5.2)."""
    agent = _fix(entry("agent"))
    assert "add the copy index <0..2>" in agent
    assert ('or write "every": ["posts"]' in agent) is every
    human = _fix(entry("human"))
    assert "is instanced (3 copies) and the face names no copy" in human
    assert "build 'posts' as one mesh, or ask the human" in human
    assert "add the copy index" not in human and '"every"' not in human
