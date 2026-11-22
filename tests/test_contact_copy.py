"""contact's ``[output, copy]`` side (DESIGN 5.2): one copy of an
instanced output named as a subject of its own, so a part that rests on
one copy of an arrayed partner states that, where the plain form is
existential over the copies."""

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError, BuildError
from moraine3d.verify.validate import validate_outputs
from support.engine import evaluate_records


def _doc(entries, groups=None, deck_span=(2.7, 3.3), decks=1) -> str:
    """Four beams along +z, one meter apart, each 0.4 m^2 on top, and a
    deck lying on them over ``deck_span`` in z: by default on the last
    one alone (copy 3). ``decks`` above 1 arrays the deck along +x,
    the further copies clear of every beam."""
    z0, z1 = deck_span
    nodes = [
        {"id": "b1", "op": "box", "params": {"size": [1.0, 0.2, 0.4]}},
        {"id": "g1", "op": "transform", "inputs": ["b1"],
         "params": {"translate": [0.0, 0.1, 0.0]}},
        {"id": "a1", "op": "array", "inputs": ["g1"],
         "params": {"count": 4, "step_translate": [0.0, 0.0, 1.0]}},
        {"id": "d1", "op": "box", "params": {"size": [1.0, 0.1, z1 - z0]}},
        {"id": "d2", "op": "transform", "inputs": ["d1"],
         "params": {"translate": [0.0, 0.25, (z0 + z1) / 2]}},
        {"id": "p1", "op": "box", "params": {"size": [0.2, 0.2, 0.2]}},
        {"id": "p2", "op": "transform", "inputs": ["p1"],
         "params": {"translate": [5.0, 0.1, 0.0]}},
    ]
    if decks > 1:
        nodes.append({"id": "d3", "op": "array", "inputs": ["d2"],
                      "params": {"count": decks, "step_translate": [3.0, 0.0, 0.0]}})
    doc = {
        "build_version": "fixture",
        "nodes": nodes,
        "outputs": {"beams": "a1", "deck": "d3" if decks > 1 else "d2", "post": "p2"},
        "asserts": entries,
    }
    if groups:
        doc["groups"] = groups
    return json.dumps(doc)


def _run(entries, groups=None, **shape) -> dict:
    sc = asset_mod.loads(_doc(entries, groups, **shape))
    recs = evaluate_records(sc)
    return validate_outputs(
        {n: r.value() for n, r in recs.items()},
        checks_policy=sc.checks, asserts=sc.asserts, groups=sc.groups,
    )


def _contact(outputs, **keys) -> dict:
    return {"id": "c", "kind": "contact", "outputs": outputs, "owner": "agent", **keys}


def _entry(report: dict) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == "c")


def _rows(report: dict) -> list[dict]:
    return [w for w in report.get("warnings", []) if w["warning"] == "contact_unquantified"]


def test_a_named_copy_is_judged_alone(tmp_path, monkeypatch):
    """An ``[output, copy]`` side is that one copy (DESIGN 5.2): the
    copy the part rests on passes, another copy of the same output
    fails by its own distance under its label, on either side."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run([_contact(["deck", ["beams", 3]])])
    e = _entry(report)
    assert e["ok"] is True and e["distance"] <= 1e-5
    assert e["outputs"] == ["deck", ["beams", 3]]
    for outputs in (["deck", ["beams", 0]], [["beams", 0], "deck"]):
        e = _entry(_run([_contact(outputs)]))
        assert e["ok"] is False
        assert e["distance"] == pytest.approx(2.5, abs=1e-6)
        assert "'beams[0]'" in e["fix"] and "'deck'" in e["fix"]


def test_a_named_copy_is_not_left_existential(tmp_path, monkeypatch):
    """The entry is about the one copy, so it names no instanced output
    under ``unchecked`` and draws no ``contact_unquantified`` row, where
    the same pair written by name draws both (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run([_contact(["deck", ["beams", 3]])])
    assert "unchecked" not in _entry(report) and _rows(report) == []
    report = _run([_contact(["deck", "beams"])])
    assert _entry(report)["unchecked"] == {"instanced_copies": ["beams"]}
    [row] = _rows(report)
    assert '"every": ["beams"]' in row["note"]
    assert '["beams", <copy>] in place of "beams"' in row["note"]


def test_the_row_names_fixes_the_entry_accepts_over_a_group(tmp_path, monkeypatch):
    """The row's fix is written in the names the entry takes (DESIGN
    5.2): where the instanced output is a group's member, ``every``
    names the group and the copy form replaces the group's side with
    the member's own copy, and both spellings parse."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    groups = {"base": ["beams", "post"]}
    report = _run([_contact(["deck", "base"])], groups=groups)
    [row] = _rows(report)
    assert row["outputs"] == ["beams"]
    assert '"every": ["base"]' in row["note"]
    assert '["beams", <copy>] in place of "base" (a group has no copy)' in row["note"]
    for fixed in (
        _contact(["deck", "base"], every=["base"]), _contact(["deck", ["beams", 3]]),
    ):
        assert _rows(_run([fixed], groups=groups)) == []


def test_only_the_indexed_side_stops_being_existential(tmp_path, monkeypatch):
    """The other side keeps its conditions (DESIGN 5.2): an instanced
    output there is still named under ``unchecked`` and still draws its
    row, until ``every`` judges its copies."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    report = _run([_contact([["beams", 3], "deck"])], decks=2)
    e = _entry(report)
    assert e["ok"] is True and e["unchecked"] == {"instanced_copies": ["deck"]}
    [row] = _rows(report)
    assert row["outputs"] == ["deck"]
    report = _run([_contact([["beams", 3], "deck"], every=["deck"])], decks=2)
    e = _entry(report)
    assert e["ok"] is False and e["every"] == {"deck": 2}
    assert e["detail"]["failing_copies"][0][0] == "deck[1]"
    assert "unchecked" not in e and _rows(report) == []


def test_min_area_reads_the_named_copys_area_alone(tmp_path, monkeypatch):
    """A side named whole sums the shared area over its copies; an
    indexed side reads the one copy's (DESIGN 5.2): a deck bearing 0.4
    m^2 on each of two beams meets ``min_area`` 0.6 against the beams
    by name and against neither beam by index."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    span = {"deck_span": (1.7, 3.3)}  # over beams 2 and 3
    e = _entry(_run([_contact(["deck", "beams"], min_area=0.6)], **span))
    assert e["ok"] is True and e["area"] == pytest.approx(0.8)
    for copy in (2, 3):
        e = _entry(_run([_contact(["deck", ["beams", copy]], min_area=0.6)], **span))
        assert e["ok"] is False and e["area"] == pytest.approx(0.4)
        assert f"'beams[{copy}]' share 0.4 m^2" in e["fix"]
        assert _entry(
            _run([_contact(["deck", ["beams", copy]], min_area=0.4)], **span)
        )["ok"] is True


def test_a_copy_the_output_lacks_fails_with_the_reason(tmp_path, monkeypatch):
    """Whether the output has the copy is an evaluation fact (DESIGN
    5.2): an index out of range and an index on a plain mesh fail the
    entry with the reason, never a guessed copy."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    e = _entry(_run([_contact(["deck", ["beams", 4]])]))
    assert e["ok"] is False and "distance" not in e
    assert "4 copies; copy index 4 is out of range (0..3)" in e["fix"]
    assert "name a copy the output has" in e["fix"]
    e = _entry(_run([_contact([["deck", 0], "post"])]))
    assert e["ok"] is False and "plain mesh and has no copy 0" in e["fix"]
    assert "drop the copy index" in e["fix"]


def test_a_human_owned_copy_is_never_steered_off_its_index(tmp_path, monkeypatch):
    """The index is the requirement's own (DESIGN 5.2): a human-owned
    entry whose copy is missing is pointed at the geometry, never at
    dropping or changing the index."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    e = _entry(_run([_contact([["deck", 3], "post"], owner="human")]))
    assert e["ok"] is False
    assert "build 'deck' as an array or scatter that has the copy" in e["fix"]
    assert "drop" not in e["fix"] and "not the copy index" in e["fix"]
    e = _entry(_run([_contact(["deck", ["beams", 7]], owner="human")]))
    assert e["ok"] is False and "build the copy" in e["fix"]
    assert "name a copy" not in e["fix"] and "not the copy index" in e["fix"]


def test_a_named_copy_shares_the_prototype(tmp_path, monkeypatch):
    """Instancing stays instancing in the check (DESIGN 5.2): a named
    copy is a view of the output's prototype and one transform, so
    naming many copies holds no world-space duplicate of the mesh."""
    import numpy as np

    from moraine3d.verify.asserts.common import Context, PairScope

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(_doc([]))
    values = {n: r.value() for n, r in evaluate_records(sc).items()}
    ctx = Context(values)
    proto = ctx.tri_source("beams")[0][0]
    for i in range(4):
        a = _contact(["deck", ["beams", i]])
        s = PairScope(ctx, {"outputs": a["outputs"], "owner": "agent"}, a, 1e-5)
        src = s.srcs[f"beams[{i}]"]
        assert src[0] is proto and len(src[1]) == 1
        assert np.shares_memory(src[1], ctx.tri_source("beams")[0][1])
        assert s.inst == []
    assert not any("[" in k for k in ctx._tri_sources)


def test_every_and_min_area_read_the_named_copy(tmp_path, monkeypatch):
    """The other side keeps its quantifier and the bearing form reads
    the copy's own faces (DESIGN 5.2)."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    e = _entry(_run([_contact([["beams", 3], "deck"], min_area=0.3, owner="human")]))
    assert e["ok"] is True and e["area"] == pytest.approx(0.4)
    e = _entry(_run([_contact([["beams", 2], "deck"], min_area=0.3)]))
    assert e["ok"] is False and e["area"] == 0.0
    e = _entry(_run([_contact([["beams", 3], "deck"], every=["deck"])]))
    assert e["ok"] is True and e["every"] == {"deck": 1}


def test_a_copy_side_parses_by_its_output():
    """Parse-time shape (DESIGN 5.2): a side is a name or ``[output,
    copy]`` with a non-negative integer index over a declared output; a
    group has no copy, the two sides name different outputs, and
    ``every`` cannot quantify the output a side indexes."""
    asset_mod.loads(_doc([_contact([["beams", 0], "deck"])]))
    asset_mod.loads(_doc([_contact([["beams", 0], ["deck", 0]])]))
    for bad in (
        ["deck", ["beams", -1]], ["deck", ["beams", True]], ["deck", ["beams"]],
        ["deck", ["beams", 0, 1]], ["deck", [3, 0]], ["deck", ["beams", "last"]],
        ["beams", ["beams", 0]], [["beams", 0], ["beams", 1]],
    ):
        with pytest.raises(AssetError) as exc:
            asset_mod.loads(_doc([_contact(bad)]))
        assert exc.value.code == "bad_assert_outputs"
        assert "[output, copy]" in str(exc.value)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_doc([_contact(["deck", ["rails", 0]])]))
    assert "unknown output 'rails'" in str(exc.value)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_doc(
            [_contact(["deck", ["base", 0]])], groups={"base": ["beams", "post"]},
        ))
    assert "names one copy of one output" in str(exc.value)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_doc([_contact(["deck", ["beams", 3]], every=["beams"])]))
    assert exc.value.code == "bad_assert_outputs" and "drop the copy index" in str(exc.value)


def test_the_copy_side_is_contacts_alone():
    """``clearance`` against another output already holds for every
    copy, so it takes names only and refuses the copy side (DESIGN
    5.2)."""
    entry = {"id": "c", "kind": "clearance", "outputs": ["deck", ["beams", 0]],
             "min": 0.01, "owner": "agent"}
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_doc([entry]))
    assert exc.value.code == "bad_assert_outputs"
    assert "[output, copy]" not in str(exc.value)


def test_an_absent_output_named_through_a_copy_fails_as_absent(tmp_path, monkeypatch):
    """An entry reaches the output its copy side names (DESIGN 5.2):
    with that output absent the entry fails under ``absent``, as one
    that named it whole does."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(_doc([_contact(["deck", ["beams", 3]])]))
    recs = evaluate_records(sc)
    values = {n: r.value() for n, r in recs.items() if n != "beams"}
    report = validate_outputs(
        values, checks_policy=sc.checks, asserts=sc.asserts, groups=sc.groups,
        absent=["beams"],
    )
    e = _entry(report)
    assert e["ok"] is False and e["absent"] == ["beams"]


def test_the_copy_side_builds_from_the_recipe(tmp_path):
    """The recipe spelling reaches the IR as written (DESIGN 4.8), and
    a float index is refused where an integer is the copy's index."""
    from support.recipes import build_recipe, ir

    body = (
        'beam = mo.box(id = "beam", size = [1.0, 0.2, 0.4])\n'
        'mo.output("beams", mo.array(beam, id = "row", count = 4, '
        'step_translate = [0.0, 0.0, 1.0]))\n'
        'mo.output("deck", mo.box(id = "deck", size = [1.0, 0.1, 0.6]))\n'
    )
    doc = ir(build_recipe(
        tmp_path / "a",
        body + 'mo.assert_contact(id = "end", outputs = ["deck", ["beams", 3]])\n',
    ))
    [entry] = doc["asserts"]
    assert entry["outputs"] == ["deck", ["beams", 3]]
    with pytest.raises(BuildError) as exc:
        build_recipe(
            tmp_path / "b",
            body + 'mo.assert_contact(id = "end", outputs = ["deck", ["beams", 6 / 2]])\n',
        )
    assert exc.value.code == "bad_assert" and "takes an integer" in str(exc.value)
