"""Output groups: the root ``groups`` section (DESIGN 4.1) and the
assert-side name resolution it feeds (DESIGN 5.2).

A piece built from several outputs — a shelf that is a board and its
posts — is named once and stands for its members wherever an assert
takes an output name: the union for the distance kinds, embedded's pair
form and extent's output form; refused where a kind names one output's
own copies or faces.
"""

import io
import json
from contextlib import redirect_stdout

import pytest

from moraine3d.app.cli import main
from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError
from moraine3d.verify.validate import validate_outputs
from support import flatten
from support.engine import evaluate_records


def _box(nid: str, size, translate, inputs=None):
    nodes = [{"id": f"{nid}_b", "op": "box", "params": {"size": list(size)}}]
    nodes.append({
        "id": nid, "op": "transform", "inputs": [f"{nid}_b"],
        "params": {"translate": list(translate)},
    })
    return nodes


def _shelves(
    entries: list[dict], *, groups=None, floor=False, floor_top=0.0,
    probe_x=None, image=False, lift_b=0.0,
) -> str:
    """Two shelves, each a board resting on two posts (touching by
    construction), 2 m apart in x; optionally a floor the posts stand on
    (a positive ``floor_top`` raises its top face off the ground plane,
    embedding the posts that deep), a probe box near shelf A, an image
    output, and shelf B lifted off the floor."""
    nodes: list[dict] = []
    outputs: dict[str, str] = {}
    for tag, x0, lift in (("a", 0.0, 0.0), ("b", 2.0, lift_b)):
        nodes += _box(f"{tag}_board", [1.0, 0.05, 0.4], [x0, 1.025 + lift, 0.0])
        nodes += _box(f"{tag}_post", [0.05, 1.0, 0.05], [x0 - 0.475, 0.5 + lift, 0.0])
        nodes.append({
            "id": f"{tag}_posts", "op": "array", "inputs": [f"{tag}_post"],
            "params": {"count": 2, "step_translate": [0.95, 0.0, 0.0]},
        })
        outputs[f"shelf_{tag}_board"] = f"{tag}_board"
        outputs[f"shelf_{tag}_posts"] = f"{tag}_posts"
    if floor:
        thick = floor_top if floor_top > 0 else 0.1
        nodes += _box("floor", [6.0, thick, 6.0], [1.0, floor_top - thick / 2, 0.0])
        outputs["floor"] = "floor"
    if probe_x is not None:
        nodes += _box("probe", [0.2, 0.2, 0.2], [probe_x, 1.0, 0.0])
        outputs["probe"] = "probe"
    if image:
        nodes.append({"id": "tex", "op": "checker_texture", "params": {"tiles": 2, "size": [8, 8]}})
        outputs["tex"] = "tex"
    doc: dict = {"build_version": "fixture", "nodes": nodes, "outputs": outputs}
    if groups is None:
        groups = {
            "shelf_a": ["shelf_a_board", "shelf_a_posts"],
            "shelf_b": ["shelf_b_board", "shelf_b_posts"],
        }
    if groups:
        doc["groups"] = groups
    doc["asserts"] = entries
    return json.dumps(doc)


def _report(asset_json: str) -> dict:
    sc = asset_mod.loads(asset_json)
    recs = evaluate_records(sc)
    return validate_outputs(
        {n: r.value() for n, r in recs.items()},
        checks_policy=sc.checks, asserts=sc.asserts, groups=sc.groups,
    )


def _entry(report: dict, aid: str) -> dict:
    return next(e for e in report["asserts"] if e["assert"] == aid)


def test_groups_parse_serialize_one_per_line_and_leave_other_assets_unchanged():
    """Root ``groups`` (DESIGN 4.1): optional, serialized one group per
    line right after ``outputs`` (it grows per piece), omitted when
    absent — an asset without the key round-trips byte-identically."""
    sc = asset_mod.loads(_shelves([]))
    assert sc.groups == {
        "shelf_a": ["shelf_a_board", "shelf_a_posts"],
        "shelf_b": ["shelf_b_board", "shelf_b_posts"],
    }
    text = asset_mod.dumps(sc)
    assert (
        '"shelf_b_posts": "b_posts"\n},\n"groups": {\n'
        '"shelf_a": ["shelf_a_board", "shelf_a_posts"],\n'
        '"shelf_b": ["shelf_b_board", "shelf_b_posts"]\n},\n"asserts"'
    ) in text
    assert asset_mod.dumps(asset_mod.loads(text)) == text
    assert asset_mod.ROOT_KEY_ORDER.index("groups") == asset_mod.ROOT_KEY_ORDER.index("outputs") + 1

    plain = asset_mod.loads(_shelves([], groups={}))
    assert plain.groups is None
    assert '"groups"' not in asset_mod.dumps(plain)


def test_groups_rejections_are_actionable():
    """``bad_group`` for every malformed spelling: not an object, empty,
    a bad or colliding name (groups and outputs share one namespace,
    case-folded like outputs), fewer than two distinct members, an
    unknown member."""
    def load(groups):
        doc = json.loads(_shelves([], groups={}))
        doc["groups"] = groups
        return asset_mod.loads(json.dumps(doc))

    for groups, needle in [
        ([["shelf_a_board"]], "non-empty object"),
        ({}, "non-empty object"),
        ({"bad name": ["shelf_a_board", "shelf_a_posts"]}, "must match"),
        ({"shelf_a_board": ["shelf_a_posts", "shelf_b_posts"]}, "collides with output"),
        ({"Shelf_A_Board": ["shelf_a_posts", "shelf_b_posts"]}, "collides with output"),
        ({"one": ["shelf_a_board"]}, "2 or more distinct"),
        ({"dup": ["shelf_a_board", "shelf_a_board"]}, "2 or more distinct"),
        ({"mixed": ["shelf_a_board", 3]}, "2 or more distinct"),
        ({"ghost": ["shelf_a_board", "nothing"]}, "unknown output 'nothing'"),
        ({"a": ["shelf_a_board", "shelf_a_posts"], "b": ["a", "shelf_b_board"]},
         "unknown output 'a'"),
    ]:
        with pytest.raises(AssetError) as exc:
            load(groups)
        assert exc.value.code == "bad_group", groups
        assert needle in exc.value.message, groups
    # Two groups may share a member: a shelf and "all shelves" both name it.
    sc = load({"a": ["shelf_a_board", "shelf_a_posts"],
               "all": ["shelf_a_board", "shelf_a_posts", "shelf_b_board"]})
    assert sc.groups["all"][0] == "shelf_a_board"


def test_group_in_one_vs_rest_clearance_excludes_its_own_members():
    """The finding behind the section (DESIGN 5.2): ``"outputs":
    [group], "others": true`` keeps the piece clear of every geometry
    output outside it — its own members, touching by construction, are
    not partners; an output added later is; the entry echoes the
    resolution under ``members``."""
    entry = {"id": "a_clear", "kind": "clearance", "outputs": ["shelf_a"],
             "others": True, "min": 0.5, "owner": "human"}
    e = _entry(_report(_shelves([entry], image=True)), "a_clear")
    assert e["ok"] is True and e["others"] is True
    assert e["members"] == {"shelf_a": ["shelf_a_board", "shelf_a_posts"]}
    assert e["pairs"] == 2  # shelf_b's two outputs; the image is no partner
    assert e["nearest"] in ("shelf_b_board", "shelf_b_posts")
    assert abs(e["distance"] - 1.0) < 1e-6

    report = _report(_shelves([entry], probe_x=0.7))
    e = _entry(report, "a_clear")
    assert report["ok"] is False and e["ok"] is False
    assert e["pairs"] == 3 and e["nearest"] == "probe"
    assert abs(e["distance"] - 0.1) < 1e-6
    assert e["detail"]["failing_partners"] == [["probe", e["distance"]]]
    assert "'shelf_a' and 'probe' come within 0.1 m" in e["fix"]


def test_group_pair_distances_are_the_union_minimum():
    """A group on either side of a pair is the union of its members: the
    closest approach over member pairs, for clearance and contact alike
    (DESIGN 5.2)."""
    entries = [
        {"id": "apart", "kind": "clearance", "outputs": ["shelf_a", "shelf_b"],
         "min": 0.5, "owner": "human"},
        {"id": "stands", "kind": "contact", "outputs": ["shelf_a", "floor"],
         "owner": "agent"},
        {"id": "b_stands", "kind": "contact", "outputs": ["floor", "shelf_b"],
         "owner": "agent"},
    ]
    report = _report(_shelves(entries, floor=True))
    apart = _entry(report, "apart")
    assert apart["ok"] is True and abs(apart["distance"] - 1.0) < 1e-6
    assert apart["members"] == {
        "shelf_a": ["shelf_a_board", "shelf_a_posts"],
        "shelf_b": ["shelf_b_board", "shelf_b_posts"],
    }
    stands = _entry(report, "stands")
    # The board never reaches the floor; the posts do — the union does.
    assert stands["ok"] is True and stands["distance"] == 0.0
    assert stands["unchecked"] == {"instanced_copies": ["shelf_a_posts"]}
    assert _entry(report, "b_stands")["ok"] is True

    lifted = _report(_shelves(entries, floor=True, lift_b=0.02))
    assert _entry(lifted, "apart")["ok"] is True
    b = _entry(lifted, "b_stands")
    assert b["ok"] is False and abs(b["distance"] - 0.02) < 1e-6
    assert "closest approach between 'floor' and 'shelf_b' is 0.02 m" in b["fix"]


def test_group_every_quantifies_each_members_copies():
    """``every`` naming a group checks every copy of every member
    against the other side; the counts are reported per member and a
    failing copy is named ``member[i]`` (DESIGN 5.2)."""
    groups = {"legs": ["shelf_a_posts", "shelf_b_posts"]}
    entry = {"id": "legs_down", "kind": "contact", "outputs": ["legs", "floor"],
             "every": ["legs"], "owner": "human"}
    e = _entry(_report(_shelves([entry], groups=groups, floor=True)), "legs_down")
    assert e["ok"] is True
    assert e["every"] == {"shelf_a_posts": 2, "shelf_b_posts": 2}
    assert e["members"] == {"legs": ["shelf_a_posts", "shelf_b_posts"]}
    assert "unchecked" not in e and e["distance"] == 0.0

    e = _entry(_report(_shelves([entry], groups=groups, floor=True, lift_b=0.02)), "legs_down")
    assert e["ok"] is False
    assert [c for c, _ in e["detail"]["failing_copies"]] == ["shelf_b_posts[0]", "shelf_b_posts[1]"]
    assert all(abs(d - 0.02) < 1e-6 for _, d in e["detail"]["failing_copies"])
    assert e["every_max"][0] == "shelf_b_posts[0]" or e["every_max"][0] == "shelf_b_posts[1]"


def test_group_every_judges_a_plain_member_apart_from_the_others():
    """``every`` naming a group quantifies every member's copies, distance
    and ``min_area`` alike (DESIGN 5.2): a plain member is its one copy,
    so three crates each seated on 0.18 m^2 miss ``min_area`` 0.2 one by
    one although the group shares 0.54, and a lifted crate fails by name
    although its neighbors touch."""
    def crates(lift: float) -> str:
        nodes, outputs = _box("floor", [4.0, 0.1, 2.0], [0.0, -0.05, 0.0]), {"floor": "floor"}
        for i in range(3):
            y = 0.15 + (lift if i == 1 else 0.0)
            nodes += _box(f"crate_{i}", [0.6, 0.3, 0.3], [i - 1.0, y, 0.0])
            outputs[f"crate_{i}"] = f"crate_{i}"
        return json.dumps({
            "build_version": "fixture", "nodes": nodes, "outputs": outputs,
            "groups": {"crates": ["crate_0", "crate_1", "crate_2"]},
            "asserts": [
                {"id": "seated", "kind": "contact", "outputs": ["crates", "floor"],
                 "every": ["crates"], "min_area": 0.2, "owner": "agent"},
                {"id": "seated_low", "kind": "contact", "outputs": ["crates", "floor"],
                 "every": ["crates"], "min_area": 0.15, "owner": "agent"},
                {"id": "union", "kind": "contact", "outputs": ["crates", "floor"],
                 "min_area": 0.2, "owner": "agent"},
            ],
        })

    report = _report(crates(0.0))
    e = _entry(report, "seated")
    assert e["ok"] is False
    assert e["every"] == {"crate_0": 1, "crate_1": 1, "crate_2": 1}
    assert abs(e["area"] - 0.54) < 1e-6 and abs(e["area_min"][1] - 0.18) < 1e-6
    assert [c for c, _ in e["detail"]["bare_copies"]] == ["crate_0", "crate_1", "crate_2"]
    assert _entry(report, "seated_low")["ok"] is True
    assert _entry(report, "union")["ok"] is True  # the existential form: the union

    e = _entry(_report(crates(0.02)), "seated_low")
    assert e["ok"] is False
    [(name, d)] = e["detail"]["failing_copies"]
    assert name == "crate_1" and abs(d - 0.02) < 1e-6
    assert e["every_max"][0] == "crate_1" and e["distance"] == 0.0


def test_contact_quantified_on_one_side_draws_no_unquantified_warning():
    """The `contact_unquantified` row is for an entry that quantified
    nothing (DESIGN 5.2): "every post on some slab" judged one side
    copy by copy, so the instanced side it left existential stays in
    `unchecked` and the report's warnings carry no row for it. An
    `every` over one plain output judges the union alone, and its row
    stays."""
    nodes = _box("slab_b", [0.3, 0.1, 0.3], [0.0, -0.05, 0.0])
    nodes.append({"id": "slabs", "op": "array", "inputs": ["slab_b"],
                  "params": {"count": 2, "step_translate": [1.0, 0.0, 0.0]}})
    nodes += _box("post_b", [0.1, 1.0, 0.1], [0.0, 0.5, 0.0])
    nodes.append({"id": "posts", "op": "array", "inputs": ["post_b"],
                  "params": {"count": 2, "step_translate": [1.0, 0.0, 0.0]}})

    def warnings(entry: dict) -> tuple[dict, list[dict]]:
        report = _report(json.dumps({
            "build_version": "fixture", "nodes": nodes,
            "outputs": {"slabs": "slabs", "posts": "posts", "base": "slab_b"},
            "asserts": [entry],
        }))
        rows = [w for w in report.get("warnings", []) if w["warning"] == "contact_unquantified"]
        return _entry(report, "stand"), rows

    plain = {"id": "stand", "kind": "contact", "outputs": ["posts", "slabs"], "owner": "agent"}
    e, rows = warnings(plain)
    assert e["ok"] is True and [w["outputs"] for w in rows] == [["posts", "slabs"]]
    e, rows = warnings(dict(plain, every=["posts"]))
    assert e["ok"] is True and e["unchecked"] == {"instanced_copies": ["slabs"]}
    assert rows == []
    vacuous = dict(plain, outputs=["base", "posts"], every=["base"])
    e, rows = warnings(vacuous)
    assert e["ok"] is True and e["every"] == {"base": 1}
    assert [w["outputs"] for w in rows] == [["posts"]]


def test_group_embedded_pair_and_extent_unite_members():
    """embedded's pair form measures the union's intersection depth and
    cedes every member cross pair from the guard; extent's output form
    reads the members' united bounds (DESIGN 5.2)."""
    entries = [
        {"id": "legs_in_floor", "kind": "embedded", "outputs": ["legs", "floor"],
         "min": 0.01, "max": 0.03, "owner": "human"},
        {"id": "a_width", "kind": "extent", "outputs": ["shelf_a"],
         "measure": "size_x", "value": 1.0, "owner": "human"},
        {"id": "a_height", "kind": "extent", "outputs": ["shelf_a"],
         "measure": "size_y", "value": 1.05, "owner": "human"},
    ]
    groups = {
        "legs": ["shelf_a_posts", "shelf_b_posts"],
        "shelf_a": ["shelf_a_board", "shelf_a_posts"],
    }
    # The floor's top raised 0.02: the posts sink that deep into it.
    report = _report(_shelves(entries, groups=groups, floor=True, floor_top=0.02))
    emb = _entry(report, "legs_in_floor")
    assert emb["ok"] is True and 0.01 <= emb["depth"] <= 0.03
    guard = _entry(report, "no_overlap")
    assert guard["ok"] is True
    assert guard["ceded"]["pairs"] == [["floor", "shelf_a_posts"], ["floor", "shelf_b_posts"]]
    assert _entry(report, "a_width")["ok"] is True
    height = _entry(report, "a_height")
    assert height["ok"] is True and abs(height["measured"] - 1.05) < 1e-6  # float32 storage
    assert report["ok"] is True


def test_group_is_refused_where_a_kind_names_one_output():
    """count, mated, extent's face form, clearance ``self`` and
    embedded's copy forms name one output's own copies or faces; a
    group there is refused with its members listed (DESIGN 5.2)."""
    def load(entry):
        return asset_mod.loads(_shelves([dict(entry, owner="agent")]))

    for entry, needle in [
        ({"id": "n", "kind": "count", "outputs": ["shelf_a"], "value": 2},
         "count counts one output's copies"),
        ({"id": "m", "kind": "mated", "faces": [["shelf_a", "top"], ["shelf_b_board", "bottom"]]},
         "mated names a tagged face set of one output"),
        ({"id": "x", "kind": "extent", "faces": ["shelf_a", "top"], "measure": "size_x", "value": 1.0},
         "extent names a tagged face set of one output"),
        ({"id": "s", "kind": "clearance", "outputs": ["shelf_a", "shelf_b_board"], "self": True, "min": 0.001},
         "'self' judges one output's own copy pairs"),
        ({"id": "o", "kind": "clearance", "outputs": ["shelf_a"], "others": True, "self": True, "min": 0.001},
         "'self' judges one output's own copy pairs"),
        ({"id": "e", "kind": "embedded", "outputs": ["shelf_a"], "self": True},
         "'self'/'connected' judge one output's own copy pairs"),
        ({"id": "c", "kind": "embedded", "outputs": ["shelf_a"], "connected": True, "min": 0.01},
         "'self'/'connected' judge one output's own copy pairs"),
    ]:
        with pytest.raises(AssetError) as exc:
            load(entry)
        assert exc.value.code == "bad_assert_outputs", entry
        assert "'shelf_a' is an output group" in exc.value.message, entry
        assert needle in exc.value.message, entry
        assert "['shelf_a_board', 'shelf_a_posts']" in exc.value.message, entry
    # An unknown name lists both vocabularies.
    with pytest.raises(AssetError) as exc:
        load({"id": "g", "kind": "clearance", "outputs": ["shelf_a", "ghost"], "min": 0.001})
    assert "groups: ['shelf_a', 'shelf_b']" in exc.value.message


def test_two_names_of_one_entry_must_stand_for_disjoint_outputs():
    """A relating kind takes two disjoint subjects (DESIGN 5.2):
    distinct names stop being enough once a name can be a group, so a
    group beside one of its own members — or beside a group sharing
    one — is refused, naming the shared outputs. An output measured
    against itself is distance 0 whatever the geometry does."""
    groups = {
        "shelf_a": ["shelf_a_board", "shelf_a_posts"],
        "all_boards": ["shelf_a_board", "shelf_b_board"],
    }

    def load(entry):
        return asset_mod.loads(_shelves([dict(entry, owner="agent")], groups=groups))

    for entry, shared in [
        ({"id": "c", "kind": "clearance", "outputs": ["shelf_a", "shelf_a_board"],
          "min": 0.5}, "['shelf_a_board']"),
        ({"id": "t", "kind": "contact", "outputs": ["shelf_a_board", "shelf_a"]},
         "['shelf_a_board']"),
        ({"id": "g", "kind": "contact", "outputs": ["shelf_a", "all_boards"]},
         "['shelf_a_board']"),
        ({"id": "e", "kind": "embedded", "outputs": ["shelf_a", "all_boards"],
          "min": 0}, "['shelf_a_board']"),
    ]:
        with pytest.raises(AssetError) as exc:
            load(entry)
        assert exc.value.code == "bad_assert_outputs", entry
        assert f"both stand for {shared}" in exc.value.message, entry
        assert "disjoint subjects" in exc.value.message, entry
    # A group beside an output outside it, and two groups that share
    # nothing, stand: overlap is the refusal, not the group name.
    assert load({"id": "ok", "kind": "clearance",
                 "outputs": ["shelf_a", "shelf_b_posts"], "min": 0.5}) is not None
    assert load({"id": "ok2", "kind": "clearance",
                 "outputs": ["all_boards", "shelf_a_posts"], "min": 0.5}) is not None


def test_group_with_an_image_member_is_unevaluatable():
    """A group is geometry only at check time (the parser cannot know an
    output's kind): an image member fails the entry with the reason,
    never a silent skip (DESIGN 5.2)."""
    groups = {"mixed": ["shelf_a_board", "tex"]}
    entry = {"id": "m", "kind": "clearance", "outputs": ["mixed", "shelf_b_board"],
             "min": 0.5, "owner": "agent"}
    e = _entry(_report(_shelves([entry], groups=groups, image=True)), "m")
    assert e["ok"] is False and "output 'tex' is an image" in e["fix"]


def test_sub_asset_groups_resolve_in_the_consumer_bundle(tmp_path):
    """A referenced asset's groups travel with its assert bundle (DESIGN
    4.6): the sub-asset's group assert is checked in its own namespace
    from the consumer's validate."""
    ws = tmp_path / "ws"
    proj = ws / "yard"
    (proj / "subassets").mkdir(parents=True)
    sub = json.loads(_shelves([
        {"id": "a_clear", "kind": "clearance", "outputs": ["shelf_a"],
         "others": True, "min": 0.5, "owner": "human"},
    ]))
    (proj / "subassets" / "shelves.mora").write_text(
        asset_mod.dumps(asset_mod.loads(json.dumps(sub))), encoding="utf-8", newline="\n"
    )
    root = {
        "build_version": "fixture",
        "nodes": [
            {"id": "ref", "op": "asset_ref",
             "params": {"asset": "subassets/shelves.mora", "output": "shelf_a_board"}},
        ],
        "outputs": {"board": "ref"},
        "asset_checks": {"grounded": False},  # the board alone floats; the posts stay behind
    }
    (proj / "yard.mora").write_text(
        asset_mod.dumps(asset_mod.loads(json.dumps(root))), encoding="utf-8", newline="\n"
    )
    flatten.write_project(proj, proj)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["validate", str(proj)])
    report = json.loads(buf.getvalue())
    assert rc == 0, buf.getvalue()
    rows = [r for r in report["ref_assets"] if r["asset"] == "subassets/shelves.star"]
    assert rows and rows[0]["asserts_total"] == 2 and rows[0]["asserts_ok"] == 2
