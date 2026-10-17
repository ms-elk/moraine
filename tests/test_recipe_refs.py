"""References in the build (DESIGN 4.6, 4.8): the closure a build
covers, the variant a consumer's overrides define, and what the build
writes."""

import hashlib
import json

import pytest

from moraine3d.errors import BuildError
from moraine3d.ops import asset_ref as ref_op
from moraine3d.recipe.build import build
from moraine3d.recipe.version import build_version
from support.links import link_dir as _link_dir
from support.recipes import build_recipe, ir, refusal, workspace, write_files

pytestmark = pytest.mark.real_stamp

PANEL = '''w = mo.param("width", 1.5, min = 0.5, max = 2.0)
n = mo.param("ribs", 3, min = 1, max = 9)
mo.output("panel", mo.box(id = "panel", size = [w, 1.0, 0.05]))
mo.output("ribs", mo.array(mo.box(id = "rib", size = [0.02, 1.0, 0.1]), id = "ribs", count = n, step_translate = [w / n, 0, 0]))
'''


def _ref(overrides: str = "", output: str = "panel", id: str = "rf") -> str:
    extra = f", overrides = {overrides}" if overrides else ""
    return (
        f'{id} = mo.asset_ref("subassets/panel.star", "{output}", id = "{id}"{extra})\n'
        f'mo.output("{id}", {id})\n'
    )


def _names(result) -> list[str]:
    return sorted(p.relative_to(result.root).as_posix() for p in result.assets)


# -- the closure -------------------------------------------------------------


def test_a_build_covers_the_reference_closure(tmp_path):
    """DESIGN 4.6: a build covers the reference closure, the root and
    its sub-assets, so the evaluator always finds the IR an `asset_ref`
    points at."""
    result = build_recipe(tmp_path, _ref(), files={"subassets/panel.star": PANEL})
    assert _names(result) == ["build/asset.mora", "build/subassets/panel.mora"]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in result.written) == _names(result)
    for path, built in result.assets.items():
        assert path.read_bytes().decode("utf-8") == built.text
    assert {p.name for p in result.sources} == {"asset.star", "panel.star"}


def test_the_reference_names_the_ir(tmp_path):
    """DESIGN 4.6: in the recipe a reference names the recipe file, and
    the build resolves each to the IR path."""
    result = build_recipe(tmp_path, _ref(), files={"subassets/panel.star": PANEL})
    [node] = ir(result)["nodes"]
    assert node["params"] == {"asset": "subassets/panel.mora", "output": "panel"}


@pytest.mark.parametrize("path", [
    "panel.star", "subassets/deep/panel.star", "subassets/panel.mora",
    "subassets/../panel.star", "subassets/panel.requirements.star",
])
def test_a_reference_path_is_exactly_one_level(tmp_path, path):
    """DESIGN 4.2: an `asset_ref` path must be exactly
    `subassets/<name>.star` in the recipe."""
    err = refusal(tmp_path, f'mo.output("a", mo.asset_ref("{path}", "panel", id = "a"))\n')
    assert err.code == "ref_layout"
    assert err.detail["site"] == "asset.star:1"


@pytest.mark.parametrize("path", [
    "../other/subassets/panel.star", "/abs/panel.star", "C:/abs/panel.star",
])
def test_a_reference_never_leaves_the_project(tmp_path, path):
    """DESIGN 4.6: an in-project reference path must resolve inside the
    project directory, and `build` rejects escapes."""
    err = refusal(tmp_path, f'mo.output("a", mo.asset_ref("{path}", "panel", id = "a"))\n')
    assert err.code == "ref_escape"
    assert err.detail == {"site": "asset.star:1", "path": path, "node": "a"}


@pytest.mark.parametrize("call,code", [
    ('mo.asset_ref("subassets/panel.star", "nope", id = "a")', "ref_unknown_output"),
    ('mo.asset_ref("subassets/gone.star", "panel", id = "a")', "recipe_not_found"),
    ('mo.asset_ref("subassets/panel.star", id = "a")', "bad_param"),
    ('mo.asset_ref("subassets/panel.star", "panel", "x", id = "a")', "bad_input"),
    ('mo.asset_ref("subassets/panel.star", "panel", id = "a", output = "panel")', "bad_input"),
    ('mo.asset_ref("subassets/panel.star", "panel", id = "a", project = "yard")', "bad_param"),
    ('mo.asset_ref(id = "a", output = "panel")', "bad_param"),
])
def test_a_reference_the_build_cannot_resolve_fails_at_its_line(tmp_path, call, code):
    """DESIGN 4.8: an unknown name fails the build with the site of the
    call that named it."""
    err = refusal(
        tmp_path, f'x = 1\nmo.output("a", {call})\n',
        files={"subassets/panel.star": PANEL},
    )
    assert err.code == code
    assert err.detail["site"] == "asset.star:2"


def test_the_build_detects_reference_cycles_and_reports_the_chain(tmp_path):
    """DESIGN 4.6: the reference graph is a DAG; the build detects
    reference cycles and reports the chain."""
    err = refusal(tmp_path, 'mo.output("a", mo.asset_ref("subassets/a.star", "a", id = "a"))\n', files={
        "subassets/a.star": 'mo.output("a", mo.asset_ref("subassets/b.star", "b", id = "a"))\n',
        "subassets/b.star": 'mo.output("b", mo.asset_ref("subassets/a.star", "a", id = "b"))\n',
    })
    assert err.code == "ref_cycle"
    assert err.detail["chain"] == [
        "asset.star", "subassets/a.star", "subassets/b.star", "subassets/a.star",
    ]


def test_an_error_inside_a_referenced_asset_names_its_own_line(tmp_path):
    """DESIGN 6: every evaluating command builds first, the closure
    included, and a build failure is reported with the recipe site."""
    err = refusal(tmp_path, _ref(), files={
        "subassets/panel.star": 'x = 1\nwhile x:\n    pass\n',
    })
    assert err.code == "unsupported_syntax"
    assert err.detail["site"] == "subassets/panel.star:2"


# -- overrides and variants ----------------------------------------------------


def test_an_overridden_reference_builds_its_own_ir(tmp_path):
    """DESIGN 4.6: an override is a value handed to the referenced
    asset's build, so an overridden reference builds its own IR under
    its override identity, and a use with no overrides shares the
    sub-asset's own."""
    result = build_recipe(
        tmp_path,
        _ref(id = "plain") + _ref('{"width": 1.0}', id = "narrow")
        + _ref('{"width": 1.0}', output = "ribs", id = "narrow_ribs"),
        files={"subassets/panel.star": PANEL},
    )
    identity = ref_op.override_identity({"width": 1.0})
    assert _names(result) == [
        "build/asset.mora",
        f"build/subassets/panel.{identity}.mora",
        "build/subassets/panel.mora",
    ]
    variant = result.assets[tmp_path / "build" / "subassets" / f"panel.{identity}.mora"]
    doc = json.loads(variant.text)
    assert doc["params"][0]["value"] == 1.0
    assert doc["nodes"][0]["params"]["size"][0] == 1.0
    # The derived value moved with the knob: the recipe ran under it.
    assert doc["nodes"][2]["params"]["step_translate"][0] == 1.0 / 3


def test_the_identity_is_the_hash_of_the_normalized_overrides(tmp_path):
    """DESIGN 4.6: the identity is the 24-hex hash of the normalized
    overrides."""
    token = json.dumps({"ribs": 5, "width": 1.0}, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    assert ref_op.override_identity({"width": 1.0, "ribs": 5}) == expected
    assert ref_op.override_identity({}) == ""
    # Two spellings of one value are one variant: the declaration
    # decides the number type, so 1 and 1.0 land as the same knob value.
    result = build_recipe(
        tmp_path,
        _ref('{"width": 1, "ribs": 5.0}', id = "a") + _ref('{"ribs": 5, "width": 1.0}', id = "b"),
        files={"subassets/panel.star": PANEL},
    )
    assert _names(result) == ["build/asset.mora", f"build/subassets/panel.{expected}.mora"]
    assert [n["params"]["overrides"] for n in ir(result)["nodes"]] == [
        {"width": 1.0, "ribs": 5}, {"width": 1.0, "ribs": 5},
    ]


@pytest.mark.parametrize("overrides,code", [
    ('{"width": 2.5}', "param_out_of_range"),
    ('{"width": "wide"}', "param_type_mismatch"),
    ('{"depth": 1.0}', "param_unknown"),
])
def test_an_override_the_declaration_refuses_fails_at_the_reference(tmp_path, overrides, code):
    """DESIGN 4.6: errors reuse the declaration contract, and the build
    is where it fails."""
    err = refusal(tmp_path, "x = 1\n" + _ref(overrides), files={"subassets/panel.star": PANEL})
    assert err.code == code
    assert err.detail["site"] == "asset.star:2"


def test_a_sibling_roots_variant_is_written_beside_the_sibling(tmp_path):
    """DESIGN 4.6: a variant is written beside the IR of its asset, in
    the referenced project's `build/`, whoever offers the overrides."""
    root = workspace(tmp_path, {
        "lamp": {"lamp_post.star": (
            'h = mo.param("shaft", 3.0, min = 0.5, max = 3.5)\n'
            'cap = mo.asset_ref("subassets/cap.star", "cap", id = "cap_ref", overrides = {"size": h / 10})\n'
            'mo.output("post", mo.cylinder(id = "post", radius = 0.05, height = h))\n'
            'mo.output("cap", cap)\n'
        ), "subassets/cap.star": (
            's = mo.param("size", 0.2, min = 0.05, max = 0.5)\n'
            'mo.output("cap", mo.sphere(id = "cap", radius = s))\n'
        )},
        "station": {"station.star": (
            'a = mo.asset_ref(project = "lamp", output = "post", id = "a", overrides = {"shaft": 1.0})\n'
            'b = mo.asset_ref(project = "lamp", output = "post", id = "b")\n'
            'mo.output("a", a)\n'
            'mo.output("b", b)\n'
        )},
    })
    result = build(root / "station" / "station.star")
    lamp = ref_op.override_identity({"shaft": 1.0})
    assert _names(result) == sorted([
        "station/build/station.mora",
        f"lamp/build/lamp_post.{lamp}.mora",
        "lamp/build/lamp_post.mora",
        f"lamp/build/subassets/cap.{ref_op.override_identity({'size': 0.1})}.mora",
        f"lamp/build/subassets/cap.{ref_op.override_identity({'size': 0.3})}.mora",
    ])
    nodes = ir(result)["nodes"]
    assert nodes[0]["params"] == {
        "project": "lamp", "asset": "lamp_post.mora", "output": "post",
        "overrides": {"shaft": 1.0},
    }
    assert nodes[0]["site"] == "station/station.star:1"
    assert nodes[1]["params"] == {"project": "lamp", "asset": "lamp_post.mora", "output": "post"}


@pytest.mark.parametrize("name,code", [
    ("nope", "unknown_project"),
    ("lib", "unknown_project"),
    (".moraine", "unknown_project"),
    ("two", "ambiguous_asset"),
])
def test_a_sibling_is_named_by_its_discovery_identity(tmp_path, name, code):
    """DESIGN 4.6: a cross-project reference names a sibling project by
    its discovery identity, never by relative path."""
    root = workspace(tmp_path, {
        "yard": {"yard.star": f'mo.output("a", mo.asset_ref(project = "{name}", output = "x", id = "a"))\n'},
        "two": {"a.star": "x = 1\n", "b.star": "x = 1\n"},
        "lib": {"parts.star": "x = 1\n"},
    })
    with pytest.raises(BuildError) as e:
        build(root / "yard" / "yard.star")
    assert e.value.code == code
    assert e.value.detail["site"] == "yard/yard.star:1"


def test_a_second_top_level_asset_names_where_the_rest_lives(tmp_path):
    """DESIGN 4.2: a second top-level asset file is refused, naming
    `lib/` for definitions and `subassets/` for parts."""
    root = workspace(tmp_path, {
        "yard": {"yard.star": 'mo.output("a", mo.asset_ref(project = "two", output = "x", id = "a"))\n'},
        "two": {"a.star": "x = 1\n", "b.star": "x = 1\n", "a.requirements.star": "K = 1\n"},
    })
    with pytest.raises(BuildError) as e:
        build(root / "yard" / "yard.star")
    assert "lib/" in e.value.message and "subassets/" in e.value.message
    assert e.value.detail["assets"] == ["a.star", "b.star"]


def test_a_reference_names_one_location(tmp_path):
    """DESIGN 4.6: a reference names a sub-asset by its path or a
    sibling project by its name, never both; the file name beside
    `project` in the IR is the build's to write."""
    root = workspace(tmp_path, {
        "yard": {"yard.star": (
            'mo.output("a", mo.asset_ref("lamp_post.star", "post", '
            'project = "lamp", id = "a"))\n'
        )},
        "lamp": {"lamp_post.star": 'mo.output("post", mo.box(id = "post"))\n'},
    })
    with pytest.raises(BuildError) as e:
        build(root / "yard" / "yard.star")
    assert e.value.code == "bad_param"
    assert "exactly one" in e.value.message
    assert e.value.detail["site"] == "yard/yard.star:1"


def test_a_project_outside_a_workspace_has_no_siblings(tmp_path):
    """DESIGN 4.6: cross-project references are allowed inside a
    workspace."""
    err = refusal(tmp_path / "yard", 'mo.output("a", mo.asset_ref(project = "lamp", output = "x", id = "a"))\n')
    assert err.code == "ref_no_workspace"


def test_a_sub_assets_requirements_are_built_into_every_variant(tmp_path):
    """DESIGN 5.2: a sub-asset's requirements are evaluated with the
    sub-asset's asserts, once per override identity, so a kit's own
    requirements hold for every variant a consumer builds."""
    result = build_recipe(
        tmp_path, _ref(id = "plain") + _ref('{"width": 1.0}', id = "narrow"),
        files={
            "subassets/panel.star": PANEL,
            "subassets/panel.requirements.star": (
                'mo.assert_extent(id = "thickness", outputs = ["panel"], measure = "size_z", value = 0.05)\n'
                'mo.budget(materials = 2)\n'
            ),
        },
    )
    for path, built in result.assets.items():
        if path.parent.name == "subassets":
            doc = json.loads(built.text)
            assert [(a["id"], a["owner"]) for a in doc["asserts"]] == [("thickness", "human")]
            assert doc["budgets"] == {"materials": 2}


# -- what the build writes -----------------------------------------------------


def test_the_ir_is_stamped_and_canonical(tmp_path):
    """DESIGN 4.8: the build writes the IR in canonical form, stamped
    with its `build_version` (DESIGN 4.1)."""
    from moraine3d.core import asset as asset_mod

    result = build_recipe(tmp_path, _ref(), files={"subassets/panel.star": PANEL})
    for path in result.assets:
        text = path.read_bytes().decode("utf-8")
        assert "\r" not in text
        sc = asset_mod.loads(text)
        assert sc.build_version == build_version()
        assert asset_mod.dumps(sc) == text


def test_a_build_that_changes_nothing_writes_nothing(tmp_path):
    """DESIGN 4.4: identical inputs produce byte-identical IR, so a
    second build finds its own bytes on disk."""
    first = build_recipe(tmp_path, _ref(), files={"subassets/panel.star": PANEL})
    stamps = {p: p.stat().st_mtime_ns for p in first.assets}
    again = build(tmp_path / "asset.star")
    assert again.written == [] and again.unchanged == 2
    assert {p: p.stat().st_mtime_ns for p in again.assets} == stamps
    assert [b.text for b in again.assets.values()] == [b.text for b in first.assets.values()]


def test_a_failed_build_writes_nothing(tmp_path):
    """DESIGN 6: a build failure is the command's failure; the IR on
    disk stays the last one a build finished."""
    build_recipe(tmp_path, _ref(), files={"subassets/panel.star": PANEL})
    before = (tmp_path / "build" / "asset.mora").read_bytes()
    write_files(tmp_path, {"asset.star": _ref() + "mo.box(id = \"rf\", size = [1, 1, 1])\n"})
    with pytest.raises(BuildError):
        build(tmp_path / "asset.star")
    assert (tmp_path / "build" / "asset.mora").read_bytes() == before
    assert not list(tmp_path.rglob("*.tmp*"))


def test_the_build_writes_under_build_and_nowhere_else(tmp_path):
    """DESIGN 4.8: nothing but the build writes under `build/`, and the
    build writes nowhere else."""
    write_files(tmp_path, {"asset.star": _ref(), "subassets/panel.star": PANEL})
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    build(tmp_path / "asset.star")
    new = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert new and all(p.relative_to(tmp_path).parts[0] == "build" for p in new)


def test_the_build_never_writes_through_a_link(tmp_path):
    """DESIGN 4.5: Moraine writes only what is its own; a `build/`
    spelled by a link leads outside the project, so the build refuses
    it and the link's target stays untouched."""
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "yard"
    write_files(project, {"asset.star": _ref(), "subassets/panel.star": PANEL})
    try:
        _link_dir(outside, project / "build")
    except OSError as e:
        pytest.skip(f"cannot create a directory link here: {e}")
    with pytest.raises(BuildError) as e:
        build(project / "asset.star")
    assert e.value.code == "build_is_link"
    assert list(outside.iterdir()) == []


def test_a_sub_asset_spelled_by_a_link_out_of_the_project_is_refused(tmp_path):
    """DESIGN 4.6: references never leave the workspace; an in-project
    reference path must resolve inside the project directory."""
    outside = tmp_path / "outside"
    write_files(outside, {"panel.star": PANEL})
    project = tmp_path / "yard"
    write_files(project, {"asset.star": _ref()})
    try:
        _link_dir(outside, project / "subassets")
    except OSError as e:
        pytest.skip(f"cannot create a directory link here: {e}")
    with pytest.raises(BuildError) as e:
        build(project / "asset.star")
    assert e.value.code == "ref_escape"


def test_the_report_names_what_was_built(tmp_path):
    """DESIGN 6: `build` reports `build_version`, the files written,
    and per asset the node, assert, and output counts."""
    result = build_recipe(tmp_path, _ref(), files={"subassets/panel.star": PANEL})
    report = result.report()
    assert report["ok"] is True
    assert report["build_version"] == build_version()
    assert report["written"] == ["build/asset.mora", "build/subassets/panel.mora"]
    assert report["assets"]["build/subassets/panel.mora"] == {
        "recipe": "subassets/panel.star", "nodes": 3, "asserts": 0, "outputs": 2,
    }
    json.dumps(report)


# -- one variant, several consumers ----------------------------------------------


LAMP = (
    'h = mo.param("height", 2.0, min = 1.0, max = 4.0)\n'
    'mo.output("post", mo.transform(mo.box(id = "b", size = [0.1, h, 0.1]), '
    'id = "post", translate = [0, h / 2, 0]))\n'
)


def test_two_projects_that_offer_the_same_overrides_read_one_variant(tmp_path, monkeypatch):
    """DESIGN 4.6: a variant's identity is its asset and its overrides,
    so two projects of one closure that offer a sibling the same
    overrides read one file, and both references evaluate."""
    from support.engine import Evaluation

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    root = workspace(tmp_path / "ws", {
        "lamp": {"lamp.star": LAMP},
        "station": {"station.star": (
            'mo.output("lamp", mo.asset_ref(project = "lamp", output = "post", '
            'id = "lamp", overrides = {"height": 3.0}))\n'
        )},
        "yard": {"yard.star": (
            'a = mo.asset_ref(project = "station", output = "lamp", id = "a")\n'
            'b = mo.transform(mo.asset_ref(project = "lamp", output = "post", '
            'id = "b_ref", overrides = {"height": 3}), id = "b", translate = [2, 0, 0])\n'
            'mo.output("a", a)\n'
            'mo.output("b", b)\n'
        )},
    })
    result = build(root / "yard" / "yard.star")
    lamp = ref_op.override_identity({"height": 3.0})
    assert _names(result) == sorted([
        "yard/build/yard.mora", "station/build/station.mora",
        f"lamp/build/lamp.{lamp}.mora",
    ])
    assert all((result.root / n).is_file() for n in _names(result))
    records = Evaluation(
        result.target.asset, irs={p: b.asset for p, b in result.assets.items()},
    ).records()
    assert records["a"].result_hash and records["b"].result_hash
    # From disk, as the next command opens them.
    from moraine3d.core import asset as asset_mod
    again = Evaluation(asset_mod.load(result.target.ir_path)).records()
    assert again["a"].key == records["a"].key


def test_a_variant_two_projects_read_is_one_instance(tmp_path, monkeypatch, capsys):
    """DESIGN 4.6: a sub-asset's asserts are checked once per (asset,
    override identity), so identical uses dedupe; a variant two
    projects of one closure read is verified once."""
    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    root = workspace(tmp_path / "ws", {
        "lamp": {
            "lamp.star": LAMP,
            "lamp.requirements.star": (
                'mo.assert_extent(id = "tall", outputs = ["post"], '
                'measure = "size_y", value = 9.0)\n'
            ),
        },
        "station": {"station.star": (
            'mo.output("lamp", mo.asset_ref(project = "lamp", output = "post", '
            'id = "lamp", overrides = {"height": 3.0}))\n'
        )},
        "yard": {"yard.star": (
            'a = mo.asset_ref(project = "station", output = "lamp", id = "a")\n'
            'b = mo.transform(mo.asset_ref(project = "lamp", output = "post", '
            'id = "b_ref", overrides = {"height": 3}), id = "b", translate = [2, 0, 0])\n'
            'mo.output("a", a)\n'
            'mo.output("b", b)\n'
        )},
    })
    assert main(["validate", str(root / "yard")]) == 1
    report = json.loads(capsys.readouterr().out)
    red = [a for a in report["asserts"] if not a["ok"]]
    assert [(a["asset"], a["assert"], a["overrides"]) for a in red] == [
        ("lamp/lamp.star", "tall", {"height": 3.0}),
    ]


def test_a_refused_build_has_written_nothing_behind_a_link(tmp_path):
    """DESIGN 4.5: the build writes inside the project and nowhere a
    link leads. Every path is judged before any file is written, a
    directory the build would have to create behind the link included,
    so the link's target stays empty whatever order the IRs sort in."""
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "yard"
    # `yard.mora` sorts after `subassets/`, whose directory does not
    # exist behind the link yet.
    write_files(project, {"yard.star": _ref(), "subassets/panel.star": PANEL})
    try:
        _link_dir(outside, project / "build")
    except OSError as e:
        pytest.skip(f"cannot create a directory link here: {e}")
    with pytest.raises(BuildError) as e:
        build(project / "yard.star")
    assert e.value.code == "build_is_link"
    assert str(tmp_path) not in e.value.message
    assert list(outside.iterdir()) == []


def test_a_reference_path_that_is_no_string_names_the_recipes_spelling(tmp_path):
    """DESIGN 4.6: in the recipe a sub-asset is referenced as
    `subassets/<name>.star`; the refusal names that spelling, never the
    IR's."""
    err = refusal(
        tmp_path, 'mo.output("x", mo.asset_ref(asset = 5, output = "o", id = "x"))\n',
    )
    assert err.code == "bad_param"
    assert "subassets/<name>.star" in err.message and ".mora" not in err.message
    assert err.detail["site"] == "asset.star:1"


def test_the_build_names_a_requirements_file_not_yet_written(tmp_path):
    """DESIGN 4.6: `serve` watches every requirements file of the
    closure and every `params.json`; the build names the file whether
    or not it exists, so creating it is an edit a watcher sees, and
    only the files read are the revision."""
    write_files(tmp_path, {"asset.star": _ref(), "subassets/panel.star": PANEL})
    result = build(tmp_path / "asset.star")
    rel = lambda files: sorted(p.relative_to(result.root).as_posix() for p in files)  # noqa: E731
    assert rel(result.sources) == ["asset.star", "subassets/panel.star"]
    assert rel(result.named) == [
        "asset.requirements.star", "asset.star", "params.json",
        "subassets/panel.requirements.star", "subassets/panel.star",
    ]
