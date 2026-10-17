"""The human's accepted knob values, ``params.json``, as the build
reads them (DESIGN 4.6, 4.8, 7.2)."""

from __future__ import annotations

import json

import pytest

from moraine3d.core import accepted
from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.errors import BuildError, MoraineError
from moraine3d.recipe.build import build

from support.recipes import build_recipe, ir, workspace, write_files

KNOB = (
    'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
    'mo.output("main", mo.box(id = "bx", size = [w, 1, 1]))\n'
)


def _knob(doc: dict, pid: str = "w") -> dict:
    return next(d for d in doc["params"] if d["id"] == pid)


# -- the file ------------------------------------------------------------------


def test_the_file_holds_one_id_per_line_in_id_order():
    """DESIGN 7.2: `params.json` is pure data, one `param_id` per line
    in canonical form, so the diff of an accept is the changed lines
    alone."""
    text = accepted.dumps({"width": [2.0, 1.0], "depth": 0.5, "label": "A"})
    assert text == (
        '{\n"depth": 0.5,\n"label": "A",\n"width": [2.0, 1.0]\n}\n'
    )
    assert accepted.loads(text.encode("utf-8"), "params.json") == {
        "depth": 0.5, "label": "A", "width": [2.0, 1.0],
    }
    assert accepted.dumps({}) == "{\n}\n"


def test_a_byte_order_mark_is_no_part_of_the_file(tmp_path):
    """DESIGN 7.2: `params.json` is the human's file, so it is read as
    an editor of theirs may have saved it."""
    write_files(tmp_path, {"asset.star": KNOB})
    (tmp_path / "params.json").write_bytes(b'\xef\xbb\xbf{\n"w": 2.5\n}\n')
    result = build(tmp_path / "asset.star", write=False)
    assert _knob(ir(result))["value"] == 2.5


def test_a_write_replaces_the_file_whole(tmp_path):
    """DESIGN 7.2: the file is written whole through the one project
    writer, and a changed value moves one line."""
    path = tmp_path / "params.json"
    accepted.write(path, {"a": 1, "b": 2.5, "c": True})
    before = path.read_bytes()
    assert b"\r" not in before
    accepted.write(path, {"a": 1, "b": 2.75, "c": True})
    after = path.read_bytes()
    changed = [
        (x, y) for x, y in zip(before.split(b"\n"), after.split(b"\n")) if x != y
    ]
    assert changed == [(b'"b": 2.5,', b'"b": 2.75,')]
    assert [p.name for p in tmp_path.iterdir()] == ["params.json"]


@pytest.mark.parametrize("text, said, line", [
    ('{"w": 2.5', "not valid JSON", 1),
    ('[1, 2]\n', "does not hold a JSON object", None),
    ('{\n"w": 2.5,\n"w": 2.0\n}\n', "holds 'w' twice", 2),
    ('{\n"a b": 1\n}\n', "is no param id", 2),
    ('{\n"w": null\n}\n', "no knob takes", 2),
    ('{\n"w": {"x": 1}\n}\n', "no knob takes", 2),
    ('{\n"w": NaN\n}\n', "non-finite", None),
])
def test_a_file_the_build_cannot_read_fails_the_build(tmp_path, text, said, line):
    """DESIGN 8.2: errors remain for an asset the runtime cannot read.
    The error names the file, the line where one is known, and what
    the file holds."""
    write_files(tmp_path, {"asset.star": KNOB, "params.json": text})
    with pytest.raises(BuildError) as e:
        build(tmp_path / "asset.star", write=False)
    assert e.value.code == "bad_params_file"
    assert said in e.value.message and "params.json" in e.value.message
    assert e.value.detail["file"] == "params.json"
    assert e.value.detail.get("line") == line


# -- the order of resolution ---------------------------------------------------


def test_an_accepted_value_takes_precedence_over_the_default(tmp_path):
    """DESIGN 4.8: values resolve in this order: the recipe's default,
    the accepted value in `params.json`, then a consumer's override.
    Later values take precedence. `mo.param` returns the resolved
    value, so the graph is built from it."""
    doc = ir(build_recipe(tmp_path / "plain", KNOB, write=False))
    assert _knob(doc)["value"] == 2.0 and "from" not in _knob(doc)
    doc = ir(build_recipe(
        tmp_path / "accepted", KNOB, write=False,
        files={"params.json": '{\n"w": 2.5\n}\n'},
    ))
    assert _knob(doc)["value"] == 2.5 and _knob(doc)["from"] == "accepted"
    assert _knob(doc)["default"] == 2.0
    assert doc["nodes"][0]["params"]["size"] == [2.5, 1.0, 1.0]


def test_a_consumers_value_takes_precedence_over_the_accepted_one(tmp_path):
    """DESIGN 4.8: a consumer's override is the last of the three. The
    viewer's overlay is a consumer's value (DESIGN 4.6)."""
    write_files(tmp_path, {"asset.star": KNOB, "params.json": '{\n"w": 2.5\n}\n'})
    result = build(tmp_path / "asset.star", offered={"w": 1.5}, strict_offers=False)
    doc = ir(result)
    assert _knob(doc)["value"] == 1.5 and _knob(doc)["from"] == "override"
    assert _knob(doc)["default"] == 2.0


def test_an_unset_is_the_state_without_the_line(tmp_path):
    """DESIGN 7.2: a `null` value unsets the knob in the overlay: the
    preview shows the recipe's default, as the state without the
    line. The entry then names no other source. A consumer's `null`
    is no unset: it is refused like any value of another kind."""
    write_files(tmp_path, {"asset.star": KNOB, "params.json": '{\n"w": 2.5\n}\n'})
    doc = ir(build(tmp_path / "asset.star", offered={"w": None}, strict_offers=False))
    assert _knob(doc) == {
        "id": "w", "value": 2.0, "min": 1.0, "max": 3.0, "site": "asset.star:1",
    }
    assert doc["nodes"][0]["params"]["size"] == [2.0, 1.0, 1.0]
    with pytest.raises(BuildError) as e:
        build(tmp_path / "asset.star", offered={"w": None})
    assert e.value.code == "param_type_mismatch"


def test_an_accepted_value_takes_the_knobs_number_type(tmp_path):
    """DESIGN 4.6: a value offered to a knob is normalized to the
    number type the knob declares, and an accepted value is read by
    the same rule."""
    result = build_recipe(
        tmp_path, KNOB, write=False, files={"params.json": '{\n"w": 3\n}\n'},
    )
    assert '"value": 3.0' in result.target.text


def test_a_siblings_accepted_value_reaches_its_consumer(tmp_path):
    """DESIGN 4.6: a project's `params.json` holds the accepted values
    of its root asset's knobs, and the root is built under them
    whoever builds it, so a project reads the same alone and inside
    its consumer. An override still takes precedence."""
    root = workspace(tmp_path, {
        "rail": {
            "rail.star": (
                'g = mo.param("gauge", 1.0, min = 0.5, max = 2.0)\n'
                'mo.output("rail", mo.box(id = "bx", size = [g, 1, 1]))\n'
            ),
            "params.json": '{\n"gauge": 1.5\n}\n',
        },
        "yard": {
            "yard.star": (
                'mo.output("a", mo.asset_ref(project = "rail", output = "rail", id = "a"))\n'
                'mo.output("b", mo.asset_ref(project = "rail", output = "rail", id = "b",'
                ' overrides = {"gauge": 0.75}))\n'
            ),
        },
    })
    result = build(root / "yard" / "yard.star", write=False)
    by_name = {p.name: json.loads(b.text) for p, b in result.assets.items()}
    plain = by_name.pop("rail.mora")
    assert _knob(plain, "gauge")["value"] == 1.5
    assert _knob(plain, "gauge")["from"] == "accepted"
    by_name.pop("yard.mora")
    (variant,) = by_name.values()
    assert _knob(variant, "gauge")["value"] == 0.75
    assert _knob(variant, "gauge")["from"] == "override"
    alone = build(root / "rail" / "rail.star", write=False)
    assert alone.target.text == result.assets[alone.target.ir_path].text


def test_an_accepted_value_is_no_part_of_a_variants_identity(tmp_path):
    """DESIGN 4.6: a variant's identity is its asset and its
    overrides, so an accepted value moves the IR's content and never
    its name."""
    files = {
        "rail": {"rail.star": (
            'g = mo.param("gauge", 1.0, min = 0.5, max = 2.0)\n'
            'h = mo.param("height", 0.2, min = 0.1, max = 0.5)\n'
            'mo.output("rail", mo.box(id = "bx", size = [g, h, 1]))\n'
        )},
        "yard": {"yard.star": (
            'mo.output("b", mo.asset_ref(project = "rail", output = "rail", id = "b",'
            ' overrides = {"gauge": 0.75}))\n'
        )},
    }
    before = build(workspace(tmp_path / "a", files) / "yard" / "yard.star", write=False)
    files["rail"]["params.json"] = '{\n"height": 0.3\n}\n'
    after = build(workspace(tmp_path / "b", files) / "yard" / "yard.star", write=False)
    names = lambda r: sorted(p.name for p in r.assets)  # noqa: E731
    assert names(before) == names(after)
    variant = next(b for p, b in after.assets.items() if p.name.count(".") == 2)
    assert _knob(json.loads(variant.text), "height")["value"] == 0.3


def test_a_sub_assets_knob_takes_no_value_from_the_file(tmp_path):
    """DESIGN 4.6: an override reaches only the directly referenced
    asset's declared params, and `params.json` holds the root asset's
    knobs, so a sub-asset's knob is set by the asset that references
    it. An id the root declares no knob for is reported, never
    applied."""
    write_files(tmp_path, {
        "asset.star": 'mo.output("a", mo.asset_ref("subassets/part.star", "p", id = "a"))\n',
        "subassets/part.star": (
            'w = mo.param("w", 0.5, min = 0.1, max = 3.0)\n'
            'mo.output("p", mo.box(id = "p", size = [w, 1, 1]))\n'
        ),
        "params.json": '{\n"w": 2.5\n}\n',
    })
    result = build(tmp_path / "asset.star", write=False)
    part = next(b for p, b in result.assets.items() if p.name == "part.mora")
    assert _knob(json.loads(part.text))["value"] == 0.5
    assert "from" not in _knob(json.loads(part.text))
    (row,) = [w for w in result.warnings if w["warning"] == "accepted_value_unused"]
    assert row["file"] == "params.json" and row["params"] == ["w"]
    assert row["count"] == 1 and "fix" in row
    alone = build(tmp_path / "subassets" / "part.star", write=False)
    assert _knob(ir(alone))["value"] == 0.5 and alone.warnings == []


# -- findings about an accepted value ------------------------------------------


def test_an_accepted_value_a_narrowed_range_left_outside_is_reported(tmp_path):
    """DESIGN 4.6: an asset's own value outside its range is a failing
    entry of the report, an accepted value that a narrowed range left
    outside it included. The build does not refuse it."""
    result = build_recipe(
        tmp_path, KNOB.replace("max = 3.0", "max = 2.2"), write=False,
        files={"params.json": '{\n"w": 2.5\n}\n'},
    )
    (entry,) = params_mod.range_entries(result.target.asset)
    assert entry["ok"] is False and entry["value"] == 2.5
    assert entry["from"] == "accepted" and entry["site"] == "asset.star:1"
    assert "'params.json'" in entry["fix"] and "viewer" in entry["fix"]


def test_a_value_of_another_kind_leaves_the_default_and_fails_its_entry(tmp_path):
    """DESIGN 8.2: a finding about an asset the runtime can still
    evaluate is a failing entry beside every other entry. An accepted
    value of another kind than the knob's cannot be the knob's value,
    so the build keeps the recipe's default and the report says so."""
    result = build_recipe(tmp_path, (
        'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
        't = mo.param("tag", "a")\n'
        'mo.output("main", mo.material(mo.box(id = "bx", size = [w, 1, 1]),'
        ' id = "mt", name = t))\n'
    ), write=False, files={"params.json": '{\n"tag": 3,\n"w": [1.0, 2.0]\n}\n'})
    doc = ir(result)
    assert _knob(doc)["value"] == 2.0 and _knob(doc)["mismatch"] == [1.0, 2.0]
    assert _knob(doc, "tag")["value"] == "a" and "from" not in _knob(doc, "tag")
    entries = {e["param"]: e for e in params_mod.range_entries(result.target.asset)}
    assert set(entries) == {"w", "tag"}
    for entry in entries.values():
        assert entry["ok"] is False and "params.json" in entry["fix"]
    assert entries["tag"]["mismatch"] == 3


def test_a_mismatch_is_the_projects_finding_and_not_a_variants(tmp_path):
    """DESIGN 4.8: a consumer's override takes precedence, so the
    value a variant is built under was never the file's to give."""
    write_files(tmp_path, {"asset.star": KNOB, "params.json": '{\n"w": "wide"\n}\n'})
    result = build(tmp_path / "asset.star", offered={"w": 1.5}, strict_offers=False)
    assert "mismatch" not in _knob(ir(result))


def test_the_loader_reads_the_two_keys_and_refuses_another_source():
    """DESIGN 4.1: a `params` entry says where its value came from
    when that is not the recipe's default, and then holds the default
    too."""
    doc = {
        "build_version": "fixture",
        "nodes": [{"id": "bx", "op": "box", "params": {"size": [1, 1, 1]}}],
        "params": [{
            "id": "w", "value": 2.0, "from": "accepted", "default": 1.5,
            "min": 1.0, "max": 3.0,
        }],
        "outputs": {"main": "bx"},
    }
    sc = asset_mod.loads(json.dumps(doc))
    assert '"id": "w", "value": 2.0, "from": "accepted", "default": 1.5' in asset_mod.dumps(sc)
    for broken in ({"from": "default"}, {"default": None}, {"default": "a"}):
        bad = json.loads(json.dumps(doc))
        bad["params"][0].update(broken)
        if broken.get("default", 0) is None:
            del bad["params"][0]["default"]
        with pytest.raises(MoraineError) as e:
            asset_mod.loads(json.dumps(bad))
        assert e.value.code == "bad_param_decl"
    bad = json.loads(json.dumps(doc))
    del bad["params"][0]["from"]
    with pytest.raises(MoraineError) as e:
        asset_mod.loads(json.dumps(bad))
    assert e.value.code == "bad_param_decl"


# -- the revision --------------------------------------------------------------


def test_the_file_is_named_before_it_exists_and_joins_the_revision(tmp_path):
    """DESIGN 4.6: `serve` watches every `params.json`. DESIGN 6: the
    revision is over every contributing file's bytes, `params.json`
    among them."""
    write_files(tmp_path, {"asset.star": KNOB})
    first = build(tmp_path / "asset.star", write=False)
    path = (tmp_path / "params.json").resolve()
    assert path in first.named and path not in first.sources
    write_files(tmp_path, {"params.json": '{\n"w": 2.5\n}\n'})
    second = build(tmp_path / "asset.star", write=False)
    assert path in second.sources
    assert second.contents[path] == b'{\n"w": 2.5\n}\n'
    hashes = {
        asset_mod.content_hash(r.sources, r.root, read=r.contents)
        for r in (first, second)
    }
    assert len(hashes) == 2


# -- what the report says of an accepted value ---------------------------------


def test_the_report_names_the_knobs_that_hold_an_accepted_value(tmp_path, monkeypatch):
    """DESIGN 6: `--failures-only` names the knobs that hold an
    accepted value (`params_accepted`), since a default the agent
    edits moves nothing while one stands. A knob without a range has
    an entry where `params.json` holds a value for it."""
    from moraine3d.app import cli

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_LOG_DIR", "")
    asset = (
        'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
        'd = mo.param("d", 1.0, min = 0.5, max = 3.0)\n'
        't = mo.param("tag", "a")\n'
        'body = mo.transform(mo.box(id = "bx", size = [w, 1, d]), id = "up",'
        ' translate = [0, 0.5, 0])\n'
        'mo.output("main", mo.material(body, id = "mt", name = t))\n'
    )
    project = write_files(tmp_path / "p", {"asset.star": asset})
    plain, _ = cli._validate_report(project / "asset.star", None, True)
    assert "params_accepted" not in plain and plain["params_total"] == 2
    write_files(project, {"params.json": '{\n"tag": "b",\n"w": 2.5\n}\n'})
    full, ok = cli._validate_report(project / "asset.star", None, False)
    assert ok is True
    assert {e["param"]: e.get("from") for e in full["params"]} == {
        "w": "accepted", "d": None, "tag": "accepted",
    }
    brief, _ = cli._validate_report(project / "asset.star", None, True)
    assert brief["params_accepted"] == ["w", "tag"]
    assert (brief["params_total"], brief["params_ok"]) == (3, 3)
    assert "params" not in brief


def test_validate_takes_knob_values_without_writing_any_file(tmp_path, monkeypatch, capsys):
    """DESIGN 6: `validate --param ID=VALUE` judges the asset under the
    values given, over the accepted ones: the report names them, a
    value of the wrong kind is the error a consumer gets, one outside
    the declared range fails the knob's entry, and neither
    `params.json` nor the IR under `build/` is written."""
    from moraine3d.app.cli import main

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "proj"
    asset = (
        'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
        'body = mo.transform(mo.box(id = "bx", size = [w, 1, 1]), id = "up",'
        ' translate = [0, 0.5, 0])\n'
        'mo.output("main", body)\n'
    )
    write_files(project, {"asset.star": asset, "params.json": '{\n"w": 2.5\n}\n'})
    assert main(["validate", str(project)]) == 0
    capsys.readouterr()
    accepted_bytes = (project / "params.json").read_bytes()
    built = (project / "build" / "asset.mora").read_bytes()

    assert main(["validate", str(project), "--param", "w=1.5"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["params_given"] == {"w": 1.5}
    entry = next(e for e in report["params"] if e["param"] == "w")
    assert entry["value"] == 1.5 and entry["from"] == "override"
    assert main(["validate", str(project), "--failures-only", "--param", "w=1.5"]) == 0
    assert json.loads(capsys.readouterr().out)["params_given"] == {"w": 1.5}
    assert (project / "params.json").read_bytes() == accepted_bytes
    assert (project / "build" / "asset.mora").read_bytes() == built

    # A value outside the declared range is judged, and fails the
    # knob's entry alone (DESIGN 6).
    assert main(["validate", str(project), "--param", "w=9"]) == 1
    report = json.loads(capsys.readouterr().out)
    entry = next(e for e in report["params"] if e["param"] == "w")
    assert entry["ok"] is False and entry["value"] == 9.0 and entry["from"] == "override"
    assert "for this run alone" in entry["fix"] and "widen the declaration" in entry["fix"]
    assert report["params_given"] == {"w": 9} and report["ok"] is False
    assert (project / "params.json").read_bytes() == accepted_bytes
    assert main(["validate", str(project), "--param", "depth=1"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "param_unknown"
    assert main(["validate", str(project), "--param", "w"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "bad_param_flag"
    assert main(["validate", str(project / "build" / "asset.mora"), "--param", "w=1.5"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "param_needs_recipe"


def test_a_given_value_is_read_by_the_knobs_kind_and_kept_out_of_the_cache(
    tmp_path, monkeypatch, capsys,
):
    """DESIGN 6: a string knob takes `--param`'s VALUE as typed, an id
    given twice is refused, and the pass evaluates into the preview
    bucket (DESIGN 4.5), so a sweep leaves the persistent store as the
    file state's own validate left it."""
    from moraine3d.app.cli import main
    from moraine3d.cache import store

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    asset = (
        'w = mo.param("w", 2.0, min = 1.0, max = 3.0)\n'
        't = mo.param("tag", "a")\n'
        'body = mo.transform(mo.box(id = "bx", size = [w, 1, 1]), id = "up",'
        ' translate = [0, 0.5, 0])\n'
        'mo.output("main", mo.material(body, id = "mt", name = "m" + t))\n'
    )
    project = write_files(tmp_path / "proj", {"asset.star": asset})
    assert main(["validate", str(project)]) == 0
    capsys.readouterr()
    before = store.stats()

    assert main(["validate", str(project), "--param", "tag=2", "--param", "w=1.5"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["params_given"] == {"tag": "2", "w": 1.5}
    after = store.stats()
    assert {k: v for k, v in after.items() if k != "preview"} == {
        k: v for k, v in before.items() if k != "preview"
    }

    assert main(["validate", str(project), "--param", "w=1.5", "--param", "w=2.5"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "bad_param_flag"
    assert main(["validate", str(project), "--param", "w=wide"]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "bad_param_flag" and "with the default" in err["message"]
