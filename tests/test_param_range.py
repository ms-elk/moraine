"""A declared knob's value against its declared range (DESIGN 6, 7.2):
reported as an entry of the validate report, never a refusal that
replaces it."""

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError
from support import projects
from support.engine import Evaluation, export_asset
from moraine3d.verify.validate import (
    append_ref_asserts,
    failures_only,
    validate_outputs,
)


def _asset(lo: float, hi: float, **extra) -> str:
    """A body with a lid resting on it: two outputs, one authored
    assert, and one knob on the body's size declared over [lo, hi]."""
    doc = {
        "build_version": "fixture",
        "nodes": [
            {"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "up", "op": "transform", "inputs": ["bx"],
             "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "ld", "op": "box", "params": {"size": [1.0, 0.2, 1.0]}},
            {"id": "ld_up", "op": "transform", "inputs": ["ld"],
             "params": {"translate": [0.0, 1.1, 0.0]}},
        ],
        "params": [
            {"id": "width", "value": [1.0, 1.0, 1.0], "node": "bx", "param": "size", "min": lo, "max": hi},
        ],
        "outputs": {"body": "up", "lid": "ld_up"},
        "asserts": [
            {"id": "lid_on_body", "kind": "contact",
             "outputs": ["body", "lid"], "owner": "human"},
        ],
    }
    doc.update(extra)
    return json.dumps(doc)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def _project(tmp_path, text: str):
    proj = tmp_path / "proj"
    projects.write_project(proj, text, name="proj")
    return proj


def test_value_outside_the_range_loads_and_fails_its_entry():
    """DESIGN 6: a knob whose value sits outside its declared range is a
    failing entry carrying the value and the range; the asset loads."""
    sc = asset_mod.loads(_asset(2.0, 5.0))
    [entry] = params_mod.range_entries(sc)
    assert entry["param"] == "width" and entry["ok"] is False
    assert (entry["value"], entry["min"], entry["max"]) == ([1.0, 1.0, 1.0], 2.0, 5.0)
    assert "'bx.size'" in entry["fix"] and "[2.0, 5.0]" in entry["fix"]


def test_a_failing_knob_names_the_recipe_line_that_declares_it(cache):
    """DESIGN 6: every entry that names a knob carries its recipe site,
    and the fix names the line to edit."""
    proj = _project(cache, _asset(2.0, 5.0))
    [entry] = params_mod.range_entries(projects.load(proj))
    assert entry["ok"] is False
    assert "the default at proj.star:1" in entry["fix"]


def test_value_inside_the_range_still_carries_its_numbers():
    """DESIGN 6: every entry carries the value the evaluation used and
    the declared range, pass or fail; the ends are inclusive."""
    sc = asset_mod.loads(_asset(1.0, 5.0))
    assert params_mod.range_entries(sc) == [
        {"param": "width", "ok": True, "value": [1.0, 1.0, 1.0], "min": 1.0, "max": 5.0}
    ]


def test_one_component_outside_fails_the_vector():
    """DESIGN 7.2: a vector's range applies to every component."""
    doc = json.loads(_asset(0.5, 5.0))
    doc["nodes"][0]["params"]["size"] = [1.0, 0.2, 1.0]
    doc["params"][0]["value"] = [1.0, 0.2, 1.0]
    [entry] = params_mod.range_entries(asset_mod.loads(json.dumps(doc)))
    assert entry["ok"] is False


def test_a_knob_without_a_range_has_no_entry():
    """DESIGN 6: the list holds the knobs that declare a range; a text
    or boolean knob has nothing to judge."""
    doc = json.loads(_asset(1.0, 5.0))
    doc["nodes"].append({"id": "mt", "op": "material", "inputs": ["up"],
                         "params": {"name": "steel"}})
    doc["params"] = [
        {"id": "finish", "value": "steel", "node": "mt", "param": "name"}
    ]
    assert params_mod.range_entries(asset_mod.loads(json.dumps(doc))) == []


def test_a_malformed_declaration_is_still_refused():
    """DESIGN 8.2: input the runtime cannot read is an error; only a
    finding it can evaluate past becomes a report entry."""
    with pytest.raises(AssetError) as e:
        asset_mod.loads(_asset(5.0, 2.0))
    assert e.value.code == "bad_param_decl"


def test_validate_reports_every_entry_beside_the_failing_knob(cache, capsys):
    """DESIGN 6, 8.2: the knob's entry fails and gates, and the outputs,
    the asserts, and the asset checks are all still reported."""
    proj = _project(cache, _asset(2.0, 5.0))
    assert main(["validate", str(proj)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert set(report["outputs"]) == {"body", "lid"}
    assert all(o["ok"] for o in report["outputs"].values())
    assert [(a["assert"], a["ok"]) for a in report["asserts"]] == [
        ("no_overlap", True), ("lid_on_body", True),
    ]
    assert [(p["param"], p["ok"]) for p in report["params"]] == [("width", False)]


def test_validate_is_green_with_the_knob_in_range(cache, capsys):
    """DESIGN 6: a knob inside its range passes its entry."""
    proj = _project(cache, _asset(0.5, 5.0))
    assert main(["validate", str(proj)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert [(p["param"], p["ok"]) for p in report["params"]] == [("width", True)]


def test_a_asset_without_ranged_knobs_has_no_params_key():
    """DESIGN 6: like budgets, the list is present when declared."""
    report = validate_outputs({}, params=[])
    assert "params" not in report


def test_failures_only_carries_params():
    """DESIGN 6: `--failures-only` compresses the list to its counts and
    its failing entries, like every gating list."""
    sc = asset_mod.loads(_asset(2.0, 5.0))
    report = validate_outputs({}, params=params_mod.range_entries(sc))
    slim = failures_only(report)
    assert (slim["ok"], slim["params_total"], slim["params_ok"]) == (False, 1, 0)
    assert slim["params"] == report["params"]
    sc = asset_mod.loads(_asset(1.0, 5.0))
    slim = failures_only(validate_outputs({}, params=params_mod.range_entries(sc)))
    assert (slim["ok"], slim["params_total"], slim["params_ok"]) == (True, 1, 1)
    assert "params" not in slim


def test_export_stays_behind_the_same_gate(cache):
    """DESIGN 6: green never outruns export — a failing knob entry
    blocks the export with the report, and nothing is written."""
    proj = _project(cache, _asset(2.0, 5.0))
    sc = projects.load(proj)
    result = export_asset(sc, profile="gltf", out_dir=proj / "exports")
    assert result["ok"] is False
    assert [p["ok"] for p in result["validation"]["params"]] == [False]
    assert not (proj / "exports").exists()


def _consumer_report(cache, part_text: str, ref: dict):
    proj = cache / "proj"
    projects.write_project(proj, {
        "build_version": "fixture",
        "nodes": [{"id": "rf", "op": "asset_ref", "params": ref}],
        "outputs": {"main": "rf"},
    }, name="root", subassets={"part": part_text})
    sc = projects.load(proj)
    ev = Evaluation(sc)
    values = {n: r.value() for n, r in ev.records().items()}
    report = validate_outputs(
        values, checks_policy=sc.checks, asserts=sc.asserts,
        asset_checks=sc.asset_checks, params=params_mod.range_entries(sc),
    )
    return append_ref_asserts(report, ev.ref_asserts())


def test_a_referenced_assets_knob_fails_the_consumer(cache):
    """DESIGN 4.6, 6: a referenced asset's knob outside its range gates
    the consumer's validate as a stamped entry, never a silent green."""
    report = _consumer_report(
        cache, _asset(2.0, 5.0), {"asset": "subassets/part.mora", "output": "body"},
    )
    assert report["ok"] is False
    [entry] = report["params"]
    assert entry["param"] == "width" and entry["ok"] is False
    assert entry["asset"] == "subassets/part.star"
    [row] = report["ref_assets"]
    assert (row["params_total"], row["params_ok"]) == (1, 0)


def test_a_referenced_assets_green_knobs_fold_into_its_row(cache):
    """DESIGN 4.6: a bundle's green entries fold into the per-identity
    row as counts, so a kit costs one row, not one line per knob."""
    report = _consumer_report(
        cache, _asset(0.5, 5.0), {"asset": "subassets/part.mora", "output": "body"},
    )
    assert report["ok"] is True
    assert "params" not in report
    [row] = report["ref_assets"]
    assert (row["params_total"], row["params_ok"]) == (1, 1)


def test_an_override_moves_the_knob_back_into_range(cache):
    """DESIGN 4.6: the entry judges the value the identity evaluates
    with, so an override inside the range is green and carries its
    `overrides` on the row."""
    report = _consumer_report(
        cache, _asset(2.0, 5.0),
        {"asset": "subassets/part.mora", "output": "body",
         "overrides": {"width": [2.0, 2.0, 2.0]}},
    )
    assert "params" not in report
    [row] = report["ref_assets"]
    assert (row["params_total"], row["params_ok"]) == (1, 1)
    assert row["overrides"] == {"width": [2.0, 2.0, 2.0]}
