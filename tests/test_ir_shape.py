"""The IR's shape (DESIGN 4.1): the build's stamp, the site of every
entry, ids as scope paths, and the knob entry."""

import json

import pytest

from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.engine import tiers
from moraine3d.errors import MoraineError, AssetError
from support.engine import evaluate_asset


def _doc(**extra) -> dict:
    doc = {
        "build_version": "fixture",
        "nodes": [
            {"id": "post/shaft", "op": "box", "params": {"size": [0.1, 2.0, 0.1]},
             "site": "yard/yard.star:4"},
            {"id": "slab", "op": "box", "params": {"size": [2.0, 0.1, 2.0]},
             "site": "yard/yard.star:9"},
        ],
        "params": [
            {"id": "height", "value": 2.0, "label": "Height", "min": 1.0,
             "max": 3.0, "site": "yard/yard.star:2"},
        ],
        "outputs": {"post": "post/shaft", "slab": "slab"},
        "asserts": [
            {"id": "post/stands", "kind": "contact", "outputs": ["post", "slab"],
             "owner": "agent", "site": "yard/yard.star:12"},
        ],
    }
    doc.update(extra)
    return doc


def _loads(doc: dict) -> asset_mod.Asset:
    return asset_mod.loads(json.dumps(doc))


def test_the_stamp_leads_the_document_and_round_trips():
    """DESIGN 4.1: `build_version` sits at the document root, the
    version of the build that wrote the file."""
    sc = _loads(_doc())
    assert sc.build_version == "fixture"
    text = asset_mod.dumps(sc)
    assert text.split("\n")[1] == '"build_version": "fixture",'
    assert asset_mod.dumps(asset_mod.loads(text)) == text


@pytest.mark.parametrize("stamp", [None, 0, ""])
def test_a_document_without_a_stamp_is_refused(stamp):
    """DESIGN 4.1: the stamp is an identity check, so a document that
    carries none names no build."""
    doc = _doc()
    if stamp is None:
        del doc["build_version"]
    else:
        doc["build_version"] = stamp
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_build_version"
    assert "moraine build" in e.value.message


def test_a_schema_number_is_refused_naming_the_stamp():
    """DESIGN 4.1: the IR has no version of its own; a root `version`
    is an unknown key, and the refusal names what the root carries."""
    with pytest.raises(AssetError) as e:
        _loads(_doc(version=0))
    assert e.value.code == "bad_root"
    assert "build_version" in e.value.message


def test_sites_round_trip_on_nodes_asserts_and_knobs():
    """DESIGN 4.1: every node carries its `site`, the recipe file and
    line that created it, and each assert the site of the call that
    declared it."""
    sc = _loads(_doc())
    assert sc.nodes["post/shaft"].site == "yard/yard.star:4"
    text = asset_mod.dumps(sc)
    assert '"params": {"size": [0.1, 2.0, 0.1]}, "site": "yard/yard.star:4"}' in text
    assert '"owner": "agent", "site": "yard/yard.star:12"}' in text
    assert '"max": 3.0, "site": "yard/yard.star:2"}' in text


def test_an_ir_written_by_hand_may_omit_the_site():
    """DESIGN 4.1: the site is the build's guarantee; the loader reads
    a document without one and writes none back."""
    doc = _doc()
    for entry in doc["nodes"] + doc["asserts"] + doc["params"]:
        del entry["site"]
    sc = _loads(doc)
    assert sc.nodes["slab"].site is None
    assert '"site"' not in asset_mod.dumps(sc)


@pytest.mark.parametrize("site", ["", 7, ["yard.star", 4]])
def test_a_site_that_is_not_a_location_is_refused(site):
    """DESIGN 4.1: the site is the recipe file and line."""
    doc = _doc()
    doc["nodes"][0]["site"] = site
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_site"


def test_the_site_joins_no_cache_key(tmp_path, monkeypatch):
    """DESIGN 4.5: keys are content-addressed; where a node was written
    is location, so moving a line re-keys nothing."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    here = evaluate_asset(_loads(_doc()))
    doc = _doc()
    doc["nodes"][0]["site"] = "yard/yard.star:40"
    there = evaluate_asset(_loads(doc))
    assert there["post"]["key"] == here["post"]["key"]


@pytest.mark.parametrize(
    "bad", ["/shaft", "post/", "post//shaft", "post/9shaft", "post shaft"],
)
def test_a_node_id_is_a_path_of_names(bad):
    """DESIGN 4.8: the scope separator is `/`, the root scope is empty,
    so a root node's id carries no leading separator, and each segment
    is a portable name."""
    doc = _doc()
    doc["nodes"][0]["id"] = bad
    doc["outputs"]["post"] = bad
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_node_id"


@pytest.mark.parametrize("bad", ["/stands", "post/", "post/9stands", "post stands"])
def test_an_assert_id_is_a_path_of_names(bad):
    """DESIGN 4.8: assert ids derive as node ids do, under the same
    separator and segment grammar."""
    doc = _doc()
    doc["asserts"][0]["id"] = bad
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_assert_id"


@pytest.mark.parametrize("name", ["post/top", "a/b"])
def test_the_separator_stays_out_of_export_identities(name):
    """DESIGN 4.8: the separator is admitted in node and assert ids and
    never in output, group, or param names."""
    doc = _doc()
    doc["outputs"][name] = "slab"
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_output_name"
    doc = _doc()
    doc["params"][0]["id"] = name
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_param_id"


def test_a_knob_entry_carries_the_resolved_value():
    """DESIGN 4.1: a knob's entry holds the value the build resolved,
    and the schema and the report read it there."""
    sc = _loads(_doc())
    [entry] = params_mod.build_schema(sc)
    assert entry == {
        "param_id": "height", "label": "Height", "bound": "build",
        "kind": "number", "value": 2.0, "min": 1.0, "max": 3.0,
    }
    assert params_mod.range_entries(sc) == [
        {"param": "height", "ok": True, "value": 2.0, "min": 1.0, "max": 3.0,
         "site": "yard/yard.star:2"}
    ]


def test_a_knob_outside_its_range_names_where_its_value_is_written():
    """DESIGN 6: a failing knob entry carries the fix, and the fix names
    the recipe line that declares the knob."""
    doc = _doc()
    doc["params"][0]["value"] = 4.0
    [entry] = params_mod.range_entries(_loads(doc))
    assert entry["ok"] is False
    assert "yard/yard.star:2" in entry["fix"]


def test_a_knob_without_a_value_is_refused():
    """DESIGN 4.1: the entry holds the resolved value."""
    doc = _doc()
    del doc["params"][0]["value"]
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_param_decl"


def test_a_knob_that_reaches_the_graph_through_the_recipe_is_the_safe_tier():
    """DESIGN 7.2: a `build`-bound knob is always `evaluation`."""
    assert tiers.classify(_loads(_doc())) == {"height": {"tier": "evaluation"}}


def test_a_build_bound_knob_cannot_be_patched_into_the_ir():
    """DESIGN 7.2: a knob is `direct` only when the build recorded that
    its value reached one node param whole; any other moves through a
    rebuild."""
    sc = _loads(_doc())
    with pytest.raises(MoraineError) as e:
        params_mod.apply_overrides(sc, {"height": 2.5})
    assert e.value.code == "param_build_bound"
    assert e.value.detail == {"param": "height"}


def _direct() -> dict:
    doc = _doc()
    doc["params"] = [
        {"id": "size", "value": [2.0, 0.1, 2.0], "node": "slab",
         "param": "size", "min": 0.05, "max": 4.0},
    ]
    return doc


def test_a_direct_knob_names_its_node_param_and_patches_both():
    """DESIGN 7.2: `direct` records that patching the param equals a
    rebuild, so a patch moves the entry and the node together."""
    sc = _loads(_direct())
    [entry] = params_mod.build_schema(sc)
    assert (entry["bound"], entry["node_id"], entry["param"]) == (
        "direct", "slab", "size",
    )
    params_mod.apply_overrides(sc, {"size": [3.0, 0.1, 3.0]})
    assert sc.nodes["slab"].params["size"] == [3.0, 0.1, 3.0]
    assert sc.params[0]["value"] == [3.0, 0.1, 3.0]
    assert asset_mod.dumps(asset_mod.loads(asset_mod.dumps(sc))) == asset_mod.dumps(sc)


def test_a_direct_knob_whose_value_left_its_node_is_refused():
    """DESIGN 7.2: the record says the value reached the param whole and
    unmodified, so the two agree."""
    doc = _direct()
    doc["nodes"][1]["params"]["size"] = [2.0, 0.2, 2.0]
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "param_value_mismatch"


@pytest.mark.parametrize("half", ["node", "param"])
def test_a_direct_record_names_both_halves(half):
    """DESIGN 4.1: `node` and `param` name one node param together."""
    doc = _direct()
    del doc["params"][0][half]
    with pytest.raises(AssetError) as e:
        _loads(doc)
    assert e.value.code == "bad_param_decl"
