"""Parameter interaction tiers (DESIGN 7.2): the graph-derived
classification of public knobs — seed / factor / evaluation — and its
conservative default."""

import json

from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.engine import tiers
from support.ir import with_knob_values
from support.engine import Evaluation


def _asset(nodes, params, outputs):
    return asset_mod.loads(
        json.dumps(
            with_knob_values({
                "build_version": "fixture",
                "nodes": nodes,
                "params": params,
                "outputs": outputs,
            })
        )
    )


_BOX = {"id": "bx", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
_PAINT = {
    "id": "mt",
    "op": "material",
    "inputs": ["bx"],
    "params": {"name": "paint", "roughness": 0.5},
}


def test_seed_param_classifies_as_seed():
    """A knob bound to an op's SEED_PARAMS is the seed tier (DESIGN 7.2:
    an unordered stochastic stream selector — a reroll control, never a
    slider)."""
    sc = _asset(
        [
            _BOX,
            {"id": "gr", "op": "grid", "params": {}},
            {
                "id": "sc",
                "op": "scatter",
                "inputs": ["gr", "bx"],
                "params": {"count": 4, "seed": 1},
            },
        ],
        [{"id": "spread", "node": "sc", "param": "seed", "min": 0, "max": 9999}],
        {"main": "sc"},
    )
    assert tiers.classify(sc) == {"spread": {"tier": "seed"}}


def test_material_factor_param_classifies_as_factor():
    """A knob on a material factor uniform with an asset-unique name is
    the factor tier and names its uniform (DESIGN 7.2: the client may
    drive the named material's channel live, with no evaluation)."""
    sc = _asset(
        [_BOX, _PAINT],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0}],
        {"main": "mt"},
    )
    assert tiers.classify(sc) == {
        "rough": {
            "tier": "factor",
            "uniform": {"material": "paint", "channel": "roughness"},
        }
    }


def test_structural_and_geometry_params_classify_as_evaluation():
    """Anything outside the positive seed/factor metadata is the
    evaluation tier (DESIGN 7.2: every doubt lands in the safe tier) —
    a material's structural ``name`` and a geometry param alike."""
    sc = _asset(
        [_BOX, _PAINT],
        [
            {"id": "ident", "node": "mt", "param": "name"},
            {"id": "size", "node": "bx", "param": "size", "min": 0.1, "max": 5.0},
        ],
        {"main": "mt"},
    )
    assert tiers.classify(sc) == {
        "ident": {"tier": "evaluation"},
        "size": {"tier": "evaluation"},
    }


def test_declared_range_outside_channel_range_demotes():
    """DESIGN 7.2: factor requires the declaration's [min, max] inside
    the channel's valid range (UNIFORM_RANGES) — the factor tier skips
    evaluation, so the declared range is the only validation a scrub
    gets, and a wider range would let an unevaluated, op-invalid value
    poison the overlay and reach the asset file through accept."""
    sc = _asset(
        [_BOX, _PAINT],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 2.0}],
        {"main": "mt"},
    )
    assert tiers.classify(sc)["rough"] == {"tier": "evaluation"}
    below = _asset(
        [_BOX, _PAINT],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": -0.5, "max": 1.0}],
        {"main": "mt"},
    )
    assert tiers.classify(below)["rough"] == {"tier": "evaluation"}


def test_unbounded_channel_range_keeps_factor():
    """DESIGN 7.2: normal_scale's channel range is [0, inf) — any declared
    upper bound is inside it, so a wide normal_scale knob stays factor."""
    paint = dict(_PAINT, params={"name": "paint", "normal_scale": 1.0})
    sc = _asset(
        [_BOX, paint],
        [{"id": "ns", "node": "mt", "param": "normal_scale", "min": 0.0, "max": 8.0}],
        {"main": "mt"},
    )
    assert tiers.classify(sc)["ns"] == {
        "tier": "factor",
        "uniform": {"material": "paint", "channel": "normal_scale"},
    }


def test_duplicate_material_name_demotes_to_evaluation():
    """A factor knob demotes when its material name is not unique among
    the asset's authored identities (DESIGN 7.2: the client applies the
    uniform by name, so a shared name would repaint a sibling output a
    real evaluation would not touch)."""
    other_box = {"id": "bx2", "op": "box", "params": {"size": [2.0, 1.0, 1.0]}}
    twin = {
        "id": "mt2",
        "op": "material",
        "inputs": ["bx2"],
        "params": {"name": "paint", "roughness": 0.8},
    }
    sc = _asset(
        [_BOX, _PAINT, other_box, twin],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0}],
        {"a": "mt", "b": "mt2"},
    )
    assert tiers.classify(sc)["rough"] == {"tier": "evaluation"}


def test_asset_ref_rename_collision_demotes_to_evaluation():
    """An asset_ref's material_rename targets join the authored-identity
    set (DESIGN 7.2): a rename onto a material node's name collides the
    same way a second material node would."""
    ref = {
        "id": "rf",
        "op": "asset_ref",
        "params": {
            "asset": "subassets/part.mora",
            "output": "main",
            "material_rename": {"steel": "paint"},
        },
    }
    sc = _asset(
        [_BOX, _PAINT, ref],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0}],
        {"a": "mt", "b": "rf"},
    )
    assert tiers.classify(sc)["rough"] == {"tier": "evaluation"}


_KIT = {
    "build_version": "fixture",
    "nodes": [
        {"id": "kb", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
        {
            "id": "km",
            "op": "material",
            "inputs": ["kb"],
            "params": {"name": "paint", "roughness": 0.6},
        },
    ],
    "outputs": {"main": "km"},
}


def _kit_project(tmp_path, ref_params: dict):
    """A root binding ``paint`` with a roughness knob beside a reference
    to a kit whose output binds ``paint`` with identical content; returns
    the loaded root and the reference-borne names of its evaluation."""
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    (proj / "subassets" / "part.mora").write_text(json.dumps(_KIT), encoding="utf-8")
    paint = dict(_PAINT, params={"name": "paint", "roughness": 0.6})
    ref = {"id": "rf", "op": "asset_ref", "params": ref_params}
    (proj / "proj.mora").write_text(
        json.dumps(
            with_knob_values({
                "build_version": "fixture",
                "nodes": [_BOX, paint, ref],
                "params": [
                    {"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0}
                ],
                "outputs": {"a": "mt", "b": "rf"},
            })
        ),
        encoding="utf-8",
    )
    sc = asset_mod.load(proj / "proj.mora")
    ev = Evaluation(sc)
    names = tiers.reference_material_names(ev.results(), ev.material_origins())
    return sc, names


def test_root_material_shared_with_kit_demotes_to_evaluation(tmp_path):
    """A factor knob demotes when an un-renamed reference carries the
    knob's material name into the root (DESIGN 7.2): the client applies
    the uniform by name across every loaded output, so the scrub would
    repaint the kit's part, which a real evaluation of the edit would
    not touch — and with identical content on both sides nothing else
    catches the pair (DESIGN 9.1's gate fails only on two contents)."""
    sc, names = _kit_project(
        tmp_path, {"asset": "subassets/part.mora", "output": "main"}
    )
    assert names == {"paint"}
    assert tiers.classify(sc, reference_names=names)["rough"] == {"tier": "evaluation"}
    # The graph walk alone cannot see a referenced file's names: the
    # check is graph-only without them.
    assert tiers.classify(sc)["rough"]["tier"] == "factor"


def test_renamed_kit_material_keeps_factor(tmp_path):
    """A reference whose ``material_rename`` gives the kit's material its
    own name carries nothing un-renamed into the root (DESIGN 7.2 /
    4.6: the consumer named it), so the root knob stays factor."""
    sc, names = _kit_project(
        tmp_path,
        {
            "asset": "subassets/part.mora",
            "output": "main",
            "material_rename": {"paint": "kit_paint"},
        },
    )
    assert names == frozenset()
    assert tiers.classify(sc, reference_names=names)["rough"] == {
        "tier": "factor",
        "uniform": {"material": "paint", "channel": "roughness"},
    }


def test_dropping_downstream_consumer_keeps_factor():
    """A downstream MATERIAL_POLICY "dropped" consumer does not demote
    (DESIGN 7.2: a dropping consumer merely makes the knob inert on that
    path, which a by-name client update renders correctly by updating
    nothing)."""
    other_box = {"id": "bx2", "op": "box", "params": {"size": [2.0, 1.0, 1.0]}}
    cut = {"id": "cut", "op": "boolean", "inputs": ["mt", "bx2"], "params": {}}
    sc = _asset(
        [_BOX, _PAINT, other_box, cut],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0}],
        {"main": "cut"},
    )
    assert tiers.classify(sc)["rough"]["tier"] == "factor"


def test_unknown_op_classifies_as_evaluation():
    """Classification never raises on an asset evaluation would reject
    (DESIGN 7.2): a knob bound to an unknown op lands in the safe tier."""
    sc = _asset(
        [{"id": "ty", "op": "torus", "params": {"radius": 1.0}}],
        [{"id": "r", "value": 1.0, "node": "ty", "param": "radius", "min": 0.1, "max": 2.0}],
        {"main": "ty"},
    )
    assert tiers.classify(sc) == {"r": {"tier": "evaluation"}}


def test_unknown_op_anywhere_demotes_every_knob():
    """DESIGN 7.2: an asset holding any op the registry cannot resolve
    classifies every knob `evaluation` — the deciding facts (the
    authored-identity set, the consumer edges) cannot be traced past it,
    so a would-be factor knob demotes even when the unknown op merely
    consumes its material downstream."""
    unknown = {"id": "zz", "op": "future_bake", "inputs": ["mt"], "params": {}}
    sc = _asset(
        [_BOX, _PAINT, unknown],
        [{"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0}],
        {"main": "zz"},
    )
    assert tiers.classify(sc) == {"rough": {"tier": "evaluation"}}


def test_build_schema_carries_tier_and_uniform():
    """build_schema merges the tier map into each entry — ``tier`` on
    every row, ``uniform`` on factor rows only (DESIGN 7.2 param_schema
    shape)."""
    sc = _asset(
        [_BOX, _PAINT],
        [
            {"id": "rough", "node": "mt", "param": "roughness", "min": 0.0, "max": 1.0},
            {"id": "size", "node": "bx", "param": "size", "min": 0.1, "max": 5.0},
        ],
        {"main": "mt"},
    )
    rows = {e["param_id"]: e for e in params_mod.build_schema(sc, tiers.classify(sc))}
    assert rows["rough"]["tier"] == "factor"
    assert rows["rough"]["uniform"] == {"material": "paint", "channel": "roughness"}
    assert rows["size"]["tier"] == "evaluation"
    assert "uniform" not in rows["size"]
    # Tierless build (no map) stays valid for the second schema consumer.
    assert "tier" not in params_mod.build_schema(sc)[0]
