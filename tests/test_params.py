"""param() declarations (DESIGN 7.2): asset-level parsing and
canonical dump, schema building, and preview-overlay application."""

import pytest

from moraine3d.core import params as params_mod
from moraine3d.core import asset as asset_mod
from moraine3d.errors import MoraineError, AssetError

_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx", "op": "box", "params": {"size": [1.0, 2.0, 1.0]}},
{"id": "nz", "op": "noise_texture", "params": {"seed": 3, "size": 64}}
],
"params": [
{"id": "width", "value": [1.0, 2.0, 1.0], "node": "bx", "param": "size", "label": "Box size", "min": 0.1, "max": 5.0, "step": 0.1},
{"id": "seed", "value": 3, "node": "nz", "param": "seed", "min": 0, "max": 99, "step": 1}
],
"outputs": {"main": "bx"}
}
"""


def _mutate(lines_replace: dict[str, str]) -> str:
    text = _ASSET
    for old, new in lines_replace.items():
        assert old in text
        text = text.replace(old, new)
    return text


def test_params_parse_and_roundtrip():
    sc = asset_mod.loads(_ASSET)
    assert len(sc.params) == 2
    text = asset_mod.dumps(sc)
    # One declaration per line, fixed key order, before outputs.
    assert '{"id": "width", "value": [1.0, 2.0, 1.0], "node": "bx", "param": "size", "label": "Box size", "min": 0.1, "max": 5.0, "step": 0.1}' in text
    assert text.index('"params": [') < text.index('"outputs": ')
    sc2 = asset_mod.loads(text)
    assert sc2.params == sc.params


def test_params_asset_without_block():
    plain = _mutate({'"params": [\n{"id": "width", "value": [1.0, 2.0, 1.0], "node": "bx", "param": "size", "label": "Box size", "min": 0.1, "max": 5.0, "step": 0.1},\n{"id": "seed", "value": 3, "node": "nz", "param": "seed", "min": 0, "max": 99, "step": 1}\n],\n': ""})
    sc = asset_mod.loads(plain)
    assert sc.params is None
    assert '"params": [' not in asset_mod.dumps(sc).replace('"params": {', "")


@pytest.mark.parametrize(
    "old,new,code",
    [
        ('"id": "width"', '"id": "9bad"', "bad_param_id"),
        ('"id": "seed"', '"id": "width"', "duplicate_param_id"),
        ('"id": "seed"', '"id": "time"', "reserved_param_id"),
        ('"node": "nz"', '"node": "nope"', "bad_param_node"),
        ('"param": "seed"', '"param": "gain"', "bad_param_ref"),
        ('"label": "Box size", ', '"labell": "x", ', "bad_param_decl"),
        ('"min": 0.1, "max": 5.0, "step": 0.1', '"step": 0.1', "bad_param_decl"),
        ('"min": 0.1, "max": 5.0', '"min": 6.0, "max": 5.0', "bad_param_decl"),
        # An explicit null is a present, non-numeric range key - refused
        # at the door, never a TypeError from the range comparisons.
        ('"min": 0.1,', '"min": null,', "bad_param_decl"),
        ('"max": 5.0,', '"max": null,', "bad_param_decl"),
        ('"step": 0.1}', '"step": null}', "bad_param_decl"),
    ],
)
def test_params_validation_errors(old, new, code):
    with pytest.raises(AssetError) as e:
        asset_mod.loads(_mutate({old: new}))
    assert e.value.code == code


def test_build_schema():
    sc = asset_mod.loads(_ASSET)
    schema = params_mod.build_schema(sc)
    assert [s["param_id"] for s in schema] == ["width", "seed"]
    w = schema[0]
    assert w == {
        "param_id": "width",
        "label": "Box size",
        "bound": "direct",
        "node_id": "bx",
        "param": "size",
        "kind": "vector",
        "value": [1.0, 2.0, 1.0],
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
    }
    assert schema[1]["label"] == "seed"  # label defaults to the id
    assert schema[1]["kind"] == "number"


def test_apply_overrides():
    sc = asset_mod.loads(_ASSET)
    params_mod.apply_overrides(sc, {"seed": 7, "width": [2.0, 2.0, 2.0]})
    assert sc.nodes["nz"].params["seed"] == 7
    assert sc.nodes["bx"].params["size"] == [2.0, 2.0, 2.0]
    # Schema reflects the overlay, and says whose value it shows.
    (w, seed) = params_mod.build_schema(sc)
    assert seed["value"] == 7 and seed["from"] == "override" and seed["default"] == 3
    assert w["default"] == [1.0, 2.0, 1.0]


def test_a_patch_of_none_unsets_the_knob():
    """DESIGN 7.2: `null` unsets the knob in the overlay: the preview
    shows the recipe's default, as the state without the line, and
    the entry names no other source. A knob holding its default
    already is left as it is."""
    sc = asset_mod.loads(_ASSET)
    params_mod.apply_overrides(sc, {"seed": 7})
    params_mod.apply_overrides(sc, {"seed": None, "width": None})
    assert sc.nodes["nz"].params["seed"] == 3
    (w, seed) = params_mod.build_schema(sc)
    assert seed["value"] == 3 and "from" not in seed and "default" not in seed
    assert w["value"] == [1.0, 2.0, 1.0] and "from" not in w


def test_apply_overrides_unknown_param():
    sc = asset_mod.loads(_ASSET)
    with pytest.raises(MoraineError) as e:
        params_mod.apply_overrides(sc, {"nope": 1})
    assert e.value.code == "param_unknown"


@pytest.mark.parametrize("bad", ["a string", [1.0, 2.0], True])
def test_apply_overrides_type_mismatch(bad):
    sc = asset_mod.loads(_ASSET)
    with pytest.raises(MoraineError) as e:
        params_mod.apply_overrides(sc, {"width": bad})
    assert e.value.code == "param_type_mismatch"


def test_schema_defaults_integer_step():
    sc = asset_mod.loads(_mutate({'"min": 0, "max": 99, "step": 1': '"min": 0, "max": 99'}))
    schema = params_mod.build_schema(sc)
    assert schema[1]["step"] == 1  # int-valued param without declared step


def test_apply_overrides_coerces_integral_float():
    sc = asset_mod.loads(_ASSET)
    params_mod.apply_overrides(sc, {"seed": 7.0})
    assert sc.nodes["nz"].params["seed"] == 7
    assert isinstance(sc.nodes["nz"].params["seed"], int)


def test_apply_overrides_out_of_range():
    """An override must sit inside the declared [min, max]: the range is
    the only validity knowledge an opaque reference exposes, so it is
    part of the declaration contract (DESIGN 4.6)."""
    sc = asset_mod.loads(_ASSET)
    with pytest.raises(MoraineError) as e:
        params_mod.apply_overrides(sc, {"seed": 200})
    assert e.value.code == "param_out_of_range"
    assert e.value.detail == {"param": "seed", "min": 0, "max": 99}


def test_apply_overrides_vector_component_out_of_range():
    """Per-component, the loader's own range rule."""
    sc = asset_mod.loads(_ASSET)
    with pytest.raises(MoraineError) as e:
        params_mod.apply_overrides(sc, {"width": [1.0, 9.0, 1.0]})
    assert e.value.code == "param_out_of_range"


def test_apply_overrides_boundary_values_accepted():
    """Bounds are inclusive, matching the report's range rule."""
    sc = asset_mod.loads(_ASSET)
    params_mod.apply_overrides(sc, {"width": [5.0, 0.1, 5.0], "seed": 99})
    assert sc.nodes["bx"].params["size"] == [5.0, 0.1, 5.0]
    assert sc.nodes["nz"].params["seed"] == 99


def test_duplicate_binding_rejected():
    text = _mutate({'"node": "nz", "param": "seed"': '"node": "bx", "param": "size"'})
    with pytest.raises(AssetError) as e:
        asset_mod.loads(text)
    assert e.value.code == "duplicate_param_binding"


def test_value_outside_declared_range_is_reported_not_refused():
    """DESIGN 6: the loader judges a declaration's form; a value outside
    the declared range is an entry of the validate report."""
    text = _mutate({'"min": 0, "max": 99': '"min": 10, "max": 99'})  # seed is 3
    sc = asset_mod.loads(text)
    assert [(e["param"], e["ok"]) for e in params_mod.range_entries(sc)] == [
        ("width", True), ("seed", False),
    ]


_INT_VECTOR_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "nz", "op": "noise_texture", "params": {"seed": 3, "size": [64, 64]}}
],
"params": [
{"id": "texsize", "value": [64, 64], "node": "nz", "param": "size", "min": 4, "max": 2048}
],
"outputs": {"main": "nz"}
}
"""


def test_apply_overrides_coerces_int_vector_components():
    """JSON clients send every number as a float — per-component for
    vectors too: an int-strict [w, h] size must not reject every
    override a JSON client can express (a control that can never
    succeed)."""
    sc = asset_mod.loads(_INT_VECTOR_ASSET)
    params_mod.apply_overrides(sc, {"texsize": [128.0, 128.0]})
    assert sc.nodes["nz"].params["size"] == [128, 128]
    assert all(isinstance(c, int) for c in sc.nodes["nz"].params["size"])


def test_schema_defaults_integer_step_for_int_vectors():
    """Integer-valued params (scalar or all-int vector) without a
    declared step get step 1, so clients never generate fractional
    values an int-strict op rejects on every drag."""
    sc = asset_mod.loads(_INT_VECTOR_ASSET)
    schema = params_mod.build_schema(sc)
    assert schema[0]["step"] == 1


def test_range_keys_rejected_on_non_numeric_kinds():
    """Range keys apply to number/vector params only: on a text/boolean
    declaration they would reach the override range check as a
    number-vs-non-number comparison — a raw TypeError instead of a
    structured error — so the loader rejects the declaration."""
    text = _mutate(
        {
            '"params": {"seed": 3, "size": 64}': '"params": {"seed": 3, "size": 64, "mode": "fbm"}',
            '{"id": "seed", "value": 3, "node": "nz", "param": "seed", "min": 0, "max": 99, "step": 1}': '{"id": "m", "value": "fbm", "node": "nz", "param": "mode", "min": 0}',
        }
    )
    with pytest.raises(AssetError) as e:
        asset_mod.loads(text)
    assert e.value.code == "bad_param_decl"


def test_apply_overrides_keeps_float_params_float():
    """The symmetric coercion: a float-valued param stays float when a
    JSON client sends an end-of-range integral as an integer —
    otherwise the int-flipped value would hand the knob the integer
    default step (a 0..1 slider degrades into a toggle) and churn
    cache keys on a value the op reads identically."""
    sc = asset_mod.loads(_ASSET)
    params_mod.apply_overrides(sc, {"width": [5, 1, 0.5]})
    assert sc.nodes["bx"].params["size"] == [5.0, 1.0, 0.5]
    assert all(isinstance(c, float) for c in sc.nodes["bx"].params["size"])
    schema = {e["param_id"]: e for e in params_mod.build_schema(sc)}
    assert schema["width"]["step"] == 0.1  # declared step, not the int default
