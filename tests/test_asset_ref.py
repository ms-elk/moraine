"""asset_ref: cross-file reuse, cache aliasing, cycle/escape/layout
rejection, per-use overrides and evaluation identity, cross-project
references, vendoring provenance (DESIGN 4.6)."""

import json

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core import sources as sources_mod
from moraine3d.errors import MoraineError, OpError, AssetError
from moraine3d.ops import asset_ref as asset_ref_mod
from support import flatten, projects
from support.engine import evaluate_asset, Evaluation
from support.links import link_dir as _link_dir

PART = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [0.5, 0.5, 0.5]}},
{"id": "tr01", "op": "transform", "inputs": ["bx01"], "params": {"translate": [3.0, 0.0, 0.0]}},
{"id": "ar01", "op": "array", "inputs": ["tr01"], "params": {"count": 3, "step_translate": [1.0, 0.0, 0.0]}}
],
"params": [
{"id": "size", "value": [0.5, 0.5, 0.5], "node": "bx01", "param": "size", "min": 0.1, "max": 2.0},
{"id": "count", "value": 3, "node": "ar01", "param": "count", "min": 1, "max": 10}
],
"outputs": {"solo": "bx01", "row": "ar01"}
}
"""

ROOT = """{
"build_version": "fixture",
"nodes": [
{"id": "rf01", "op": "asset_ref", "params": {"asset": "subassets/part.mora", "output": "row"}}
],
"outputs": {"main": "rf01"}
}
"""


def ref_asset(asset: str, output: str) -> str:
    return json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "rf01", "op": "asset_ref", "params": {"asset": asset, "output": output}}
            ],
            "outputs": {"main": "rf01"},
        }
    )


def multi_ref_asset(refs: dict[str, dict]) -> str:
    """A root asset of asset_ref nodes: output name -> asset_ref params."""
    return json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": f"rf{i:02d}", "op": "asset_ref", "params": params}
                for i, params in enumerate(refs.values(), 1)
            ],
            "outputs": {
                name: f"rf{i:02d}" for i, name in enumerate(refs, 1)
            },
        }
    )


ALPHA = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}
],
"params": [
{"id": "width", "value": [1.0, 1.0, 1.0], "node": "bx01", "param": "size", "min": 0.1, "max": 4.0}
],
"outputs": {"main": "bx01"}
}
"""


def write_workspace(tmp_path, beta_params: dict, alpha_text: str = ALPHA):
    """Two-project workspace; beta's root is one asset_ref with the given
    params. A sibling's root outputs plus its declarations are the whole
    public interface a cross-project ref can touch (DESIGN 4.6)."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True, exist_ok=True)
    (ws / "alpha").mkdir(exist_ok=True)
    (ws / "alpha" / "alpha.mora").write_text(alpha_text, encoding="utf-8")
    (ws / "beta").mkdir(exist_ok=True)
    (ws / "beta" / "beta.mora").write_text(
        multi_ref_asset({"main": beta_params}), encoding="utf-8"
    )
    return ws


def write_project(tmp_path, root_text=ROOT, part_text=PART):
    """The fixtures as IR files, the form the evaluator reads when an
    IR is handed over by path."""
    (tmp_path / "subassets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "root.mora").write_text(root_text, encoding="utf-8")
    (tmp_path / "subassets" / "part.mora").write_text(part_text, encoding="utf-8")
    return tmp_path / "root.mora"


def build_fixtures(project):
    """Translate every fixture the project holds into a recipe (DESIGN
    10) and build the project. Returns the root's IR, written under
    ``build/`` with the closure and its variants (DESIGN 4.6)."""
    return projects.built(flatten.write_project(project, project)).target.ir_path


def build_project(tmp_path, root_text=ROOT, part_text=PART):
    """``write_project``, built: an overridden reference opens the IR
    its overrides define, which only a build writes (DESIGN 4.6)."""
    write_project(tmp_path, root_text, part_text)
    return build_fixtures(tmp_path)


def build_workspace(ws, project: str):
    """Translate every project of a fixture workspace and build one."""
    for child in sorted(p for p in ws.iterdir() if p.is_dir() and list(p.glob("*.mora"))):
        flatten.write_project(child, child)
    return projects.built(ws / project).target.ir_path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def test_ref_is_identity_over_referenced_output(cache):
    root = write_project(cache / "proj")
    r = evaluate_asset(asset_mod.load(root))
    sub = evaluate_asset(asset_mod.load(cache / "proj" / "subassets" / "part.mora"))
    # Same cache key: location params never enter the key (DESIGN 4.6),
    # and instancing passes through untouched.
    assert r["main"]["key"] == sub["row"]["key"]
    assert r["main"]["stats"]["instances"] == 3
    assert r["main"]["stats"] == sub["row"]["stats"]


def test_subasset_stays_individually_evaluable(cache):
    root = write_project(cache / "proj")
    sub_path = sources_mod.resolve(root.parent / "subassets" / "part.mora")
    r = evaluate_asset(asset_mod.load(sub_path))
    assert set(r) == {"solo", "row"}


def test_ref_dirty_propagation_across_files(cache):
    root = write_project(cache / "proj")
    r1 = evaluate_asset(asset_mod.load(root))
    # Unchanged sub-asset: second evaluation is a pure cache hit.
    assert evaluate_asset(asset_mod.load(root))["main"]["key"] == r1["main"]["key"]
    # Editing the sub-asset re-keys the composite output.
    doc = json.loads(PART)
    doc["nodes"][0]["params"]["size"] = [0.7, 0.7, 0.7]
    doc["params"][0]["value"] = [0.7, 0.7, 0.7]
    (cache / "proj" / "subassets" / "part.mora").write_text(json.dumps(doc), encoding="utf-8")
    r2 = evaluate_asset(asset_mod.load(root))
    assert r2["main"]["key"] != r1["main"]["key"]


def test_ref_chain_from_subasset_resolves_project_relative(cache):
    proj = cache / "proj"
    mid = """{
"build_version": "fixture",
"nodes": [
{"id": "rf01", "op": "asset_ref", "params": {"asset": "subassets/part.mora", "output": "solo"}},
{"id": "tr01", "op": "transform", "inputs": ["rf01"], "params": {"translate": [0.0, 1.0, 0.0]}}
],
"outputs": {"lifted": "tr01"}
}
"""
    root = write_project(proj, root_text=ref_asset("subassets/mid.mora", "lifted"))
    (proj / "subassets" / "mid.mora").write_text(mid, encoding="utf-8")
    r = evaluate_asset(asset_mod.load(proj / "root.mora"))
    assert r["main"]["stats"]["bounds"]["min"][1] == 0.75


def test_ref_cycle_rejected_with_chain(cache):
    proj = cache / "proj"
    root = write_project(
        proj,
        root_text=ref_asset("subassets/part.mora", "main"),
        part_text=ref_asset("subassets/other.mora", "main"),
    )
    (proj / "subassets" / "other.mora").write_text(
        ref_asset("subassets/part.mora", "main"), encoding="utf-8"
    )
    with pytest.raises(AssetError) as exc:
        evaluate_asset(asset_mod.load(root))
    assert exc.value.code == "ref_cycle"
    assert exc.value.detail["chain"] == [
        "root.mora",
        "subassets/part.mora",
        "subassets/other.mora",
        "subassets/part.mora",
    ]


def test_ref_escape_rejected(cache):
    proj = cache / "proj"
    (cache / "outside.mora").write_text(PART, encoding="utf-8")
    for path in ("../outside.mora", "/abs/outside.mora", "C:/abs/outside.mora"):
        root = write_project(proj, root_text=ref_asset(path, "row"))
        with pytest.raises(AssetError) as exc:
            evaluate_asset(asset_mod.load(root))
        assert exc.value.code == "ref_escape", path


def test_ref_layout_enforced(cache):
    """References must be exactly 'subassets/<name>.mora' (DESIGN 4.2):
    no top-level assets, no nesting, no traversal that lands in-project."""
    proj = cache / "proj"
    write_project(proj)
    (proj / "toplevel.mora").write_text(PART, encoding="utf-8")
    (proj / "subassets" / "nested").mkdir(parents=True, exist_ok=True)
    (proj / "subassets" / "nested" / "deep.mora").write_text(PART, encoding="utf-8")
    for path in (
        "toplevel.mora",
        "root.mora",
        "subassets/nested/deep.mora",
        "subassets/../toplevel.mora",
    ):
        root = write_project(proj, root_text=ref_asset(path, "row"))
        with pytest.raises(AssetError) as exc:
            evaluate_asset(asset_mod.load(root))
        assert exc.value.code == "ref_layout", path


def test_ref_bad_params_rejected(cache):
    proj = cache / "proj"
    cases = [
        ({"asset": "subassets\\part.mora", "output": "row"}, "bad_param"),
        ({"asset": "subassets/part.json", "output": "row"}, "bad_param"),
        ({"output": "row"}, "bad_param"),
        ({"asset": "subassets/part.mora"}, "bad_param"),
        # Exactly one location form; 'project' is a name, never a path;
        # 'overrides' must be an object (DESIGN 4.6).
        ({"asset": "subassets/part.mora", "project": "other", "output": "row"}, "bad_param"),
        ({"project": "other/sub", "output": "row"}, "bad_param"),
        ({"project": "..\\other", "output": "row"}, "bad_param"),
        ({"asset": "subassets/part.mora", "output": "row", "overrides": ["size"]}, "bad_param"),
        ({"asset": "subassets/nope.mora", "output": "row"}, "ref_not_found"),
        ({"asset": "subassets/part.mora", "output": "nope"}, "ref_unknown_output"),
    ]
    for params, code in cases:
        doc = json.loads(ROOT)
        doc["nodes"][0]["params"] = params
        root = write_project(proj, root_text=json.dumps(doc))
        with pytest.raises(MoraineError) as exc:
            evaluate_asset(asset_mod.load(root))
        assert exc.value.code == code, params


def test_ref_takes_no_inputs(cache):
    proj = cache / "proj"
    doc = json.loads(ROOT)
    doc["nodes"].insert(0, {"id": "bx01", "op": "box", "params": {}})
    doc["nodes"][1]["inputs"] = ["bx01"]
    root = write_project(proj, root_text=json.dumps(doc))
    with pytest.raises(MoraineError) as exc:
        evaluate_asset(asset_mod.load(root))
    assert exc.value.code == "bad_inputs"


def test_ref_requires_file_backed_asset(cache):
    sc = asset_mod.loads(ROOT)  # in-memory: no path to resolve against
    with pytest.raises(AssetError) as exc:
        evaluate_asset(sc)
    assert exc.value.code == "ref_no_project"


def test_evaluation_collects_asset_files_for_watch(cache):
    root = write_project(cache / "proj")
    ev = Evaluation(asset_mod.load(root))
    ev.results()
    assert ev.asset_files == {(cache / "proj" / "subassets" / "part.mora").resolve()}


def test_missing_ref_target_still_joins_watch_set(cache):
    """A referenced-but-missing sub-asset must be watched, so creating the
    file later triggers a hot reload without touching the root asset."""
    proj = cache / "proj"
    root = write_project(proj, root_text=ref_asset("subassets/new.mora", "row"))
    ev = Evaluation(asset_mod.load(root))
    with pytest.raises(AssetError) as exc:
        ev.results()
    assert exc.value.code == "ref_not_found"
    assert (proj / "subassets" / "new.mora").resolve() in ev.asset_files
    # Creating the file resolves the failure on the next evaluation.
    (proj / "subassets" / "new.mora").write_text(PART, encoding="utf-8")
    assert evaluate_asset(asset_mod.load(root))["main"]["stats"]["instances"] == 3


def test_provenance_roundtrip_and_ignored_by_evaluation(cache):
    proj = cache / "proj"
    root = write_project(proj)
    r1 = evaluate_asset(asset_mod.load(root))
    doc = json.loads(PART)
    doc["provenance"] = {"source": "examples/part/part.mora", "content_hash": "sha256:00"}
    sub = asset_mod.loads(json.dumps(doc))
    assert sub.provenance["content_hash"] == "sha256:00"
    text = asset_mod.dumps(sub)
    assert asset_mod.loads(text).provenance == sub.provenance
    (proj / "subassets" / "part.mora").write_text(text, encoding="utf-8")
    # Provenance is location metadata: the composite result key is unchanged.
    assert evaluate_asset(asset_mod.load(root))["main"]["key"] == r1["main"]["key"]


def test_bad_provenance_rejected():
    doc = json.loads(PART)
    doc["provenance"] = "not-an-object"
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_provenance"


# -- per-use overrides (DESIGN 4.6) ---------------------------------------


def test_override_token_shapes():
    """Empty overrides are the plain identity; values are canonically
    serialized but never numerically folded (int coercion depends on the
    referenced asset's declaration, unknowable at token time)."""
    assert asset_ref_mod.override_token({}) == ""
    assert asset_ref_mod.override_token({"b": 1, "a": 2}) == \
        asset_ref_mod.override_token({"a": 2, "b": 1})
    assert asset_ref_mod.override_token({"w": 2}) != asset_ref_mod.override_token({"w": 2.0})


def test_overrides_must_have_string_keys():
    with pytest.raises(OpError) as exc:
        asset_ref_mod.ref_params(
            {"asset": "subassets/p.mora", "output": "row", "overrides": {1: 2}}
        )
    assert exc.value.code == "bad_param" and exc.value.detail["param"] == "overrides"


def test_override_variants_are_separate_records(cache):
    """Differently-overridden uses of one file are distinct evaluations
    with distinct cache entries; an un-overridden sibling keeps aliasing
    the referenced output's record (DESIGN 4.6)."""
    proj = cache / "proj"
    root = build_project(
        proj,
        root_text=multi_ref_asset(
            {
                "big": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"size": [0.7, 0.7, 0.7]},
                },
                "plain": {"asset": "subassets/part.mora", "output": "row"},
            }
        ),
    )
    r = evaluate_asset(asset_mod.load(root))
    sub = evaluate_asset(asset_mod.load(proj / "build" / "subassets" / "part.mora"))
    assert r["big"]["key"] != r["plain"]["key"]
    assert r["plain"]["key"] == sub["row"]["key"]
    assert r["big"]["stats"]["bounds"]["min"][1] == -0.35
    assert r["plain"]["stats"]["bounds"]["min"][1] == -0.25


def test_overridden_ref_golden(cache):
    """Golden pin for an overridden reference: the override lands on the
    referenced asset's declared param before evaluation, so the result is
    exactly the referenced output at the overridden value."""
    proj = cache / "proj"
    root = build_project(
        proj,
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"size": [0.7, 0.7, 0.7], "count": 4},
                }
            }
        ),
    )
    r = evaluate_asset(asset_mod.load(root))
    s = r["main"]["stats"]
    assert s["instances"] == 4
    assert s["bounds"]["min"] == [2.65, -0.35, -0.35]  # the row starts at x = 3
    assert s["bounds"]["max"] == [6.35, 0.35, 0.35]


def test_override_reuse_is_warm_hit(cache):
    """An override variant is a first-class cache citizen: re-evaluating
    the same overridden reference is a pure warm pass (DESIGN 4.5)."""
    proj = cache / "proj"
    root = build_project(
        proj,
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"size": [0.7, 0.7, 0.7]},
                }
            }
        ),
    )
    evaluate_asset(asset_mod.load(root))
    ev = Evaluation(asset_mod.load(root))
    ev.records()
    p = ev.profile()
    assert p["misses"] == 0 and p["hits"] > 0


def test_override_change_and_revert(cache, monkeypatch):
    """Changing an override re-keys only the overridden use; reverting it
    returns to the original keys without recomputation; a clean cache
    regenerates the same keys (determinism, DESIGN 4.4/4.5)."""
    proj = cache / "proj"

    def root_text(size):
        return multi_ref_asset(
            {
                "big": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"size": [size, size, size]},
                },
                "plain": {"asset": "subassets/part.mora", "output": "solo"},
            }
        )

    root = build_project(proj, root_text=root_text(0.7))
    r1 = evaluate_asset(asset_mod.load(root))
    build_project(proj, root_text=root_text(0.9))
    r2 = evaluate_asset(asset_mod.load(root))
    assert r2["big"]["key"] != r1["big"]["key"]
    assert r2["plain"]["key"] == r1["plain"]["key"]  # unrelated sibling untouched
    build_project(proj, root_text=root_text(0.7))
    ev = Evaluation(asset_mod.load(root))
    r3 = {name: rec.key for name, rec in ev.records().items()}
    assert r3["big"] == r1["big"]["key"]
    assert ev.profile()["misses"] == 0  # revert is a pure warm pass
    # Clean cache: full regeneration reproduces identical keys.
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache / "cache2"))
    r4 = evaluate_asset(asset_mod.load(root))
    assert r4["big"]["key"] == r1["big"]["key"]


def test_empty_overrides_are_plain_identity(cache):
    """`"overrides": {}` and no overrides key are one identity: both alias
    the referenced output's record outright (DESIGN 4.6)."""
    proj = cache / "proj"
    root = write_project(
        proj,
        root_text=multi_ref_asset(
            {
                "a": {"asset": "subassets/part.mora", "output": "row"},
                "b": {"asset": "subassets/part.mora", "output": "row", "overrides": {}},
            }
        ),
    )
    ev = Evaluation(asset_mod.load(root))
    recs = ev.records()
    assert recs["a"] is recs["b"]


def test_override_errors_reuse_declaration_contract(cache):
    """Override validation IS the declaration contract (one interface,
    two consumers): unknown id -> param_unknown naming the declared set,
    kind mismatch -> param_type_mismatch, outside the declared min/max
    -> param_out_of_range; all enriched with the referencing node and
    asset."""
    proj = cache / "proj"
    cases = [
        ({"nope": 1.0}, "param_unknown"),
        ({"count": "three"}, "param_type_mismatch"),
        ({"count": 99}, "param_out_of_range"),
    ]
    for overrides, code in cases:
        with pytest.raises(MoraineError) as exc:
            build_project(
                proj,
                root_text=multi_ref_asset(
                    {
                        "main": {
                            "asset": "subassets/part.mora",
                            "output": "row",
                            "overrides": overrides,
                        }
                    }
                ),
            )
        assert exc.value.code == code, overrides
        assert exc.value.detail["node"] == "rf01"
        assert exc.value.detail["asset"] == "subassets/part.star"
        assert exc.value.detail["site"] == "root.star:1"


def test_chained_override_reaches_direct_referent_only(cache):
    """An override addresses the directly referenced asset's declarations
    only (DESIGN 4.6 opacity); forwarding a deeper knob awaits the
    re-export binding form (future work)."""
    proj = cache / "proj"
    mid = {
        "build_version": "fixture",
        "nodes": [
            {
                "id": "rf01",
                "op": "asset_ref",
                "params": {"asset": "subassets/part.mora", "output": "solo"},
            },
            {
                "id": "tr01",
                "op": "transform",
                "inputs": ["rf01"],
                "params": {"translate": [0.0, 1.0, 0.0]},
            },
        ],
        "params": [
            {"id": "lift", "value": [0.0, 1.0, 0.0], "node": "tr01", "param": "translate", "min": -5.0, "max": 5.0}
        ],
        "outputs": {"lifted": "tr01"},
    }
    write_project(
        proj,
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/mid.mora",
                    "output": "lifted",
                    # 'size' is part.mora's declaration, not mid.mora's.
                    "overrides": {"size": [0.7, 0.7, 0.7]},
                }
            }
        ),
    )
    (proj / "subassets" / "mid.mora").write_text(json.dumps(mid), encoding="utf-8")
    with pytest.raises(MoraineError) as exc:
        build_fixtures(proj)
    assert exc.value.code == "param_unknown"
    assert "lift" in exc.value.message  # the declared set is enumerated
    assert exc.value.detail["asset"] == "subassets/mid.star"


def test_inner_plain_ref_dedups_across_override_variants(cache):
    """Two override variants of one asset share the plain identity of an
    inner un-overridden reference: the inner target evaluates once, only
    the variants' own nodes fork (DESIGN 4.6 evaluation identity)."""
    proj = cache / "proj"
    mid = {
        "build_version": "fixture",
        "nodes": [
            {
                "id": "rf01",
                "op": "asset_ref",
                "params": {"asset": "subassets/part.mora", "output": "solo"},
            },
            {
                "id": "tr01",
                "op": "transform",
                "inputs": ["rf01"],
                "params": {"translate": [0.0, 1.0, 0.0]},
            },
        ],
        "params": [
            {"id": "lift", "value": [0.0, 1.0, 0.0], "node": "tr01", "param": "translate", "min": -5.0, "max": 5.0}
        ],
        "outputs": {"lifted": "tr01"},
    }
    root = write_project(
        proj,
        root_text=multi_ref_asset(
            {
                "lo": {
                    "asset": "subassets/mid.mora",
                    "output": "lifted",
                    "overrides": {"lift": [0.0, 1.0, 0.0]},
                },
                "hi": {
                    "asset": "subassets/mid.mora",
                    "output": "lifted",
                    "overrides": {"lift": [0.0, 2.0, 0.0]},
                },
            }
        ),
    )
    (proj / "subassets" / "mid.mora").write_text(json.dumps(mid), encoding="utf-8")
    ev = Evaluation(asset_mod.load(build_fixtures(proj)))
    r = ev.records()
    assert r["lo"].key != r["hi"].key
    p = ev.profile()
    # part's bx01 once + one transform per variant. A leaked per-variant
    # part identity would show up as an extra disk hit on bx01's key.
    assert (p["hits"], p["misses"]) == (0, 3)


def test_override_can_retarget_inner_ref(cache):
    """A declaration may bind an asset_ref's own text params (existing
    param_kind rules); the inner identity derives from the mutated
    instance, so a retargeted inner reference resolves to the overridden
    target (DESIGN 4.6)."""
    proj = cache / "proj"
    mid = {
        "build_version": "fixture",
        "nodes": [
            {
                "id": "rf01",
                "op": "asset_ref",
                "params": {"asset": "subassets/part.mora", "output": "solo"},
            }
        ],
        "params": [{"id": "src", "value": "solo", "node": "rf01", "param": "output"}],
        "outputs": {"picked": "rf01"},
    }
    root = write_project(
        proj,
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/mid.mora",
                    "output": "picked",
                    "overrides": {"src": "row"},
                }
            }
        ),
    )
    (proj / "subassets" / "mid.mora").write_text(json.dumps(mid), encoding="utf-8")
    r = evaluate_asset(asset_mod.load(build_fixtures(proj)))
    assert r["main"]["stats"]["instances"] == 3  # 'row', not the declared 'solo'


def test_drop_warnings_stay_producer_side(cache):
    """Advisory warnings name edit sites, and a node inside a referenced
    asset is one the consumer cannot edit (DESIGN 4.1): a sub-asset's
    attribute drop is the producer's own validate's finding and is never
    re-raised in a consumer — under an override identity or the plain
    one."""
    proj = cache / "proj"
    part = {
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {}},
            {"id": "u1", "op": "unwrap", "inputs": ["b1"], "params": {}},
            {"id": "b2", "op": "box", "params": {"size": [0.4, 2.0, 0.4]}},
            {
                "id": "d1",
                "op": "boolean",
                "inputs": ["u1", "b2"],
                "params": {"mode": "difference"},
            },
        ],
        "params": [{"id": "cut", "value": [0.4, 2.0, 0.4], "node": "b2", "param": "size", "min": 0.1, "max": 3.0}],
        "outputs": {"main": "d1"},
    }
    root = build_project(
        proj,
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/part.mora",
                    "output": "main",
                    "overrides": {"cut": [0.3, 2.0, 0.3]},
                }
            }
        ),
        part_text=json.dumps(part),
    )
    ev = Evaluation(asset_mod.load(root))
    ev.records()
    assert ev.attribute_drop_warnings() == []
    # The producer sees it in its own validate.
    ev = Evaluation(projects.load(proj / "subassets" / "part.star"))
    ev.records()
    (w,) = ev.attribute_drop_warnings()
    assert w["node"] == "d1" and w["dropped"] == ["TEXCOORD_0"] and "asset" not in w


def test_ops_used_covers_override_instances(cache):
    """ops_used resolves every record's asset instance through the
    (path, token) registry — an override instance's ops must appear."""
    root = build_project(
        cache / "proj",
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"size": [0.7, 0.7, 0.7]},
                }
            }
        ),
    )
    ev = Evaluation(asset_mod.load(root))
    ev.records()
    used = ev.ops_used()
    assert used["asset_ref"] == 2
    assert {"box", "array"} <= set(used)


def test_overridden_composite_reexports_byte_identical(cache):
    """The export determinism invariant survives overridden references:
    two exports of a composite with an overridden asset_ref are
    byte-identical (DESIGN 4.4, Day-One Invariant)."""
    from support.engine import export_asset

    proj = cache / "proj"
    doc = json.loads(
        multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"size": [0.7, 0.7, 0.7]},
                }
            }
        )
    )
    doc["checks"] = {"main": {"not_sunk": False}}  # centered row: waived
    root = build_project(proj, root_text=json.dumps(doc))
    out = proj / "exports"
    result = export_asset(asset_mod.load(root), profile="gltf", out_dir=out)
    assert result["ok"] is True, result
    first = (out / "root.glb").read_bytes()
    export_asset(asset_mod.load(root), profile="gltf", out_dir=out)
    assert (out / "root.glb").read_bytes() == first


# -- cross-project references (DESIGN 4.6) --------------------------------


def test_cross_project_ref_targets_sibling_root_output(cache):
    """A cross-project reference names a sibling by its discovery
    identity and one of its root asset's named outputs; the record
    aliases the sibling's own evaluation (location is not content)."""
    ws = write_workspace(cache, {"project": "alpha", "output": "main"})
    r = evaluate_asset(asset_mod.load(build_workspace(ws, "beta")))
    a = evaluate_asset(asset_mod.load(ws / "alpha" / "build" / "alpha.mora"))
    assert r["main"]["key"] == a["main"]["key"]
    assert r["main"]["stats"]["bounds"]["max"][0] == 0.5


def test_cross_project_ref_with_overrides(cache):
    """Overrides address the sibling root asset's declarations exactly
    like an in-project reference (one interface)."""
    ws = write_workspace(
        cache,
        {"project": "alpha", "output": "main", "overrides": {"width": [2.0, 1.0, 1.0]}},
    )
    r = evaluate_asset(asset_mod.load(build_workspace(ws, "beta")))
    assert r["main"]["stats"]["bounds"]["max"][0] == 1.0


def _without_recipes(ws):
    for recipe in ws.rglob("*.star"):
        recipe.unlink()


@pytest.mark.parametrize("overrides", [{}, {"width": [2.0, 1.0, 1.0]}])
def test_cross_project_ref_resolves_from_the_ir_alone(cache, overrides):
    """DESIGN 4.6: the build writes the sibling root's IR by file name
    beside `project`, so an IR handed over by path reaches the sibling
    with no recipe left to read."""
    ref = {"project": "alpha", "output": "main"}
    if overrides:
        ref["overrides"] = overrides
    ws = write_workspace(cache, ref)
    root = build_workspace(ws, "beta")
    built = json.loads(root.read_text(encoding="utf-8"))
    assert built["nodes"][0]["params"]["asset"] == "alpha.mora"
    _without_recipes(ws)
    r = evaluate_asset(asset_mod.load(root))
    assert r["main"]["stats"]["bounds"]["max"][0] == (1.0 if overrides else 0.5)


def test_cross_project_ref_the_build_did_not_resolve(cache):
    """DESIGN 4.6: a project reference that names no IR was never
    built, and the refusal names the build."""
    ws = write_workspace(cache, {"project": "alpha", "output": "main"})
    with pytest.raises(AssetError) as exc:
        evaluate_asset(asset_mod.load(ws / "beta" / "beta.mora"))
    assert exc.value.code == "ref_unresolved"
    assert exc.value.detail["node"] == "rf01"
    assert exc.value.detail["project"] == "alpha"
    assert "moraine build" in exc.value.message


@pytest.mark.parametrize("asset", ["subassets/part.mora", "../alpha.mora", "alpha.star", ".mora"])
def test_cross_project_ref_names_the_root_ir_by_file_name(cache, asset):
    """DESIGN 4.6: beside `project` the IR carries a bare file name,
    so the reference reaches the sibling's root IR and nothing else."""
    ws = write_workspace(cache, {"project": "alpha", "asset": asset, "output": "main"})
    with pytest.raises(MoraineError) as exc:
        evaluate_asset(asset_mod.load(ws / "beta" / "beta.mora"))
    assert exc.value.code == "bad_param"
    assert exc.value.detail["param"] == "asset"


def test_cross_project_unknown_project(cache):
    """The name is validated against the workspace roster (the path-free
    choke point) — the error enumerates the valid set."""
    ws = write_workspace(cache, {"project": "gamma", "output": "main"})
    with pytest.raises(MoraineError) as exc:
        build_workspace(ws, "beta")
    assert exc.value.code == "unknown_project"
    assert exc.value.detail["projects"] == ["alpha", "beta"]
    assert exc.value.detail["node"] == "rf01"


def test_cross_project_requires_workspace(cache):
    """The 'project' form only resolves inside a workspace; a standalone
    project is pointed at vendoring instead."""
    proj = cache / "solo"
    proj.mkdir()
    (proj / "solo.mora").write_text(
        multi_ref_asset({"main": {"project": "alpha", "output": "main"}}),
        encoding="utf-8",
    )
    flatten.write_project(proj, proj)
    with pytest.raises(MoraineError) as exc:
        projects.built(proj)
    assert exc.value.code == "ref_no_workspace"
    assert exc.value.detail["project"] == "alpha"
    assert "vendor" in exc.value.message


def test_cross_project_self_reference_is_cycle(cache):
    ws = write_workspace(
        cache,
        {"project": "beta", "output": "main"},
        alpha_text=multi_ref_asset({"main": {"project": "alpha", "output": "main"}}),
    )
    # alpha -> alpha via its own project name is a one-file cycle.
    with pytest.raises(MoraineError) as exc:
        build_workspace(ws, "alpha")
    assert exc.value.code == "ref_cycle"


def test_cross_project_cycle_chain_is_workspace_relative(cache):
    """A cross-project cycle reports the file chain workspace-relative,
    as every site in a workspace is (DESIGN 4.1)."""
    ws = write_workspace(
        cache,
        {"project": "alpha", "output": "main"},
        alpha_text=multi_ref_asset({"main": {"project": "beta", "output": "main"}}),
    )
    with pytest.raises(MoraineError) as exc:
        build_workspace(ws, "alpha")
    assert exc.value.code == "ref_cycle"
    assert exc.value.detail["chain"] == [
        "alpha/alpha.star",
        "beta/beta.star",
        "alpha/alpha.star",
    ]


def test_cross_project_target_joins_watch_set(cache):
    """DESIGN 4.6: `serve`'s watch closure extends across the
    workspace; the sibling's recipe is one of the files the build
    read."""
    ws = write_workspace(cache, {"project": "alpha", "output": "main"})
    build_workspace(ws, "beta")
    result = projects.built(ws / "beta")
    assert (ws / "alpha" / "alpha.star").resolve() in result.sources


def test_cross_project_ambiguous_sibling_root(cache):
    """A sibling with several top-level assets cannot be referenced — the
    'one root asset per project' invariant is what makes the interface
    addressable. The error speaks the ref context: no absolute paths, no
    CLI-only advice."""
    ws = write_workspace(cache, {"project": "alpha", "output": "main"})
    for child in ("alpha", "beta"):
        flatten.write_project(ws / child, ws / child)
    flatten.write_doc(json.loads(ALPHA), ws / "alpha", "extra")
    with pytest.raises(MoraineError) as exc:
        projects.built(ws / "beta")
    assert exc.value.code == "ambiguous_asset"
    assert exc.value.detail["node"] == "rf01"
    assert exc.value.detail["project"] == "alpha"
    assert str(ws) not in exc.value.message
    assert "subassets/" in exc.value.message


def test_cross_project_conflicted_root_diagnosed(cache):
    """A .moraine/ marker beside a top-level asset is the invalid
    layout (DESIGN 4.2) — the build fails with workspace_asset_conflict
    naming the stray root asset as the fix, never 'moraine init', and
    without leaking the absolute root path."""
    root = cache / "degen"
    (root / ".moraine").mkdir(parents=True)
    flatten.write_doc(json.loads(ALPHA), root, "stray")
    flatten.write_doc(
        json.loads(multi_ref_asset({"main": {"project": "other", "output": "main"}})),
        root / "child", "child",
    )
    with pytest.raises(MoraineError) as exc:
        projects.built(root / "child")
    assert exc.value.code == "workspace_asset_conflict"
    assert "child project directory" in exc.value.message
    assert "moraine init" not in exc.value.message
    assert str(root) not in exc.value.message


def test_cross_project_link_spelled_sibling_keeps_one_namespace(cache):
    """The asset registry and the record memo must share one resolved
    path namespace: a junction/symlink-spelled sibling must not strand
    its records where ops_used / attribute_drop_warnings cannot resolve
    them (the export manifest would silently omit the sibling's ops)."""
    ws = write_workspace(cache, {"project": "gamma", "output": "main"})
    try:
        _link_dir(ws / "alpha", ws / "gamma")
    except OSError as e:  # e.g. symlink-restricted environment
        pytest.skip(f"cannot create a directory link here: {e}")
    ev = Evaluation(asset_mod.load(build_workspace(ws, "beta")))
    ev.records()
    assert "box" in ev.ops_used()


def test_cross_project_link_spelled_sibling_under_overrides(cache):
    """DESIGN 4.6: a variant is named after the project its asset
    belongs to, so a link-spelled sibling's variant is the file the
    build wrote under the sibling's own name."""
    ws = write_workspace(
        cache,
        {"project": "gamma", "output": "main", "overrides": {"width": [2.0, 1.0, 1.0]}},
    )
    try:
        _link_dir(ws / "alpha", ws / "gamma")
    except OSError as e:  # e.g. symlink-restricted environment
        pytest.skip(f"cannot create a directory link here: {e}")
    r = evaluate_asset(asset_mod.load(build_workspace(ws, "beta")))
    assert r["main"]["stats"]["bounds"]["max"][0] == 1.0


def test_cross_project_external_link_refused(cache):
    """DESIGN 4.6 containment: a sibling spelled by a link whose target
    lies outside the workspace is refused (project_outside_workspace) —
    a reference must never reach files git-sync, the watch closure, and
    root-relative revision framing cannot see."""
    outside = cache / "outside"
    outside.mkdir()
    (outside / "gamma.mora").write_text(ALPHA, encoding="utf-8")
    ws = write_workspace(cache, {"project": "gamma", "output": "main"})
    try:
        _link_dir(outside, ws / "gamma")
    except OSError as e:  # e.g. symlink-restricted environment
        pytest.skip(f"cannot create a directory link here: {e}")
    flatten.write_project(outside, outside)
    with pytest.raises(MoraineError) as exc:
        build_workspace(ws, "beta")
    assert exc.value.code == "project_outside_workspace"
    assert exc.value.detail["node"] == "rf01"
    assert exc.value.detail["project"] == "gamma"
    assert str(outside) not in exc.value.message


@pytest.mark.parametrize("overrides", [{}, {"width": [2.0, 1.0, 1.0]}])
def test_cross_project_ref_through_a_link_below_the_project_is_refused(
    cache, overrides,
):
    """DESIGN 4.6: the evaluator holds containment over the IR it
    opens, a sibling's own or a variant beside it, so a link below the
    project directory that leads out of the workspace is refused."""
    import shutil

    linked = "alpha/build"

    ref = {"project": "alpha", "output": "main"}
    if overrides:
        ref["overrides"] = overrides
    ws = write_workspace(cache, ref)
    root = build_workspace(ws, "beta")
    outside = cache / "outside"
    shutil.move(ws / linked, outside)
    try:
        _link_dir(outside, ws / linked)
    except OSError as e:  # e.g. symlink-restricted environment
        pytest.skip(f"cannot create a directory link here: {e}")
    with pytest.raises(AssetError) as exc:
        evaluate_asset(asset_mod.load(root))
    assert exc.value.code == "project_outside_workspace"
    assert exc.value.detail["node"] == "rf01"
    assert str(outside) not in exc.value.message


@pytest.mark.parametrize("name", ["gamma", ".moraine", "exports", "Lib"])
def test_cross_project_ref_to_what_is_no_project(cache, name):
    """DESIGN 4.2, 4.6: a sibling is a directory directly under the
    workspace root that is neither hidden nor layout, so any other name
    is refused as no project, listing the ones built here."""
    ws = write_workspace(cache, {"project": "alpha", "output": "main"})
    build_workspace(ws, "beta")
    for layout in (".moraine", "exports", "lib"):
        (ws / layout / "build").mkdir(parents=True, exist_ok=True)
        (ws / layout / "build" / "alpha.mora").write_text(ALPHA, encoding="utf-8")
    doc = json.loads(multi_ref_asset(
        {"main": {"project": name, "asset": "alpha.mora", "output": "main"}}
    ))
    root = ws / "beta" / "build" / "beta.mora"
    root.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AssetError) as exc:
        evaluate_asset(asset_mod.load(root))
    assert exc.value.code == "unknown_project"
    assert exc.value.detail["projects"] == ["alpha", "beta"]
    assert exc.value.detail["node"] == "rf01"
    assert str(ws) not in exc.value.message


def test_a_variant_that_is_a_link_out_of_the_project_is_refused(cache):
    """DESIGN 4.6: an in-project reference resolves inside the project
    directory, the variant its overrides name included."""
    proj = cache / "proj"
    root = build_project(proj, root_text=json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "rf01", "op": "asset_ref", "params": {
            "asset": "subassets/part.mora", "output": "row",
            "overrides": {"count": 4},
        }}],
        "outputs": {"main": "rf01"},
    }))
    identity = asset_ref_mod.override_identity({"count": 4})
    variant = proj / "build" / "subassets" / f"part.{identity}.mora"
    outside = cache / "outside.mora"
    variant.replace(outside)
    try:
        variant.symlink_to(outside)
    except OSError as e:  # e.g. symlink-restricted environment
        pytest.skip(f"cannot create a file link here: {e}")
    with pytest.raises(AssetError) as exc:
        evaluate_asset(asset_mod.load(root))
    assert exc.value.code == "ref_escape"
    assert exc.value.detail["node"] == "rf01"


def test_cross_project_ref_from_subasset(cache):
    """The workspace root derives from the referencing asset's own
    project, so a sub-asset can hold the cross-project ref."""
    ws = cache / "ws"
    (ws / ".moraine").mkdir(parents=True)
    (ws / "alpha" / "subassets").mkdir(parents=True)
    (ws / "alpha" / "alpha.mora").write_text(
        ref_asset("subassets/mid.mora", "main"), encoding="utf-8"
    )
    (ws / "alpha" / "subassets" / "mid.mora").write_text(
        multi_ref_asset({"main": {"project": "beta", "output": "main"}}),
        encoding="utf-8",
    )
    (ws / "beta").mkdir()
    (ws / "beta" / "beta.mora").write_text(ALPHA, encoding="utf-8")
    r = evaluate_asset(asset_mod.load(build_workspace(ws, "alpha")))
    assert r["main"]["stats"]["bounds"]["max"][0] == 0.5


def test_integral_float_override_lands_identical_key(cache):
    """DESIGN 4.6: overrides are normalized to the number type each
    knob declares, so `4` and `4.0` offered to one knob are one variant
    and one cache key."""
    proj = cache / "proj"
    root = build_project(
        proj,
        root_text=multi_ref_asset(
            {
                "a": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"count": 4},
                },
                "b": {
                    "asset": "subassets/part.mora",
                    "output": "row",
                    "overrides": {"count": 4.0},
                },
            }
        ),
    )
    r = evaluate_asset(asset_mod.load(root))
    assert r["a"]["key"] == r["b"]["key"]
    assert r["a"]["stats"]["instances"] == 4


def test_retargeted_inner_asset_param_still_walks_the_grammar(cache):
    """An override that rewrites an inner ref's 'asset' param re-enters
    the full location contract — escapes are rejected exactly as if the
    path were authored in the file."""
    proj = cache / "proj"
    mid = {
        "build_version": "fixture",
        "nodes": [
            {
                "id": "rf01",
                "op": "asset_ref",
                "params": {"asset": "subassets/part.mora", "output": "solo"},
            }
        ],
        "params": [{"id": "src_asset", "value": "subassets/part.mora", "node": "rf01", "param": "asset"}],
        "outputs": {"picked": "rf01"},
    }
    root = write_project(
        proj,
        root_text=multi_ref_asset(
            {
                "main": {
                    "asset": "subassets/mid.mora",
                    "output": "picked",
                    "overrides": {"src_asset": "../../outside.star"},
                }
            }
        ),
    )
    (proj / "subassets" / "mid.mora").write_text(json.dumps(mid), encoding="utf-8")
    with pytest.raises(MoraineError) as exc:
        build_fixtures(proj)
    assert exc.value.code == "ref_escape"


def test_drop_warnings_never_re_raised_for_any_identity(cache):
    """Two identities of one referenced file raise nothing in the
    consumer (DESIGN 4.1 producer-side rule) — neither the plain nor the
    overridden instance is the consumer's edit site."""
    proj = cache / "proj"
    part = {
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {}},
            {"id": "u1", "op": "unwrap", "inputs": ["b1"], "params": {}},
            {"id": "b2", "op": "box", "params": {"size": [0.4, 2.0, 0.4]}},
            {
                "id": "d1",
                "op": "boolean",
                "inputs": ["u1", "b2"],
                "params": {"mode": "difference"},
            },
        ],
        "params": [{"id": "cut", "value": [0.4, 2.0, 0.4], "node": "b2", "param": "size", "min": 0.1, "max": 3.0}],
        "outputs": {"main": "d1"},
    }
    root = build_project(
        proj,
        root_text=multi_ref_asset(
            {
                "plain": {"asset": "subassets/part.mora", "output": "main"},
                "cutd": {
                    "asset": "subassets/part.mora",
                    "output": "main",
                    "overrides": {"cut": [0.3, 2.0, 0.3]},
                },
            }
        ),
        part_text=json.dumps(part),
    )
    ev = Evaluation(asset_mod.load(root))
    ev.records()
    assert ev.attribute_drop_warnings() == []


# -- sub-asset asserts through asset_ref (DESIGN 4.6) -----------------------


ASSERTED_PART = json.dumps(
    {
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {}},
            {"id": "g1", "op": "transform", "inputs": ["b1"],
             "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "b2", "op": "box", "params": {}},
            {"id": "g2", "op": "transform", "inputs": ["b2"],
             "params": {"translate": [0.0, 1.7, 0.0]}},
        ],
        "params": [
            {"id": "lift", "value": [0.0, 1.7, 0.0], "node": "g2",
             "param": "translate", "min": -5.0, "max": 5.0}
        ],
        "outputs": {"lower": "g1", "upper": "g2"},
        "asserts": [
            {"id": "sep", "kind": "clearance", "min": 0.1,
             "outputs": ["lower", "upper"], "owner": "human"}
        ],
    }
)


def _sub_entries(report):
    """The sub-asset's authored assert entries still listed: failing
    ones, and green ones with something to read (`unchecked`, `ceded`,
    `stronger`) — the rest fold into `ref_assets` (DESIGN 4.6). The
    built-in guard rides every report (the consumer's carries no
    `asset`; each bundle's does) and is pinned by its own tests below."""
    return [e for e in report["asserts"] if "asset" in e and not e.get("builtin")]


def _rows(report):
    """The per-(asset, identity) fold rows, keyed by asset and overrides."""
    return {(r["asset"], json.dumps(r.get("overrides"), sort_keys=True)): r
            for r in report.get("ref_assets", [])}


def _sub_guards(report):
    return [e for e in report["asserts"] if "asset" in e and e.get("builtin")]


def _asserted_root(cache, refs: dict[str, dict], part_text: str = ASSERTED_PART):
    proj = cache / "proj"
    doc = json.loads(multi_ref_asset(refs))
    # The consumer picks elevated kit parts: a parts-bin root, so
    # asset-level grounding is waived (DESIGN 4.1 asset_checks).
    doc["asset_checks"] = {"grounded": False}
    return build_project(proj, root_text=json.dumps(doc), part_text=part_text)


def _ref_assert_report(root):
    from moraine3d.verify.validate import append_ref_asserts, validate_outputs

    sc = asset_mod.load(root)
    ev = Evaluation(sc)
    values = {n: r.value() for n, r in ev.records().items()}
    report = validate_outputs(
        values, checks_policy=sc.checks, asserts=sc.asserts,
        asset_checks=sc.asset_checks,
    )
    return append_ref_asserts(report, ev.ref_asserts())


def test_subasset_asserts_checked_through_consumer(cache):
    """Sub-asset asserts are checked by every consuming root's validate
    (DESIGN 4.6): consuming an asset through asset_ref does not launder
    its asserts away, and the entry names its asset."""
    root = _asserted_root(
        cache, {"main": {"asset": "subassets/part.mora", "output": "upper"}}
    )
    report = _ref_assert_report(root)
    assert report["ok"] is True
    # Green and nothing to read: folded to the per-identity row, which
    # counts the guard and `sep` both green; no entry line survives.
    assert _sub_entries(report) == []
    assert report["ref_assets"] == [
        {"asset": "subassets/part.star", "asserts_total": 2, "asserts_ok": 2,
         "params_total": 1, "params_ok": 1}
    ]


def test_subasset_assert_failure_gates_consumer_validate(cache):
    """A failing sub-asset assert gates the consuming validate's ok
    (DESIGN 5.2: asserts are never waivable), and the entry carries the
    override identity it failed under."""
    root = _asserted_root(
        cache,
        {
            "main": {
                "asset": "subassets/part.mora",
                "output": "upper",
                "overrides": {"lift": [0.0, 1.55, 0.0]},
            }
        },
    )
    report = _ref_assert_report(root)
    assert report["ok"] is False
    (entry,) = _sub_entries(report)
    assert entry["ok"] is False and "clearance min" in entry["fix"]
    assert entry["asset"] == "subassets/part.star"
    assert entry["overrides"] == {"lift": [0.0, 1.55, 0.0]}


def test_subasset_asserts_per_override_identity(cache):
    """Sub-asset asserts run once per (file, override identity) — exactly
    the identity under which overrides change the geometry (DESIGN 4.6):
    one file consumed under two identities yields two entries."""
    root = _asserted_root(
        cache,
        {
            "plain": {"asset": "subassets/part.mora", "output": "upper"},
            "sunk": {
                "asset": "subassets/part.mora",
                "output": "upper",
                "overrides": {"lift": [0.0, 1.55, 0.0]},
            },
        },
    )
    report = _ref_assert_report(root)
    assert report["ok"] is False
    # One file, two identities: the plain one folds green, the sunk one
    # keeps its failing entry — and both have a row.
    (failing,) = _sub_entries(report)
    assert failing["ok"] is False and failing["overrides"] == {"lift": [0.0, 1.55, 0.0]}
    rows = _rows(report)
    assert len(rows) == 2
    plain = rows[("subassets/part.star", "null")]
    sunk = rows[("subassets/part.star", json.dumps({"lift": [0.0, 1.55, 0.0]}, sort_keys=True))]
    assert (plain["asserts_total"], plain["asserts_ok"]) == (2, 2)
    assert (sunk["asserts_total"], sunk["asserts_ok"]) == (2, 1)


def test_subasset_asserts_deduped_across_identical_uses(cache):
    """Identical uses of one sub-asset share one instance, so its asserts
    are checked once, not once per use site (DESIGN 4.6 identity)."""
    root = _asserted_root(
        cache,
        {
            "a": {"asset": "subassets/part.mora", "output": "upper"},
            "b": {"asset": "subassets/part.mora", "output": "lower"},
        },
    )
    report = _ref_assert_report(root)
    assert len(report["ref_assets"]) == 1 and _sub_entries(report) == []


def test_nested_subasset_asserts_checked(cache):
    """Asserts in transitively referenced sub-assets are checked too:
    every loaded instance is covered, not just direct references."""
    proj = cache / "proj"
    (proj / "subassets").mkdir(parents=True, exist_ok=True)
    (proj / "subassets" / "inner.mora").write_text(ASSERTED_PART, encoding="utf-8")
    passthrough = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "rf01", "op": "asset_ref",
                 "params": {"asset": "subassets/inner.mora", "output": "upper"}}
            ],
            "outputs": {"upper": "rf01"},
        }
    )
    root = _asserted_root(
        cache,
        {"main": {"asset": "subassets/part.mora", "output": "upper"}},
        part_text=passthrough,
    )
    report = _ref_assert_report(root)
    assert report["ok"] is True
    rows = _rows(report)
    # The passthrough carries only the guard; inner carries guard + sep.
    assert rows[("subassets/part.star", "null")]["asserts_total"] == 1
    inner = rows[("subassets/inner.star", "null")]
    assert (inner["asserts_total"], inner["asserts_ok"]) == (2, 2)


def test_subasset_assert_evaluates_unconsumed_outputs(cache):
    """A sub-asset assert referencing an output no consumer evaluates
    still evaluates it — the assert's scope is the sub-asset's own
    namespace."""
    root = _asserted_root(
        cache,
        # Only `upper` is consumed; `sep` also references `lower`.
        {"main": {"asset": "subassets/part.mora", "output": "upper"}},
    )
    report = _ref_assert_report(root)
    # The 0.2 m gap measured — "lower" was evaluated even though the
    # root never consumes it (an unevaluatable `sep` would fail with the
    # reason and stay listed).
    assert report["ok"] is True and _sub_entries(report) == []
    (row,) = report["ref_assets"]
    assert (row["asserts_total"], row["asserts_ok"]) == (2, 2)


def test_subasset_reached_only_through_unconsumed_output_is_checked(cache):
    """Every loaded instance is covered, including one reachable only
    through an output no consumer evaluates (DESIGN 4.6: consuming an
    asset through `asset_ref` never launders its asserts away): the
    root consumes the kit's `body`; the kit's unconsumed `cap` is an
    asset_ref into a lid whose clearance assert fails, and that failure
    gates the consumer's validate and export."""
    from support.engine import export_asset

    proj = cache / "proj"
    (proj / "subassets").mkdir(parents=True, exist_ok=True)
    lid = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {}},
                {"id": "b2", "op": "box", "params": {}},
                {"id": "g2", "op": "transform", "inputs": ["b2"],
                 "params": {"translate": [0.0, 1.05, 0.0]}},
            ],
            "outputs": {"lower": "b1", "upper": "g2"},
            "asserts": [
                {"id": "gap", "kind": "clearance", "min": 0.5,
                 "outputs": ["lower", "upper"], "owner": "human"}
            ],
        }
    )
    (proj / "subassets" / "lid.mora").write_text(lid, encoding="utf-8")
    kit = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "b1", "op": "box", "params": {}},
                {"id": "rf01", "op": "asset_ref",
                 "params": {"asset": "subassets/lid.mora", "output": "upper"}},
            ],
            "outputs": {"body": "b1", "cap": "rf01"},
        }
    )
    root = _asserted_root(
        cache,
        {"main": {"asset": "subassets/part.mora", "output": "body"}},
        part_text=kit,
    )
    report = _ref_assert_report(root)
    assert report["ok"] is False
    (failing,) = _sub_entries(report)
    assert failing["asset"] == "subassets/lid.star"
    assert failing["ok"] is False and "clearance min" in failing["fix"]
    rows = _rows(report)
    assert set(rows) == {("subassets/part.star", "null"), ("subassets/lid.star", "null")}
    assert rows[("subassets/lid.star", "null")]["asserts_ok"] == 1  # the guard

    result = export_asset(asset_mod.load(root), profile="gltf", out_dir=proj / "exports")
    assert result["ok"] is False
    assert any(
        e.get("asset") == "subassets/lid.star" and e["ok"] is False
        for e in result["validation"]["asserts"]
    )


GUARDED_KIT = json.dumps(
    {
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {}},
            {"id": "g1", "op": "transform", "inputs": ["b1"],
             "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "b2", "op": "box", "params": {}},
            {"id": "g2", "op": "transform", "inputs": ["b2"],
             "params": {"translate": [0.0, 1.5, 0.0]}},
        ],
        "params": [
            {"id": "lift", "value": [0.0, 1.5, 0.0], "node": "g2",
             "param": "translate", "min": -5.0, "max": 5.0}
        ],
        "outputs": {"lower": "g1", "upper": "g2"},
    }
)


def test_subasset_guard_covers_overridden_unconsumed_outputs(cache):
    """The built-in overlap guard runs in every reference bundle (DESIGN
    4.6): a consumer override that sinks two kit outputs into each other
    fails the consumer's validate even though it consumes only one of
    them — the kit's outputs as the overrides shaped them, consumed or
    not — while the unoverridden identity passes."""
    sunk = {"asset": "subassets/part.mora", "output": "lower",
            "overrides": {"lift": [0.0, 1.3, 0.0]}}
    root = _asserted_root(cache, {"main": sunk}, part_text=GUARDED_KIT)
    report = _ref_assert_report(root)
    assert report["ok"] is False
    (guard,) = _sub_guards(report)
    assert guard["asset"] == "subassets/part.star"
    assert guard["overrides"] == {"lift": [0.0, 1.3, 0.0]}
    assert guard["ok"] is False
    assert guard["detail"]["failing_pairs"][0][:2] == ["lower", "upper"]

    plain = {"asset": "subassets/part.mora", "output": "lower"}
    root = _asserted_root(cache, {"main": plain}, part_text=GUARDED_KIT)
    report = _ref_assert_report(root)
    assert _sub_guards(report) == []  # green, nothing ceded: folded
    assert report["ref_assets"] == [
        {"asset": "subassets/part.star", "asserts_total": 1, "asserts_ok": 1,
         "params_total": 1, "params_ok": 1}
    ]
    assert report["ok"] is True


def test_subasset_guard_cedes_embeds_through_nested_references(cache):
    """A bundle's guard inherits the cede of the references its own
    asset makes (DESIGN 5.2 cede through references, in every bundle): a
    kit that rigidly places two outputs of an embedding kit keeps their
    declared embed ceded, so the consumer stays green."""
    proj = cache / "proj"
    (proj / "subassets").mkdir(parents=True, exist_ok=True)
    (proj / "subassets" / "inner.mora").write_text(EMBED_KIT, encoding="utf-8")
    mid = json.dumps(
        {
            "build_version": "fixture",
            "nodes": [
                {"id": "rf01", "op": "asset_ref",
                 "params": {"asset": "subassets/inner.mora", "output": "lower"}},
                {"id": "rf02", "op": "asset_ref",
                 "params": {"asset": "subassets/inner.mora", "output": "upper"}},
                {"id": "t1", "op": "transform", "inputs": ["rf01"],
                 "params": {"translate": [3.0, 0.0, 0.0]}},
                {"id": "t2", "op": "transform", "inputs": ["rf02"],
                 "params": {"translate": [3.0, 0.0, 0.0]}},
            ],
            "outputs": {"lower": "t1", "upper": "t2"},
        }
    )
    root = _asserted_root(
        cache, {"main": {"asset": "subassets/part.mora", "output": "lower"}},
        part_text=mid,
    )
    report = _ref_assert_report(root)
    guards = {g["asset"]: g for g in _sub_guards(report)}
    assert guards["subassets/part.star"]["ceded"] == {"pairs": [["lower", "upper"]]}
    assert guards["subassets/inner.star"]["ceded"] == {
        "pairs": [["lower", "upper"]], "self": ["weave"],
    }
    assert all(g["ok"] for g in guards.values()) and report["ok"] is True


# -- cede through references (DESIGN 5.2) ------------------------------------


EMBED_KIT = json.dumps(
    {
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {}},
            {"id": "g1", "op": "transform", "inputs": ["b1"],
             "params": {"translate": [0.0, 0.5, 0.0]}},
            {"id": "b2", "op": "box", "params": {}},
            {"id": "g2", "op": "transform", "inputs": ["b2"],
             "params": {"translate": [0.0, 1.3, 0.0]}},
            {"id": "g3", "op": "transform", "inputs": ["g1"],
             "params": {"translate": [0.0, 0.0, 4.0]}},
            {"id": "a1", "op": "array", "inputs": ["g3"],
             "params": {"count": 3, "step_translate": [0.8, 0.0, 0.0]}},
        ],
        "params": [
            {"id": "lift", "value": [0.0, 1.3, 0.0], "node": "g2",
             "param": "translate", "min": -5.0, "max": 5.0}
        ],
        "outputs": {"lower": "g1", "upper": "g2", "weave": "a1"},
        "asserts": [
            {"id": "seam", "kind": "embedded", "outputs": ["lower", "upper"],
             "min": 0.05, "max": 0.3, "owner": "agent"},
            {"id": "free", "kind": "embedded", "outputs": ["weave"],
             "self": True, "owner": "agent"},
        ],
    }
)


def _embed_consumer(cache, nodes, outputs):
    proj = cache / "proj"
    doc = {"build_version": "fixture", "nodes": nodes, "outputs": outputs,
           "asset_checks": {"grounded": False}}
    return build_project(proj, root_text=json.dumps(doc), part_text=EMBED_KIT)


def _cede_report(root):
    from moraine3d.verify.validate import validate_outputs

    sc = asset_mod.load(root)
    ev = Evaluation(sc)
    values = {n: r.value() for n, r in ev.records().items()}
    report = validate_outputs(
        values, checks_policy=sc.checks, asserts=sc.asserts,
        asset_checks=sc.asset_checks, ref_embeds=ev.embedded_via_refs(),
    )
    guard = next(e for e in report["asserts"] if e.get("builtin"))
    return report, ev, guard


def _kit_refs(*outputs):
    return [
        {"id": f"rf{i:02d}", "op": "asset_ref",
         "params": {"asset": "subassets/part.mora", "output": out}}
        for i, out in enumerate(outputs, 1)
    ]


def test_ref_embeds_cede_rigidly_placed_kit(cache):
    """Embeds declared inside a referenced asset travel with its parts
    (DESIGN 5.2 cede through references): consumer outputs reaching the
    same reference identity through one rigid placement inherit the
    kit's pair embeds, and a copy-form embed cedes the consumer
    output's own copy pairs."""
    nodes = _kit_refs("lower", "upper", "weave") + [
        {"id": f"t{i}", "op": "transform", "inputs": [f"rf{i:02d}"],
         "params": {"translate": [2.0, 0.0, 0.0], "rotate_deg": [0.0, 30.0, 0.0]}}
        for i in (1, 2, 3)
    ]
    root = _embed_consumer(
        cache, nodes, {"a_lower": "t1", "a_upper": "t2", "a_weave": "t3"}
    )
    report, ev, guard = _cede_report(root)
    assert ev.embedded_via_refs() == {
        "pairs": [["a_lower", "a_upper"]], "self": ["a_weave"],
    }
    assert guard["ok"] is True
    assert guard["ceded"] == {"pairs": [["a_lower", "a_upper"]],
                              "self": ["a_weave"]}
    assert report["ok"] is True


def test_ref_embeds_do_not_cede_across_placements_or_scale(cache):
    """A consumer that re-places one part has re-made the joint: outputs
    under different composed placements do not map, and a scaled
    placement (embed depths change) breaks the trace entirely (DESIGN
    5.2 cede through references)."""
    nodes = _kit_refs("lower", "upper") + [
        {"id": "t1", "op": "transform", "inputs": ["rf01"],
         "params": {"translate": [2.0, 0.0, 0.0]}},
        {"id": "t2", "op": "transform", "inputs": ["rf02"],
         "params": {"translate": [2.0, 0.1, 0.0]}},  # re-seated: deeper joint
    ]
    root = _embed_consumer(cache, nodes, {"a_lower": "t1", "a_upper": "t2"})
    report, ev, guard = _cede_report(root)
    assert ev.embedded_via_refs() == {"pairs": [], "self": []}
    assert guard["ok"] is False  # the consumer owns the re-made joint

    nodes = _kit_refs("lower", "upper") + [
        {"id": f"t{i}", "op": "transform", "inputs": [f"rf{i:02d}"],
         "params": {"scale": 2.0}}
        for i in (1, 2)
    ]
    root = _embed_consumer(cache, nodes, {"s_lower": "t1", "s_upper": "t2"})
    report, ev, guard = _cede_report(root)
    assert ev.embedded_via_refs() == {"pairs": [], "self": []}
    assert guard["ok"] is False

    # Rigidity is exact, not within a relative tolerance: a near-unit
    # scale is still a scaled placement.
    nodes = _kit_refs("lower", "upper") + [
        {"id": f"t{i}", "op": "transform", "inputs": [f"rf{i:02d}"],
         "params": {"scale": 1.000001}}
        for i in (1, 2)
    ]
    root = _embed_consumer(cache, nodes, {"s_lower": "t1", "s_upper": "t2"})
    _report, ev, _guard = _cede_report(root)
    assert ev.embedded_via_refs() == {"pairs": [], "self": []}


def test_ref_embeds_follow_placement_traits(cache):
    """The cede walk is op-trait-driven (`PLACEMENT`, DESIGN 5.2 cede
    through references): an identity op between the reference and the
    output keeps the trace, and a data-placed op re-makes the joint —
    the pair it places does not map."""
    nodes = _kit_refs("lower", "upper") + [
        {"id": f"u{i}", "op": "unwrap", "inputs": [f"rf{i:02d}"],
         "params": {"mode": "box"}}
        for i in (1, 2)
    ] + [
        {"id": f"t{i}", "op": "transform", "inputs": [f"u{i}"],
         "params": {"translate": [2.0, 0.0, 0.0]}}
        for i in (1, 2)
    ]
    root = _embed_consumer(cache, nodes, {"a_lower": "t1", "a_upper": "t2"})
    report, ev, guard = _cede_report(root)
    assert ev.embedded_via_refs() == {"pairs": [["a_lower", "a_upper"]], "self": []}
    assert guard["ok"] is True and report["ok"] is True

    nodes = _kit_refs("lower", "upper") + [
        {"id": "deck", "op": "box", "params": {"size": [4.0, 0.2, 4.0]}},
        {"id": "gd", "op": "transform", "inputs": ["deck"],
         "params": {"translate": [0.0, 0.1, 0.0]}},
        {"id": "top", "op": "tag", "inputs": ["gd"],
         "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
        {"id": "p1", "op": "place", "inputs": ["top", "rf01"],
         "params": {"anchor": "top"}},
        {"id": "p2", "op": "place", "inputs": ["top", "rf02"],
         "params": {"anchor": "top"}},
    ]
    root = _embed_consumer(
        cache, nodes, {"deck": "top", "a_lower": "p1", "a_upper": "p2"}
    )
    report, ev, guard = _cede_report(root)
    assert ev.embedded_via_refs() == {"pairs": [], "self": []}
    assert guard["ok"] is False  # both seated on one plane: a re-made joint


def test_placement_traits_are_declared_consistently():
    """Every op declares `PLACEMENT` from the walk's vocabulary, and an
    affine one supplies the matrix the walk composes (ops registry
    contract)."""
    from moraine3d import ops

    for name in ops.names():
        mod = ops.get(name)
        placement = getattr(mod, "PLACEMENT", None)
        assert placement in (None, "identity", "affine", "rigid"), name
        if placement == "affine":
            assert callable(getattr(mod, "matrix", None)), name


def test_ref_embeds_do_not_cede_across_override_identities(cache):
    """A different override identity may change the geometry, so the
    sub-asset's embed does not vouch for a cross-identity pair (DESIGN
    5.2 cede through references)."""
    nodes = [
        {"id": "rf01", "op": "asset_ref",
         "params": {"asset": "subassets/part.mora", "output": "lower"}},
        {"id": "rf02", "op": "asset_ref",
         "params": {"asset": "subassets/part.mora", "output": "upper",
                    "overrides": {"lift": [0.0, 1.25, 0.0]}}},
    ]
    root = _embed_consumer(cache, nodes, {"a_lower": "rf01", "a_upper": "rf02"})
    report, ev, guard = _cede_report(root)
    assert ev.embedded_via_refs() == {"pairs": [], "self": []}
    assert guard["ok"] is False  # 0.25 m overlap, undeclared at the root


def test_subasset_dimension_asserts_ride_through_consumer(cache):
    """A sub-asset's dimension asserts (DESIGN 5.2) are checked by the
    consuming root like every other kind — the face form included, whose
    entry names `faces` rather than `outputs` — and gate its validate."""
    part = json.loads(ASSERTED_PART)
    part["nodes"].insert(3, {"id": "t2", "op": "tag", "inputs": ["b2"],
                             "params": {"name": "cap", "select": {"normal_y": {"gt": 0.9}}}})
    part["nodes"][4]["inputs"] = ["t2"]  # g2 now transforms the tagged box
    part["asserts"] = [
        {"id": "cap", "kind": "extent", "faces": ["upper", "cap"], "measure": "min_y",
         "value": 2.2, "owner": "human"},
        {"id": "one", "kind": "count", "outputs": ["upper"], "value": 2, "owner": "agent"},
    ]
    root = _asserted_root(
        cache, {"main": {"asset": "subassets/part.mora", "output": "upper"}},
        part_text=json.dumps(part),
    )
    report = _ref_assert_report(root)
    assert report["ok"] is False
    (entry,) = _sub_entries(report)
    assert entry["assert"] == "one" and entry["measured"] == 1
    assert entry["asset"] == "subassets/part.star"
    assert report["ref_assets"] == [
        {"asset": "subassets/part.star", "asserts_total": 3, "asserts_ok": 2,
         "params_total": 1, "params_ok": 1}
    ]


def test_provenance_is_ignored_by_evaluation(cache):
    """DESIGN 4.6: a vendored copy records its provenance (source path
    and content hash, an optional document-root field ignored by
    evaluation) — with the field absent, present, or carrying different
    content, every output of the asset keeps the same cache key and
    result hash, whether evaluated directly or through a reference."""
    proj = cache / "proj"
    root = write_project(proj)
    sub_path = proj / "subassets" / "part.mora"

    def outputs(asset_path) -> dict:
        return {
            name: (rec.key, rec.result_hash)
            for name, rec in Evaluation(asset_mod.load(asset_path)).records().items()
        }

    plain_sub = outputs(sub_path)
    plain_root = outputs(root)
    assert set(plain_sub) == {"solo", "row"}
    for provenance in (
        {"source": "examples/part/part.mora", "content_hash": "sha256:00"},
        {"source": "elsewhere/part.mora", "content_hash": "sha256:ff", "root": "kits"},
    ):
        doc = json.loads(PART)
        doc["provenance"] = provenance
        sub_path.write_text(asset_mod.dumps(asset_mod.loads(json.dumps(doc))), encoding="utf-8")
        assert asset_mod.load(sub_path).provenance == provenance
        assert outputs(sub_path) == plain_sub
        assert outputs(root) == plain_root
