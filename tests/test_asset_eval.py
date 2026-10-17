"""Asset loading, validation, determinism, and cache behavior."""

import json
import pathlib

import pytest

from moraine3d.core import asset as asset_mod
from moraine3d.core import sources as sources_mod
from moraine3d.errors import MoraineError, AssetError
from moraine3d.ops import box
from support import flatten
from support.engine import evaluate_asset

ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [1.0, 2.0, 3.0]}}
],
"outputs": {"main": "bx01"}
}
"""


def test_load_valid():
    sc = asset_mod.loads(ASSET)
    assert sc.build_version == "fixture"
    assert list(sc.outputs) == ["main"]
    assert sc.nodes["bx01"].op == "box"


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda d: d.pop("build_version"), "bad_build_version"),
        (lambda d: d.update(build_version=1), "bad_build_version"),
        (lambda d: d.update(version=0), "bad_root"),
        (lambda d: d["nodes"].append(dict(d["nodes"][0])), "duplicate_id"),
        (lambda d: d.update(outputs={"main": "nope"}), "bad_output_ref"),
        (lambda d: d.update(outputs={}), "bad_outputs"),
    ],
)
def test_load_rejects(mutate, code):
    doc = json.loads(ASSET)
    mutate(doc)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == code


def test_evaluate_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(ASSET)
    r1 = evaluate_asset(sc)
    glb = pathlib.Path(r1["main"]["glb"])
    assert glb.is_file()
    first_bytes = glb.read_bytes()

    # Re-evaluate: same key (cache hit), identical bytes after cache clear.
    r2 = evaluate_asset(sc)
    assert r2["main"]["key"] == r1["main"]["key"]
    glb.unlink()
    (tmp_path / "cache" / f"{r1['main']['key']}.meta.json").unlink()
    r3 = evaluate_asset(sc)
    assert glb.read_bytes() == first_bytes
    assert r3["main"]["stats"] == r1["main"]["stats"]


def test_mesh_result_hash_deterministic():
    m1 = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    m2 = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    assert m1.result_hash() == m2.result_hash()
    assert m1.result_hash() != box.evaluate([], {"size": [1.0, 2.0, 3.1]}, None).result_hash()


def test_unknown_op_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    doc = json.loads(ASSET)
    doc["nodes"][0]["op"] = "torus"
    sc = asset_mod.loads(json.dumps(doc))
    with pytest.raises(MoraineError) as exc:
        evaluate_asset(sc)
    assert exc.value.code == "unknown_op"


RECIPE = 'mo.output("main", mo.box(id = "bx01", size = [1.0, 2.0, 3.0]))\n'


def test_resolve_project_dir(tmp_path):
    """DESIGN 6: `PROJECT` is a project directory holding its root asset
    file or an explicit recipe path; a `.mora` path is accepted as the
    IR it is."""
    (tmp_path / "crate.star").write_text(RECIPE)
    (tmp_path / "crate.requirements.star").write_text("")
    assert sources_mod.resolve(tmp_path).name == "crate.star"
    assert sources_mod.resolve(tmp_path / "crate.star").name == "crate.star"
    (tmp_path / "loose.mora").write_text(ASSET)
    assert sources_mod.resolve(tmp_path / "loose.mora").name == "loose.mora"


def test_resolve_ignores_what_is_not_an_asset_file(tmp_path):
    """DESIGN 4.2: an asset file is a `*.star` file that is not a
    requirements file. A directory named like one, a requirements
    file, and the IR never classify a project."""
    (tmp_path / "old.star").mkdir()
    (tmp_path / "crate.requirements.star").write_text("")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "crate.mora").write_text(ASSET)
    (tmp_path / "stray.mora").write_text(ASSET)
    with pytest.raises(AssetError) as exc:
        sources_mod.resolve(tmp_path)
    assert exc.value.code == "asset_not_found"
    (tmp_path / "crate.star").write_text(RECIPE)
    assert sources_mod.resolve(tmp_path).name == "crate.star"


def test_resolve_rejects_empty_and_ambiguous(tmp_path):
    """DESIGN 4.2: a directory with exactly one top-level asset file is
    a project; a second one is refused, naming `lib/` for definitions
    and `subassets/` for parts."""
    with pytest.raises(AssetError) as exc:
        sources_mod.resolve(tmp_path)
    assert exc.value.code == "asset_not_found"
    (tmp_path / "a.star").write_text(RECIPE)
    (tmp_path / "b.star").write_text(RECIPE)
    with pytest.raises(AssetError) as exc:
        sources_mod.resolve(tmp_path)
    assert exc.value.code == "ambiguous_asset"
    assert "lib/" in exc.value.message and "subassets/" in exc.value.message
    assert exc.value.detail["assets"] == ["a.star", "b.star"]


def test_canonical_dumps_roundtrip():
    sc = asset_mod.loads(ASSET)
    text = asset_mod.dumps(sc)
    # One node per line: the node's line is exactly one line of the output.
    node_lines = [l for l in text.splitlines() if l.startswith('{"id"')]
    assert len(node_lines) == 1
    sc2 = asset_mod.loads(text)
    assert sc2.nodes["bx01"].params == sc.nodes["bx01"].params
    assert sc2.outputs == sc.outputs


def test_growing_root_objects_serialize_entry_per_line():
    """DESIGN 4.1: root containers whose entries grow per output —
    ``outputs`` and ``checks`` — serialize one entry per line like the
    root lists; fixed-keyset objects (``budgets``) stay inline."""
    doc = json.loads(ASSET)
    doc["nodes"].append({"id": "bx02", "op": "box", "params": {}})
    doc["outputs"] = {"main": "bx01", "side": "bx02"}
    doc["checks"] = {"main": {"not_sunk": False}, "side": {"not_enclosed": False}}
    doc["budgets"] = {"materials": 3}
    text = asset_mod.dumps(asset_mod.loads(json.dumps(doc)))
    lines = text.splitlines()
    assert '"outputs": {' in lines and '"main": "bx01",' in lines
    assert '"checks": {' in lines and '"main": {"not_sunk": false},' in lines
    assert '"budgets": {"materials": 3}' in lines
    assert asset_mod.loads(text).checks == doc["checks"]


def test_duplicate_json_keys_rejected():
    """json.loads resolves duplicate object keys last-wins before any
    validation can see them — a duplicated output name would silently
    drop a binding while validate stays green (never silent, P6)."""
    text = """{
"build_version": "fixture",
"nodes": [
{"id": "a", "op": "box", "params": {}},
{"id": "b", "op": "box", "params": {}}
],
"outputs": {"main": "a", "main": "b"}
}
"""
    with pytest.raises(AssetError) as e:
        asset_mod.loads(text)
    assert e.value.code == "duplicate_key"


# -- loader: unhashable values, unknown keys, encoding, casing --------------


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda d: d.update(outputs={"main": ["bx01"]}), "bad_output_ref"),
        (lambda d: d.update(params=[{"id": "w", "value": 1.0, "node": ["bx01"],
                                      "param": "size", "min": 0, "max": 9}]),
         "bad_param_node"),
        (lambda d: d.update(asserts=[{"id": "a", "kind": ["no_overlap"],
                                       "outputs": "all"}]), "bad_assert_kind"),
        (lambda d: d.update(asserts=[{"id": "a", "kind": "contact",
                                       "outputs": [["main"], ["main"]]}]),
         "bad_assert_outputs"),
    ],
)
def test_unhashable_json_values_are_asset_errors(mutate, code):
    """A list/object where the loader expects a name (an output ref, a
    param declaration's node, an assert kind, an assert outputs element)
    is an AssetError with the existing code, never a raw TypeError from a
    membership test (DESIGN 8.2: every loader failure is actionable)."""
    doc = json.loads(ASSET)
    mutate(doc)
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == code


def test_unknown_node_and_root_keys_rejected():
    """DESIGN 4.1 never-silent loading: a key the loader does not know
    (a typo'd 'param', a stray 'note', a root 'meta') is rejected with
    the accepted keys listed — not carried into evaluation with every
    default and dropped again by dumps."""
    doc = json.loads(ASSET)
    doc["nodes"][0] = {"id": "bx01", "op": "box", "param": {"size": [2, 2, 2]}, "note": "hi"}
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_node"
    assert "'note'" in exc.value.message and "'param'" in exc.value.message
    assert "params" in exc.value.message
    doc = json.loads(ASSET)
    doc["meta"] = 1
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(json.dumps(doc))
    assert exc.value.code == "bad_root"
    assert "'meta'" in exc.value.message and "outputs" in exc.value.message


def test_load_rejects_non_utf8_and_unreadable_files(tmp_path, monkeypatch):
    """An asset saved in a legacy codepage, or one the OS refuses to read,
    is an AssetError naming the file and the fix — never a raw
    UnicodeDecodeError/OSError traceback out of the CLI."""
    bad = tmp_path / "legacy.mora"
    # A legacy single-byte codepage: 0xE9 alone is not valid UTF-8.
    bad.write_bytes(ASSET.replace("bx01", "bé").encode("cp1252"))
    with pytest.raises(AssetError) as exc:
        asset_mod.load(bad)
    assert exc.value.code == "bad_encoding"
    assert "UTF-8" in exc.value.message and "legacy.mora" in exc.value.message

    good = tmp_path / "ok.mora"
    good.write_text(ASSET, encoding="utf-8")

    def denied(self):
        raise PermissionError(13, "held by another process")

    # load() reads raw bytes (CRLF must reach the canonical comparison).
    monkeypatch.setattr("pathlib.Path.read_bytes", denied)
    with pytest.raises(AssetError) as exc:
        asset_mod.load(good)
    assert exc.value.code == "asset_unreadable"
    assert "ok.mora" in exc.value.message


def test_project_root_folds_subassets_casing(tmp_path):
    """DESIGN 4.2/4.6: the project root of a sub-asset is its parent's
    parent whatever casing the on-disk 'subassets' directory carries
    (resolve() restores on-disk casing; the layout slot is the same)."""
    sub = tmp_path / "proj" / "Subassets"
    sub.mkdir(parents=True)
    (sub / "a.mora").write_text(ASSET, encoding="utf-8")
    assert asset_mod.project_root(sub / "a.mora") == (tmp_path / "proj").resolve()
    assert asset_mod.project_root(tmp_path / "proj" / "p.mora") == (tmp_path / "proj").resolve()
    assert asset_mod.project_root(sub / "a.star") == (tmp_path / "proj").resolve()


def test_an_ir_under_build_belongs_to_its_project(tmp_path):
    """DESIGN 4.6: the IR mirrors the project's layout under `build/`,
    a variant beside the IR of its asset, so an IR belongs to the
    project whose `build/` holds it and its references resolve there."""
    ws = tmp_path.resolve()
    identity = "0123456789abcdef01234567"
    cases = {
        ws / "yard" / "build" / "yard.mora": ("yard", "yard/build"),
        ws / "yard" / "build" / "subassets" / "gate.mora": ("yard", "yard/build"),
        ws / "yard" / "build" / "subassets" / f"gate.{identity}.mora": ("yard", "yard/build"),
        ws / "lamp" / "build" / f"lamp_post.{identity}.mora": ("lamp", "lamp/build"),
    }
    for path, (project, home) in cases.items():
        assert asset_mod.project_root(path) == ws / project, path
        assert asset_mod.ir_home(path) == ws / home, path
    recipes = [sources_mod.recipe_of(p).relative_to(ws).as_posix() for p in cases]
    assert recipes == [
        "yard/yard.star", "yard/subassets/gate.star", "yard/subassets/gate.star",
        "lamp/lamp_post.star",
    ]
    assert sources_mod.recipe_of(ws / "loose.mora") is None


def test_dumps_never_writes_infinity():
    """Canonical serialization is standard JSON: an inf that reached a
    node param (an overflow literal like 1e999 parses without hitting
    parse_constant) raises instead of writing an 'Infinity' token, and
    the error is actionable (DESIGN 8.2): it names the node and the
    param path carrying the value and the fix, never a bare failure."""
    sc = asset_mod.loads(ASSET.replace("[1.0, 2.0, 3.0]", "[1e999, 2.0, 3.0]"))
    with pytest.raises(AssetError) as exc:
        asset_mod.dumps(sc)
    err = exc.value
    assert err.code == "nonfinite_number"
    assert "'bx01'" in err.message and "'size[0]'" in err.message
    assert "finite number" in err.message
    assert err.detail == {"node": "bx01", "param": "size[0]"}
    # The same rule reaches a nested value and one outside every node.
    nested = ASSET.replace(
        '"params": {"size": [1.0, 2.0, 3.0]}',
        '"params": {"size": [1.0, 2.0, 3.0], "extra": {"deep": [0, {"k": 1e999}]}}',
    )
    with pytest.raises(AssetError) as exc:
        asset_mod.dumps(asset_mod.loads(nested))
    assert exc.value.detail == {"node": "bx01", "param": "extra.deep[1].k"}
    rooted = ASSET.replace(
        '"outputs": {"main": "bx01"}',
        '"outputs": {"main": "bx01"},\n"provenance": {"source": "x", "content_hash": "y", "n": 1e999}',
    )
    with pytest.raises(AssetError) as exc:
        asset_mod.dumps(asset_mod.loads(rooted))
    assert exc.value.code == "nonfinite_number"
    assert exc.value.detail == {"path": "provenance.n"}


# -- cache key contract (DESIGN 4.5) -----------------------------------------


def test_impl_version_bump_rekeys_the_record(tmp_path, monkeypatch):
    """Rule 3 / DESIGN 4.5: op_impl_version is inside every cache key, so
    a bump re-evaluates into a second record instead of serving the old
    one."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(ASSET)
    k1 = evaluate_asset(sc)["main"]["key"]
    monkeypatch.setattr(box, "OP_IMPL_VERSION", box.OP_IMPL_VERSION + 1)
    k2 = evaluate_asset(sc)["main"]["key"]
    assert k1 != k2
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2


def test_carry_version_is_in_every_key(tmp_path, monkeypatch):
    """DESIGN 4.5: the evaluator-level CARRY_VERSION (material/face-group
    carry rules, npz record layout) is folded into every cache key, so a
    change there can never serve records written under the old rules."""
    from moraine3d.engine import evaluator

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = asset_mod.loads(ASSET)
    k1 = evaluate_asset(sc)["main"]["key"]
    monkeypatch.setattr(evaluator, "CARRY_VERSION", evaluator.CARRY_VERSION + 1)
    k2 = evaluate_asset(sc)["main"]["key"]
    assert k1 != k2


def test_declared_kernel_version_is_in_the_key(tmp_path, monkeypatch):
    """DESIGN 4.5: the resolved version of every kernel an op declares in
    KERNELS joins its cache key, so a dependency upgrade or a shared-kernel
    bump re-evaluates instead of serving the record written under the old
    implementation — while an op declaring no kernel keeps a key without
    the field, so pure ops keep their records across the mechanism."""
    import importlib.metadata

    from moraine3d.engine import evaluator
    from moraine3d.ops import unwrap

    assert unwrap.KERNELS == ("core.mesh",) and box.KERNELS == ()
    assert isinstance(evaluator._kernel_version("core.mesh"), int)
    assert evaluator._kernel_version("trimesh") == importlib.metadata.version("trimesh")
    bare = evaluator._cache_key("box", 1, {}, [])
    assert evaluator._cache_key("box", 1, {}, [], None, ()) == bare
    assert evaluator._cache_key("box", 1, {}, [], None, ("core.mesh",)) != bare

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    doc = {
        "build_version": "fixture",
        "nodes": [
            {"id": "bx01", "op": "box", "params": {}},
            {"id": "uw01", "op": "unwrap", "inputs": ["bx01"], "params": {}},
        ],
        "outputs": {"main": "bx01", "uv": "uw01"},
    }
    sc = asset_mod.loads(json.dumps(doc))
    r1 = evaluate_asset(sc)
    real = evaluator._kernel_version
    monkeypatch.setattr(evaluator, "_kernel_version", lambda name: f"{real(name)}+1")
    r2 = evaluate_asset(sc)
    assert r1["uv"]["key"] != r2["uv"]["key"]
    assert r1["main"]["key"] == r2["main"]["key"]


def test_node_salted_ops_get_distinct_records(tmp_path, monkeypatch):
    """DESIGN 4.4/4.5: two USES_NODE_ID nodes with identical params
    (scatter) salt their streams with the node id, so they get distinct
    cache keys and different transforms — never one shared record."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    doc = {
        "build_version": "fixture",
        "nodes": [
            {"id": "p", "op": "box", "params": {"size": [0.1, 0.1, 0.1]}},
            {"id": "g", "op": "grid", "params": {}},
            {"id": "s1", "op": "scatter", "inputs": ["g", "p"], "params": {"count": 5, "seed": 1}},
            {"id": "s2", "op": "scatter", "inputs": ["g", "p"], "params": {"count": 5, "seed": 1}},
        ],
        "outputs": {"a": "s1", "b": "s2"},
    }
    from support.engine import evaluate_records

    recs = evaluate_records(asset_mod.loads(json.dumps(doc)))
    assert recs["a"].key != recs["b"].key
    ta, tb = recs["a"].value().transforms, recs["b"].value().transforms
    assert ta.shape == tb.shape == (5, 4, 4)
    assert not (ta == tb).all()


# -- non-finite geometry stays JSON-safe ------------------------------------


def _nonfinite_asset(size: str) -> str:
    return ASSET.replace("[1.0, 2.0, 3.0]", size)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_nonfinite_geometry_stats_are_json_safe(tmp_path, monkeypatch, capsys):
    """Overflowed geometry (a translate of 1e39 overflows float32; a box
    of 1e308 overflows its own bounds) must reach stats, eval, and serve
    as standard JSON — non-finite bounds are null — while the GLB writer
    refuses it with nonfinite_mesh and validate keeps reporting
    finite=false (the designed reporter of the condition)."""
    from moraine3d.app.cli import main
    from support.engine import Evaluation, evaluate_records
    from moraine3d.verify.validate import validate_outputs

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    big = tmp_path / "big"
    big.mkdir()
    (big / "big.mora").write_text(_nonfinite_asset("[1e308, 1e308, 1e308]"), encoding="utf-8")
    flatten.write_project(big, big)
    far = tmp_path / "far"
    far.mkdir()
    (far / "far.mora").write_text(
        ASSET.replace(
            '{"id": "bx01", "op": "box", "params": {"size": [1.0, 2.0, 3.0]}}',
            '{"id": "bx01", "op": "box", "params": {"size": [1.0, 2.0, 3.0]}},\n'
            '{"id": "tr01", "op": "transform", "inputs": ["bx01"], '
            '"params": {"translate": [1e39, 0.0, 0.0]}}',
        ).replace('"main": "bx01"', '"main": "tr01"'),
        encoding="utf-8",
    )
    flatten.write_project(far, far)
    for proj in (big, far):
        sc = asset_mod.load(proj / f"{proj.name}.mora")
        stats = evaluate_records(sc)["main"].stats
        assert None in stats["bounds"]["min"] or None in stats["bounds"]["max"]
        # The meta is re-read as a valid hit (null components are a legal
        # record shape), not self-healed away as corruption.
        assert evaluate_records(sc)["main"].stats == stats
        assert main(["stats", str(proj)]) == 0
        out = json.loads(capsys.readouterr().out)  # parses: standard JSON
        assert out["outputs"]["main"]["bounds"] == stats["bounds"]
        assert main(["eval", str(proj)]) == 1
        err = json.loads(capsys.readouterr().out)["error"]
        assert err["code"] == "nonfinite_mesh" and "main" in err["message"]
        with pytest.raises(MoraineError) as exc:
            Evaluation(sc).results()
        assert exc.value.code == "nonfinite_mesh"
        values = {n: r.value() for n, r in evaluate_records(sc).items()}
        report = validate_outputs(values)
        finite = [c for c in report["outputs"]["main"]["checks"] if c["check"] == "finite"]
        assert finite and finite[0]["ok"] is False


# -- ephemeral preview bucket (DESIGN 4.5) ------------------------------------

_CHAIN = ASSET.replace(
    '{"id": "bx01", "op": "box", "params": {"size": [1.0, 2.0, 3.0]}}',
    '{"id": "bx01", "op": "box", "params": {"size": [1.0, 2.0, 3.0]}},\n'
    '{"id": "tr01", "op": "transform", "inputs": ["bx01"], '
    '"params": {"translate": [0.0, 0.0, 0.0]}}',
).replace('"main": "bx01"', '"main": "tr01"')


def _chain_asset(translate):
    doc = json.loads(_CHAIN)
    doc["nodes"][1]["params"]["translate"] = translate
    return asset_mod.loads(json.dumps(doc))


def test_preview_pass_writes_aside_to_bucket(tmp_path, monkeypatch):
    """DESIGN 4.5 ephemeral preview bucket: a preview pass takes
    persistent hits first (the clean upstream of a scrubbed subgraph is
    file history), lands every miss it computes in the bucket — derived
    GLB beside its record — and leaves the persistent store's file set
    byte-identical: exploration never grows the persistent cache, and
    bucket bytes never join the session write accounting."""
    from moraine3d.cache import store as cache_mod
    from support.engine import Evaluation

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    cdir = tmp_path / "cache"
    evaluate_asset(_chain_asset([0.0, 0.0, 0.0]))  # file state -> persistent
    cache_mod.take_wrote()
    before = sorted(
        (p.name, p.stat().st_size) for p in cdir.iterdir() if p.is_file()
    )
    ev = Evaluation(_chain_asset([0.0, 1.0, 0.0]), preview=True)
    r = ev.results()
    prof = ev.profile()
    assert prof["hits"] == 1 and prof["misses"] == 1  # box hit, transform missed
    assert r["main"]["bucket"] is True
    assert pathlib.Path(r["main"]["glb"]).parent == cdir / "preview"
    assert (cdir / "preview" / f"{r['main']['key']}.npz").exists()
    after = sorted(
        (p.name, p.stat().st_size) for p in cdir.iterdir() if p.is_file()
    )
    assert after == before
    assert cache_mod.session_wrote() == 0
    # The same preview again: served from the bucket, nothing recomputed.
    ev2 = Evaluation(_chain_asset([0.0, 1.0, 0.0]), preview=True)
    assert ev2.results()["main"]["key"] == r["main"]["key"]
    prof2 = ev2.profile()
    assert prof2["hits"] == 2 and prof2["misses"] == 0


def test_file_state_recomputes_into_persistent_no_promotion(tmp_path, monkeypatch):
    """DESIGN 4.5: no promotion — a later file-state evaluation of a key
    the bucket already holds recomputes into the persistent store, so
    "the persistent cache indexes file history" stays literally true."""
    from support.engine import Evaluation

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    cdir = tmp_path / "cache"
    evaluate_asset(_chain_asset([0.0, 0.0, 0.0]))
    Evaluation(_chain_asset([0.0, 1.0, 0.0]), preview=True).results()
    r = evaluate_asset(_chain_asset([0.0, 1.0, 0.0]))  # previewed value applied
    assert "bucket" not in r["main"]
    assert (cdir / f"{r['main']['key']}.npz").exists()
    assert pathlib.Path(r["main"]["glb"]).parent == cdir


def test_the_ir_writer_is_atomic_and_cleans_its_temp(tmp_path):
    """DESIGN 4.5 (deletion safety): the one writer of IR files emits
    exactly the text it is given, and on failure its in-flight temp
    never lingers beside the IR."""
    sc = _chain_asset([0.0, 0.0, 0.0])
    target = tmp_path / "s.mora"
    asset_mod.write_text(target, asset_mod.dumps(sc))
    assert target.read_text(encoding="utf-8") == asset_mod.dumps(sc)
    bad = tmp_path / "missing-dir" / "s.mora"
    with pytest.raises(MoraineError) as exc:
        asset_mod.write_text(bad, asset_mod.dumps(sc))
    assert exc.value.code == "asset_unwritable"
    assert not list(tmp_path.rglob("*.tmp*"))
