"""Contract hardening: winding-safe inputs, strict parameter
validation, portable output names, cache self-healing, the registry
contract, and the cost tiering of stats/eval."""

import ast
import importlib
import importlib.metadata
import inspect
import json
import re
from importlib import resources
from pathlib import Path

import pytest

from conftest import SRC_ROOT, package_sources
from moraine3d import ops
from moraine3d.core import mesh as mesh_mod
from moraine3d.core import asset as asset_mod
from moraine3d.errors import MoraineError, OpError, AssetError
from moraine3d.ops import common as common_mod
from moraine3d.ops import transform as transform_op
from support.engine import evaluate_records, evaluate_asset, Evaluation


def _asset(nodes: list[dict], outputs: dict[str, str]) -> "asset_mod.Asset":
    return asset_mod.loads(
        json.dumps({"build_version": "fixture", "nodes": nodes, "outputs": outputs})
    )


BOX = {"id": "bx01", "op": "box", "params": {}}


# -- registry contract -------------------------------------------------------




def _op_sources() -> dict[str, str]:
    return {
        name: Path(ops.get(name).__file__).read_text(encoding="utf-8")
        for name in ops.names()
    }


def test_registry_contract():
    """ops/__init__.py module contract: every op declares PARAMS
    (frozenset), an int OP_IMPL_VERSION >= 1, INPUT_ARITY as (min, max)
    with max None = unbounded, a documented ATTRIBUTE_POLICY, a callable
    evaluate, and well-formed optional tier metadata — SEED_PARAMS a
    frozenset within PARAMS, UNIFORM_PARAMS a dict from PARAMS members
    into the UNIFORM_CHANNELS vocabulary (DESIGN 7.2: a channel outside
    the viewer's mapping would silently render as a dead knob)."""
    for name in ops.names():
        mod = ops.get(name)
        assert isinstance(mod.PARAMS, frozenset), name
        assert isinstance(mod.OP_IMPL_VERSION, int) and mod.OP_IMPL_VERSION >= 1, name
        lo, hi = mod.INPUT_ARITY
        assert isinstance(lo, int) and lo >= 0, name
        assert hi is None or (isinstance(hi, int) and hi >= lo), name
        assert mod.ATTRIBUTE_POLICY in ("preserved", "dropped"), name
        # This closed value set is also the interaction-tier classifier's
        # consumed-op guard (engine.tiers, DESIGN 7.2): a third,
        # material-consuming policy value must teach the classifier
        # before it can pass here.
        assert getattr(mod, "MATERIAL_POLICY", "preserved") in ("preserved", "dropped"), name
        seed_params = getattr(mod, "SEED_PARAMS", frozenset())
        assert isinstance(seed_params, frozenset) and seed_params <= mod.PARAMS, name
        uniform = getattr(mod, "UNIFORM_PARAMS", {})
        assert isinstance(uniform, dict) and set(uniform) <= mod.PARAMS, name
        assert set(uniform.values()) <= ops.UNIFORM_CHANNELS, name
        assert isinstance(mod.KERNELS, tuple), name
        assert len(set(mod.KERNELS)) == len(mod.KERNELS), name
        assert callable(mod.evaluate), name


def test_seed_params_cover_every_seed_param():
    """DESIGN 7.2: a param named ``seed`` selects an unordered stochastic
    stream, so an op declaring one must claim it in SEED_PARAMS —
    otherwise the knob would classify as an orderable slider (the wrong
    control, and its declared range an accidental cache-budget
    declaration)."""
    for name in ops.names():
        mod = ops.get(name)
        if "seed" in mod.PARAMS:
            assert "seed" in getattr(mod, "SEED_PARAMS", frozenset()), name


def test_node_salted_streams_declare_uses_node_id():
    """DESIGN 4.4/4.5 and rand.py: an op that derives streams with the
    node id (stream_u64 / uniform) declares USES_NODE_ID so the node id
    joins its cache key — and only such ops do (the evaluator reads the
    flag with a default of False, so a missing flag would silently share
    one record between two identical-param nodes)."""
    for name, src in _op_sources().items():
        salted = "stream_u64" in src or "uniform(" in src
        assert salted == bool(getattr(ops.get(name), "USES_NODE_ID", False)), name


KERNEL_MODULES = ("core.mating", "core.predicates", "core.rand", "core.mesh")
KERNEL_DISTS = ("manifold3d", "trimesh")


def _kernel_imports(tree: ast.AST) -> tuple[set[str], set[str], dict[str, str]]:
    """The kernels a module's imports reach, the sibling ``ops`` modules
    it imports, and the local names bound to kernel functions (so a
    helper module's functions can be traced to the kernels they call).
    A dataclass or exception class imported from a kernel module is not
    a dependency; ``core.mesh`` counts only its declared helpers."""
    kernels: set[str] = set()
    siblings: set[str] = set()
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in KERNEL_DISTS:
                    kernels.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 2 and node.module == "core":
                for a in node.names:
                    if f"core.{a.name}" in KERNEL_MODULES:
                        kernels.add(f"core.{a.name}")
            elif node.level == 2 and node.module in KERNEL_MODULES:
                mod = importlib.import_module(f"moraine3d.{node.module}")
                for a in node.names:
                    counts = inspect.isfunction(getattr(mod, a.name))
                    if node.module == "core.mesh":
                        counts = a.name in mesh_mod.KERNEL_HELPERS
                    if counts:
                        kernels.add(node.module)
                        bound[a.asname or a.name] = node.module
            elif node.level == 1:
                if node.module is None:
                    siblings.update(a.name for a in node.names)
                else:
                    siblings.add(node.module.split(".")[0])
    return kernels, siblings, bound


def _common_kernels() -> dict[str, set[str]]:
    """Per function of ``ops/common.py``: the kernels it calls."""
    tree = ast.parse(Path(common_mod.__file__).read_text(encoding="utf-8"))
    _, _, bound = _kernel_imports(tree)
    out: dict[str, set[str]] = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            out[fn.name] = {
                bound[n.id] for n in ast.walk(fn)
                if isinstance(n, ast.Name) and n.id in bound
            }
    return out


def test_kernels_declaration_matches_imports():
    """DESIGN 4.5: an op's KERNELS declaration equals the kernels its
    module imports, in both directions — a distribution (manifold3d,
    trimesh) or a shared in-package kernel module, reached directly,
    through the ``common`` helpers it calls, or through a sibling op it
    imports — so a consumed kernel cannot go undeclared and a declared
    one cannot be stale; every declared in-package kernel carries an
    int KERNEL_VERSION and every declared distribution is installed.
    numpy is never declared (its arithmetic is bit-tolerant, DESIGN 4.4)."""
    common = _common_kernels()
    direct: dict[str, set[str]] = {}
    edges: dict[str, set[str]] = {}
    for name in ops.names():
        tree = ast.parse(Path(ops.get(name).__file__).read_text(encoding="utf-8"))
        kernels, siblings, _ = _kernel_imports(tree)
        if "common" in siblings:
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "common"
                ):
                    kernels |= common.get(n.attr, set())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "common":
                for a in n.names:
                    kernels |= common.get(a.name, set())
        direct[name] = kernels
        edges[name] = {s for s in siblings if s in ops.names()}
    reached = {name: set(k) for name, k in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, sibs in edges.items():
            for s in sibs:
                if not reached[s] <= reached[name]:
                    reached[name] |= reached[s]
                    changed = True
    for name in ops.names():
        declared = ops.get(name).KERNELS
        assert set(declared) == reached[name], (name, declared, sorted(reached[name]))
        assert "numpy" not in declared, name
        for k in declared:
            if "." in k:
                assert k in KERNEL_MODULES, (name, k)
                version = importlib.import_module(f"moraine3d.{k}").KERNEL_VERSION
                assert isinstance(version, int) and version >= 1, (name, k)
            else:
                assert k in KERNEL_DISTS, (name, k)
                assert importlib.metadata.version(k)
    assert mesh_mod.KERNEL_HELPERS <= {
        n for n, v in vars(mesh_mod).items() if inspect.isfunction(v)
    }


def test_stochastic_ops_take_explicit_seed():
    """DESIGN 4.4: no global RNG — every op that draws on rand.py takes
    an explicit ``seed`` param, and nothing under src/moraine3d imports
    numpy's or Python's global random state (vendored code excluded)."""
    for name, src in _op_sources().items():
        if re.search(r"^\s*from \.\.core import rand|^\s*from \.\.core\.rand import", src, re.M):
            assert "seed" in ops.get(name).PARAMS, name
    for py in package_sources():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("random", "numpy.random"), py
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in ("random", "numpy.random"), py
                if node.module == "numpy":
                    assert all(a.name != "random" for a in node.names), py
        assert "np.random" not in py.read_text(encoding="utf-8"), py


def test_tessellation_ceilings_within_shared_budget():
    """DESIGN 5: per-dimension tessellation ceilings are absurdity
    guards, and the shared cell budget bounds their product - grid and
    sphere satisfy it structurally (their maximum product equals it),
    so raising a per-op ceiling must confront the budget here; revolve,
    whose hard joints duplicate rings past it, enforces it at runtime."""
    from moraine3d.ops import common, grid, revolve, sphere

    assert common.TESSELLATION_CELLS_MAX == 4096 * 4096
    assert grid.SEGMENTS_MAX * grid.SEGMENTS_MAX <= common.TESSELLATION_CELLS_MAX
    assert sphere.SEGMENTS_MAX * sphere.SEGMENTS_MAX <= common.TESSELLATION_CELLS_MAX
    # box's three face pairs multiply past the budget at max segments -
    # the runtime check (test_box) is the guard, as for revolve.
    from moraine3d.ops import box

    assert 2 * 3 * box.SEGMENTS_MAX * box.SEGMENTS_MAX > common.TESSELLATION_CELLS_MAX
    # revolve's worst-case ring count (every interior joint hard, both
    # ends capped) exceeds the budget at max segments - the runtime
    # check is reachable, not dead code.
    worst_rings = revolve.PROFILE_MAX - 2 + (revolve.PROFILE_MAX - 2) + 2
    assert worst_rings * revolve.SEGMENTS_MAX > common.TESSELLATION_CELLS_MAX


def test_runtime_strings_are_ascii():
    """Runtime strings - anything a print or raise can hand a console -
    stay pure ASCII: a legacy-codepage Windows console (e.g. cp932)
    crashes naive piping consumers on anything wider, and error text is
    the agent's working material (DESIGN 2 P2). Docstrings never reach a
    console and are exempt (comments are invisible to ast); input-parsing
    literals that must carry wide characters are allowlisted by exact
    value - never messages. Help pages get the same rule end to end in
    tests/test_cli.py::test_every_help_page_is_ascii."""
    from moraine3d.core.journal import POINTER_RE

    allowed = {POINTER_RE.pattern}
    offenders = []
    for py in package_sources():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and node.value not in allowed
                and not node.value.isascii()
            ):
                offenders.append(f"{py.relative_to(py.parents[3])}:{node.lineno}")
    assert not offenders, "non-ASCII runtime strings:\n" + "\n".join(offenders)


def test_operators_doc_headings_match_registry():
    """The shipped reference is the agent-facing contract (DESIGN 5, 10):
    every registered op has a heading ``<name> (op_impl_version N)`` in
    one of its files and N is the module's OP_IMPL_VERSION — no silent
    drift in either direction."""
    from moraine3d.app.cli import REFERENCE_FILES, reference_text

    text = "\n".join(reference_text(name) for name in REFERENCE_FILES)
    found = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^#{2,3} ([a-z_]+) \(op_impl_version (\d+)\)", text, re.M)
    }
    assert set(found) == set(ops.names())
    for name, version in found.items():
        assert version == ops.get(name).OP_IMPL_VERSION, name


# -- winding safety ----------------------------------------------------------


def test_extrude_negative_distance_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = _asset(
        [
            {"id": "g1", "op": "grid", "params": {}},
            {"id": "e1", "op": "extrude", "inputs": ["g1"], "params": {"distance": -0.5}},
        ],
        {"o": "e1"},
    )
    with pytest.raises(OpError) as exc:
        evaluate_records(sc)
    assert exc.value.detail["param"] == "distance"


def test_transform_mirror_on_instances_rejected():
    proto = ops.get("box").evaluate([], {}, None)
    inst = ops.get("array").evaluate([proto], {"count": 2}, None)
    with pytest.raises(OpError) as exc:
        transform_op.evaluate([inst], {"scale": [-1.0, 1.0, 1.0]}, None)
    assert exc.value.code == "bad_input"
    # The same mirror on a plain mesh stays legal (winding is flipped there).
    transform_op.evaluate([proto], {"scale": [-1.0, 1.0, 1.0]}, None)


def test_array_negative_step_scale_rejected():
    proto = ops.get("box").evaluate([], {}, None)
    with pytest.raises(OpError) as exc:
        ops.get("array").evaluate([proto], {"count": 3, "step_scale": -1.0}, None)
    assert exc.value.detail["param"] == "step_scale"


# -- parameter validation ----------------------------------------------------


def test_unknown_param_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = _asset([{"id": "bx01", "op": "box", "params": {"siez": [2.0, 2.0, 2.0]}}], {"o": "bx01"})
    with pytest.raises(OpError) as exc:
        evaluate_records(sc)
    assert exc.value.code == "unknown_param"
    assert "siez" in exc.value.message and "size" in exc.value.message


def test_unknown_param_in_sub_asset_names_its_asset(tmp_path, monkeypatch):
    """DESIGN 4.6 / 8.2: node ids are per file, so an evaluation error in
    a referenced asset locates its asset as well as its node —
    unknown_param like unknown_op and bad_inputs."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    (proj / "subassets" / "part.mora").write_text(json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "bx01", "op": "box", "params": {"siez": [1.0, 1.0, 1.0]}}],
        "outputs": {"main": "bx01"},
    }), encoding="utf-8")
    (proj / "proj.mora").write_text(json.dumps({
        "build_version": "fixture",
        "nodes": [{"id": "rf01", "op": "asset_ref", "params": {
            "asset": "subassets/part.mora", "output": "main"}}],
        "outputs": {"o": "rf01"},
    }), encoding="utf-8")
    with pytest.raises(OpError) as exc:
        evaluate_records(asset_mod.load(proj / "proj.mora"))
    assert exc.value.code == "unknown_param"
    assert exc.value.detail["node"] == "bx01"
    assert exc.value.detail["asset"] == "subassets/part.mora"


def test_unknown_op_and_bad_inputs_name_the_node(tmp_path, monkeypatch):
    """Every evaluation error locates its node (DESIGN 8.2 actionable
    errors): unknown_op and bad_inputs carry the node id like
    unknown_param does."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(OpError) as exc:
        evaluate_records(_asset([{"id": "t1", "op": "torus", "params": {}}], {"o": "t1"}))
    assert exc.value.code == "unknown_op" and exc.value.detail["node"] == "t1"
    with pytest.raises(OpError) as exc:
        evaluate_records(_asset(
            [BOX, {"id": "x1", "op": "extrude", "inputs": ["bx01", "bx01"], "params": {}}],
            {"o": "x1"},
        ))
    assert exc.value.code == "bad_inputs" and exc.value.detail["node"] == "x1"


def test_nonfinite_json_literal_rejected():
    text = ('{"build_version": "fixture", "nodes": [{"id": "b", "op": "box", '
            '"params": {"size": [NaN, 1.0, 1.0]}}], "outputs": {"o": "b"}}')
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(text)
    assert exc.value.code == "bad_json"


def test_nonfinite_param_rejected():
    with pytest.raises(OpError):
        transform_op.matrix({"translate": [float("nan"), 0.0, 0.0]})
    with pytest.raises(OpError):
        transform_op.matrix({"scale": float("inf")})
    with pytest.raises(OpError):
        ops.get("box").evaluate([], {"size": [float("inf"), 1.0, 1.0]}, None)


def test_scatter_bool_params_strict():
    proto = ops.get("box").evaluate([], {}, None)
    target = ops.get("grid").evaluate([], {}, None)
    with pytest.raises(OpError) as exc:
        ops.get("scatter").evaluate(
            [proto, target], {"count": 2, "align_normal": "false"}, None
        )
    assert exc.value.detail["param"] == "align_normal"


def test_geometry_ops_reject_image_inputs():
    # An Image reaching a geometry op must be an actionable OpError,
    # never a raw AttributeError or an Instances value wrapping a texture.
    img = ops.get("checker_texture").evaluate([], {"size": 8}, None)
    proto = ops.get("box").evaluate([], {}, None)
    target = ops.get("grid").evaluate([], {}, None)
    cases = [
        ("transform", [img]),
        ("array", [img]),
        ("scatter", [img, target]),
        ("scatter", [proto, img]),
    ]
    for op_name, inputs in cases:
        with pytest.raises(OpError) as exc:
            ops.get(op_name).evaluate(inputs, {}, None)
        assert exc.value.code == "bad_input", op_name


def test_seed_ceiling_is_actionable():
    # scatter/noise_displace previously took seed via common.integer with
    # no ceiling; 2**100 reached rand.py as a raw OverflowError.
    proto = ops.get("box").evaluate([], {}, None)
    target = ops.get("grid").evaluate([], {}, None)
    for op_name, inputs in [("scatter", [proto, target]), ("noise_displace", [target])]:
        with pytest.raises(OpError) as exc:
            ops.get(op_name).evaluate(inputs, {"seed": 2**100}, None)
        assert exc.value.detail["param"] == "seed", op_name


def test_time_root_key_refused_with_hint():
    """DESIGN 4.1: `time` is a reserved name, not a root key - a file
    still carrying the line is refused, and the error says to delete it."""
    text = (
        f'{{"build_version": "fixture", "time": 0.0, "nodes": '
        f'[{json.dumps(BOX)}], "outputs": {{"o": "bx01"}}}}'
    )
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(text)
    assert exc.value.code == "bad_root"
    assert "'time' is a reserved name" in str(exc.value)


# -- output names ------------------------------------------------------------


@pytest.mark.parametrize("name", ["bad/name", "bad name", "1bad", "..", "über"])
def test_output_name_charset_rejected(name):
    with pytest.raises(AssetError) as exc:
        _asset([BOX], {name: "bx01"})
    assert exc.value.code == "bad_output_name"


def test_output_name_case_collision_rejected():
    with pytest.raises(AssetError) as exc:
        _asset([BOX], {"Main": "bx01", "main": "bx01"})
    assert exc.value.code == "output_name_collision"


# -- cache -------------------------------------------------------------------


def test_cache_corruption_self_heals(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache))
    sc = _asset([BOX], {"o": "bx01"})
    evaluate_records(sc)
    npz = next(cache.glob("*.npz"))
    npz.write_bytes(b"not a zip")
    # Cache hit reads stats from meta; the corrupt value is hit on load.
    rec = evaluate_records(sc)["o"]
    with pytest.raises(MoraineError) as exc:
        rec.value()
    assert exc.value.code == "cache_corrupt"
    assert not npz.exists()
    # The record was removed, so the next run re-evaluates cleanly.
    assert evaluate_records(sc)["o"].value().stats()["verts"] == 24


def test_cache_meta_corruption_self_heals(tmp_path, monkeypatch):
    # A corrupt meta.json must not surface as a raw JSONDecodeError, and
    # the broken record pair must not survive to fail the next run too.
    cache = tmp_path / "cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache))
    sc = _asset([BOX], {"o": "bx01"})
    evaluate_records(sc)
    meta = next(cache.glob("*.meta.json"))
    meta.write_text("{not json", encoding="utf-8")
    # Recovers within the same pass: the pair is dropped and re-evaluated.
    assert evaluate_records(sc)["o"].stats["verts"] == 24
    assert json.loads(meta.read_text(encoding="utf-8"))["stats"]["verts"] == 24


def test_cache_dir_unwritable_is_actionable(tmp_path, monkeypatch):
    # An unusable cache location must not surface as a raw OSError
    # traceback: the error names the fix.
    from moraine3d.core.paths import cache_dir

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file where the cache dir should go")
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(blocker))
    with pytest.raises(MoraineError) as exc:
        cache_dir()
    assert exc.value.code == "cache_unwritable"
    assert "MORAINE_CACHE_DIR" in exc.value.message


def test_glb_filename_carries_writer_version(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache))
    sc = _asset([BOX], {"o": "bx01"})
    r = evaluate_asset(sc)
    from moraine3d.core.glb import GLB_VERSION

    assert r["o"]["glb_file"].endswith(f"-g{GLB_VERSION}-o.glb")
    assert (cache / r["o"]["glb_file"]).is_file()


# -- CLI cost tiering --------------------------------------------------------


def test_records_do_not_materialize_glbs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache))
    sc = _asset([BOX], {"o": "bx01"})
    records = evaluate_records(sc)
    assert records["o"].stats["verts"] == 24
    assert not list(cache.glob("*.glb"))


def test_eval_only_output_skips_others(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(cache))
    sc = _asset(
        [
            BOX,
            # A second output whose op would fail loudly if evaluated.
            {"id": "bad1", "op": "extrude", "inputs": ["bx01"], "params": {"distance": -1.0}},
        ],
        {"good": "bx01", "bad": "bad1"},
    )
    r = evaluate_asset(sc, only="good")
    assert list(r) == ["good"]
    with pytest.raises(AssetError) as exc:
        Evaluation(sc).records(only="nope")
    assert exc.value.code == "unknown_output"


def test_write_atomic_lost_rename_benign_only_for_cache_paths(tmp_path, monkeypatch):
    """Concurrent writers of the same key produce identical bytes, so a
    Windows open-file lock on the destination (PermissionError from
    os.replace) is a benign lost race for content-addressed cache names.
    Without a destination it is a cache that refuses writes
    (cache_unwritable, naming MORAINE_CACHE_DIR). Export paths are not
    content-addressed (same name, new bytes): there the lock is a loud
    export_locked failure that leaves no tmp behind in the export dir."""
    from moraine3d.cache.records import write_atomic
    from moraine3d.core.paths import cache_dir
    from moraine3d.engine.export import _write_out

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    cdir = cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    def deny(src, dst):
        raise PermissionError(13, "held open by another process")

    monkeypatch.setattr("moraine3d.core.atomic.os.replace", deny)

    # Cache path, destination exists: the lost rename is benign and the
    # temp file is cleaned up (through the deletion choke point).
    dest = cdir / ("0" * 24 + ".npz")
    real_replace_bytes = b"already-written identical bytes"
    dest.write_bytes(real_replace_bytes)
    write_atomic(dest, b"same-key bytes")
    assert dest.read_bytes() == real_replace_bytes
    assert not list(cdir.glob("*.tmp*"))

    # Cache path, destination missing: nothing to fall back on.
    with pytest.raises(MoraineError) as exc:
        write_atomic(cdir / ("1" * 24 + ".npz"), b"x")
    assert exc.value.code == "cache_unwritable"
    assert "MORAINE_CACHE_DIR" in exc.value.message
    assert not list(cdir.glob("*.tmp*"))

    # Non-content-addressed path (exports): always a loud failure, and
    # the in-flight tmp never lingers next to the user's export.
    exp = tmp_path / "asset.glb"
    exp.write_bytes(b"old export")
    with pytest.raises(MoraineError) as exc:
        _write_out(exp, b"new export")
    assert exc.value.code == "export_locked"
    assert "asset.glb" in exc.value.message and "close it" in exc.value.message
    assert exp.read_bytes() == b"old export"
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["asset.glb"]


def test_write_atomic_unwritable_cache_dir_is_actionable(tmp_path, monkeypatch):
    """A cache dir that exists but refuses the write itself (ACL,
    read-only mount, disk full) surfaces as cache_unwritable naming
    MORAINE_CACHE_DIR — the same error as the mkdir probe — not a raw
    OSError traceback."""
    from moraine3d.cache.records import write_atomic
    from moraine3d.core.paths import cache_dir

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    cdir = cache_dir()

    def deny(self, data):
        raise PermissionError(13, "access denied")

    monkeypatch.setattr("pathlib.Path.write_bytes", deny)
    with pytest.raises(MoraineError) as exc:
        write_atomic(cdir / ("2" * 24 + ".npz"), b"x")
    assert exc.value.code == "cache_unwritable"
    assert "MORAINE_CACHE_DIR" in exc.value.message


def test_write_atomic_reaped_tmp_is_cache_missing(tmp_path, monkeypatch):
    """A record whose in-flight tmp vanished before the rename (a
    concurrent full cache clear, which reaps tmp files without a
    freshness guard) is the same race records.load_value names: cache_missing,
    fix = re-run — never a raw FileNotFoundError."""
    from moraine3d.cache.records import write_atomic
    from moraine3d.core.paths import cache_dir

    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    cdir = cache_dir()

    def gone(src, dst):
        raise FileNotFoundError(2, "tmp reaped")

    monkeypatch.setattr("moraine3d.core.atomic.os.replace", gone)
    with pytest.raises(MoraineError) as exc:
        write_atomic(cdir / ("3" * 24 + ".npz"), b"x")
    assert exc.value.code == "cache_missing"
    assert "re-run" in exc.value.message


def test_concurrent_clear_reported_as_missing_not_corrupt(tmp_path, monkeypatch):
    """A record that vanishes between the hit check and the lazy load (a
    concurrent `cache clear`, which has no freshness guard) is a normal
    race, not corruption — the error names the actual cause and the fix
    is a plain re-run."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    sc = _asset([BOX], {"main": "bx01"})
    evaluate_records(sc)
    rec = evaluate_records(sc)["main"]  # hit: the value loads lazily
    for p in (tmp_path / "cache").glob("*.npz"):
        p.unlink()  # what a concurrent clear does behind our back
    with pytest.raises(MoraineError) as exc:
        rec.value()
    assert exc.value.code == "cache_missing"
    assert "clear" in exc.value.message


def test_packaged_text_files_are_lf():
    """The wheel's text payload — the source, the agent guide and
    reference, the workspace templates, and the example assets — is LF
    on every platform: `moraine init` copies the guide and templates
    verbatim (DESIGN 6), and an example asset's bytes are compared to
    its canonical text as written (DESIGN 4.1), so a carriage return in
    the package would ship a noncanonical asset and a CRLF workspace."""
    files = list(package_sources())
    files += sorted((SRC_ROOT / "templates").iterdir())
    files += sorted((SRC_ROOT / "docs").rglob("*.md"))
    files += sorted((SRC_ROOT / "docs" / "examples").rglob("*.star"))
    assert len(files) > 100
    crlf = [str(p.relative_to(root)) for p in files if b"\r" in p.read_bytes()]
    assert crlf == []
