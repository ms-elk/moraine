"""The stamp at the door (DESIGN 4.1): the runtime reads the IR its
own build wrote, and refuses any other."""

import json

import pytest

from moraine3d.app.cli import main
from moraine3d.core import asset as asset_mod
from moraine3d.core.sources import recipe_of
from moraine3d.engine.evaluator import Evaluation
from moraine3d.errors import MoraineError
from moraine3d.recipe.build import build
from moraine3d.recipe.version import build_version
from support.recipes import write_files

pytestmark = pytest.mark.real_stamp

PART = 'mo.output("p", mo.transform(mo.box(id = "b", size = [1, 1, 1]), id = "p", translate = [0, 0.5, 0]))\n'
ASSET = 'mo.output("p", mo.asset_ref("subassets/part.star", "p", id = "p"))\n'


def _restamp(path, stamp: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert build_version() in text
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace(build_version(), stamp))


def _error(argv, capsys) -> dict:
    assert main(argv) == 1
    return json.loads(capsys.readouterr().out)["error"]


def test_an_ir_another_build_wrote_is_refused(tmp_path, capsys):
    """DESIGN 4.1: a file carrying any other stamp is stale, and the
    loader refuses it (`version_mismatch`) naming `moraine build`."""
    project = write_files(tmp_path / "yard", {"yard.star": PART})
    ir = build(project / "yard.star").target.ir_path
    assert main(["stats", str(ir)]) == 0
    capsys.readouterr()
    _restamp(ir, "0+older")
    err = _error(["stats", str(ir)], capsys)
    assert err["code"] == "version_mismatch"
    assert (err["detail"]["found"], err["detail"]["expected"]) == ("0+older", build_version())
    assert "moraine build" in err["message"]
    # The project itself is rebuilt by any command, which is the fix.
    assert main(["stats", str(project)]) == 0
    assert build_version() in ir.read_text(encoding="utf-8")


def test_a_stale_referenced_ir_is_refused_where_it_is_opened(tmp_path, monkeypatch, capsys):
    """DESIGN 4.1, 4.6: an IR handed over by path opens its references
    from disk, and a referenced IR another build wrote is refused with
    the node that names it."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET, "subassets/part.star": PART,
    })
    result = build(project / "yard.star")
    _restamp(project / "build" / "subassets" / "part.mora", "0+older")
    err = _error(["stats", str(result.target.ir_path)], capsys)
    assert err["code"] == "version_mismatch"
    assert err["detail"]["node"] == "p"
    assert err["detail"]["file"] == "subassets/part.star"


def test_an_evaluation_refuses_a_referenced_ir_another_build_wrote(tmp_path, monkeypatch):
    """DESIGN 4.1: a file carrying any other stamp is stale, and the
    loader refuses it; an evaluation is given the stamp to hold the IRs
    it opens to, so none is constructed without one."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET, "subassets/part.star": PART,
    })
    result = build(project / "yard.star")
    _restamp(project / "build" / "subassets" / "part.mora", "0+older")
    with pytest.raises(MoraineError) as e:
        Evaluation(
            asset_mod.load(result.target.ir_path),
            source_of=recipe_of, stamp=build_version(),
        ).records()
    assert e.value.code == "version_mismatch"
    assert (e.value.detail["found"], e.value.detail["expected"]) == ("0+older", build_version())
    assert e.value.detail["node"] == "p"
    assert "run 'moraine build'" in e.value.message
    with pytest.raises(TypeError):
        Evaluation(asset_mod.load(result.target.ir_path))


def test_a_tree_another_build_wrote_whole_is_refused(tmp_path, monkeypatch):
    """DESIGN 4.1: the runtime reads the IR its own build wrote, so a
    referenced IR is held to the running build's stamp and not to the
    stamp of the asset that references it."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": ASSET, "subassets/part.star": PART,
    })
    root = build(project / "yard.star").target.ir_path
    for ir in (root, project / "build" / "subassets" / "part.mora"):
        _restamp(ir, "0+older")
    with pytest.raises(MoraineError) as e:
        Evaluation(
            asset_mod.load(root), source_of=recipe_of, stamp=build_version(),
        ).records()
    assert e.value.code == "version_mismatch"
    assert e.value.detail["file"] == "subassets/part.star"


def test_an_ir_with_no_recipe_is_named_as_one_to_read(tmp_path, capsys):
    """DESIGN 4.1, 8.2: an error names the shortest path to a fix; an
    IR no recipe sits beside cannot be rebuilt, so the refusal does not
    name a rebuild."""
    from conftest import IR_FIXTURES

    example = IR_FIXTURES / "box" / "box.mora"
    err = _error(["validate", str(example)], capsys)
    assert err["code"] == "version_mismatch"
    assert "run 'moraine build'" not in err["message"]
    assert "moraine docs recipe" in err["message"]


LAMP = (
    'h = mo.param("height", 2.0, min = 1.0, max = 4.0)\n'
    'mo.output("post", mo.transform(mo.box(id = "b", size = [0.1, h, 0.1]), '
    'id = "post", translate = [0, h / 2, 0]))\n'
)


def _variant(tmp_path):
    """A closure whose root and whose sibling read one variant of
    `lamp`: the root's IR and the variant."""
    from moraine3d.ops import asset_ref as ref_op
    from support.recipes import workspace

    root = workspace(tmp_path / "ws", {
        "lamp": {"lamp.star": LAMP},
        "station": {"station.star": (
            'mo.output("lamp", mo.asset_ref(project = "lamp", output = "post", '
            'id = "lamp", overrides = {"height": 3.0}))\n'
        )},
        "yard": {"yard.star": (
            'mo.output("a", mo.asset_ref(project = "station", output = "lamp", id = "a"))\n'
            'mo.output("b", mo.asset_ref(project = "lamp", output = "post", '
            'id = "b", overrides = {"height": 3}))\n'
        )},
    })
    ir = build(root / "yard" / "yard.star").target.ir_path
    identity = ref_op.override_identity({"height": 3.0})
    return ir, root / "lamp" / "build" / f"lamp.{identity}.mora"


def test_a_variant_is_held_to_the_stamp(tmp_path, monkeypatch):
    """DESIGN 4.1, 4.6: a variant is an IR like any other, and one
    carrying another stamp is refused at the reference that opens
    it."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ir, variant = _variant(tmp_path)
    _restamp(variant, "0+older")
    with pytest.raises(MoraineError) as e:
        Evaluation(
            asset_mod.load(ir), source_of=recipe_of, stamp=build_version(),
        ).records("b")
    assert e.value.code == "version_mismatch"
    assert (e.value.detail["node"], e.value.detail["file"]) == ("b", "lamp/lamp.star")


def test_a_failure_inside_a_variant_names_the_asset_it_was_built_from(tmp_path, monkeypatch):
    """DESIGN 4.6, 8.2: node ids are per file, so a failure raised
    while a variant is evaluated names the asset the variant was built
    from, and its line is that asset's."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ir, variant = _variant(tmp_path)
    text = variant.read_text(encoding="utf-8")
    assert text.count('"size": [0.1, 3.0, 0.1]') == 1
    with open(variant, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace('"size": [0.1, 3.0, 0.1]', '"size": [0.1, 1e999, 0.1]'))
    with pytest.raises(MoraineError) as e:
        Evaluation(
            asset_mod.load(ir), source_of=recipe_of, stamp=build_version(),
        ).records("b")
    assert e.value.code == "bad_param"
    assert (e.value.detail["asset"], e.value.detail["node"], e.value.detail["site"]) == (
        "lamp/lamp.star", "b", "lamp/lamp.star:2",
    )


def test_a_stale_ir_below_a_referenced_asset_names_that_asset(tmp_path, monkeypatch, capsys):
    """DESIGN 4.6, 8.2: node ids are per file, so a refusal raised at a
    reference inside a referenced asset names that asset, and its line
    is that asset's."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    project = write_files(tmp_path / "yard", {
        "yard.star": (
            'mo.output("q", mo.box(id = "p", size = [1, 1, 1]))\n'
            'mo.output("p", mo.asset_ref("subassets/gate.star", "p", id = "gate"))\n'
            'mo.waive_asset("grounded")\n'
        ),
        "subassets/gate.star": (
            'x = 1\n'
            'mo.output("p", mo.asset_ref("subassets/part.star", "p", id = "p"))\n'
        ),
        "subassets/part.star": PART,
    })
    result = build(project / "yard.star")
    _restamp(project / "build" / "subassets" / "part.mora", "0+older")
    err = _error(["validate", str(result.target.ir_path)], capsys)
    assert err["code"] == "version_mismatch"
    assert (err["detail"]["asset"], err["detail"]["node"]) == ("subassets/gate.star", "p")
    assert err["detail"]["site"] == "subassets/gate.star:2"


def test_an_overridden_project_reference_opens_its_variant_by_name(tmp_path, monkeypatch):
    """DESIGN 4.6: the evaluator uses `overrides` to derive the
    identity and open the variant beside the IR `asset` names, so the
    variant is all an overridden reference reads, and a reference the
    build has not given an `asset` names no file."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    ir, variant = _variant(tmp_path)
    variant.with_name("lamp.mora").unlink(missing_ok=True)
    records = Evaluation(
        asset_mod.load(ir), source_of=recipe_of, stamp=build_version(),
    ).records()
    assert records["b"].result_hash == records["a"].result_hash
    text = ir.read_text(encoding="utf-8")
    assert text.count('"asset": "lamp.mora", ') == 1
    with open(ir, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace('"asset": "lamp.mora", ', ""))
    with pytest.raises(MoraineError) as e:
        Evaluation(
            asset_mod.load(ir), source_of=recipe_of, stamp=build_version(),
        ).records()
    assert (e.value.code, e.value.detail["node"]) == ("ref_unresolved", "b")
