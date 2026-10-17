"""Build tests (DESIGN 10): the round trip over the IR corpus, the
byte-identical rebuild, and the refusals that keep the boundary."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import IR_FIXTURES
from moraine3d.recipe.build import build
from moraine3d.recipe.version import build_version
from support import flatten
from support.recipes import build_recipe, ir, refusal

pytestmark = pytest.mark.real_stamp

EXAMPLES = IR_FIXTURES
PROJECTS = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())
GOLDEN = Path(__file__).parent / "golden" / "build_sums.json"


def _built(fixture: Path, tmp_path: Path):
    return build(flatten.write_project(fixture, tmp_path / fixture.name))


def _fixture_of(ir_path: Path, fixture: Path, project: Path) -> Path | None:
    """The fixture file an IR was built from; None for a variant, which
    no fixture file holds."""
    rel = ir_path.relative_to(project / "build")
    return None if rel.name.count(".") > 1 else fixture / rel


# -- round trip ----------------------------------------------------------------


@pytest.mark.parametrize("fixture", PROJECTS, ids=lambda p: p.name)
def test_the_flat_recipe_of_a_fixture_builds_to_the_fixture(fixture, tmp_path):
    """DESIGN 10: the translator turns an IR fixture into a flat recipe
    whose build is equivalent to the fixture: the same node ids,
    outputs, groups, params, asserts, checks, and budgets."""
    result = _built(fixture, tmp_path)
    project = (tmp_path / fixture.name).resolve()
    compared = 0
    for ir_path, built in result.assets.items():
        source = _fixture_of(ir_path, fixture, project)
        if source is None:
            continue
        doc = json.loads(source.read_text(encoding="utf-8"))
        assert flatten.differences(doc, json.loads(built.text)) == []
        compared += 1
    assert compared >= 1


@pytest.mark.parametrize(
    "ir_file", sorted(EXAMPLES.rglob("*.mora")),
    ids=lambda p: p.relative_to(EXAMPLES).as_posix(),
)
def test_the_corpus_is_in_canonical_form(ir_file):
    """DESIGN 4.1: one node per line, fixed key order — a fixture is
    already in canonical form, so `dumps(loads(text)) == text` and the
    corpus shows the form the build writes."""
    from moraine3d.core import asset as asset_mod

    text = ir_file.read_text(encoding="utf-8")
    assert asset_mod.dumps(asset_mod.loads(text)) == text


def test_the_corpus_covers_the_vocabulary():
    """DESIGN 10: the IR corpus proves the builtins cover the IR. What
    the corpus does not reach is named here, so a gap is a decision
    and not an oversight."""
    from moraine3d import ops
    from moraine3d.core import asserts as vocab
    from moraine3d.core import asset as asset_mod

    used_ops, used_kinds, used_keys = set(), set(), set()
    for mora in EXAMPLES.rglob("*.mora"):
        doc = json.loads(mora.read_text(encoding="utf-8"))
        used_ops |= {n["op"] for n in doc["nodes"]}
        used_kinds |= {a["kind"] for a in doc.get("asserts") or []}
        used_keys |= set(doc)
    assert set(ops.names()) - used_ops == set()
    assert set(vocab.AUTHORABLE_KINDS) - used_kinds == set()
    # No shipped example waives an asset-level check; the declaration's
    # own test builds one (test_recipe_builtins).
    assert set(asset_mod.ROOT_KEY_ORDER) - used_keys == {"asset_checks"}


def test_the_translator_orders_definitions_before_uses():
    """DESIGN 10: a flat recipe has one call per node, ids kept,
    definitions ordered before their uses."""
    nodes = [
        {"id": "top", "op": "transform", "inputs": ["mid"], "params": {}},
        {"id": "mid", "op": "transform", "inputs": ["base"], "params": {}},
        {"id": "other", "op": "box", "params": {}},
        {"id": "base", "op": "box", "params": {}},
    ]
    assert [n["id"] for n in flatten.ordered(nodes)] == ["base", "mid", "top", "other"]


def test_a_knob_is_wired_into_the_call_that_consumes_it(tmp_path):
    """DESIGN 4.6: an override is a value handed to the referenced
    asset's build; the flat recipe of a fixture hands the knob's value
    to the node param the fixture names."""
    doc = {
        "build_version": "fixture",
        "nodes": [{"id": "bx", "op": "box", "params": {"size": [1.0, 2.0, 1.0]}}],
        "params": [{"id": "size", "value": [1.0, 2.0, 1.0], "node": "bx",
                    "param": "size", "min": 0.5, "max": 4.0}],
        "outputs": {"main": "bx"},
    }
    target = flatten.write_doc(doc, tmp_path, "asset")
    assert "size = p_size" in target.read_text(encoding="utf-8")
    offered = build(target, offered={"size": [3.0, 3.0, 3.0]})
    assert ir(offered)["nodes"][0]["params"]["size"] == [3.0, 3.0, 3.0]


def test_the_file_decides_the_owner_in_translation(tmp_path):
    """DESIGN 5.2: every assert carries its owner in the IR, and the
    owner is the file; the translator puts a human-owned entry and a
    budget where they keep that owner."""
    fixture = EXAMPLES / "shelf_pair"
    doc = json.loads((fixture / "shelf_pair.mora").read_text(encoding="utf-8"))
    owners = {a["id"]: a["owner"] for a in doc["asserts"]}
    result = _built(fixture, tmp_path)
    assert {a["id"]: a["owner"] for a in ir(result)["asserts"]} == owners
    requirements = (tmp_path / "shelf_pair" / "shelf_pair.requirements.star")
    humans = [i for i, o in owners.items() if o == "human"]
    assert requirements.is_file() == bool(humans or doc.get("budgets"))


# -- rebuild -------------------------------------------------------------------


def _sums(tmp_path: Path) -> dict[str, str]:
    """sha256 of every IR the corpus builds to, by its path under the
    corpus. The stamp's line is left out of the hash: it moves with the
    vocabulary, and its own test pins it."""
    out = {}
    for fixture in PROJECTS:
        result = _built(fixture, tmp_path)
        for ir_path, built in result.assets.items():
            lines = built.text.split("\n")
            assert lines[1] == f'"build_version": "{build_version()}",'
            body = "\n".join(lines[:1] + lines[2:])
            rel = ir_path.relative_to(tmp_path.resolve()).as_posix()
            out[rel] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return dict(sorted(out.items()))


def test_the_same_recipe_builds_byte_identical_twice(tmp_path):
    """DESIGN 10: the same recipe builds byte-identical twice. DESIGN
    4.4: the build is a pure function of the recipe files and the tool
    version."""
    assert _sums(tmp_path / "a") == _sums(tmp_path / "b")


def test_the_build_does_not_depend_on_the_hash_seed(tmp_path):
    """DESIGN 4.4: identical inputs produce byte-identical IR; nothing
    the build orders depends on a process's hash seed."""
    fixture = EXAMPLES / "tower_yard"
    target = flatten.write_project(fixture, tmp_path / "tower_yard")
    script = (
        "import hashlib, json, sys\n"
        "from moraine3d.recipe.build import build\n"
        "r = build(sys.argv[1], write=False)\n"
        "print(json.dumps(sorted(hashlib.sha256(b.text.encode()).hexdigest()"
        " for b in r.assets.values())))\n"
    )
    seen = set()
    for seed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", script, str(target)],
            capture_output=True, text=True, env=env, check=True,
        )
        seen.add(out.stdout.strip())
    assert len(seen) == 1


def test_the_corpus_builds_to_the_frozen_bytes(tmp_path):
    """DESIGN 10: the same recipe builds byte-identical across the two
    platforms. The sums are frozen in the repository, so the suite
    passing on Windows and on macOS is the comparison."""
    sums = _sums(tmp_path)
    if os.environ.get("MORAINE_REGEN_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        with open(GOLDEN, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(sums, indent=1) + "\n")
    assert sums == json.loads(GOLDEN.read_text(encoding="utf-8"))


# -- refusals ------------------------------------------------------------------


def test_no_builtin_returns_an_evaluated_value(tmp_path):
    """DESIGN 4.8: a recipe never observes an evaluated value (a bound,
    a count, a face frame), and no builtin returns one. DESIGN 11,
    Invariant 10."""
    for expr, said in (
        ("node.bounds", "has no attribute"),
        ("node.id", "has no attribute"),
        ("node + 1", "not supported"),
        ("len(node)", "not supported"),
        ("float(node)", ""),
    ):
        err = refusal(tmp_path / "v", f'node = mo.box(id = "a", size = [1, 1, 1])\nx = {expr}\n')
        assert err.code == "recipe_error" and said in err.message
        assert err.detail["site"] == "asset.star:2"
    result = build_recipe(tmp_path / "d", '''node = mo.box(id = "a", size = [1, 1, 1])
made = [mo.output("a", node), mo.group("g", ["a", "b"]), mo.waive_asset("grounded"),
        mo.assert_count(id = "one", outputs = ["a"], value = 1)]
mo.output("b", mo.box(id = "b", size = [1, 1, 1]))
ok = made == [None, None, None, None]
mo.output("c", mo.box(id = "c", size = [1 if ok else 2, 1, 1]))
''')
    assert ir(result)["nodes"][-1]["params"]["size"][0] == 1


def test_the_build_runs_before_any_evaluation(tmp_path, monkeypatch):
    """DESIGN 4.8: the build needs no evaluator and runs before any
    evaluation; an asset no operator could evaluate still builds."""
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    build_recipe(tmp_path / "p", 'mo.output("a", mo.box(id = "a", size = "not a size"))\n')
    assert not (tmp_path / "cache").exists()


def test_a_human_threshold_cannot_be_built_from_the_recipe(tmp_path):
    """DESIGN 5.2: a requirements file loads nothing and no builtin in
    it returns a recipe value, so its thresholds can only be its own
    constants and the arithmetic it writes over them."""
    asset = 'W = 2.0\nmo.output("slab", mo.box(id = "slab", size = [W, 0.1, W]))\n'
    for line, code in (
        ('load("asset.star", "W")', "requirements_load"),
        ('load("lib/dims.star", "W")', "requirements_load"),
        ('w = mo.param("w", 2.0, min = 1.0, max = 3.0)', "requirements_builtin"),
        ('w = mo.box(id = "probe", size = [1, 1, 1])', "requirements_builtin"),
        ("w = W", "unknown_name"),
    ):
        err = refusal(tmp_path / code / str(len(line)), asset, files={
            "asset.requirements.star": f"{line}\n", "lib/dims.star": "W = 2.0\n",
        })
        assert err.code == code
        assert err.detail["site"] == "asset.requirements.star:1"
