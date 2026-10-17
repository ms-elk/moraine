"""`moraine docs` addresses every builtin by name and shows the recipe
spelling beside each contract (DESIGN 4.8, 6): the assert kinds are
sections of their own, each op and kind section opens with the call
the recipe writes, and `moraine docs api` lists the whole `mo`
namespace, all generated from the registries the prelude is."""

import json
import re

import pytest

from moraine3d import ops
from moraine3d.app.cli import main, reference_text
from moraine3d.core import asserts as vocab
from moraine3d.recipe import api, prelude

_HEAD = re.compile(r"^(#{2,3}) ([a-z_]+) \((op_impl_version \d+|assert kind)\)(, .*)?$", re.M)


def _docs(section, capsys):
    rc = main(["docs", section])
    return rc, capsys.readouterr().out


def test_every_name_under_mo_has_one_signature():
    """The generated listing covers the prelude's names exactly, in its
    order, and nothing the prelude lacks."""
    text = api.text()
    sigs = [l for l in text.splitlines() if l.startswith("mo.")]
    assert [s.split("(")[0][3:] for s in sigs] == prelude.names()
    for name in prelude.names():
        assert api.signature(name).startswith(f"mo.{name}(")
    assert api.signature("warp_drive") is None
    assert "math.atan2(...)" in text and "math.pi" in text


def test_signatures_follow_the_registries():
    """An op's signature names its inputs by arity, by role where the
    op declares one, and by keyword where the recipe passes them so,
    and every param it accepts, an integer param marked; an assert
    kind's names every key; the reference-op spelling is the
    two-position form."""
    assert api.operator("box") == "mo.box(id = ..., size = ..., segments = int)"
    assert api.operator("boolean") == "mo.boolean(a, b, ..., id = ..., mode = ...)"
    assert api.operator("mate").startswith("mo.mate(base, part, [leader], id = ..., ")
    assert api.operator("scatter").startswith("mo.scatter(target, prototype, id = ..., ")
    assert api.operator("material").startswith(
        "mo.material(a, base_color_texture = node, roughness_texture = node, "
        "metallic_texture = node, normal_texture = node, id = ..., name = ..., "
    )
    assert "slots" not in api.operator("material")
    assert api.operator("asset_ref").startswith("mo.asset_ref(asset, output, id = ..., ")
    for name in ops.names():
        sig = api.operator(name)
        for param in ops.get(name).PARAM_KINDS:
            if name == "asset_ref" and param in ("asset", "output", "project"):
                continue  # spelled by position (the project form is prose)
            if param == getattr(ops.get(name), "SLOT_PARAM", None):
                continue  # written by the build from the input keywords
            assert f"{param} = " in sig, (name, param)
    assert api.assertion("count") == (
        "mo.assert_count(id = ..., outputs = ..., value = int, min = int, max = int)"
    )
    for kind in vocab.AUTHORABLE_KINDS:
        sig = api.assertion(kind)
        for key in vocab.KINDS[kind].KEY_KINDS:
            assert f"{key} = " in sig, (kind, key)


def test_every_op_and_kind_section_opens_with_its_spelling():
    """Each op section and each assert-kind section carries the
    generated signature as its first line, so the spelling beside the
    contract is the build's own."""
    seen = set()
    for name in ("ops.md", "textures.md", "assertions.md"):
        text = reference_text(name)
        for m in _HEAD.finditer(text):
            section = m.group(2)
            after = text[m.end():].lstrip("\n")
            expected = f"Recipe: `{api.signature(section)}`"
            assert after.startswith(expected), (name, section, after[:120])
            seen.add(section)
    assert set(ops.names()) <= seen
    assert vocab.AUTHORABLE_KINDS <= seen


def test_docs_answers_each_kind_by_name(capsys):
    """`moraine docs mated` (and every kind) returns that kind's own
    section and names its examples; the shared rules stay under *Asset
    assertions*."""
    for kind in sorted(vocab.AUTHORABLE_KINDS):
        rc, reply = _docs(kind, capsys)
        assert rc == 0, kind
        assert reply.startswith(f"### {kind} (assert kind)\n\nRecipe: `mo.assert_{kind}(")
        assert reply.splitlines()[-1].startswith("examples: examples/"), kind
        assert "### " not in reply[4:]
    rc, reply = _docs("Asset assertions", capsys)
    assert rc == 0 and "quantifier" in reply.lower()
    assert '- `kind: "contact"`' not in reply


def test_docs_api_is_listed_and_generated(capsys):
    """`moraine docs api` prints the generated listing, and the index
    names it beside the recipe's file."""
    rc, reply = _docs("api", capsys)
    assert rc == 0 and reply == api.text() + "\n"
    assert main(["docs"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [d["doc"] for d in listing["docs"]][:3] == ["recipe.md", "api", "asset.md"]
    (title,) = listing["docs"][1]["sections"]
    assert _docs(title, capsys) == (rc, reply)


@pytest.mark.parametrize("name", ["box", "scatter", "material", "noise_texture"])
def test_docs_op_reply_still_carries_numbers(name, capsys):
    rc, reply = _docs(name, capsys)
    assert rc == 0 and reply.splitlines()[2].startswith("Recipe: `mo.")
    assert any(line.startswith("numbers: ") for line in reply.splitlines())
