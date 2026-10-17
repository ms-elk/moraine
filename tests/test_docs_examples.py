"""The example assets live in the package and print from `moraine docs`
(DESIGN 5: an operator's example doubles as usage documentation; DESIGN
6: the docs command).

The reference cites `examples/<name>`; every citation resolves to a
shipped asset, every op section cites one, and the section reply
carries the asset — so a workspace agent reads the contract and a use
of it in one reply. A selection idiom the shipped assets teach — a tag
only the intended feature can satisfy — is pinned by removing the
feature and watching the entry fail.
"""

import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from conftest import IR_FIXTURES
from moraine3d.app.cli import (
    EXAMPLE_CITATION, REFERENCE_FILES, example_assets, example_text, main, reference_text,
)
from moraine3d.core import asset as asset_mod
from support import flatten

def _docs(*args) -> tuple[int, dict | str]:
    """The reply: a section's text, or the object of the listing and
    of an error."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["docs", *args])
    out = buf.getvalue()
    return rc, json.loads(out) if out.startswith("{") else out


def test_every_citation_resolves_and_every_op_section_cites():
    """Each `examples/<name>` the reference cites is a shipped project,
    and each op section (`<name> (op_impl_version N)`) cites at least
    one — the usage-documentation promise, section by section."""
    shipped = {p.split("/")[1] for p in example_assets()}
    text = "\n".join(reference_text(f) for f in REFERENCE_FILES)
    cited = set(EXAMPLE_CITATION.findall(text))
    assert cited and cited <= shipped, cited - shipped
    for f in REFERENCE_FILES:
        t = reference_text(f)
        heads = [(m.start(), m.group(1)) for m in re.finditer(r"^#{2,3} (\S+).*$", t, re.M)]
        for k, (pos, name) in enumerate(heads):
            end = heads[k + 1][0] if k + 1 < len(heads) else len(t)
            body = t[pos:end]
            if re.match(r"^#{2,3} [a-z_]+ \(op_impl_version", body):
                assert EXAMPLE_CITATION.search(body), f"{f}: section {name} cites no example"


def test_docs_section_reply_carries_the_cited_assets():
    """`moraine docs <section>` names the shipped examples the section
    cites, in citation order, once each, as the next call asks for
    them; the asset is not printed with the contract, and a section
    citing none carries no line."""
    rc, reply = _docs("scatter")
    assert rc == 0
    assert reply.splitlines()[-1] == (
        "examples: examples/scatter_field  (moraine docs examples/<name>)"
    )
    assert "mo.output(" not in reply
    rc, asset = _docs("examples/scatter_field")
    assert rc == 0 and "mo.scatter(" in asset
    rc, reply = _docs("Asset assertions")
    assert reply.splitlines()[-1].startswith(
        "examples: examples/shelf_pair, examples/scatter_field, examples/pallet  "
    )
    rc, reply = _docs("checks")
    assert rc == 0 and "examples: " not in reply


def test_docs_examples_index_and_examples_slash_name():
    """`moraine docs examples` is the index section naming every shipped
    project; `moraine docs examples/<name>` prints that project's root
    asset, `examples/<name>/subassets/<sub>` a sub-asset; an unknown
    name lists the projects and hints near misses."""
    rc, reply = _docs("examples")
    assert rc == 0 and reply.startswith("## Examples")
    for p in example_assets():
        assert f"`{p.split('/')[1]}`" in reply
    rc, reply = _docs("examples/shelf_pair")
    assert rc == 0
    assert reply == example_text("examples/shelf_pair/shelf_pair.star").rstrip() + "\n"
    # The human's file beside the root asset prints by name too.
    rc, reply = _docs("examples/shelf_pair/requirements")
    assert rc == 0 and "mo.budget(materials = 2)" in reply
    assert 'mo.group("shelf_a", ["shelf_a_posts", "shelf_a_boards"])' in reply
    rc, reply = _docs("examples/box/requirements")
    assert rc == 1 and reply["error"]["code"] == "unknown_example"
    rc, reply = _docs("examples/tower_yard/subassets/water_tower")
    assert rc == 0
    assert reply == example_text("examples/tower_yard/subassets/water_tower.star").rstrip() + "\n"
    rc, reply = _docs("examples/shelf")
    assert rc == 1 and reply["error"]["code"] == "unknown_example"
    assert reply["error"]["detail"]["did_you_mean"] == ["shelf_pair", "shelf_row"]
    assert "shelf_pair" in reply["error"]["detail"]["examples"]
    # The listing names both forms.
    rc, listing = _docs()
    assert "examples/<name>" in listing["usage"]


def _validate(project: Path) -> tuple[int, dict]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["validate", str(project)])
    return rc, json.loads(buf.getvalue())


@pytest.mark.parametrize("band", [True, False], ids=["feature-bound", "vacuous"])
def test_removing_the_feature_fails_a_feature_bound_selection(tmp_path, band):
    """The selection idiom `examples/shelf_pair` carries, pinned on its
    IR fixture (DESIGN 5.2 —
    what a green entry proves): `cap_top` names the post cap by a band
    only the cap reaches (`center_y > 0.81`), so building the post
    without its cap turns the entry red (`empty_selection`); the same
    tag written by normal alone would keep matching the bare post's
    top and stay green — the vacuous selection the reviewer's question
    exists to catch."""
    doc = json.loads((IR_FIXTURES / "shelf_pair" / "shelf_pair.mora").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in doc["nodes"]}
    nodes["ps04"]["inputs"] = ["ps02"]  # the post without its cap
    if not band:
        nodes["ps04"]["params"]["select"] = {"normal_y": {"gt": 0.9}}
    proj = tmp_path / "shelf_pair"
    proj.mkdir()
    (proj / "shelf_pair.mora").write_text(
        asset_mod.dumps(asset_mod.loads(json.dumps(doc))), encoding="utf-8", newline="\n"
    )
    flatten.write_project(proj, proj)
    rc, report = _validate(proj)
    if band:
        assert rc == 1
        assert report["error"]["code"] == "empty_selection"
        assert report["error"]["detail"]["node"] == "ps04"  # the cap_top tag
    else:
        # Green on the wrong feature: the extent then reads the bare post
        # top (0.8) against the cap's 0.82 and is the only thing that fails.
        failing = [e["assert"] for e in report["asserts"] if not e["ok"]]
        assert failing == ["cap_height"]
