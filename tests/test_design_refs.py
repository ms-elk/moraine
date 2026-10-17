"""DESIGN reference integrity (CLAUDE.md document rule 2).

Code, tests, and shipped docs cite only DESIGN's shipped-spec
chapters (1 through 11). Future Work holds no landed contract to
cite: the moment an implementation wants to cite an item there, that
item has landed, so its spec moves to the owning chapter first.
Future Work items are unnumbered and cited by name
(`Future Work: <item>`); every numeric citation must exist as a
DESIGN.md heading and every cited item name must resolve to a
Future Work heading, so references cannot rot silently.
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

from conftest import SRC_ROOT, package_sources

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "DESIGN.md"

# Explicit reference forms only ("DESIGN 4.5", "DESIGN §4.5", "§4.5");
# bare numbers in prose are not treated as references.
REF = re.compile(r"(?:DESIGN[ §]+|§)(\d+(?:\.\d+)?)")

LAST_SHIPPED_CHAPTER = 11

FUTURE_WORK_CITE = re.compile(r"Future Work: (?!<)")


def _shipped_docs():
    """Every non-Python surface that ships or fronts the product: the
    reference and guide (recursively), the workspace templates, the
    viewer client, the README, the root tour docs, and the skill."""
    yield from sorted((SRC_ROOT / "docs").rglob("*.md"))
    yield from sorted((SRC_ROOT / "templates").glob("*.md"))
    yield SRC_ROOT / "app" / "viewer" / "app.js"
    yield ROOT / "README.md"
    yield ROOT / "README.ja.md"
    yield ROOT / "docs" / "tour.md"
    yield ROOT / "docs" / "tour.ja.md"
    yield ROOT / "skills" / "moraine" / "SKILL.md"


def _scanned_files():
    yield from package_sources()
    yield from sorted((ROOT / "tests").glob("*.py"))
    yield from _shipped_docs()


def test_scanned_surfaces_exist():
    """The scan lists are pinned by name, so a moved or renamed surface
    fails here instead of silently leaving the scan."""
    missing = [p for p in _scanned_files() if not p.is_file()]
    assert not missing, [str(p.relative_to(ROOT)) for p in missing]


def _future_work_items() -> list[str]:
    text = DESIGN.read_text(encoding="utf-8")
    start = text.index("\n## Future Work\n")
    return re.findall(r"^### (.+)$", text[start:], re.M)


def test_design_refs_cite_existing_shipped_chapters():
    text = DESIGN.read_text(encoding="utf-8")
    chapters = set(re.findall(r"^## (\d+)\.", text, re.M))
    sections = set(re.findall(r"^### (\d+\.\d+)", text, re.M))
    problems = []
    paths = [*_scanned_files(), DESIGN, ROOT / "CLAUDE.md"]
    for path in paths:
        body = path.read_text(encoding="utf-8")
        for m in REF.finditer(body):
            num = m.group(1)
            where = f"{path.relative_to(ROOT)}:{body.count(chr(10), 0, m.start()) + 1}"
            if int(num.split(".")[0]) > LAST_SHIPPED_CHAPTER:
                problems.append(
                    f"{where}: cites DESIGN {num} — Future Work is"
                    f" unnumbered; cite the item by name, or the owning"
                    f" chapter once the spec has landed"
                )
            elif "." in num:
                if num not in sections:
                    problems.append(
                        f"{where}: cites DESIGN {num}, but DESIGN.md has"
                        f" no '### {num}' heading"
                    )
            elif num not in chapters:
                problems.append(
                    f"{where}: cites DESIGN {num}, but DESIGN.md has no"
                    f" '## {num}.' heading"
                )
    assert not problems, "\n".join(problems)


def test_future_work_cited_by_existing_item_name():
    """A `Future Work: <item>` citation (DESIGN, CLAUDE.md)
    must open with the exact name of a Future Work heading — names are
    the chapter's only anchors, so a renamed or deleted item fails here
    instead of rotting silently. The `<item>` grammar placeholder is
    exempt."""
    items = _future_work_items()
    assert items, "no Future Work headings found in DESIGN.md"
    problems = []
    paths = [DESIGN, ROOT / "CLAUDE.md"]
    for path in paths:
        body = path.read_text(encoding="utf-8")
        for m in FUTURE_WORK_CITE.finditer(body):
            tail = body[m.end():m.end() + 80]
            if not any(tail.startswith(name) for name in items):
                line = body.count("\n", 0, m.start()) + 1
                problems.append(
                    f"{path.relative_to(ROOT)}:{line}: 'Future Work:"
                    f" {tail[:40]}...' names no Future Work heading"
                )
    assert not problems, "\n".join(problems)


def test_no_future_work_citations_in_code():
    """Code, tests, and shipped docs never cite Future Work (document rule 2):
    a citable item has landed and is cited at its owning chapter."""
    problems = []
    for path in _scanned_files():
        if path.name == Path(__file__).name:
            continue  # the pattern definition would match itself
        body = path.read_text(encoding="utf-8")
        for m in FUTURE_WORK_CITE.finditer(body):
            line = body.count("\n", 0, m.start()) + 1
            problems.append(
                f"{path.relative_to(ROOT)}:{line}: cites Future Work —"
                f" no landed contract lives there"
            )
    assert not problems, "\n".join(problems)


# Provenance stays out of code and out of DESIGN (CLAUDE.md document rule 1):
# where a change came from — a phase, a review or benchmark finding, a
# feedback entry, a decision date — lives in commit messages, never in
# comments, docstrings, or the normative text. The sanctioned test citation is
# the claim pinned (a semantics sentence or DESIGN section), so DESIGN
# references pass and origin tags fail.
PROVENANCE = re.compile(
    r"Phase [0-9]"
    r"|[Ff]inding [0-9]"
    r"|Review finding"
    r"|feedback entry [0-9]"
    r"|\(entry [0-9]"
    r"|\bentry [0-9]+'s"
    r"|field test"
    r"|\bentry-[0-9]"
    r"|Stage [A-Z][ ,)]"
    r"|batch [A-Z][0-9]"
    r"|\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\b"
)


def _prose(path: Path) -> list[tuple[int, str]]:
    """(first line, text) chunks the provenance scan reads: for a
    Python file its comments and docstrings only — a string literal
    (a log-file fixture named by its day, a protocol version) is data,
    not a remark about the code; for every other surface, the file."""
    body = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return [(1, body)]
    chunks = [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(body).readline)
        if tok.type == tokenize.COMMENT
    ]
    for node in ast.walk(ast.parse(body)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        head = node.body[0] if node.body else None
        if (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        ):
            chunks.append((head.lineno, head.value.value))
    return chunks


def test_no_provenance_in_code():
    # _scanned_files' non-Python entries are the shipped docs and the
    # viewer client, so they are held to the same bar; DESIGN joins the
    # scan because document rule 1 bans the same tags from
    # its normative text.
    paths = [*_scanned_files(), DESIGN]
    problems = []
    for path in paths:
        if path.name == Path(__file__).name:
            continue  # the pattern definition would match itself
        for start, text in _prose(path):
            for m in PROVENANCE.finditer(text):
                line = start + text.count("\n", 0, m.start())
                problems.append(
                    f"{path.relative_to(ROOT)}:{line}: provenance"
                    f" {m.group(0)!r} belongs in the commit message, not"
                    f" in code"
                )
    assert not problems, "\n".join(problems)


GUIDE = SRC_ROOT / "docs" / "guide.md"
GUIDE_BUDGET = re.compile(r"budget: (\d+) lines")
# A backticked pointer, possibly wrapped across a line; the grammar
# placeholders (`<section>`, `…`) and the bare listing are not pointers.
GUIDE_DOCS_POINTER = re.compile(r"`moraine docs\s+([^`<…]+?)`")


def test_guide_docs_pointers_resolve(capsys):
    """Every `moraine docs <section>` pointer the guide carries (the
    routing rows and when-to-read lines of CLAUDE.md document rule 4) resolves
    through the command it names: a renamed or re-cut reference section
    fails here instead of sending the agent to `unknown_section`."""
    from moraine3d.app.cli import main

    text = GUIDE.read_text(encoding="utf-8")
    pointers = sorted({" ".join(m.group(1).split()) for m in GUIDE_DOCS_POINTER.finditer(text)})
    assert pointers, "guide.md carries no `moraine docs <section>` pointer"
    problems = []
    for section in pointers:
        code = main(["docs", section])
        out = capsys.readouterr().out
        if code != 0:
            problems.append(f"`moraine docs {section}`: {json.loads(out)['error']['message']}")
    assert not problems, "\n".join(problems)


def test_guide_holds_its_declared_budget():
    """The guide declares its own size ("budget: N lines", a comment
    under its preamble): the file may not
    exceed the budget it declares, so a feature landing cannot grow
    the guide silently — the addition displaces a line, moves to
    `moraine docs` or the error channel (CLAUDE.md document rule 4), or
    raises the declared budget as a reviewable diff."""
    text = GUIDE.read_text(encoding="utf-8")
    m = GUIDE_BUDGET.search(text)
    assert m, "guide.md declares no 'budget: N lines' in its preamble"
    budget = int(m.group(1))
    lines = text.count("\n")
    assert lines <= budget, (
        f"guide.md is {lines} lines against its declared budget of"
        f" {budget} — displace a line, move content to moraine docs or"
        f" the error channel, or raise the budget deliberately"
    )



def test_the_shipped_reference_is_ascii_and_names_no_design_section():
    """DESIGN 6: `moraine docs SECTION` prints a section's own text, so
    the reference is ASCII, which every console prints, and so are the
    example files and the generated listing it prints the same way.
    The design document is not shipped, so the reference a workspace
    agent reads carries no pointer into it."""
    from moraine3d.recipe import api

    docs = ROOT / "src" / "moraine3d" / "docs"
    problems = []
    for path in sorted((docs / "reference").glob("*.md")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.isascii():
                problems.append(f"{path.name}:{n}: non-ASCII")
            if "DESIGN" in line:
                problems.append(f"{path.name}:{n}: names DESIGN")
    for path in sorted((docs / "examples").rglob("*.star")):
        if not path.read_text(encoding="utf-8").isascii():
            problems.append(f"{path.relative_to(docs).as_posix()}: non-ASCII")
    if not api.text().isascii():
        problems.append("the generated api listing: non-ASCII")
    assert not problems, "\n".join(problems)
