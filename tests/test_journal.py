"""Project journal index parsing and the `moraine init` retirement pass
(DESIGN 4.7 format, DESIGN 6 report)."""

from __future__ import annotations

import json
from pathlib import Path

from moraine3d.core import journal
from moraine3d.app.cli import main

SAMPLE = """# kiosk — journal

A market kiosk. Free prose before the anchors.

## Entries
### sizing-rationale
- [moraine 0.7.2] Counter height 0.9 — the standing-use reference. -> #counter
- [moraine 0.7.1, workaround] `smooth_normals` angle must exceed 360/segments.
- [moraine 0.7.2, WORKAROUND] When the watch dies, restart serve. → #watch
### rejected-variants
- [moraine 0.7.2] A separate output for the roof edge was rejected.
- An old-format line (no stamp).
  - nested bullet, never counted
* [moraine 0.7.0] asterisk bullets count too

## Details
### counter
Measurements and the rejected variants.
- [moraine 0.0.1, workaround] a stamped bullet inside Details is body text, not an entry
### watch
Body.
### orphan
no entry points here
"""


def test_parse_entries_kinds_pointers_and_unstamped():
    """DESIGN 4.7: one entry per column-0 bullet under Entries, opening
    with `[moraine <version>]` or `[moraine <version>, workaround]`; a
    trailing `-> #slug` / `→ #slug` names its body; other column-0
    bullets are unstamped; nothing under Details is an entry."""
    parsed = journal.parse(SAMPLE)
    assert [(e.line, e.version, e.kind) for e in parsed.entries] == [
        (7, "0.7.2", "permanent"),
        (8, "0.7.1", "workaround"),
        (9, "0.7.2", "workaround"),
        (11, "0.7.2", "permanent"),
        (14, "0.7.0", "permanent"),
    ]
    assert [e.pointer_slug for e in parsed.entries] == [
        "counter", None, "watch", None, None
    ]
    assert all(e.pointer_path is None for e in parsed.entries)
    assert parsed.unstamped == 1  # the nested bullet is not an entry candidate
    assert parsed.detail_slugs == ["counter", "watch", "orphan"]
    assert parsed.all_slugs == ["sizing-rationale", "rejected-variants", "counter", "watch", "orphan"]


def test_legacy_layout_without_anchors_still_parses():
    """Pre-format journals (no Entries/Details anchors) still scan: every
    stamped bullet is an entry, every other column-0 bullet unstamped."""
    text = "# old\n\n## constraints that shape the build\n- root params are one node.\n- [moraine 0.6.0, workaround] old limit.\n"
    parsed = journal.parse(text)
    assert len(parsed.entries) == 1 and parsed.entries[0].kind == "workaround"
    assert parsed.unstamped == 1


def test_scan_file_dangling_both_directions_and_sibling_pointers(tmp_path):
    """A pointer without its heading, a duplicate slug (ambiguous), and a
    Details heading no entry points to are each dangling; a
    `-> <relative>.md#slug` pointer resolves against a sibling file."""
    (tmp_path / "journal.md").write_text(SAMPLE, encoding="utf-8")
    rep = journal.scan_file(tmp_path / "journal.md")
    assert rep.dangling == 1  # `orphan`
    text = (
        "## Entries\n"
        "- [moraine 1.0] missing body -> #nowhere\n"
        "- [moraine 1.0] ambiguous -> #dup\n"
        "- [moraine 1.0] sibling ok -> journal/topic.md#remote\n"
        "- [moraine 1.0] sibling missing -> journal/topic.md#absent\n"
        "- [moraine 1.0] sibling file missing -> journal/none.md#x\n"
        "## Details\n### dup\n### dup\n"
    )
    (tmp_path / "journal.md").write_text(text, encoding="utf-8")
    (tmp_path / "journal").mkdir()
    (tmp_path / "journal" / "topic.md").write_text("### remote\nbody\n", encoding="utf-8")
    rep = journal.scan_file(tmp_path / "journal.md")
    # nowhere, dup (count 2), absent, none.md → 4; both `dup` headings are
    # pointed at, so they are not orphans.
    assert rep.dangling == 4


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    (ws / "journal.md").write_text(
        "## Entries\n- [moraine 0.1.0, workaround] root-level stale\n", encoding="utf-8"
    )
    for name in ("beta", "alpha"):
        (ws / name).mkdir()
        (ws / name / f"{name}.star").write_text("", encoding="utf-8")
    (ws / "alpha" / "journal.md").write_text(SAMPLE, encoding="utf-8")
    # beta has no journal; a non-project dir is never scanned
    (ws / "assets").mkdir()
    (ws / "assets" / "journal.md").write_text("- [moraine 0.0.0, workaround] never\n", encoding="utf-8")
    return ws


def test_scan_lists_stale_workarounds_with_relative_paths(tmp_path):
    """DESIGN 6: the pass lists workaround entries whose stamp differs
    from the installed version, root journal first then projects in
    sweep order, with workspace-relative paths and counts."""
    ws = _workspace(tmp_path)
    review, counts = journal.scan(ws, "0.7.2")
    assert [(r["journal"], r["line"], r["version"]) for r in review] == [
        ("journal.md", 2, "0.1.0"),
        ("alpha/journal.md", 8, "0.7.1"),
    ]
    assert review[1]["entry"].startswith("`smooth_normals`")
    assert counts == {"scanned": 2, "entries": 6, "unstamped": 1,
                      "dangling": 1, "skipped": 0,
                      "unstamped_by_journal": {"alpha/journal.md": 1}}
    # A different installed version flips which entries are stale —
    # stateless: the comparison is stamp != installed, nothing persisted.
    review2, _ = journal.scan(ws, "0.7.1")
    assert [r["line"] for r in review2] == [2, 9]


def test_init_reports_journal_review(tmp_path, capsys, monkeypatch):
    """`moraine init` runs the pass read-only: journals are byte-identical
    afterwards, the report carries `version`, `journal_review`, and
    `journals`, and an unreadable journal degrades per journal (DESIGN
    6): named on stderr, counted as skipped, the others still scanned."""
    ws = _workspace(tmp_path)
    before = (ws / "alpha" / "journal.md").read_bytes()
    monkeypatch.setattr("moraine3d.app.cli.dist_version", lambda: "0.7.2")
    assert main(["init", str(ws)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "0.7.2"
    assert [r["journal"] for r in out["journal_review"]] == ["journal.md", "alpha/journal.md"]
    assert out["journals"]["scanned"] == 2 and out["journals"]["dangling"] == 1
    assert (ws / "alpha" / "journal.md").read_bytes() == before

    (ws / "journal.md").write_bytes(b"\xff\xfe not utf-8 \x80")
    assert main(["init", str(ws)]) == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert [r["journal"] for r in out["journal_review"]] == ["alpha/journal.md"]
    assert out["journals"]["scanned"] == 1 and out["journals"]["skipped"] == 1
    assert "journal skipped: journal.md" in captured.err


def test_bom_journal_keeps_its_first_line(tmp_path):
    """A UTF-8 BOM written by an editor must not eat line 1: a stamped
    entry there is parsed and counted like any other (DESIGN 4.7)."""
    text = "\ufeff- [moraine 0.1.0, workaround] first line -> #a\n## Details\n### a\n"
    (tmp_path / "journal.md").write_text(text, encoding="utf-8")
    rep = journal.scan_file(tmp_path / "journal.md")
    assert [(e.line, e.kind) for e in rep.parsed.entries] == [(1, "workaround")]
    assert rep.dangling == 0


def test_fenced_blocks_are_content_not_structure(tmp_path):
    """Headings and bullets inside a ``` fence (a journal-layout example
    quoted in a journal) are body text: they open no section, define no
    slug, and count as no entry — the same fence rule `moraine docs`
    applies to the shipped reference."""
    text = (
        "## Entries\n"
        "- [moraine 1.0] real -> #a\n"
        "```\n"
        "## Details\n"
        "### a\n"
        "- [moraine 0.0.1, workaround] quoted example\n"
        "- quoted unstamped\n"
        "```\n"
        "- [moraine 1.0] still an entry\n"
        "## Details\n"
        "### a\n"
    )
    parsed = journal.parse(text)
    assert [e.line for e in parsed.entries] == [2, 9]
    assert parsed.unstamped == 0
    assert parsed.detail_slugs == ["a"] and parsed.all_slugs == ["a"]
    (tmp_path / "journal.md").write_text(text, encoding="utf-8")
    assert journal.scan_file(tmp_path / "journal.md").dangling == 0


def test_unreadable_sibling_pointer_is_dangling(tmp_path):
    """A sibling body file that is not UTF-8 resolves like a missing one:
    the pointer is dangling, the journal itself still scans."""
    (tmp_path / "journal.md").write_text(
        "## Entries\n- [moraine 1.0] body -> notes.md#x\n", encoding="utf-8"
    )
    (tmp_path / "notes.md").write_bytes(b"\xff\xfe### x\n")
    rep = journal.scan_file(tmp_path / "journal.md")
    assert len(rep.parsed.entries) == 1 and rep.dangling == 1


def test_workaround_claim_key_parses_and_groups(tmp_path):
    """DESIGN 4.7: `[moraine <v>, workaround: <key>]` names the tool
    limit; the retirement pass collapses same-key entries — across
    journals — into one row per limit, sitting where the key first
    appears, while keyless entries stay one row each."""
    parsed = journal.parse(
        "## Entries\n"
        "- [moraine 0.7.1, workaround: seam-weld] rail A workaround.\n"
        "- [moraine 0.7.1, workaround:seam-weld ] rail B workaround.\n"
        "- [moraine 0.7.1, workaround] keyless.\n"
        "- [moraine 0.7.1] permanent, no key.\n"
    )
    assert [e.key for e in parsed.entries] == ["seam-weld", "seam-weld", None, None]

    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    (ws / "journal.md").write_text(
        "## Entries\n"
        "- [moraine 0.7.0, workaround: seam-weld] root instance.\n",
        encoding="utf-8",
    )
    (ws / "alpha").mkdir()
    (ws / "alpha" / "alpha.star").write_text("", encoding="utf-8")
    (ws / "alpha" / "journal.md").write_text(
        "## Entries\n"
        "- [moraine 0.7.1, workaround: seam-weld] alpha instance.\n"
        "- [moraine 0.7.1, workaround] keyless stale.\n"
        "- [moraine 0.7.2, workaround: seam-weld] fresh - not stale.\n",
        encoding="utf-8",
    )
    review, counts = journal.scan(ws, "0.7.2")
    assert len(review) == 2
    group, single = review
    assert group == {
        "key": "seam-weld", "entries": 2,
        "journals": ["journal.md", "alpha/journal.md"],
        "versions": ["0.7.0", "0.7.1"],
        "entry": "root instance.",
    }
    assert single["journal"] == "alpha/journal.md" and "key" not in single
    assert counts["entries"] == 4 and counts["unstamped_by_journal"] == {}


def test_a_bare_stamp_line_is_prose_and_stamps_nothing(tmp_path):
    """DESIGN 4.7: a stamp counts only at the head of an entry bullet —
    a column-0 line holding only a stamp is prose, and the unstamped
    bullets after it stay unstamped (the rejected file-level default
    would have made every later subject-side observation a workaround
    of a version that never saw it)."""
    text = (
        "# old field journal\n"
        "[moraine 0.6.0, workaround: legacy-era]\n"
        "- first legacy limit. -> #a\n"
        "- [moraine 0.7.2, workaround] individually stamped.\n"
        "- The human chose the wide window for the sea view.\n"
        "## Details\n### a\nbody\n"
    )
    parsed = journal.parse(text)
    assert [(e.line, e.version, e.kind) for e in parsed.entries] == [
        (4, "0.7.2", "workaround"),
    ]
    assert parsed.unstamped == 2
    assert parsed.unstamped_pointers == [(None, "a")]

    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    (ws / "journal.md").write_text(text, encoding="utf-8")
    review, counts = journal.scan(ws, "0.7.2")
    assert review == []
    assert counts["unstamped"] == 2 and counts["unstamped_by_journal"] == {
        "journal.md": 2
    }


def test_entries_anchor_scopes_the_index():
    """DESIGN 4.7: with the anchors present, `## Entries` is the index
    and the preamble and any other section are prose — a stamped
    working note kept above the anchor is neither an entry nor an
    unstamped bullet; a journal without the anchor is read whole."""
    text = (
        "# kiosk — journal\n"
        "- [moraine 0.6.0, workaround] a preamble note, not an entry.\n"
        "- an unstamped preamble bullet.\n"
        "## Entries\n"
        "- [moraine 0.7.2, workaround] the one entry.\n"
        "- the one unstamped observation.\n"
        "## Notes\n"
        "- [moraine 0.6.0, workaround] another section: prose.\n"
        "## Details\n### a\nbody\n"
    )
    parsed = journal.parse(text)
    assert [(e.line, e.version) for e in parsed.entries] == [(5, "0.7.2")]
    assert parsed.unstamped == 1
    assert parsed.detail_slugs == ["a"]


def test_unstamped_entries_point_at_bodies_too(tmp_path):
    """The reference's `dangling` — "a pointer without its `### slug`,
    or a Details section no entry points to" — counts unstamped
    entries as entries: the subject-side norm is unstamped, so a
    journal whose bodies are all pointed at reports 0, and an unstamped
    pointer at a missing slug is the dangling one."""
    p = tmp_path / "journal.md"
    p.write_text(
        "## Entries\n- Counter height 0.9. -> #counter\n- Roof trim rejected. -> #trim\n\n"
        "## Details\n### counter\nbody\n\n### trim\nbody\n",
        encoding="utf-8",
    )
    rep = journal.scan_file(p)
    assert rep.parsed.unstamped == 2 and rep.parsed.unstamped_pointers == [
        (None, "counter"), (None, "trim")
    ]
    assert rep.dangling == 0
    p.write_text("## Entries\n- Points nowhere. -> #nowhere\n", encoding="utf-8")
    assert journal.scan_file(p).dangling == 1


def test_wrapped_entry_keeps_its_pointer(tmp_path):
    """One entry is one column-0 bullet, continuation lines included: a
    pointer closing the last wrapped line names the body, for stamped
    and unstamped bullets alike, while a nested bullet stays inside its
    entry and a blank line or the next column-0 line ends it."""
    p = tmp_path / "journal.md"
    p.write_text(
        "## Entries\n"
        "- [moraine 0.7.2] A long stamped observation that wraps\n"
        "  onto a second line. -> #long\n"
        "- An unstamped one that wraps\n"
        "  and carries a nested bullet\n"
        "  - nested, not an entry\n"
        "  before it points. -> #wrapped\n"
        "\n"
        "  indented prose after a blank line is not a continuation -> #stray\n"
        "- [moraine 0.7.2] no pointer\n"
        "## Details\n### long\nx\n\n### wrapped\ny\n",
        encoding="utf-8",
    )
    rep = journal.scan_file(p)
    assert [e.pointer_slug for e in rep.parsed.entries] == ["long", None]
    assert rep.parsed.entries[0].text == "A long stamped observation that wraps"
    assert rep.parsed.unstamped == 1
    assert rep.parsed.unstamped_pointers == [(None, "wrapped")]
    assert rep.dangling == 0


def test_a_pointer_on_the_entry_own_line_survives_a_continuation(tmp_path):
    """The pointer closes one of the entry's lines — its own, or the
    last of a wrapped one: a body pointed at from the first line stays
    pointed at when a nested bullet or more prose follows, stamped and
    unstamped alike, so its Details section is not dangling."""
    p = tmp_path / "journal.md"
    p.write_text(
        "## Entries\n"
        "- [moraine 0.7.2] Stamped, points then carries a note. -> #first\n"
        "  - a nested note, inside the entry\n"
        "- Unstamped, points then wraps. -> #second\n"
        "  continuing prose after the pointer\n"
        "## Details\n### first\nx\n\n### second\ny\n",
        encoding="utf-8",
    )
    rep = journal.scan_file(p)
    assert [e.pointer_slug for e in rep.parsed.entries] == ["first"]
    assert rep.parsed.unstamped_pointers == [(None, "second")]
    assert rep.dangling == 0
