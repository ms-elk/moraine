"""Project journal index (DESIGN 4.7) — the read-only side of the
retirement pass that ``moraine init`` runs (DESIGN 6).

A journal is one markdown file. Its ``## Entries`` section holds one
entry per column-0 bullet, each opening with a version stamp:
``- [moraine <version>] ...`` (permanent) or
``- [moraine <version>, workaround] ...`` (version-bound), optionally
with a claim key — ``[moraine <version>, workaround: <key>]`` — naming
the tool limit, so the pass can group the entries of one limit. An
entry may end with a pointer to a body — ``-> #slug`` into the file's
own ``## Details`` section (a ``### slug`` heading) or
``-> <relative>.md#slug`` into a sibling file; the pointer closes one
of the entry's lines — its own, or the last of a wrapped one — and
unstamped bullets point the same way. Stamps older or newer
than the installed version on *workaround* entries are what the pass
lists; nothing here writes to a journal — retiring an entry is the
agent's or the human's act.

With an ``## Entries`` heading present, only the bullets under it are
read: the preamble and any other section are prose. Files without the
two anchors (the pre-stamp layout) still parse: a stamped bullet
anywhere outside ``## Details`` is an entry, any other column-0
bullet counts as unstamped.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from . import workspace as workspace_mod

JOURNAL_NAME = "journal.md"
ENTRIES_HEADING = "entries"
DETAILS_HEADING = "details"

_HEADING2_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")
_HEADING3_RE = re.compile(r"^###\s+(?P<slug>\S+)\s*$")
_BULLET_RE = re.compile(r"^[-*]\s+")
ENTRY_RE = re.compile(
    r"^[-*]\s+\[moraine\s+(?P<version>[^\s,\]]+)"
    r"(?:\s*,\s*(?P<kind>workaround)(?:\s*:\s*(?P<key>[^\]]*?))?)?\s*\]"
    r"\s*(?P<text>.*)$",
    re.IGNORECASE,
)
POINTER_RE = re.compile(
    r"(?:→|->)\s*(?P<path>[^\s#]+\.md)?#(?P<slug>\S+)\s*$"
)

ENTRY_PREVIEW_CHARS = 100

# Journals are read as utf-8-sig: an editor-written BOM must not eat the
# first line (a stamped entry on line 1 would otherwise vanish from the
# pass); the file's own encoding is not rewritten.
JOURNAL_ENCODING = "utf-8-sig"


def unfenced_lines(lines: Iterable[str]) -> Iterator[tuple[int, str]]:
    """``(index, line)`` for every line outside a ``` fence, the fence
    lines themselves excluded: a ``#`` heading or a bullet inside a
    fenced example (a journal layout, an asset file) is content, not
    structure. Shared by the journal parser and ``moraine docs``."""
    fenced = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            yield i, line


@dataclass(frozen=True)
class Entry:
    line: int  # 1-based
    version: str
    kind: str  # "permanent" | "workaround"
    text: str
    pointer_path: str | None  # relative file for a sibling body, else None
    pointer_slug: str | None
    key: str | None = None  # workaround claim key (one name per tool limit)


@dataclass
class Parsed:
    entries: list[Entry] = field(default_factory=list)
    unstamped: int = 0
    detail_slugs: list[str] = field(default_factory=list)  # under ## Details
    all_slugs: list[str] = field(default_factory=list)  # every ### heading
    # (path, slug) pointers of the unstamped bullets: they name bodies
    # like any entry, so `scan_file` resolves them the same way.
    unstamped_pointers: list[tuple[str | None, str]] = field(default_factory=list)


def _is_heading(raw: str, title: str) -> bool:
    h2 = _HEADING2_RE.match(raw.rstrip())
    return bool(h2) and h2.group("title").strip().lower() == title


def _stamp_parts(m) -> tuple[str, str, str | None]:
    kind = "workaround" if m.group("kind") else "permanent"
    key = (m.group("key") or "").strip() or None
    return m.group("version"), kind, key


def _pointer(lines: list[str]) -> tuple[str | None, str | None]:
    """The bullet's pointer: the first of its lines one closes. Joining
    the lines first would drop a pointer that closes the entry's own
    line before a continuation follows — POINTER_RE anchors at the end
    of what it is given, and only the last line would still match."""
    for line in lines:
        p = POINTER_RE.search(line)
        if p:
            return p.group("path"), p.group("slug")
    return None, None


def _entry(
    i: int, version: str, kind: str, key: str | None, lines: list[str]
) -> Entry:
    """``lines`` is the whole bullet: its first line the preview text,
    any of its lines the one the pointer closes."""
    path, slug = _pointer(lines)
    return Entry(
        line=i, version=version, kind=kind, text=lines[0],
        pointer_path=path, pointer_slug=slug, key=key,
    )


def parse(text: str) -> Parsed:
    """Pure parse of one journal file's text. A bullet runs until the
    next column-0 line or blank line: its indented continuation lines
    (a wrapped entry, a nested bullet) belong to it, so a pointer
    closing any of its lines is the entry's pointer."""
    out = Parsed()
    lines = list(unfenced_lines(text.splitlines()))
    has_entries = any(_is_heading(line, ENTRIES_HEADING) for _, line in lines)
    # Without the Entries anchor the whole file is the index.
    in_entries = not has_entries
    in_details = False
    # The bullet being read: line, stamp parts or None, its lines.
    open_: list | None = None

    def close() -> None:
        nonlocal open_
        if open_ is None:
            return
        i, stamp, bullet = open_
        open_ = None
        if stamp is None:
            out.unstamped += 1
            path, slug = _pointer(bullet)
            if slug is not None:
                out.unstamped_pointers.append((path, slug))
        else:
            out.entries.append(_entry(i, *stamp, bullet))

    for i0, raw in lines:
        i = i0 + 1  # 1-based
        line = raw.rstrip()
        h2 = _HEADING2_RE.match(line)
        if h2:
            close()
            title = h2.group("title").strip().lower()
            in_details = title == DETAILS_HEADING
            if has_entries:
                in_entries = title == ENTRIES_HEADING
            continue
        h3 = _HEADING3_RE.match(line)
        if h3:
            close()
            out.all_slugs.append(h3.group("slug"))
            if in_details:
                out.detail_slugs.append(h3.group("slug"))
            continue
        if in_details or not in_entries:
            continue
        if open_ is not None and line and raw[0] in " \t":
            open_[2].append(line.strip())
            continue
        close()
        m = ENTRY_RE.match(line)
        if m:
            open_ = [i, _stamp_parts(m), [m.group("text")]]
            continue
        if _BULLET_RE.match(line):
            open_ = [i, None, [_BULLET_RE.sub("", line, count=1)]]
    close()
    return out


@dataclass
class FileReport:
    path: Path
    parsed: Parsed
    dangling: int


def scan_file(path: Path) -> FileReport:
    """Parse one journal and resolve its body pointers — own ``##
    Details`` slugs for ``#slug``, a sibling file's ``###`` headings for
    ``<relative>.md#slug``. Dangling = a pointer whose target heading is
    missing or ambiguous (duplicate slug), plus a ``## Details`` heading
    no entry points to. Stamped and unstamped bullets point alike: an
    unstamped subject-side entry's body is as real as a stamped one's."""
    parsed = parse(path.read_text(encoding=JOURNAL_ENCODING))
    own = parsed.detail_slugs
    dangling = 0
    pointed: set[str] = set()
    sibling_cache: dict[str, list[str]] = {}
    pointers = [
        (e.pointer_path, e.pointer_slug)
        for e in parsed.entries if e.pointer_slug is not None
    ] + parsed.unstamped_pointers
    for pointer_path, pointer_slug in pointers:
        if pointer_path is None:
            targets = own
            pointed.add(pointer_slug)
        else:
            key = pointer_path
            if key not in sibling_cache:
                sib = path.parent / Path(*key.split("/"))
                try:
                    sibling_cache[key] = parse(
                        sib.read_text(encoding=JOURNAL_ENCODING)
                    ).all_slugs
                except (OSError, UnicodeDecodeError):
                    sibling_cache[key] = []  # unreadable target: dangling
            targets = sibling_cache[key]
        if targets.count(pointer_slug) != 1:
            dangling += 1
    dangling += sum(1 for s in own if s not in pointed)
    return FileReport(path=path, parsed=parsed, dangling=dangling)


def journal_paths(root: Path) -> list[Path]:
    """The workspace journal plus every project's, existing files only,
    in sweep order (root first, then projects lexicographically)."""
    paths = [root / JOURNAL_NAME]
    for name in workspace_mod.discover(root).projects:
        paths.append(root / name / JOURNAL_NAME)
    return [p for p in paths if p.is_file()]


def scan(root: Path, installed: str) -> tuple[list[dict], dict]:
    """The retirement pass: ``(journal_review, journals)`` for the
    ``init`` report. Review rows are workaround entries whose stamp is
    not the installed version; paths are workspace-relative (reports get
    quoted into committed files, DESIGN 6). Entries sharing a claim key
    collapse into one row per limit — "is this entry stale" becomes "is
    this limit fixed", answered once — sitting where the key first
    appears in sweep order; keyless entries stay one row each. Counts
    carry per-journal unstamped totals (``unstamped_by_journal``) so a
    legacy journal is named, not just summed. The pass degrades per
    journal: one that cannot be read (OSError, not UTF-8) is named on
    stderr and counted under ``skipped``; the rest are still scanned."""
    review: list[dict] = []
    groups: dict[str, dict] = {}
    counts: dict = {
        "scanned": 0, "entries": 0, "unstamped": 0, "dangling": 0,
        "skipped": 0, "unstamped_by_journal": {},
    }
    for path in journal_paths(root):
        rel = path.relative_to(root).as_posix()
        try:
            rep = scan_file(path)
        except (OSError, UnicodeDecodeError) as e:
            print(f"journal skipped: {rel}: {e}", file=sys.stderr)
            counts["skipped"] += 1
            continue
        counts["scanned"] += 1
        counts["entries"] += len(rep.parsed.entries)
        counts["unstamped"] += rep.parsed.unstamped
        if rep.parsed.unstamped:
            counts["unstamped_by_journal"][rel] = rep.parsed.unstamped
        counts["dangling"] += rep.dangling
        for e in rep.parsed.entries:
            if e.kind != "workaround" or e.version == installed:
                continue
            if e.key is not None:
                g = groups.get(e.key)
                if g is None:
                    g = {
                        "key": e.key,
                        "entries": 0,
                        "journals": [],
                        "versions": [],
                        "entry": e.text[:ENTRY_PREVIEW_CHARS],
                    }
                    groups[e.key] = g
                    review.append(g)
                g["entries"] += 1
                if rel not in g["journals"]:
                    g["journals"].append(rel)
                if e.version not in g["versions"]:
                    g["versions"].append(e.version)
            else:
                review.append(
                    {
                        "journal": rel,
                        "line": e.line,
                        "version": e.version,
                        "entry": e.text[:ENTRY_PREVIEW_CHARS],
                    }
                )
    return review, counts
