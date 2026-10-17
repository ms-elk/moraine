# Journal format

## journal (project journal format)

`journal.md` (one per project, optionally one at the workspace root)
is the agent-written record of the *why*: rationale,
corrections, accepted and rejected variants, failed approaches. The
guide carries the writing rules (declarative observations, narrowest
covering level, the human's language, never what `stats` derives);
this section is the file format the tool reads.

Layout is one file, two sections:

```
# kiosk - journal
(free preamble: the asset, the session's aim)

## Entries
### dimensions            <- optional topic sub-headings, ignored by the tool
- Counter height 0.9: the human's standing-counter reference; the 1.1 first draft read as a bar. -> #counter
- [moraine 0.7.2, workaround] `smooth_normals` needs `angle_max_deg` > 360/segments; the default 30 leaves a 12-segment post faceted.
### rejected
- A separate output for the roof trim was rejected: one welded shell, and a split only adds overlap-guard pairs.

## Details
### counter
(the body an entry points to: measurements, the path not taken, the
 evidence. Read only when its entry is relevant)
```

- `## Entries` is the part every session reads; one entry is **one
  column-0 bullet** (its indented continuation lines included); the
  preamble and any other section are prose the tool does not read.
  `## Details` holds bodies, one `### <slug>` section each, read on
  demand when one of an entry's lines ends with `-> #slug` (the
  Unicode arrow U+2192 is accepted too). Slugs are unique within the
  file. A pointer of the form `-> <relative>.md#slug` targets a
  sibling file, the shape a journal takes if it is ever split into
  topic files.
- A **version stamp** (the value of `moraine --version` at writing
  or last verification) marks what the tool's version can falsify;
  a subject-side observation (the asset, the human, the design:
  true regardless of tool version) carries **no stamp**.
  `[moraine <version>]` marks an observation of this version's
  behavior. Its consumer is the next upgrade: re-read it then.
  `[moraine <version>, workaround]` marks a version-bound limit (a
  bug, a diagnosis of tool behavior, and what was done about it).
  Write the symptom so it can be re-verified. Whatever a product fix
  would make false carries a stamp. A workaround stamp may carry a
  **claim key** (`[moraine <version>, workaround: <key>]`, one
  short kebab-case name per tool limit, shared by every entry that
  limit forced) so the retirement pass can ask "is this limit
  fixed" once per limit instead of once per entry. A stamp counts
  only at the head of an entry bullet.
- The retirement pass: `moraine init` (run at the workspace root after
  every Moraine upgrade) reads every journal (it never writes one)
  and lists under `journal_review` each `workaround` entry whose stamp
  is not the installed version (`journal`, `line`, `version`, `entry`;
  entries sharing a claim key collapse to one row per limit: `key`,
  `entries`, `journals`, `versions`, and the first entry's preview),
  plus counts under `journals` (`scanned`, `entries`, `unstamped` with
  its per-journal `unstamped_by_journal`,
  `dangling`, that is a pointer without its `### slug`, or a Details
  section no entry points to). Re-verify each listed entry against the
  new version in the asset itself: restore what the workaround
  replaced, a value or a construction, run `validate` once, revert (a
  limit on a command is re-tried in place), and delete the entry when
  the limit is gone or re-stamp it when it still holds. No probe project:
  an entry whose site the asset no longer holds is deleted unverified.
  Unstamped bullets are the subject-side norm (and the
  legacy state of pre-format journals); they are counted, not listed.
  When touching a legacy entry, stamp it only if this version can
  falsify it.
- Journals are read by the engine only here; nothing about their
  content enters evaluation, caching, or export.
