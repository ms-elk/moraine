# CLI reference (build, query, screenshot, export, pins, mcp, cache, sessions)

### build (CLI, recipe to IR)

`moraine build [PROJECT] [--diff [PATH]] [--check]` runs the project's
recipe and the recipes it references, and writes each asset's IR
under its project's `build/`. Every evaluating command (`validate`,
`stats`, `query`, `eval`, `export`, `screenshot`, `serve`) builds
first, so `build` alone is for reading what a recipe expands to, or
for the refusal with nothing evaluated. `PROJECT` is a project
directory or an asset file. A requirements file and a `lib/` file are
built with their asset and are not targets. The report lists each IR
with its `recipe` and its `nodes` / `asserts` / `outputs` counts, the
files `written`, and the count left `unchanged`, since a build that
changes nothing touches no file. `warnings` carries
`copies_not_instanced` (see *load*), `threshold_tautology` (see
*requirements*), and `accepted_value_unused`, an id `params.json`
holds that the root asset declares no knob for.
`unread` names an asset file the build ran and could not read for how
it uses its names, whose knobs are all staged fields. A failure is
one error naming the `site` to edit (see *refusals*).

`--diff` reports what the edit reached, per asset that changed:
`nodes` and `asserts` `added`, `removed`, and `changed` (a changed
row names its `site` and each changed param as `[before, after]`),
human-owned asserts under `asserts.human`, the `outputs` made
`dirty`, and the root keys whose `declarations` changed. Under
`nodes`, `renamed` pairs the added and removed nodes that are one
subgraph under two ids (`from`, `to`; equal content is all a pair
shares, so a node removed and an equal one added elsewhere pair
too), and marks a pair
`stream_changed` where the id salts the op's streams, as `scatter`'s
does, so the result changed with the name. Run it after
an edit inside a function or a loop to see every node the line
produced. The base is the IR on disk before the build. `--diff PATH`
compares against an IR file or a `build/` directory instead, which is
how two revisions are compared: build each, then name the other's
`build/`. An asset the base holds no IR of, as another project's is
when one project's `build/` was saved, is listed under `not_compared`
and compared with nothing; an asset only a new reference reads is
reported as added. To compare those too, save
the workspace and name each project's `build/` in turn. A list past
ten entries names ten and counts the rest under `by_site`, the recipe
line that made them, and under `by_caller`, the outermost call they
came through, where they came through one. A line moved by a comment
is no change.

`--check` builds to memory and writes nothing. `check.differs` names
each IR on disk that is not what the recipe builds to now, with the
first differing `line` quoted as `built` and `on_disk`, `missing` the
IRs the disk does not hold, and `stale` what the next build removes.
Any difference exits 1.

The build of a project's root removes from that project's `build/`
the IR no project of the workspace reads any more (`removed`): the
variant of an override since changed, a sub-asset once built alone.
A variant sits in the `build/` of the project its asset belongs to,
so a sibling's variant that a changed override left goes with the
sibling's next root build. A file there that the build did not write
is named under `unrecognized` and left, and a file written within the
last minute is left for a later build. `blocked` names what stops
every removal:
an IR the build could not read (`files`), until it is built again,
or a project holding more than one top-level asset file (`assets`).

`consumers` names the projects that reference this one (`projects`),
read from their IR on disk, and the projects that could not be asked
(`not_built`, `unread`).

An IR handed to a command by path (`moraine validate
build/<name>.mora`) is evaluated as it is, with no build. It is
refused when another build wrote it (`version_mismatch`).

### query (CLI, read side)

`moraine query [PROJECT] [--select '<predicate JSON>'] [--tag NAME]
[--instance I] [--output NAME]` is a deterministic read-only query over evaluated
outputs (records come from the cache; no GLB materialization). Per
mesh output it reports `{"faces", "matched"}` plus, when matched > 0,
total `area`, the `bounds` of the matched faces, and their
area-weighted `centroid`, so token cost scales with the answer.
`--tag NAME` reads the faces a tag caught, by its name, and with
`--select` those of them the predicate matches. An output without the
tag is skipped with the tags it carries, and a tag no output carries
is an error (`unknown_tag`). `--instance I` reads copy I of an
instanced output where it stands in the world (`frame: "copy I
(world)"`), which is the frame `place` and `mate` take an arrayed
anchor in. Image outputs are skipped with a structured note; an
instanced output is read on its **prototype in the prototype's local
frame** (the frame `tag` writes in) and says so (`instanced`: copy
count, `frame`), so the query works across `project` refs without
opening the referenced project's files. Exit code 0. Zero matches is
an answer, not a failure. A number the requirement *states* is not a
query read into prose: declare it as an `extent` / `count` assert (see
*Asset assertions*) and let every `validate` re-read it.

### screenshot (CLI, the pixel tier)

`moraine screenshot [PROJECT] [--view front|iso|... | --azimuth DEG
--elevation DEG] [--distance M] [--target X,Y,Z] [--size WxH] -o
shot.png` renders pixels through the running server's viewer (it
opens a tab if none is connected; `no_viewer` names the URL to open).
Pixels are tier 3: your diagnostic for what no number covers, never
the acceptance gate. That is the human's, deciding on your report.

**What a shot renders.** The asset's *current file state*, never a
slider preview or the tab's displayed generation. `--param
ID=VALUE`, repeatable, shoots the asset under a value of a declared
knob in place of it, as `validate --param` judges one (see *checks*):
no file is written and no viewer moves, and the report names the
values under `params_given`. The report's
`project` / `asset_hash` / `port` say exactly what was shot, and
`watch: "dead"` means the file watch died and is re-arming (your
shot is still current; the human's display may lag until it does;
`validate` / `stats` carry the same word under `serve`). Without a
browser, report visual verification as incomplete. A hand-rolled
fallback renderer is diagnosis at best, never acceptance.

**Another workspace's server.** A server of another workspace on the
port is not yours: `validate` / `stats` name it as
`serve.other_root`, `screenshot` and `pins` are refused
(`wrong_workspace`), and `serve` reports `port_in_use` naming the
holder. Serve this workspace with `--port <other>` and pass the same
`--port` to those commands.

**Framing a decisive shot.** One view decides little, and a subject
spanning under ~150 px on its long side decides nothing. Get closer
(`--target` / `--distance`, written `--target=X,Y,Z` when X is
negative, because a leading minus reads as an option otherwise;
`--size` only when you cannot): changed *fractions* are
resolution-invariant, so nothing warns you.

- An assembly: a full view plus a second azimuth >= ~60 deg apart (an
  interpenetration reads as "attached" from the angle that hides it).
- A joint: a close-up (`--target` on the joint, short `--distance`)
  shot **edge-on**. Sight along the contact plane, perpendicular to
  the contact normal; facing a contact flat-on cannot show its gap at
  any distance. For an instanced contact pick a copy deliberately
  (the nearest, an end, plus one representative). "The joint" is not
  one place.
- An interior: the free camera (`--target` inside, `--azimuth` /
  `--elevation` / `--distance` through the opening. Fixed views
  cannot see it).
- The asset's intended viewing distance: a few-millimeter step
  passes every numeric tier and vanishes at thumbnail resolution;
  detail that reads at 1 m may be noise at 10 m, or gone. Either way
  it is not what the viewer sees.

An output meant to be visible that contributes no pixels to the shots
you took is a defect. Reposition it, use it as a cutter, or remove it.

### export (CLI, the engine handoff)

`moraine export [PROJECT] [--profile gltf] [-o DIR] [--bake-instances]
[--recompute]` requires a green `validate` (waivers included) and
writes one GLB per asset. Each named output is one root node (no
parent asset node), in the order of the output names, and instancing
is kept via `EXT_mesh_gpu_instancing` as a *required* extension: an
importer without it refuses the file (`--bake-instances` projects
instanced outputs to plain meshes for those; the manifest records the
flag). It also writes a `*.manifest.json` recording tool and kernel
versions. A relative `-o` resolves against
the project, so `-o exports/` lands inside the project from any
directory (`screenshot -o` stays cwd-relative). `--recompute` proves
byte-identical re-export, never overwriting a cache record; a mismatch
is a red report that writes nothing, so the last good export stays.

Before an engine handoff, run `moraine validate --profile gltf` and
read its report precisely: profile check failures gate the export,
`level: "warn"` entries are advisory (missing UVs are fine on
untextured materials). Verify the import with the unmodified GLB:
hierarchy (one root node per named output), materials, emissive
parts, orientation, scale, instances.

### pins (CLI, the pointing channel)

`moraine pins list | add --output NAME (--select '{...}' | --face N)
[--instance I] | clear [--id N]` on the running server. A pin is the
human clicking geometry in the
viewer; `pins list` returns each one resolved into your vocabulary
(`output`, `instance`, `face`, its `groups`, face fields, and the
click's world `position`) so you act on structure, never on a guess
from a screenshot. A pin whose recorded `asset_hash` differs from the
current report's predates your edits. Confirm before acting on it.
Pins go both ways: `pins add` marks where *you* mean in the human's
viewer. Pins are pointing gestures, not annotations: they die with
the server, clear on a project switch, and fall off past the store's
cap. Never persist a pin's raw face index into the graph. Indices go
stale; when a pointed selection must survive, fix it as a `tag` built
from the pin's `groups` and face fields. `pins clear` when done.

### mcp (CLI, the tool surface)

`moraine mcp` exposes the read/verify commands (`stats`, `validate`,
`query`, `export`, `screenshot`, `docs`) as MCP tools over stdio, so
an MCP client can drive the verify loop without a shell; each tool
answers with the same compact JSON its command prints. Assets are
still edited as recipe files. A client registers the installed
command, for example:

```
claude mcp add moraine -- moraine mcp
```

or, in a client's JSON configuration,
`{"mcpServers": {"moraine": {"command": "moraine", "args": ["mcp"]}}}`.

### cache (CLI, the machine-level store)

The evaluation cache is machine-level, tool-managed, and regenerable.
It lives in the OS cache directory (`MORAINE_CACHE_DIR`
overrides), never in a workspace, and is size-capped
(`MORAINE_CACHE_MAX_BYTES`, default 4 GiB, `0` disables the cap).
Preview evaluations from the viewer go to its ephemeral `preview/`
bucket, outside the cap. `moraine cache stats` reports size and
composition, the bucket separately. `moraine cache evict` runs the cap
now. `moraine cache clear` empties everything, the bucket included.
None of these is a determinism check (`eval --recompute` is, and it
never overwrites a record), and the cache only ever deletes files it
wrote itself.

### sessions (CLI, the session log)

The runtime keeps a session log of what its entry points did, so a
round of authoring can be read back as counts: one event
per CLI command (the command, its outcome, its time, the bytes it
printed, the asset revision it saw, and a summary of what `validate`
or `stats` found), and from `serve` one per start and stop, per
reload, per screenshot, and per accept. Names, hashes, counts, and
outcomes only: no asset content, no knob value, and nothing a knob
scrub or a pin click in the viewer does. It is one file per UTC day
under the OS log directory (`MORAINE_LOG_DIR` overrides; set it empty
to turn logging off), never inside a workspace, kept 30 days
(`MORAINE_LOG_KEEP_DAYS`; `0` keeps everything), and read by `moraine
sessions` and the viewer's timeline alone: nothing leaves the machine.
`moraine sessions [PATH]`
summarizes the events under a workspace root, a project, or any
directory above them (default the cwd) as sessions (runs of events
closer than `--gap` minutes, default 30; `--last N` newest, `--all`)
and, per project, rounds: one asset revision carrying at least one
verification, with its command counts, `validate_ok`, `failed_checks`,
`ended_by`, `changed_lines`, and wall time. Nothing under the path is
an actionable `no_sessions` error, never an empty report.
