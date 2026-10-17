# Moraine authoring guide

Agent guide for authoring assets in a Moraine workspace. This file is
**maintained by the tool**: `moraine init` keeps it matching the
installed Moraine version, and local edits are overwritten on refresh.
The human's rules live in the workspace's `AGENTS.md` (which overrides
this guide); your observations — rationale, corrections, accepted and
rejected variants — belong in `journal.md`. The recipe *is* the
asset: the agent edits it, the build expands it, the runtime
evaluates it deterministically, and the viewer shows the result.

Reference beyond this guide is one command away: `moraine docs` lists
the shipped reference by file and section, `moraine docs <section>`
prints one (an op, an assert kind, the recipe, predicates, texture
patterns; `api` is every builtin with its arguments) and names the
examples it cites; `moraine --help` lists the commands.
This guide carries the rules that must shape an asset *before* the
tool can speak; the reference carries the contracts to consult while
composing. A line earns its place here only if the violation it
guards is **silent** (no check, no error, and no reference section
this guide routes to catches it), or it changes the **subject,
structure, or public surface** before building starts, or it is
needed **every session** — everything else lives in `moraine docs`,
`--help`, or the error channel.
<!-- budget: 442 lines -->

## What can and cannot be built {#scope}

Conventions, normative for every asset: meters, real-world scale ·
right-handed · +Y up · +Z forward (asset front faces +Z) · CCW
winding · UVs v-down, (0, 0) at the texture's top-left. Engine
conversions happen only in export profiles, never in the asset.

The kernel builds rule-describable structure: primitive stock (box,
cylinder, sphere, grid) shaped by displacement, extrusion,
revolution, booleans, chamfers, and instancing. Check the subject
against what has **no operator** before committing to it:

- Profiles revolve but do not extrude: `revolve` sweeps an open
  `[radius, y]` polyline a full 360° around +Y (cones, tapers,
  turned profiles); a drawn outline cannot be extruded along a line,
  and partial revolutions have no op — linear profiled parts stay
  boolean combinations of the primitive stock.
- Outputs must be closed solids: an open surface (a `grid`, a
  displaced sheet) cannot be an output, so terrain is a displaced
  sheet **plus the `extrude` that closes it**; a scatter *target*
  may stay open.
- Curved or coaxial mating, placement on displaced terrain, and
  placing an `asset_ref`'d kit have no op — those transforms stay
  authored, so a composite asset's positioning falls entirely there.
  No texture compositing (one source per slot), no `scatter` masks,
  alpha-only transparency.
- Sculptural form — quality living in vertex-level nuance, no
  nameable degree of freedom — is out of scope by design: say so
  rather than approximating it.

Routing — at these branch points, read the named contract **before**
the plan commits (each row's limit changes plans, not parameters):

| Deciding | Read first (`moraine docs …`) | The plan-changing limit |
|---|---|---|
| write or restructure a recipe | `recipe` | a node's id is its identity; inside a function every node is named |
| rest a part on a surface | `place` | on-top only, single horizontal plane |
| butt / fit / dock faces | `mate` + `mated` | planar face sets; a contained joint stays authored, checked by `mated` `mode: "contained"` |
| select faces | *Face predicates* | triangles, not logical faces; groups drop on topology rebuilds |
| repeat parts | `array` / `scatter` | transform frame; stochastic outputs cannot become knobs |
| terrain / relief | `grid`, `noise_displace`, `extrude`, `boolean` | open surfaces cannot be outputs |
| chamfer edges | `bevel` | convex solids only |
| turn a rotational part | `revolve` | the solid always contains the axis — a ring / torus has no op |
| texture a surface | *patterns* + `unwrap` | box/planar projection only; one source per slot |
| assign materials | `material` | name is the export identity; a factor knob on a shared name is never live |
| place another project's asset | `asset_ref` | an override reaching a material needs `material_rename` |
| state a requirement relative to a knob or another part | `extent` + `requirements` | a threshold is a number: `param_value` takes no arithmetic, no assert compares two measurements — substitute, and say which |

## Workspace & projects {#workspace}

One repository is one **workspace**: the root holds `AGENTS.md`
(human-owned rules), optionally a workspace `journal.md`, and this
guide under `.moraine/` (committed — tool-managed, refreshed by
`moraine init`, so upgrades show up as diffs). Each **project** is a
child directory directly under the root, holding one root asset
`<name>.star` (the recipe), the human's `<name>.requirements.star`
and `params.json` (accepted knob values), definitions under `lib/`,
sub-assets under `subassets/` (placed via `asset_ref`), its
`journal.md`, non-regenerable inputs in `external/` (committed), and
two regenerable directories: the IR the build writes in `build/` and
profile outputs in `exports/`. The project is the unit of evaluation
and export; references never leave the workspace — an in-project
`asset_ref` targets `subassets/`, a cross-project one names a sibling
project and one of its **root asset's** named outputs (root outputs
plus its knobs are a project's whole public interface), and reuse
across repositories stays vendoring.

Edit the recipe, never the IR under `build/` (the next build
overwrites it). The cache, pins, and journal know a node by its id,
and a `scatter`'s id seeds its streams: keep it when restructuring
(moving a part into a function, pass the id it had), and never reuse
one. `moraine build --diff` names what an edit `renamed`.

## What the project publishes {#public}

Choosing the root outputs is a public decision: it fixes the
granularity consumers get (publish the repeat unit and the end pieces
separately and a consumer can `array` the unit; publish the finished
run and it cannot), and the face groups those outputs carry are part
of the surface — consumers `place`/`mate` against them: publish a
tag on every face a consumer will seat or dock against (an untagged
kit makes each consumer re-tag it); a renamed tag or topology
rebuild upstream of a root output breaks them. Each named output is
also one named child in the engine hierarchy at export, so
granularity is a hierarchy and draw-call decision too. Export writes
the children by name: the order of declaration means nothing.

Counts are choices, not defaults — the asset is real-time content:
spend triangles where the silhouette or a close view needs them and
nowhere else (a small fitting never needs a hero part's resolution);
a material identity is a draw state, a part a hierarchy node.
`moraine stats` reports `asset.verts`/`tris` (instances multiplied
in) and `asset.materials`, so each traces to a choice.

Declare the asset's human-tweakable knobs with `mo.param` (contract:
`moraine docs knobs`): a handful of high-level, **self-contained**
knobs, not every param. A knob can be declarable and still broken,
so test each candidate against what fails silently:

- Assert the joints it moves — else a slider breaks one while
  `validate` stays green. A knob whose bad values fail loudly takes a
  narrowed range.
- A `material` factor knob on a name several parts share goes to
  every use of the name: one left out is one name with two contents,
  which fails `validate`.
- The valid range is one contiguous interval (no islands; a vector
  param needs all components valid together), and the knob shows in
  the asset's *assembled* outer form — a kit's interior knobs vanish
  once it is placed.
- It must hold **in combination** with the already-declared knobs —
  two individually-safe knobs can collide at their extremes.
- **Sweep the declared range before declaring**: `moraine validate
  --param <id>=<value>` judges a value over an accepted one and
  touches no file and no viewer. Declare the range you actually
  accept, not a slider-friendly one. Sweeps only catch loud failures:
  a rotation `array` closing a full turn has exactly one valid
  `count`; below it the ring is incomplete and `validate` stays green.
- A `material` factor knob moves no geometry, so a sweep sees nothing
  at any value: only pixels check its range. Shoot both ends and a
  value between (`moraine screenshot --param <id>=<value>`, the
  recipe untouched), and confirm neighboring values differ — a
  saturated range looks right at both ends. The map's minimum times
  the factor floor is the effective floor.

Repetition belongs to the placing side: a kit exposes its unit, a
consumer sets the count with its own `array` — one `array` per
output, all reading one count in the consumer's recipe. The viewer
builds a control per knob, and its cost follows how the knob is
written: a name bound once and handed whole to one `material` factor
or one `seed` by a top-level call is live (a slider that repaints, a
reroll); any other use — arithmetic, a list, a second use, a
function, an `overrides` — is a field behind Apply that rebuilds the
recipe. Write the knobs you want live that way. A preview is visible
only in the tab that scrubs it — never a `screenshot` or another tab.
The human lands a pick with Accept, which writes `params.json`:
**never edit that file**. Its values outrank your defaults
(`validate` names them under `params_accepted`): when a default of
yours must take effect, ask the human to Reset to defaults in the
panel.

For taste over a stochastic stream ("show me a few options"), sweep
the seed as a contact sheet — read `moraine docs variation-grid` first.

## Build order {#order}

The stages: `moraine serve` on the workspace root first, once a
project holds an asset file (backgrounded; one per workspace —
`port_in_use` names the holder: this workspace's is the one to use,
another's means `--port`; it opens the human's viewer if none is
watching, on the project the human picks) → subject
(against {#scope}) → public surface ({#public}) → asserts → geometry
→ materials → knobs declared and swept. Rules that keep the middle
stages working:

- **Author assert-first** — the test-first habit, applied to assets.
  Before building geometry, list every verifiable requirement in the
  request, then land each as asset content in the same edit that
  introduces the output it names. Solid-based checks stay red until
  the outputs close, by design. An elevated output passes the
  ground checks with no waiver, but nothing verifies what holds it
  up: write the `contact`/`place` (or `mated`) in the same edit.
  Never waive a sunk part — fix penetration by translating; waive
  `not_sunk` only for deliberate below-ground geometry, asset-level
  `grounded` only for out-of-asset support. Build from the
  requirements file's constants (`load` them), so the human's number
  shapes the part and judges it. Never judge an assert of your own by
  a value your asset computes for the geometry: it moves with the
  part and cannot fail (`threshold_tautology` names such an asset
  constant, not an expression over a knob or a loaded one). The
  build then converges against `validate`, not against re-reading
  the request, and the asset documents its own acceptance criteria.
- **Pick the assert kind by what the requirement says** (contracts by
  name: `moraine docs <kind>`): "must not
  interpenetrate" is the built-in guard's job, over every pair and
  copy pair from the first edit — nothing to write;
  `contact` for "touches / rests on", with `min_area` where a part
  must be *seated* — an edge or a corner on a face is distance 0 like
  a face, so without it a stair whose steps graze their stringers
  reads green; `mated` for "butts / fits / docks" — `mode:
  "coincident"` when the two faces register to each other, `mode:
  "contained"` when a smaller face seats anywhere on a larger one;
  `clearance` with the stated gap as `min` for "keep N apart";
  `embedded` (`min`/`max` depth band) for "embeds / sinks into" —
  the guard cedes declared pairs to it: plan embeds freely;
  `extent` / `count` for a stated dimension, position, height off
  the ground, or copy count — a number you would otherwise `query`
  and write into prose. Over instanced outputs, quantify — see
  {#copies}.
- **The file sets the owner, the requirement's provenance picks the
  file**, not who types the entry: a constraint the human stated
  goes in `<name>.requirements.star` even when you draft it — and
  when they stated the goal and you derived the number (the
  requirement is theirs; your arithmetic is translation). The asset
  file is for your own invariants — including every requirement you
  invent when the human set only the subject — and for a
  **substituted checkable claim** standing in for what the human
  asked; record the substitution in the journal, naming their words.
  **Append to a requirements file as you transcribe; change or
  delete a line there only on the human's word.** A request that
  states a requirement's new number approves that one change: make
  it and report the line before and after. Any other you propose,
  and wait: a number you derived, a request that names the change
  but leaves the number to you. A stated dimension whose datum is
  open (center or face, ground or floor) is a question before
  transcription; proceeding without the answer, say so in the
  report. A failing human-owned assertion means fix the geometry,
  not the constraint.
- **Tag, don't index**: reference element sets by `tag` (a predicate
  fixed into the graph as a named face group) — the selection
  re-evaluates under upstream edits; an index list silently goes
  stale. Pick robust thresholds against the faces a later `boolean`
  may add (a surviving group can silently change in *what it
  matches*) and against curvature — `{"normal_y": {"gt": 0.9}}`
  reads "points up" on planar outputs; a cylinder's stepped normals
  slip it, so pair with `center_*`. Tag **after** topology rebuilds
  ({#breakage}). On an instanced output the predicate reads the
  prototype's local frame; pick the copy by index downstream
  (`instance` / `face_instance`).
- **Place and mate by face, not arithmetic**: to rest a part, tag the
  surface and use `place`; to butt, fit, or dock, tag both faces and
  use `mate`, with sibling outputs of the same assembly riding along
  as `[anchor, part, leader]`. A position you compute follows a knob
  too, but it restates where you believe the face is; the op reads
  the face the evaluation built. The op constructs the relation and
  the assertion (`contact` / `mated`) declares it from the evaluated
  outputs; the assertion outlives a later replacement of the op, so
  write both. An arrayed anchor is
  no reason to fall back to arithmetic: tag the prototype and pass
  `instance = <copy index>`. A joint a `boolean` union consumes
  before any output cannot carry `mated` — there `mate`'s structured
  errors and the boolean margins are the guard. A contained joint's
  position is design intent — no op solves it; author the transform
  and declare it with `mated` `mode: "contained"`.
- **Query when a number must be read, not designed**: `moraine query`
  interrogates the evaluated asset with the tag vocabulary — its
  debugger. Query what a tag actually catches (a sliver invisible to
  stats and screenshots shows up in its `area`); when a screenshot
  contradicts the numbers, trust `query` — it reads evaluated
  geometry, no viewer in between.
- **Reuse before re-authoring**: a chain of nodes needed more than
  once is a function; an assembly with its own knobs and asserts is
  a sub-asset under `subassets/`, placed with `asset_ref` — never a
  second hand-built copy; per-use variation comes from `overrides`
  on declared param ids, never a copy of the file with one number
  changed. Asserts inside a sub-asset keep gating in every consuming
  root's `validate`. A reference is a monorepo dependency, not a
  package: when you change a shared asset, fix every consuming root
  in the same edit and prove it with a workspace-root `validate`.
  Contracts and limits: `moraine docs asset_ref`.

## Copies {#copies}

- Instancing stays instancing: `array`/`scatter` emit prototype +
  transforms. Never bake copies by hand. `bake_instances` is
  sanctioned exactly where the next op needs a real mesh — a
  `boolean` cutter, a `place`d or scattered Instances output, an
  anchor spanning arrayed copies, any other mesh-only input (the
  rule is the need, not the list). Everything else stays instanced
  through export (transform the prototype *before* the array node,
  never bake after it); `export --bake-instances` is a per-export
  delivery choice, never a reason for an in-graph bake.
- Copies that differ (a per-copy look, a subset picked by seed) leave
  instancing: separate outputs lose `count` and `every`, one merged
  mesh is unchecked inside ({#done}). Report which you chose.
- Every stochastic op takes an explicit `seed` — same asset, same
  result, always. A stochastic count or seed fails `validate` for
  some values by nature, so it makes a poor knob, and a scattered
  field's requirements go through `every`, never one lucky seed.
- Checks over an instanced output default to the **union** of its
  copies (`contact` is existential — one near copy passes for all).
  Write `every = [name]` on `contact` and `mated`, `every = True`
  on `extent` when you mean every copy, `self = True` on `clearance`
  for spacing between copies. An `unchecked` count in a report is an
  entry that did not: quantify it, or report why not. Overlap that
  *is* the
  construction — a unit embedding into its neighbor, copies weaving
  freely — is a declared `embedded` (`connected` for one continuous
  run), whose `min` also fails copies drifting *apart* — the direction
  no overlap check sees. A positioned copy gets `mated` on its faces.

## How meshes break {#breakage}

- **Topology rebuilds (`boolean`, `extrude`, …) lose attributes**
  (tags, UVs with their material, curved shading): each loss reaching
  an output is warned, and the recovery (`tag`, `unwrap` →
  `material`, `smooth_normals`) goes **after** the rebuild. One hole
  is silent: a boolean output with **no** material gets no
  `flat_shading` warning, so an untextured curved part still needs
  its `smooth_normals`.
- **Boolean margins**: union operands must clearly overlap (≥ 1 mm
  embed) or clearly protrude; a cutter must clearly overshoot the
  faces it cuts. Exactly-coplanar contact breeds degenerate sliver
  triangles (`validate` flags them at the source, judged relative to
  shape, so a pass survives `asset_ref` transforms). **Embed, never
  butt-join; overshoot, never kiss.**
- **Symptoms surface downstream of causes**: a broken input flows
  through later ops and fails somewhere else. The kernels refuse the
  known cases loudly (an inverted `extrude` direction, `bevel` on
  non-convex input, displacing a flat-shaded mesh) — when an op
  refuses, fix the input it names; never work around it in a way
  that re-silences the defect.

## The look {#look}

- Pipeline per part: geometry → `unwrap` (UVs) → `material` as the
  **last** node, after any topology rebuild ({#breakage}).
- `material` `name` is the export identity: reuse one name+params
  across parts to share the material (one draw state); same name
  with different params fails `validate`. An `overrides` that
  reaches a referenced kit's material changes its content under the
  kit's name — give that use its own identity with
  `material_rename = {"<kit name>": "<this use's name>"}` on the
  `asset_ref`. When the engine target has a material budget, declare
  it for the asset you export: `mo.budget(materials = N)` in its
  requirements file gates `validate`; an undeclared budget is not
  checked (`moraine docs budgets`).
- Set factors before reaching for textures — a factor is free, a
  texture costs cache and GLB bytes. Texture values are LINEAR
  (colors too — no sRGB math in assets).
- Legibility comes from materials, not geometry: faces with the same
  normal shade identically at any depth, so an opening or recess
  reads only through a material difference — and parts sharing one
  material must not sit edge to edge on screen, or they merge into
  one shape. `emissive` adds brightness, not light or transparency;
  never model colored layers as concentric opaque solids.
- The patterns that keep large surfaces legible are in the
  reference: `moraine docs patterns`, read before texturing one.

## When is it done {#done}

Edit → the viewer hot-reloads (<2 s when cached) → verify in tiers,
cheapest first.

1. `moraine validate` — the built-in checks (contract: `moraine docs
   checks`) plus your asserts, which gate the same way; run it first.
   `--failures-only` collapses healthy outputs and `unchecked` to
   counts. After editing a project that other projects reference,
   validate the **workspace root** — the sweep is the only check that
   reaches the consumers (`not_swept` names assets it could not).
2. `moraine stats` — verts/tris/bounds, for reading; a dimension
   that must hold is an `extent` assert (it and the ground checks
   measure actual vertices, where an instanced output's stats
   bounds transform the prototype's box corners).
3. `moraine screenshot` — pixels, your call: shoot when a number
   cannot settle the question, never to re-check a claim a green
   assert already proves, and never ask the human to relay what the
   viewer shows when the endpoint can tell you. Framing a decisive
   shot: `moraine docs screenshot`.

Acceptance is the human's; your report is what they decide on. Say
which tier proved each claim, and when you did not look, say so —
"validate green, not shot" is a true report, "verified" is not.

**What green does not prove.** `validate` proves the checks it lists
— not visual usefulness, and not every structural property. Still
green:

- An enclosure formed by several outputs together — `not_enclosed`
  catches only an output fully inside another **single** opaque
  output (waive an intentional interior part:
  `mo.waive("<output>", "not_enclosed")`). In an
  assembly, count contributing outputs yourself.
- Counts that must agree across outputs (one arm longer than its
  rail) — derive both from one name, or guard the seam with `mated`.
- Parts merged into one output overlapping or drifting apart — no
  check looks inside an output.
- Whether `emissive` reads as a light source — no tier measures
  light: the viewer draws brightness, the engine decides lighting.
- Anything you never asserted.

## Engine handoff {#handoff}

Before an engine handoff, run `moraine validate --profile gltf` and
read the GLB contract and the import checklist: `moraine docs export`.

## When the human points {#pins}

"This one" or "here" in the human's message — whatever the language
— usually names a pin they clicked in the viewer. Before acting on
such deixis, run `moraine pins list` and act on the structure it
returns, never on a guess from a screenshot; the pin's lifetime and
how to keep a pointed selection: `moraine docs pins`.

## Project journal {#journal}

- Before working in a project, read its `journal.md` and, if
  present, the workspace `journal.md` — the `## Entries` index; a
  body under `## Details` only when a relevant entry points at it.
  Format, stamps, retirement: `moraine docs journal`.
- Record what the workspace cannot say: rationale behind dimensions
  and construction choices, the human's corrections, accepted and
  rejected variants, approaches that failed. The asset records the
  *what*; the journal is where the *why* survives the session. Never
  record what the workspace already derives (counts, bounds,
  structure). Skip recording when the human says the session is off
  the record.
- One entry is one **observation** — a claim line plus its
  grounding, terse (a body that grows moves under `## Details`).
  Stamp what the tool's version can falsify (`[moraine <version>]`);
  a workaround stamped with a version you are not running is a
  hypothesis: reproduce it in the asset, then delete or re-stamp.
- Entries are declarative observations, never imperatives — one pick
  is one pick, not "user prefers X". An entry that wants to be
  phrased "always do X" is a promotion candidate: record the
  observation, flag it, and say so — moving a rule into `AGENTS.md`
  is the human's act.
- Place each entry at the narrowest level that covers it (workspace
  `journal.md` only for observations you already know are
  project-independent); retire as you append — an observation the
  asset can now carry as an assert, an entry a later entry
  contradicts, and a promoted observation all leave the journal.
  Write in the human's working language, ids and op names verbatim,
  and terse — the index is read every session. The journal biases
  your proposals; it never constrains them. Never store project
  knowledge in account-side memory; workspace files are the only
  source of truth.
