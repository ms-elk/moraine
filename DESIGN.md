# Moraine — Design Document

*A living design document — the normative source of truth for
Moraine's design, updated with the implementation it governs.*

**Authorship:** Designed with Claude, implemented by Claude via Claude Code.\
**Platforms:** Windows and macOS (Apple silicon).

---

## Concept

> **Moraine lets you build, refine, and extend 3D assets together
> with an AI agent — the agent writes the recipe, tests verify your
> declared requirements, you decide what they become.**

## 1. Overview

Moraine is an AI-native procedural modeling tool for real-time 3D
assets. It began as the tool its author — a 3D graphics engineer
without modeling craft — wanted: a way to make the 3D assets the
author wants. The author is the tool's calibration floor, not its
ceiling. If it works with no craft, it works, and what it can make
scales with the direction a user brings. The audience is anyone who
can specify a 3D asset and judge the result — graphics engineers,
technical artists, 3D artists. The more craft and eye a user
brings, the sharper their half of the loop becomes: intent, taste,
and acceptance (P4) scale with expertise.

As an AI-native tool, Moraine is designed along two lines:

- **For the agent,** an authoring interface designed for agents, with
  all deterministic work — evaluation, verification, and assembly —
  handled by the tool (P2, P6, §8).
- **For the human,** an interface for directing that work: declaring
  requirements, pointing at geometry, exploring parameters, and
  deciding what ships (P4, P5).

The name comes from glacial moraines — landforms deposited by a procedural
process. A sibling project by the same author: `alpine` (CPU path tracer).
Moraine produces the worlds that renderer draws.

### 1.1 What Moraine produces

Real-time 3D assets: game-engine-grade meshes and PBR materials usable in
games, AEC/manufacturing visualization, digital twins, XR, and web product
viewers. The first export profile is `gltf`, verified on Unity (URP)
via glTFast (§9.1). The guarantee Moraine makes is *engineering
quality*: correct real-world scale, clean topology, batching-friendly
materials, deterministic naming, style-consistent modular kits, and a
part hierarchy with its instancing kept through export — the
structure that pivots and LODs are added to (§9.1).

### 1.2 Why Moraine (positioning)

- **vs. asset stores:** stores require searching and settling, and
  the exact asset may not exist. Moraine builds what is declared and
  verifies it. The cost is production time, but only for the *first*
  asset: a generator is a reusable capital good whose marginal cost
  per variation approaches zero, and it is consumed *by other
  assets*. Authored once, a door generator yields a whole family of
  variants, instantiated across a kit's buildings at per-use values.
  Recipes can circulate between users
  (Future Work: Library & commons) as fork-and-adapt starting points,
  never as fixed assets to settle for.
- **vs. image-to-3D generators (Tripo/Meshy etc.):** their mesh
  quality is improving fast, and the raw engine-readiness of a single
  generated mesh should be expected to keep improving. That gap is
  not the moat (Future Work: Generative nodes plans to consume these
  as generator nodes for sculptural form). The structural difference
  is what the output *is*. A generation is a one-off sample:
  approximately what the image showed, never exactly what was
  specified, with nowhere to declare scale, dimensions, mating, or
  budgets, and no source that regenerates it after a manual edit. A
  recipe is exact to what is declared, refined by a diff to the file,
  and consistent across a kit by construction ("the same crate,
  200 mm wider" is one override). The engine-ready guarantees come
  from the graph, never from a generator's best effort.
- **vs. existing DCCs + AI plugins (Houdini, Blender, Maya):** those
  tools spent decades optimizing UIs for *human* hands and eyes, and
  AI is retrofitted on top. Even where a headless API exists, the
  session is stateful, the representation is token-expensive to read,
  and evaluation carries no determinism or cache contract. Moraine
  inverts this: the AI-readable representation (a hermetic Starlark
  recipe, the operator graph it expands to, a wire protocol,
  structured tool responses) is the source of truth, and the human UI
  is a *projection* of it. Counter-hypothesis,
  stated honestly: incumbent DCCs may close the gap for
  language→node-network generation. What they cannot cheaply retrofit
  is first-class AI access to all state, token-efficient
  representations, automated verification loops, and operator
  semantics designed for AI authors. The last is the hardest, because
  redesigning a node library around predicate addressing and
  intent-sized steps would break decades of muscle memory and asset
  compatibility. Moraine competes on these.
- **vs. an agent scripting a DCC:** the strongest alternative for an
  agent. A Python script against a DCC's API is software too, so the
  cost of writing one is no argument. The difference is what the
  agent must build around the geometry. In the script every relation
  the asset has is the agent's code: the checker, tested by the agent
  that wrote the geometry; the assembly that joins the parts' exports;
  the propagation of a part's parameter into that assembly. Each
  request grows it, for the next agent to read in full and trust as
  code. In Moraine the prover and the assembler are the product's,
  and the agent writes a declaration, a line per requirement, read by
  a successor against the requirement rather than as code. Where a
  requirement outruns the declared vocabulary the agent writes a
  checker here too, as the exception a successor can find. The agent
  still reads the recipe, but what goes is the code that was never
  the request.

The common thread: Moraine's working artifact is a recipe that
carries its own requirements. Each requirement is declared in the
file, has an owner, and is verified by the tool on every round. Once
accepted, it keeps gating every later round until its owner revises
or retires it (§5.2), so a requirement is kept, not frozen.

AI-native, here, means that the tool, not the agent, owns the
modeling concepts and handles the deterministic work built on them.
The agent interprets intent, authors and revises recipes, and
diagnoses failures. The tool assembles referenced parts with per-use
overrides, verifies declared contact and clearance, and resolves
what the human pointed at into a structured target. Each result
comes back as actionable feedback (§8.2). A request therefore costs
the agent only the change it names, not the machinery around it.

The next request is the easy one: what was built is a part to
consume, what was accepted is a requirement that still holds, what
the human points at is the target. An asset grows by requests, across
sessions and hands, instead of being rebuilt. This is the Concept's
*extend* clause.

### 1.3 Scope and non-goals

**Scope.** Moraine deliberately targets **rule-describable
structure**: mechanical parts, furniture, architecture, modular
kits, and the natural forms procedural rules generate (terrain,
rock outcrops, foliage, scatter fields). The boundary is
representational, not aesthetic. What stays out is *sculptural*
form — shapes whose quality lives in vertex-level nuance rather
than in any nameable degree of freedom, regardless of model
capability (Future Work: Generative nodes). Two strengths coincide here:
it is the LLM's strongest modeling domain (operator-graph
construction is essentially programming), and it is procedural
methodology's home turf, as three decades of Houdini practice show.

**Non-goals.** The exclusions below follow from that alignment.
Implementers must not add sculpt-leaning features opportunistically.
The planned path for sculptural form is a different one entirely
(Future Work: Generative nodes).

- Characters and sculptural shapes (creatures, cloth drape, hand-sculpted detail). See Future Work: Generative nodes.
- Skeletal animation (bones + skinning). See Future Work: Animation.
- Scene-level cinematics / keyframe editors (engine timeline territory).
- Photoreal hand-painted texturing (procedural materials only; external image
  assets may be referenced later via `external/`).
- Re-implementing Houdini's node library. Operators wrap mature libraries.
- A graph-editing GUI for humans. Humans scrub parameters; the agent
  edits structure.

---

## 2. Design Principles

**P0 — AI-first architecture.** The machine-readable representation is the
source of truth; every human-facing surface is a projection. Every principle
below is a corollary.

**P1 — Non-destructive by construction.** The asset is a recipe and the
operator graph it builds, never baked geometry. Evaluation is
re-derivation. History is the recipe plus its version control.
(Houdini's .hip stores recipes, not geometry. Moraine adopts the same
separation with a diff-friendly text format.)

**P2 — Token economy.** The agent is invoked only for what requires
intelligence (authoring/editing recipes, diagnosing from screenshots).
Everything deterministic — evaluation, validation, caching, export — runs
locally with zero agent involvement. Tool responses are compact
structured data and errors are machine-readable and terse, under the
rules of §8.2.

**P3 — Don't reinvent operators.** Geometry kernels come from mature
libraries (manifold3d for booleans, trimesh for mesh plumbing, numpy for
array math, xatlas later for unwrap). Moraine's runtime is an orchestrator,
structurally similar to trimesh itself: pure-Python glue over compiled kernels.

**P4 — Roles are explicit.**
- *Agent:* interprets intent, authors recipes (§4.8), transcribes
  requirements (§5.2), and iterates on structured feedback (§8.1).
- *Tool:* owns the modeling concepts and deterministic execution —
  assembly, evaluation, caching, verification, export, and serving
  the viewer. Never calls the agent.
- *Human:* owns intent, requirements (§5.2), taste, and acceptance at
  every round, and directs the work through requests, pointing,
  parameter scrubbing, and selection among variations. What ships,
  and what the asset becomes, is a human decision (§1.2). Humans do
  not edit graph structure through the UI.

**P5 — Interfaces per audience.** Humans get visual/spatial/comparative
surfaces (viewer, sliders, variation grids). AI gets textual/structural
surfaces (JSON, protocol, validation reports). Bridging widgets
translate between the two. 3D deixis is the shipped one (click a
primitive → a structured pin the agent reads, §7.2). Further
directions are in Future Work: AI-native UI.

**P6 — Operators are designed for AI authors, not human hands.** Human-first
DCC operators assume a mouse and an eye: interactive selections, perceptual
micro-iteration, errors caught visually. Much of good operator design is
shared. The divergence is on exactly three axes, and Moraine goes
all-in on the AI side of each. *Addressing:* elements are referenced
by predicate (angle threshold, orientation, provenance, tag), never
by index lists. Indices break under upstream edits, predicates
re-evaluate. *Step size:* one operator ≈ one sentence of intent —
one *verifiable postcondition*, so operator size is judged by
testability, not prose length. Spatial arithmetic the agent would
derive wrongly (the steady error class the before/after visual diff
direction records — Future Work: AI-native UI) moves into the kernel
as semantic anchors and relative frames.
*Loop closure:* where a human's eye catches a bad result, the agent needs
a numerically verifiable contract — documented effects on stats,
assertable without pixels, and structured actionable failure (§8.2). The
admission rubric in §5 operationalizes this principle for every
operator added.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Agent harness (intelligence; external to the tool)        │
│  edits recipes, calls CLI, reads reports                   │
└───────────────┬────────────────────────────────────────────┘
                │ files + CLI, or MCP (§8.4)
┌───────────────▼────────────────────────────────────────────┐
│  moraine runtime (Python 3.12)                             │
│  recipe build (Starlark) · graph interpreter ·             │
│  hash cache · validators · exporters                       │
│  kernels: numpy / trimesh / manifold3d (wheels)            │
│  local HTTP + WebSocket server (Viewer Protocol)           │
└───────────────┬────────────────────────────────────────────┘
                │ Viewer Protocol (§7)
┌───────────────▼────────────────────────────────────────────┐
│  Clients: three.js browser viewer (first)                  │
└────────────────────────────────────────────────────────────┘
```

- **Language:** Python 3.12. Rationale: the geometry-kernel ecosystem is
  Python-bound. Recipes are Starlark (§4.8), evaluated in-process by a
  compiled interpreter shipped as a wheel, so an edit costs no compile
  step (fast agentic loops). Performance-critical paths already live in
  C++ kernels. Escape hatch: any single hot operator may be rewritten in
  Rust (PyO3) or numba behind its stable signature.
- **Dependency rule:** every dependency MUST ship binary wheels for Windows
  and macOS (Apple silicon) on PyPI. This is what guarantees `pip
  install` works without a compiler on both platforms. Intel macOS is
  not a supported platform. The Starlark interpreter's wheels do not
  cover it, and Moraine maintains no interpreter of its own.
- **Performance model:** the viewer renders cached meshes at 60 fps (GPU
  budget). Graph evaluation is asynchronous to rendering with an
  interactivity budget of ~100 ms–1 s per dirty-subgraph
  re-evaluation. The 16 ms budget belongs to clients, never to the
  runtime.
- **Implementation discipline:** array-oriented code only. No per-vertex or
  per-pixel Python loops. Vectorize with numpy or push into a kernel.
- **Package layering:** the runtime is a strict stack. A layer imports
  only itself and what lies below it (lazy imports included), and the
  import graph is acyclic. Each layer is a folder of the same name,
  except `errors`, a single module at the package root, beside which
  the root `__init__` holds only the installed-version lookup
  (`dist_version`). A module sits
  in the layer of the concern it implements, not in the lowest layer
  its imports would allow (the project journal is a workspace
  document and so is `core`; the screenshot camera is viewer
  vocabulary and so is `app`, though it imports nothing above
  `errors`).

  ```
  ┌─────────┬────────────────────────────────────┬────────────────────────────────────┐
  │ app     │ entry points: CLI, server + viewer │                                    │
  │ recipe  │ Starlark build: recipe → IR (§4.8) │ imports only errors, core, ops     │
  │ engine  │ graph evaluation, export           │                                    │
  │ verify  │ validation over evaluated values   │ reads values, never the evaluator  │
  │         │ + assert kinds (`asserts/`)        │ a kind: core, kernels, `common`    │
  │ ops     │ operators + their shared helpers   │ imports only errors, core, helpers │
  │ cache   │ store, records                     │                                    │
  │ core    │ data model and codecs              │                                    │
  │         │ + assert vocabulary (`asserts/`)   │ a kind: `numbers`, `common`        │
  │ errors  │ the shared floor                   │                                    │
  └─────────┴────────────────────────────────────┴────────────────────────────────────┘
        imports point down only                  (rules beyond the order)
  ```

  The right column holds what the order alone does not say. `verify` sits
  below `engine` by design: it reads evaluated values and never the
  evaluator, so a verdict cannot depend on how evaluation is implemented.
  `recipe` sits directly below `app`, so only the entry points import
  it. It imports only `errors`, `core`, and `ops`, because its
  builtins are generated from the operator registry and the assert
  registry and a build never evaluates (§4.8). The interpreter is
  imported under `recipe` alone, and the layers that read the IR
  (`cache`, `ops`, `verify`, `engine`) import neither the recipe-file
  rules nor the workspace roster in `core`. So a change of recipe
  language stops at `recipe`, those rules, and the entry points.
  An operator imports only `errors`, `core`, and the `ops` package
  below its root: the shared helpers and, where one operator's mesh
  mapping is another's building block, a sibling operator. The
  registry (the `ops` package root) and `cache` are closed to it. So
  adding an operator cannot break another and every operator is
  testable alone. The assert vocabulary has the same shape
  on both sides of the layer line — one kind, one module: its
  parse-time shape and defaults under `core/asserts/`, its evaluation
  under `verify/asserts/`, a registry at each package root, and a kind
  module knowing only the kernels and its side's shared helpers (the
  evaluation side also reads its own parse-side twin and the parse
  side's shared helpers) — never a registry, a sibling kind, or the
  dispatcher. A kind is added by two files and two registry lines.
  Tests pin the layering and keep the two registries equal.
- **Packaging:** import name and PyPI distribution name are one
  identifier, **`moraine3d`**. `moraine` was taken on PyPI (name and
  top-level import alike, inviting same-environment collisions), and
  a separator-free name keeps install and import identical — the
  `open3d`/`manifold3d` pattern. So an agent's guess is always right.
  The command, cache directory, and workspace dir stay `moraine`; the
  brand stays **Moraine**. Repo: `ms-elk/moraine`. Nothing importable
  from `moraine3d` is public, so its modules move between releases
  without notice.

---

## 4. Asset Format

An asset has two forms. The **recipe** is a Starlark file (§4.8): the
**save file**, the **unit of version control**, and the **surface the
agent edits**. The **IR** is a JSON document, an operator graph with
parameters, written by `moraine build` as the recipe's deterministic
expansion. The IR is the input to evaluation, the cache, validation,
export, the viewer protocol, and MCP. It is a build artifact: written to
`build/` inside the project, regenerable, and never hand-edited (a hand
edit is overwritten by the next build, which is the whole guard). §4.1
states the IR's rules and §4.8 the recipe's. The sections between hold
what both share: layout, conventions, determinism, the cache,
references, and the journal. Everything the runtime does past the build
reads the IR alone, so an operator, an assert kind, a cache record, or
an export profile never knows a recipe exists. The recipe exists so that
a value stated once is derived everywhere it is used and a repeated
shape costs its rule, not its expansion.

IR files carry the extension **`.mora`**. The content is plain JSON, and
the extension marks the one file only the build writes and only the
runtime reads, so a glob finds every IR and never `params.json` or any
other JSON a project holds. An IR keeps its recipe's name: the root
asset's is `build/<name>.mora`, a sub-asset's
`build/subassets/<name>.mora`, and an asset used under overrides has
one IR per override identity beside its own (§4.6). There is exactly
one **root asset** per project, the one
asset file at the project directory's top level, so CLI commands can
take just the project directory (§4.2).

### 4.1 Format rules

- **One node per line; fixed key order within a node.** Rationale:
  a readable, diffable artifact. The build's `--diff` (§6) reports
  over it, and a human debugging a build reads it. The same
  rule covers every root container that grows per output. The lists
  (`nodes`, `params`, `asserts`) and the `outputs`/`groups`/`checks`
  objects serialize one entry per line, because an entry-per-output
  key must review, diff, and merge entry-wise. Fixed-keyset root
  objects (`budgets`, `provenance`, `asset_checks` — it grows per
  check kind, never per output) stay inline.
- **`"build_version"`** at document root, the version of the build
  that wrote the file (§4.4). It is an identity check, not a schema
  number. The runtime reads the IR its own build wrote, so a file
  carrying any other stamp is stale, and the loader refuses it
  (`version_mismatch`) naming `moraine build`. The IR has no version
  of its own because its shape is the build's output, so a change to
  it is a `build_version` change already. Nothing migrates an IR,
  because nothing needs to: the recipe is rebuilt. Compatibility
  lives on the recipe side. A change to the build's contract (§4.4)
  ships with its refusal, so a recipe written against the older
  contract fails the build naming the new spelling (§4.8), never a
  rewrite. A change that needs the author's decision (which
  assert replaces a retired one, whether an overlap is intent or
  defect) ships the same way, and the checks the change strengthened
  go red on the real cases.
- **Deterministic unique `id`** per node, derived by the build from
  the recipe's names (§4.8). All references — edges, selections,
  cache keys, chat↔graph links — use ids, never array indices or
  display names. A named node keeps its id until the recipe renames
  it, which is what lets the cache, the pins, and the journal keep
  naming a node across rounds. Every node also carries its `site` —
  the recipe file and line that created it, workspace-relative — so a
  report never makes its reader search the recipe for a node. A node
  made inside a function carries `via` as well, the calls it was
  reached through (§4.8). The loader reads an IR with or without
  them, so an IR handed to the runtime by path (§6) needs none. They
  are location, not content, and join no cache key.
- **Named outputs:** `"outputs": {"crate_a": <node-id>, ...}`,
  declared in the recipe by `mo.output` (§4.8). One
  graph may produce a whole modular kit. Export names derive
  deterministically from output names, so re-exports never break
  engine-side references. Because they become cache/export filenames
  and engine object names, names match `[A-Za-z][A-Za-z0-9_-]*` and
  are unique case-insensitively. The order outputs are declared in
  carries no meaning (§9.1).
- **Root `"params"`:** the asset's declared knobs (§4.8), one entry
  per line in declaration order. An entry holds the knob's `id`, the
  `value` the build resolved for it, the declaration's `label`,
  `min`, `max`, and `step`, and the `site` of the call that declared
  it. `from` names the source of a value that is not the recipe's
  default, `accepted` (the project's `params.json`) or `override` (a
  consumer, §4.8), and `default` then holds the recipe's. `mismatch`
  holds an accepted value of another kind
  than the knob's, which the build did not take. `node` and `param`,
  present together or not at all, are the build's record that the
  value reached exactly that one node param whole (`direct`, §7.2),
  and the two values agree. The schema (§7.2) and the report (§6)
  read the value from the entry, so neither depends on where the
  recipe used it.
- **Reserved name `time`** (Future Work: Animation): refused as a
  public parameter id (`reserved_param_id`) and, as a root key, like
  any unknown key, with the error naming the reservation. No file
  carries it until an operator consumes it (§11).
- **Optional root `"checks"`:** per-output validation policy —
  waivers only, `"checks": {"<output>": {"<check>": false}}`. Keys
  must name declared outputs and known checks, and any value but
  `false` is refused (never silently unchecked; an opt-in spelling
  stays reserved). The waiver lives in the asset so the intent is
  documented and survives edits. Evaluation ignores the key. The
  waivable checks: `not_sunk` — the output's true minimum y no lower
  than tolerance below the ground plane — for geometry that
  intentionally extends below ground (a foundation, a terrain slab);
  `not_enclosed` — full enclosure by an opaque output, judged by
  manifold3d volume containment, never a bounds test (opaque means
  base-color alpha exactly 1) — for a deliberate interior part. An
  instanced output fails `not_enclosed` only when every copy is
  enclosed, and is never a container (per-copy enclosure has no
  single answer). A green entry names the output under
  `unchecked.instanced_copies` whenever some copy's box lies inside a
  container's box. The ground checks split by direction: sinking is
  an output property (the report names the culprit; the fix is one
  upward translate), resting on the ground an asset property
  (`asset_checks`, below). A per-output two-sided ground check was
  rejected: multi-part assets ground only their base, so it turned
  routine elevation into a waiver re-declared by every referencing
  asset. Instead, an elevated output passes with no declaration, and
  what holds it up is stated where support lives, as a
  `contact`/`mated` assert (§5.2). The unwaivable geometry checks
  judge by shape, not absolute size: `no_degenerate_tris` flags a
  triangle whose height over its longest edge is ≤ 1e-9 m. An
  absolute-area bar sits inside fp noise for meter-scale slivers, and
  this bar is four orders above the perturbation a baked transform
  introduces, so a producing asset's green survives `asset_ref` reuse
  while a latent sliver is red at its source.
- **Optional root `"asset_checks"`:** asset-level policy over the
  outputs as a whole, the same waiver-only spelling in one flat
  object (`"asset_checks": {"grounded": false}`), reported as the
  top-level `asset_checks` list and gating like the per-output
  checks. `grounded` — the true minimum y across all geometry outputs
  reaches the ground plane within tolerance — is the floating
  direction of the split above. Red means the whole asset hovers,
  and the fix is one downward translate. The waiver documents support
  outside the asset (a hanging fixture; a parts-bin sub-asset).
  Deriving groundedness from support asserts was rejected: a consumer
  output reaches a referenced asset's asserts through no provenance
  that covers the general case (a boolean or a re-scatter erases
  which sub-asset output a value was), so every referencing asset
  would re-declare its supports — the disease the split removes.
  `checks` inheritance through `asset_ref` falls to the same
  argument: routine elevation no longer declares anything.
- **Optional root `"groups"`:** output groups — one object after
  `outputs`, `"groups": {"shelf_a": ["shelf_a_frame",
  "shelf_a_posts"]}`, one group per line. A name for a piece built
  from several outputs, taken by an assert wherever it takes an
  output name (§5.2). Groups and outputs share one namespace
  (collisions refused, case-folded like output names). A group holds
  two or more distinct declared outputs, never another group. An
  output may belong to several groups. Absent when the asset declares
  none. Evaluation, export, and the cache ignore the key.
- **Root `"asserts"`:** declarative constraints as asset content,
  checked deterministically at eval time and reported through
  `validate` — gating, never waivable. Each entry carries the `site`
  of the recipe call that declared it, and under `from_param` the
  knob a threshold was taken from, where one was (§4.8). The format
  rule ends here; the semantics live in §5.2.
- **Root `"budgets"`:** asset-declared acceptance budgets, one inline
  object after `asserts` — `"budgets": {"materials": N}`. Acceptance
  criteria are human-owned and belong in the reviewable recipe, not
  in a tool flag: a declared budget **gates** `validate` like an
  assertion (the report's `budgets` list; raising the number is an
  asset edit the human approves), and an undeclared budget is **not
  checked at all**. There is no built-in default, since an advisory
  nobody declared goes unread. `stats` reports the same counts over
  the same scope so headroom is visible before declaring. A budget
  counts what the asset's outputs **ship**, reference-borne content
  included. Budgets are performance bounds, performance is a property
  of the exported asset, so an assembly's budget bounds its own glb.
  (Counting only what an asset authors was rejected: an assembly's
  budget would then bound nothing it ships.) Unlike
  asserts (§4.6), a sub-asset's declared budgets are not re-run by
  consumers: each asset's declaration gates in its own `validate`.
  Vocabulary: `materials` — distinct shipped material identities
  (counting: §9.1); `triangles` — rendered triangle total, instance
  counts multiplied in (the `stats` number); `parts` — exportable
  parts (§9: mesh outputs; an instanced output is one part, an image
  output none; `objects` was rejected as naming nothing in the asset
  format or glTF). `materials` is the kind `validate` checks.
  `triangles` and `parts` are refused as unknown today, never
  silently unchecked. Evaluation ignores the key.
- **Instancing is first-class:** scatter/array nodes emit
  `{prototype: <node-id>, transforms: [...]}`, never merged geometry.
  Merging is the explicit operator `bake_instances` — the one
  deliberate Instances → Mesh opt-in, unblocking `boolean` cutters.
- **Vertex attributes** use glTF's semantics verbatim (`COLOR_n` /
  `TEXCOORD_n` / `TANGENT` — §4.3's ecosystem-standard rule). Each
  op's semantics contract declares per attribute whether it is
  *preserved* or *dropped* — never silent (§5 rubric 5). `validate`
  warns on drops that reach an output — the slot still absent at a
  root output the value flows into. A loss re-authored downstream, or
  a value that only cuts or anchors, is not reported (a warning on
  every cutter trains waiving). The same trait pattern
  (`ATTRIBUTE_POLICY`) governs materials (`MATERIAL_POLICY`), named
  face groups (`tag`), and shading (`SHADING_POLICY`): a `boolean`
  output with a curved surface reaching `material` with its flat
  shading intact gets a `flat_shading` warning, per material node,
  walking across `asset_ref` boundaries (foreign sources labeled
  `asset:id`), gated on face-normal spread at `smooth_normals`'s
  default threshold (an all-planar output shades identically either
  way), advisory and never gating. Whether an op's own flat output
  is named as the source is its declaration (`FLAT_SHADING_SOURCE`):
  `boolean` declares it and `bevel` does not. Every such
  warning names an edit
  site, so it is raised for root-asset nodes only. A node inside a
  referenced asset is the producer's, reported by the producer's own
  validate. `SHADING_POLICY` `replaced` ends the causal chain
  (`smooth_normals` rebuilds shading; `boolean`/`bevel` emit flat
  output regardless of input), and `CARRIED_INPUTS` names the inputs
  whose surface reaches the output — the ones the walk follows —
  leaving out reference-only inputs (`scatter`'s target, `place`'s
  anchor base), which never reach the output surface and are not
  walked.

### 4.2 Directory layout of a workspace

```
workspace/            # one git repository = one workspace
  AGENTS.md           # human-owned workspace rules (multi-agent standard; scaffolded once by `moraine init`)
  journal.md          # agent-owned workspace-level journal (§4.7; created on first entry)
  .moraine/           # tool territory: the shipped authoring guide, synced to the install (committed)
  <project>/
    <name>.star               # the root asset's recipe (§4.8; agent-owned; committed; one per project, name free)
    <name>.requirements.star  # its human-owned requirements: asserts, budgets, requirement constants (§5.2; optional; committed)
    params.json       # human-accepted knob values (§7.2; committed; written by the viewer's accept)
    journal.md        # agent-owned project journal (§4.7)
    lib/              # optional definition-only recipe files, loaded by the assets (§4.8)
    subassets/        # optional sub-asset recipes referenced by the root (§4.6)
      <name>.star                # one sub-asset (agent-owned; committed)
      <name>.requirements.star   # that sub-asset's human-owned requirements (§5.2; optional; committed)
    external/         # non-regenerable inputs (committed)
    build/            # the IR the build writes (§4; regenerable, never edited; ignored or committed, the human's call)
    exports/          # profile outputs (gitignored or released)
```

One repository is one **workspace** bundling any number of projects
(one repository per asset proved too heavy to manage). The
**project** is the unit of evaluation, reference scope (§4.6), and
export: one root asset per project directory. An **asset file** is a
`*.star` file that is not a requirements file (`*.requirements.star`,
§5.2). Resolution rule: a directory with exactly one top-level asset
file is a project. A second one is refused, naming `lib/` for
definitions and `subassets/` for parts. A directory holding
`.moraine/` and no top-level asset file is a workspace. A directory
holding both is invalid. Every classification site refuses it with
the fix named, and `init` refuses a target holding a top-level asset
file rather than create the conflict. (Reading the combination as
a one-project workspace was rejected: one stray root asset would
silently flip the root's classification, and the one-asset case is
already the standalone project directory.) The IR under
`build/` never classifies anything. The root's name is free,
typically the asset's. The project's identity stays its directory
name (§4.6) whatever the file is called, and a named file is
recognizable wherever it is opened or copied. (A fixed root name was
rejected: it made every copied root anonymous, named the root and a
sub-asset by two rules, and forced a rename at each vendoring.)
`moraine init` scaffolds the workspace form above. `serve` over a
workspace is §7.2.

**Discovery.** Projects live directly under the workspace root. The
sweep examines only the root's immediate child directories, skipping
dot-directories and the reserved layout names (`subassets`, `lib`,
`build`, `exports`, `external`, `custom`, matched case-insensitively).
`custom` names nothing the tool reads. It is held for what a workspace
defines beyond the shipped vocabulary
(Future Work: Operator extensions), since a name a project has
taken cannot be reserved afterwards. A reserved name is
one a project directory may contain and never be, in a workspace or
outside one. A directory of that name holding a top-level asset file
is refused wherever a project is classified
(`reserved_project_name`), with the rename named. Skipping it would
leave an asset that evaluates when addressed and is missing when
swept. A child
directory with a top-level asset file is
a project, its directory name is its identity in sweep output, and
sweeps run in lexicographic name order — deterministic, no manifest.
The sweep names what it did not reach: asset files below the
project level ride the sweep report as `not_swept` (always present,
so the empty list is the positive statement). Arbitrary-depth
discovery is not rejected, only deferred: relaxing root-direct
later breaks no workspace, while the bare rule applied at depth
would need exclusion machinery just to skip `subassets/`.

**Ownership of the root files.** Content belongs to the human
(`AGENTS.md` — the multi-agent instruction-file standard, so a
workspace is readable by any authoring agent), the guide under
`.moraine/` to the factory, journals to the agent. Inside a project,
the requirements files and `params.json` are the human's, the asset
files, `lib/`, and the journal the agent's (§5.2). `init` syncs the
guide: overwritten to the installed version when they differ,
reported (created/updated/current), never merged. `AGENTS.md` is the
one create-once file `init` never touches again. It keeps a prose
pointer to the guide for harnesses that read it without an import
mechanism
(prose survives rewriting and translation). A harness that does not
read `AGENTS.md` needs its own instruction file importing
`AGENTS.md`, which reaches the guide through that pointer. That
file is the human's, never scaffolded and never written by the
onboarding skill (§8.5): the tool ships no harness-specific file, so
it accumulates no adapter per harness and never competes with a
file the human keeps. No `.gitignore` is scaffolded either: whether
`build/` and `exports/` are ignored or released is the human's
call.

**Language.** Shipped content — the guide, operator docs, templates as
scaffolded — stays English, one versioned source. `AGENTS.md` and the
journals are written in the human's working language (default: the
language the human uses in conversation), with node ids, output
names, and op names verbatim: the journal's primary reader is the
human who prunes it.

**Sub-asset root.** A recipe whose parent directory is named
`subassets/` belongs to the project one level up, so a project
directory must not itself be named `subassets`. The layout is
enforced at build and at evaluation: an `asset_ref` path must be
exactly `subassets/<name>.star` in the recipe and
`subassets/<name>.mora` in the IR (one level). Starting strict is the
compatible direction; nesting can be allowed later.

**The cache is not part of the project.** It is tool-managed and
machine-level, in the OS cache directory (`%LOCALAPPDATA%\moraine\Cache`,
`~/Library/Caches/moraine`; `MORAINE_CACHE_DIR` overrides). Keys are
content-addressed (§4.5), hence location-independent: one cache
serves every project, and users never see, back up, or gitignore it.
A workspace-local cache (gitignored, `Library/`-style) was rejected
as the default and stays the explicit opt-in via `MORAINE_CACHE_DIR`
(hermetic CI, self-contained checkouts): artists legitimately manage
workspaces through cloud sync rather than git, sync providers have no
reliable folder exclusion, and cache churn — high-frequency atomic
renames of binary records — is hostile to sync clients. Unbounded
growth, the one real liability of an invisible cache, is the size
cap of §4.5.

**`external/` vs the cache** is the load-bearing distinction: cache
entries can always be deleted and re-derived from the recipe; what
`external/` holds (reference images, imported meshes) cannot. v1 is
fully procedural, so `external/` starts empty. The directory name is
reserved (above) so that a reference form into it, when one lands,
has a settled home. `build/` sits on the cache's side of the line.
It lives in the project rather than in the machine-level cache so
that a person or an agent has a stable path beside the recipe at
which to read and diff the IR.

### 4.3 Units & conventions (normative)

- Units: meters. Real-world scale always.
- Axes: right-handed, +Y up, +Z forward — an asset's front faces +Z (the
  glTF 2.0 convention, which is also Unity's forward).
- Winding: counter-clockwise front faces.
- UVs: the glTF texture-coordinate origin — u to the right, v downward,
  (0, 0) at the image's top-left texel. Projection unwraps sign their axes
  to it, so a map with an intrinsic up (weathering running down a wall)
  reads upright from outside the surface rather than inverted.
- Tangents: MikkTSpace.
- Normal maps: +Y up ("OpenGL convention") **as required by the glTF 2.0
  specification**. This is a texture-encoding convention, unrelated to
  the rendering API. A target measured to need −Y (§9.2) gets a
  G-channel flip in its export profile, never in the IR.
- Color spaces: albedo sRGB; normal/roughness/metallic/AO linear.
  Emissive is a linear factor, not a texture slot.
- PBR model: glTF 2.0 metallic-roughness.

### 4.4 Determinism

- No global RNG anywhere. Every stochastic node takes an explicit `seed`
  parameter. Per-node streams derive as `hash(seed, node_id, index)`
  (`scatter`). The index is a function of the copy and of the draw
  (a position, a yaw, a scale), never of how many copies there are,
  so a count that grows appends copies and moves none. The stream
  function is pinned (`rand.py`: sha256 of the node id, folded with
  seed and index through a splitmix64 finalizer).
  The lattice-noise operators (`noise_displace`, `noise_texture`,
  `streaks`, `brick_texture`) draw from a seed-keyed lattice without
  the node id, so identical noise fields are shareable across nodes.
  A consumer bounded to a few lattice planes reserves its own, and
  the octave consumers stride the seed, so two same-seed maps never
  track each other. The 3D field of `noise_displace` reads every
  plane, so it shares single corners with them, which is corner
  sharing, not map-level tracking. `rand.py` is a declared kernel
  (§4.5), so changing either function changes outputs and is a
  `KERNEL_VERSION` bump that rekeys every consumer.
- Identical IR must produce identical meshes (bit-tolerant: within
  float tolerance) across runs and across Windows/macOS.
- The build is a pure function of the recipe files, `params.json`,
  and the tool version. Identical inputs produce byte-identical IR,
  on every platform. Starlark guarantees the recipe side (§4.8),
  and the build owns the rest: the interpreter's floats are
  Python floats, the IR is written by the runtime's own canonical
  serializer, and the `math` builtins the recipe sees depend on no
  platform's library: the arithmetic and the functions IEEE 754
  rounds exactly are the double's own, and the transcendental ones
  are computed beyond the double and rounded once to the nearest
  double. (The host's library was rejected: its inverse trigonometric
  functions differ in the last bit between the two platforms, so a
  recipe placing by `atan2` built a different IR on each.) `moraine
  build --check` (§6) is the proof.
- The proof is non-destructive: `eval --recompute` (and `export
  --recompute`, §6) recomputes every node, checks each result hash
  against the record on disk, and never overwrites a record. Identical
  bytes need no rewrite, and differing bytes are the evidence the
  check exists to surface, reported with the first offending node and
  its op, never replaced. A key with no record is stored as an
  ordinary miss would be, so a cold cache is not a reason to withhold
  the results. A machine-wide
  `cache clear` is not the check's tool: it destroys every workspace's
  records to establish a property one comparison establishes.
- The guarantee is scoped to one Moraine version and dependency set.
  `op_impl_version` prevents stale cache reuse across Moraine upgrades;
  the declared kernel versions in the key (§4.5) do the same across
  dependency upgrades. Neither is an archive of past semantics.
  Re-evaluating an old asset on a newer runtime may legitimately
  produce different geometry. Reproducing an asset exactly means
  pinning versions (the export manifest records them, §9). Changing a
  shipped operator's semantics is a breaking change to call out, never
  a silent update.
- The build's contract is versioned the same way. `build_version`
  covers the builtins' signatures, the id derivation, the resolution
  rules, the IR's shape, and the interpreter distribution's version.
  `build` reports it, the IR carries it (§4.1), and the export
  manifest records it (§9). The stamp is a contract number, the
  interpreter's version, and a hash of the generated builtins, so a
  builtin added or renamed moves it unprompted. The tool's own
  version is not part of it, and a release that changes nothing a
  recipe means leaves every IR current. A build-layer change can
  never serve a stale record, because the build keys nothing: the IR
  it writes is the content every key hashes, and the stamp joins no
  key. What such a change can alter is which IR a recipe means, and
  the version says when.

### 4.5 Cache

- Content-addressed: `key = hash(op_name, op_impl_version, params,
  upstream_result_hashes)`. The hash is sha256 over the canonical
  JSON of that payload (sorted keys, compact separators), truncated
  to 24 hex chars. The payload also carries an evaluator-level carry
  version, bumped when the evaluator's material / face-group carry
  rules or the record layout change, so evaluator-side semantics can
  never serve stale records under unchanged operator keys.
- `op_impl_version` is a per-operator integer bumped on any change to
  output semantics, bug fixes included. This prevents stale results
  silently surviving fixes. Adding a param whose default code path is
  bit-identical to the pre-param output is *not* such a change and
  does not bump. Keys hash the authored params, so existing assets
  keep their records. The bit-identity claim is pinned by a frozen
  result-hash test (the sanctioned use of a golden hash), so a
  refactor that drifts the default bits fails loudly.
- Stochastic operators that salt their streams with the node id
  (§4.4) declare it (`USES_NODE_ID`), and the node id then joins
  their key. Without it, two nodes with identical params would share
  one key while producing different results. Pure-function operators
  never key on node id.
- An operator declares the kernels its output depends on (`KERNELS`):
  an installed distribution by name (`manifold3d`, `trimesh`) or a
  shared in-package kernel module by dotted path (`core.mating`,
  `core.predicates`, `core.rand`, `core.mesh`). The key folds each
  one's resolved version: the installed version, or the module's
  `KERNEL_VERSION`, bumped like `op_impl_version` on any
  output-semantics change of a function operators consume. So
  neither a dependency upgrade nor a shared-kernel fix can serve old
  records. The field is absent when the declaration is empty. A test
  pins declarations to imports in both directions. numpy is not
  declared: its arithmetic is bit-tolerant (§4.4), and keying on it
  would rekey every operator on every patch for no protection.
- The cache stores meshes and texture buffers uniformly. During
  parameter scrubbing only the dirty downstream subgraph re-evaluates.
- The cache is shared across projects and processes (§4.2), so writes
  are atomic: records land under a temp name and are renamed into
  place. A reader sees an older record or none, never a partial one.
  Determinism (§4.4) makes concurrent writers of one key benign.

**Retention, eviction & deletion safety.** The machine-level cache
grows without bound through iteration garbage (every edit mints new
keys and orphans old ones), so relocation cannot cap it. A size cap
can, and eviction is safe by construction: every entry re-derives
from the recipe, and aliased `asset_ref` records need no special
casing for the same reason. The contract:

- **LRU via touch-on-hit.** A cache hit `os.utime`s the record's
  `.meta.json`. Eviction deletes oldest-mtime first. There is no
  index database. The filesystem stays the only ledger (the ccache
  precedent), matching the atomic-rename store and adding no
  cross-process lock or index-vs-store drift. NTFS atime is
  unreliable and is not used.
- **Cap:** `MORAINE_CACHE_MAX_BYTES` over a built-in default of a few
  GB (`0` = unlimited and skips all checks, and a negative or
  non-integer value is refused as `bad_cache_cap`; no config file for
  this
  alone). Eviction runs down to 90% of cap (hysteresis, so
  consecutive sessions don't each trip it) and skips records touched
  within the last hour. This is the freshness guard that protects
  lazily-loaded records held by running evaluations. The window is
  symmetric around now, so a future mtime (clock skew) is never
  fresh-forever, and the guard is re-checked against the live meta
  mtime just before a group is deleted. The residual window between
  that re-check and the unlink is accepted. A hit racing the delete
  sees at worst a transient `cache_missing` the rerun self-heals, and
  closing it would take a lease/lock scheme against the no-lock
  premise.
- **Leftovers:** an in-flight file outside the freshness window is a
  crash leftover, reaped before the cap is asked, since it is garbage
  at any cache size and would count against the cap forever.
- **Unit:** the record group sharing one key prefix (npz + meta +
  derived GLB/PNG evict together), deleted meta-first. That is the
  reverse of write-visibility order, so no half-visible record can
  exist.
- **Trigger: the process that grew the cache checks after growing.**
  Short-lived CLI: one `os.scandir` check at command end, after
  output is flushed, skipped entirely when the session wrote zero
  cache bytes. Warm-cache and read-only commands therefore pay
  nothing. `serve`: the same rule debounced (after a re-eval that
  wrote, at most every 10 minutes, on a background thread so viewer
  pushes are never delayed). Returning from `serve` disarms the
  counter rather than running a shutdown-blocking scan.
  `moraine cache evict` forces the routine. A crash before the check
  leaves at most one session's overshoot, recovered by the next
  writer. Concurrent evictors and writer-vs-evictor races are benign
  (losers swallow FileNotFoundError; re-created keys carry identical
  bytes).
- **Ephemeral preview bucket.** Preview evaluations — a `serve`
  parameter overlay in effect (§7.2) — must never grow the persistent
  cache: the persistent cache indexes file history, the bucket
  indexes exploration. The bucket is one flat `preview/` subdirectory
  of the cache dir with the same filename grammar. Reads read
  through: persistent hits first, then bucket hits. Every miss a
  preview computes lands in the bucket, derived artifacts beside
  their record. There is no promotion: a later file-state evaluation
  of the same key recomputes into the persistent store. Bucket bytes
  sit outside the cap, the LRU (the eviction scan ignores
  directories), and the session write accounting, so exploration can
  never arm the eviction check. Lifetime is the serve session: the
  server empties the bucket at start (a crashed session's leftovers
  have no other reaper) and at stop, unbounded in between.
  Apply-to-evaluate controls (§7.2) bound the evaluation rate.
  Concurrent servers share the one bucket. One purging under another
  is the benign race class above. Bucket records serve over
  `/preview/` (the `/cache/` route's twin, §7.2). The bucket is never
  reached through a link: a symlink or junction under its name is
  refused on write with the fix named, scans as empty for the purge,
  and serves nothing.
- **CLI:** `moraine cache stats / clear / evict` (§6), terse JSON.
  There is no per-project clear, because attribution is undefined in
  a content-addressed shared store, and a cap makes it unnecessary.
  A `clear` under a running server strands nothing: the server
  re-materializes the records it serves on the first miss (§7.2). A
  clear is not the determinism check's tool (§4.4). `stats` reports
  the preview bucket as its own section (folding uncapped bytes into
  the persistent numbers would misstate how full the capped store
  is). `clear` empties the bucket too, attributed separately: freed
  bytes, survivors of an open-file lock, and unrecognized names
  alike. `evict` never touches it.
- **Deletion safety:** Moraine must be structurally unable to delete
  anything it did not itself write as disposable bookkeeping: its
  cache data, the session log's day files (§6), an export's own
  in-flight temp file beside the export target (§9), a build's own
  in-flight temp beside the IR it writes and the IR in its own
  project's `build/` that no root reaches (§4.8), and a params
  write's own in-flight temp beside `params.json` (§7.2), each
  behind its own narrow gate. A golden test pins the full set of
  deletion sites.
  - *Allowlist choke point.* Every unlink — corrupt-record recovery
    included — goes through one tool-wide function: direct children
    of the cache dir or of its preview bucket whose full name matches
    the writer's grammar (24-hex key + known extension, or that
    name's in-flight `.tmp<pid>-<n>` form, the earlier counter-less
    `.tmp<pid>` included). There is no directory sweep, no recursion,
    no symlink following. The bucket purge is choke-point traffic,
    not a deletion site of its own, and runs behind the same gates
    as the bulk operations below. In a cache dir that fails them the
    server's start and stop purges are skipped and logged, never
    forced. Unrecognized names are
    reported, never removed, and survive even `cache clear`.
  - *Bulk-op gates.* `cache evict` / `clear` additionally require the
    cache dir to prove itself Moraine's, by carrying Moraine's own
    `CACHEDIR.TAG` (the Cache Directory Tagging standard; tag-honoring
    backup tools skip the cache as a side benefit) or, untagged, by
    holding nothing but Moraine's own cache files. They also refuse
    outright when the cache dir is a home directory, a drive root, or
    the current working directory. Tag acceptance matches Moraine's
    own tag body (the standard signature plus its "created by
    Moraine" line), never the bare signature. Another tool's tag
    marks the dir as that tool's cache, foreign content that neither
    opens the gate nor gets overwritten. A foreign tag is reported by
    `cache stats` and named by the refusal. A torn Moraine tag is
    repaired by the next write. An unreadable tag is a transient
    no-verdict, refused for that call and never remembered as
    foreign. Content provenance counts the preview bucket as
    Moraine's own infrastructure. The bucket means the actual
    directory, never a symlink or NTFS junction named after it (every
    bucket-identity check probes for both). Anything foreign *inside*
    it stays name-gated at the choke point.
  - *Frozen on-disk formats.* The tag body prefix and the cache
    filename grammar are pinned byte-exact by test against bytes
    already on user machines, so each changes only together with a
    migration that still accepts the old form. A grammar change ships
    its outgoing pattern in the legacy-name list
    (`_LEGACY_NAME_RES`), in-flight variants included. Legacy names
    count as Moraine content for provenance, are reclaimed by eviction
    and `clear`, pass the choke point, and are never written again
    (the list starts empty; the pre-release grammar change is not
    grandfathered). Name recognition flows through one
    fullmatch-anchored classifier, so the tagging, reporting, and
    deleting layers cannot drift apart.
  - *Tag stamping requires provenance.* The tag is written at the
    moment Moraine creates the directory (the only instant "created
    and empty" is fresh evidence) or on a later write into a dir
    holding nothing but Moraine's own files. A pre-existing directory
    with foreign content never acquires it, so the do-not-back-up
    marker can never spread onto user data. Stamping uses exclusive
    create (a tag appearing between judgment and write is never
    replaced). The torn-repair overwrite keeps a residual window of
    the same accepted class as the eviction race. Accepted cost: a
    tag an external actor deletes mid-process stays absent until the
    next process.
  - *Judged-dir memo.* The write path memoizes directory verdicts in
    one per-process set holding positive observations only (a
    transient failure yields no verdict) and never revalidates.
    Complexity on the refusal side of irreversible deletion is worth
    buying, complexity for misconfig-recovery convenience is not.
    Staleness is safe because the deletion gate never consults the
    memo: it reads the directory fresh on every call, pinned by test.
    A wholesale dir deletion, meanwhile, heals via creation-time
    stamping, and a cleaned-up dir regains its tag with the next
    process.
  - *Recovery bypasses the gates, never the choke point.* The
    evaluator's corrupt-record recovery deletes through the audited
    choke point but not the directory gates. Self-healing must work
    even where the gates would refuse, and stays safe because the
    names it deletes derive from its own keys. Corruption includes
    valid JSON of the wrong shape or field types. The hit path
    validates the record-shape contract, so every failing meta routes
    into the same drop-pair-and-re-evaluate recovery instead of
    surfacing later as a raw traceback with a poisoned record left
    behind (§8.2).
- **Profiling (the `profile` key, §6):** the hit path that touches is
  the hit path that counts. It is one code path, so the LRU signal
  and the hit/miss numbers can never disagree.

### 4.6 Asset references

Reuse across assets ("place the water tower in several yards")
is an ordinary operator, not a schema change:

- **`asset_ref`** — a zero-input node whose params name a sub-asset
  file (project-relative path) and one of its named outputs. Its
  result is that output's value, Mesh or Instances alike. Instancing
  passes through untouched (§4.1). In the recipe it is
  `mo.asset_ref("subassets/<name>.star", "<output>", overrides =
  {...})` for an in-project sub-asset and `mo.asset_ref(project =
  "<sibling>", output = "<output>", ...)` for a sibling project
  (below), and the build resolves each to the IR it names. For a
  sibling the IR keeps `project` and gains `asset`, the file name of
  the sibling's root IR, so the evaluator reaches every reference
  from the IR alone. A recipe names one location, never both.
- **A function is not a sub-asset.** A recipe function expands into
  its caller's IR (§4.8), so a helper called twenty times is twenty
  inline chains in one IR. The cache collapses identical chains to
  one evaluation by content, so inline expansion costs nothing at
  evaluation. A sub-asset is a *file*, chosen for what a file
  carries: an opaque namespace, a declared parameter interface,
  asserts evaluated once per override identity, single evaluation
  and serving, and the unit of vendoring. A helper wants none of
  these, and making every call a file would turn the commonest
  compression (the same shape of chain for many distinct parts) into
  file growth. A function that wants a sub-asset's properties is
  written as one. Promoting a function to a sub-asset by marker is
  Future Work: Kit generators.
- **References are opaque.** The referenced asset evaluates as its own
  graph in its own namespace. Its node ids never enter the
  referencing graph, so id uniqueness stays per-file.
- **Asserts cross references.** A sub-asset's `asserts` are checked
  by every consuming root's `validate` (and `export`), evaluated in
  the sub-asset's own namespace and local frame, once per (asset,
  override identity). That is exactly the identity under which
  overrides change the geometry, so identical uses dedupe and each
  overridden variant is verified as landed. The built-in overlap
  guard (§5.2) runs in every such bundle over all of the instance's
  outputs, consumed or not, with the instance's own embeds and those
  reaching it through its references ceded. So a consumer's override
  that sinks two kit parts into each other is caught whether or not
  the consumer imports both. Report entries carry `asset` (and
  `overrides`) and gate `ok`: consuming an asset through `asset_ref`
  never launders its asserts away. Green entries with nothing to read
  fold into one `ref_assets` row per (asset, identity) with
  `asserts_total` / `asserts_ok`. Failing entries, and green ones
  naming an `unchecked` scope, `ceded` pairs, or a `stronger` mode,
  stay listed instead, so a kit repeated under many identities cannot
  bury the entries that carry something. Root-space propagation was
  rejected as semantically wrong: sub-asset asserts are authored in
  the local frame against local output names, while the root may
  transform the consumed output downstream, so no principled world
  placement exists. Documentation-only was rejected because `mated`
  (§5.2) is authored inside kits and would inherit the
  evaluate-nowhere hole. Outputs an assert references but no consumer
  evaluates are evaluated for the check. A workspace sweep still
  skips `subassets/` as project roots (consumers carry their checks),
  and `moraine validate subassets/kit.star` keeps working for kit
  development.
- **References never leave the workspace.** An in-project reference
  path must resolve inside the project directory. `build` and `validate`
  reject escapes. `load()` is narrower still (the rule is §4.8). It
  never reaches an asset file or another project, so a sibling's
  internals are closed at the language level and the interface below is
  the only door. Within one project git versions the root asset and its
  sub-assets together, so live-follow references are reproducible by
  construction. The pinning problem that makes cross-*repository*
  references hard (Future Work: Library & commons) does not arise. The
  workspace model (§4.2) extends exactly that rationale across sibling
  projects: **cross-project references are allowed inside a workspace,
  restricted to the interface.** Such a reference names a sibling
  project by its discovery identity (§4.2: a root-direct directory name,
  validated against the discovery set), never by relative path, and one
  named output of that project's *root asset*. Name resolution enforces
  one containment rule at every consumer of a project name (the build,
  serving, sweeps): the resolved asset file must lie inside the
  resolved workspace root. The evaluator consults no roster and holds
  the same rule over the IR it opens. A link-spelled sibling
  is legal while its target stays internal, but a link pointing outside
  is refused, because it would silently break git-sync reproducibility,
  the watch closure, and root-relative revision framing, and bypass the
  vendoring rule. `subassets/` remain project-private. The root asset's
  named outputs plus its `param` declarations are a project's public
  interface, which preserves in-project refactoring freedom, keeps the
  future library boundary an interface (project@version → output), and
  starts strict, the compatible direction. A sub-asset needed from
  outside is re-exported by its author as a root named output (one
  line). Cycle detection and `serve`'s watch closure extend across the
  workspace. Cross-*repository* reuse stays vendoring, in two forms
  chosen by what the source needs. A self-contained recipe (no `load()`
  beyond its own requirements file, no `asset_ref`) is copied into
  `subassets/` with its requirements file, under its own name or one the
  consumer chooses, and owned there. A root and a sub-asset are named by
  one rule, so the copy needs no rename. A recipe with dependencies is a
  project, and is copied as a whole directory (without `build/`) beside
  the consumer's projects, where its `lib/`, `subassets/`, `external/`,
  and `params.json` resolve unchanged and the consumer references it as
  any sibling. A vendored copy records its provenance in its own text
  (`mo.vendored(source, hash)`, the source path and content hash, a
  document-root declaration ignored by evaluation) so a later migration
  to version-pinned library references (Future Work: Library & commons)
  is mechanical, and because cache keys are content-addressed and
  path-free, that migration invalidates nothing.
- **Layout:** the root asset is the project's one top-level asset
  file (§4.2). Sub-assets live directly under `subassets/`, and
  reference paths read exactly `subassets/<name>.star` (their IR
  `subassets/<name>.mora` under `build/`). The build and the
  evaluator enforce the layout (§4.2), so "one root asset per
  project" holds at every level. A sub-asset is a complete,
  individually buildable and evaluable asset (`moraine eval
  subassets/water_tower.star` works), so the authoring loop (§8.1)
  applies unchanged while building a part.
- **Cache:** the referenced output's result hash joins the `asset_ref`
  cache key exactly as an upstream result hash (§4.5). The path and
  output name are location, not content, and stay out of the key.
  Dirty propagation across files and content-level dedup follow for
  free. A plain reference owns no record of its own (it aliases the
  referenced one). Only a `material_rename` (below) materializes a
  derived record, keyed by the referenced result hash, the mapping,
  and `asset_ref`'s op version.
- **The reference graph is a DAG.** The build detects reference cycles
  and reports the chain. A build covers the reference closure: the
  root, its sub-assets, and every sibling project it names, so the
  evaluator always finds the IR an `asset_ref` points at. Every
  asset's requirements file (§5.2) is part of that closure whether
  or not a recipe loads it. `serve` watches every reachable recipe
  file, every requirements file, every `lib/` file they load, and
  every `params.json`. A requirements file is watched before it
  exists, so writing the first one is an edit like any other.
  Editing any of them hot-reloads its dependents.

**Per-use parameter overrides** — the exposed-parameter interface
that makes a reusable asset adjustable rather than frozen ("a water
tower at three heights" needs an interface, not three copies):

- **One interface, two consumers.** The `mo.param` declarations (§4.8)
  that drive the parameter panel (§7.1) mark an asset's public knobs.
  `asset_ref` overrides consume the same declarations (the IR's root
  `params` list). Each declares an immutable `param_id` whose value the
  recipe consumes wherever it likes, and the id's surviving renames is
  what makes it addressable from a referencing asset.
- **An overridden reference is its own build.** An override is a
  value handed to the referenced asset's build, so an overridden
  reference builds an IR of its own, a **variant**, and a use with no
  overrides shares the asset's plain IR. A variant's identity is its
  asset and its overrides. Overrides are normalized to the number
  type each knob declares, so `1` and `1.0` offered to one knob are
  one variant. A variant is written beside the plain IR of its asset,
  in the referenced asset's own project, whoever offers the overrides:
  `build/subassets/<name>.<identity>.mora` for a sub-asset and
  `build/<name>.<identity>.mora` for a root, `<identity>` being the
  24-hex hash of the normalized overrides. Two projects that offer a
  sibling the same overrides read one file, and every IR's references
  resolve in the project whose `build/` holds it. An asset file's name
  never ends in a dot and 24 hex digits, the form that marks a
  variant. (Writing a sibling root's variant in the consuming
  project's `build/` was rejected. A project's `build/` takes the
  variants of its sub-assets from its consumers' builds either way,
  and the second place gave each consumer a copy that an evaluation
  had to recognize as one instance.)
- **The evaluator never applies overrides.** The `asset_ref` IR node
  keeps `asset` or `project`, `output`, and the normalized
  `overrides`, so reports, dedup, and record aliasing keep their
  identity key. The evaluator uses `overrides` only to derive the
  identity and open the variant. A variant's declared params already
  hold the overridden values, and applying them again would be a
  second, recipe-blind resolution of the same knob.
- **Evaluation identity, not cache keys.** Overrides join the
  reference's params and land on the referenced asset's declared
  params, so cache keys follow content as always. What changes is
  *evaluation identity*, which gains a normalized-override dimension
  (the record memo and the asset registry key on it; without it two
  differently-overridden uses of one asset silently collide). An
  un-overridden reference keeps the record aliasing unchanged.
- **Errors reuse the declaration contract** (`param_unknown` /
  `param_type_mismatch`), and the declared min/max is part of it: an
  override outside the range fails with `param_out_of_range`,
  enforced in the build's one param resolution, which both consumers
  (the viewer's overlay and a reference's overrides) feed, so the rule
  is shared and the build is where it fails. The range is the only
  validity knowledge an opaque reference exposes, and starting strict is
  the compatible direction (a waiver form can come later). The refusal
  covers a value a consumer offers. An asset's own value outside its
  range is a failing entry of the report (§6).
- **Two scope rules keep references opaque.** An override reaches
  only the *directly* referenced asset's declared params. An asset in
  between forwards a deeper knob by declaring a knob of its own and
  passing its value on in the `overrides` of its own reference, so a
  chain stays a sequence of declared interfaces. And the viewer's
  parameter panel shows the root asset's declarations only (overrides
  are values in the file, orthogonal to the panel's live overlay).
- **Accepted values are the root's.** A project's `params.json`
  (§7.2) holds the values of the knobs the panel shows, its root
  asset's, so a sub-asset's knob is set by the asset that references
  it. The root is built under them whoever builds it, so a project
  reads the same alone and inside a consumer. A value accepted in a
  sibling therefore moves every consumer that does not override the
  knob. The sibling's report names them (`consumers`, §6).
- **Per-use material identity — `material_rename`.** Overrides that
  reach a material change its content under the kit's name, so two
  uses of one kit bind one name to two contents and the consumer
  cannot repair it in the referenced file. `asset_ref` takes
  `"material_rename": {"<name>": "<per-use name>"}` — the explicit
  consumer-side identity, the material-instance / material-variant
  shape of the engines. Auto-derived names were rejected because
  per-instance namespacing of a kit's materials yields one identity
  per instance with duplicated content, the outcome the explicit form
  avoids. The rename is applied to the *received value* at the
  reference boundary, never inside the referenced asset and never as
  part of its override identity. A renamed ref is a one-input derived
  record over the referenced output's record, while the sub-asset
  instance, its asserts, and every un-renamed use stay shared. It is
  depth-agnostic (a material name is the output's public export
  identity wherever it was authored, so names arriving through nested
  references rename too) and loud: every key must name a material
  present on the received output (`material_rename_unknown` lists
  what is there). A silent no-op is exactly the
  overrides-not-reaching-output failure class. A renamed identity is
  authored by the consuming asset (the material-origin walk, §9.1),
  and the `validate` uniqueness gate's fix routes a reference-borne
  collision to the ref node and this key.
- **Precedents.** Houdini HDAs solve pinning with namespaced versioned
  definitions plus embed-and-unlock vendoring, the model for the
  future library layer. USD composes by path references, the
  template for the in-project rule. Blender's linked libraries show
  how machine-specific paths break when a workspace moves, which is
  why references may never escape the workspace.

### 4.7 Project journal (`journal.md`)

Four kinds of project knowledge are worth accumulating: **aesthetic
preference** (chosen vs. rejected seeds and variants, what the user's
adjectives turned out to mean), **conventions** (naming, default
dimensions, kit style defaults), **correction patterns** (mistakes
the agent made repeatedly and was corrected on), and **rationale** (what
was tried and failed, what a human's words turned out to require). The
guard against scope creep: never record what the workspace already
derives (counts, bounds, structure), and never record what the recipe
carries.

**Comment or journal.** The recipe has comments (§4.8), so the two
carriers split by one test: whether the knowledge has a line to sit
beside. A recipe comment belongs to a line and says what that line's
expression cannot: the constraint behind a value, the reason a
construction was chosen. It never states a measured number, because a
number the tool can measure belongs in an assert, where it is checked,
or nowhere. The journal keeps what has no line: the alternative that
was rejected (its lines are gone), the correction the agent kept
needing, the preference behind a pick. A derivation itself is neither.
It is a named expression in the recipe. The two never point at each
other, and a version stamp (below) counts only in the journal, so the
retirement pass has one carrier to scan.

The shipped guide's "Project journal" section carries the agent-side
rules (read rule, entry format, retirement, declarative observations,
placement, language). The file format the tool reads is the `moraine
docs journal` reference. Consolidation and topic-file mechanics stay
in Future Work: Journal mechanics.

- **Name and placement.** The file is `journal.md`: "memory" collides
  with every harness's own memory features, and "journal" says what
  the file is, observations the agent judged worth recording, not an
  unjudged log. One journal lives in each project directory, beside
  the asset whose *why* it records, because the project directory is
  the portable unit (§4.6 vendoring). Workspace-spanning observations
  go to an optional workspace-level `journal.md` (created on first
  entry). The placement rule is "the narrowest level that covers the
  entry", with a "do not record this session" flag respected at every
  level. Sharing *across* workspaces, if ever, rides the library
  layer (Future Work: Library & commons).
- **Layout — one file, an index and on-demand bodies.** `## Entries`
  is the index every session reads: one entry per column-0 bullet,
  the line being the whole entry for most observations. `## Details`
  holds the bodies that need more than a line, one `### <slug>`
  section each, read only when an entry points at it (`-> #slug`;
  slugs unique per file). A pointer may also name a sibling file
  (`-> <relative>.md#slug`), so a journal that outgrows one file can
  be split by a deterministic move without changing an entry. Topic
  sub-headings inside Entries are free. The two anchors are what the tool
  recognizes. A journal without them still scans (every stamped bullet
  outside Details is an entry). One file rather than an index
  plus entry files, because the primary reader is the human who
  prunes it: one page, one diff, one history.
- **Entry format — the stamp marks what a version can falsify.** An
  entry about the tool opens with the version it was written or last
  verified against, the value of `moraine --version` (§6).
  `[moraine <version>]` marks an observation of this version's
  behavior, whose consumer is the next upgrade. The stamp is matched
  case-insensitively, and a pointer's arrow may be written `->` or
  `→`.
  `[moraine <version>, workaround]` marks a version-bound limit,
  written with its symptom so it can be re-verified. Whatever a
  product fix would make false carries a stamp; a subject-side
  observation (about the asset, the human, the design) carries none.
  A workaround stamp may carry a **claim key**,
  `[moraine <version>, workaround: <key>]`, one short name per tool
  limit shared by every entry that limit forced, so the retirement
  pass asks "is this limit fixed" once instead of "is this entry
  stale" per entry. A stamp counts only at the head of an entry
  bullet. A file-level default stamp is not offered, since it cannot
  tell legacy bullets from the unstamped subject-side observations
  appended after it.
- **Retirement.** A workaround entry is an observation against the
  version it names, not a fact. When its stamp is not the running
  version the agent reproduces the symptom before acting on it, in
  the asset itself: it restores what the workaround replaced, a value
  or a construction, runs `validate` once, and reverts (a limit on a
  command is re-tried in place), then deletes the entry (limit gone)
  or re-stamps it (limit holds). An entry whose site the asset
  no longer holds is deleted unverified: the observation no longer
  shapes the asset, and the limit, if it holds, is the tool's
  feedback, not the journal's. The tool's part is the **retirement
  pass in `moraine init`**, the command run after every upgrade: it
  reads every journal in the workspace, never writes one, and
  reports each stale-stamped workaround entry under `journal_review`
  and the journal counts under `journals` (shape: §6), claim-keyed
  entries collapsed to one row per limit.
  Stateless by design: no version file, the comparison is stamp
  against installed, so the list repeats until the entries are
  retired and disappears on its own. Three further criteria are the
  agent's, applied as it appends: an observation the asset can now
  carry as an assert or check, an entry a later entry contradicts,
  and an observation promoted into the human-owned file all leave
  the journal. Unstamped entries are counted, never listed. A legacy
  entry is stamped only if the running version can falsify it, as the
  agent next touches it.
- **Workspace completeness — the journal is the repository's
  record.** Any agent, from any vendor, on a fresh machine must reach
  full project context from the repository alone. Whatever
  session-persistence mechanism the host harness offers is a
  disposable cache, never the authoritative copy, and the harness's
  own dedup rule (don't store what the repository already records)
  enforces the boundary without fighting it (P0; model- and
  client-independent, §8). Harness-side
  memory remains the right layer for user-universal traits that
  transcend Moraine.
- **Read rule.** The guide instructs, in the imperative, that before
  working in a project directory the agent reads that project's
  `journal.md` and, if present, the workspace `journal.md`: the
  Entries index, and a Details body only when a relevant entry points
  at it. Prose, not harness wiring: the rule is the same on every
  harness, and an import would be one harness's.
- **Recording.** One pick is recorded as one pick, never silently
  promoted to a standing preference. Entries are **declarative
  observations, never imperatives**: the grammatical form carries the
  epistemic status and keeps an always-read journal distinguishable
  from the human-owned rules file. An entry that wants to be phrased
  "always do X" is a promotion candidate. Record the observation,
  flag it, never write the rule.
- **Promotion — two distinct moves, both decided by the human.**
  *Horizontal* (project journal → workspace journal) widens scope
  while staying observation-level. It requires evidence from more
  than one project, and the workspace entry cites its instances.
  Under the read rule the agent never routinely sees sibling
  projects' journals, so recurrence is spotted by the human while
  pruning. The agent's only write-time judgment is initial placement.
  *Vertical* (journal → the human-owned file) changes normative
  force, bias becoming rule. Its evidence sits in the journals the
  agent reads every session, so the agent marks the entry as a
  candidate and the human decides.
- **Application.** Preferences bias proposals; they never constrain.
  The agent may propose against a recorded preference where the design
  warrants it, and staleness is expected, because the journal is
  editable history, not ground truth.
- **Rollout — the permanent shape.** The shipped guide instructs the
  modeling agent to append candidates, and the human prunes. No
  surface gets its own auto-capture channel, variation-grid picks
  included, since a standing capture channel is a standing cost
  without standing evidence (P2).

### 4.8 Recipes

A recipe is a **Starlark** file, the dialect of Python built for
describing build graphs. It was chosen because the discipline
Moraine needs of a recipe is the language's own specification, not
a rule laid over a general language:

- *Deterministic.* The same files build the same IR (§4.4). The
  language has no clock, no randomness, and no environment to read,
  and a dictionary iterates in insertion order.
- *Hermetic.* A build reads only the files its recipe names. The
  language has no I/O and no import. `load()` is the one way to
  another file, and the build decides what it reaches.
- *Terminating.* A build finishes. There is no `while`, and a `for`
  iterates a finite sequence. The build adds the one bound the
  language leaves to it, a cap on call depth.

In Python each of these is a rule the agent is asked to keep and a
lint and a sandbox are asked to catch. Here a recipe that breaks one
cannot be built. So `serve` evaluates the agent's text on every
save, and building a recipe is never a code-execution decision,
whoever wrote it.

What the language keeps of Python is its syntax, its functions, and
its loops. An agent's prior transfers, so there is no grammar to
teach, and a value is stated once where a data format would write
out every copy (§4).

**What the language lacks.** What the agent's prior must unlearn is
small and loud: a construct Starlark lacks fails the build with the
spelling to use instead, the actionable-error rule (§8.2) applied to
a language. Recursion is bounded, not refused. A call chain that
reaches the cap fails the build with the chain and the loop over
`range()` to write instead.

**The boundary: recipes compute numbers, operators compute
geometry.** A recipe computes only from declared values: literals,
its own arithmetic, declared params. It never observes an evaluated
value (a bound, a count, a face frame), and no builtin returns one.
Placement that depends on evaluated shape is an operator's (`place`,
`mate`, §5.1), face selection is a predicate's (`tag`), stochastic
sampling is an operator's with a `seed`, and verification is an
assert's, because only an assert has an owner and a report row.
Starlark's `fail()` is refused at the top level for the same reason a
check script was (§5.2). It would stop the build with one message
where a report carries every finding. Inside a function it is the
function's own refusal of what it was handed, and fails the build
with its message. The consequence of the boundary is Invariant 10
(§11): the build needs no evaluator and runs before any evaluation.
Load-time expansion feeding a static graph is the architecture the
language was designed for.

**Builtins.** The functions that build the IR live under one host
namespace, `mo`, generated from the registries (§3) so the recipe's
vocabulary and the IR's are one:

- *Operators:* `mo.<op>(inputs..., id = ..., <param> = ...)` for
  every registered op, bare names identical to the IR's `"op"`
  strings, inputs as node handles, params by keyword with the IR's
  names and validation. An input is passed by position. Where an
  operator's inputs play different roles (a base and a part, a target
  and a prototype) the operator names them, and a recipe may pass
  such an input by its name: the call then says which is which, and
  holds when an input is added or the order moves. The IR keeps the
  positions. An input an operator may take or leave (a `material`'s
  textures) is passed by keyword alone, the keyword naming the slot
  it fills, and the build writes the filled slots beside the inputs.
  (Textures by position beside a list of their slots, the IR's own
  shape, was rejected for the recipe: two one-channel textures
  swapped are a valid asset, and no check sees it.) Inputs held in a
  list are expanded after the keywords, the language's own order, as
  in
  `mo.boolean(id = "stair", mode = "union", *treads)`. No builtin
  takes a list in place of its inputs. The node's id is the keyword
  `id`, the IR's own key, and never `name`, because `material` and
  `tag` take a `name` param of their own. An input's name is a
  keyword of the builtin too, so the registry test that keeps the
  builtins equal to the IR refuses an operator declaring a param
  called `id` or named like one of its inputs. The call returns an
  immutable node
  handle. There is one spelling: no method chaining, no `op_` prefix,
  no sub-namespace, so a call site reads as its IR line. A param
  holds what the IR holds, so a value that is not a finite number is
  refused.
- *Declarations, one per IR root key:* `mo.output(name, node)`,
  `mo.group(name, [outputs])`, `mo.param(id, default)` with `min`,
  `max`, `label`, and `step` as keywords, `mo.budget(...)` (in a
  requirements file only,
  §5.2), `mo.waive(output, check)` and `mo.waive_asset(check)` for
  the waivers of §4.1, and `mo.vendored(source, hash)` (§4.6).
  `mo.param` returns the knob's resolved value. Values resolve in
  this order: the recipe's default, the root's accepted value in
  `params.json` (§4.6, §7.2), then a consumer's override (§4.6). Later
  values take precedence.
- *Asserts:* `mo.assert_<kind>(...)`, one function per registered
  kind, never a generic call taking the kind as data. Every argument
  is a keyword, outputs and groups named by string, with the IR's
  argument names (`tol`, `min`, `max`, `every`, `self`). There is no
  `owner` argument: the
  file decides (§5.2). An unknown name fails the build, which is
  what makes assert-first a build state (§8.1).
- *In a requirements file:* the assert functions, `mo.budget`,
  `mo.group` for a group its asserts name (§5.2), and
  `mo.param_value(id)`, an opaque handle to one of that asset's
  declared knobs, usable whole as a threshold (`value`, `tol`,
  `min`, `max`, or `min_area`) and nowhere else. Arithmetic on a
  handle, or a
  comparison by size, is refused (`param_value_arithmetic`) with the
  fix of writing the derived quantity as a constant, since a handle
  resolves only after the asset has declared its knobs. Equality and
  truth the language gives every value (a handle equals no number
  and is true), so a condition on a handle cannot be refused where
  it stands. The build refuses the handle instead, once the file has
  run: one that became no assert's threshold (`param_value_unused`)
  was read some other way, and a requirements file cannot branch on
  a knob's value. The
  knob is a number, and the entry names it (`from_param`, §4.1), so
  a requirement an accepted value moved says which knob moved it. No
  operator, no declaration that creates a node, and no `load()` (the
  reason is §5.2).
- *Language-level names stay bare:* `range`, `struct`, `load`, and
  `math` (the recipe's own, §4.4). `mo.` marks exactly what
  reaches the IR, so a call without it is recipe arithmetic. The bare
  names a recipe resolves are frozen by a test, as the call-stack
  text below is: a name the interpreter gains or drops with its pin
  would change the language recipes are written in, and it fails
  Moraine's own suite, never a user's build.
- *Every name under `mo` is the tool's.* A workspace defines none,
  and an operator's name matches `[a-z][a-z0-9_]*`. So a builtin
  added later collides with nothing a workspace wrote, and whatever
  a workspace comes to define is spelled outside both.

**Numbers.** A recipe's numbers are the language's: `/` gives a
float and `//` between integers an integer. Every operator param and
every assert key declares the number type it reads, a float, an
integer, or neither. A float param given an integer holds it as a
float, so `90` and `360 / 4` build one IR and share one cache key.
The conversion is the one the operator makes when it reads the param,
made by the build, so it changes no result. An integer param given a
float refuses it whatever its value and names `int(x)` as the
spelling to write, since `//` on a float stays a float. Numbers
inside a list or a dict take their param's type. `moraine docs <op>`
lists each number param's type from the declaration the build reads.
A knob keeps the type of its declared value (§4.6). (Deciding the
type from the value, a whole float written as an integer, was
rejected: a param would hold an integer at one knob value and a float
at the next, and an integer param would take a quotient that stops
being whole when its knob moves.)

**Names and ids.** A node that anything can point at carries an `id`:
every output's node, every op that salts its streams with its id
(`USES_NODE_ID`, §4.5), and any node an assert, a pin, or a journal
entry may name. Ids are unique within an asset, and a duplicate fails
the build. A function that creates nodes takes `id` as its first
argument and creates them under `<id>/`, so its ids are deterministic,
collision-free, and read as their construction (`pipe_left_3/barrel`).
The separator is `/`, admitted in node and assert ids and never in
output, group, or param names (which become export identities,
§4.1). Each segment is a portable name (`[A-Za-z][A-Za-z0-9_-]*`),
the grammar of those names, so an id reads the same in a report, a
pin, and a journal entry. Ids are case-sensitive.

A node may be left unnamed only at an asset's top level. It gets
`<op><n>`, `n` counting that op's unnamed nodes in program order,
which is legible enough beside its `site` and moves only when an
unnamed node of the same op is added or removed above it. Assert ids
derive the same way. Inside a function every node and assert is
named, the function writing the whole id (`id = id + "/barrel"`),
and a call that leaves one unnamed fails the build with that
spelling. The interpreter tells the build which function a call sits
in and never the arguments that function received, so the build
cannot know the id a function was given. An id written in full is
also what lets a recipe keep the id a node already had, and with it
every stream that id salts. A requirements file names every assert
it declares, since an ordinal would rename a requirement each time
one is added above it.

**Sites.** A builtin learns its call site from the interpreter's
call stack. `mo` reaches the host through a Starlark prelude, one
thin function per builtin, which reads the stack and hands it to the
host function with the call's arguments. The build takes the
innermost recipe frame as the `site` (§4.1), so a node made inside a
helper carries the helper's line. The outer frames are its `via`, the
site of each call on the way, outermost first, written beside the
`site` on every node, assert, and knob made inside a function. A line
shared by every caller of a helper then still leads to the call that
asked for this entry. A chain longer than six keeps its two ends and
counts the middle.

The stack is read through the interpreter's debugging extension,
`call_stack()`, which returns the traceback as text. The interpreter
documents that text as written for a human reader, neither stable
nor meant to be parsed. Moraine parses it, under four rules that
contain the dependency. The interpreter distribution is pinned to
one exact version, which is part of `build_version` (§4.4). A test
freezes the shape of the text, so a format change fails Moraine's
own suite when the pin moves and never a user's build. A node or an
assert whose site cannot be read fails the build, and none is
written without one. The extension is loaded for the prelude alone,
so a recipe cannot observe where it was called from and the boundary
above holds.

**Functions and files.** A function that creates nodes is a macro,
expanded into its caller's IR under the id the call passes. Why it
is not a sub-asset is §4.6. Helpers shared by the root and its
sub-assets live in `lib/*.star`, definition-only files. A `lib/`
file that creates a node or declares an output, param, or assert at
its top level fails the build. `load()` has three rules:

1. An asset loads `lib/` files of its own project and its own
   requirements file, whose constants it may build against (§5.2),
   and nothing else.
2. A `lib/` file loads other `lib/` files and no requirements file.
   A helper that needs a requirement constant takes it as an
   argument.
3. An asset file (the root or `subassets/<name>.star`) is never
   loaded, only referenced.

The three share one reason: a loaded file runs, and a requirements
file or an asset file run under another asset would register its
declarations there. A recipe file reached through a link is refused,
since the rules judge the path a recipe spells. A requirements file
is never an asset wherever it sits (§4.2), so it is never an
`asset_ref` target and never a command's target (§6). Recipe
comments follow §4.7.

**Repetition.** A loop whose body emits an unchanged prototype under
a per-copy transform writes by hand what `array` and `scatter` emit
as instances, and the export pays for it in meshes (Invariant 6).
`build` detects the pattern (three or more sibling nodes whose
content, all but a transform, hashes equal, placed by one step) and
warns with the `array` to write (`copies_not_instanced`), in its own
report and among `validate`'s warnings. Copies placed by no step draw
no warning, since no operator emits them as instances and a warning
without a fix trains its reader to pass warnings by. A loop is for
copies that differ by rule (a stair whose treads shorten, a rank of
pipes whose lengths follow an arch, a per-copy assert) and for
placing sub-asset references under different overrides.

**Build.** `moraine build` (§6) resolves the target's reference
closure, runs each recipe with the params of §7.2, and writes the IR
in canonical form under each project's `build/`. An asset is built in
two phases. *Declaration:* its recipe runs, loading what it loads.
Its requirements file (§5.2) runs exactly once per build whether a
`load()` reached it first or the build runs it after the recipe, and
every assert, budget, and `mo.param_value` call it makes is
recorded, never resolved. *Resolution:* with every declaration in,
the build resolves output and group names, knob ids, and handles,
and an unknown name fails here with the site of the call that named
it. Order inside a file therefore matters to values (a constant is
bound before an expression uses it, as in any program) but not to
names. Nothing but the build writes under `build/`, and nothing
reads a recipe but the build.

**What `build/` holds.** A build leaves what earlier builds wrote: the
variant of an override that has since moved, the IR of a sub-asset
once built on its own. A reader that opens `build/` by path would take
those for current, so the written build of a project's root removes
them from that project's `build/`. A file is removed when it is the
build's and no longer needed. Whether it is the build's is asked of
the file alone: a direct child of `build/` or `build/subassets/`,
neither reached through a link, under the writer's extension,
beginning as an IR begins. Anything else there is reported as
`unrecognized` and left, except a writer's own in-flight
temporary, which is that writer's to rename or remove. Whether it
is still needed is asked of the workspace, because a `build/` has
more than one writer: a project
that references a sibling under overrides writes the variant beside
the sibling, and with it the variants of the sibling's own
references. Every project's root IR is a root,
named by the project's root asset, a reference leads to the IR it
names, and what no root reaches is removed. A variant its consumer
has stopped reading therefore stays beside the sibling, opened by no
reference, until the sibling's root is built. A root that cannot be
read stops the removal, since the walk then knows less than the disk
holds, and the report names it as `blocked`. So does a project that
holds more than its one root asset, which has no one root to be read
from. A file written within the last minute is left for a later
build: a build writes its closure file by file, so a variant can land
beside this project before the root that names it lands beside its
own. What the removal keeps true is that no IR a root reaches names
an IR that is gone. It does not keep a consumer's IR current with a
sibling's recipe, which only the consumer's build does, and
`build --check` answers. (Removing what the build in hand did not
write was rejected: the owner's build would remove the variants a
consumer wrote beside it. A ledger of written files was rejected
while every file under `build/` names itself by its first bytes,
since the walk needs no state that can part from the disk.)

---

## 5. Operators & Assertions

The modeling vocabulary in three parts: operators build (the table
below), predicates address (§5.1), assertions verify (§5.2).

The operators (the table, plus `asset_ref`, §4.6) wrap mature
kernels. Signatures are the stable contract. Implementations may be
swapped (numba/Rust) without touching assets. A new operator is
admitted under the rubric below. The table says what each operator
does and what it
runs on. Canonical per-op contracts — parameters, failure modes,
attribute policy — live in the shipped reference (`moraine docs`).

| Operator | Kernel | What it does |
|---|---|---|
| `box / cylinder / sphere / grid` | numpy | primitives |
| `transform` | numpy | TRS, instancing-aware |
| `extrude` | numpy/trimesh | faces along their normal or a vector (a vector opposing every normal sweeps the flipped sheet; tangent or mixed-orientation vectors, and flat-shaded closed inputs, are actionable errors) |
| `boolean` | manifold3d | union / difference / intersection; output is flat-shaded |
| `revolve` | numpy | solid of revolution of an inline `[radius, y]` profile around +Y, always watertight (ends auto-close); inward-facing and degenerate profiles are actionable errors; joint shading crease-thresholded like `smooth_normals`. The inline profile retrofits to the curve value type when it lands (Future Work: Operator extensions) |
| `array` / `scatter` | numpy | emit prototype + transforms (instanced) |
| `noise_displace` | numpy | fBm displacement along normals; flat-shaded input is refused with the `smooth_normals` fix named, never welded (welding would merge the attributes `preserved` promises to keep) |
| `bevel` | own | single-segment chamfer; `segments > 1` is refused until a multi-segment kernel lands. No matching edge is a `no_effect` error, and manifold/degeneracy/volume post-conditions gate the result with `broken_result` — never a silent pass-through |
| `unwrap` | own | planar/box projection UVs; a packed atlas is a further `mode` to come |
| `smooth_normals` | numpy/trimesh | angle-threshold auto-smooth; restores curved shading after boolean/bevel |
| `bake_instances` | numpy | explicit Instances → Mesh merge — the one deliberate opt-in (§4.1); tiles face groups |
| `material` / texture nodes | numpy | `noise_texture` (fbm/ridged/billow), `brick_texture`, `checker_texture`, `colorize`, `ramp`, `streaks` (fixed vertical; orientation comes from the part's UVs), `normal_map` emit linear-float `Image` values cached like meshes; `material` binds a glTF metallic-roughness material to a mesh — name = export identity (§9.1), content-dedup'd, carried by `MATERIAL_POLICY` (§4.1) |
| `tag` / `place` | numpy | `tag` fixes a face-predicate selection into the graph as a named group; `place` anchors a part `on_top` of a tagged group — §5.1 |
| `mate` | numpy | face-to-face placement through the shared mating frame — §5.1; paired with the `mated` assertion (§5.2) |

**Admission rubric (the P6 rubric).** A candidate operator is
judged against P6. The question a candidate must answer is *does it
reduce the agent's error rate or the token cost per accepted asset*
(evidence: the authoring benchmark and field builds), not whether
other DCCs have it:

1. **Predicate addressing** — element references are queries (angle,
   orientation, provenance, tags), never index lists that break under
   upstream edits.
2. **Absorbs spatial arithmetic** — accepts semantic anchors /
   relative frames rather than making the agent derive coordinates.
3. **Token-compact signature** — few orthogonal parameters, names
   consistent across operators so the agent's priors transfer.
4. **Numerically verifiable contract** — documented effect on stats
   (counts / bounds / volume), so results are assertable without
   pixels.
5. **Total and deterministic** — never silently produces garbage.
   Failures are structured and actionable (§8.2).

Two exclusions follow. Operators that presuppose interactive
selection or brush/gizmo-style micro-manipulation are out. They
optimize for the hands Moraine's authors don't have. And the rubric
cuts against mega-operators: a 50-parameter building generator is a
recipe function or a sub-asset and eventually a library entry
(Future Work: Library & commons), not an operator, because its
semantics contract would be unverifiable. The same guard applies to
recipe functions with the same force. Hermeticity bounds what a
function *can* do, not whether it reads as a declaration, so
functions stay one-intent-sized too. Operators stay
one-intent-sized (P3). Candidates are collected in
Future Work: Operator extensions.

**What an operator ships.** Each operator documents its semantics
(input → output meaning) in one short paragraph, independent of
implementation. Golden-mesh tests (§10) pin them. Every operator
also ships a minimal example asset, read per authoring pattern, not
per op. Ops admitted as one pattern may share one
composed example, each reference section citing it. The examples
live in the package (`docs/examples/`, one project directory each, so
every install ships them). The section that cites one names it
(`moraine docs <op>`), and any prints by name (`moraine docs
examples/<name>`), so a workspace agent reads the contract first and
an asset that uses it when it needs one. The repository's root
`examples/` is an ignored scratch checkout of them. The package tree
stays a delivery, never a workspace (§4.2).

Tessellation ceilings are absurdity guards, not quality bars. Asset
quality is owned by the asset's declared budgets (§4.1). Each
tessellation parameter caps at 4096, and the product of two such
dimensions is bounded by a shared cell budget (4096 × 4096). An op
whose tessellation can multiply past it (`revolve`, whose hard joints
duplicate rings) refuses before allocating.

Operators do not delegate to headless Blender: it ships no PyPI wheel
(§3), it would make results depend on the host environment against
the determinism and cache contract (§11), and it would blur the
positioning of §1. A kernel that extends an operator (a multi-segment
bevel, a packed atlas) lands behind the same signature, accepting a
value it refused or adding a `mode`. Asset files never change, and
neither does what an asset already builds: a kernel that returns
another result for the same params is a new value of a param, not a
new implementation of an old one.

### 5.1 Predicate addressing: tags and anchored placement (P6)

The `tag`/`place`/`mate` rows above share one face-predicate
vocabulary — designed together, not piecemeal (P6) — that is also
consumed by asset asserts (§5.2) and `moraine query` (§6). That is
one semantics readable at query time, fixable into the graph, and
assertable at validate time.

- **Predicates:** `normal_*`, `center_*`, `area`; comparators
  `gt/ge/lt/le/eq+tol`; AND-composition. Cardinality is explicit:
  `exactly_one` (empty and ambiguous selections error),
  `one_or_more` (empty errors), `any` (an empty selection is valid; a
  consumer for which it is meaningless refuses it itself). A
  selection outside its declared cardinality is a structured error,
  never a silent pick. Exact thresholds and ascending face-index
  ordering keep results deterministic near boundaries.
- **`tag`** fixes a selection into the graph as a named face group.
  On an instanced input it tags the prototype in its local frame,
  copies untouched. A world-space predicate over a chosen copy would
  make one tag mean different faces per copy, so a consumer tags a
  received instanced output at the point of use. Groups ride
  result_hash/cache/stats, propagate per `ATTRIBUTE_POLICY` (dropped
  with a `groups_dropped` warning on topology rebuilds), are tiled by
  `bake_instances`, and are never exported.
- **`place`** positions a part by semantic relation instead of
  agent-derived coordinates. This is the same error class as the
  §4.1 ground checks, attacked from the generation side. Mode
  `on_top`: the anchor group's area-weighted centroid + max-y resting
  plane, translation-only, with an upward gate and a single-surface
  gate (an anchor whose faces are not coplanar is refused,
  `anchor_not_coplanar`, rather than resting on an average). An
  Instances anchor
  takes a required 0-based `instance` index selecting the copy whose
  world-space group frame anchors the part (out-of-range errors
  instead of going stale). The part input stays mesh-only.
- **`mate`** positions a part by bringing two tagged faces into
  coincidence. It is the any-orientation generalization of `place`,
  absorbing the hand-computed translation and rotation every assembly
  seam otherwise needs. Inputs `[anchor_base, part]`, or
  `[anchor_base, part, leader]` when the part is a rider. Sibling
  outputs of one assembly carry no mating face of their own and
  receive the rigid motion that mates the leader's face — one solved
  frame, every relation an input edge. The **mating frame** is one
  definition shared by the op and the `mated` assertion (§5.2),
  derived from a tagged face set's world-space triangles. The set
  must be one planar surface: face normals within 1° of the mean
  (slivers below 1e-6 of the set's area carry no direction and are
  not tested), vertices within 1e-4 m of the plane. Mixed,
  opposite-facing, or edge-disconnected sets are structured errors,
  never a guessed frame. *Origin* = the centroid of the set's single
  opening when it has one, else the area-weighted centroid. A holed
  wall face registers on its hole, a tube end on its bore, because
  the opening is the feature a joint registers on (two or more
  openings: `ambiguous_opening`). *Normal* = area-weighted face
  normal. *Secondary axis* = world +Y projected into the plane, +Z
  (asset front, §4.3) for a face within 1° of level. This is the
  upright rule, so an assembly keeps its up and a horizontal mate
  degenerates to the pure translation `place` performs. The solve
  maps part normal → −anchor normal and secondary → secondary,
  applies the explicit `roll_deg` about the anchor normal (a square
  mouth mates identically at 0 and 90 degrees — roll is declared,
  never inferred), then translates origin onto origin. The centroid
  fallback is silent at execution: a bore rim that lost triangles to
  its predicate has no closed opening, and the paired `mated` shares
  the frame and passes. So the op reports its provenance: `validate`
  lists every root `mate` node under `mates` with the origin word per
  role (`anchor` / `face`: `opening` or `centroid`), and the `mated`
  entry echoes the words. A word, not a warning: a blank-face mate
  legitimately reads `centroid`. The op solves coincidence only. A
  contained joint (a part somewhere on a larger blank face) has no
  unique solution, so its position stays authored and only the
  assertion's `contained` mode (§5.2) checks it. A `Frame` value
  type, curved and coaxial mating, and roll inference are not in the
  slice. Instanced anchor bases and face carriers take the required
  `instance` / `face_instance` copy index. An instanced part moves as
  a set. The refusals (unknown/empty tag, non-planar, degenerate,
  disconnected, ambiguous opening, missing copy index) each name a
  terminating remedy, and they are the only guard for a joint
  consumed by a `boolean` union before any output, which cannot carry
  `mated`. The pair is descriptive, never prescriptive, about output
  granularity. The guide does not steer authors toward splitting
  outputs so that joints become assertable, because the assert's
  benefit is authoring-time while the draw-call cost ships in the
  asset, a tradeoff the agent cannot weigh. A validate-only output
  is the mechanism to revisit if union-interior joints break silently
  in the field.

Further anchor modes (*flush with*, *centered on face*) and further
assert kinds remain candidates in Future Work: Operator extensions.
Anchor rotation and the secondary axis are settled above by `mate`'s
frame.

### 5.2 Asset assertions

The checking half of the vocabulary: operators *solve* (`place`
derives a frame, `boolean` an intersection), assertions *verify* the
evaluated result. This is the same predicate/named-output semantics
§5.1's write side fixes into the graph, applied from the read side
at validate time. An op that solves is paired with an independent
check (`place` with `contact`; `mate` with `mated`; `array` and
`scatter` with `count`), so no solver certifies its own solution.

Declarations, not code: a check written as a script the author runs
against the result was rejected as the carrier for any kind. The
prover would be written by the author of the geometry it proves,
live outside the asset's owner model, and never enter the report's
roster. What a green entry proves is that the declared condition
holds of the faces and outputs the declaration names, measured
exactly. Whether the declaration expresses the requirement — that a
predicate written for the top shelf does not select the post tops —
is the reviewer's question, never answered by inferring intent from
names. A selection the wrong feature can satisfy is a recipe defect,
and rewriting it so only the intended feature satisfies it is the
recipe fix ownership allows. It is distinct from the loosening the
owner model forbids.

**Common contract** — every kind obeys these; a kind's paragraph
below adds only its measurement and judgment:

- Root `"asserts"` (§4.1) is checked deterministically at eval time
  and reported through `validate` — gating, never waivable. An
  unevaluatable entry (an image, empty or non-finite geometry, a
  face without a frame) fails with the reason. An unknown output name
  is refused at load. Sub-asset
  asserts gate every consuming root's validate, per override
  identity (§4.6).
- **Every entry carries its measurement, pass or fail** — the
  distance, angle and offset, depth, or `measured` value. The
  dimension kinds and `embedded` also echo the requirement (the
  resolved `value`/`tol`, or the band with an omitted edge as
  `null`). The pair kinds name the resolved tolerance in the failure
  text. A green report that withholds the
  number forces its reader to raise a tolerance until the entry fails
  and read the gap from the failure. That is a re-run over every
  assert, measured to dominate a revision's tool-wait, so the exact
  search runs once, in the check.
- **Units are the kind's own measure.** `tol`, `min`, `max`, and
  `value` are meters on every length kind and copies on `count`,
  never a unit that depends on which other keys are present: an
  entry is readable on its face.
- **Storage-noise floor.** Vertices are float32, so a gap built
  exactly 0.15 wide reads 0.149999973. Every length comparison
  against an authored `tol` or band edge therefore carries a floor of
  1e-6 m: a miss under it passes, and the entry says `noise_floor`.
  It says so on the green verdict the floor decided, never on a
  failing entry, since an entry making many comparisons may waive one
  and still fail on another. Beyond ~16 m one ulp exceeds 1e-6 and a
  tolerance is the author's to widen. The overlap guard's bar is not
  an authored number and takes no floor.
- **Names resolve to pieces.** A root `groups` name (§4.1) resolves,
  wherever an entry takes an output name, to the union of its members
  — per kind, the last column of the quantifier table. A kind that
  names one output's own copies or faces refuses a group with the
  members listed. The two names a relating kind takes must stand for
  disjoint sets of outputs (an output on both sides would be measured
  against itself: distance 0 whatever the geometry does), relations
  between a group's members stay entries over output names, and the
  entry echoes the resolution under `members`. Rejected: a `group`
  key inside the entry (it serves the one-vs-rest form but not a pair
  of pieces); grouping by reference source (partners dropped with no
  line in the file).
- **Quantifier keys, one spelling each.** A cross-output check sees
  an instanced output as the union of its copies. The opt-in keys
  cover what the union cannot. `"every": [name, ...]` (contact and
  mated, the kinds with two subjects; names drawn from the entry's
  `outputs` or `faces`, so the key says whose copies) or `"every":
  true` (extent, with one subject) checks each copy individually,
  reporting the checked counts as `every` and failing copies as `name[i]`.
  `"self": true` (clearance, embedded) covers every copy pair of the
  entry's instanced outputs, per-copy, never merged, reported as
  `self_pairs` with failing pairs labeled `name[i]`. `"connected":
  true` (embedded) adds one-body connectivity. A boolean quantifier
  key is `true` or absent, because a written `false` would be a
  second spelling of the off state (a name list refuses a boolean).
  The names are explicit rather than implied from whatever is
  instanced, so an entry's meaning cannot change when an output
  later becomes an array.
- **The honesty roster.** A green entry that quantified less than it
  reads names the gap: `unchecked` (a scope not quantified —
  `instanced_copies` for a contact without `every`,
  `instanced_self_pairs` for a clearance without `self`), `ceded`
  (pairs handed to a declared embed), `stronger` (a stronger `mated`
  mode that also holds). The report's `warnings` carry
  `contact_unquantified` as a row naming the `every` fix, because the
  weakening it marks arrives with no file diff and no red verdict. An
  output arrayed after the entry was written turns "touches its
  support" into "some copy touches". The row is for an entry that
  quantified nothing: one whose `every` judged a side copy by copy
  (more than one copy, or a group's members) chose its quantifier,
  and the side it left existential stays in `unchecked` alone, since
  a warning its author cannot clear teaches reading past
  warnings. `--failures-only` rolls the
  roster up rather than folding it away (§6).
- **Owner.** Every authored entry carries `owner: human|agent`
  in the IR, derived from the file that declared it (ownership,
  below).

**The kinds.**

- *The built-in overlap guard* (report kind `no_overlap`, never
  authored): every validate checks every geometry-output pair and
  every copy pair of each instanced output for material
  interpenetration, with no entry in the asset. Built in because the
  requirement is universal — what everyone writes every time is a
  default — and because a guard that exists only if remembered
  contradicts the report's honesty stance. The entry rides the
  report's `asserts` with `builtin: true` and the reserved id
  `no_overlap` (an authored `no_overlap` kind is refused with the
  redirect to `embedded`; so is a user assert under the reserved id).
  Criterion: exact manifold3d intersection, judged by mean
  penetration depth (2·volume / surface area over the intersection's
  *load-bearing* faces) against contact's float32 floor, 1e-5 m. An
  fp flush-contact slab is a few ulps of its coordinate thick, so a
  tighter bar fails correctly flush geometry at asset scale.
  Load-bearing, because coplanar contact (a cutter seated in the
  pocket it cut; flush construction) attaches kernel-exact
  zero-thickness sheets whose area would dilute a real defect under
  the bar. Residue leaves the measure by component (one thinner than
  1e-8 m, the kernel-exact floor), then by the exact overlap, in each
  plane, of the faces of one orientation with the faces of the other,
  computed as a 2D intersection in the plane's own frame, so a face
  only partly under a sheet keeps the rest of its area. No length in
  the measure is near the kernel's resolution, so no position or size
  of the intersection reaches it. An intersection that
  is nothing but residue is flush contact, depth 0. A box
  overlap thinner than the bar on any axis is under the bar before
  any solid is built, so flush-built kits clear their
  touching pairs without kernel work. One entry, `pairs`/`self_pairs`
  counted, failing pairs only; touching passes. The copy pass keeps
  copies per-copy, failing pairs `name[i]`. A non-watertight
  prototype fails only where box overlap leaves copy pairs
  unresolved. The guard's parameters are spec-frozen: no asset-level
  off switch (the escape is the declared `embedded`), no criterion
  knob (a volume bar grows with contact area, so a
  small-cross-section part escapes deeper under the same number),
  and changing them is a breaking release, since an asset's verdict
  would change with no file diff. Its admission test binds any future
  built-in assert: universal field-authored text, a human-owned
  requirement sentence, and a pair-level structure per-output checks
  cannot express. Defaults otherwise stay checks (§4.1).
- *`contact` and `clearance`*: the minimum surface distance between
  two named outputs, measured on world-space triangles
  (watertightness not required), must be ≤ `tol` (contact, default
  1e-5 m) or ≥ `min` (clearance, required), both inclusive.
  Crossing surfaces count as distance 0: contact composes with the
  overlap guard into "resting contact", clearance fails on any touch
  or pierce. A fully contained output is invisible to surface
  distance. Material separation stays the guard's domain and
  enclosure `not_enclosed`'s. Both take two named outputs, never a
  universal scope on both sides (universal contact is contradictory;
  universal clearance outlaws flush assembly). Clearance rejects
  `min: 0` (vacuously true). A quantified entry adds the loosest copy
  (`every_max`, contact) or the closest copy pair (`self_min`,
  clearance — a doubling box search, so a green spacing check names
  its tightest gap without measuring every pair). Beside a pair,
  clearance's `self` holds both outputs' copy pairs to the pair's
  `min`. Over one output alone it measures that output's copy pairs
  and no partner, a spacing stated under a `min` of its own.
  (Rejected: `self` as a list of the
  pair's names. It keeps an output that overlaps itself by design out
  of the scope, and still judges the spacing by the gap to a partner
  the requirement never mentioned.) Clearance alone has a one-vs-rest form,
  `"outputs": [a], "others": true`: `a` keeps `min` from every other
  geometry output the asset has at validate time (images are not
  partners), so a part added later is covered instead of being a pair
  nobody wrote. The entry reports `pairs`, `nearest` with its
  `distance`, and `detail.failing_partners`. `self` composes over
  `a`'s copies. If there are no other geometry outputs, the
  one-vs-rest check passes vacuously. Contact refuses the key.
  *Tolerance ownership:* contact's `tol` is a numerics floor owned by
  the tool. Above ~16 m one float32 ulp exceeds 1e-6 and a
  correctly flush contact reads as a gap; 1e-5 keeps every ≤ 160 m
  asset safe while staying 100× below the 1 mm modeling-error floor.
  Clearance's bound is a design quantity owned by the author, so it
  is `min`, the key a band's lower edge has on every other kind, with
  no default. (`tol` was rejected as its name: on every other kind
  `tol` is an allowance around a target, and `tol = 0.05` reads as
  the opposite of "at least 5 cm apart". A default was rejected with
  it: a design value an entry need not write moves the verdict with
  no diff in the file.)
  *The bearing form.* Distance alone cannot tell a seat from a graze:
  an edge or a corner on a face is distance 0 like a face on a face,
  so instanced steps meeting their stringers along an edge pass every
  contact. `min_area` (square meters, author-owned like clearance's
  `min`) requires the face area the two subjects share to be at least
  that much, beside the distance condition. The shared area is the
  overlap of the two face sets in a common plane: over the triangle
  pairs whose normals are anti-parallel within a tool-owned floor
  (0.01°, `mated`'s default) and whose planes lie within `tol` of each
  other, the in-plane intersection areas summed, exact because each
  side's triangles partition its own surface. The entry carries
  `area`, and under `every` the copy sharing least as `area_min` and
  the copies short of the bound as `bare_copies`, each copy measured
  against the other subject as a whole; a group's area sums its
  members' faces. The storage-noise floor applies to the area's
  side length. A curved seat shares area only where the two
  tessellations put a facet on one plane, so a shaft in a bore is
  `mated`'s question. A written 0 is refused, being the plain form.
  The measure has no up: a step let into a stringer bears on a
  vertical face, which a form reading the face's orientation (a
  `bearing` kind, "a rests on b") would miss; and the sentence is
  contact's own, "touches, rests on", with a second measure, so it is
  a key and not a kind (the rule `mated`'s modes follow).
- *`mated`*: two tagged faces, `"faces": [[output, tag], [output,
  tag]]` (a third element is the 0-based copy index, required when
  that output is instanced; `faces`, not `outputs`, because a part
  has many faces), must satisfy the entry's `mode`. The frame is the
  `mate` op's (§5.1), read from the evaluated outputs' world-space
  faces — never from the op's transform, so the declaration survives
  a later replacement of the op. `"coincident"` (the default): (a)
  normals anti-parallel within `angle_tol_deg` (default 0.01), (b)
  mating-frame origins within `tol` (default 1e-5 m; both defaults
  tool-owned numerics floors). It is the check `contact` cannot be: a
  part slid across its port still touches at distance 0, while its
  origins are the slide apart. `"contained"` covers the joint
  coincidence cannot: a smaller part butted against a larger blank
  face, whose centroids legitimately differ. It first checks stage
  (a). It then requires the faces to be coplanar within `tol` along
  the mating normal, and the part's (`faces[1]`'s — order carries the
  roles) contour to lie inside the anchor's outer contour. Inside
  means boundary vertices inside
  or within `tol` of the rim, the rim allowance honored by the
  edge-crossing test too, so a flush-seated edge is containment while
  a part edge leaving the contour beyond the allowance fails outright
  (a non-convex anchor's spanned notch is caught even with every
  vertex inside). Holes in the anchor face are outside the criterion
  (a blank flange may cover a bore). The in-plane position is free by
  design, `contact` and the overlap guard carrying the seated-joint
  half. Entries carry the measured angle and, in coincident mode,
  the offset and distance, or in contained mode the plane gap, the
  outside reach and the crossing count, each face's origin provenance
  word (`opening` / `centroid`), and the mode, so a switch to the
  weaker registration is visible in the report, not just the file
  diff. A green
  contained entry whose origins also register within `tol` carries
  `stronger: "coincident"`. Under `every` (the table) a named
  output's face carries no index, two named outputs pair copy by copy
  and differing counts fail the entry with both numbers, and one
  named output puts each copy against the other face as written. The
  entry reports the pair count and each
  measure's maximum over the pairs (`angle_deg_max`, `distance_max`,
  and so on), as contact's `every_max` reports the loosest copy. Indexed
  entries do not replace the key. A recipe writes them in a loop only
  by copying a copy count it cannot observe (§4.8), and a copy added
  later is then a joint no entry names and no report line mentions.
  (Rejected as the pairing: any copy of the other output, contact's
  existential, under which a part registered on its neighbor's seat
  passes.) Contained is a mode, not a kind: both
  modes answer one requirement sentence ("this part butts that face")
  over one subject. A kind is warranted only when the subject or the
  sentence differs, as between `extent` and `count`. Roll and contour
  equality are not checked in either mode. The explicit `roll_deg`
  on the op is the author's statement.
- *`embedded`*: the positive form of the penetration-depth axis. It
  is the declared intent that material interpenetrates, judged on
  the guard's load-bearing depth measure in a band `min` ≤ depth ≤
  `max` (meters). The symmetry closes a hole the guard alone cannot:
  an overlap check has no upper bound on the *away* direction, so a
  repeat unit that must embed into its neighbor can silently separate
  while every overlap entry stays green. The requirement is
  two-sided, and a waive-shaped allowance would erase the check
  without stating it. Three forms (the table): the *pair form* names
  two outputs and measures their intersection. The *copy forms* name
  one output: `self` requires every
  copy pair overlapping above the defect floor to lie in the band,
  `connected` additionally requires the copies to form one body
  through embeds ≥ `min` (connected components over copy pairs:
  stronger than "no isolated copy", and stronger than contact, since
  copies link through material embeds, never mere touch). One output
  is one body, so a deliberate two-body output is split into
  outputs. `min` defaults to 1e-3 m for the pair form and to 0 for
  `self`. The pair default is the "robustly embedded" floor, so an
  omitted `min` is the union-seam rule written as a declaration,
  never a floorless one. Changing it is a breaking release, since
  an entry's verdict would move with no diff in its file. A written
  pair `min` 0 is the explicit free-overlap spelling for a depth
  with no design value (a scatter sunk into displaced
  terrain measures whatever its seed gives). A positive `min` is a
  positive claim: copies that have all separated fail it. `connected`
  rejects a written 0, and every form rejects a band edge above 0
  and at or below the guard's defect floor (a written 0 is the
  free-overlap spelling, not a floor). Overlap that shallow reads as
  flush contact and could never be policed. `max` is optional: a
  parameter-dependent depth has no statable ceiling, and forcing one
  would enshrine a measured number. Every entry carries the resolved
  band with `max: null` as the visible spelling of
  declared-unbounded. The pure free-overlap declarations (a pair
  `min` 0 without `max`, a `self` without bounds) measure nothing: a
  band admitting every depth reads no verdict from the kernel, so the
  exact intersection (measured as the largest single cost of a dense
  scatter's validate) is skipped and the entry reports
  `measured: false` (the copy form adding the broad-phase
  `candidate_pairs`). Their content is the declaration itself, owned
  and traveling through `asset_ref`. A band is one kind, not two
  one-sided entries: the declaration that opens a hole in the guard
  must carry its own ceiling in the same entry, or the ceiling half
  goes unwritten the way unpaired waivers do. *Cede.* A declared
  embed takes its pairs over from the guard (the pair form its
  cross-output pair, the copy forms that output's copy pairs) and
  polices them under its own band whether it passes or fails. The
  guard reports them under `ceded`: never a silent gap, never a
  double fire. Cede is definitional: the guard's requirement sentence
  reads "no interpenetration *outside declared embedded bands*", so
  writing an embed adds an agent-owned checkable claim rather than
  weakening the human-owned guard. (A per-output checks waiver was
  rejected for intended overlap: a waiver has no `owner` and does not
  cross `asset_ref`, exactly the two properties a declared intent
  needs.) *Cede through references (§4.6).* Embeds declared inside a
  referenced asset travel with its parts. A consumer output that
  reaches an `asset_ref` through ops declaring how they place their
  input maps into that reference's (asset, override identity). That
  qualification is an op trait, never a name list:
  position-preserving ops, rigid `transform`s composed into one
  placement, and data-placed ops (`place`, `mate`) that move the part
  as one body. Two consumer outputs mapping into the same identity
  under the same composed placement, neither data-placed, inherit the
  sub-asset's declared pair embeds, and an output inheriting a
  copy-form embed cedes its own copy pairs. A rigid motion of the
  whole set keeps every pair's depth, which the sub-asset's own
  per-identity validation already proved. A scaled placement, a
  data-placed (re-seated) pair, or a different override identity does
  not map. Those pairs stay with the consumer's guard until the
  consumer declares them, since a consumer that re-parameterizes or
  re-seats a kit is re-making its joints. The same trace runs inside
  every per-identity bundle, so a kit that itself references an
  embedding kit keeps that cede.
- *`extent` and `count` — the dimension kinds*: `query`'s own
  measurements made gating. A requirement that states a number — a
  total rise, a diameter, a height off the ground, a copy count — is
  otherwise read back interactively and written into prose, where it
  rots the moment the geometry changes. *`extent`*: one subject,
  `"outputs": [a]`, the output's world-space vertex bounds with its
  copies united (actual transformed vertices, never transformed box
  corners, so a tilted copy of a curved prototype reads true where
  `stats` bounds cannot), or `"faces": [output, tag]` / `[output,
  tag, copy]`, one tagged face set on one copy (the index required on
  an instanced output, as `mated` requires it). It is read through
  one `measure` token, `size|min|max|center` × `x|y|z`, the face
  predicates' spelling along an axis. The judgment is `value` within
  `tol` (default 1e-3 m, the author's allowance, the scale at which
  a stated dimension is met) or a `min`/`max` band with at least one
  edge, never both in one entry. No radius key: a diameter is a `size_*`. The ground
  reads as `min_y`: the per-output statement the asset-level
  `grounded` check (§4.1) deliberately does not make. Under `every`
  (the table) the face form drops the index, and the entry reports
  the measurements' span as `measured_range`. The two forms state
  different requirements (a
  flight's total rise vs. every leg reaching the ground), so an entry
  without `every` over an instanced output is not a weakened one and
  carries no `unchecked`. *`count`*: `"outputs": [a]`, the output's
  copy count (a plain mesh counts one) against `value` or an integer
  `min`/`max` band (`min` equal to `max` is refused: that band is
  `value`). It is the check `array` and `scatter` are paired with,
  read from the evaluated copies rather than the parameter. A
  required count held only in a node parameter is agent-editable
  with no approval and no report line (the leak §8.3 measures). In a
  human-owned entry it is a requirement, and an output later baked
  or unioned into one mesh reads one and fails, where `contact`'s
  `every` would silently read stronger.
- *Quantifiers over instanced outputs, and groups* — the map the
  keys follow:

  | kind | against the other output | between the copies of one output | a group name in place of an output |
  |---|---|---|---|
  | overlap guard | ∀ by construction (any copy's intersection is the union's) | every copy pair, by default | outputs only; a declared embed between groups cedes the member cross pairs |
  | `contact` | existential on the union; `every` makes it ∀ per named output, distance and `min_area` alike | none: all pairs touching is unsatisfiable beyond two copies, and a "no isolated copy" (∀∃) form has no demand — if it arrives it is a new spelling, not `self` | the members' union; `every` naming a group quantifies every member's copies |
  | `clearance` | ∀ by construction (the union minimum is the per-copy minimum), so `every` is refused | `self` — every copy pair ≥ `min`; over one output alone, the copy form | the union; the one-vs-rest partners are the geometry outputs outside the group (its own members touch by construction); `self` refused |
  | `embedded` | pair form on the unions (all-pairs embed is unsatisfiable, like contact's ∀) | `self` — band over every overlapping pair; `connected` — the band plus one body | pair form: the union's intersection depth; `self`/`connected` refused |
  | `mated` | one indexed copy per face; `every` judges every copy of the named outputs, two named outputs paired by index | — | refused: a group has no faces |
  | `extent` | no partner: the output form measures the union of the copies, the face form one indexed copy | `every` — each copy's own bounds or face set against the judgment | output form: the members' united bounds, `every` every member's copies; face form refused |
  | `count` | no partner: the copies are the measure | — | refused: a sum of copy counts is no output's count |

**Ownership.** Every asset may have a requirements file beside it,
`<name>.requirements.star` for the asset `<name>.star`, root and
sub-asset alike, in the recipe's language with the builtins of
§4.8. Every assert carries `owner: human|agent` in the IR, and the
owner is the file: an entry declared in a requirements file is
human-owned, an entry declared in an asset file is agent-owned. The
file is the one whose top level is running. A function an asset calls
declares the asset's entry wherever the function was written, so a
threshold the agent passes never becomes a human-owned one. Budgets
live in the requirements file, since they are human-owned acceptance
criteria by definition (§4.1). The agent *appends* to a requirements
file when it transcribes a request (assert-first, §8.1). Editing or
deleting a line there is the change that needs human approval (the
agent proposes, per P4). A request that states a requirement's new
number is the approval of that one edit, the human's number written
in place of the human's number. A number the agent derived from the
request puts the agent's arithmetic in that place, and waits for
approval; so does a request that names the change and leaves the
number to the agent. Agent-owned invariants stay freely editable.
A requirements file addresses its own asset's outputs, groups, and
knobs in that asset's local frame. A group a human-owned entry names
is declared in that file, since a group's members are the entry's
reach, and one declared in the asset file is the agent's to narrow
with no diff in the human's file: the build warns of it
(`group_not_owned`), with the move as the fix. A sub-asset's is evaluated
with the sub-asset's asserts, once per override identity (§4.6), so
a kit's own requirements (a roof thickness, a column diameter, what
rests on what) stay with the kit, hold for every variant a consumer
builds, and are checked when the kit is verified alone. An asset
without the file has no human-owned entries but the guard below.

Ownership matters because the asset author is the agent, and an
assertion it may freely edit is one it can weaken to pass (the
self-acceptance gap, §8.3). The file carries it rather than a field
for two reasons. No entry can lack an owner, so none is ever guessed,
and either guess would be wrong: `agent` hands a forgotten human
requirement to the agent to weaken, and `human` freezes the agent's
own invariants behind approval. And a requirement sitting inside the
agent's loops can be dropped by a refactor without a line that reads
as a deletion, while a requirements file changes only by a diff of
that file, which `build --diff` lists under its own key (§6). Profile
policy lives outside the asset (§9). Ownership follows the
requirement through translation: a threshold the agent derives from
a human-stated goal is still human-owned. The arithmetic is
translation, not authorship. Only a **substituted claim** is
agent-owned: a different checkable requirement standing in for what
the human asked, because the stated one has no assert spelling. The
substitution is recorded in the project journal naming the human's
words. Failure messages offer tolerance-weakening advice only on
agent-owned entries: a human-owned entry's message says "fix the
geometry", never steering the reader toward the forbidden constraint
edit.

**A human-owned threshold cannot be built from the recipe.** A
language lets the number that built the geometry become the number
that judges it, at which point the requirement moves with the
mistake and the entry can never fail. The guard is structural, not
syntactic. A requirements file loads nothing and no builtin in it
returns a recipe value (§4.8), so its thresholds can only be its own
constants and the arithmetic and loops it writes over them, or a
knob's resolved value (`mo.param_value`, human-owned data through the
viewer) used whole (§4.8). A list of six roof widths written there
and iterated into six asserts is the human's enumeration, not a
measurement. The dependency runs the other way by design. An asset
may `load()` the requirement constants and build against them, so a
requirement changed by the human moves the geometry and the check
together in the one direction that is sound. A constant the agent
chose lives in the asset, and an agent-owned entry that reuses it is
reported as a warning (`threshold_tautology`). The agent may weaken
its own invariants but should see itself doing it.

The build finds the reuse in the asset file's text, as it finds a
knob's use (§7.2): a threshold written with a name the asset binds
at its top level or loads from `lib/`, where the same name is read
outside every assert. It reports one row per constant, with the line
that binds it and the entries it judges, in `build`'s report and
among `validate`'s warnings. A name loaded from the requirements
file draws no row, nor does a value computed from such names alone,
nor a knob's value. A threshold that reaches its assert as a
function's argument or a loop's variable is not traced. (Comparing a
threshold with the values of the nodes it judges was rejected:
thresholds and params share values such as 0 and 1 by chance far
more often than by a shared constant.)

The built-in overlap guard's owner is `human` by **spec assignment**,
because "no part may interpenetrate another" is a standing human
requirement, not a tool invariant. The assignment is load-bearing.
Were the guard tool-owned, an agent could gut it by authoring its own
asset-wide policy without approval. Human-owned, its only softening
is the declared `embedded` — visible, banded, and sanctioned as a
substituted claim.

**Checked, never solved.** Satisfying asserts is the agent's job in
the verify loop, not a kernel solver's. A CAD-style constraint solver
is out of scope. The clause governs **assertions**. The checking
layer never mutates geometry to satisfy a failing constraint.
Operators *are* solvers (`place`, `boolean`, `scatter`, `unwrap` each
solve a sub-problem) and may solve, under the contracts ops already
carry: a deterministic, unique solution; everything consulted appears
as an input edge (the DAG stays honest); independent checking, per
the pairing rule above; and a structured, actionable error when
unsolvable.

---

## 6. Runtime CLI

Deterministic, agent-free commands (P2):

```
moraine --version                           # {"version": "..."} — the version journal stamps carry (§4.7)
moraine init [DIR]                          # scaffold a workspace: AGENTS.md created once, guide synced to the install (§4.2); journal retirement pass (§4.7)
moraine build [PROJECT] [--check] [--diff [PATH]]   # recipe → IR under build/ (§4.8); --check compares against the IR on disk and writes nothing; --diff reports the graph change since the last build (or against the IR at PATH)
moraine eval [PROJECT] [--output NAME] [--recompute]   # build, then evaluate and populate the cache; --recompute = the determinism check (§4.4)
moraine stats [PROJECT]                     # part/tri/vert counts, bounds, materials — compact JSON
moraine validate [PROJECT] [--profile gltf] [--param ID=VALUE]   # profile validation report + asserts — compact JSON; --param judges under a knob value and writes nothing
moraine query [PROJECT] [--output NAME] [--select '{...}'] [--tag NAME] [--instance I]   # face-predicate query over outputs (§5.1), or the faces a tag caught — compact JSON
moraine export [PROJECT] [--profile gltf] [-o exports/] [--bake-instances] [--recompute]
moraine serve [PROJECT] [--no-open]         # HTTP+WS server + viewer, watches recipe files and params (workspace root: all projects, viewer picker); opens the viewer unless a tab is already connected (§7.1)
moraine screenshot [PROJECT] [--view front|iso|... | --azimuth DEG --elevation DEG] [--distance M] [--target X,Y,Z] [--size WxH] [--param ID=VALUE] -o shot.png   # via running server; shoots PROJECT (default: cwd's); --param shoots it under a knob value and writes nothing
moraine docs [SECTION]                      # shipped reference: recipe API, asset format, op contracts, textures, checks, assertions, CLI, journal, examples, and the generated `mo` namespace; a section prints alone, as text, naming the sections under it and the examples it cites; SECTION = examples/<name> prints one
moraine mcp                                 # MCP server over stdio: stats, validate, query, export, screenshot, docs as tools (§8.4)
moraine sessions [PATH] [--gap MIN] [--last N | --all]   # session-log summary: sessions, per-project rounds, command counts
moraine pins list|add|clear [PROJECT]       # deixis pins on the running server (§7.2): read the human's, place the agent's
moraine cache stats|clear|evict             # cache size/composition, full clear, forced cap enforcement (§4.5)
```

All output is terse machine-readable JSON on stdout. Human-readable
rendering of the same data is the viewer's job (§7). The one
exception is a page of the shipped reference: `docs SECTION` prints
the section's own text, which a reader would otherwise need a second
process to take out of the JSON. The reference is ASCII, so it prints
on every console. The `docs` listing and every error stay JSON.
Every evaluated
report (`build`, `eval`, `stats`, `validate`, `query`; a sweep's
`workspace` aggregate) leads with `moraine_version`, the key the
export manifest (§9) and the `screenshot` report already carry: a
second install on the same machine is the ordinary way a green report
gets read against the wrong version.

**Targets.** `PROJECT` is a project directory holding its root asset
file (default: cwd) or an explicit recipe path (which may be a sub-asset
under `subassets/`, §4.6). Every evaluating command builds first, the
closure included (§4.6), and a build failure is the command's failure,
reported with the recipe site. A `.mora` path is accepted as an escape
hatch for the runtime's own debugging (it evaluates that IR as given)
and never classifies a project (§4.2). Read-side commands (`validate`,
`stats`) also accept a workspace directory (§4.2) and sweep its projects
into one rollup: `{"projects": {<name>: <report>}, "workspace":
<aggregate>}`, the per-project shape identical to the single-project
command so same-named outputs never collide. The sweep exits 0 only when
every project passes, 1 when any fails to load, evaluate, or validate. A
workspace with no projects is an error (`empty_workspace`), never an
empty green sweep. The aggregate carries `not_swept` (§4.2) and stays a
plain per-project sum under cross-project references. Every project's
GLB is self-contained, so the sum reads as shipped-per-project content,
not a deduplicated source count. `serve` accepts a workspace root and
serves all its projects, starting on the lexicographically first (§7.2).
Write-side commands (`eval`, `export`) take one project or asset per
invocation. Pointed at a workspace, they fail with an actionable
`workspace_target` error listing its projects.

**Paths in reports.** A relative `export -o` resolves against the
project (an export is a project artifact, so `-o exports/` lands
inside the project from any invocation directory). Absolute paths are
taken as given. Windows drive-relative or rooted spellings are
refused rather than written somewhere silent. A sub-asset sharing its
stem with the root asset is refused at export, since the out dir is
shared per project. `export`'s `glb`/`manifest` paths are
cwd-relative when the target is inside cwd, so the in-project and
workspace-root flows quote journal-safely. Otherwise they echo the
resolved path (the author pointed there). `screenshot` reports a
workspace-relative asset path, never an absolute one, since reports
get quoted into committed journals. `screenshot -o` stays
cwd-relative: a screenshot is a session artifact, not project
content.

**`init`** targets the workspace root — `AGENTS.md` created once, the
guide re-synced (§4.2) — and runs the journal retirement
pass (§4.7). Its report carries `version`, `journal_review` (one row
per keyless stale-stamped workaround entry — `journal`, `line`,
`version`, `entry` — with claim-keyed entries collapsed to one row
per limit: `key`, `entries`, `journals`, `versions`), `journals`
(`scanned`, `entries`, `unstamped` with `unstamped_by_journal`,
`dangling`, `skipped`). An unreadable journal degrades to a stderr
line and a `skipped` count, never a failed scaffold. The scaffold
itself reports `initialized`, the files `created` and `skipped`, and
the `guide` state.

**`build`** writes the IR of the target and its reference closure
under each project's `build/` and reports `build_version` (§4.4),
the files written and the count left `unchanged`, the files
`removed`, the names `unrecognized`,
what `blocked` a removal in the project's `build/` (§4.8), the asset
files left `unread` (§7.2), and per asset the node, assert, and
output counts, and the build time (`ms`). Its errors carry the
recipe `site` and `via` (§4.8).
A `params.json` that is not one object of knob values fails the
build too (`bad_params_file`), naming the file and the line.
`--check` builds to memory and
compares with the IR on disk, file by file: `ok`, or each file that
differs with its first differing line quoted from both and each file
the disk does not hold, and never a write. It names as `stale` what
the next written build would remove. `--diff` reports
the graph change the last edit made, so an edit inside a loop or a
function shows which nodes it reached: nodes added, removed, and
changed (with the changed params), asserts added, removed, and
changed with human-owned entries listed under their own key, and
the outputs made dirty. `serve` runs the same diff on every watch
reload and logs its counts. The comparison base is the IR on disk
before the build, or the IR file or `build/` directory `--diff PATH`
names, so a review compares two revisions by building each, and the
tool never reads version control. The diff compares by id, so the
order of a file is no change and neither is a `site`. Of the nodes
added and removed, the pairs that are one subgraph under two ids are
named `renamed`, since the cache, pins, and the journal know a node
by its id (§4.1). Equal content is all a pair shares, so a node
removed and an equal one added elsewhere pair too. A pair whose op
salts its streams with the node id is marked `stream_changed`, the
new id being a new result (§4.5). An asset is
compared with the one the same reference read before, so a changed
override, which opens a variant under another name, reads as the
values it moved. Under a named base, an asset an existing reference
read whose IR the base does not hold (another project's, where one
project's `build/` was saved) is left out and listed
(`not_compared`), and the outputs that read it are not dirty for it.
An asset only a new reference reads is new. Read as all added, it buried an unedited consumer's
empty diff. Past ten entries a list names ten and counts the rest by
the recipe line that made them (`by_site`), and where they came
through calls, by the outermost call as well (`by_caller`), since an
edit every part reaches through one helper is otherwise one row.

**The `consumers` key.** A project's own checks do not run the checks
of the projects that reference it, so a change to a referenced
project can break a consumer while every report of the project stays
green. `build` and `validate` name the consumers: `projects`, the
other projects of the workspace whose roots reach an asset of this
project on disk, `not_built`, the projects with no root IR to read a
reference from, and `unread`, those whose IR could not be read, with
the sweep of the workspace root as the `fix`. Where no project is
known to reference this one, the `fix` says how to find out, not
what to check. The list is what the IR on disk says, so a project is
named until its own build says otherwise. The key is absent where
every list is empty, and it never gates.

**`query --select`** evaluates one face predicate (§5.1) over
evaluated outputs and reports match count, total area,
matched-face bounds, and the area-weighted `centroid` per output, so
token cost scales with the answer, never the asset. `--tag NAME`
answers for the faces a tag caught, alone or narrowed by `--select`.
The tag's predicate is written from the recipe's values, so a
predicate retyped for the query would not be the tag, and the
centroid is the origin `place` anchors on (§5.1), which the recipe
cannot compute where a rebuild shaped the face. An output without the
tag is skipped with the tags it carries, and a tag no output carries
is an error. An instanced output is read on its prototype in its
local frame — the frame `tag` writes in — and says so with its copy
count. `--instance I` reads copy I where it stands in the world, the
frame an arrayed anchor is taken in. `query` reads outputs, so an
anchor on a node that is none is read by publishing it for the
question.

**The `profile` key** rides `eval`/`stats` with disk-cache hit/miss
counts, total per-node wall time, and the five slowest misses.
Misses are the only optimization targets, so a large asset never
dumps one line per node. The touch-on-hit that feeds the counters is
the §4.5 LRU signal. Under `--recompute` (§4.4) `eval` and `export`
add `recompute` — `nodes`, `matched`, `unrecorded`, and on a mismatch
`mismatched_total` with the first few `mismatched` nodes and a `fix`
naming the op — and exit 1 with the report intact. A mismatched
output carries `mismatched: true` in place of its artifact, and
`export` then reports `ok: false` and writes nothing (a GLB from
values the cache disagrees with is not an export to trust; the last
good export stays).

**`validate` report keys beyond the checks.** Every entry that names a
node or an assert carries its recipe `site` (§4.1), and one made
inside a function its `via` (§4.8). When several failing entries share
one site the report adds `by_site`, a rollup of counts per site, so
one wrong expression in a loop reads as one line with a count rather
than as its expansion. Where the entries of a site were reached
through more than one call, its row counts them by their `via`.
`material_name_unique` per
output (one material name bound to two contents — the GLB writer's hard
error, mirrored so green never outruns `export`). The fix names the
colliding outputs and, for a reference-borne collision, the `asset_ref`
node and `material_rename` (§4.6). When the asset declares budgets
(§4.1; counting rules §9.1), the report includes a top-level `budgets`
list. Each entry contains `ok`, `count`, `limit`, `identities` with
per-identity outputs and transparent/emissive flags, and `fix` on
failure. When the asset declares knobs that carry a range (§7.2), the
report includes a top-level `params` list, one gating entry per knob
in declaration order: `ok`, the `value` the evaluation used, its
`from` (§4.1), `min`, `max`, and `fix` on failure. The `fix` of an
accepted value names `params.json` and the viewer, since the agent
does not edit that file. A knob without a range has an entry where
`params.json` holds a value for it, and the entry fails where that
value is a `mismatch` (§4.1). A referenced asset's knobs gate the
consumer per override identity as its asserts do (§4.6). A failing
one is listed with its `asset`, and the green ones fold into the
`ref_assets` row as `params_total` / `params_ok`.
`--failures-only` compresses every gating list to counts plus
failures (`budgets_total` / `budgets_ok` and so on) and keeps the
honesty machinery as counts. It names the knobs that hold an accepted
value (`params_accepted`), since a default the agent edits moves
nothing while one stands and the green entry that says so is counted
away. A green entry carrying an `unchecked`,
`ceded`, or `stronger` key (§5.2) is counted under that key at the top
level, `entries` and, by name, what the entries hold
(`"ceded": {"entries": 1, "pairs": 312}`), in a green report and a red
one alike. The rows are the full report's, and a failing entry keeps
its own keys whole. `ref_assets` (§4.6), `mates` (§5.1), `by_site`,
`consumers`, `warnings`, and `serve` pass through. A report that says
less still says that something was not checked, and how much.
(Keeping the rows was rejected: a green report of a large asset then
runs to tens of kilobytes, and its reader greps the counts out.)
`stats` rolls the per-output
entries up under `asset` in the budget's scope: `parts`, `tris`/`verts`
as rendered totals with instance counts multiplied in, and
`asset.materials` (`count`, `identities`, the declared `budget` when
present) from cached metadata alone. Content identity is `validate`'s
job, and the two counts agree whenever the uniqueness gate is green.

**`validate --param ID=VALUE`**, repeatable, judges one project
under the values given for its declared knobs: each takes precedence
over an accepted value and must be of the knob's kind, as a
consumer's offer must (§4.6). One outside the declared range is
judged all the same, and fails the knob's entry in `params`: a
range is stated for a reason, and where the shape breaks beyond it
is found by looking, with no declaration widened and put back. A
string knob takes VALUE
as typed, every other knob as JSON, and an id given twice is
refused. Neither `params.json` nor the IR is written, the evaluation
goes to the preview bucket (§4.5), and the viewer does not move. The
report names the values (`params_given`). A declared range is swept
this way, one knob and in combination, while an accepted value
stands. `screenshot --param` takes the same values and shoots the
asset under them: the report names them (`params_given`), and the
shot is then no evidence of the file state. A material factor moves
no geometry, so its range is checked in pixels, and this is the way
to do it with the recipe and the human's viewer untouched.

**The `serve` key.** `validate` and `stats` probe the server on
`--port` (`GET /api/status`, answered from memory, under a fixed
budget alongside the command's own work) and, when it serves the
same root, carry `{port, project, watch}` — the file-watch liveness
the screenshot report also names (§7.2), the one fact agent-side
checks cannot see — plus `listen_rearmed` when the listener has died
and re-armed (§7.2). A server of another root is carried as
`{port, other_root}`, so it is never taken for this root's. No server
or a foreign listener leaves the key out, never an error. A request
on the pins and screenshot channels names the root its command speaks
for, and a server of another root refuses it (`wrong_workspace`)
before resolving the project, where a project of the same name would
answer in the caller's place. `serve` claims its port before it
evaluates anything, and on a held port reports `port_in_use` naming
the holder's root and nothing else. Every server channel — this
probe, pins, screenshot — first checks that the port accepts a
connection within a budget a listening server always meets (the
kernel completes a loopback handshake even mid-evaluation). A
firewalled port drops the SYN, and without the check "no server"
would cost the channel's full timeout and read as a busy server.

**`screenshot`** renders the current file state of the project the
command speaks for (`PROJECT`, default the working directory's;
outside any project, the displayed one) through the dedicated render
session (§7.2). The camera is a named view or free azimuth/elevation
(exclusive with `--view`; the §7.2 convention), refined by
`--distance`/`--target` (default: auto-fit). The report names what
was rendered: port, project, workspace-relative recipe path, asset
content hash (24-hex over every contributing recipe file's bytes),
moraine version, watch liveness, and the resolved camera. On
`no_viewer` the CLI opens the viewer in the human's browser itself
and retries while the tab connects, waiting first for a tab to appear
so two shots racing on an empty server do not each add one. A host
with no browser gets the original error, and the launcher runs under
a bounded wait so a foreground browser cannot hang the command.

**`pins`** talks to the running server's pin store over HTTP
(`/api/pins`, origin-gated like the screenshot endpoint; no viewer
tab required), addressed like `screenshot` to the project the command
speaks for. `list` returns its pins with the current revision hash,
so staleness is checkable per pin. When the viewer shows another
project the result is empty, naming the displayed project, and
`add`/`clear` are refused (`not_displayed`, §7.2).
`add --output NAME --select '{...}'` places a pin by predicate (P6),
or `--face N` by face index, `--instance I` where the output is
instanced. `clear [--id N]` removes one or all. Resolution rules,
record shape, and lifetime are the §7.2 contract. The report is the
resolved record itself, so a placement is its own verification.

**Session log.** The runtime keeps its own record of what its entry
points did. It is the durable instrument behind every measurement of
the authoring loop (the authoring benchmark reads its counts from it;
a hand count from a transcript is not evidence). Events: one `command`
per CLI invocation but `serve` and `mcp` (each tool call records its
own); from the server `serve_start` (a workspace names its
`projects`) / `serve_stop`, `watch_dead` / `watch_alive` and
`listen_dead` / `listen_alive` per outage and re-arm (§7.2), one
`reload` per file-state evaluation (`trigger`
`startup|watch|set_asset|cache_lost`), one
`accept` per viewer accept (sorted `param_ids`, the outcome, and the
`validate` summary of the accepted state, §7.2, left out where the
pick could move no check, the project was left, or the server stopped
before the check), and one `screenshot` per render. The log records
what the user is shown: a knob scrubbed in the viewer and a pin placed
there are the human's own gestures, which no surface reads back, so
neither leaves an event; an accept does, because it reaches the file
and closes a round. Names, never
values: knob values are asset content and the file diff is their record.
Every event carries `t`
(UTC, ISO-8601), `v` (the moraine version), `pid`, and — where the
command reached an asset — `root` (the enclosing workspace root, or the
project directory outside one), `project`, and `asset_hash`, the 24-hex
content hash over every contributing recipe file's bytes (the root,
every requirements file, the sub-asset and `lib/` files reached,
`params.json`), framed from the same root by every writer so one
revision has one name (a sweep carries a `projects` map). A build
looks at each file once and the hash is over what it found, so a file
saved or created while a command runs is the next revision. A `command`
event adds `cmd`, `ok`, the error `code`, `ms`, `out_bytes` (the stdout
volume the agent read), `via` when another entry point ran the command
in-process (`mcp`, §8.4 — the same command, so the same event), and a
compact `summary` of the verify outcome (`validate`: output count,
failed outputs, red gating checks, failed asserts, budgets, and knobs,
warning count; `build`: the counts, assets, files written, and build
time; `stats`/`eval`: node, output, triangle and instance counts and
the cache profile, `eval --output` counting the evaluated subgraph;
`screenshot`: camera, size, bytes). A `reload` carries `ok`, `ms`, the
error `code`, and on a watch reload `changed_lines` — lines added plus
removed across the recipe files since the previous file state, measured
(a wholesale rewrite past a bound is counted coarsely rather than paying
a quadratic diff on the serving loop) — and the `build --diff` counts
(nodes and asserts added, removed, changed). `changed_lines` leaves
`params.json` out, since a pick is no edit. A missing file is part of a
revision (a dangling `asset_ref`). Any other unreadable file leaves the
event hash-less — an honest gap, never a hash of a file set that did not
exist on disk. The log holds no asset content. What it cannot measure:
model calls and tokens (§8.2 — the tool never initiates them). The
recorded proxies are screenshot count and bytes, edited asset lines, and
output bytes.

Location and discipline: the log is machine-level, one file per UTC
day under the OS log directory (`MORAINE_LOG_DIR` overrides; empty
turns logging off). It is never inside the workspace (§4.2) and never
inside the cache (the log is not regenerable, and §4.5's deletion
gates recognize only cache files). Recording never changes a
command's stdout or exit code: an unwritable log directory costs one
stderr line. The log is a rolling window. The first write of a new
day retires day files older than `MORAINE_LOG_KEEP_DAYS` (default
30; `0` keeps everything), narrowly (the §4.5 discipline scaled to
one file kind): only files directly in the log directory whose whole
name is a day file's, dated before the cutoff, never the current day,
and whose first line proves Moraine wrote them. Durable evidence is
the summary report, archived wherever a measurement is recorded. The
log's absolute `root` stays in the machine-local file.

**`moraine sessions`** is the summary report over the log. For `PATH`
(a workspace root, a project directory, or any directory above them —
default cwd) it selects the events whose `root` is at or under it (a
workspace member's project directory narrows to that project's
events) and derives **sessions** — event runs separated by less than
`--gap` minutes (default 30; `--last N` newest, default 1, or
`--all`) — and, per
project, **rounds**: one asset revision (one `asset_hash`) carrying
at least one verification (`validate`, `stats`, `screenshot`, `eval`,
`query`, `export`, or the `validate` an accept runs, §7.2). Revisions
never verified fold into the next round, a subgraph evaluation
(`--output`) is marked `partial` and opens no round, and hash-less
events attach to the current round (events before the first revision
fold into its round, and alone they open none). Each round reports its
command counts, `validate_ok` and `failed_checks`, `ended_by` (the last
verifier before the next edit — the mechanical "which tier caught it"),
`changed_lines`, `shots`, and wall time. Each project reports its
rounds, verify counts, last known `nodes`, reloads, accepts (picks
that reached the file without an agent round trip and were verified
there, §7.2), and `eval_errors` — every failed command, reload,
render, and accept attributed to the project. Each session reports
its span in minutes, command counts,
`out_bytes`, and totals. Sweep-wide quantities stay on the command,
never multiplied into each project. Root-less command events
(`screenshot`, `docs`, `cache`, `init`, `pins`, `sessions`) join the
session whose span, extended by the gap, contains them, as ambient
counts only (a render is counted once, from the server's `screenshot`
event). The default report reads day files newest first and stops when
the requested sessions are covered. Nothing under the root is an
actionable `no_sessions` error, never an empty green report. "Session"
here is the report's grouping of timestamps — unrelated to the cache's
per-process write counter (§4.5). The server reads the same log with the
same summarizer and serves the newest session to the viewer as the
session timeline (`timeline`, §7.2).

---

## 7. Viewer & Viewer Protocol

### 7.1 First viewer

Browser-based, three.js, served by `moraine serve`. three.js is vendored
into the package (`app/viewer/vendor/`, license notice in
`THIRD_PARTY_NOTICES.md`), never CDN-loaded, for three reasons. The
viewer must work offline with no failure modes outside the tool's
control (the verify loop cannot distinguish "CDN down" from "my asset
broke"). The renderer version is pinned so screenshots — acceptance
inputs — stay reproducible per wheel version (the viewer-side
counterpart of §4.4). And plain vendored ES modules keep
npm/node/bundlers out of the toolchain (the JS-side counterpart of the
§3 dependency rule). The viewer ships WebGLRenderer. A move to
WebGPURenderer is a client-internal swap, invisible to the protocol. No
Electron/Tauri wrapper. Features: hot reload on asset change (<2 s
edit-to-pixels), orbit camera, screenshot endpoint, a parameter panel
auto-generated from `mo.param` declarations, the accept-time
verification warning (§7.2), a project picker on
workspace serve mirroring the server's reported active project
(§7.2), pin display, and the session timeline panel. On evaluation
failure the
viewer retains the last good state (§8.1). This section owns what the
viewer *shows* and how the human *operates* it. The messages and state
transitions behind it are §7.2.

**Opening.** `serve` opens the viewer in the human's default browser
once it is serving, the dev-server convention. It opens only when
nobody is watching: a client that connects while the grace window
runs means a tab already shows this server (tabs re-attach on their
own after a restart). The window is sampled rather than checked once
at its end, so it can be long enough for a retry the browser has
throttled without delaying the open when a tab answers early. A tab
throttled past the window is the accepted residual, and the client
retries immediately on becoming visible. `--no-open` suppresses the
open, a host with no browser silently gets none, and the `viewer` log
event (`opened` / `connected` / `unavailable`) tells the agent at
serve time whether a human can see anything.

**Lighting and exposure.** One rig serves the interactive view and the
screenshot scene alike: hemisphere, key and fill lights, a neutral
environment map so metallic and glossy surfaces read as material
rather than silhouette, and a lit ground at y = 0 whose shadow gives
every view its contact and scale cues, re-framed to the asset bounds
on every load and hidden while the asset extends below ground. The rig
renders through one fixed linear exposure, calibrated so a matte white
surface facing the key sits just below clipping and mid-gray renders
mid-gray, so a shot's tones can be read against the authored factors.
Rejected: a per-shot exposure control (evidence renders must stay
comparable across sessions) and a filmic curve (its shoulder and black
offset move the mid and dark tones). Viewer chrome never occludes file
state.

**Camera.** The orbit camera is undamped (every DCC trains the
expectation that the camera stops when the hand stops) and keeps the
stock web-viewer mapping of left-drag orbit, right-drag pan, wheel
zoom, the standard of the in-browser glTF viewer (§4.3's
ecosystem-standard rule). A DCC-style mapping was rejected: DCC
mappings disagree with each other and lean on middle-drag and
modifier keys that collide with browser and OS behavior. Modifier keys
never remap the camera, because modifiers belong to selection. The stock
control's modified-left-drag pan alias is disabled so a multi-pick with
a few pixels of hand jitter cannot become a silent
camera move. Screenshot requests render on a persistent offscreen
canvas, so the interactive view is never resized or repainted by a
shot.

**Parameter panel.** Knobs render by their interaction tier (§7.2),
grouped so the cost model is legible from the layout alone: live knobs,
then seeds, then the staged group with its Apply control directly
beneath the fields it applies (an Apply floating above unrelated live
sliders reads as governing them). Within the live group, knobs cluster
under their material name once more than one material is represented; a
single-material panel stays flat. The controls:

| tier | control | why |
|---|---|---|
| factor | slider driving the material uniform on every input tick | real-time, no server round trip on the visual path; the client re-applies factor uniforms from the latest schema after every GLB load so a loaded material's last-evaluated factors never undo an active preview |
| evaluation | entry field (typed or stepped), staged behind Apply; commits clamp to the declared range | a slider's drag affordance promises exactly the live response this tier lacks, so it would misread as broken rather than staged |
| seed | value readout, reroll, back/forward over a client-local history | a seed is unordered; ordering it on a slider turns its declared range into an accidental cache-budget declaration |

A knob whose value the human's file holds shows that value as
`accepted`, with the recipe's default beside it, and the two stand
while a preview moves the field. A knob a consumer's override set
is marked `override`. The footer's Reset to defaults previews the recipe's
default for every accepted knob at once (the unset, §7.2, one
evaluation), so the human sees which values are theirs and can hand
the project back to the agent. Accept then removes the lines. The
control is whole-panel, like the two landings beside it: a
knob-by-knob control put a button on every accepted row for a rare
act. Accepting the default's number in place of the unset was
rejected: the line would stand, and a default the agent later edits
would move nothing (§4.8).

One placement rule governs Apply, Accept, Discard, and Reset to
defaults: **always present, disabled while it has nothing to act
on**. A control that appears on
first use is undiscoverable, and the enabled state already says
"something is staged". Staged fields carry a pending accent. While an
Apply's evaluation runs, the control carries the busy state (disabled
and relabeled until the reply) and each field it sent carries an
in-flight mark, a spinning glyph after the label, distinct from the
pending accent and cleared when the reply arrives. Motion is what
peripheral vision catches, a color shift is not. The HUD also
indicates evaluation in progress. A reroll shows the
pick at the click and carries the same busy glyph until that pick's
reply, and the reroll and the two history steps are disabled until
then. A pick is judged by its result, and a seed is unordered, so a
pick made before the last one shows only delays the one that gets
looked at. Accept, Discard, and Reset to defaults scope the whole
overlay, so they render together, apart from the staged group's
Apply, and asymmetrically: Accept is the one emphasized control,
alone at the footer's right; the other two are quiet at the left,
text-weight, so a hand reaching for the landing cannot slip onto the
discard. None warrants a confirmation, because an accept is a
reviewable diff, a discard a redone scrub, and a reset a preview the
discard drops.

**Asset totals.** The display shows asset totals (the rollup `moraine
stats` reports, §6; the preview's when one is active), never a
per-output list, which runs off the window at asset scale while
reading as the whole asset. Image outputs stay out of the line by the
rule that keeps them out of `parts`. The thumbnail strip is their
surface.

**Timeline panel.** Renders the active project's slice of the
`timeline` payload (§7.2) under one rule: it shows only what the
co-present human can act on, so analysis quantities (wall time,
aggregates, totals) stay in `moraine sessions`. The always-visible
header is two signals, progress (the verify-round count) and attention
(amber only while the newest verified revision failed validate, naming
the checks), never a cumulative error count, which outlives its fixes.
Failure history stays legible as the crossed rows in the newest-first
round list, which rests collapsed under the header and opens on a
click, so the two signals are what the panel shows at rest.
Quantities are measured, in their own units and plain
words (shots, commands, lines edited), never estimated tokens, which
the runtime cannot measure (§8.2). A frozen panel says it is frozen
(`stale`, §7.2).

**Pins and selection.** Pin display is client-internal (§7.2 owns the
records). The highlight is derived from the *displayed* geometry on
every refresh, so it follows topology-stable parameter changes, the
tab's own preview included, and it occludes like asset content so a
pointing gesture reads in context. Click selection is primitive-unit:
a click pin highlights the whole clicked primitive (the output, or
that one copy of an instanced output), while the record still carries
the exact face and spot. A predicate pin highlights exactly its
matched polygons, the fine-grained selection tool in the agent's own
vocabulary. Gestures follow the DCC selection convention: a plain
click makes the clicked primitive the sole selection, a background
click clears, and shift-click adds or removes one primitive (these
gestures and `pins clear` own removal; the list carries no delete
control). A gesture changes the highlight at once, from the click's
own raycast; the pins broadcasts that answer it replace that display,
and a refused select restores the recorded selection. The list draws
the record alone. Every pin draws in the one selection accent, so a
highlight can never be mistaken for a material tint. Identical
targets draw once at one opacity, and a marker sphere appears only
for a pin the highlight cannot show (output gone, topology changed,
oversized matched set). Which pin is which is the list's job: rows
name the primitive (`output[copy]`) with the id trailing (the agent's
currency for `unpin`), never the clicked triangle's tag membership,
which would come and go with the click. The list carries the stale
badges and jump-to-pin, and hovering a row singles its pin out.

### 7.2 Viewer Protocol (first-class contract)

The runtime and all display clients communicate exclusively through this
protocol (WebSocket JSON messages + HTTP for bulk payloads). Client-side
rendering and controls are §7.1. This section is the messages, their
state, and the guarantees between them.

**Overlays.** Each connection carries its own preview overlay, the
knob values `set_param` holds, ephemeral and never written to a
file. Every broadcast and every snapshot replayed to a connecting
tab holds file state, and the re-evaluated preview pair
(`asset_updated` + `param_schema`) goes to the overlaying client
alone, so one client's scrubbing is invisible to every other client
and to the agent's evidence channels, which resolve file state. A
reload drops every overlay, on its failure path too, and announces
the drop to each client whose non-empty overlay it discarded with
`preview_dropped {trigger}`, the trigger naming the reload's cause
(`watch`, `set_asset`, `cache_lost`) or `revert`, the one other way
an overlay ends. An agent edit landing mid-exploration therefore
reads as the file winning. (A `preview_active` flag on shared state
was rejected: with the overlay client-local there is nothing to
flag.)

- `asset_updated {project, outputs, glb_urls, image_urls, stats, asset}`
  — runtime → client after re-eval. `image_urls`: image-valued outputs
  get cached sRGB preview PNGs instead of GLBs. `project`, the active
  project's directory name, is the `select` seam, naming the asset a pin
  belongs to. `stats` is per output and `asset` the rollup `moraine
  stats` reports (§6), so the display's totals and the CLI's are one
  derivation (a preview pair rolls up the preview). The urls are
  content-addressed, so a served record is cached by the browser as
  immutable, and a miss on one of those paths is never cacheable
  (`no-store`). Otherwise a cached 404 would sit under the very name a
  re-evaluation restores. A miss on a record the server is serving is
  the cache emptied under it (a machine-wide `cache clear` from another
  process, §4.5). The server re-materializes with one file-state reload
  (trigger `cache_lost`, a reload like any other) and re-broadcasts.
  A tab whose load already failed re-asks on a bounded backoff until
  a newer generation lands, so a clear never strands a connected
  tab.
- `workspace {projects, changed}` — runtime → client, workspace mode
  only: the project roster, replayed first on every connect.
  Absence means single-project serve and no picker. `changed` names the
  roster projects whose recipe files changed on disk while another
  project was displayed, cleared for a project when it is displayed
  again. It is re-sent whenever the set changes, full state, idempotent.
  With several sessions authoring in one workspace, the picker's mark is
  the only signal a session's work sends to a human who is looking
  elsewhere.
- `param_schema {params: [{param_id, label, bound, node_id?, param?,
  kind, value, from?, default?, accepted?, tier, uniform?, min?, max?,
  step?}],
  overridden}` —
  runtime → client, sent after every `asset_updated`. `bound` is `build`
  by default: the value reaches the graph through a rebuild. It is
  `direct` only when the build has recorded that the declared value
  reached exactly one node param whole and unmodified, and nowhere else,
  so that patching that param equals a rebuild. A value placed inside a
  vector, used twice, combined with anything, or consulted by a
  condition is `build`. The build makes the record from the recipe's
  text, since `mo.param` returns a plain value that carries no mark
  of its origin. It is made where the asset's top level binds the
  value to one name and the file writes that name once more, as a
  whole keyword of an operator call the top level makes once. A use
  inside a function or a loop is `build`, as are a value the param
  holds in another number type, one passed on in a reference's
  `overrides` (§4.6), and one a threshold follows (`mo.param_value`,
  §4.8), which is read in a second place. An asset file the reading
  cannot take is `build`
  throughout and named under `unread` in the build's report. `direct`
  is an optimization, not a second semantics, and a test pins that a
  direct patch and a rebuild produce the same IR. (Rebuilding under
  moved values and comparing the IR was rejected: it costs two builds
  per knob, and a condition whose threshold sits between the two
  values goes unseen.) `from` and `default` are the entry's (§4.1).
  `accepted` is the value the human's file holds for the knob, read
  from the file, so it stands in the schema of a preview, whose
  `value` the overlay replaced.
  Values reflect the receiving client's own
  preview overlay if one is active, file state otherwise, and
  `overridden` lists the
  param ids the overlay holds (empty for file state). The overlay is
  server state, so the client's Accept and Discard take their enabled
  state from this field rather than from a guess. Kinds: `number |
  vector | boolean | text`, derived from the declaration's default
  (§4.8), never from where the value lands. A scalar knob may sit inside
  a vector (`translate = [0, level, 0]`) and stays a scalar knob. Number
  and vector declarations must carry min/max. A knob whose value sits
  outside its declared range is reported, not refused (§6). The
  schema carries the value as it is, and the panel marks the knob and
  shows the number, since a control's own range would otherwise pull
  it to the nearest end unseen. `tier` is the knob's
  **parameter interaction tier** — `factor | evaluation | seed` —
  auto-classified from the graph on every schema build, never authored
  (a text override can rename a material mid-session, moving knobs
  between tiers). `seed`: the bound param is in the op's `SEED_PARAMS`.
  `factor`: the bound param is in the op's `UNIFORM_PARAMS` (params
  affecting only the bound material's glTF factor uniforms; texture
  values multiply factors, so the promise holds with textures bound),
  AND the declaration's [min, max] sits inside the channel's valid range
  (`UNIFORM_RANGES` — the factor tier skips evaluation, so the declared
  range is the only validation a scrub gets), AND the bound node's
  material name is unique among the identities the root asset authors
  and absent from the names un-renamed references carry into the root
  (the client applies the uniform by material name across every loaded
  output, so a shared name would let one knob repaint a sibling; a kit
  binding the root's name with identical content passes the §9.1
  uniqueness check, so only the tier keeps an accepted scrub from
  binding one name to two contents), AND every downstream consumer over
  carried inputs keeps `MATERIAL_POLICY` `preserved`/`dropped`.
  `evaluation` otherwise. A `build`-bound knob is always `evaluation`.
  Its apply reruns the recipe (the table below), safe by the language
  guarantee (§4.8). Every doubt lands in the safe tier, so
  misclassification can cost an extra evaluation but never show stale
  geometry. An asset holding any op the registry cannot resolve
  classifies every knob `evaluation`. `factor` entries carry `uniform
  {material, channel}` (`UNIFORM_CHANNELS`). The classification doubles
  as the guide's cheap-knob criterion for `params` selection: factor
  knobs are free, seed knobs one evaluation per reroll, evaluation knobs
  one per apply.
- `set_param {param_id, value}` / `set_param {params: {param_id: value,
  ...}}` — client → runtime, the *only* preview mutation clients may
  perform (structure edits are recipe edits by the agent; the only
  client-side path to the project is `accept`). It addresses the
  immutable `param_id` the recipe declares (§4.8, §4.6), never the
  node. A `null` value unsets the knob: the overlay then holds the
  state without the human's line, which is the recipe's default,
  previewed under the knob's tier contract like any value. A client
  can state a value and cannot state a line that is not there, so
  the unset is a value of the overlay's own. A
  failing override is reverted and reported via `asset_error` to the
  overlaying client (last good state stays up, §8.1). The per-tier
  contract:

  | tier | message | runtime work | reply |
  |---|---|---|---|
  | factor | one coalesced single-key `set_param` per gesture (drag end, not the drag) | updates the overlay and checks the declaration's type and range *without evaluating* (no assertion runs in a preview) — load, prune, overlay | `param_schema` alone: zero evaluations, zero cache writes; a later evaluation-tier apply lands the scrubbed factor values. Live `base_color` alpha approximates via renderer transparency until the next evaluation trues up `alphaMode` |
  | evaluation | the batch form on Apply — N staged knobs, one evaluation | evaluates; a failing batch reverts every batch key (a partial overlay would preview a state the client never staged). A `build`-bound knob in the batch first reruns the recipe in-process with the overlay's values; a `direct` one patches the IR value without a rebuild | `asset_updated` + `param_schema`. The form carries the contract, not the key count: a one-entry batch still evaluates and never takes the factor fast path |
  | seed | one single-key `set_param` per reroll, sent immediately; the pick is a fresh integer uniform in the declared range, chosen client-side (a user gesture, not an op stream — §4.4 is about op determinism, and the value travels fully recorded) | evaluates like an evaluation knob; rerolls are never coalesced — each pick is a distinct proposal with its own reply | the pair, per pick |

  Client-local staged edits, seed histories, and vector send-bases are
  not part of the overlay: they survive a drop, and they are
  asset-scoped. A project switch discards them, since param ids are
  asset-scoped names and a survivor would apply one project's values to
  a colliding id in another. No live path for evaluation knobs at
  release.
- `accept {}` — client → runtime: write this connection's
  preview overlay to the project's `params.json`; `accepted {params,
  dropped?}` back to the accepting client (immediate feedback;
  the durable confirmation is the watch reload that follows and
  reaches every client). Accept
  is the "you decide" landing: the human's pick reaches the project
  without relaying the value through the agent, over the same diff
  and reload path an agent edit takes. It writes exactly the state
  the tab displays. The overlay holds declared knob values and
  nothing else, so structural edits stay impossible by construction,
  and every value has already passed the declaration contract.
  `params.json` is the human's file (§4.2): pure data, one
  `param_id` per line in canonical form, written whole through the
  one project writer, so the diff is the changed lines alone. The
  lines are in id order, so a recipe edit that reorders the
  declarations moves none of them. Accept
  never rewrites the recipe source, and an agent edit never touches
  the human's file. An unset knob's line leaves with the write. A
  line whose id the asset declares no knob for
  draws a warning from the build (`accepted_value_unused`), leaves
  with the next write, and is named under `dropped`, since no edit
  of the agent's may remove it. A server showing a sub-asset alone
  refuses the accept (`accept_not_root`, §4.6), naming the root's
  knob and the reference's `overrides`, and one showing an IR opened
  by path has no project to write into (`accept_no_project`). How an
  accepted value resolves against the recipe's default and a
  consumer's override is §4.8. Staleness is
  closed by the drop rule, not a lock: an overlay that still exists
  to accept was built against the bytes on disk (the window between
  an external write and its watch event is the same accepted race
  class as concurrent agent edits). An empty overlay reaching the
  runtime anyway is refused with the actionable error. After an
  accept the file state *is* the accepted state, so the evidence
  channels show it.
- `accept_validated {project, params, ok, failed}`, or
  `{project, params, checked, error}`, or
  `{project, params, checked, moves_no_check}` —
  runtime → client, after the reload that lands an accept: the
  runtime runs `validate` over the accepted file state of that
  project and sends every client the report's gating summary, the
  failing entries by id. Each row of `failed` holds the
  `entry`, its report `section`, its `owner`, and the `site` an
  authored entry has (a builtin check has none). A
  budget is the human's, a knob's range the agent's, and a check
  every asset is held to is `builtin`. The accept
  event carries the same summary (§6). Validating the saved state
  reports a requirement broken by a pick without adding validation
  cost to each gesture. The check runs apart from the reload, since
  `validate` over a large asset outlasts what a reload may wait. It
  judges the state that reload evaluated, for the accepts that reload
  landed, and a state replaced before its check ended is judged by
  the reload that replaced it. Where the accepted state does not
  build, the message carries `checked: false` with the `error`, and
  the verdict waits for a state that does. The check never gates the
  write. The human's
  pick is a requirement (P4), and a red entry after it says the
  recipe must catch up, not that the pick was refused. A sibling that
  consumes the project through a reference is not re-validated: its
  verdict is its own (§4.6).
  An accept that can move no entry of the report is not validated:
  the reload that lands it answers `checked: false` with
  `moves_no_check`. Its knobs are all `factor`-tier, on channels no
  verdict reads. `validate` reads every factor for finiteness and
  for the count of material identities, and the tier's conditions
  hold both still: the declared range sits inside the channel's, and
  the material's name is the asset's only one. It reads one channel
  for more: a base color's alpha decides whether an output encloses
  another (`not_enclosed`, §4.1), so a color knob is validated. The
  channels a verdict reads are declared beside the checks, and a test
  moves every other channel across its range and finds each entry
  unchanged. The answer says that the pick moves no check, and not
  that the state is green: an asset red before the pick is red after
  it. The question is asked of the state the reload built, the one
  the check would judge, so a recipe edit that lands with the accept
  is taken as it stands. Two accepts are validated all the same: one
  of a knob that fails its own entry in the file state (§6), which
  the new value turns green, and one made while the warning stands,
  which the warning's own check validates anyway.
  The `moves_no_check` answer goes to the accepting client alone, as
  `accepted` did. A verdict, and the `checked: false` of a state
  that does not build, reach every client, since the line they keep
  (below) is shared.
- *The accept warning.* A red `accept_validated` is shown, not merely
  sent, because the human may have accepted without knowing an assert
  existed. Between the write and its verdict the line says the
  accepted knobs are being validated, since silence would read as a
  verdict. Then the viewer keeps one line up: the accepted knobs,
  that validate is red, and that the agent resolves it. What is red
  and
  whose move it is stay out of the line. The fix is a recipe edit,
  the agent's work, settled in conversation in the human's language,
  so an explanation in the viewer is one the human can read and
  cannot act on, and `validate` gives the agent the entries with
  their owner and the knob that moved each (`from_param`, §4.8).
  While the warning stands, each watch reload of that project re-runs
  the same `validate` and updates or clears it, so it is never stale,
  and a green project pays nothing. The runtime holds the warning
  with the project, sends it whole to every client each time and
  replays it to a connecting one, so a tab shows the last message and
  decides nothing. A re-root ends it. The tool
  never undoes on its own. If it did, an agent-owned red would veto
  the human's pick, and a human-owned red is the human's to resolve
  either way (the pick or the requirement is wrong, and only they
  know which). Nor does the warning offer an undo. The pick that was
  the mistake is replaced by the next pick, over the same path, and
  the warning clears with it.
- `revert {}` — client → runtime: drop this connection's overlay and
  show the file again. This is Accept's other half. Without it the human
  can reach a scrubbed state but not the one they are judging it
  against. The answer is the shared file-state pair replayed to this
  connection: no evaluation, no cache write, no broadcast, no new
  revision, the drop announced as a reload's is. An empty overlay is
  refused as for accept.
- `screenshot_request {request_id, protocol, glb_urls, camera, size}` —
  runtime → client — and `screenshot_result {request_id, protocol, png |
  error}` — client → runtime. The dedicated render session:
  the runtime evaluates the file's current state of the project the
  request addresses, or the values `screenshot --param` offers (§6),
  never a client's overlay. The target is a roster name, the
  displayed project by default. A single-project server refuses a
  foreign name (`not_served`). So with several sessions sharing one
  server and one tab, each shoots its own asset and the human's picker
  stays the human's. It resolves the camera server-side and sends the
  GLB urls (`/cache/`, or `/preview/` under `--param`) plus a
  renderer-neutral pose `camera {position, target, up, fov}`
  (cartesian meters, vertical fov in degrees),
  consumable by a rasterizer or a path tracer with no client-side
  framing logic. The client renders those GLBs in a throwaway scene
  (assets + the §7.1 lighting rig, environment map and ground shadow
  catcher included; grid, axes, and pin helpers are viewer chrome, not
  file state) and answers with base64 PNG, or `error` with the reason
  (fetch/parse failures report instead of timing out). `protocol` is a
  version stamp echoed back. A result without the current stamp comes
  from a pre-upgrade tab and is ignored, not resolved, so a current
  client can still answer within the timeout. Only when nobody current
  answers does the failure report `stale_client`. The CLI guards the
  other direction: a 200 response that is not the `{meta, png}` envelope
  reports `stale_server` instead of writing a corrupt file. Camera
  resolution (`camera.py`, the single view-roster source): named views
  are presets over the free camera (azimuth/elevation/distance/target).
  Azimuth 0 looks from +Z (front), 90 from +X (right), elevation 0
  horizontal, +90 top — right-handed +Y-up per §4.3. Auto-fit frames the
  union AABB's bounding sphere (fov 45°, slack 1.35). The free camera
  closes the interior-acceptance gap fixed views cannot.
- `watch_dead {error}` — runtime → client, sticky, replayed on every
  later connect: the file watch itself died (watch root deleted, OS
  watch resources exhausted; per-event reload errors retry on their
  own), so hot reload is off and the display may be stale until the
  watch re-arms. Re-arming is one discipline shared by every watcher and
  the listener: retried with doubling backoff up to a cap, never while
  the root itself is missing. An attempt that dies at birth is the same
  outage (logged, no second message), one that outlives the birth window
  is announced armed, and the backoff resets only once a watch has held
  for a stable period, so a watcher that keeps dying late never flaps.
  Every death and re-arm is logged (server log and session log).
- `watch_alive {}` — runtime → client: the watch re-armed. It clears the
  sticky `watch_dead`, and the one catch-up reload that follows
  re-derives the display from the file for the edits that landed while
  dead. The screenshot report carries the same liveness as `watch:
  "alive" | "dead"` (the shot itself stays fresh, because it
  re-evaluates the file, but a human looking at the tab must not trust
  it as current). `validate`/`stats` carry it under `serve` (§6).
- `listen_dead {error}` / `listen_alive {}` — runtime → client: the
  listener stopped accepting while every connected socket kept streaming
  — a half-dead server. The platform's accept loop closes the listening
  socket on an accept error (a tab navigating away mid-handshake), and
  nothing the server says through its open sockets can notice. Every
  screen and report reads healthy, and only a new tab or command finds
  the port closed. So liveness is checked from outside the sockets, an
  interval apart: the listening socket under the site, and a loopback
  connect made off the event loop (an evaluation blocking the loop
  delays the verdict but can never run its budget out, §6). A failed
  check marks the outage once (logged, session-recorded, told to the
  connected tabs) and re-arms the listener under the watch discipline:
  the dead site dropped, a fresh one bound and probed. Only a site this
  server bound counts: a re-bind refused because another process took
  the port is a death whatever that process would answer (a foreign
  listener is no server, as for the CLI probe). The server therefore
  runs its own site rather than the framework's run loop and installs
  the termination-signal handler that loop provided, so a kill still
  runs the shutdown hooks. `listen_alive` clears the sticky flag. A new
  connection is itself the proof, so a tab clears it on connect and it
  is never replayed. Status carries `listen` and `listen_rearmed`, the
  count of healed outages. That count is the one trace a half-death
  leaves in a report. The screenshot report and the `serve` key (§6)
  carry the count when non-zero.
- `select {action: "pin" | "replace" | "unpin" | "clear", ...}` — client
  → runtime; 3D deixis, the explicit pointing channel (§8.3). `pin` and
  `replace` carry `{output, face, instance?, position}` from the click's
  raycast (the copy index is required on an instanced output — a frame
  is never guessed). `replace` makes the new pin the sole selection as
  one atomic mutation (one broadcast; the store is cleared only after a
  successful resolution, so the gesture cannot race a concurrent pin
  writer into spurious errors). A plain click on a primitive that is
  already pinned keeps that pin, id preserved, and unpins the others
  one by one. `unpin` carries `{id}`. The runtime
  resolves the pin against the file-state evaluation the last reload
  holds (an evidence channel, so no overlay and no build stands
  between the click and the record) into `{id, output, node_id,
  instance, face, groups,
  face_center, face_normal, area, position, bary, source, asset_hash,
  t}`: the face's tag-group membership plus its predicate-vocabulary
  fields, local to the prototype's frame, with the revision it was
  resolved against (framed over the full watched file set, so the hash
  is comparable to the server revision). `bary` (the click's barycentric
  coordinates, clamped into the triangle) lets a client re-derive the
  pointed spot on whatever shape the face currently has, and `position`
  is that point re-expressed on the file-state face. The agent's HTTP
  surface (`/api/pins`, §6) accepts the same face form and a predicate
  form `{output, predicate, instance?}` resolved to the matched faces'
  area-weighted centroid with the selection summary on the record (an
  empty match is the standard `empty_selection` error), and answers for
  the displayed project only (a read of another roster project is
  empty and names the displayed one, and a mutation is refused,
  `not_displayed`). A predicate pin's record carries the `predicate`
  and its selection `summary` in place of the face fields. Failures
  answer the pointing client only, as structured errors. A pin is an
  ephemeral pointing gesture and may carry the raw face index that names
  the hit. Anything durable goes through predicates and tags (P6). The
  record's `groups` and face fields are exactly the vocabulary for
  building that predicate. Lifetime: pins live in server memory for the
  active asset only. A re-root (`set_asset`) drops them, the store is
  capped (oldest evicted first: a gesture stream, not storage), and
  everything dies with the server. A file reload does *not* drop them,
  because the agent's fix must not erase the gesture that asked for it.
  Staleness is derivable from each pin's `asset_hash`. (Persistent,
  versioned annotation is deliberately out: Future Work: AI-native UI.)
- `pins {project, asset_hash, pins}` — runtime → client: the full pin
  state, idempotent (receivers replace, never merge), sent after every
  pin mutation and re-root and replayed on connect when pins exist. The
  replay comes after the state pair, since markers need the asset they
  point into. A predicate pin's record here additionally carries
  `faces`, the matched face indices (omitted past a size cap). It is a
  display-only field on the visual channel (P5). The textual surfaces
  return the record without it.
- `timeline {root, gap_minutes, session, stale?}` — runtime → client:
  the newest session's summary under the server's watch root, derived
  from the session log by the reader `moraine sessions` uses (§6).
  `session` is that report's session object, field for field, so the
  panel and the CLI read the same numbers and the rounds vocabulary
  stays defined once. Full state, idempotent, replayed on connect when
  available, re-sent when the log directory changes. The server watches
  the log directory alongside its asset watch, so CLI verify commands
  refresh the panel without polling, and an unchanged derivation is
  never re-broadcast (day files are machine-shared, so a sibling root's
  events wake the watch with nothing new). Absence means logging is off
  and no panel. The payload carries names, never paths. `stale: true` is
  the frozen-panel flag: the log watch died, or the newest session
  vanished under an existing payload. It is sticky and replayed like
  `watch_dead`, cleared when the log watch re-arms. It is detection and
  report only. The payload is root-scoped and survives a re-root. The
  panel renders the active project's slice (§7.1).
- `set_asset {project}` — client → runtime; re-root the running server
  to a different project without a restart: swap the watched file,
  re-evaluate, answer with `asset_updated`. One running viewer hops
  between projects instead of one server per asset. The server stays
  single-active-asset, and `asset_updated` carries the active project's
  identity so a `select` is unambiguous across a switch. The last
  `asset_error {project, error}` is sticky. A connecting tab receives
  it after the state pair, pins, and timeline, so a tab opened while
  the file is broken sees the failure beside the last good state.
  Contracts:
  *targets are names, never paths*. `project` is a name from the
  workspace roster (the §4.2 discovery sweep). An asset-file-path form
  was rejected because nothing client-chosen may touch the filesystem.
  *The roster is re-read from disk*: every message that resolves a
  name (`set_asset`, a screenshot, a pins call) runs the sweep first,
  and so does the watch when a batch adds or removes a root child (a
  rename is both) or touches an asset file directly under a child the
  roster does not list or deletes one under a child it does, so a
  project created after start-up is served without a restart and the
  picker learns of it unasked. The sweep
  reads the root, never a client string. A sweep the layout rules
  refuse (§4.2) keeps the last roster and is the triggering message's
  error, or one logged line from the watch. A project removed while
  displayed stays on display with its last good state until its next
  reload fails.
  *Failure semantics are asymmetric:* an unresolvable
  target (unknown name, `asset_not_found`, `ambiguous_asset`) swaps
  nothing. A resolvable-but-broken target does re-root with the last
  good state kept on display (§8.1), so fixing its file completes the
  switch via hot reload. `asset_error` carries the active project so the
  picker mirrors the reported project rather than assuming a failed
  switch never moved. *One fixed watch root:* the file watch is one
  fixed recursive root narrowed per-event to the reachable asset set.
  The root is the workspace root, or in single mode the project
  directory, widened to the enclosing workspace root when the served
  project is a workspace member, so sibling files reached by
  cross-project references (§4.6) hot-reload. The watcher never
  restarts, and the startup log names the chosen `watch_root`.
- `time` — reserved name, as an `asset_updated` field and as a client
  message type, for animation (Future Work: Animation). Neither is
  carried until then. The protocol ships no unused field: adding one is
  additive, and the client ignores message types it does not know.

**Security boundary (normative):** the server binds to `127.0.0.1` only.
The protocol is a local, single-user contract, and anything wider is a
separate explicit mode. A hostile web page can still attempt localhost
WebSocket connections, so `set_asset` rests on two preconditions:
WebSocket `Origin` checking, and a target restriction satisfied
structurally by the name-based message (no client-supplied path ever
reaches the filesystem). `Origin` checking compares scheme, host, and
port with the server's own, so a page on another loopback port is
another program's and is refused like a foreign host. A request without
`Origin` is a non-browser client (the CLI, tools). The same gate guards
the screenshot, status, and pins endpoints.

**Known client roster (design targets):** the three.js viewer
(shipped) and **alpine**, the author's CPU path tracer, as an offline
high-quality preview client. A protocol that both a rasterizer and a
path tracer can consume is the litmus test of its design.

---

## 8. AI Integration

**Model agnosticism holds by construction:** the tool never embeds or
calls a model (P2, P4), every agent-facing surface is documented (the
asset format, CLI JSON, the Viewer Protocol, and MCP as the
cross-vendor front door, §8.4), and the token-economy rules of §8.2
are agent-generic, so any agent that can edit text files and run CLI
commands drives Moraine. What does not transfer between models is
calibration: the self/author acceptance gap of §8.3 is per-model
data, re-measured per agent, never assumed.

### 8.1 The core loop

```
Agent edits the recipe (<name>.star; str_replace, minimal diff)
  → runtime detects change, builds, re-evaluates the dirty subgraph
  → viewer hot-reloads (<2 s)
  → Agent reads validate + stats (and a screenshot when a number cannot settle it)
  → Agent verifies, iterates
```

If evaluation fails, the viewer keeps rendering the last good state
and the failure is returned to the agent as a structured, actionable
error. A broken asset is never shown to the human. A build failure
takes the same path, its error naming the recipe file and line
(§4.8). A one-line recipe edit inside a loop changes many nodes. The
reload's graph diff (§6) tells the agent which.

The guide's entry into this loop is **assert-first**, the Concept's
"tests verify" clause as authoring procedure:

1. Write the requirements first. The request's verifiable
   requirements are appended to the requirements file as
   `mo.assert_<kind>` declarations and budgets (§5.2). The agent's
   own invariants and waivers go to the asset. Interpenetration
   needs no entry, because the built-in overlap guard covers every
   pair from the first edit.
2. Build. An assert over an output not yet declared fails the build
   (`unknown_output`): the red state the loop starts in.
3. Declare and build the outputs until `build` and then `validate`
   are green. The finished project carries its own acceptance
   criteria in a file the human can read.

A requirement states intent and comes from the request. A
measurement copied from the result into an assert is a defect, not a
contract. Normative recipe: the guide's authoring-loop section.

### 8.2 Token-economy rules (binding on integrations)

- The tool never initiates agent calls.
- Responses to the agent: JSON stats/validation only; screenshots only on
  explicit request; no unsolicited base64, no mesh dumps.
- A recipe states a repeated shape once, as a function or a loop
  (§4.8), so the agent reads and edits the rule and never its
  expansion.
- Errors and validation reports are *actionable*: they state what
  failed and the shortest path to a fix ("7 materials, budget is 5;
  piece_a and piece_c can share one"), never a bare failure flag.
  Actionable errors shorten the agentic loop, which is itself token
  optimization.
- A finding about an asset the runtime can still evaluate is a
  failing entry beside every other entry, as a knob outside its
  declared range is (§6). An error in place of the report hides every
  unrelated entry behind one finding, and isolating that finding
  costs a round per guess. Errors remain for an asset the runtime
  cannot read or evaluate, and for a value a consumer offers that the
  declaration refuses (§4.6).
- An on-demand reply is one contract: `moraine docs SECTION` answers
  with that section alone and names the sections under it and the
  examples it cites (§6). What an agent reads before its first recipe
  line is mostly the reference, so a reply carrying its neighbors or
  its examples whole would be paid by every reader of one contract.
- Guide residency: know-how consumed only during a specific working
  phase lives in the on-demand channels — `moraine docs`, actionable
  errors, `validate` reports — never in the resident authoring guide.
  Resident lines are reserved for rules that must shape behavior
  before the tool can speak. Every resident sentence is a standing
  token cost in every session.
- Future UI marks operations that consume tokens distinctly from
  local deterministic operations (Future Work: AI-native UI).

### 8.3 The refinement asymmetry

Procedural assets make the *mechanism* of adjustment cheap for an LLM:
a tweak is a one-line diff on a semantic parameter, re-evaluated in
milliseconds. But the *judgment* of fine visual adjustment stays
expensive. The agent perceives through discrete, low-resolution, paid
screenshots, where a human scrubs a slider under continuous feedback.
The agent's own "looks done" threshold cannot serve as the acceptance
gate, because measured self-scores have run optimistic against the
author's, and even a perfectly calibrated agent leaves acceptance a
human role (P4). Only the genuinely spatial errors need pixels at all.
The design treats this asymmetry as structural:

- **Verification is tiered, cheapest first.** Tier 1: deterministic
  validation (zero tokens — intersections, scale, manifoldness are
  the engine's job). Tier 2: numeric stats (cheap — counts, bounds,
  volume). Tier 3: pixels, reserved for the residual class of errors
  only vision catches — low-resolution thumbnails by default, full
  renders only when diagnosing. A claim a tier-1 check has proved is
  not re-verified at tier 3 (measured: a per-output shot set that
  re-shot asserted gaps dominated a revision's pixel cost), and no
  shot is mandated. Pixels are the agent's diagnostic, never the
  acceptance gate. The acceptance gate is the human's (P4), deciding
  on the agent's report, which names the tier each claim was proved at
  and says when nothing was looked at. Investment in operators and
  validators pays twice: it raises the quality ceiling *and* shrinks
  tier-3 usage.
- **Promote recurring visual errors to validators.** When an error
  class is caught by screenshot more than once (parts poking below
  ground), it becomes a deterministic tier-1 check. Each promotion
  permanently converts a pixel loop into a free one.
- **Route continuous taste to the human.** Parameter scrubbing,
  variation grids, and selection are the human's surface (P4). The
  agent owns structure and spec compliance, where correctness is
  numeric. For taste over a stochastic stream the agent-side
  counterpart is the reference's variation-grid recipe over existing
  surfaces: N seed edits, same-camera thumbnails, one presented
  sheet, the pick typed in chat and landed as an ordinary edit,
  reaching the journal through human promotion like any other
  preference (§4.7), never a capture channel. A human at the viewer
  scrubbing a declared reroll knob is already on the cheaper path.
- **Widen the pointing channel, not the describing channel.** 3D
  deixis (the §7.2 pin channel) exists because localizing "this looks
  wrong, here" in language is the costliest part of the loop.

### 8.4 MCP

The CLI/protocol surface is wrapped as an MCP server so any MCP
client can drive Moraine. Claude surfaces first, other MCP-capable
agents by the same server (§8).

**Scope:** the read/verify surface only — `stats`, `validate`,
`query`, `export`, `screenshot`, `docs` as tools over the in-process
runtime. Clients with file access edit recipes directly and
need nothing more. For them the server's value is discovery and a
shell-free verify loop. A write surface (`asset_edit`, `param_set`),
the only editing path for file-access-less clients, is deferred until
such demand is observed. Contracts are built on observed demand, and
the write contract carries real design weight (edit granularity,
enforcing the §4.1 format rules server-side).

**Naming:** registry name `io.github.ms-elk/moraine`; local server
key `moraine`. The name identifies the product itself exposed over
MCP, not a separate bridge package. The install identifier stays
`moraine3d` (§3).

**Contract:** `moraine mcp` serves over stdio (newline-delimited
JSON-RPC, the tools capability only), so a client is configured with
the command `moraine` and the argument `mcp`, the installed tool
(§8.5) and no second entry point. Each tool is the CLI command of the
same name run in-process through the CLI's own parser and handler.
Its arguments are the command's arguments as a JSON object (`query`'s
predicate an object, `screenshot`'s `target` a three-number array),
and its result is the compact JSON the command prints, relayed as
text. So there is one contract and one report shape whichever surface
drove it (§6). A command that ran and failed answers with its own
error object under `isError`. Only the caller's protocol mistakes (an
unknown tool or argument) are JSON-RPC errors. `screenshot` writes
the PNG as the command does and adds it as image content, the one
place the two surfaces differ, since a file-access-less client has no
other way to see the render. Every tool call appends the command's
session-log event marked `via: "mcp"` (§6). The contract's carrier
inside a workspace is `moraine mcp --help`, not the authoring guide,
which says nothing about MCP (the equivalence guards no silent
violation and shapes no plan). The process's stdout is the transport,
so the CLI's printer runs quiet for an in-process run and anything
else a handler prints is diverted to stderr. The protocol subset
(`initialize`, `tools/list`, `tools/call`, `ping`) is implemented
directly rather than through the reference SDK, which would add a
dozen transitive packages to a tool whose dependency list is six (§3).
The SDK's client is the conformance test instead, as a development
dependency.

### 8.5 Onboarding

Beside the CLI, Moraine ships an agent-side onboarding surface: a
skill in the open Agent Skills format (`skills/moraine/SKILL.md`) and
a plugin marketplace manifest under `.claude-plugin/`, so an agent
harness installs the tool and reaches the shipped guide in one
command. The repository root is the plugin. Only the delivery is
harness-specific: competing harnesses read the same skill directory
from their own locations, and a second, harness-shaped copy would be
the drift the single shipped guide exists to prevent (§8.2).

**One install path, one version per machine.** Moraine is a tool run
beside a workspace, never a dependency of it, so it installs once per
machine as a tool (`uv tool install moraine3d`; `uv tool upgrade` to
move), and the README and the skill name that one path and no other.
A `pip install` lands in whatever interpreter is active, and two
`moraine` commands on one machine let a green report be read against
the wrong version. The consequence is accepted: a machine runs one
version, and its workspaces move together. Which version built what
is the record's job (the export manifest, §9; every evaluated report,
§6; journal stamps, §4.7), so an output from an earlier version is
reproduced by running that recorded version once
(`uvx moraine3d@<version>`), not by keeping it installed.

**Scope: setup only,** install and upgrade alike. The skill checks
`moraine --version`, installs or upgrades `moraine3d`, and runs
`moraine init`. When `init` reports `AGENTS.md` as created, the skill
asks the human's working language, in the language the human is
already using and with that language as the default answer, and
translates the fresh `AGENTS.md` into it. That is the one setup step
a command cannot perform (`init` ships English; language and
ownership of `AGENTS.md` are §4.2), and the gate on `created` is what
makes the upgrade path skip it. The skill writes no harness file: the
agent reaches the guide through `AGENTS.md`, on every harness alike
(§4.2), and the skill ends by sending the agent to the guide.
Everything past setup delegates to `.moraine/guide.md`. The skill's
trigger survives a request that is not in English, and its
description says so. The upgrade moment
belongs to the skill because it arrives before any Moraine command
has run, and the re-sync it owes the workspace is exactly `init`
(§4.2). The onboarding files keep no copy of the version: the skill
installs whatever the tool installer resolves, and `init` syncs the
guide to it (§3).

---

## 9. Export Profiles

The IR is target-neutral (glTF-aligned). Export is per-profile
validation + conversion. Conventions conversion (a normal-map G flip)
is always the profile's job, never the IR's.

**A profile is a measured delta, and its name follows what that delta
is specific to.** `gltf` is the container of §4.3 and the checks every
real-time consumer shares. It is specific to nothing narrower, so the
container names it. A delta a *verified import path* is measured to
need takes that target's name: an engine name is the honest name for
an engine's own requirements. What a name may never do is outrun its
delta. A target whose importer performs the conversion itself needs
no profile, however large the engine (Unity (URP) via glTFast, §9.1).
Either kind may carry a target's name: **validation** adds checks
over identical bytes; **conversion** changes the bytes and can leave
the container's specification behind (a −Y normal-map flip is no
longer conforming glTF). That is a choice to ship a non-conforming
file rather than configure the importer.

**The artifact pair.** An export writes `<stem>.glb` +
`<stem>.manifest.json`. Both payloads are built before either write
(a build failure writes nothing). Each write is atomic, but the pair
is not, so the contract makes the gap detectable. The GLB is written
first and the manifest last, as the commit marker, and the manifest
records `glb_hash` (24-hex truncated sha256 of the GLB bytes, the
identity format of §4.5/§6). A manifest whose `glb_hash` does not
match the GLB beside it is an interrupted export, already surfaced
through the error channel, and re-running the export heals it.
Rollback is not attempted, since re-export is deterministic (§4.4),
which is also what makes byte-identical re-export checkable against
the manifest rather than merely claimed. The manifest records the
versions that built the file (§4.4), the build's included.

**Export-time instance baking.** `moraine export --bake-instances`
projects every Instances output to a plain mesh at the export edge
(transforms applied to tiled prototype copies; materials, attributes,
and face groups carried per `bake_instances` semantics). The default
remains keep, because Invariant 6 governs the IR and the default
path. The flag is an explicit edge projection in the class of profile
conversions, never a silent merge. The flag joins the manifest
together with the `bake_instances` op's impl version (byte-identical
re-export holds per flag value). Cache keys are untouched. Validation
and asserts run on the authored values either way, because the
projection changes delivery format, never exportability. So only the
TRS-representability check is skipped (validating baked meshes
instead flipped green flush/enclosure layouts to blocking failures).
Granularity is per export run. Per-output selection is deferred until
demanded. Baking multiplies the prototype's triangles by its copy
count. The export report carries no built-in threshold for that
total, since `stats` already reports the multiplied count and a
budget nobody declared is not checked (§4.1). The flag doubles as the
compatibility path for importers
without `EXT_mesh_gpu_instancing` and as the documented fix the
sheared-instance-transform export error points to. The projection
runs the same evaluate-and-carry step as an in-graph `bake_instances`
node, so the two paths cannot drift. In-graph baking is for semantic
necessity only (boolean cutters, `place` mesh parts, anchors spanning
arrayed copies), delivery format a per-export choice.

**Emissive strength.** glTF caps `emissiveFactor` at 1 per component.
Factors above 1 export normalized to the unit range with the intensity
in `KHR_materials_emissive_strength`, listed in `extensionsUsed` and
never in `extensionsRequired`. An importer without the extension
still reads the color and loses only the strength, and
`EXT_mesh_gpu_instancing` stays the one required extension (dropping
it would lose geometry). The same writer serves the viewer
(`GLB_VERSION` bumped; the vendored loader honors the extension).

### 9.1 The `gltf` profile (first)

- Container: glTF 2.0 (GLB), the §4.3 conventions verbatim. The
  profile converts nothing: its content is validation.
- Verified consumer: Unity (URP) via glTFast. The zero-fix-up claim
  is measured on that import path. glTFast performs at import every
  conversion a Unity profile would otherwise carry, which is why the
  delta is zero.
- Textures: glTF-native metallic-roughness only (glTFast performs the
  URP `MetallicSmoothness` conversion itself, so a sidecar set would
  be dead weight). Roughness/metallic author as separate grayscale
  maps and pack into the metallicRoughnessTexture G/B channels at the
  GLB edge. Base color sRGB-encodes there too (IR images stay
  linear float, §4.3). Tangents are not written. For normal-textured
  primitives without `TANGENT` the glTF spec directs importers to
  generate MikkTSpace tangents (a SHOULD), so the §4.3 convention is
  delegated to the consumer and the in-engine verification pass
  holds the line. Revisit and write tangents if it shows seams. A
  non-glTFast Unity path is what would give a `unity` profile its
  first content: measure that path, then emit MetallicSmoothness
  there, never in `gltf`.
- Preserves part hierarchy and pivots (animation and LOD depend on
  this — never flatten to a single mesh). v1 scope: the hierarchy is
  exactly the named-output level. Each named output is one part,
  parts are never merged, and every part's pivot is the asset origin
  (author assets so the origin is the natural pivot). A richer
  `Assembly` value type is tracked
  (Future Work: Other tracked extensions).
- Preserves instancing (`EXT_mesh_gpu_instancing`).
- Writes the parts in the order of their names, case folded, and the
  materials in the order the parts first use them. A recipe
  refactored into functions and loops declares its outputs in another
  order, and the same content then exports the same bytes, so a
  byte-identical re-export stays the evidence that nothing changed.
  (Writing the declared order and stating it as meaning was rejected:
  every refactor would carry the order by hand.)
- Validation: every output is a mesh, normals present, UVs present
  (advisory on an untextured output, blocking on a textured one),
  instance transforms TRS-representable. These are the
  profile's own checks, some stricter than the specification (a part
  without normals imports flat-shaded: conforming, and almost never
  the intent). Scale sanity is a plain `validate` check, and naming
  determinism is a property of the writer. Geometry and material
  budgets are asset-declared (§4.1), never profile constants.
- **Material identity** is profile-independent and lives in plain
  `validate` (report shape: §6). *Name uniqueness gates:* one name
  with two contents fails `material_name_unique`. Plain export
  already hard-failed on it, so green must not outrun export. *The
  budget counts distinct content identities* (name + factors +
  textures) over what the asset ships (§4.1), plus the neutral
  default once when any output binds no material (it ships as a real
  glTF material). Shared identities
  count once, so a kit's deliberate kit-wide sharing is never
  flagged, and transparent and emissive materials are structurally
  their own identities, which the failure fix names. *The
  material-origin walk* — the carried closure (`CARRIED_INPUTS`, the
  shading walk's rule) over op traits `IS_REFERENCE` /
  `authors_material` (a local `material` node, or an `asset_ref`
  carrying `material_rename`) — does not scope the budget. It routes
  the uniqueness fix (a collision on a reference-borne name names the
  `asset_ref` node and `material_rename`, the consumer's only handle
  on a name authored elsewhere) and scopes the knob panel's authored
  identities (§7.2). *What a material costs:* an SRP-batcher rationale
  was rejected, because Unity's SRP Batcher batches per shader
  variant and keeps per-material constants resident, so material
  count is not its constraint. Each distinct identity costs one
  render-state change and one submesh boundary that
  static/dynamic/GPU-instancing batching cannot cross, plus
  kit-vocabulary sprawl. The budget is that discipline, declared per
  asset by the human who knows the target, never an engine-derived
  constant.

### 9.2 Later profiles

- **Web:** KTX2 texture compression, Draco/meshopt, hard size
  budgets. (Cheap: the first viewer already consumes glTF.)
- **USD:** the interop point with manufacturing pipelines. Future.
- **Unreal:** unmeasured. Its importer may absorb the normal-map G
  flip the way glTFast absorbs the URP conversion, in which case
  there is no profile to write. A measured need (the flip, or
  validation encoding Unreal's own requirements) earns one, carrying
  the engine's name and the non-conformance of §9 with it. Future.

---

## 10. Testing

Two layers realize the Concept's "tests verify": the repository's own
tests below prove the *vocabulary* (operator semantics stay
trustworthy across implementation swaps and platforms). Per-asset
`validate`, checks (§4.1), and asserts (§5.2) prove each one. This
section is the first layer.

- **Golden-mesh tests:** per operator, assert vertex/face counts, bounds,
  volume within tolerance. These pin operator semantics and make
  implementation swaps (numba/Rust) safe. Every shipped example recipe
  builds and validates green.
- **Build tests,** three:
  - *Round trip.* A test-side translator (test infrastructure, never
    a command, §4.1) turns an IR fixture into a flat
    recipe (one call per node, ids kept, definitions ordered before
    their uses) whose build is equivalent to the fixture: the same
    node ids, outputs, groups, params, asserts, checks, and budgets,
    and the same result hash for every output. Bytes are not
    compared, since the build adds `site` and orders nodes by
    definition. Id equality keeps every `USES_NODE_ID` stream where
    it was. The IR corpus proves the builtins cover the IR.
  - *Rebuild.* The same recipe builds byte-identical twice and across
    the CI platforms.
  - *Refusals.* The id derivation, the boundary rule, and the
    human-threshold rule (§4.8, §5.2) each have a refusal test.
- **Determinism tests:** same asset twice → identical hashes on one
  machine. Across the CI runners the cross-platform
  check is the shared golden fixtures (fixed counts/bounds/volumes with
  tolerances): result hashes need not match across OSes, geometry within
  tolerance must (§4.4).
- **Export validation tests:** `gltf` profile output re-parsed and checked
  (hierarchy, instancing, texture assignment, naming stability).

---

## 11. Day-One Invariants

Ten decisions that cost one line now but a migration to retrofit. All are
binding from day one. (Common principle: *stability of references and declaration of
semantics cannot be retrofitted.*)

1. The build's version stamped on the IR, refused on mismatch;
   recipe compatibility by refusal naming the fix (§4.1, §4.4, §4.8).
2. Deterministic node ids derived from the recipe's names; all
   references by id (§4.1, §4.8).
3. Units & conventions block, declared normatively (§4.3).
4. Determinism: explicit per-node seeds, no global RNG (§4.4).
5. `op_impl_version` inside every cache key (§4.5).
6. Instancing represented as instancing, end to end (§4.1, §9.1).
7. Named multiple outputs per graph; deterministic export naming (§4.1).
8. Part hierarchy + pivots preserved through export (§9.1; v1 scope: flat
   named-output parts, pivot = asset origin).
9. Reserved name `time`, in the schema and the Viewer Protocol (§4.1,
   §7.2).
10. Recipes compute numbers, operators compute geometry: a recipe
    never reads an evaluated value (§4.8).

Invariants 8–9 exist to keep animation (Future Work: Animation) a
cheap later addition rather than a migration. The name is the
reservation. The field lands with the first operator that consumes
it, as an additive change.

---

## Future Work

Items here are directions, not landed contracts. They are cited by
name — `Future Work: <item>` — never by number: names survive the
reordering and deletion that are this chapter's lifecycle. When an
item lands, its spec moves to the owning chapter and the item is
deleted. What stays in an item is what remains unbuilt and the
constraints already decided for it.

### AI-native UI

Guiding principle: *the UI's job is to raise the bandwidth of
human→agent communication (point and choose instead of describe),
lower the cost of agent→human verification (diffs and comparisons
at a glance), and keep deterministic interactions token-free.*
Committed: P5 and the shipped surfaces these build on — the §7.2 pin
channel and screenshots. Candidate directions, roughly by expected
value:

1. **Rich deixis** — persistent, versioned annotations and
   bidirectional markup beyond the shipped ephemeral pointing gesture
   (§7.2).
2. **Knob focus** — the agent directs attention *within* the declared
   knob set: a pin-channel verb naming param ids, the viewer floating
   and accenting the named knobs, so a round's decision arrives
   framed ("these two decide the silhouette — scrub and accept").
   Deixis pointed at parameters instead of geometry. It is the
   session-ephemeral counterpart of custom inspectors below, and the
   next answer to panel density after the tier grouping (§7.1). Cheap
   by construction — it rides the pin infrastructure.
3. **Variation grids** — contact-sheet UI over N seeds. The UI-less
   recipe ships in the reference (`moraine docs variation-grid`,
   §8.3). What stays here is the grid UI itself, its
   preference-capture rationale closed by the journal's
   automation-degree decision (§4.7).
4. **Screenshot annotation** — scribble on a render, send as an image
   to the agent. The red-pen review metaphor.
5. **Token-cost visibility** — UI visually distinguishes local
   deterministic commands from agent-bound ones.
6. **Before/after visual diff** — any single change (a parameter
   scrub, a recipe edit) reviewable as an A/B screenshot toggle. It
   is the visual analog of reviewing a code diff, and the
   human-facing proof of non-destructive editing. Blind authoring
   produces spatial-arithmetic errors at a steady rate and quality
   rises sharply under visual-feedback iteration, so this attacks
   the highest-leverage variable in that quality model.
7. **Agent-authored custom inspectors** — a per-generator inspector
   definition (floor/bay slider groups for a building generator),
   stored as data beside the asset and served deterministically
   thereafter. Authoring cost amortizes as generators do
   (Future Work: Token economy), and regeneration rides the
   structural edits the agent is already making.

### Animation

- **In scope, first:** mechanical animation — time-driven node TRS
  sampled and baked to glTF animation clips (fans, doors, cranes).
  Rides on Day-One Invariants 8–9 (§11): `time` lands as a recipe
  declaration that builds to an optional root key with a default,
  an additive change every rebuild absorbs (§4.1).
- **One clock per composite.** The root asset owns `time`
  and it propagates into referenced assets over the override plumbing
  of §4.6, so a composite animates as one timeline and a sub-asset's
  own clock is ignored when referenced (the USD single-stage-timeline
  and Houdini global-playbar precedents converge here). Per-use
  retiming, if ever needed, is optional `time_offset` / `time_scale`
  params on `asset_ref`, a value declared at the reference site, never
  a second clock. `time` joins the cache key of exactly the nodes
  whose operators consume it, so record aliasing (§4.6) stays valid
  for time-independent sub-assets.
- **In scope, second:** vertex animation textures (VAT) — procedural
  motion baked to textures, reusing the texture-node + cache
  infrastructure. Playback is a shader concern.
- **Out of scope:** skeletal animation (bones + skinning). Skinning
  quality depends on hand-authored weights (the craft Moraine exists
  to avoid), and rigid-part hierarchies already cover mechanical
  assets. glTF skinning can be added later without schema breakage.
- **Open — a runtime expression for `time`-driven values,** decided
  when animation is scheduled. Build-time expressions exist (the
  recipe, §4.8), and the recipe cannot see `time`, which is an
  evaluation-time input by definition. Candidates: **none** (the
  recipe emits sampled values or per-key parameters), a runtime
  expression over `time` alone. Evaluate "none" first, against real
  animation authoring, since a runtime expression language is the
  on-ramp to re-implementing a DCC's expression layer inside
  evaluation.

### Token economy

Token cost is measured, never assumed, by the instruments §6, §7.2,
and §8.3 ship, and optimization is prioritized against the authoring
benchmark. What stays future:

1. **Representation efficiency at scale** — a graph outline on
   request (ids, op names, connectivity only) for the expansion a
   recipe hides, and the remaining **asset queries** — dependency
   and spatial questions ("which nodes contribute to this bbox",
   "what is downstream of parameter X") over the graph and its
   evaluated records, token cost scaling with the answer. The
   face-predicate form is the shipped precedent (§5.1 vocabulary,
   one expression readable at query time and fixable into the
   recipe).
2. **Reuse as amortization** — authoring tokens amortize across
   every asset a shared generator later produces
   (Future Work: Library & commons).
3. **Recipe refactoring with equivalence proof** — the agent
   refactors a grown recipe as it refactors any code (extract a
   function from repeated calls, simplify, drop dead nodes), and a
   refactor is accepted only if the evaluated outputs' content
   hashes are unchanged. Determinism (§4.4) and content-addressed
   records (§4.5) make "provably output-identical" a one-bit check.
   One honest edge: ids derive from the recipe's names, and ops
   declaring `USES_NODE_ID` make an id change observable in their
   output, so a refactor either keeps those names or the check
   legitimately fails. Never special-case it.

### Generative nodes

Sculptural form stays outside the operator set (§1.3) and arrives
through dedicated generative 3D models (image-to-3D, text-to-3D),
integrated as **non-deterministic generator nodes** once their
topology is production-grade (deformable edge flow, clean quads,
intentional UVs). A generative node emits a base mesh,
and downstream procedural nodes handle refinement, arraying, and
variation. The agent's role stays directorial: it commissions the
generator rather than sculpting through operators. The
texture-node + cache pattern (§4.5) is the template, with one
addition. A generator's output breaks
determinism (§4.4), so it is **frozen** into `external/` as a
non-regenerable, provenance-recorded input (the day-one receptacle of
§4.2), accepted by the human like any imported asset. The boundary is
normative: everything Moraine claims — determinism, verification,
reviewability — attaches to the recipe around the generated mesh,
never to the generator's output, which keeps the positioning of §1.2
intact.

### Journal mechanics

The journal itself is shipped (§4.7). What stays future:

- **Split into topic files:** if a journal outgrows one file, a
  deterministic move: each `### <slug>` body to `journal/<topic>.md`
  by its Entries sub-heading, pointers rewritten to the sibling-file
  form the shipped grammar already accepts, the Details section
  removed. It is reviewed by the human as a diff. No model is in the
  loop.
- **Consolidation pass:** if recurrence is shown to go unnoticed in
  practice, an on-request pass, never a scheduled job (P2 — a
  standing LLM cost needs standing evidence). Input is the
  journals alone (the workspace holds no session logs — §6 — so
  workspace completeness makes journals the only record to mine).
  Output is proposed edits that cite instances: new generalizations
  only, stated observations only, corrections replacing what they
  correct. The human reviews them as an ordinary git diff. Nothing
  promotes automatically (§4.7).

### Library & commons

A shared layer where proven generators, subgraph recipes, and
sub-assets accumulate across a user's workspaces, and beyond it a
commons of recipes circulating between users. The shared unit is a
recipe, never a baked asset. Browsing is inspiration for the human's
P4 role, and adoption is fork-and-adapt: the agent reads the shared
recipe, not the render, and reconstructs the approach to the user's
own spec. Two guards keep it a commons rather than the asset store
§1.2 critiques. Trust: a recipe is hermetic Starlark plus
`external/`, evaluated by Moraine's own builtins and operators, so
consuming one is never a code-execution decision (§4.8). Framing:
fork-and-remix, never download-and-ship. No central infrastructure:
it starts git-native, shared repositories plus provenance pinning.

Until reuse is seen to want to cross workspaces at all, it is the
§4.6 reference inside a workspace and vendoring across them.
Promotion into the library is a curation decision, never automatic,
under three constraints:

- A library reference keeps the `asset_ref` contract and changes only
  where the file comes from. Cache keys stay content-addressed and
  path-free, so vendored and library-served copies of one content
  share entries, and the provenance a vendored copy records (§4.6)
  becomes the pinned reference by a field move.
- Fork-on-write runs both ways, vendor to reference and reference to
  vendor, so a project can always take local ownership.
- An entry is a parameterized asset (§4.6 overrides), and the catalog
  lists its interface: a visual gallery for the human, the recipe
  plus a one-line entry for the agent (P5).

Short of the library, two items stand on conditions of their own:

- **`moraine vendor`.** The command form of the §4.6 copy rules, for
  when vendoring by hand is seen to go wrong: given a source recipe
  or project, it chooses the form by the source's dependencies,
  copies the closure, and refuses a name that collides with an
  existing project or sub-asset, offering the rename.
- **Workspace-level definitions.** A `lib/` at the workspace root,
  loadable by every project, would share helpers without vendoring.
  It stays closed, because a project's interface is its outputs and
  params (§4.6) and a shared definition file couples projects on
  internals, until helper duplication across sibling projects costs
  more than that coupling would.

### Kit generators

A marker that promotes a recipe function to a sub-asset generator:
each call would build its own IR under `build/subassets/`, the
call's arguments becoming overrides on the generated asset's
declared params, with an `asset_ref` in the caller. Not built: the
plain function and the sub-asset file cover the observed cases
(§4.6), and a marker would let the commonest helper become a file by
accident, until rewrites show the same helper turned into a
sub-asset file by hand more than occasionally.

### Lights

Light placement is in scope — lamp posts, machine indicators, and
modular kits with lighting are engine-asset territory.

- **Scope: punctual lights only** — directional / point / spot, the
  set of `KHR_lights_punctual` (ratified; supported by three.js,
  glTFast, Blender, and Unreal's importer). Area lights are excluded
  (never ratified). IES profiles are tracked but not initial scope.
- **Physical units, per the glTF spec:** candela for point/spot, lux
  for directional. This joins the §4.3 conventions block when
  implemented.
- **First non-mesh value type.** A `light` operator emits a Light
  value that flows through `transform` and `array` like any other
  value. `validate`/`stats` gain type dispatch (light counts,
  intensity sanity) rather than assuming Mesh. This is the precedent
  for further non-geometry values (curves, cameras).
- **Export expands light arrays to individual glTF light nodes.**
  `EXT_mesh_gpu_instancing` is mesh-only, so Invariant 6 governs
  meshes. Light instancing is preserved in the IR and expanded only
  at the export boundary.
- **Viewer:** three.js punctual lights map 1:1. Asset lights suppress
  (or toggle against) the viewer's rig, so an asset that authors its
  own lighting is previewable as it will look in-engine.
- **Pairing convention:** a lamp asset is an emissive material (looks
  lit) plus a light node (casts light). The material half exists.

### Scenes

The word *scene* is held for the engine's meaning: an asset whose
outputs place other assets, with lights (Future Work: Lights) and
later cameras. A scene is a kind of asset, never a second unit, so
one root asset per project (§4.2) holds. What it adds is
verification in world space (clearance between placed assets, a
walkable path, light coverage) and an export that keeps each placed
asset's identity (§9).

### Operator extensions

Extensions beyond the shipped operator set: operators, the value types and
assert kinds they need, the constraints already decided, and what a
workspace would define. They are collected from authoring evidence
(the benchmark and field builds) and domain-coverage gaps.
None are commitments. An operator the tool ships is admitted under
the §5 rubric and ships what §5 requires (semantics paragraph,
golden-mesh test, example asset).

**Decided constraints on the shipped set:** an image-combine op
(`mix`/`multiply`) was deliberately deferred. `streaks` owns a slot
outright or is colorized alone. `streaks`'s fixed
vertical direction keeps a `direction` param as additive headroom.

**Next additions (evidence-backed, high confidence):**

- `mirror` — symmetry as a named operation. `transform` already keeps
  a mesh's faces outward under a negative-determinant scale and refuses
  one on an instance set, so the operator adds the instance case and
  the plane as the stated intent.

**AI-native group (the P6 payoff):** operator classes uncommon as a
*unified authoring contract* in human-first DCCs. Pieces exist there
(Blender's Geometry Nodes selection fields are predicate-shaped) but
not as the agent's standard addressing with one vocabulary shared
between query and edit, ambiguity handled mechanically, and results
wired into validation. Candidates beyond the shipped core (§5.1,
§5.2):

- Anchored-placement extensions — further modes (*flush with*,
  *centered on face*).
- Shared placement across outputs — two outputs that must land on
  the same scattered positions (a crown and a trunk per tree) need
  one sampling shared by both placements, and a scatter belongs to
  one output: a helper called twice scatters twice. The fix is an
  operator's, not the recipe's (a recipe cannot read positions,
  §4.8): a scatter that emits frames consumed by several placements,
  so one sampling places every part.
- `distribute` — declared count / edge-margin / mutual-spacing layout
  inside a tagged face. It turns a knob that
  needs guarding asserts into one the op guarantees by construction.
- Further assert kinds — prioritized by kit friction. Symmetry is
  demoted for lack of demand.
  Axis-restricted clearance is pooled. Each promotion converts design
  intent held in a modeler's head into tier-1 verification (§8.3).

**Curve-typed group (design together, not piecemeal):**

- A 2D profile / curve value type — the second non-mesh value type.
  The Light precedent (Future Work: Lights) applies. `revolve`'s
  inline `profile` retrofits to it when it lands (§5).
- `sweep` — profile along a polyline/helix: pipes, railings,
  moldings, threads, springs. A polyline sweep is numpy-native and is
  the likelier route to the edge quality `bevel`'s `segments > 1`
  defers (§5).
- `wireframe` — mesh edges → beams/pipes (trusses, bracing, cages,
  railings). Expressible over `sweep`.

**Engine-readiness group:**

- `decimate` / LOD chain — §1.1 keeps the part structure LODs attach
  to, but no planned operator produces them. `fast-simplification`
  ships the required wheels (§3).
- `text3d` — signage, unit numbers, engraved part marks (freetype-
  and earcut-class kernels with wheels exist).
- `shell` / `thicken` — containers, sheet metal, pipe walls. A
  simple normal-offset first, a level-set offset kernel later as a
  further `mode` (§5; also the tracked route to a multi-segment
  `bevel`; manifold3d ships
  level-set extraction, but its binding evaluates the field per
  point, so adoption needs the field evaluation kernel-side).
- `set_color` / color-by-tag — vertex colors (`COLOR_0`, the §4.1
  attribute model) as tint and wear masks. Unity URP's stock Lit
  shader ignores vertex color, and a factor-only `material` node is
  the cheaper answer for telling kit parts apart.
  The likely first real need is AI-facing — painting a `tag`'s
  selection into vertex color so a screenshot shows whether a
  predicate selected what was intended.
- Placement helpers (`ground` / align / stack) — dormant. The §4.1
  ground checks catch the whole error class for free, and the
  evidence points at asserts and anchored placement, not more
  validators.

**Workspace-defined operators:** an operator a workspace holds and
the agent implements, for a shape the shipped set cannot build.
Undesigned. Two reservations keep its arrival additive: the
directory name `custom` (§4.2) and the `mo` namespace (§4.8).

**Workspace-defined measures:** a measurement a workspace holds and
the agent implements, for a requirement the shipped assert kinds
cannot state. Undesigned, under the same two reservations. One
constraint is fixed: a workspace defines how a value is measured and
never how it is judged. The comparison, its tolerance, the entry's
owner, and its row in the report stay the tool's, which is what
keeps it apart from the check script §5.2 rejects.

**Explicitly not candidates:** subdivision surfaces, lattice/cage
deformation, sculpt-style operators — the on-ramp to sculptural
modeling. The §1.3 scope guard applies and the planned route is
Future Work: Generative nodes. A first-class SDF representation (an
SDF value type with primitive/blend nodes) is likewise out, should it
be re-proposed: level-set meshing emits dense unstructured triangles
(against the §1.1 real-time promise), faces of a fused field cannot
be attributed to source nodes (breaks §5.1 tag/provenance
addressing), a second modeling vocabulary doubles the agent's error
surface (P3), and exposed smooth blends are themselves an on-ramp to
sculptural form. SDF enters only as an internal kernel behind
existing mesh-in/mesh-out signatures, where provenance stays "this
node's output" and the recipe never changes.

### Performance & scale

Scale is *unverified*, not *verified-good*: nothing measured on
small assets says how hundreds of nodes, deep `asset_ref` chains,
dense boolean stacks, or a texture-inflated cache behave.
Optimization targets come from the shipped profiling data (the
`profile` key, §6), never intuition. Tracked work, in dependency
order, all pending profile data:

1. **Scale benchmark.** A fixed synthetic stress suite alongside the
   authoring benchmark (generated assets at 10²–10³ nodes; deep
   reference chains; high-poly boolean stacks; 10⁴–10⁵ instances),
   recorded append-only so runs stay comparable.
2. **Evaluation speedup, in escalation order:** (a) per-operator
   kernel rewrites behind stable signatures (the §3 escape hatch);
   (b) parallel evaluation of independent subgraphs — determinism
   (§4.4) and content-addressed records (§4.5) make this safe by
   construction; (c) only then anything architectural. No speedup
   work without a profile showing the §3 budget actually breached.
3. **Cache directory sharding**, when record counts call for it
   (records are already zlib-compressed npz).

### Other tracked extensions

- An `Assembly`/`Part` value type (nested parts, local pivots, stable
  part ids) if kit assembly outgrows flat named outputs (§9.1).
- `alpine` path-traced client (§7.2). A final-quality look at a
  finished asset feeds the human's acceptance (P4). It is a beauty
  check, not a verification tier (§8.3 is untouched). Guards: it is
  just a Viewer Protocol client, swappable by construction; the web
  viewer remains the reference client; no Moraine feature may ever
  require it.
- Source assets in `external/` (reference images composited
  into textures; imported meshes as graph inputs).
- CI verification demo: the published sample workspace running
  `moraine validate` and a byte-identical re-export check in GitHub
  Actions — the recipe's software toolchain (§1.2) shown as a green
  check per pull request. Cheap by construction.

### Beyond 3D asset authoring

**3D as a human-agent interface:** Structured 3D can serve as a
shared communication space between humans and agents, enabling
pointing, demonstration, and explanation in both directions as part
of a dialogue. From this perspective, Moraine is at its core a
runtime that lets an agent produce structured 3D efficiently and
precisely, with asset authoring as its first use. Potential
applications include interactive explanation, guided inspection, and
education, among other forms of spatial human-agent interaction.
