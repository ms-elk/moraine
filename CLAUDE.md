# CLAUDE.md — Moraine

AI-native procedural modeling tool for real-time 3D assets. The Concept:
**Moraine lets you build, refine, and extend 3D assets together with an
AI agent — the agent writes the recipe, tests verify your declared
requirements, you decide what they become.**

**This file instructs the developer role only.** When authoring assets —
in a user's Moraine project, or in-repo under
`src/moraine3d/docs/examples/` or for the
benchmark — follow the modeling-agent guide instead
(`src/moraine3d/docs/guide.md`; `moraine init` ships it to workspaces as
`.moraine/guide.md`).

Read `DESIGN.md` before non-trivial work — the Concept heads it;
section references below point there. DESIGN carries no schedule.

Positioning (§1.2) is binding wherever the product speaks. The story
is an asset that grows by requests: what was built is a part to
consume, what was accepted is a requirement that still holds, what
the human points at is the target. The tool, not the agent, carries
the assembly and the proof, the human owns intent and acceptance, and
determinism and version control are the floor.

## Project facts

- Python 3.12. Import & PyPI name `moraine3d` (one identifier, §3);
  CLI stays `moraine`. Repo `ms-elk/moraine`. Windows + macOS.
- Kernels: numpy / trimesh / manifold3d (§3). Every new dependency
  MUST ship Windows+macOS binary wheels on PyPI (§3).
- Asset = a Starlark recipe, `<name>.star`, the source of truth and
  the file the agent edits (§4.8; one root asset per project
  directory, sub-assets under `subassets/` via `asset_ref`, §4.6).
  `moraine build` expands it to the IR, a JSON operator graph in
  `build/<name>.mora` (one node per line, fixed key order, ids from
  the recipe's names), which everything past the build reads
  and nothing edits (§4).
- Workspace model (§4.2): one git repository = one workspace
  of many projects; the project stays the unit of evaluation and
  export. References stay workspace-internal (§4.6); project
  journals per §4.7.
- Non-regenerable inputs live in `external/`, committed; the
  evaluation cache is regenerable (§4.2, §4.5).

## Conventions (normative — §4.3)

Meters · right-handed · +Y up · +Z forward (asset front faces +Z) ·
CCW winding · UVs v-down, (0, 0) at the texture's top-left ·
MikkTSpace tangents · normal maps +Y ("OpenGL", per glTF spec) · albedo sRGB, data maps linear · glTF 2.0 metallic-roughness.
Target-specific conversions happen ONLY in export
profiles, never in the IR, and a profile exists only where a delta is
measured (§9). When a convention is chosen or revisited, default to the
surrounding ecosystem's standard (glTF, Unity) over a Moraine-local choice
converted at the edges (the +Z-forward flip is the precedent).

## Implementation rules

1. **Array-oriented only.** No per-vertex / per-pixel Python loops. Vectorize
   with numpy or push into a kernel. Hot operators may move to numba/PyO3
   behind their existing signature — never change the signature for speed.
2. **Determinism.** No global RNG. Every stochastic op takes explicit `seed`;
   derive per-node streams as `hash(seed, node_id, index)`, or read a
   seed-keyed lattice for shareable noise fields (§4.4).
3. **Cache keys** include `op_impl_version`. Bump it on ANY output-semantics
   change, including bug fixes. An additive param whose default path is
   bit-identical to the pre-param output is not such a change — no bump,
   but pin the bit-identity with a frozen result-hash test (§4.5).
   An op whose streams are salted with the node
   id (rule 2) declares `USES_NODE_ID = True` so the node id joins its key
   (§4.5). An op that calls a kernel declares it in `KERNELS`; a
   shared kernel module's `KERNEL_VERSION` is bumped like
   `op_impl_version`.
4. **Instancing stays instancing** — scatter/array emit prototype+transforms,
   through to export (`EXT_mesh_gpu_instancing`). Never silently merge.
5. **Day-One Invariants (§11) are binding.** If a change
   would violate one, stop and surface it instead of working around it.
6. **Operator semantics are the contract.** Each op gets a one-paragraph
   semantics doc + golden-mesh test (counts, bounds, volume tolerance) before
   its implementation is considered done; an assert kind is the same
   contract in two files, `core/asserts/<kind>.py` (shape, defaults)
   and `verify/asserts/<kind>.py` (evaluation), registered at each
   package root (§3). Every op ships a minimal example asset
   under `src/moraine3d/docs/examples/` (§5; candidates are
   admitted under its rubric). Test docstrings
   cite the claim they pin (the semantics sentence or DESIGN section), not
   the mechanism.
7. **Extending an operator never changes what it returns.** A kernel
   that does more (a multi-segment `bevel`, a packed atlas for
   `unwrap`) accepts a value the operator refused or adds a `mode`;
   the same params keep returning the same result (§5).
8. **Scope guard.** Rule-describable structure only (§1.3):
   mechanical parts, furniture, architecture, modular kits, and the
   natural forms procedural rules generate (terrain, outcrops,
   foliage, scatter fields) in scope; sculptural form — quality in
   vertex-level nuance, no nameable degree of freedom — out.
   Do not add sculpt-leaning features, however helpful they seem — the
   planned path for sculptural form is generative nodes, not operators
   (Future Work: Generative nodes). If a request appears to
   need sculptural capability,
   say so and stop.
9. **Deletion safety.** `external/` and the cache are the two sides
   of one line; never blur it (§4.2). Every unlink goes through the
   one audited choke point, and Moraine stays structurally unable to
   delete anything it did not itself write — read §4.5 before
   touching any deletion path.

## Document rules

1. **DESIGN.md is the single source of truth for Moraine's design.** A
   decision that changes or settles design lands there with the
   implementation, at the section it affects, as normative text — no
   dates, phase or batch names, or finding numbers (that provenance
   lives in commit messages). Record a rejected alternative only if
   it could be re-proposed, and then as the rationale against it.
   Code, commits, and conversation never outrun DESIGN.
2. **DESIGN Future Work is never cited from code, tests, or shipped
    docs** — a citable item has landed: move its spec to the owning
    chapter first, then cite the new home. Section numbers in §1–§11
    are stable anchors — never renumbered; content moves, numbers
    don't. Future Work is unnumbered and cited by name
    (`Future Work: <item>`) under the conventions its own chapter
    intro states; contracts move to the owning chapter the moment
    they are decided — normative-sounding detail accreting in Future
    Work is the signal. `tests/test_design_refs.py` enforces the
    citation ban, heading existence, and item-name resolution.
3. **DESIGN names no other project document.** It carries no
    schedule, record, or evidence pointers, so nothing downstream of
    it can force an edit to it. This file and the README are the entry
    points and may name anything in the repository.
4. **The agent guide grows by trigger lines, not prose.** A new
    feature ships its agent documentation as a `moraine docs` section
    plus at most one guide line — a routing-table row or a
    when-to-read pointer naming the moment it applies. Prose
    residence in `src/moraine3d/docs/guide.md` requires the guide
    preamble's admission test (silent violation / plan-shaping /
    needed every session), with the passing criterion named in the
    commit message. The guide's declared line budget is test-enforced
    (`tests/test_design_refs.py`): an addition displaces a line,
    moves to `moraine docs` or the error channel, or raises the
    budget as a deliberate, reviewable diff. The budget bounds how
    many rules compete for the agent's attention, not tokens: what an
    agent reads before its first recipe line is mostly `moraine
    docs`, so a line leaves by the admission test, never to make
    room. When a rule's moment is a command, prefer the strongest
    carrier — that command's own output or error text — over any
    document.

## Token economy — product design rules (§2 P2, §8.2)

These shape what you *build*, not just how you behave; the agent-side
counterparts live in `src/moraine3d/docs/guide.md`.

- CLI/tool output: terse machine-readable JSON. Stats and validation reports,
  never mesh dumps, never unsolicited images. The one exception is a
  page of the shipped reference (`moraine docs SECTION`), printed as
  its own text (§6).
- Errors are actionable: state what failed and the shortest path to a fix,
  never a bare failure flag.
- On evaluation failure, the viewer keeps the last good state; report the
  failure through the error channel, never by breaking the displayed asset.
- Screenshots only on explicit request.

## Commands

```
moraine --version               # terse JSON; the version journal stamps carry (§4.7)
moraine serve [PROJECT]         # server + viewer + file watch (workspace root: all projects, viewer picker)
moraine init [DIR]              # scaffold a workspace (AGENTS.md once, guide synced from the package; journal retirement pass, §4.7)
moraine build [PROJECT]         # recipe -> IR under build/; every evaluating command builds first (§4.8/§6)
moraine eval / stats / validate / query / export / screenshot / docs   # see §6
moraine mcp                     # MCP server over stdio: the read/verify commands as tools (§8.4)
moraine sessions [PATH]         # session-log summary: sessions, per-project rounds, command counts (§6)
moraine pins list|add|clear     # deixis pins on the running server (§7.2/§6)
moraine cache stats|clear|evict # size (preview bucket separate) / full clear (bucket too) / run the cap now (§4.5)
pytest                          # golden-mesh + determinism tests
```

`PROJECT` = a project dir (default: cwd) or an asset-file path (an
IR path is evaluated as it is, with no build);
`stats`/`validate`/`serve` also take a workspace root (§6).
