# Selection, anchoring & assertions

## Selection, anchoring & assertions

One vocabulary, designed together: the same face predicate an agent
uses to *read* an asset (`moraine query`) is the one it *fixes into the
graph* (`tag`), references from placement (`place`, `mate`), and guards
with asset assertions. Priors transfer across all of them.

### Face predicates (shared vocabulary)

A predicate is a JSON object mapping per-face fields to comparator
objects; all comparisons AND together (across fields and within one).
Example: `{"normal_y": {"gt": 0.9}, "center_y": {"ge": 1.0}}`.

- Fields: `normal_x/y/z` (unit face normal; a degenerate face has a
  zero normal and never matches a normal test), `center_x/y/z` (face
  centroid, meters), `area` (m^2).
- Comparators: `gt`, `ge`, `lt`, `le`, and `eq` with optional `tol`
  (default 1e-6): |value - eq| <= tol.
- Cardinality (`expect` on consuming ops): `one_or_more` (default;
  an empty selection is `empty_selection`), `exactly_one` (empty is
  `empty_selection`, more than one `ambiguous_selection`), `any`
  (an empty selection is valid. A consumer for which it is
  meaningless refuses it itself, e.g. `empty_group` on a frame
  lookup). A selection outside its declared cardinality is a
  structured error, never a silent pick.
  Predicates address **triangles**, not logical faces: a box
  face is 2 triangles, so its top matches 2 entries. `exactly_one`
  suits a genuinely single-triangle selection, not a quad face.
- Determinism: fields derive from evaluated geometry; selections are
  ascending face-index order. Thresholds compare exactly, so choose
  robust values (`{"gt": 0.9}`, not `{"gt": 0.99999}`).
- Frame on instanced geometry: `tag` on an Instances input and
  `moraine query` on an instanced output evaluate the fields on the
  **prototype in its local frame**. The copies are transforms applied
  afterwards, so a rotated or translated copy's world normal/center
  differs. Select the face on the prototype, then address the copy by
  index (`place`/`mate` `instance` / `face_instance`, the `mated` copy
  index), never by a world-space predicate.

Selections materialize as **face groups** on a Mesh: name -> sorted
unique face indices. Groups ride `result_hash`, the cache record, and
`stats` (`"groups": {name: face_count}`, numerically verifiable), but
never the exported GLB (authoring-side addressing, not engine data).
Propagation follows `ATTRIBUTE_POLICY`: topology-preserving ops carry
groups through; rebuilds drop them with a `groups_dropped` warning in
`validate` (re-tag after the rebuild). `bake_instances` tiles groups
per copy; `array`/`scatter` carry them on the prototype.

### tag (op_impl_version 2)

Recipe: `mo.tag(a, id = ..., name = ..., select = ..., expect = ...)`

Fixes a predicate selection into the graph as a named face group on its
single geometry input, the write side of predicate addressing. An
instanced input is tagged on its prototype (tags ride the prototype;
the copies pass through untouched; the predicate reads the prototype's
local frame, see *Face predicates*), so a received instanced output can
be tagged at the point of use and one copy's faces addressed through
`place`/`mate` `instance` / `face_instance` and `mated` copy indices. Geometry,
attributes, and material pass through unchanged; the output gains
`face_groups[name]` = the matched faces, re-evaluated on every upstream
change (the selection survives edits the way an index list cannot), and
the name documents intent in the graph itself. Params: `name` (required,
`[A-Za-z][A-Za-z0-9_-]*`; colliding with an existing group errors,
because overwriting would hide a stale tag), `select` (required
predicate), `expect` (cardinality, default `one_or_more`).
Deterministic and pure.
Example: `examples/shelf_pair`.

### place (op_impl_version 2)

Recipe: `mo.place(base, part, id = ..., anchor = ..., mode = ..., offset = ..., instance = int)`

Anchored placement: positions its part input by semantic relation to a
tagged anchor instead of agent-derived coordinates. Inputs: `base`
(mesh carrying the tag), then `part` (mesh). The output is the
**translated part only**. The base flows on through its own edge, so
part separation and named outputs are preserved. Anchor frame
(canonical): origin = area-weighted centroid of the anchor group's
faces (where the anchor is an output, `moraine query --tag <tag>`
reads it as `centroid`, with `--instance <i>` for an arrayed anchor's
copy, so an `offset` to a designed position is that position minus
it); resting
plane = max vertex y over those faces; v1 anchors
translate only. The group must form a single horizontal surface: a
vertex-y spread above 1e-4 m errors (`anchor_not_coplanar`). A
mixed-height anchor would put the centroid between surfaces and rest
the part over empty space; tag one surface per anchor.
Params: `anchor` (required; unknown names error listing
available groups), `mode` = `on_top` (v1's single mode: part
bounds-min-y onto the resting plane, x/z bounds-center onto the
centroid), `offset` `[x, y, z]` meters added after (default 0), in
**world axes**, never in the anchor's own frame (a radial offset on a
rotated copy is still a hand-computed world vector). The
anchor's summed normal must point upward (y > 0) or the op errors
(`anchor_not_upward`), never a guessed frame. Contract, verifiable
from stats: result `bounds.min.y` = plane y + offset y; x/z center =
centroid + offset. Normals, UVs, attributes, material, and the part's
own face groups are preserved (pure translation).

Per-instance anchoring: the anchor base may be **Instances**. Tags
ride the prototype, so `instance: <int>` (0-based copy index,
**required** for an instanced anchor, because a frame is never
guessed) selects the copy whose transformed group frame anchors the
part. The
frame is computed in world space, so the contract above holds
unchanged, and an out-of-range index is an error, not a stale
reference. The part input stays mesh-only: an instanced part is an
error with guidance (insert a `bake_instances` node between the
array/scatter node and this input, place the prototype before
array/scatter, or transform the instance set). Deterministic and pure.
Example: `examples/shelf_row`.

### mate (op_impl_version 1)

Recipe: `mo.mate(base, part, [leader], id = ..., anchor = ..., instance = int, face = ..., face_instance = int, mode = ..., roll_deg = ...)`

Face-to-face placement: positions its part by bringing two tagged faces
into coincidence. It is the any-orientation generalization of `place`,
for the butt joints, fitted openings, and dockings an assembly otherwise
places by hand-computed translate + rotate. Inputs: `base`
(geometry carrying the `anchor` tag), `part` (mesh or instance set),
optional `leader` (geometry carrying the `face` tag when the part does
not, see *riders* below). The output is the **moved part only**. The
base and the leader flow on through their own edges.

Mating frame (shared with the `mated` assertion, so the op and its
check cannot drift): **origin** = the centroid of the face set's single
opening when it has one (a port, a tube end, a flange bore, that is
the feature a joint registers on; a holed wall face registers on its
hole, not on the face's area centroid), else the area-weighted face
centroid; **normal** = area-weighted face normal; **secondary axis** =
world +Y projected into the face plane (a face within 1 deg of level
uses +Z, the asset front). The part is rotated so its face normal
opposes the anchor normal and its secondary axis aligns with the
anchor's (the upright rule: an assembly keeps its up, a horizontal mate
is a pure translation), then rolled by `roll_deg` about the anchor
normal, then translated origin onto origin. Each face set must be one
planar surface (face normals within 1 deg of the mean, vertices within
1e-4 m of the plane; slivers below 1e-6 of the set's area are not
tested) with at most one opening (loops that touch at a point stay
distinct); otherwise the op errors
(`anchor_not_planar` / `face_not_planar`, `anchor_degenerate` /
`face_degenerate` for zero area or canceling normals,
`anchor_disconnected` / `face_disconnected` for edge-disconnected
patches, `ambiguous_opening` for two or more openings), never a
guessed frame; each message names the tag to split or narrow.
Unknown or empty tags error listing what is available
(`unknown_anchor` / `unknown_face`, `empty_anchor` / `empty_face`).

Params: `anchor` (required; face group on input 0), `face` (required;
face group on the leader, or on the part when there is no leader),
`mode` = `coincident` (v1's single mode), `roll_deg` (default 0;
right-hand rule about the mated normal. A square mouth mates
identically at 0 and 90 deg, so the roll is explicit, never inferred),
`instance` / `face_instance` (0-based copy index, **required** when
the anchor base / the face-carrying input is instanced, rejected
otherwise). Contract, verifiable from stats and `moraine query`: the
output's own `face` group lies on the anchor plane with its frame
origin on the anchor origin and its normal opposing the anchor normal.
The `mated` assertion checks exactly this from the evaluated outputs.
Normals, UVs, attributes (TANGENT rotated as a surface direction),
material, and the part's own face groups are preserved: a rigid
motion. An instanced part moves as a set (transforms pre-multiplied,
prototype untouched, so instancing stays instancing).

Riders: an assembly's sibling outputs (the ribs, deck, and conduit that
share one hull's frame) carry no mating face of their own. Give them
the hull as a third input, `[anchor_base, part, leader]`, and the
part receives the rigid motion that mates the leader's `face` to the
anchor: one solved frame, no repeated arithmetic, every relation an
input edge. The leader's tag is read, the leader itself is not emitted.

Pair it with `mated`: the op constructs the relation, the assertion
declares it from the evaluated outputs (implementation vs
specification. The assertion outlives a later replacement of the op).
A joint consumed by a `boolean` union before it reaches an output
cannot carry `mated`; there the op's structured errors above and the
boolean margin discipline are the guard. Deterministic and pure.
Example: `examples/wall_duct`.

Provenance: `validate` lists every root `mate` node under `mates`, as
`{"node", "anchor", "face"}` with the origin word per role, `opening`
or `centroid`. The fallback is silent at execution: a bore rim that
lost triangles to its predicate has no closed opening, so the frame
falls back to the area centroid, the part mates off the bore, and the
paired `mated` (same frame) passes. Where the design registers on an
opening, expect `opening`; a `centroid` there means the tag lost the
rim. A blank-face mate legitimately reads `centroid`, which is why
this is a word on the report and not a warning.

### Asset assertions (`mo.assert_<kind>`, the rules every kind shares)

Declarative constraints as asset content, checked deterministically by
`validate` and gating its `ok` like tier-1 checks, never waivable.
A red entry is fixed in the geometry when the declaration is right;
a declaration that is wrong (it names faces the intended feature
does not own, or an output since renamed) is rewritten to state the
requirement (a human-owned one with the human's approval); no entry
is loosened to pass, and widening a tolerance is a requirement
change, the owner's to make. A green entry means the declared
condition holds of exactly the faces and outputs the entry names;
whether the declaration says what the requirement says is yours to
review. A tag written for the top shelf that also matches the post
tops when no shelf reaches its band passes as satisfied, so a
selection that only the intended feature can satisfy (two faces of
it, or a band nothing else reaches) is part of the declaration, and
the way to check one is to remove or move the feature and watch the
entry fail.
Asserts you author inside a sub-asset gate every consuming root's
validate too, checked in the sub-asset's own namespace once per
override identity (entries carry `asset` and, when present,
`overrides`; green entries with nothing to read fold into one
`ref_assets` row per identity (`asserts_total` / `asserts_ok`),
while failing ones and green ones carrying `unchecked`, `ceded`, or
`stronger` stay listed). Each entry is one recipe call,
`mo.assert_<kind>(id = ..., <key> = ...)`, and each kind's section
opens with its call and its keys (`contact`, `clearance`, `mated`,
`embedded`, `extent`, `count`). The report names an entry by that id
under the key `assert` (`"assert": "<id>"`). An output name in a call
may be a group's (see *groups*). The IR writes an entry as one
JSON object per line under `"asserts"`, the same keys with `kind`
and `owner` beside them (see *A complete minimal asset*). `tol`, `min`, `max`, and
`value` carry the kind's own measure (meters on every length kind,
copies on `count`), never a unit that
depends on which other keys are present. Every length comparison
against a `tol` or band edge has a storage-noise floor of 1e-6 m: a
measurement that misses its bound by less passes, and the entry then
carries `noise_floor: 1e-06`, on a green verdict only, since an entry
that makes many comparisons (a one-vs-rest clearance, an `every`
contact) can waive one and still fail on another. Vertices are float32
(~1e-7 relative), so a gap built exactly 0.15 wide reads 0.149999973.
Write the tolerance the requirement states (`0.15`), never `0.1499` to
dodge the storage; a miss beyond the floor is real; the overlap guard's
bar takes none.

- Interpenetration needs no entry: the built-in guard checks every
  output pair and every copy pair (see *no_overlap*).
- `self = True` (clearance; embedded's copy forms
  reuse the spelling under their own semantics, see `embedded`)
  additionally checks **every copy pair of each instanced output** in
  the entry's scope, the spacing failure the
  union view cannot see. Copies stay per-copy (instancing is never
  merged): per-copy AABB broad-phase, the exact kernel on survivors,
  the pair's surface distance >= `min`. The entry reports the checked
  count as `self_pairs` (0 when the scope has no instanced output.
  The entry is then vacuous and legal before the first array/scatter
  exists) and failing copy pairs under `detail.failing_pairs` as
  `[name[i], name[j], distance]` rows. **Without the
  opt-in**, an entry whose scope holds instanced outputs names them
  under `unchecked.instanced_self_pairs`. The report states what it
  did not check.
- Quantifiers over instanced outputs, and a group name in place of
  an output, at a glance:

  | kind | against the other output | between the copies of one output | a group name (`mo.group`) |
  |---|---|---|---|
  | overlap guard | the union already fails on any copy (holds for every copy by construction) | every copy pair, by default | outputs only; an `embedded` between groups cedes the member pairs |
  | `contact` | existential on the union. `every = [name]` makes it hold for every copy, distance and `min_area` alike | none (all pairs touching is unsatisfiable; a "no isolated copy" form is not offered) | the members' union; `every = [group]`, every copy of every member (counts per member) |
  | `clearance` | the union minimum is the per-copy minimum (holds for every copy by construction) | `self = True`, every copy pair >= `min`; over one output alone, the copy form | the union; `others = True` partners are the outputs outside the group; `self` refused |
  | `embedded` | pair form on the unions | `self = True`, band over overlapping pairs; `connected = True`, the band plus one body | pair form: the union's intersection depth; `self`/`connected` refused |
  | `mated` | a copy index per face. `every = [name]` judges every copy of the named outputs, two named outputs paired by index | - | refused (a group has no faces) |
  | `extent` | `outputs` form: the union of the copies; `faces` form: one indexed copy | `every = True`, each copy's own bounds or face set | `outputs` form: the members' united bounds, `every` every member's copies; `faces` form refused |
  | `count` | the copies are the measure | - | refused (a sum of copy counts is no output's count) |

  Without the opt-in an entry names what it did not quantify under
  `unchecked` (`instanced_self_pairs` / `instanced_copies`); copy
  pairs policed by a declared embed appear under the guard's
  `ceded`, not `unchecked`. `--failures-only` keeps the three keys as
  counts (see *checks*).
- `owner`: `"human"` or `"agent"`, on every entry, written by the
  build from the file that declared it: the requirements file is the
  human's, an asset file the agent's (see *requirements*). Human-owned
  entries are
  requirements: the agent proposes changes but does not edit them
  without approval (process + report visibility; the owner rides every
  validate entry so a weakened requirement is visible). Agent-owned
  entries are invariants the agent records for itself and may edit
  freely. The criterion is the **provenance of the requirement**, not
  who typed the call: a constraint the human stated ("no part may
  interpenetrate another") goes in the requirements file even when
  the agent drafts the entry; the agent's own recorded invariants stay
  in the asset file.
- Checked, never solved: asserts only check. `validate`
  never moves geometry to satisfy one. Solving belongs to operators
  (`place`, `mate`, `scatter`, `boolean` each solve a sub-problem under
  their own contracts: deterministic result, inputs as graph edges,
  structured error when unsolvable), which is why an op and its
  assert come as a pair. `mate` constructs the joint, `mated` checks
  it from evaluated geometry; `array`/`scatter` emit the copies,
  `count` checks them from the evaluated output.
Examples: `examples/shelf_pair` (groups, the one-vs-rest clearance, a
contained mate over every board, a feature-bound extent, count),
`examples/scatter_field` (contact quantified per copy, free-overlap
embeds on a scatter), `examples/pallet` (per-copy contact and the
copy form of clearance).

### no_overlap (the built-in overlap guard)

The built-in overlap guard (report kind `no_overlap`; never
authored. Its entry carries `builtin: true` and the reserved id
`no_overlap`): every validate checks **every geometry-output pair
and every copy pair of each instanced output** for material
interpenetration, with no entry in the file. A intersected with B,
computed exactly by the manifold3d kernel on world-space solids,
must not be deeper than
1e-5 m (the same float32 storage-noise floor as contact's default.
A flush-contact fp slab is a few ulps of its coordinate thick, so a
tighter bar fails correct geometry at tens-of-meters scale). Depth
is the intersection's mean thickness (2 x volume /
surface area) over its load-bearing faces. Zero-thickness sheets
from coplanar contact (flush construction; a cutter seated in the
pocket it cut) are excluded, so the criterion is independent of
contact area and stable under coplanar residue: flush contact of
any size passes, material interpenetration fails; resting
assemblies need no exclusion list, and outputs added later are
covered automatically. Copy pairs stay per-copy (instancing is
never merged; a pair whose box overlap is thinner than the bar on
any axis, disjoint or merely touching, is under the bar with no
solid built, cross-output and copy alike, so a flush-built kit
clears its touching pairs for free). The guard also runs inside
every referenced asset's per-identity bundle, over all of that
asset's outputs as the consumer's overrides shaped them (consumed
or not), with the asset's own embeds ceded. Its entry there
carries `asset`. The report stays one entry: `pairs`
and `self_pairs` counted, failing pairs listed only on failure
(`detail.failing_pairs`, `[a, b, volume, depth]` rows with copies
as `name[i]`). Image outputs are outside its domain; a
non-watertight output fails it with the reason unless bounds
disjointness already proves its pairs clear, and a non-watertight
prototype fails the copy pass only when AABB overlap leaves copy
pairs unresolved. The guard's owner is `human` by spec assignment
("no part may interpenetrate another" is the standing human
requirement), so it is never weakened, only **ceded to declared
embeds**: an `embedded` entry takes its pairs over (see
`embedded`), and the guard reports them under `ceded`
(`pairs` / `self`).

### groups (`mo.group`, several outputs under one name)

A piece built from several outputs (a board
and its posts, touching by construction) is named once with
`mo.group("shelf_a", ["shelf_a_board", "shelf_a_posts"])` (two or
more distinct outputs, never another group; one namespace with
output names) and stands for the **union of its members** wherever
an entry takes an output name (per kind, the last column of the
quantifier table below);
`outputs = ["shelf_a"], others = True` is the one line "clear of
every other piece". A group a requirement names is declared in the
requirements file (see *requirements*). A refusal lists the members,
so the fix is a copy.
Relations between members (the board on its posts) stay entries
over output names; the overlap guard's pairs stay outputs. The two
names a relating kind takes must stand for disjoint outputs. A
group beside one of its own members is refused, since that output
would be measured against itself. The entry echoes the resolution
as `members`. Example: `examples/shelf_pair`.

### contact (assert kind)

Recipe: `mo.assert_contact(id = ..., outputs = ..., every = ..., tol = ..., min_area = ...)`

The minimum surface distance between the two
named outputs, measured on world-space triangles, must be <= `tol`
(meters; default 1e-5, the float32 storage-noise floor. Raise it
explicitly only for assets far beyond ~160 m; it absorbs *storage*
noise only. A coordinate you authored rounded, a part typed at
0.333 against a face at 1/3, is a real gap of the rounding size, so
seat parts with `place`/`mate` or state the tol). Crossing surfaces
count as distance 0, so contact plus the overlap guard
asserts **resting contact**, that is "touches its support" as a
checkable constraint. Watertightness is not required (raw surfaces
measure); an image, empty (zero-triangle), or non-finite output is
unevaluatable and fails with the reason. Every entry, green or
failing, contact or clearance, reports the true closest approach as
`distance`, the number you would otherwise raise a tolerance to
read; a quantified entry adds the loosest copy (`every_max`, contact)
or the closest copy pair (`self_min`, clearance), each with its
`name[i]` label. An instanced output counts as
the **union of its copies** here, which makes the guard existential.
One near copy passes for all: "the boulders meet the ground" is true
when one boulder does. `every = [name, ...]` (names drawn from the
entry's `outputs`. Contact has two subjects, so the key says whose
copies; `extent`, with one, takes `every = True`) quantifies it:
every copy of each
named output must individually be within `tol` of the other output.
The entry reports the checked copy counts as `every` (a plain-mesh
name counts 1. It is vacuous, and legal before the first array/scatter
exists; a group's plain member counts 1 too and is judged apart from
the other members, labeled by its name) and failing copies as
`name[i]` with their distance
(`failing_copies`). **Without the opt-in**, a contact entry over
instanced outputs names them under `unchecked.instanced_copies`,
and the report's `warnings` carry a `contact_unquantified` row for
it. An output arrayed after the entry was written weakens the
entry to "some copy touches" with no file diff, so the row is where
the weakening shows. An entry whose `every` judges one side copy by
copy and leaves the other existential ("every foot on some pad")
draws no row; the other side stays under `unchecked`. An `every`
over one plain output judges nothing more, and the row stays.
**`min_area`** (square meters, yours to state
like clearance's `min`) is the bearing form: an edge or a corner on
a face is distance 0 like a face on a face, so a part that must be
*seated*, not merely touching, states the face area it shares with
its support: `mo.assert_contact(id = "bear", outputs = ["stringers",
"steps"], every = ["steps"], min_area = 0.02)`. The shared area is
the overlap of the two surfaces where their facets lie on one plane
(anti-parallel within 0.01 deg, within `tol` along the normal); the
entry reports it as `area`, under `every` the copy sharing least as
`area_min` and the copies short of the bound as `bare_copies` (each
copy's area is against the other output as a whole, so a step borne
by two stringers counts both); the noise floor applies to the side
length of the area. A curved seat
therefore shares none unless both tessellations agree (a shaft in a
bore is `mated`'s question). A written `0` is refused: leave the key
out to require touch alone.

Examples: `examples/anchored_crate` (two rests), `examples/pallet` (per copy).

### clearance (assert kind)

Recipe: `mo.assert_clearance(id = ..., outputs = ..., others = ..., self = ..., min = ...)`

The minimum surface distance (`contact`'s measure) must be
>= `min` (meters, > 0). It is one scalar over the whole pair,
the closest approach in any direction, never a per-axis gap ("keep N
above" on a spiral measures the diagonal to the neighboring tread,
not the rise). `min` is the design gap, required: "robustly
separate" is `min = 0.001`, a physics-simulation target 0.01 to
0.04 m (engine contact offsets). There is no `tol` and no `max`:
parts that must stay within a distance are `contact`'s. Touching or
crossing surfaces fail at distance 0. A **fully contained** output is
invisible to surface distance (the surface-to-surface gap is
genuinely positive): material separation is the overlap guard's domain
and enclosure the `not_enclosed` check's. Pair them when that is
the actual requirement. Over instanced outputs `every` does not
apply (see the quantifier table below); the gap **between copies**
is `self = True` ("keep the lockers N apart"). Written over one
output alone, `outputs = [a], self = True`, it is the copy
form: `a`'s copy pairs and nothing else, under a `min` that is the
spacing's own, so a spacing never borrows a partner output or its
gap (the entry reports `self_pairs` and `self_min`, no `distance`).
Beside a pair it adds the copy pairs of both outputs under the
pair's `min`. **"Clear of
everything else"** is `outputs = [a], others = True`: `a` must
keep `min` from every other geometry output the asset has when
`validate` runs. An output added later is covered automatically,
where a list of pair entries would leave it a pair nobody wrote.
The entry reports `pairs` (partners counted), `nearest` and its
`distance`, and failing partners as `detail.failing_partners`
`[other, distance]`; `self` composes over `a`'s copies as usual;
images are not partners, and neither are a group's own members;
with no other output the entry is vacuous
and legal. Not on contact (everything touching is a contradiction).

Examples: `examples/shelf_pair` (one-vs-rest), `examples/pallet` (the copy form).

### mated (assert kind)

Recipe: `mo.assert_mated(id = ..., faces = ..., every = ..., mode = ..., tol = ..., angle_tol_deg = ...)`

Two tagged faces, `faces = [[output, tag],
[output, tag]]`, each entry optionally carrying a 0-based copy index
as a third element when that output is instanced (required then).
Mode `"coincident"` (the default): the faces must be coincident,
that is stage (a) their normals anti-parallel within
`angle_tol_deg` (default 0.01), stage (b) their mating-frame origins
within `tol` (meters; default 1e-5, the same float32 floor as
contact, but the origin comparison inherits float32 *relative*
error, so the default's headroom thins with distance from the world
origin: a design-exact joint ~12 m out already measures a few 1e-6;
raise `tol` deliberately for assets tens of meters across).
The frame is the `mate` op's (see *mate*: opening-aware
origin, area-weighted normal), read from the **evaluated outputs'
world-space faces**, never from an op's transform, so `mated`
declares what `mate` constructs and keeps declaring it when the
construction changes. It is
the check `contact` cannot be: a duct slid across its wall port still
touches (distance 0) and does not interpenetrate, yet its origins
are the slide apart, so mated fails with the measured `offset`, and
the entry carries `detail` (`angle_deg`, `offset`, `distance`, and
`origin`, that is the two faces' frame provenance words, `opening` /
`centroid`, see *mate*) when both frames exist, pass or fail,
numbers first. The report entry echoes
the entry's `mode` (like `owner`), so a switch to the weaker
in-plane registration is visible in the report.
Mode `"contained"`: the joint where a smaller part butts a larger
blank face. The face centroids legitimately differ, so coincidence
would fail correct geometry. `faces = [[anchor...], [part...]]`, and
**order carries the roles**: `faces[0]` is the containing face. The
same stage (a), then the faces must be coplanar within `tol` along
the mating normal and the part's contour must lie inside the
anchor's **outer** contour (boundary vertices inside or within `tol`
of the rim, and the crossing test honors the same allowance. A
part edge seated flush with the anchor's edge is containment; a
part boundary edge leaving the contour beyond `tol` fails outright,
so a non-convex anchor's spanned notch is caught even with
every vertex inside). Holes in the anchor face are ignored: a blank
flange may cover a bore. The in-plane position within the anchor is
free by design. Pair the entry with `contact` (seated); the
overlap guard carries the rest. `detail` carries `angle_deg`, `plane_gap`,
`outside`, `crossings`, `origin`. A green contained entry whose
origins also register within `tol` carries `stronger: "coincident"`
on the entry line: it under-claims (assert the stronger mode) and
nothing else reports that.
In either mode, a face without
a frame fails with the reason: an unknown tag (the available tags
are listed), an instanced output without a copy index, an
out-of-range index, an empty, non-planar, disconnected, or
multi-opening face set (the `mate` refusals, as reasons). Roll about
the mated normal and contour *equality* are not checked in this
version.
**`every = [name, ...]`** (names drawn from the entry's `faces`)
judges the joint on every copy of the named outputs, and a face of
a named output carries no copy index. Two named outputs pair copy
by copy, crown `i` on trunk `i`: `{"faces": [["trunks", "seat"],
["crowns", "socket"]], "every": ["trunks", "crowns"]}`, and their
copy counts must agree, a mismatch failing with both counts. One
named output puts each of its copies against the other face as
written: every stringer butting one platform face is `{"faces":
[["platform", "end"], ["stringers", "end"]], "every":
["stringers"], "mode": "contained"}`. The entry reports the copy
counts as `every` and, under `detail`, `pairs` and each measure's
maximum over the pairs (`angle_deg_max`, `distance_max`, or
`plane_gap_max` / `outside_max` / `crossings_max`); failing pairs
ride the fix as face labels and `detail.failing_pairs` with their
own numbers. One entry per copy index states the same thing only
while the copy count holds: a copy added later is a joint no entry
names, where `every` covers it.

Examples: `examples/wall_duct` (coincident), `examples/shelf_pair` (contained, over every board).

### embedded (assert kind)

Recipe: `mo.assert_embedded(id = ..., outputs = ..., self = ..., connected = ..., min = ..., max = ...)`

The positive form of the depth axis, the
**declared intent that material interpenetrates**, judged on the
guard's load-bearing depth measure in a declared band
`min` <= depth <= `max` (meters). That depth is 2 x V/A of the
intersection, not the geometric sink: a cylinder of radius 0.18
sunk 0.05 reads 0.039 (r x h / (r + h)) because the side wall counts
in the area. A stated sink is an `extent` `min_y` on the sunk
output; `embedded` states the permission and its band guards it.
Use it where the overlap *is* the construction: a repeat unit
embedding into its neighbor, a part sunk into terrain. These are
cases a waive would erase without stating the
requirement, and where the **away direction** matters: parts that
separate fail `min`, which no overlap check can see. Three forms:
- **Pair form**, `outputs = [a, b]`: the two outputs'
  intersection depth must lie in the band (instanced outputs as the
  union of copies). `min` defaults to 1e-3 (the >= 1 mm embed rule
  as a floor); a written `min` 0 is the **free-overlap spelling**
  for a depth with no design value (a scatter sunk into displaced
  terrain measures whatever its seed gives, and a positive floor
  there would be a claim about one seed): the pair is ceded from
  the guard and reported, and passes whether or not the parts meet.
- **`self = True`**, `outputs = [a]`: every copy pair of `a`
  overlapping above the defect floor (1e-5 m) must lie in the band.
  Its `min` defaults to 0, so both bounds omitted is the **pure
  free-overlap declaration**: copies may interpenetrate, said
  visibly, with an owner, traveling through `asset_ref`, none of
  which a checks waiver gives. A written positive `min` is a
  positive claim: copies that have all separated fail it.
- **`connected = True`**, `outputs = [a]`: `self`'s band, plus
  the copies must form **one body** through embeds >= `min` (a rail
  of overlapping segments, a chain of modules). Strictly stronger
  than "every copy has a neighbor" (a chain broken into halves
  still fails) and links through material embeds only, never mere
  touch. One output is one body: split a deliberate two-body
  output into outputs. On failure the entry carries `components`
  and `detail.components` (`[size, first_copy]` rows).

`max` is optional: state it when the design magnitude is known (a
4.8 mm seam gets `[0.004, 0.006]`); omit it when the depth is
parameter-dependent (terrain sink). The entry then shows
`max: null`, the visible spelling of declared-unbounded. A band
edge above 0 and at or below the defect floor (1e-5 m) is refused
at parse. The guard treats overlap that shallow as flush contact,
so the band could never be policed (a written `min` 0 is the
free-overlap spelling, not a floor). Entries
always carry the resolved band and the measured `depth` (pair
form) or `overlapping` count and `depth_range` (copy forms),
except the pure free-overlap declarations (pair `min` 0 without
`max`; `self` with no bounds), which measure nothing: a band
admitting every depth needs no kernel, so they report
`measured: false` (the `self` form with the copy pairs it would have
measured as `candidate_pairs`) and pass on an open (non-watertight)
mesh too;
failing copy pairs ride `detail.failing_pairs`. Unevaluatable
entries fail with the reason; the copy forms are vacuous
(`self_pairs` 0) on a plain-mesh output, so the declaration is
legal before the first array/scatter node exists.

**Cede:** a declared embed takes its pairs over from the built-in
overlap guard: the pair form its
cross-output pair, the copy forms that output's copy pairs. The
embedded entry polices them under its own band; the guard
reports what it ceded under `ceded` (`pairs` / `self`) and checks
the rest. Embeds declared **inside a referenced asset** travel
with its parts: two consumer outputs that reach the same reference
identity through one rigid placement (`transform`s composed; ops
that leave positions untouched, that is `tag`, `material`, `unwrap`,
`smooth_normals`, pass through) inherit that asset's pair embeds,
and an output inheriting a copy-form embed cedes its own copy pairs
even through `place`/`mate` (a rigid motion of the whole set keeps
every copy pair's depth). A scaled placement, a pair re-seated by
`place`/`mate`, or a different override identity does not map, and
the consumer declares those pairs itself. The bounded idiom pairs
naturally:
`embedded` states the intent, and the band's `max` is its guard.

Examples: `examples/brick_building` (pair form), `examples/scatter_field` (free overlap).

### extent (assert kind)

Recipe: `mo.assert_extent(id = ..., outputs = ..., faces = ..., measure = ..., every = ..., value = ..., tol = ..., min = ..., max = ...)`

**A stated dimension as a gate**, the number you
would otherwise read with `query` or `stats` and write into prose
(where it silently rots on the next edit). One subject: `"outputs":
[a]`, the output's world-space vertex bounds, copies united,
measured on actual transformed vertices (so a tilted copy of a
curved prototype reads true where `stats` bounds transform box
corners); or `faces = [output, tag]` / `[output, tag, copy]`, one
tagged face set on one copy (the copy index is required on an
instanced output, as in `mated`; the union is the `outputs` form).
`measure` names the bound to read, one token: `size_x|y|z`,
`min_x|y|z`, `max_x|y|z`, `center_x|y|z` (meters along that axis,
the face-predicate field spelling). The judgment is `value` within
`tol` (default 1e-3, your allowance; a stated design tolerance is
an explicit `tol`) **or** a `min`/`max` band
with at least one edge, never both in one entry. For an
override-controlled dimension, assert its allowed range and keep
the default in the bound node parameter; pinning the default would
reject valid overrides. Examples: a
flight's total rise `outputs = ["steps"], measure = "max_y",
value = 2.1`; a flange diameter `measure = "size_x", value =
0.34`; a part lifted clear of the ground `measure = "min_y", min =
0.05` (a per-output ground statement the asset-level `grounded`
check does not make); a bore `faces = ["pipe", "bore"], measure =
"size_x"`. A diameter is a `size_*`, so halve it yourself for a
radius.
**`every = True` reads each copy on its own** instead of the union:
every leg on the ground `outputs = ["legs"], measure = "min_y",
every = True, value = 0`; every post cap level `faces =
["posts", "cap_top"], measure = "max_y", every = True, value =
0.82` (no copy index, `every` names them all; a group: every
member's copies). The entry reports the copy counts as `every`, the
measurements' span as `measured_range` in place of `measured`, and
failing copies as `name[i]` (a plain member by its name) with
their number (`failing_copies`).
The union form is not a weakened per-copy check but a different
requirement (a flight's rise, a row's width are dimensions of the
set), so it raises no `unchecked`; write `every` when the sentence
is about each part (each leg reaches the ground).
Every evaluable entry reports `measured` beside the requirement it
echoes (`value`/`tol`, or the band with the omitted edge as `null`),
pass or fail; image, empty, and non-finite subjects, an unknown tag,
or a missing copy index fail with the reason.

Examples: `examples/flange_pipe`, `examples/stairs`.

### count (assert kind)

Recipe: `mo.assert_count(id = ..., outputs = ..., value = int, min = int, max = int)`

The output's **copy count** (an instanced output's
copies, a plain mesh counting one) against `value` (`"outputs":
["saddle"], "value": 2`, what `every` cannot say: two saddles, not
one) or an integer `min`/`max` band (`outputs = ["stones"], "min":
150`; `min` equal to `max` is refused, because that is `value`). The
check `array`/`scatter` pair with: it reads the evaluated copies, so
on an instanced output it reads that node's `count` back. The entry's
content is the requirement's ownership (a count held only in a node
param is yours to edit with no approval and no report line) and the
instancing itself (an output later baked or unioned into one mesh
reads 1 and fails, the change `every` misses). Reports `measured`;
an image output is unevaluatable.

Examples: `examples/stairs`, `examples/scatter_field`.

### budgets (`mo.budget`, asset-declared acceptance budgets)

Optional, declared as `mo.budget(materials = 4)`
in the asset's requirements file. A declared budget gates `validate` like
an assertion. The report gains a top-level `budgets` list with one
entry per declared budget (`budget`, `ok`, `count`, `limit`, and for
`materials` the `identities`, each `{name, outputs, transparent,
emissive}`, plus a `fix` on failure) and `ok` is false above the
limit; an undeclared budget is not checked at all (there is no built-in
default; `moraine stats` always reports `asset.materials`, the
identities this asset ships, by name, the same scope the budget
counts, so headroom is visible before declaring one; the two counts
agree whenever `material_name_unique` is green). Budgets are acceptance
criteria the human owns, so they live in the requirements file rather
than in a tool flag. Raising one is an edit of that file the human
approves, not a silenced warning. A budget counts what this asset's outputs
ship, identities arriving through `asset_ref` included: budgets bound
the exported glb, so an assembly's declared budget covers the whole
assembly. Unlike asserts, a sub-asset's *declared* budgets are not
re-run by its consumers. Each asset's declaration gates in its own
`validate`.

`materials` counts distinct material identities this asset ships:
identity = name + factors + textures, so same name and same params on
several parts is one identity. Shared identities count once, and
sharing a kit material across many referenced pieces never raises the
count; outputs with no material share the
neutral default, which ships as a real glTF material and counts once
under `moraine_default`. Transparent (`alphaMode: BLEND`) and
emissive materials are structurally their own identities (they can
never share with an opaque one) and the failure `fix` names them so
the remaining candidates for sharing are clear. Each identity is one
draw state in the engine and a submesh boundary that static/dynamic/GPU-
instancing batching cannot cross; the budget is a per-asset
discipline declared by the human who knows the target, not an engine
hard limit.
Example: `examples/kit_variants`.

