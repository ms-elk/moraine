# Operator semantics

Each operator's semantics (input -> output meaning) is the stable contract,
independent of implementation, and is pinned by golden-mesh tests.
`op_impl_version` is bumped on ANY output-semantics
change, including bug fixes; an op that calls a kernel (manifold3d,
trimesh, or a shared in-package kernel module) declares it in
`KERNELS`, and the kernel's version joins the cache key.
Each op is one recipe call,
`mo.<op>(inputs..., id = ..., <param> = ...)` (see *recipe*). All
sizes are meters.

Vertex attributes follow the glTF vocabulary (`NORMAL`, `TEXCOORD_n`,
`COLOR_n`, `TANGENT`). Each op declares
`ATTRIBUTE_POLICY`: **preserved** (input slots beyond POSITION/NORMAL
ride through) or **dropped** (topology is rebuilt and the slots are
discarded, so re-author downstream). Never silent: `validate` warns when a
graph feeds attributes into a dropping op and the loss reaches an output
(see *checks*). Where an op's paragraph says "UVs are dropped", that
extends to all extra attribute slots.


## box (op_impl_version 1)

Recipe: `mo.box(id = ..., size = ..., segments = int)`

Emits an axis-aligned rectangular box centered at the origin (pivot at
center of volume) with edge lengths `size = [sx, sy, sz]` in meters
(default `[1, 1, 1]`; all components must be > 0; a scalar broadcasts
to a cube: `size` is scale-like, and only scale-like params accept a
scalar; translations, rotations, offsets, and directions must be
written as 3 numbers). `segments = [nx, ny, nz]` divisions per axis
(default `[1, 1, 1]`, each 1..4096, the face quads together bounded by
the shared tessellation cell budget). Each face is its own
`(nu + 1) x (nv + 1)` vertex lattice (vertices duplicated across faces
so each face has hard edges and a per-face outward normal, CCW-wound
triangles), so the default is 24 vertices and 12 triangles. Segments
do not change the shape; give them to a box that trims or cuts a
displaced surface (the `boolean` face discipline). Deterministic; no
stochastic parameters.
Example: `examples/box`; a segmented trim box: `examples/scatter_field`.

## cylinder (op_impl_version 1)

Recipe: `mo.cylinder(id = ..., radius = ..., height = ..., segments = int)`

Emits a Y-axis cylinder centered at the origin (pivot at center of
volume) with `radius` (default 0.5), `height` (default 1), and
`segments` around the circumference (default 32, 3..4096). The side is
smooth-shaded (shared ring vertices, radial normals); the caps are hard
(duplicated ring vertices, +/-Y normals) and closed with a center-vertex
fan. Vertex 0 of each ring sits at angle 0 (+X); winding is CCW from
outside. Deterministic.
Example: `examples/pipe_tee`.

## sphere (op_impl_version 1)

Recipe: `mo.sphere(id = ..., radius = ..., segments = int, rings = int)`

Emits a UV sphere centered at the origin with `radius` (default 0.5),
`segments` longitude divisions (default 32, 3..4096), and `rings`
latitude divisions (default 16, 2..4096). Smooth-shaded: shared vertices
with normals equal to the normalized position. Poles are single vertices
at +/-Y closed with triangle fans; interior latitudes are quad bands.
Winding is CCW from outside. Deterministic. When slicing with a half-space
`boolean` (e.g. cutting a hemisphere), ring parity decides where the
rim lands: with even `rings` a vertex ring lies exactly on the equator
and the cut rim keeps the full nominal radius; odd `rings` cut
mid-band, where the polygonal radius falls short of nominal (r 1.0,
rings 33 -> rim about 0.9989). This is a visible ledge against an
equal-radius cylinder, with no degenerate triangles either way.
Example: `examples/scatter_field`.

## revolve (op_impl_version 1)

Recipe: `mo.revolve(id = ..., profile = ..., segments = int, angle_max_deg = ...)`

Emits the solid of revolution of `profile`, a list of two or more
`[radius, y]` points (m) forming an open polyline in the +X half-plane,
swept a full 360 deg around the +Y axis with `segments` steps (default
32, 3..4096). Order the profile so the solid lies on its left
(bottom-to-top along the outside); a profile that would face the
surface inward is an error naming the fix (reverse the point order).
A radius within 1e-9 below zero, as an interpolation leaves at the
axis, is read as 0.
Ends close automatically: an end with radius 0 becomes a pole vertex,
an end with radius > 0 a flat horizontal cap (duplicated ring, +/-Y
normals, center-vertex fan). The result is always watertight. Radii
must be >= 0, with 0 allowed only at the ends; consecutive duplicate
points, identical endpoints, interior zero radii, exact double-backs,
self-intersections (the closure to the axis included), and profiles
enclosing no area are errors, as is a rings x segments product beyond
the 4096 x 4096 cell budget (hard joints duplicate rings, so the
per-param ceilings alone do not bound it). Circumferential shading is
smooth (shared ring vertices); a profile joint bending by more than
`angle_max_deg` (default 30) is hard (duplicated rings), a gentler one
smooth (shared ring, half-plane bisector normal). The default keeps a
dome's meridians smooth and a flange's steps crisp with no param
written. Vertex 0 of each ring sits at angle 0 (+X); winding is CCW
from outside. Deterministic. Example: `examples/turned_column`.

## grid (op_impl_version 1)

Recipe: `mo.grid(id = ..., size = ..., segments = int)`

Emits a flat rectangle in the XZ plane at y = 0, centered at the origin,
facing +Y. `size = [sx, sz]` (default `[1, 1]`); `segments = [nx, nz]`
quad divisions (default `[1, 1]`, each 1..4096). Vertices are shared,
normals all +Y, winding CCW seen from +Y. The open boundary makes it the
canonical base for `extrude` and a scatter target, and keeps it out of
`outputs`: an open surface fails `watertight` (and every solid-based
assert), so close it with `extrude` before naming it; a scatter target
may stay open. Deterministic.
Example: `examples/scatter_field`.

## transform (op_impl_version 1)

Recipe: `mo.transform(a, id = ..., translate = ..., rotate_deg = ..., scale = ...)`

Applies a TRS transform to its single input. Params: `translate`
`[x, y, z]` (default `[0, 0, 0]`); `rotate_deg` `[rx, ry, rz]` degrees
applied extrinsically in X, Y, Z order, i.e. `M = T * Rz * Ry * Rx * S`
(default `[0, 0, 0]`); `scale`, scalar or `[sx, sy, sz]`, non-zero
(default 1).
Only scale-like params (`scale`, array `step_scale`, box `size`) accept
a scalar; a bare number for `translate` or `rotate_deg` is rejected as
`bad_param` rather than broadcast. Mesh input:
vertices map through M, normals through the inverse-transpose
(renormalized), TANGENT directions through the linear part
(renormalized; a mirroring transform flips the handedness sign w); a
negative-determinant scale flips winding to keep faces
CCW-outward. Instances input: M pre-multiplies every instance transform
(the whole set moves rigidly; the prototype is untouched); a
negative-determinant M is rejected there, since the
shared prototype's winding cannot flip per instance. Mirror the
prototype mesh before the array/scatter node instead. Deterministic.

Because the order is fixed, a compound orientation usually wants two
nodes: an inward-tilted leg on a circle is tilted in its radial plane
first (`rotate_deg: [0, 0, tilt]`), then swung to azimuth in a second
transform (`rotate_deg: [0, azimuth, 0]`). Folding both into one node
applies the tilt in the wrong plane, since Rz acts before Ry. Sign
reference for the +Y-up right-handed frame: `rotate_deg: [0, 90, 0]`
maps +X to -Z.
Example: `examples/box`.

## extrude (op_impl_version 2)

Recipe: `mo.extrude(a, id = ..., distance = ..., direction = ...)`

Solidifies its single mesh input by sweeping it along an offset.
`distance` (default 0.1, > 0: a negative distance would invert the
solid; sweep the other way with a `direction` vector instead);
`direction`, `"normal"` (offset
each vertex along its vertex normal; default) or a fixed `[x, y, z]`
vector (normalized, scaled by `distance`). A vector that opposes every
face normal sweeps the flipped sheet, so the output is the intended
outward solid, exactly as if the input were mirrored first; a vector
tangent to any face, or opposing some faces while aligning with others,
is an actionable error (the sweep would self-intersect). Output =
offset copy of the surface (front, original winding) + the original
surface flipped (back) + hard-edged side quads along every boundary
edge, wound outward for positive `distance` along the surface normal.
A closed smooth-shaded input (no boundary edges) yields a two-shell
solid. Boundary edges are vertex-index edges, so a flat-shaded *closed*
surface (duplicated vertices with divergent normals, that is `box`,
`boolean`/`bevel` output) would extrude into per-face slabs with open
wedges, not a thickened shell. It is refused. Shell a solid with a
boolean difference of an inset copy, or weld it smooth first
(`smooth_normals` with `angle_max_deg` above its hardest edge. The
default 30 keeps hard edges split, which stay refused). Seam-split
closed surfaces whose duplicates carry agreeing normals (UV seams) are
smooth-shaded and extrude as their boundary edges imply. Thicken
smooth-shaded sheets (`grid`, `smooth_normals` output). Sides are
flat-shaded; front/back keep the input's shading. Input UVs are dropped
(unwrap downstream). Deterministic.
Example: `examples/scatter_field`.

## boolean (op_impl_version 1)

Recipe: `mo.boolean(a, b, ..., id = ..., mode = ...)`

CSG on two or more watertight mesh inputs via manifold3d. `mode`:
`"union"` (default), `"difference"` (input 0 minus the union of the
rest), or `"intersection"`. Inputs are welded before the operation; a
non-watertight input is an actionable error, not a silent repair; an
instanced input (an arrayed cutter) is refused. Insert a
`bake_instances` node between the array/scatter node and this input.
Margin discipline: a cutter or operand face exactly on another operand's face
(including the face a part rests on: a `difference` cut that starts at
the resting plane is a tangential cut) breeds slivers or a
non-watertight result; embed >= 1 mm or overshoot. Face discipline: a
cut through a face with no interior vertices triangulates as fans from
that face's corners (a trim box's wall crossed by a 146-vertex terrain
edge becomes 36 m slivers), so a box that trims a displaced surface
gets `segments` a few times coarser than the surface's own (1 m cells
against 0.25 m terrain segments hold the cut walls' triangles near
1 m), with the surface overshooting the box (flush walls fan
regardless); a cutter of flat faces needs none. Output
is flat-shaded (per-face duplicated vertices and normals). CSG results
are hard-surface by nature; curved-surface shading is dropped, and a
boolean output with curved surface reaching `material` without
`smooth_normals` in between gets a `flat_shading` warning from
`validate` (an all-planar output shades identically either way, so
none). Input UVs are dropped. A `union` of operands that do not touch
is legitimate, one output of several closed bodies (an assembly
shipped as one part): nothing warns and `watertight` passes.
Deterministic.
Example: `examples/pipe_tee`.

## smooth_normals (op_impl_version 1)

Recipe: `mo.smooth_normals(a, id = ..., angle_max_deg = ...)`

Recomputes shading of its single mesh input by angle threshold, the
ecosystem-standard angle-threshold auto-smooth every DCC tool ships.
Coincident vertices are
welded, then re-split so that faces meeting across an edge at up to
`angle_max_deg` (default 30, > 0) share a vertex with the
corner-angle-weighted average normal (invariant to how coplanar regions
are triangulated), while sharper edges (and non-manifold edges) keep
hard per-face corners. Stored input normals are discarded and rebuilt
from geometry; face count and winding are unchanged. Intended use:
restoring curved-surface shading after `boolean` / `bevel`, whose
outputs are flat-shaded (the threshold must exceed the facet angle of
the curved members, 360/`segments` for a cylinder, and the default 30
does nothing for a 12-segment cylinder, whose facets meet at 30 deg);
and setting the look of displaced surfaces (`noise_displace` terrain),
where nearly every edge exceeds 30 deg and the default reads faceted,
so raise `angle_max_deg` to the crease angle the surface should keep.
Input UVs are dropped (unwrap downstream).
Known limit (shared with every angle-based auto-smooth): where two
surfaces meet near-tangentially (e.g. a union of equal-radius
cylinders, whose crease dihedral falls to 0 deg at the tangent points),
the seam smooths over and shading bleeds across it; give the members
distinct radii (or otherwise avoid tangency) so the crease stays above
the threshold. Deterministic. Example: `examples/pipe_tee`.

## array (op_impl_version 1)

Recipe: `mo.array(a, id = ..., count = int, step_translate = ..., step_rotate_deg = ..., step_scale = ...)`

Linear array of its single input as instances. `count` (default 2,
1..1000000); per-step TRS `step_translate` `[x, y, z]` (default
`[0, 0, 0]`), `step_rotate_deg` `[rx, ry, rz]` (default `[0, 0, 0]`),
`step_scale`, scalar or `[sx, sy, sz]` (default 1, components > 0: a
mirroring step would alternate instance handedness against the shared
prototype's winding), composed
as the `transform` matrix M; copy i gets `M^i` (copy 0 = identity).
Steps compose about the **world origin**: `step_scale` scales copies
toward the origin (build the prototype at the origin and carry the
array with a later `transform`), while `step_rotate_deg` sweeps a
prototype placed off the origin around it, so one `array` cannot
usefully do both; chain two, or transform after.
Mesh input -> Instances of that prototype. Instances input -> composed
set (count x k transforms, same prototype), so nested arrays form grids
without merging. The copies are numbered outer first: over an input
of k copies, copy `i` (the `name[i]` of a report) is this array's step
`i // k` of the input's copy `i % k`. Emits prototype + transforms,
never baked geometry. Deterministic.
Example: `examples/crate`.

## bake_instances (op_impl_version 1)

Recipe: `mo.bake_instances(a, id = ...)`

Explicitly merges its single Instances input into one Mesh: the
prototype baked at every instance transform. Vertices map through each
instance matrix, normals through its inverse-transpose (renormalized);
faces are the prototype's, index-offset per copy; UVs and extra
attribute slots are tiled per copy (preserved), except TANGENT
directions, which map through each instance's linear part like the
geometry they frame. No params. Instance
transforms must have positive determinant (guaranteed by
`array`/`scatter`/`transform` construction) so shared winding stays
CCW-outward; violations are an actionable error. A Mesh input is an
error. Merging is explicit, and a mesh needs no baking (nothing
merges silently; this op is the explicit opt-in).
Intended use: repeated cutters for `boolean` (bolt circles, window
openings). Delivery format is not a reason to use
this op: `moraine export --bake-instances` projects instanced outputs
to plain meshes at the export edge. Deterministic. Example:
`examples/bolt_plate`.
Example: `examples/bolt_plate`.

## scatter (op_impl_version 3)

Recipe: `mo.scatter(target, prototype, id = ..., count = int, seed = int, align_normal = ..., rand_yaw = ..., scale_jitter = ...)`

Scatters the `prototype` mesh as instances at random points on the
surface of the `target` mesh: where, then what, as `place` and `mate`
take theirs. `count`
(default 10, 1..1000000; raising it appends copies and leaves the
earlier ones where they were); `seed` (default 0, 0..2^63-1); `align_normal`
(JSON bool, default false): instance +Y aligns to the surface normal,
and a level target leaves the copy's yaw unchanged (the prototype keeps
its +Z front);
`rand_yaw` (JSON
bool, default false),
uniform random rotation about the instance's up axis; `scale_jitter`
`[lo, hi]` uniform random uniform-scale range (default `[1, 1]`);
the jitter scales the prototype as authored, so a transform applied
*before* the scatter (a lift of the prototype off its origin) scales
with each copy, while a transform applied to the scattered output
does not. Sampling is uniform by triangle area. Stochastic but deterministic:
streams derive as `hash(seed, node_id, index)`, so a node
reproduces exactly; two nodes with the same seed decorrelate. Output is
prototype + transforms; the target surface itself is not emitted. An
instanced target (an arrayed floor) is refused. Insert a
`bake_instances` node between the array/scatter node and this input.
Example: `examples/scatter_field`.

## noise_displace (op_impl_version 3)

Recipe: `mo.noise_displace(a, id = ..., amplitude = ..., frequency = ..., octaves = int, lacunarity = ..., gain = ..., seed = int)`

Displaces each vertex of its single mesh input along its vertex normal
by an fBm value-noise field sampled at the vertex position. `amplitude`
(default 0.05, peak about +/-amplitude); `frequency` cycles/meter (default 1);
`octaves` (default 4, 1..10); `lacunarity` (default 2); `gain` (default 0.5);
`seed` (default 0, 0..2^63-1). The field depends on position and `seed` only (not
the node id) so equal params reproduce the identical field. Decorrelate
via `seed`. A flat-shaded input (coincident duplicated vertices with
divergent normals, that is `box`, `boolean`/`bevel` output) would
displace the
duplicates apart and tear the surface along every hard edge, so it is
refused; weld the shading first with `smooth_normals`, raising
`angle_max_deg` above the hardest edge the displacement must cross
(its default 30 keeps hard edges split, which stay refused). Coincident
duplicates whose normals agree (UV seams) displace identically and
pass. Topology and UVs unchanged; stored normals kept as-is.
Deterministic across runs and platforms; params whose top octave would
overflow the noise lattice for the mesh's coordinate range fail with an
actionable `bad_param` instead of casting platform-dependently.
Example: `examples/scatter_field`.

## bevel (op_impl_version 4)

Recipe: `mo.bevel(a, id = ..., width = ..., segments = int, angle_min_deg = ...)`

Chamfers the sharp convex edges of its single watertight mesh input.
Precondition: a **convex** solid. Each chamfer is a global half-space
trim, so on a non-convex input (a solid with a bore, pocket, or step) a
plane slices unrelated geometry; that case fails `broken_result` below,
never flows on. `width` (default 0.01), distance cut back along each
adjacent face;
`segments` (default 1): v1 supports only 1 (flat chamfer);
`angle_min_deg` (default 30): only edges whose faces meet at more than
this normal angle are beveled. v1 cuts one plane per unique sharp convex
edge (manifold3d half-space trims): exact for convex solids and small
widths. At the default `angle_min_deg` a prism or cylinder keeps its
side edges (facet angle 360/`segments`) and only the rims are
chamfered; each rim plane then also cuts a little into the two
neighboring side facets, visibly on a coarse prism. Lower
`angle_min_deg` below the facet angle to chamfer the side edges too
(a 32-segment cylinder needs `angle_min_deg` < 11.25).
Corners resolve as the intersection of their edges' planes (a shallow
apex, not a flat corner triangle); concave edges are left untouched.
Structured failures: a non-watertight input fails `not_watertight`
(close the surface before bevel); `segments` other than 1 fails
`not_implemented` (multi-segment bevel is not implemented yet); a
`width` that consumes the whole solid fails `empty_result` (reduce
`width`). A knife edge (adjacent faces meeting at ~180 deg) has no
defined chamfer plane and fails with `degenerate_edge`. Remove the
sliver geometry first. A bevel matching no edge fails with `no_effect`
(a wrong `angle_min_deg` must not read as success; the message names
the sharpest convex edge angle). A deliberate pass-through is expressed
by deleting
the node. Post-conditions gate the result: the trims must leave a valid
manifold, no degenerate (sliver) triangles, and remove no more volume
than the chamfers account for; violating any fails with `broken_result`
(the non-convex failure, where a global plane slices unrelated
geometry). Reduce `width`, or bevel the convex pieces before the
boolean that made the shape. `segments > 1` is refused until a
multi-segment kernel lands behind this signature. Output is
flat-shaded; UVs are dropped.
Deterministic.
Example: `examples/crate`.

## unwrap (op_impl_version 2)

Recipe: `mo.unwrap(a, id = ..., mode = ..., uv_scale = ..., axis = ...)`

Assigns UVs to its single mesh input. `mode`, `"box"` (default) or
`"planar"`; `uv_scale` UV units per meter (default 1); `axis`,
planar projection axis `"x" | "y" | "z"` (default `"y"`). Box mode
projects each face along the dominant axis of its normal onto the two
other axes, signed so the texture reads upright and un-mirrored from
outside the surface under glTF's v-down texture origin: a side face maps
world +Y to smaller v, so a map with an intrinsic up (weathering running
down a wall) keeps it;
vertices shared between differently-projected faces split, others stay
shared. Planar mode projects every vertex along one axis; no splits.
Projection UVs overlap by construction, fine for tiling materials.
What `box` and `planar` return is fixed; a packed atlas arrives as a
further `mode`.
Geometry is otherwise unchanged:
the face list is untouched (box mode splits vertices at projection
seams only), so face groups survive `unwrap`. Tag before or after
it, either works.
Planar mode degenerates on faces parallel to the projection axis
(their UVs collapse to a line, invisible until a normal or color map
stretches across them). Box is the default for closed solids for that
reason; planar suits sheets facing one axis. Deterministic. Caveat for
curved surfaces: box mode switches projection direction where the
dominant normal axis flips, so on a doubly-curved surface (dome,
sphere) adjacent faces sample different tile regions. The phase
mismatch shows as seams in **structured** maps (checker, brick,
streaks, any tileable pattern with a visible period); isotropic
noise-derived roughness and normal maps have no phase to break and
read fine in practice. Verify from two azimuths. Until a curved
projection mode exists, keep structured maps off such surfaces
(factor or noise-only materials), and do not read a seam seen under a
`checker_texture` test as a defect of the real material. Structure
that varies along v only (horizontal bands, weathering by height)
survives every side-projection switch (v maps to world +Y on each
side face), so a v-only map is legitimate on curved sides; only
u-phased structure breaks there, and +/-Y-facing faces map v to world z.
Example: `examples/crate`.

## asset_ref (op_impl_version 2)

Recipe: `mo.asset_ref(asset, output, id = ..., overrides = ..., material_rename = ...)`

Places a named output of another asset in this graph. Zero
inputs. `output` names an output of the referenced asset, located by
exactly one of two params: `asset`, an in-project path of the form
`subassets/<name>.star`, forward slashes only (e.g.
`"subassets/water_tower.star"`, which the IR names as the `.mora` the
build wrote; the workspace layout is enforced,
so top-level, nested, or traversing targets are rejected), or
`project`, a sibling project's directory name inside the same
workspace (a name from the discovery roster, never a path), targeting a
named output of that project's *root* asset (the IR names that asset's
own IR beside `project`, as `"asset": "<name>.mora"`, which the build
writes and a recipe never does). A project's root outputs
plus its `params` declarations are its whole public interface:
`subassets/` stay project-private, and reuse across repositories stays
vendoring (copy into `subassets/` and own it). The result is exactly
the referenced output's value, Mesh or Instances alike. Instancing
passes through untouched. The referenced asset evaluates as its own
graph in its own namespace, and the reference graph must be a DAG
(cycles rejected with the file chain, across projects too).
Optional `overrides`, an object mapping the *directly* referenced
asset's declared param ids (its root `params` list, the same
declarations that drive the viewer panel) to per-use values. The
referenced asset is built again under the overridden values, under the
declaration contract (`param_unknown` / `param_type_mismatch` /
`param_out_of_range`: a value must sit inside the declared min/max;
the range is the author's validity promise, widen the declaration if
the design truly needs more), so
differently-overridden uses of one asset are distinct evaluations with
their own cache entries, while cache keys still follow content only. A
knob declared deeper in a reference chain is out of reach by design.
The asset in between forwards it by declaring a knob of its own and
passing the value on, `overrides = {"width": mo.param("width", ...)}`.
An un-overridden asset_ref is an identity over the referenced output,
so its cached record is the referenced output's record: the location
params never enter a cache key, and editing the referenced asset
dirties every referencing composite. A referenced asset's `asserts`
are checked by every consuming root's `validate` (and `export`), in
the sub-asset's own namespace and local frame, once per override
identity. Entries carry `asset` (and `overrides`) and gate `ok`
(green ones fold to a `ref_assets` count row per identity), so
authoring asserts inside a kit keeps them gating wherever the kit is
used.
Example: `examples/kit_variants`.

### material_rename (asset_ref param)

Optional `material_rename`, an object mapping a material name carried
by the received output to this use's own name (e.g.
`{"panel_paint": "panel_paint_worn"}`): the consumer-side per-use
material identity. Overrides that reach a material change its content
while its name stays the kit's, so two uses of one kit with different
texture/factor overrides would bind one name to two contents, a
`material_name_unique` failure the consumer cannot repair in the
referenced file. The rename is applied to the *received value* at the
reference boundary (factors and textures untouched; only the exported
identity changes), reaches materials arriving through nested
references too, and never touches the referenced asset or its override
identity. A renamed ref is its own cached record over the referenced
output's record, while un-renamed uses keep aliasing it. Every key must
name a material the received output actually carries (a Mesh or
Instances carries one, so the map is effectively one entry; an absent
name, an unmaterialed output, or an image output fails
`material_rename_unknown` listing what is present); the new name is a
portable identifier, never `moraine_default`, never the old name, and
distinct per mapping. A renamed identity is authored by the consuming
asset: it counts toward that asset's `budgets.materials`. Deterministic.
Examples (for `asset_ref` as a whole): `examples/tower_yard` places a vendored
water tower twice, the second tower overriding a declared tank radius,
two towers from one file and one knob; `examples/kit_variants` places
one panel kit twice, the second use overriding its wear seed and paint
roughness under `material_rename`, with a declared material budget.
Example: `examples/kit_variants`.
