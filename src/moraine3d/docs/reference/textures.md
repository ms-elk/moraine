# Texture & material

Texture nodes emit an `Image` value, `(h, w, 1|3)` float32, **linear**
values (sRGB encoding happens only at the GLB/PNG edge, per material
slot). Images cache uniformly with meshes, are pure (no node-id salt:
identical params share one cache record), and are seamlessly tileable
unless a paragraph says otherwise. An image named as an asset output
gets a preview PNG in the viewer (sRGB-encoded for display) but is not
an exportable part. Feed it into `material` and drop it from `outputs`
before export.

## noise_texture (op_impl_version 1)

Recipe: `mo.noise_texture(id = ..., seed = int, size = int, scale = int, octaves = int, gain = ..., low = ..., high = ..., mode = ...)`

Emits a 1-channel fBm value-noise image, mapped linearly from the
normalized fBm range [-1, 1] to `[low, high]` (defaults 0 / 1), tileable
in u and v. `seed` (default 0, 0..2^63-1); `size` int or `[w, h]`
pixels, 4..2048 (default 256); `scale` integer lattice cells across
the tile (default 4, 1..2048: beyond one cell per pixel everything is
sub-pixel); `octaves` (default 4, 1..10); `gain` per-octave amplitude
falloff (default 0.5, > 0). `mode` (default `"fbm"`) selects the
per-octave shaping applied to each octave value `n` in [-1, 1] before
summation: `"fbm"` is `n` unchanged; `"ridged"` is `1 - 2|n|` (sharp
bright creases at lattice zero-crossings: rock, worn metal);
`"billow"` is `2|n| - 1` (rounded bright lumps). Every shaped octave
stays in [-1, 1], so the normalized sum and the exact `[low, high]`
mapping hold in all modes, and tiling is unchanged. The default output
is bit-identical to this op before `mode` existed (additive param,
same op_impl_version). Lacunarity is fixed at 2 so every octave's
lattice stays integer-periodic (that is what keeps the tile edge
seamless). Deterministic and pure. Example: `examples/steel_door`.

## streaks (op_impl_version 1)

Recipe: `mo.streaks(id = ..., seed = int, size = int, count = int, length = ..., width = ..., wiggle = ..., low = ..., high = ...)`

Emits a 1-channel image of vertical weathering streaks (rust wash, drip
stains), tileable in u and v. `count` streaks (default 12, 1..64), each
at a hashed u position, run down +v from a hashed head with a short
fade-in and a linear fade-out over the streak length; direction is
fixed vertical, and +v is downward on a surface `unwrap` textured, so
the drips run down the part. This is per face under box projection,
so a multi-face body needs one shared direction (see Patterns).
Background pixels are exactly `low`; streak cores approach `high`;
overlaps saturate at exactly `high` (defaults 0 / 1. Swap them for dark
streaks on a light map). All pixels lie in
[min(low, high), max(low, high)]. `length` maximum streak length as a
fraction of the tile (default 0.5, in (0, 1]; per-streak jitter
0.5..1x); `width` maximum streak width as a fraction of the tile
(default 0.02, in (0, 0.5]; per-streak jitter 0.7..1x); `wiggle`
horizontal meander amplitude in tile fractions (default 0.01, 0..0.25;
0 = straight); `seed` / `size` as in noise_texture. Deterministic and
pure. Example: `examples/steel_door`.

## checker_texture (op_impl_version 1)

Recipe: `mo.checker_texture(id = ..., size = int, tiles = int, color_a = ..., color_b = ...)`

Emits a 3-channel checkerboard: `tiles` x `tiles` squares (default 8,
1..2048: beyond one tile per pixel nothing is resolved) alternating
`color_a` (the square containing u = v = 0, default `[1, 1, 1]`) and
`color_b` (default `[0, 0, 0]`), linear RGB. `size` as in
noise_texture. An even `tiles` repeats seamlessly. Primary use:
verifying UV layout and texel density before real materials. Judge it
from a face-on view (`--view front` or a free camera square to the
surface): at oblique angles correct squares foreshorten into stripes
and read as broken UVs. Deterministic and pure. Example:
`examples/uv_check`.
Example: `examples/uv_check`.

## brick_texture (op_impl_version 4)

Recipe: `mo.brick_texture(id = ..., seed = int, size = int, rows = int, columns = int, mortar = ..., offset = ..., variation = ..., chamfer = ..., brick_color = ..., mortar_color = ..., map = ...)`

Emits a brick pattern, running bond by default, tileable (with odd
`rows` the top course meets course 0 across the v seam in the same
phase: an odd cycle cannot alternate). `map` selects
`"color"` (default; 3-channel: per-brick-varied `brick_color` on faces,
`mortar_color` in joints, blended across the chamfer ramp. The color
follows the same ramp as the height, which is what keeps the two
outputs pixel-aligned) or `"height"` (1-channel: 1 on brick faces
falling to 0 in the joints, to feed `normal_map`). Color and height nodes
with identical params are pixel-aligned. `seed` (default 0, 0..2^63-1)
drives per-brick value variation (multiplier `1 - variation *
U01(brick)`, `variation` default 0.15, 0..1); `rows` courses per tile
(default 6, 1..256); `columns` bricks per course (default 3, 1..256);
`mortar` joint thickness as a fraction of a course height (default
0.08, 0..0.9), equally thick in both directions on a square tile;
`offset` shift of each course relative to the one below, in brick
widths, taken modulo one brick (default 0.5 = running bond for every
`columns`; 0 = stack bond); `chamfer` height ramp width as a fraction
of the joint thickness (default 0.5, 0..1; 0 = hard step);
`brick_color` (linear RGB, default `[0.45, 0.18, 0.12]`) and
`mortar_color` (default `[0.78, 0.76, 0.72]`); `size` as in
noise_texture. Deterministic and pure. Example: `examples/brick_shed`.

## colorize (op_impl_version 1)

Recipe: `mo.colorize(a, id = ..., color_low = ..., color_high = ...)`

Maps its 1-channel image input through a two-color linear gradient:
`out = color_low + clamp(p, 0, 1) * (color_high - color_low)`; emits a
3-channel image the same size. `color_low` (default `[0, 0, 0]`) and
`color_high` (default `[1, 1, 1]`), linear RGB. Typical use: noise into a
concrete / grime color map. Deterministic and pure. Example:
`examples/brick_shed`.
Example: `examples/brick_shed`.

## ramp (op_impl_version 1)

Recipe: `mo.ramp(a, id = ..., stops = ..., values = ...)`

Maps its 1-channel image input through a multi-stop piecewise-linear
gradient, the N-stop generalization of `colorize`. `stops` two or
more positions in 0..1, strictly increasing (default `[0, 1]`);
`values` one entry per stop: all numbers (emits a 1-channel image,
grayscale contrast shaping for height / roughness maps) or all
`[r, g, b]` linear colors >= 0 (emits 3-channel). Input clamps to
[0, 1]; input at or below the first stop takes the first value, at or
above the last stop the last value; between stops, linear
interpolation. Output is the input's size; a tileable input yields a
tileable output. Deterministic and pure. Example: `examples/steel_door`.

## normal_map (op_impl_version 1)

Recipe: `mo.normal_map(a, id = ..., strength = ...)`

Converts its 1-channel height input into a tangent-space normal map:
3-channel, unit normals encoded `n * 0.5 + 0.5`, **+Y-up ("OpenGL")
convention as glTF requires**. Green brightens on the upper (smaller v)
flank of a bump. Gradients are wrap-around central differences, so a
tileable height map yields a tileable normal map. `strength` (default 1,
>= 0) is the height amplitude in UV-tile units, so the tilt it gives
depends on how sharp the height is: a full step across one pixel
tilts the normal by atan(`strength` x size / 2). On a 256-pixel
`brick_texture` height the mortar flanks reach 45 deg at `strength`
0.008 and 79 deg at 0.04; beyond that the flank normals lie almost
in the surface, and neither a larger `strength` nor the material's
`normal_scale` changes the picture. Hard-edged heights (mortar, plank
grooves) want `strength` near 2 / size; smooth noise takes the larger
values. Deterministic and pure.
Example: `examples/brick_shed`.

## material (op_impl_version 1)

Recipe: `mo.material(a, base_color_texture = node, roughness_texture = node, metallic_texture = node, normal_texture = node, id = ..., name = ..., base_color = ..., metallic = ..., roughness = ..., emissive = ..., normal_scale = ...)`

Binds a glTF 2.0 metallic-roughness material to its mesh (or instanced)
input 0; geometry is untouched and the material rides through cache,
viewer, and export. Each texture image is passed by the keyword of
the slot it fills: `base_color_texture` (3ch), `roughness_texture`
(1ch), `metallic_texture` (1ch), `normal_texture` (3ch); the keyword
without `_texture` is the factor. Roughness and metallic must match
in size: they pack into the glTF metallicRoughness texture G/B
channels at the GLB edge. Texture values multiply factors,
exactly as in glTF, and are quantized to 8 bits at export, silently
clipping values above 1.0; for a darkening multiplier map, set the
brightest value to 1.0. `name` is required and is the exported material
identity: identical content (name + factors + textures) dedups into
one glTF material (one draw state shared by every part binding it),
one name with two contents fails plain `validate`
(`material_name_unique`, gating; the GLB writer refuses it too), and
`moraine_default` is reserved. A name arriving through `asset_ref` is
given a per-use identity with the ref's `material_rename` (see
*asset_ref*); an asset caps the identities it authors with
`mo.budget` (see *budgets*). A knob on a factor of a material shared
by name goes to every use of the name, since moving it on one part
makes one name two contents (`material_name_unique`). Such a knob is
a staged field in the viewer, not a live one (see *knobs*). To move
one part alone, give its use a distinct name. Factors: `base_color` linear RGB(A) in
0..1 (default `[1, 1, 1, 1]`; an alpha factor below 1 exports with glTF
`alphaMode: "BLEND"`, so the GLB renders the transparency the factor
declares); `metallic` (default 0.0) and `roughness`
(default 0.8) in 0..1 (authoring defaults match the neutral material,
not glTF's fully-metallic default); `emissive` linear RGB >= 0 (default
0; components above 1 export normalized to the unit range with the
intensity in `KHR_materials_emissive_strength`, so an importer that
honors the extension renders the authored strength and one that does
not still sees the color); `normal_scale` >= 0 (default 1). Single-input geometry ops whose
`MATERIAL_POLICY` (default: their `ATTRIBUTE_POLICY`) is "preserved"
carry the material downstream; topology-rebuilding ops drop it with a
warning. Prefer binding the material as the last node of a part.
Deterministic and pure. The call and the node it builds (the IR lists
the textures after the mesh, with the slots they fill under `slots`,
which the build writes and a recipe never does):

```python
mo.material(tank_uv, base_color_texture = color_map, roughness_texture = rough_map, id = "tank_mat", name = "tank_paint", metallic = 0.15, roughness = 0.85)
```

```json
{"id": "tank_mat", "op": "material", "inputs": ["tank_uv", "color_map", "rough_map"], "params": {"name": "tank_paint", "slots": ["base_color", "roughness"], "metallic": 0.15, "roughness": 0.85}}
```
Example: `examples/shelf_pair` (unwrap, then material, then array on each
prototype), `examples/uv_check`.

## Patterns (texture legibility)

Authoring patterns for surfaces that read wrong with naive texturing.
Consult before texturing a large surface. (The ops referenced are
documented in their own sections.)

- **Repeated coplanar members** (shutter slats, planks, panel rows)
  read as one flat sheet when their faces are flush. Give them relief:
  alternate member depth by a few percent of the member height (two
  interleaved arrays at different z, 2 mm on 100 mm slats). A cheaper
  substitute is a `brick_texture` `{"map": "height", "columns": 1}`
  course map fed to `normal_map`, but each course keeps one vertical
  end joint (staggered at the default `offset`, aligned at
  `offset: 0`), so use it only where panel ends are plausible, never
  for a seamless surface. Judge the front screenshot at viewing
  distance, not zoomed in.
- **One slot, one source.** A `material` slot takes a single image:
  `base_color` cannot combine a grime map and a paint map, and `normal`
  carries either micro-relief or a structural seam, not both. Choose
  by the surface's character and move the other signal to `roughness`
  (there is no in-graph mix op). Factors multiply the texture, so a
  `base_color` factor can only darken a map: per-part brightness
  differences come from the color source (a second `colorize`), not
  from the factor.
- **Flat-factor steel reads as plastic at distance.** Give large metal
  surfaces at least a roughness map (`noise_texture`
  `mode: "ridged"` for worn metal, contrast shaped with a scalar
  `ramp`) and a low-contrast `base_color` map (`noise_texture` ->
  `ramp` with color stops); `streaks` makes a rust-wash roughness map,
  or a grime `base_color` via `colorize`. Painted or coated steel is a
  dielectric: `metallic` ~0.1 to 0.2 (high `metallic` without an
  environment map renders dark and kills texture reading, so reserve
  it for bare polished metal). Example: `examples/steel_door`.
- **Structured maps belong on surfaces `unwrap` can cover without
  phase seams.** Box-projection UVs on a doubly-curved surface (dome,
  sphere) put tile-phase discontinuities at every projection switch,
  visible in the viewer and engine alike, and easy to misread as
  texture noise at acceptance (see the `unwrap` caveat). On such
  surfaces keep to factors and isotropic noise-derived maps (roughness,
  fine normal detail: no period to break) until a curved projection
  mode lands, with one condition: an isotropic source read through
  UVs still changes *apparent scale* where the projection switches,
  so keep its amplitude low (subtle roughness variation, not bold
  blotches) or the step in grain size marks the seam anyway. A seam
  under a `checker_texture` test does not by itself
  condemn the real material.
- **Directional maps need one direction.** `streaks` runs along v and
  box projection picks v per face, so a box-projected body streaks a
  different way on its floor, its sides, and its back: three drip
  directions on one bucket. Directional wear is authorable only where
  the textured faces share one projected direction (a planar panel,
  one face of a box); elsewhere use isotropic ridged noise, which has
  no direction to disagree about. A prototype whose long faces must
  all carry the grain can be unwrapped standing on its long axis and
  then rotated flat: UVs ride the transform, so the streaks run along
  the length on every long face. Decide this before choosing the
  map, not at the screenshot.
- **Grain is a stretched isotropic map.** No texture op stretches
  along one axis, and `uv_scale` is one number, so wood grain
  comes from the UVs: scale the member to 0.1 along its length,
  `unwrap` it (box), and scale it back by 10. UVs ride the
  transform, so an isotropic `noise_texture` then runs ten times
  longer along the member on every face parallel to its length,
  and one material serves posts, rails, and rafters alike. Shrink
  the member's own long axis, before it is rotated into place; the
  wrong axis runs the grain across, and only a shot shows it.

## Variation-grid (seed-sweep recipe)

When the open question is taste over a stochastic stream ("show me a
few options"), sweep the seed instead of describing candidates. A
human at the viewer with a declared seed knob needs no grid. Its
control rerolls with history, the cheaper path; sweep when the ask
arrives in chat, the seed is undeclared, or a side-by-side comparison
is wanted.

- Pick 4 to 6 seeds spread across the declared range; for each, edit
  the seed in place and `screenshot` with one fixed camera (same view,
  target, and size for every shot, thumbnail-sized), naming each shot
  by its seed.
- Present the set as one contact sheet (compose a single image where
  your environment can; otherwise the labeled thumbnails in one
  message); the pick comes back in chat and lands as an ordinary
  one-line edit.
- Never leave the file on a mid-sweep seed: finish on the pick or the
  original.
- Cost is one evaluation and one tier-3 shot per seed. Keep the grid
  small and the thumbnails low-res. Record chosen and rejected, and
  what the human's adjectives turned out to mean, per the journal
  rules in the guide.
