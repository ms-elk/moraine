# Shipped example assets

## Examples

The shipped example projects. `moraine docs examples/<name>` prints
one's root recipe, `examples/<name>/requirements` the human's file
beside it, `examples/<name>/subassets/<sub>` a sub-asset, and an
op's section prints the recipe its `Example:` line cites. Each is a
complete project authored from a one-paragraph request. To build or
serve one, copy its directory into a workspace: the package tree is a
delivery, and a build writes beside the recipe.

| project | demonstrates | operators |
|---|---|---|
| `box` | The smallest asset: one box resting on the ground plane. Start here. | box, transform |
| `uv_check` | Checker texture on a box-unwrapped box, for reading the UV convention (v-down, top-left origin) in the viewer. | unwrap, checker_texture, material |
| `crate` | Bevel -> box unwrap -> nested instanced arrays: a 3 x 2 crate kit from one prototype. | bevel, unwrap, array |
| `pallet` | Stringer pallet, every board instanced. | box, array |
| `flange_pipe` | Through-bored pipe with two bolted flanges, one watertight solid; length and flange diameter asserted (`extent`). | cylinder, boolean |
| `stairs` | Straight flight, instanced steps, stringers cut flush to the ground; rise, width, and riser count asserted (`extent`, `count`), and every step bearing on the stringers by face (`contact` with `min_area`). | array, boolean |
| `arcade` | Wall with three semicircular arches and a coping band, one watertight mesh. | boolean, cylinder |
| `water_tower` | Tank with a hemispherical dome on a rotational array of tilted legs, braced, with an instanced-rung ladder. | sphere, array, boolean |
| `brick_building` | Boolean opening cuts, unioned trim, instanced sills and lintels. | boolean, array |
| `bolt_plate` | Eight bolt holes: rotational array -> `bake_instances` -> boolean difference. | array, bake_instances, boolean |
| `pipe_tee` | Two cylinders unioned, then angle-threshold auto-smoothed. | boolean, smooth_normals |
| `turned_column` | Revolved column profile; the capital tagged on the prototype, a colonnade of four, an architrave placed on copy 0 and asserted to rest on every column. | revolve, tag, array, place |
| `brick_shed` | Hollow brick walls placed on a concrete slab by its tagged top face; procedural brick albedo with a height-derived normal map, noise-based concrete; declared parameters. | brick_texture, normal_map, colorize, noise_texture, place, material |
| `steel_door` | Louvred steel door in a frame; painted steel from a ramped noise and streak roughness; every slat asserted to be held by the frame. | ramp, streaks, noise_texture, material, array |
| `anchored_crate` | A crate placed on a pallet's tagged deck, a smaller crate on its lid; both rests asserted as `contact`. | tag, place |
| `shelf_row` | Items placed on one copy of an instanced shelf by `instance` index. | tag, place, array |
| `wall_duct` | A duct mated to a wall port face-to-face, with a flange riding the same solve; `mated` and `contact` asserted. | tag, mate, boolean |
| `tower_yard` | A vendored water tower (`subassets/`) placed twice from one file, with one knob. | asset_ref, transform |
| `kit_variants` | One panel kit placed twice with parameter overrides and a per-use material identity, under a material budget. | asset_ref (overrides, material_rename) |
| `scatter_field` | A displaced grid extruded and cut into a ground slab, forty rocks scattered on it: every rock asserted to touch the ground, the sinks and the rock overlaps declared as free-overlap embeds, the count pinned. | grid, noise_displace, extrude, boolean, sphere, scatter |
| `shelf_pair` | Two shelves, each a pair of capped posts and two boards (unwrap -> material -> array on both prototypes), named as output groups: each shelf kept clear of everything outside it, the shelves apart, every board end mated `contained` to its post, the cap height read from a tag only the cap can satisfy. | box, boolean, tag, unwrap, material, array |
