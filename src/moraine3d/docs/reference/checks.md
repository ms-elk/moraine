# Built-in checks

## checks (built-in validate checks)

The contract of every check `validate` runs with no entry in the
file. Per check: criterion, waivability, how an instanced output is
quantified, and whether the entry appears when green. Asserts
(`mo.assert_<kind>`, gating the same way) and budgets have their own
sections. Per output, always listed, green or red:

- `finite`: no NaN/inf anywhere in the value: vertices, normals, UVs,
  extra attribute slots, instance transforms, material factors and
  texture pixels (an image output: its pixels). Not waivable. Every
  other check on a non-finite output is skipped.
- `no_degenerate_tris`: no triangle whose height over its longest
  edge falls below 1e-9 (relative to the triangle, so a pass survives
  `asset_ref` transforms); usually exactly-coplanar boolean operands,
  duplicate points, or a collapsed scale. Not waivable. Prototype only
  (copies do not change shape).
- `watertight`: the surface is closed after welding coincident
  vertices. Not waivable. Prototype only.
- `scale_sane`: the largest dimension lies in [0.001, 1000] m (an
  instanced output: over all copies). Not waivable.
- `instancing_intact`: instanced outputs only, always green with the
  copy count: instancing reached the output unmerged.
- `not_sunk`: the true minimum y over every copy's actual vertices is
  no lower than 1e-3 m below the ground plane. Waivable per output
  (`mo.waive("<output>", "not_sunk")`) for deliberate
  below-ground geometry only.

Asset-level (`asset_checks` in the report; one entry when the asset
has geometry outputs):

- `grounded`: the lowest geometry output reaches the ground plane
  within 1e-3 m, from above or below. Waivable at the asset level
  (`mo.waive_asset("grounded")`) for support outside the
  asset.

Listed only on failure, waiver, or an unquantified copy:

- `not_enclosed`: the output does not lie entirely inside another
  single-mesh output's solid that is finite, watertight, and opaque
  (base-color alpha exactly 1.0; textures carry no alpha). Containment
  is solid, by kernel volume (inner intersected with container =
  inner), never a bounds test, so a part that merely overlaps, or sits
  in a hollow, passes here (interpenetration is the overlap guard's).
  An open inner surface cannot be built as a solid and is skipped.
  That is under-reporting, never a false fail. An instanced output is
  an inner candidate only, judged over all its copies together: it
  fails when every copy is enclosed; when some copy's box lies inside
  a container's box without a failure, the green entry names the
  output under `unchecked.instanced_copies`. Per-copy enclosure is not
  tested. Waivable per output (`mo.waive("<output>", "not_enclosed")`) for an
  intentional interior part.
- `material_name_unique`: one material name bound to one content
  (name + factors + textures) across the asset's outputs; listed on
  every colliding output, the sides named with their route (authored
  here, or via which `asset_ref` node) and either fix offered. Not
  waivable: the GLB writer's hard error, mirrored.

Advisory `warnings` (never gating; root-asset nodes only. A node
inside a referenced asset is the producer's edit site and its own
validate's finding, never re-raised in a consumer):

- `attributes_dropped` / `groups_dropped` / `material_dropped`: a
  topology-rebuilding op dropped slots, face groups, or a material an
  input carried **and the loss reaches an output** (still absent at
  the outputs named); a loss re-authored downstream, or a value that
  only cuts or anchors, is not reported.
- `flat_shading`: a boolean output with curved surface (an edge
  where `smooth_normals` at its default 30 deg would change a normal)
  reaches this material with no shading-replacing op in between; an
  all-planar boolean output shades identically either way and is not
  reported.

Beyond the per-output entries the report carries: `moraine_version`,
the tool that produced the report (leading the object; the export
manifest and the `screenshot` report carry the same key, so two
installs can never be confused through their answers); `mates`, one row
per root `mate` node with the origin provenance per role (see *mate*);
`ref_assets`, one row per referenced (asset, override identity) with
`asserts_total` / `asserts_ok`, the fold of that bundle's green assert
entries (see *Asset assertions*), and `params_total` / `params_ok`
when the referenced asset declares ranged knobs; `params`, one
gating entry per declared knob that carries a range, or an accepted
value (`param`, `ok`,
`value`, `min`, `max`, `from` for a value accepted in the viewer or
set by a consumer, and `fix` on failure: a value outside the range,
or a value in `params.json` of another kind than the knob's, shown
as `mismatch`), a referenced asset's failing knobs included with
their `asset`; `by_site`, the failing entries counted by the line
they share, and by `via` where they came through more than one call, so
one wrong expression in a loop reads as one row; `consumers`, the
projects that reference this one (see *build*); `serve`, the running
server's file-watch liveness (`watch: "alive" | "dead"`, with its
`port` and `project`) when a `moraine serve` on `--port` serves this
root.
`--failures-only` keeps counts, failing entries, `warnings`, `mates`,
`ref_assets`, `by_site`, `consumers`, and `serve`, and counts every
green entry's honesty key (`unchecked`, `ceded`, `stronger`) at the
top level under that key: `entries`, and by name what they hold, as
in `"ceded": {"entries": 1, "pairs": 312}`. The rows are in the
report without the flag. A
workspace sweep nests each project's report and adds `not_swept`: the
asset files below the project level the sweep did not reach.
Every entry that names an output, an assert, or a knob carries its
`site`, the recipe file and line to edit, with `via` where it was
made inside a function (see *ids*), and `--failures-only` keeps both
on each failing entry.
`--param ID=VALUE`, repeatable, judges one project under the values
given for its declared knobs, over an accepted value, and writes no
file: neither `params.json` nor the IR, and the viewer does not move.
The report names them under `params_given`. A value outside the
declared range is judged too: its `params` entry fails (the report is
red), and the other entries say what the shape does there, which is
how to find where a range should end without editing it. VALUE is JSON
(`2.4`, `true`, `[1,2,3]`), and for a string knob the text itself.
An id given twice is refused.
`--profile <name>` appends the export profile's own checks, whose
`level: "warn"` entries are advisory.
