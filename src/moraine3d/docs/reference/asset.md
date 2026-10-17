# Asset format

The shipped reference: `moraine docs` lists it by file and section,
`moraine docs <section>` prints one section's contract together with
the example asset its `Example:` line cites (the index: *Examples*).

## A complete minimal asset

The IR is what a recipe builds to (*recipe*) and what every command
evaluates: a JSON operator graph at `build/<name>.mora`, written by
the build and read, never edited. This one is the build of the recipe
shown under *recipe*, and it passes `validate`:

```json
{
"build_version": "5+starlark-pyo3.2026.1.2+e540e009",
"nodes": [
{"id": "slab_box", "op": "box", "params": {"size": [2.0, 0.1, 2.0]}, "site": "dock.star:5"},
{"id": "slab", "op": "transform", "inputs": ["slab_box"], "params": {"translate": [0.0, 0.05, 0.0]}, "site": "dock.star:4"},
{"id": "slab_top", "op": "tag", "inputs": ["slab"], "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}, "site": "dock.star:9"},
{"id": "crate_box", "op": "box", "params": {"size": [0.4, 0.4, 0.4]}, "site": "dock.star:12"},
{"id": "crate", "op": "place", "inputs": ["slab_top", "crate_box"], "params": {"anchor": "top", "mode": "on_top", "offset": [0.3, 0.0, 0.0]}, "site": "dock.star:10"},
{"id": "bollard_box", "op": "box", "params": {"size": [0.1, 0.5, 0.1]}, "site": "dock.star:19"},
{"id": "bollard_a", "op": "transform", "inputs": ["bollard_box"], "params": {"translate": [-1.2, 0.25, 0.0]}, "site": "dock.star:18"},
{"id": "bollards", "op": "array", "inputs": ["bollard_a"], "params": {"count": 2, "step_translate": [2.4, 0.0, 0.0]}, "site": "dock.star:23"}
],
"params": [
{"id": "bollard_count", "value": 2, "node": "bollards", "param": "count", "label": "Bollards", "min": 1, "max": 6, "step": 1, "site": "dock.star:2"}
],
"outputs": {
"slab": "slab_top",
"crate": "crate",
"bollards": "bollards"
},
"asserts": [
{"id": "crate_on_slab", "kind": "contact", "outputs": ["slab", "crate"], "owner": "agent", "site": "dock.star:28"}
]
}
```

Root keys in canonical order `build_version, nodes, params, outputs,
groups, checks, asset_checks, asserts, budgets, provenance`, an
absent section omitted. Every node line is `{"id", "op", "inputs",
"params", "site", "via"}` in that order. Zero-input ops omit
`"inputs"`, while `"params"` is always present. `site` is the recipe
file and line of the call that made the entry, the line to edit, and
`via`, on an entry made inside a function, the calls on the way to
it. A number is written in its param's type: a float param reads
`2.0` however the recipe wrote it.
`build_version` names the build that wrote the file. A file carrying
another stamp is stale and is refused (`version_mismatch`), and
`moraine build` writes it again.

Layout: every section that gains an entry per output holds one entry
per line. Those are the lists (`nodes`, `params`, `asserts`) and the
objects `outputs`, `groups` (optional: a piece built from several
outputs, named once for the asserts, see *groups*), and
`checks` (output name -> waivers, `false` the only value).
`asset_checks` (see *checks*) and `budgets` (see *budgets*) are
inline. An op's input order and param names are its own section's
contract, printed with the asset that uses it.

`params` entries: one per knob the recipe declares (see *knobs*),
holding the knob's `id`, the `value` the build resolved for it, the
declared `label`, `min`, `max`, and `step`, and its `site`. `from`
names the source of a value that is not the recipe's default,
`accepted` (the project's `params.json`) or `override` (an asset that
references this one, or a preview), and `default` then holds the
recipe's. `mismatch` holds a value of
`params.json` the knob could not take. An entry that also carries
`node` and `param` records that the value is exactly that node's
param (see *knobs*).
