# Recipes

## recipe (the asset as a Starlark file)

An asset is written as a **recipe**, a Starlark file. Starlark keeps
Python's syntax and drops what makes a program unpredictable, so a
recipe reads no file, no clock, and no random source, and the same
text always builds the same asset. Every command builds the recipe
first and evaluates the IR the build wrote (format: *A complete
minimal asset*).

| file | holds | owner |
|---|---|---|
| `<name>.star` | the root asset, one per project, its name free | agent |
| `<name>.requirements.star` | that asset's requirements (see *requirements*) | human |
| `lib/<name>.star` | definitions only: functions and constants (see *load*) | agent |
| `subassets/<name>.star` | a part of the project, placed with `asset_ref` | agent |
| `subassets/<name>.requirements.star` | that part's requirements | human |
| `build/` | the IR, written by the build | the tool |

A complete recipe, which passes `validate` as it is:

```python
# dock/dock.star
count = mo.param("bollard_count", 2, min = 1, max = 6, step = 1, label = "Bollards")

slab = mo.transform(
    mo.box(id = "slab_box", size = [2.0, 0.1, 2.0]),
    id = "slab",
    translate = [0, 0.05, 0],
)
slab_top = mo.tag(slab, id = "slab_top", name = "top", select = {"normal_y": {"gt": 0.9}})
crate = mo.place(
    base = slab_top,
    part = mo.box(id = "crate_box", size = [0.4, 0.4, 0.4]),
    id = "crate",
    anchor = "top",
    mode = "on_top",
    offset = [0.3, 0, 0],
)
bollard = mo.transform(
    mo.box(id = "bollard_box", size = [0.1, 0.5, 0.1]),
    id = "bollard_a",
    translate = [-1.2, 0.25, 0],
)
bollards = mo.array(bollard, id = "bollards", count = count, step_translate = [2.4, 0, 0])

mo.output("slab", slab_top)
mo.output("crate", crate)
mo.output("bollards", bollards)
mo.assert_contact(id = "crate_on_slab", outputs = ["slab", "crate"])
```

**Reading the rest of the reference.** Each section opens with the
recipe spelling of its call; the entries and reports it then
describes are in the IR's spelling, and each IR entry is one recipe
call:

| IR | recipe |
|---|---|
| node `{"id": "crate", "op": "place", "inputs": ["slab_top", "crate_box"], "params": {"anchor": "top"}}` | `mo.place(base = slab_top, part = crate_box, id = "crate", anchor = "top")` |
| `"outputs": {"crate": "crate"}` | `mo.output("crate", crate)` |
| `"groups": {"shelf": ["frame", "posts"]}` | `mo.group("shelf", ["frame", "posts"])` |
| assert `{"id": "rests", "kind": "contact", "outputs": ["slab", "crate"], "every": ["crate"], "owner": ...}` | `mo.assert_contact(id = "rests", outputs = ["slab", "crate"], every = ["crate"])`, in the file of its owner |
| `"checks": {"slab": {"not_sunk": false}}` | `mo.waive("slab", "not_sunk")` |
| `"asset_checks": {"grounded": false}` | `mo.waive_asset("grounded")` |
| `"budgets": {"materials": 2}` | `mo.budget(materials = 2)`, in the requirements file |
| a `params` entry | `mo.param(...)` (see *knobs*) |

JSON values are written as Starlark literals: `true` is `True`,
`false` is `False`, and a key whose value would be `null` is left out.

### builtins (what `mo` and `math` hold)

`mo` and `math` are in scope in every recipe file with no `load`.
`mo.` marks exactly what reaches the IR. A call without it is recipe
arithmetic. `moraine docs api` lists every name under both with its
arguments.

- **Operators.** `mo.<op>(inputs..., id = ..., <param> = ...)` for
  every operator of the reference, under the IR's own name. Inputs
  are the nodes other operator calls returned, by position in the
  contract's order. Where an operator's inputs play different roles
  its signature names them, and the name is a keyword: write
  `mo.place(base = slab, part = crate, ...)` and
  `mo.scatter(target = ground, prototype = rock, ...)`, so the call
  says which is which. A `material`'s textures are passed by keyword
  alone (`base_color_texture = albedo`). Params are keywords with the
  IR's names. A list
  of inputs is expanded after the keywords, as in
  `mo.boolean(id = "stair", mode = "union", *treads)`. The node's id
  is the keyword `id`. `material` and `tag` take a `name` of their
  own, which is a param. The call returns the node, which a recipe
  can pass on as an input or to `mo.output` and cannot read: it has
  no fields and no arithmetic, since a recipe computes from declared
  values and never from evaluated shape.
- **`mo.asset_ref`** takes no inputs. A part of the same project is
  `mo.asset_ref("subassets/<name>.star", "<output>", id = ...)`, and a
  sibling project's root asset is `mo.asset_ref(project = "<name>",
  output = "<output>", id = ...)`. `overrides = {...}` and
  `material_rename = {...}` are keywords (contract: *asset_ref*).
- **Declarations.** `mo.output(name, node)`, `mo.group(name,
  [outputs])`, `mo.param(id, default, ...)`, `mo.waive(output,
  check)`, `mo.waive_asset(check)`, `mo.vendored(source, hash)`, and
  in a requirements file `mo.budget(materials = N)`,
  `mo.param_value(id)`, and `mo.group` for a group its asserts name
  (see *requirements*).
- **Asserts.** `mo.assert_<kind>(id = ..., <key> = ...)`, one function
  per kind, keywords only, with the IR's keys (`outputs`, `faces`,
  `every`, `self`, `others`, `tol`, `min`, `max`, `value`, `measure`,
  `mode`). There is no `owner` keyword. The file decides (see
  *requirements*).
- **`math`**: `pi`, `tau`, `e`, and `sin`, `cos`, `tan`, `asin`,
  `acos`, `atan`, `atan2`, `sqrt`, `floor`, `ceil`, `fabs`, `hypot`,
  `radians`, `degrees`, `exp`, `log`, `pow`, `fmod`. A result is the
  same on every platform: the transcendental functions are computed
  beyond the double and rounded once.

A value that reaches the IR is a finite number, a string, a boolean,
a list, or a dict with string keys. Arithmetic that overflows gives
`None`, and the build refuses it where it is passed (`bad_value`).

Each param holds one number type, which `moraine docs <op>` lists
under `numbers`. A float param takes either, so `rotate_deg = [0,
360 / count, 0]` and `[0, 90, 0]` build the same node. An integer
param (`count`, `segments`, `seed`, a texture `size`) takes an
integer and refuses a float whatever its value (`bad_param`): write
`n // 2` between integers, and `int(x)` around anything a `/` or a
float knob went into, since `12.0 // 3` is `4.0`. An assert's lengths
and angles are floats, and `count`'s `value`, `min`, and `max`
integers. A knob keeps the type of its default: `mo.param("w", 2,
...)` is an integer knob and `2.0` a continuous one.

Names resolve once the asset's files have all run. An assert may
name an output declared further down, so asserts can be written
before the geometry they judge. A name that no `mo.output` or
`mo.group` declares fails the build (`unknown_output`), which is how
a requirement written first stays visible until its output exists.

### ids (naming nodes and asserts)

An id is a string of names joined by `/`, each a letter followed by
letters, digits, `_` or `-` (`[A-Za-z][A-Za-z0-9_-]*`),
unique within its asset (`duplicate_id` names the first call's line).
The cache, pins, journal entries, and `scatter`'s random streams all
key on the id, so a node keeps its id when the recipe around it is
restructured. A renamed node is a new node.

- **At the top level of an asset file** a node may be left unnamed and
  gets `<op><n>`, `n` counting that operator's unnamed calls in
  program order (`box1`, `box2`). Name every node that something
  points at. An output's node and a `scatter` node must be named
  (`id_required`).
- **Inside a function** every node and every assert is named. A
  function that creates nodes takes `id` as its first argument, names
  what it creates under it, and gives the node it returns the id
  itself:

```python
# steps/lib/parts.star
def step(id, index, rise, run, width):
    height = rise * (index + 1)
    block = mo.box(id = id + "/block", size = [width, height, run])
    return mo.transform(block, id = id, translate = [0, height / 2, index * run])
```

Moving nodes that already exist into a function keeps their ids:
give the function the ids they had, as arguments, where the form
above would derive new ones. `moraine build --diff` names the nodes
an edit `renamed`, and marks a `scatter` among them, whose result
changed with its name.

Every node, assert, and knob carries its `site`, the recipe file and
line of the call that made it. A node made inside a function carries
the function's line, and `via`, the line of each call on the way to
it, outermost first. When a report names a line inside a helper, the
line to edit is that one if every caller fails there, and the call
in `via` if one caller does.

### load (sharing definitions)

```python
# steps/steps.star
load("lib/parts.star", "step")
load("steps.requirements.star", "RISE", "WIDTH")

RUN = 0.28

for i in range(3):
    name = "step_%d" % i
    mo.output(name, step(name, i, RISE, RUN, WIDTH))
    if i > 0:
        mo.assert_contact(id = name + "_meets_lower", outputs = ["step_%d" % (i - 1), name])
```

`load("<path>", "<symbol>", ...)` names each symbol it takes. The path
is project-relative with forward slashes. Three rules:

- An asset loads `lib/<name>.star` files of its own project and its
  own requirements file, and nothing else.
- A `lib/` file holds definitions only. A call to `mo` at its top
  level fails the build (`lib_not_definitions`). It may load other
  `lib/` files and no requirements file, so a helper that needs a
  requirement's constant takes it as an argument.
- An asset file is never loaded. Another asset's geometry arrives
  through `mo.asset_ref`.

A loop is for copies that differ by rule, as the steps above do, and
for per-copy asserts. Copies of one unchanged part are `array` or
`scatter`, which keep them instanced through export. Three or more
nodes that place one part by one step draw `copies_not_instanced`,
in `build` and among `validate`'s warnings, with the `array` call to
write as its `fix`.

### requirements (the human's file)

```python
# steps/steps.requirements.star
RISE = 0.17
WIDTH = 0.9

mo.assert_extent(id = "top_height", outputs = ["step_2"], measure = "max_y", value = 3 * RISE, tol = 0.001)
mo.assert_extent(id = "clear_width", outputs = ["step_0"], measure = "size_x", min = WIDTH)
mo.budget(materials = 1)
```

`<name>.requirements.star` sits beside the asset `<name>.star` and
holds what the human requires of it. An assert declared there is
reported with `owner: "human"`, and one declared in an asset file or
in a function an asset calls with `owner: "agent"`. The file holds
constants, asserts, budgets, and the groups its asserts name: a
group's members are the requirement's reach, so a group an asset file
declares and a requirement names draws `group_not_owned`, in `build`
and among `validate`'s warnings, with the move as its `fix`. Every
assert in it carries an `id`. It calls no operator, declares no
output, knob, or waiver, and loads nothing (`requirements_builtin`,
`requirements_load`). The build runs
it once with its asset, whether or not the asset loads it.

The asset may load the file's constants and build against them, as
`RISE` above, so the number the human requires and the number the
geometry uses are one. A `lib/` file and a sub-asset load no
requirements file, so where one of them shapes what the human's
number governs, the asset hands the number over: as an argument to
the `lib/` function, the way `steps.star` passes `RISE` and `WIDTH`
to `step` (see *load*), or as an `overrides` value for a knob the
sub-asset declares.

The reverse draws a warning. An assert of the asset's whose threshold
is a constant of the asset that also builds the geometry moves with
the geometry and cannot fail: `threshold_tautology`, one row per
constant with its `fix`, in `build` and among `validate`'s warnings.

A threshold may follow a knob: `max = mo.param_value("<param_id>")`
takes the value the knob resolved to, so a requirement moves with
what the human accepts. The handle stands whole as `value`, `tol`,
`min`, or `max` of an assert in this file and nowhere else, and names
a number knob of this file's asset. It holds no number while the file
runs, so arithmetic and `<` on it fail the build (see *refusals*).
`==` and `if` on it learn nothing (a handle equals no number and is
true), and a handle that ends up no assert's threshold fails the
build (`param_value_unused`): a requirements file cannot branch on a
knob's value. The entry's `from_param` names the knob.

### knobs (`mo.param`, the human-tweakable values)

`mo.param(id, default, min = ..., max = ..., step = ..., label = ...)`
declares one knob and returns its value, for the recipe to compute
with anywhere. A knob is a finite number, a boolean, a string, or a
vector of 2 to 4 numbers. A number or a vector needs `min` and `max`,
which apply to every component of a vector. `step` is the viewer's
slider increment only.

- A knob's value is its default, replaced by the value the human
  accepted in the viewer (`params.json`, beside the root asset),
  replaced by a consumer's override. So a default you edit moves
  nothing while an accepted value stands, and `validate` says which:
  the knob's `params` entry carries `from: "accepted"`, and
  `--failures-only` names the knob under `params_accepted`. The
  viewer's Accept alone writes `params.json`, and the viewer alone
  removes its lines: an accepted knob shows your default beside it,
  and the panel's Reset to defaults previews every default, which
  Accept then makes the file's state. Ask the human for that when a
  default of yours must take effect. To judge the asset under another
  value meanwhile, `moraine validate --param <id>=<value>` (see
  *checks*).
- A value of the asset's own outside the declared range still
  builds, a default or an accepted value alike, and `validate` fails
  that knob's `params` entry with the fix.
- A consumer sets a knob per use with `overrides` on its
  `mo.asset_ref`, and the referenced asset is built again under that
  value. A value of the wrong kind or outside the range fails the
  consumer's build at the reference's line (`param_type_mismatch`,
  `param_out_of_range`), as does an id the asset does not declare
  (`param_unknown`).
- `params.json` holds the root asset's knobs. A sub-asset's knob is
  set by the asset that references it, so to put one in the human's
  hands declare a knob in the root and pass its value on in the
  reference's `overrides`. A sibling project's accepted values reach
  every asset that references it and does not override them.
- The viewer draws a control per knob. A knob whose value is handed
  whole to one operator param, by a call the file's top level makes
  once, gets a live control where that param is a `material` factor
  (a slider that repaints as it moves) or a `seed` (a reroll). The
  name is bound once and written once:

  ```python
  r = mo.param("rough", 0.5, min = 0.0, max = 1.0)
  mo.material(body, id = "paint", name = "paint", roughness = r)
  ```

  Any other use (arithmetic, inside a list, a second use, a
  condition, a function, a loop, an `overrides`, a threshold's
  `mo.param_value`) is a staged field behind Apply, which rebuilds
  the recipe under the value.

### refusals (what the build says no to)

A build failure is one error with a `code`, the `site` to edit, and
the spelling to write instead. What Python habit meets most often:

| written | code | write instead |
|---|---|---|
| `while` | `unsupported_syntax` | `for i in range(n):`, or `array` for copies of one part |
| `class` | `unsupported_syntax` | a function returning `struct(...)` |
| `try`, `raise`, `assert` | `unsupported_syntax` | nothing to catch: a requirement is `mo.assert_<kind>(...)` |
| `import`, `from` | `unsupported_syntax` | `load("lib/<name>.star", "<symbol>")` |
| `lambda`, `global`, `is`, `del`, `with`, `yield` | `unsupported_syntax` | a named `def`, a returned value, `==`, a new list |
| an f-string, a type annotation | `unsupported_syntax` | `"%s_%d" % (name, i)`, parameter names alone |
| `f(*xs, id = "a")` | `argument_order` | `f(id = "a", *xs)` |
| `"%02d" % i`, `"%.3f" % x`, `"{:02d}".format(i)` | `unsupported_format` | `"%d" % i`; `("00" + str(i))[-2:]` to pad, `int(x * 100) / 100` to shorten |
| `round`, `sum`, `print`, `set`, `isinstance` | `unknown_name` | the error names the replacement |
| a function that calls itself past depth 50 | `call_depth` | a loop over `range(n)` |
| a node or assert left unnamed inside a function | `id_required` | `id = id + "/<name>"` |
| a node passed as a param, a list passed as an input | `bad_value`, `bad_input` | inputs by position or by the name of their role, a list expanded after the keywords |
| `owner = ...` on an assert | `bad_assert` | declare it in the owner's file |
| a float for an integer param, `count = n / 2` | `bad_param` | `n // 2`, or `int(x)` |
| `fail(...)` at a file's top level | `fail_at_top_level` | `mo.assert_<kind>(...)`; inside a function `fail` is the function's own refusal (`recipe_failed`) |
| `mo.budget` in an asset file | `budget_outside_requirements` | the requirements file |
| `mo.param_value(...) + 0.1`, `< 2` | `param_value_arithmetic` | the derived quantity as a constant of the requirements file |
| `mo.param_value` in an asset, or anywhere but a threshold | `param_value_outside_requirements`, `param_value_misplaced` | in an asset, the value `mo.param` returned |
| `if mo.param_value(...):`, `== 2`, a handle no assert takes | `param_value_unused` | requirements that hold at every value of the knob, the handle as the threshold that moves |
| a `load` the three rules exclude | `load_refused` | see *load* |
