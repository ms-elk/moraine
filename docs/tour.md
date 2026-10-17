# Tour: your first asset

English | [日本語](tour.ja.md)

This tour picks up where the README's
[Get started](../README.md#get-started) leaves off, with `moraine`
installed and a workspace from `moraine init` open in your agent,
and takes one asset from a request to an exported file. You write
the request, look at the result, and accept it. The agent does the
rest.

## 1. The workspace

A workspace holds projects, one folder each. A project's folder has
its recipe, a `.star` file, and beside it a requirements file,
`<name>.requirements.star`, where your requirements go. As the agent
works, it adds a `journal.md`, a `build/` folder for the tool's
intermediate output, and an `exports/` folder for exported files.
`AGENTS.md` at the workspace root is yours, for rules every request
should follow, such as
default dimensions or your working language.

## 2. Ask for the stool

Say what you want, and which parts of it are requirements, like
this:

> Make a stool in a `stool` folder, 0.5 m tall, with a 0.36 m square
> seat on four legs, instanced. The seat height and the seat resting
> on every leg are my requirements.

The agent writes the recipe into a new `stool/` project. The tool
generates the mesh from the recipe and verifies it, and the agent
edits the recipe until every requirement passes. You don't need to
read the recipe. Worth seeing once is where your two requirements
landed, in `stool/stool.requirements.star`:

```python
SEAT_H = 0.50
LEG_BEARING_MIN = 0.03 * 0.03

mo.assert_extent(id = "seat_height", outputs = ["seat"], measure = "max_y", value = SEAT_H)
mo.assert_contact(id = "seat_on_every_leg", outputs = ["legs", "seat"], every = ["legs"], min_area = LEG_BEARING_MIN)
```

These lines are what the tool verifies. The height is your number
as you said it. The bearing area, how much of each leg the seat must
sit on for "resting on" to hold, is the agent's translation, and the
agent asks you to confirm it. The file is yours: a line in it
changes only when you say so.

## 3. Look at it

When the agent starts work, the viewer opens in your browser. If it
does not, ask the agent for the viewer's URL. It shows the project
as it is now and follows every edit the agent makes. The verify
rounds panel at the top left lists the rounds and, while the newest
one fails, names the checks that failed. A value the agent left in
your hands, the leg size here, sits on a panel at the top right.
Change it there and click Apply to preview the result. When you're
happy with it, click Accept to save your choice to the project's
`params.json`, and the tool verifies the requirements again under
the value you chose. The recipe does not change.

<img src="images/stool.png" width="480" alt="the stool in the viewer, with the verify rounds panel at the top left and the leg size knob at the top right">

If you want to change something, or want another parameter on the
panel, say so. You can also pin the part in the viewer and say so
there. The agent edits the recipe, the tool verifies again, and the
viewer follows. What ships is what you accept.

## 4. Export it

When you're happy with the stool, ask the agent:

> Export the stool.

The agent runs the export, and the tool writes
`stool/exports/stool.glb`, a glTF 2.0 file with the parts,
materials, and instancing intact, and a manifest beside it. Export
requires validation to pass. Under the same conditions the
manifest records, such as the Moraine version, exporting an
unchanged recipe reproduces the file byte for byte.

Open it in a viewer or engine that supports glTF instancing, such
as Unity with glTFast. If yours does not, ask the agent to export
with the instances baked.

## Next time

The agent keeps a `journal.md` in the `stool/` project and reads it
before the next round, so you pick up where you left off.

## Explore the sample workspace

See how a railway station kit grows round by round through
building, refining, and extending. The
[sample workspace](https://github.com/ms-elk/moraine-sample-workspace)
holds the requests, recipes, and journals behind each decision.
