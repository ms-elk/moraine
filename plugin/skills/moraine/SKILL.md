---
name: moraine
description: "Set up or upgrade Moraine, an AI-native procedural modeling tool for real-time 3D assets, in a workspace, then hand off to the authoring guide Moraine ships. Use when the human wants to author, edit, verify, or export a 3D asset with Moraine, mentions a Moraine recipe or a Moraine workspace, asks to start a Moraine project, or asks to install, update, or upgrade Moraine. The request may come in any language, with only the tool's name in English. Setup only: everything about authoring lives in the shipped guide this skill points at."
---

# Moraine setup

This skill only readies a workspace for Moraine.

Work in the directory the human wants the workspace in.

## 1. Check the install

```
moraine --version
```

What to run next depends on the answer:

- Command missing: `uv tool install moraine3d`, once per machine,
  never into a project venv: a second `moraine` on the PATH is how a
  report gets read against the wrong version. If `uv` is missing,
  install it first (https://docs.astral.sh/uv/); it also provisions
  the Python 3.12+ Moraine requires. If `moraine` is still not found
  after the install, run `uv tool update-shell` and open a new shell.
- Installed, and the human asked to upgrade:
  `uv tool upgrade moraine3d`, then confirm `moraine --version`
  changed
- Installed, and they did not ask: nothing. Never upgrade unasked; the
  version change would re-sync the guide under a project in progress

## 2. Scaffold the workspace

```
moraine init
```

Run it after every install *and every upgrade*, in each workspace:
it syncs the authoring guide at `.moraine/guide.md` to the installed
version, and until it runs the workspace's guide is the previous
version's. Read the report. `AGENTS.md` under `created` means a fresh
workspace (step 3); `journal_review` lists `workaround` entries
stamped with another version, which the guide tells you how to
retire.

Your harness reaches the workspace through `AGENTS.md`, which points
at the guide; write no harness file of your own.

## 3. Ask the human's working language

Only when step 2 reported `AGENTS.md` under `created`. Otherwise the
file is already the human's and you leave it alone.

Ask in the language the human has been using. This skill is in
English for you; the question is for them. Their conversation
language is also the default answer: propose it and ask them to
confirm, and ask openly only when the conversation gives you nothing
to go on.

If the answer is not English, translate the freshly scaffolded
`AGENTS.md` in place, keeping its structure and headings. It stays
the human's file: you edit it at their direction.

Shipped content stays English: `.moraine/guide.md` is overwritten on
every `init`, so never translate it.

## Then read the guide

Read `.moraine/guide.md` and follow it. Setup is done; this skill
adds nothing to the guide.
