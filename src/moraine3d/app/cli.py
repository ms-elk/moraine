"""CLI (DESIGN 6). All stdout is terse machine-readable JSON.

Commands: init, build, eval, stats, validate, query, export, serve,
screenshot, docs, sessions, pins list|add|clear, mcp,
cache stats|clear|evict.

Every invocation but ``serve`` (whose server writes its own events) and
``mcp`` (whose tool calls each record their own) appends one
``command`` event to the session log (DESIGN 6) from ``main``'s tail;
handlers contribute the asset revision and a compact outcome through
``session.annotate*``.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import http.client
import socket
import json
import math
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from .. import dist_version
from ..cache import store as cache_mod
from ..core import asset as asset_mod
from ..core import sources as sources_mod
from ..core import workspace as workspace_mod
from ..engine.stats import asset_rollup
from ..errors import MoraineError
from ..recipe import build as build_mod
from ..recipe import target as target_mod
from . import camera as camera_mod
from . import session
from .server import DEFAULT_PORT, ROOT_HEADER, open_browser

# What the command wrote to stdout: the session log's ``out_bytes`` (the
# agent-read volume — a token-cost proxy) and the error code, taken from
# the one printed object. Reset per main() call. ``quiet`` keeps the
# object off stdout for an in-process caller that relays it itself (the
# MCP server, whose stdout is the transport); the accounting is unchanged.
_out: dict = {"bytes": 0, "last": None, "quiet": False}


def _print(obj: dict) -> None:
    # allow_nan=False: the output contract is standard JSON; a NaN reaching
    # here is an upstream validation bug, not something to serialize.
    line = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    _out["bytes"] += len(line.encode("utf-8")) + 1
    _out["last"] = obj
    if not _out["quiet"]:
        print(line)


def _print_text(text: str) -> None:
    """A page of the shipped reference as it is written (DESIGN 6):
    prose is read, not parsed, so it is printed without the JSON
    around it. The reference is ASCII, which every console prints."""
    _out["bytes"] += len(text.encode("utf-8")) + 1
    _out["last"] = {"text": text}
    if not _out["quiet"]:
        print(text)


def _open(target: Path):
    """The target, opened (DESIGN 6): a recipe is built with its
    closure, an IR handed over by path is read as given."""
    return target_mod.open_target(target)


def _evaluation(opened, **kw):
    from ..engine.evaluator import Evaluation

    return Evaluation(opened.asset, **opened.inputs, **kw)


def _note_asset(opened, ev, *, partial: bool = False, **summary) -> None:
    """Annotate the session-log event with the revision a handler
    evaluated: the project (by name), the root framing its content hash
    (the enclosing workspace root, else the project directory — the
    same framing the server uses), and the hash itself, over every
    recipe file that contributed. ``partial`` marks a subgraph
    evaluation."""
    sc = opened.asset
    project_dir = asset_mod.project_root(sc.path)
    root = opened.root
    session.annotate(root=str(root))
    try:
        asset_hash = opened.revision_hash(ev.asset_files)
    except OSError:
        # A contributing file went unreadable between evaluation and
        # stamping (editor save window): record the revision hash-less
        # rather than name a file set that never existed on disk.
        asset_hash = None
    session.annotate_project(
        project_dir.name,
        asset_hash=asset_hash,
        partial=True if partial else None,
        summary=summary or None,
    )


def _resolve_target(path_arg) -> Path | workspace_mod.Workspace:
    """Workspace-aware resolution (stats, validate, serve): a workspace
    directory returns its discovery; anything else resolves as a
    single asset (DESIGN 6)."""
    p = Path(path_arg)
    if p.is_dir() and workspace_mod.is_workspace(p):
        return workspace_mod.discover(p)
    asset = sources_mod.resolve(path_arg)
    # Annotate the resolved project now, before anything evaluates: a
    # command that fails mid-evaluation must still leave an attributed
    # session-log event, or per-project error accounting would depend
    # on which entry point (CLI vs serve) ran the failing evaluation.
    project_dir = asset_mod.project_root(asset.resolve())
    session.annotate(root=str(workspace_mod.enclosing_root(project_dir)))
    session.annotate_project(project_dir.name)
    return asset


def _addressed_project(path_arg: str | None) -> tuple[str | None, Path | None]:
    """The project a server-channel command speaks for (DESIGN 6) and
    the root a server for it serves: the roster name of ``path_arg``'s
    project, or None to mean "whatever the server displays". None is
    taken only when the argument was omitted and the working directory
    holds no asset — an explicit target that resolves to nothing is
    the ordinary asset_not_found error, never a silent fall-through to
    the displayed project. A workspace root as the working directory
    names the root with no project."""
    try:
        asset = sources_mod.resolve(path_arg or ".")
    except MoraineError:
        if path_arg is not None:
            raise
        # No project here, but a workspace root still says whose
        # server the command means.
        here = Path.cwd()
        return None, here.resolve() if workspace_mod.is_workspace(here) else None
    # The unresolved path: the roster keys projects by the directory
    # name under the workspace root, which resolve() can escape through
    # a junction (the server's _bind_asset keeps the same rule).
    project = asset_mod.project_root(asset)
    return project.name, workspace_mod.enclosing_root(project)


def _no_projects_error(ws: workspace_mod.Workspace) -> MoraineError:
    return MoraineError(
        "empty_workspace",
        f"no projects under workspace '{ws.root}'; a project is a "
        "child directory holding one root asset (<name>.star) - write the "
        "first one, then run this again",
    )


def _resolve_asset(path_arg) -> Path:
    """Single-asset resolution (eval, query, export): a workspace
    target is an actionable error, never a fall-through to the bare
    asset_not_found (DESIGN 6: these commands take one project per
    invocation; serve routes workspaces through _resolve_target)."""
    target = _resolve_target(path_arg)
    if isinstance(target, workspace_mod.Workspace):
        if not target.projects:
            raise _no_projects_error(target)
        raise MoraineError(
            "workspace_target",
            f"'{target.root}' is a workspace; this command takes one "
            "project - pass one of its project directories",
            projects=list(target.projects),
        )
    return target


def _sweep(ws: workspace_mod.Workspace, per_project) -> tuple[dict, list[str]]:
    """Run a per-project report function over every discovered project.

    ``per_project(asset_path) -> (report, ok)``. A ``MoraineError`` from
    one project lands in that project's slot as ``{"error": ...}`` and
    never escapes to main()'s handler, so siblings still run; any other
    exception is a bug and propagates (the same contract as a
    single-project run). ``failed`` lists projects that would have
    exited 1 standalone (error or ok=False). The cache session write
    counter accumulates across the whole sweep, so the single
    post-command _housekeep() check stays correct.
    """
    if not ws.projects:
        raise _no_projects_error(ws)
    session.annotate(root=str(ws.root.resolve()), sweep=True)
    projects: dict[str, dict] = {}
    failed: list[str] = []
    for name in ws.projects:
        try:
            report, ok = per_project(ws.resolve_asset(name))
        except MoraineError as e:
            projects[name] = {"error": e.to_dict()}
            failed.append(name)
            session.annotate_project(name, ok=False, code=e.code)
            continue
        projects[name] = report
        session.annotate_project(name, ok=ok)
        if not ok:
            failed.append(name)
    return projects, failed


def _sweep_rollup(ws: workspace_mod.Workspace, roll: dict, probe: _ServeProbe) -> dict:
    """The keys every sweep's ``workspace`` section carries beyond its
    own aggregate (DESIGN 6): ``not_swept`` — the assets below the
    project level the sweep did not reach, always present so an empty
    list is the positive statement — and ``serve`` when a server for
    this root answers."""
    roll["not_swept"] = workspace_mod.unswept_assets(ws)
    return _with_serve(roll, probe)


def _with_serve(report: dict, probe: _ServeProbe) -> dict:
    status = probe.result()
    if status is not None:
        report["serve"] = status
    return report


def _stamped(report: dict) -> dict:
    """Every evaluated report names the tool that produced it (DESIGN 6):
    ``moraine_version`` leads the object, the same key the export
    manifest and the screenshot report carry, so a report can never be
    mistaken for another install's answer."""
    return {"moraine_version": dist_version(), **report}


def _target_root(target) -> Path:
    """The root a server for this target would serve (the sweep root, or
    the project's enclosing root) — what the status probe matches."""
    if isinstance(target, workspace_mod.Workspace):
        return target.root
    return workspace_mod.enclosing_root(asset_mod.project_root(target))


def _cmd_eval(args) -> int:
    opened = _open(_resolve_asset(args.project))
    sc = opened.asset
    # only=... evaluates just that output's subgraph — other outputs are
    # neither computed nor able to fail the command.
    ev = _evaluation(opened, recompute=args.recompute)
    results = ev.results(only=args.output)
    profile = ev.profile()
    report = {"outputs": results, "profile": profile}
    summary: dict = {}
    if args.recompute:
        # The determinism check (DESIGN 4.4): a mismatch is exit 1 with
        # the report intact - the verdict and its evidence are the output.
        report["recompute"] = ev.recompute_report()
        summary["recompute_mismatched"] = report["recompute"].get("mismatched_total", 0)
    # The session summary carries the same counts as stats (DESIGN 6);
    # under --output they cover the evaluated subgraph, flagged partial.
    asset = asset_rollup({n: r["stats"] for n, r in results.items()}, len(sc.nodes))
    _note_asset(
        opened, ev, partial=args.output is not None,
        outputs=len(results), nodes=len(sc.nodes),
        tris=asset["tris"], instances=asset["instances"],
        hits=profile["hits"], misses=profile["misses"], eval_ms=profile["ms"],
        **summary,
    )
    _print(_stamped(report))
    return 1 if summary.get("recompute_mismatched") else 0


def _workspace_rollup(projects: dict[str, dict], failed: list[str]) -> dict:
    """Workspace-level aggregate for the stats sweep (DESIGN 6): a plain
    sum over each successful project's asset rollup. ``bounds`` is
    deliberately absent — projects share no coordinate space, so a union
    AABB would be a meaningless number. ``failed`` is always present so
    the shape stays stable for parsers. Cross-project-referenced
    geometry counts once per consuming project, deliberately (DESIGN 6):
    every project's exports are self-contained, so the sum reads as
    shipped-per-project content, not a dedup'd source count.
    """
    roll: dict = {"projects": len(projects), "failed": failed,
                  "outputs": 0, "parts": 0, "nodes": 0, "instances": 0,
                  "verts": 0, "tris": 0}
    for report in projects.values():
        asset = report.get("asset")
        if asset is None:
            continue
        for key in ("outputs", "parts", "nodes", "instances", "verts", "tris"):
            roll[key] += asset[key]
    return roll


def _stats_report(asset_path: Path) -> dict:
    opened = _open(asset_path)
    sc = opened.asset
    # records(), not results(): stats come from cache metadata and must
    # not pay for GLB materialization (DESIGN 8.2 tiering).
    ev = _evaluation(opened)
    records = ev.records()
    stats = {name: rec.stats for name, rec in records.items()}
    asset = asset_rollup(stats, len(sc.nodes))
    sites = ev.output_sites()
    stats = {
        name: {**sites.get(name, {}), **s}
        for name, s in stats.items()
    }
    budget = (sc.budgets or {}).get("materials")
    if budget is not None and "materials" in asset:
        # Headroom readable from stats: the declared cap beside the count.
        asset["materials"]["budget"] = budget
    profile = ev.profile()
    _note_asset(
        opened, ev, outputs=asset["outputs"], nodes=asset["nodes"],
        tris=asset["tris"], instances=asset["instances"],
        hits=profile["hits"], misses=profile["misses"], eval_ms=profile["ms"],
    )
    return {
        "asset": asset,
        "outputs": stats,
        "profile": profile,
    }


def _cmd_stats(args) -> int:
    _check_port(args.port)  # before any evaluation: a bad flag costs nothing
    target = _resolve_target(args.project)
    probe = _ServeProbe(args.port, _target_root(target))
    if isinstance(target, workspace_mod.Workspace):
        projects, failed = _sweep(target, lambda p: (_stats_report(p), True))
        roll = _sweep_rollup(target, _workspace_rollup(projects, failed), probe)
        _print({"projects": projects, "workspace": _stamped(roll)})
        return 0 if not failed else 1
    _print(_stamped(_with_serve(_stats_report(target), probe)))
    return 0


def _given_params(items: list[str] | None) -> dict[str, str]:
    """``--param ID=VALUE`` by id, each VALUE as it was typed (DESIGN
    6). An id given twice is refused: one report judges one value."""
    given: dict[str, str] = {}
    for item in items or []:
        pid, sep, raw = item.partition("=")
        if not sep or not pid:
            raise MoraineError(
                "bad_param_flag",
                f"--param takes ID=VALUE, got '{item}'; example: "
                f"--param width=2.4 --param size=[1,2,3]",
            )
        if pid in given:
            raise MoraineError(
                "bad_param_flag",
                f"--param names '{pid}' twice ({given[pid]} and {raw}); one "
                f"report judges one value, so run validate once per value",
                param=pid,
            )
        given[pid] = raw
    return given


def _offered_values(asset_path: Path, given: dict[str, str]) -> dict:
    """The typed values ``--param`` offers: a string knob takes VALUE
    as typed, so a value that reads as a number needs no quoting, and
    every other knob takes it as JSON. The kinds are the declarations'
    of the asset as its files state it, each shown by the recipe's
    default (an accepted value is the human's, not the declaration)."""
    declared = {
        d["id"]: d.get("default", d["value"])
        for d in target_mod.open_target(asset_path, write=False).asset.params or []
    }
    offered: dict = {}
    for pid, raw in given.items():
        if isinstance(declared.get(pid), str):
            offered[pid] = raw
            continue
        try:
            offered[pid] = json.loads(raw)
        except json.JSONDecodeError:
            if pid in declared:
                raise MoraineError(
                    "bad_param_flag",
                    f"--param {pid}={raw}: the value is not JSON, and "
                    f"'{pid}' is declared with the default {json.dumps(declared[pid])}; "
                    f"example: {pid}={json.dumps(declared[pid], separators=(',', ':'))}",
                    param=pid,
                ) from None
            offered[pid] = raw  # an undeclared id: the build names it
    return offered


def _validate_report(
    asset_path: Path, profile, fails_only, given: dict | None = None,
) -> tuple[dict, bool]:
    from ..engine import validation
    from ..verify.validate import add_by_site

    if given:
        if asset_mod.is_ir_file(asset_path):
            raise MoraineError(
                "param_needs_recipe",
                f"--param offers values to a recipe's knobs, and "
                f"'{asset_path.name}' is an IR, evaluated as it is; name "
                f"the project or its asset file",
            )
        # Built as a consumer's offer is (DESIGN 4.6): in memory, over
        # the accepted values, with nothing written, and evaluated into
        # the preview bucket, so a sweep evicts no record of the file
        # state (DESIGN 4.5).
        given = _offered_values(asset_path, given)
        opened = target_mod.open_target(
            asset_path, offered=given, write=False, ranged_offers=False,
        )
    else:
        opened = _open(asset_path)
    ev = _evaluation(opened, preview=bool(given))
    report, values = validation.report(opened.asset, ev)
    mates = ev.mate_frames()
    if mates:
        report["mates"] = mates
    if opened.build is not None and opened.build.warnings:
        report["warnings"] = report.get("warnings", []) + opened.build.warnings
    if opened.build is not None and opened.build.swept.readers is not None:
        # The projects that reference this one (DESIGN 6): its own
        # checks do not run theirs, so the report says who to check.
        report["consumers"] = opened.build.swept.readers
    if profile is not None:
        from ..engine.export import profile_checks

        report = profile_checks(profile, values, report)
    ok = report["ok"]
    # A report under offered values judges no revision of the files.
    _note_asset(opened, ev, partial=bool(given), **_validate_summary(report))
    add_by_site(report)
    if fails_only:
        from ..verify.validate import failures_only

        report = failures_only(report)
    if given:
        report["params_given"] = given
    return report, ok


def _validate_summary(report: dict) -> dict:
    """The session log's compact verify outcome (DESIGN 6): which gating
    checks, asserts, budgets, and knob ranges went red — read from the full report,
    before --failures-only reshapes it. Warn-level checks are advisory
    and stay out of ``failed_checks``."""
    outputs = report.get("outputs") or {}
    failed_checks: list[str] = []
    failed_outputs = 0
    for out in outputs.values():
        red = [
            c["check"] for c in out.get("checks", [])
            if not c.get("ok") and c.get("level") != "warn"
        ]
        if red:
            failed_outputs += 1
        failed_checks.extend(c for c in red if c not in failed_checks)
    failed_checks.extend(
        c["check"] for c in report.get("asset_checks") or []
        if not c.get("ok") and c["check"] not in failed_checks
    )
    return {
        "outputs": len(outputs),
        "failed_outputs": failed_outputs,
        "failed_checks": failed_checks,
        "asserts_failed": sum(1 for a in report.get("asserts") or [] if not a.get("ok")),
        "budgets_failed": sum(1 for b in report.get("budgets") or [] if not b.get("ok")),
        "params_failed": sum(1 for p in report.get("params") or [] if not p.get("ok")),
        "warnings": len(report.get("warnings") or []),
    }


def _cmd_validate(args) -> int:
    _check_port(args.port)  # before any evaluation: a bad flag costs nothing
    target = _resolve_target(args.project)
    given = _given_params(args.param)
    if given and isinstance(target, workspace_mod.Workspace):
        raise MoraineError(
            "param_needs_project",
            "--param offers values to one project's knobs; name the "
            "project, not the workspace root",
        )
    probe = _ServeProbe(args.port, _target_root(target))
    if isinstance(target, workspace_mod.Workspace):
        projects, failed = _sweep(
            target,
            lambda p: _validate_report(p, args.profile, args.failures_only),
        )
        roll = _sweep_rollup(
            target,
            {"ok": not failed, "projects": len(projects), "failed": failed},
            probe,
        )
        _print({"projects": projects, "workspace": _stamped(roll)})
        return 0 if not failed else 1
    report, ok = _validate_report(target, args.profile, args.failures_only, given)
    _print(_stamped(_with_serve(report, probe)))
    return 0 if ok else 1


def _cmd_query(args) -> int:
    from ..core import predicates

    select = None
    if args.select is not None:
        try:
            select = json.loads(args.select)
        except json.JSONDecodeError as e:
            raise MoraineError(
                "bad_predicate",
                f"--select must be valid JSON ({e}); "
                f'example: {{"normal_y": {{"gt": 0.9}}}}',
            )
        predicates.validate_predicate(select, where="--select")
    elif args.tag is None:
        raise MoraineError(
            "bad_predicate",
            "query needs --select '<predicate JSON>', --tag <name>, or both "
            "(the tag's faces the predicate matches)",
        )
    opened = _open(_resolve_asset(args.project))
    ev = _evaluation(opened)
    records = ev.records(only=args.output)
    _note_asset(opened, ev, partial=args.output is not None, outputs=len(records))
    sites = ev.output_sites()
    rows = {
        name: predicates.output_query(rec.value(), select, args.tag, args.instance)
        for name, rec in records.items()
    }
    lacking = [r for r in rows.values() if "tags" in r]
    if args.tag is not None and lacking and len(lacking) == sum(
        "skipped" not in r or "tags" in r for r in rows.values()
    ):
        # No output carries the tag: a name mistyped, or a tag a rebuild
        # dropped, is an error and not a report of skipped rows.
        carried = sorted({t for r in lacking for t in r["tags"]})
        raise MoraineError(
            "unknown_tag",
            f"no output carries a tag '{args.tag}'; the outputs' tags: "
            f"{carried if carried else 'none (a tag node goes after the last topology rebuild)'}",
            tags=carried,
        )
    for row in rows.values():
        row.pop("tags", None)
    _print(_stamped({
        "outputs": {
            name: {**sites.get(name, {}), **row} for name, row in rows.items()
        }
    }))
    return 0


def _note_build(result) -> None:
    """Annotate the session-log event with the revision a build read:
    the project, the root, and the content hash over every recipe file
    that contributed (DESIGN 6)."""
    project_dir = asset_mod.project_root(result.target.recipe)
    session.annotate(root=str(result.root))
    try:
        asset_hash = asset_mod.content_hash(
            result.sources, result.root, read=result.contents,
        )
    except OSError:
        asset_hash = None
    session.annotate_project(
        project_dir.name, asset_hash=asset_hash,
        summary={**result.target.counts, "assets": len(result.assets),
                 "written": len(result.written), "build_ms": result.ms},
    )


def _cmd_build(args) -> int:
    """Build the target's recipe and its reference closure into IR
    under each project's ``build/`` (DESIGN 4.8, 6)."""
    recipe = _resolve_asset(args.project)
    if asset_mod.is_ir_file(recipe):
        raise MoraineError(
            "not_an_asset",
            f"'{recipe.name}' is an IR, what a build writes; name the "
            "project or its asset file",
            file=recipe.name,
        )
    result = build_mod.build(recipe, write=False)
    # Read before the build writes: the base of the diff is the IR on
    # disk before this build (DESIGN 6).
    changed = _build_diff(result, args.diff) if args.diff is not None else None
    if args.check:
        build_mod.look(result)
    else:
        build_mod.write_out(result)
    _note_build(result)
    report = result.report(wrote=not args.check)
    if args.check:
        report["check"] = build_mod.compare(result)
        report["ok"] = "fix" not in report["check"]
    if changed is not None:
        report["diff"] = changed
    _print(_stamped(report))
    return 0 if report["ok"] else 1


def _build_diff(result, base: str) -> dict:
    """``build --diff``'s report (DESIGN 6): the graph change from the
    IR on disk before the build, or from the IR file or the ``build/``
    directory ``base`` names."""
    from ..engine import diff as diff_mod

    target = result.target.ir_path
    base_target = None
    if base:
        given = Path(base)
        if given.is_dir():
            home = asset_mod.project_root(result.target.recipe) / asset_mod.BUILD_DIR
            base_target = given.resolve() / target.relative_to(home)
        else:
            base_target = given.resolve()
        if not base_target.is_file() or not asset_mod.is_ir_file(base_target):
            raise MoraineError(
                "diff_base_not_found",
                f"--diff names '{base}', which holds no IR of "
                f"'{target.name}' to compare against; name an IR file, or "
                f"the build/ directory of the project as another revision "
                f"built it",
                path=base,
            )
    diffs = diff_mod.closure_diff(
        target, {p: b.asset for p, b in result.assets.items()},
        diff_mod.from_disk, base_target,
    )
    if diffs[target].uncompared:
        raise MoraineError(
            "diff_base_not_found",
            f"--diff names '{base}', whose IR of '{target.name}' could not "
            f"be read as one this build wrote; build that revision again "
            f"with this install and name its build/",
            path=base,
        )

    def shown(path: Path) -> str:
        try:
            return path.relative_to(result.root).as_posix()
        except ValueError:
            return path.as_posix()

    return diff_mod.report(diffs, shown, base or "disk")


def _cmd_export(args) -> int:
    from ..engine.export import export_asset

    asset_path = _resolve_asset(args.project)
    opened = _open(asset_path)
    sc = opened.asset
    project_dir = asset_mod.project_root(asset_path)
    # A sub-asset sharing its stem with the project's root asset would
    # write the root's <stem>.glb + manifest pair - a silent overwrite,
    # since every out dir is shared per project.
    if (
        asset_path.resolve().parent.name.lower() == asset_mod.SUBASSETS_DIR
        and sources_mod.asset_files(project_dir)
    ):
        root_asset = sources_mod.resolve(project_dir)
        if root_asset.stem == asset_path.stem:
            raise MoraineError(
                "export_stem_collision",
                f"sub-asset '{asset_path.name}' shares its stem with the "
                f"project's root asset, so its export would overwrite "
                f"'{root_asset.stem}.glb' - rename the sub-asset file",
            )
    # A relative -o anchors to the project, not the shell cwd: an export
    # is a project artifact, so the guide's `-o exports/` idiom lands
    # inside the project from any invocation directory. screenshot -o
    # stays cwd-relative by design - a screenshot is a session artifact.
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        # Windows drive-relative ('D:exports') and rooted ('\\exports')
        # spellings are not absolute, yet joining would discard the
        # project anchor - refuse rather than write somewhere silent.
        if out_dir.drive or out_dir.root:
            raise MoraineError(
                "bad_out_dir",
                f"-o '{args.out}' is drive-relative or rooted; pass a "
                "plain relative path (resolved against the project) or "
                "a fully absolute path",
            )
        out_dir = project_dir / out_dir
    ev = _evaluation(opened, recompute=args.recompute)
    summary: dict = {}
    recompute: dict | None = None
    if args.recompute:
        # The determinism verdict comes first: a mismatch against the
        # cached records means the recomputed bytes are not the ones a
        # warm cache would have exported - not an export to trust, so
        # nothing is written and the last good export stays in place
        # (the same rule as a validation failure, DESIGN 8.2).
        ev.records()
        recompute = ev.recompute_report()
        summary["recompute_mismatched"] = recompute.get("mismatched_total", 0)
    if recompute is not None and summary["recompute_mismatched"]:
        result = {
            "ok": False, "profile": args.profile, "outputs": {},
            "recompute": recompute, "fix": recompute["fix"],
        }
    else:
        result = export_asset(
            sc, profile=args.profile, out_dir=out_dir,
            bake_instances=args.bake_instances, evaluation=ev,
        )
        if recompute is not None:
            result["recompute"] = recompute
        if opened.build is not None and opened.build.warnings:
            # An export is no quieter than the validate it gates on:
            # what the build warned of, the export's report carries.
            carrier = result if result["ok"] else result["validation"]
            carrier["warnings"] = carrier.get("warnings", []) + opened.build.warnings
    _note_asset(
        opened, ev, profile=args.profile, ok=result["ok"],
        outputs=len(result.get("outputs") or {}), **summary,
    )
    _print(result)
    return 0 if result["ok"] else 1


def _check_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise MoraineError(
            "bad_port", f"--port must be in 1..65535, got {port}"
        )


def _server_request(
    port: int, path: str, *, method: str,
    body: dict | None = None, timeout_code: str, timeout: float = 60.0,
    root: Path | None = None,
) -> bytes:
    """One request against the running server, the shared error mapping
    of the CLI's server channels (pins, screenshot, status): server
    errors unwrap into their own codes, a timeout reports
    ``timeout_code``, a listener that does not speak HTTP reports
    ``stale_server``, and no server maps to ``no_server``. The caller
    decodes the 200 body — each channel's ``stale_server`` answer names
    its own contract. ``root`` names the root the caller speaks for, so
    a server of another root refuses the request (``wrong_workspace``)
    instead of answering it from its own projects."""
    _check_port(port)
    no_server = (
        f"no moraine server on port {port}; run 'moraine serve', or pass "
        "--port if serve listens on another port"
    )
    if not _reachable(port):
        raise MoraineError("no_server", no_server)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            **({"Content-Type": "application/json"} if body is not None else {}),
            # Percent-encoded: a header carries latin-1 only.
            **({ROOT_HEADER: urllib.parse.quote(str(root))} if root is not None else {}),
        },
        method=method,
    )
    timeout_msg = (
        f"no response within {timeout:g} s; the server may still be evaluating a "
        f"heavy asset - check the 'moraine serve' terminal on port "
        f"{port} and retry"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(payload)["error"]
            raise MoraineError(err["code"], err["message"], **err.get("detail", {}))
        except (json.JSONDecodeError, KeyError, TypeError):
            raise MoraineError("http_error", f"server returned {e.code}")
    except TimeoutError:
        # The header wait times out as a bare TimeoutError (urllib wraps
        # only the request phase in URLError), and the server can hold a
        # response through a whole evaluation — plus, on the screenshot
        # channel, the viewer wait.
        raise MoraineError(timeout_code, timeout_msg)
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), TimeoutError):
            raise MoraineError(timeout_code, timeout_msg)
        raise MoraineError("no_server", no_server)
    except http.client.HTTPException as e:
        # The response phase is not wrapped by urllib: a listener that
        # is not an HTTP server (BadStatusLine), or one whose body ends
        # early (IncompleteRead), surfaces raw.
        raise MoraineError(
            "stale_server",
            f"the listener on port {port} did not answer as a moraine server "
            f"({type(e).__name__}); check the port, or restart 'moraine serve'",
        )


# Loopback connect budget for the pre-check every server channel runs
# before its request: a listening server completes the handshake in the
# kernel, at once, even mid-evaluation, so a connect that takes longer
# means no server. Without the check, a port nobody listens on is not
# refused on every host — the Windows firewall drops the SYN, and the
# connect runs to the channel's full timeout: 2 s added to every
# `validate`/`stats` with no serve running, 30 s to a screenshot, and
# "no_server" misreported as a timeout.
_CONNECT_TIMEOUT_S = 0.5


def _reachable(port: int) -> bool:
    """Whether something accepts TCP connections on the loopback port
    within ``_CONNECT_TIMEOUT_S``; the channel's request then decides
    whether it is a moraine server."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=_CONNECT_TIMEOUT_S).close()
        return True
    except OSError:
        return False


# A status probe is a loopback GET the server answers from memory; with
# the connect pre-check, no server returns at once. The budget bounds
# the wait when the server is busy in an evaluation (it answers nothing
# meanwhile), and the probe runs on a thread alongside the command's
# own work, so the wait overlaps the evaluation instead of following it.
_STATUS_TIMEOUT_S = 2.0


def _status(port: int) -> dict | None:
    """What a ``moraine serve`` on the port answers to the status
    contract, its ``root`` resolved; None for no server, a foreign
    listener, or a stale build."""
    try:
        raw = _server_request(
            port, "/api/status", method="GET",
            timeout_code="status_timeout", timeout=_STATUS_TIMEOUT_S,
        )
        status = json.loads(raw.decode("utf-8"))
        if not isinstance(status, dict) or "watch" not in status:
            return None
        return {**status, "root": Path(status["root"]).resolve()}
    except (MoraineError, OSError, ValueError, KeyError, TypeError):
        return None


def _serve_status(port: int, root: Path) -> dict | None:
    """The running server's watch liveness for the target's root (DESIGN
    6): ``{"port", "project", "watch"}`` when a ``moraine serve`` on the
    port answers the status contract for the same root, and ``{"port",
    "other_root"}`` when it answers for another, so a server that is
    not this root's is never taken for it. No server, a foreign
    listener, or a stale build read as "nothing known", never as an
    error: the report is about the asset, and this key is the one fact
    only the server holds (the human's display goes stale when the
    file watch dies while every agent-side check keeps passing)."""
    status = _status(port)
    if status is None:
        return None
    if status["root"] != root.resolve():
        return {"port": port, "other_root": str(status["root"])}
    out = {"port": port, "project": status.get("project"), "watch": status["watch"]}
    if status.get("listen_rearmed"):
        # A listener that died and re-armed in this server's life
        # (DESIGN 7.2): the one trace a healed half-death leaves.
        out["listen_rearmed"] = status["listen_rearmed"]
    return out


class _ServeProbe:
    """The status probe started before the command's own work and read
    when the report is assembled: a daemon thread carries
    ``_serve_status`` so its wait overlaps the evaluation; ``result``
    joins with whatever is left of the probe's budget (measured from
    the start) and reads "nothing known" when the probe has not
    answered by then — never a stall after the report is ready."""

    def __init__(self, port: int, root: Path):
        self._result: dict | None = None
        self._deadline = time.monotonic() + _STATUS_TIMEOUT_S
        self._thread = threading.Thread(
            target=self._run, args=(port, root), daemon=True, name="moraine-status",
        )
        self._thread.start()

    def _run(self, port: int, root: Path) -> None:
        self._result = _serve_status(port, root)

    def result(self) -> dict | None:
        self._thread.join(max(0.0, self._deadline - time.monotonic()))
        return None if self._thread.is_alive() else self._result


def _pins_request(
    port: int, method: str, addressed: tuple[str | None, Path | None],
    query: str = "", body: dict | None = None,
) -> dict:
    """One request against the running server's pin surface (DESIGN 6),
    addressed to ``project`` (None: the displayed one), over the shared
    server-channel error mapping; a 200 that is not the pins contract
    reports `stale_server`."""
    project, root = addressed
    if project is not None:
        if body is not None:
            body = {**body, "project": project}
        else:
            sep = "&" if query else "?"
            query = f"{query}{sep}project={urllib.parse.quote(project)}"
    raw = _server_request(
        port, f"/api/pins{query}", method=method, body=body,
        timeout_code="pins_timeout", root=root,
    )
    try:
        out = json.loads(raw.decode("utf-8"))
        if not isinstance(out, dict):
            raise ValueError("not an object")
    except ValueError:
        raise MoraineError(
            "stale_server",
            f"the server on port {port} did not answer the pins contract; "
            f"it likely predates this CLI - restart 'moraine serve'",
        )
    return out


def _cmd_pins_list(args) -> int:
    report = _pins_request(args.port, "GET", _addressed_project(args.project))
    _print(report)
    # Root-less like `screenshot`; a pin leaves no event of its own
    # (DESIGN 6), so the summary is the only trace of the command.
    session.annotate(
        summary={"pins": len(report.get("pins") or []), "port": args.port}
    )
    return 0


def _cmd_pins_add(args) -> int:
    body: dict = {"output": args.output}
    if args.select is not None:
        try:
            body["predicate"] = json.loads(args.select)
        except json.JSONDecodeError as e:
            raise MoraineError(
                "bad_predicate",
                f"--select must be valid JSON ({e}); "
                f'example: {{"normal_y": {{"gt": 0.9}}}}',
            )
    if args.face is not None:
        body["face"] = args.face
    if args.instance is not None:
        body["instance"] = args.instance
    report = _pins_request(
        args.port, "POST", _addressed_project(args.project), body=body,
    )
    _print(report)
    session.annotate(summary={"pins": report.get("pins"), "port": args.port})
    return 0


def _cmd_pins_clear(args) -> int:
    query = f"?id={args.id}" if args.id is not None else ""
    report = _pins_request(
        args.port, "DELETE", _addressed_project(args.project), query=query,
    )
    _print(report)
    session.annotate(
        summary={"cleared": report.get("cleared"), "port": args.port}
    )
    return 0


def _cmd_serve(args) -> int:
    from .server import run

    _check_port(args.port)
    try:
        target = _resolve_target(args.project)
        workspace = None
        project = None
        if isinstance(target, workspace_mod.Workspace):
            if not target.projects:
                raise _no_projects_error(target)
            # Initial asset = first project. An ambiguous or missing
            # first asset fails as terse CLI JSON here, before the
            # server starts; an asset whose *content* is broken still
            # serves — the picker is the recovery path (DESIGN 7.2).
            workspace = target
            project = target.projects[0]
            target = workspace.resolve_asset(project)
        root = _target_root(workspace or target).resolve()
        try:
            run(target, port=args.port, workspace=workspace, project=project,
                open_browser=not args.no_open)
        except MoraineError as e:
            if e.code != "port_in_use":
                raise
            raise _port_owner_error(args.port, root, e) from None
    finally:
        # serve settles eviction through its own debounced background
        # check; disarm the session counter on every exit path —
        # resolution errors included — so the CLI tail never runs a
        # synchronous full-cache scan at serve shutdown.
        cache_mod.take_wrote()
    return 0


def _port_owner_error(port: int, root: Path, taken: MoraineError) -> MoraineError:
    """``port_in_use`` naming whose server holds the port (DESIGN 6):
    this root's is the one to use, another root's is not, and a
    listener that does not answer the status contract leaves the
    bind's own message."""
    status = _status(port)
    if status is None:
        return taken
    if status["root"] == root:
        return MoraineError(
            "port_in_use",
            f"a 'moraine serve' of this root already runs on port {port}: "
            f"use it, or pass --port <other> for a second server",
            root=str(root),
        )
    return MoraineError(
        "port_in_use",
        f"port {port} is held by the 'moraine serve' of another root "
        f"('{status['root']}'), which knows none of this root's projects; "
        f"pass --port <other> here, and the same --port to screenshot, "
        f"pins, validate, and stats",
        other_root=str(status["root"]),
    )


# On `no_viewer`: how long to wait for a tab that may already be on its
# way before opening one (a sibling session's recovery, or the human),
# how long to keep retrying once this command opened one, and the poll
# interval for both.
_VIEWER_JOIN_WAIT_S = 2.0
_VIEWER_OPEN_WAIT_S = 20.0
_VIEWER_OPEN_POLL_S = 1.0


def _screenshot_request(port: int, body: dict, root: Path | None = None) -> bytes:
    """The shot request, with the one recovery the error text asks for
    done by the CLI itself (DESIGN 6): on `no_viewer`, open the viewer
    in the human's browser and retry while the tab connects — the tab
    was the agent's render surface and its absence blocks the visual
    tier, so the agent takes the step it would otherwise have to ask
    the human for. It waits for a tab to appear before opening one:
    sessions share the viewer, so two shots racing on an empty server
    must not each add a tab. No browser (headless host, launcher
    failure): the original error stands, actionable as before."""
    def request() -> bytes:
        return _server_request(
            port, "/api/screenshot", method="POST", body=body,
            timeout_code="screenshot_timeout", root=root,
        )

    def until(deadline: float) -> bytes:
        """Retry while the answer stays `no_viewer`; the last one is
        raised once the budget is spent."""
        while True:
            try:
                return request()
            except MoraineError as e:
                if e.code != "no_viewer" or time.monotonic() >= deadline:
                    raise
            time.sleep(_VIEWER_OPEN_POLL_S)

    try:
        return until(time.monotonic() + _VIEWER_JOIN_WAIT_S)
    except MoraineError as e:
        if e.code != "no_viewer":
            raise
        first = e
    if not open_browser(f"http://127.0.0.1:{port}/"):
        raise first
    return until(time.monotonic() + _VIEWER_OPEN_WAIT_S)


def _cmd_screenshot(args) -> int:
    _check_port(args.port)
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        raise MoraineError("bad_size", "size must look like 512x512")
    for key in ("azimuth", "elevation", "distance"):
        v = getattr(args, key)
        if v is not None and not math.isfinite(v):
            # Fast local check; camera.resolve is the authority.
            raise MoraineError("bad_camera", f"{key} must be a finite number")
    if args.view is not None and (args.azimuth is not None or args.elevation is not None):
        # Fast local check; the server's camera.resolve is the authority
        # on the contract (DESIGN 7.2).
        raise MoraineError(
            "camera_conflict",
            "view and azimuth/elevation are exclusive; drop --view to "
            "place the camera freely",
        )
    body: dict = {"size": [w, h]}
    project, root = _addressed_project(args.project)
    if project is not None:
        body["project"] = project
    given = _given_params(args.param)
    if given:
        if project is None:
            raise MoraineError(
                "param_needs_project",
                "--param offers values to one project's knobs; name the "
                "project, or run from its directory",
            )
        body["params"] = _offered_values(sources_mod.resolve(args.project or "."), given)
    for key in ("view", "azimuth", "elevation", "distance"):
        v = getattr(args, key)
        if v is not None:
            body[key] = v
    if args.target is not None:
        try:
            target = [float(v) for v in args.target.split(",")]
        except ValueError:
            raise MoraineError("bad_camera", "target must look like 0,0.5,0")
        if len(target) != 3 or not all(math.isfinite(v) for v in target):
            raise MoraineError("bad_camera", "target must look like 0,0.5,0")
        body["target"] = target
    raw = _screenshot_request(args.port, body, root)
    try:
        envelope = json.loads(raw.decode("utf-8"))
        meta = envelope["meta"]
        png = base64.b64decode(envelope["png"], validate=True)
    except (ValueError, KeyError, TypeError):
        # Not the {meta, png} envelope: a 200 from a pre-upgrade server
        # (raw PNG body). The version-skew answer, like the viewer-side
        # protocol echo: name the fix, never write a corrupt file.
        raise MoraineError(
            "stale_server",
            f"the server on port {args.port} did not answer with the "
            f"expected JSON envelope; it likely predates this CLI - "
            f"restart 'moraine serve'",
        )
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
    except OSError as e:
        raise MoraineError(
            "write_failed",
            f"cannot write '{out}' ({e.__class__.__name__}); pass -o <file "
            f"path in a writable directory>",
        ) from e
    # Relay the server's provenance record whole — hand-picking fields
    # here would silently drop any meta the server adds later.
    _print({"screenshot": str(out), **meta, "bytes": len(png), "port": args.port})
    # Root-less and project-less by construction: the CLI cannot know
    # which root the answering server serves, so naming meta's project
    # here would let a foreign server's shot open rounds in another
    # workspace's report. The server's own ``screenshot`` event owns the
    # render's provenance and all per-project accounting; this event
    # contributes ambient command counts and ``out_bytes`` only.
    session.annotate(
        summary={
            **({"view": meta["view"]} if "view" in meta else {}),
            "size": [w, h], "bytes": len(png), "port": args.port,
        },
    )
    return 0


def _sync_file(dest: Path, src: bytes) -> str:
    """Factory-owned file sync (the guide): overwritten to the
    installed version when they differ, reported as
    created/updated/current, never merged (DESIGN 4.2)."""
    if not dest.exists():
        state = "created"
    elif dest.read_bytes() != src:
        state = "updated"
    else:
        return "current"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src)
    return state


def _cmd_init(args) -> int:
    """Scaffold a workspace from the shipped templates (DESIGN 4.2, 6).
    Ownership is per file: AGENTS.md is human-owned —
    created once, never overwritten. The authoring guide
    (`.moraine/guide.md`) is factory-owned — re-synced to the
    installed version on every init, reported, never merged. Harness
    instruction files are the human's, never scaffolded.
    The same run is the journal retirement pass (DESIGN 4.7): every
    journal is read (never written) and the workaround entries whose
    stamp is not the installed version are listed for re-verification.
    """
    from importlib import resources

    from ..core import journal as journal_mod

    target = Path(args.dir)
    if workspace_mod.is_project(target):
        # Scaffolding here would create the workspace/project layout
        # conflict (DESIGN 4.2) — refuse before writing anything.
        raise MoraineError(
            "workspace_asset_conflict",
            f"'{target}' holds a top-level asset file, so init would make it "
            "read as both a workspace root and a project; move the asset "
            "into a child project directory and re-run init",
        )
    created: list[str] = []
    skipped: list[str] = []
    # Filesystem failures (target is an existing file, a file named
    # .moraine, an unwritable directory) must surface as the CLI's terse
    # JSON errors, not tracebacks — same contract as cache_unwritable.
    try:
        template = resources.files("moraine3d") / "templates"
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(template.iterdir(), key=lambda e: e.name):
            if not entry.is_file():
                continue
            dest = target / entry.name
            if dest.exists():
                skipped.append(entry.name)
                continue
            dest.write_bytes(entry.read_bytes())
            created.append(entry.name)
        guide_state = _sync_file(
            target / workspace_mod.MORAINE_DIR / "guide.md",
            (resources.files("moraine3d") / "docs" / "guide.md").read_bytes(),
        )
    except OSError as e:
        raise MoraineError(
            "init_failed",
            f"cannot scaffold '{target}' ({e.__class__.__name__}: {e}); "
            "the target must be a writable directory (and '.moraine' must "
            "not be a file)",
        ) from e
    installed = dist_version()
    # The pass is advisory: an unreadable journal is reported on stderr
    # and skipped inside scan; a failure of the pass itself (workspace
    # discovery) must not turn a successful scaffold into a failure —
    # report it on stderr and keep the JSON contract whole.
    try:
        review, counts = journal_mod.scan(target, installed)
    except MoraineError as e:
        print(f"journal scan skipped: {e}", file=sys.stderr)
        review, counts = [], {"scanned": 0, "entries": 0, "unstamped": 0,
                              "dangling": 0, "skipped": 0,
                              "unstamped_by_journal": {}}
    _print({"initialized": str(target), "created": created,
            "skipped": skipped, "guide": guide_state,
            "version": installed,
            "journal_review": review, "journals": counts})
    return 0


# The shipped reference, one file per topic, in reading order (DESIGN
# 6): `moraine docs` lists them, a section name resolves across them.
REFERENCE_FILES = (
    "recipe.md", "asset.md", "ops.md", "textures.md", "checks.md",
    "assertions.md", "cli.md", "journal.md", "examples.md",
)
# The one section no file holds: the `mo` namespace, generated from
# the registries (recipe/api.py) so it cannot lag the build.
API_DOC = "api"
API_TITLE = "api (the `mo` namespace, generated from the registries)"


def reference_text(name: str) -> str:
    from importlib import resources

    return (resources.files("moraine3d") / "docs" / "reference" / name).read_text(
        encoding="utf-8"
    )


# A reference section cites its example as `examples/<name>`; the
# recipe lives in the package under docs/examples/<name>/<name>.star,
# and the section's reply names it for `moraine docs examples/<name>`.
EXAMPLE_CITATION = re.compile(r"`examples/([A-Za-z0-9_]+)`")


def example_assets() -> list[str]:
    """Shipped example asset files as `examples/...` paths, sorted (a
    requirements file is printed beside its asset, not listed)."""
    from importlib import resources

    base = resources.files("moraine3d") / "docs" / "examples"
    out = []
    for project in base.iterdir():
        if not project.is_dir():
            continue
        for f in project.iterdir():
            if sources_mod.is_asset_file(f.name):
                out.append(f"examples/{project.name}/{f.name}")
        subs = project / "subassets"
        if subs.is_dir():
            out.extend(
                f"examples/{project.name}/subassets/{f.name}"
                for f in subs.iterdir() if sources_mod.is_asset_file(f.name)
            )
    return sorted(out)


def example_text(path: str) -> str:
    """The text of a shipped asset by its `examples/...` path."""
    from importlib import resources

    rel = path.split("/")[1:]
    return (resources.files("moraine3d") / "docs" / "examples").joinpath(*rel).read_text(
        encoding="utf-8"
    )


def _example_exists(path: str) -> bool:
    from importlib import resources

    rel = path.split("/")[1:]
    return (resources.files("moraine3d") / "docs" / "examples").joinpath(*rel).is_file()


def _cited_examples(text: str) -> list[str]:
    """The shipped examples a section cites, as the `examples/<name>`
    a reader asks for, in citation order, once each."""
    shipped = set(example_assets())
    return [
        f"examples/{name}"
        for name in dict.fromkeys(EXAMPLE_CITATION.findall(text))
        if f"examples/{name}/{name}{sources_mod.RECIPE_EXT}" in shipped
    ]


def _print_example(query: str) -> int:
    """`moraine docs examples/<name>` / `examples/<name>/subassets/<sub>`
    / `examples/<name>/requirements` (the human's file beside the
    root asset)."""
    shipped = example_assets()
    q = query.rstrip("/")
    ext = sources_mod.RECIPE_EXT
    if q.endswith(ext):
        q = q[: -len(ext)]
    parts = q.split("/")
    candidates = [f"{q}/{parts[-1]}{ext}", f"{q}{ext}"] if len(parts) >= 2 else []
    hit = next((c for c in candidates if c in shipped), None)
    if (
        len(parts) == 3 and parts[-1] == "requirements"
        and f"examples/{parts[1]}/{parts[1]}{ext}" in shipped
    ):
        # The human's file beside a shipped root: the one path the
        # listing does not name, resolved under the project it names.
        req = f"examples/{parts[1]}/{parts[1]}{sources_mod.REQUIREMENTS_EXT}"
        if not _example_exists(req):
            raise MoraineError(
                "unknown_example",
                f"'{parts[1]}' ships no requirements file; 'moraine docs "
                f"examples/{parts[1]}' prints its root asset",
                example=parts[1],
            )
        hit = req
    if hit is None:
        names = sorted({p.split("/")[1] for p in shipped})
        near = [n for n in names if parts[-1].lower() in n.lower()] if len(parts) >= 2 else []
        hint = f"; did you mean: {', '.join(near)}" if near else ""
        raise MoraineError(
            "unknown_example",
            f"no shipped example '{query}'{hint}; 'moraine docs examples' lists "
            f"them, 'moraine docs examples/<name>' prints one",
            examples=names, **({"did_you_mean": near} if near else {}),
        )
    _print_text(example_text(hit).rstrip())
    return 0


def _cmd_docs(args) -> int:
    """The shipped reference (DESIGN 6): asset format, op contracts,
    checks, assertions, CLI, journal — one file per topic, sliced by
    section. The listing and an error are JSON; a section is its own
    text, one heading's worth, with the sections under it and the
    examples it cites named for the next call."""
    from ..core.journal import unfenced_lines

    docs: dict[str, list[str]] = {name: reference_text(name).splitlines() for name in REFERENCE_FILES}
    headings: list[tuple[str, int, int, str]] = []  # (file, line index, level, title)
    # A `#` line inside a fenced example (a journal layout, an asset
    # file) is content, not a section boundary.
    for name, lines in docs.items():
        for i, line in unfenced_lines(lines):
            m = re.match(r"^(#{2,3}) (.+)$", line)
            if m:
                headings.append((name, i, len(m.group(1)), m.group(2).strip()))
    names = [
        re.split(r"[\s(]", t, maxsplit=1)[0].rstrip(",:") for _, _, _, t in headings
    ]
    if not args.section:
        listing = [
            {"doc": name, "sections": [t for f, _, _, t in headings if f == name]}
            for name in REFERENCE_FILES
        ]
        # The generated listing of the `mo` namespace sits beside the
        # recipe's own file: what exists, then what each name means.
        listing.insert(1, {"doc": API_DOC, "sections": [API_TITLE]})
        _print({
            "docs": listing,
            "usage": "moraine docs <section> prints that section alone, as "
                     "text, and names the sections under it and the examples "
                     "it cites; <section> is the title's first "
                     "whitespace-delimited token or the full title (e.g. "
                     "'moraine docs place', 'moraine docs mated', 'moraine "
                     "docs checker_texture'); 'moraine docs api' lists every "
                     "builtin with its arguments; 'moraine docs examples' "
                     "lists the shipped assets, 'moraine docs "
                     "examples/<name>' prints one",
        })
        return 0
    if args.section.lower().startswith("examples/"):
        return _print_example(args.section)
    q = args.section.lower()
    if q in REFERENCE_FILES:
        # The index's `doc` is a file, not a section: answer with what
        # it holds, so the next call names one.
        _print({
            "doc": q,
            "sections": [t for f, _, _, t in headings if f == q],
            "usage": "moraine docs <section>; a file name lists its sections",
        })
        return 0
    if q == API_DOC or q == API_TITLE.lower():
        from ..recipe import api

        _print_text(api.text())
        return 0
    # Exact first-token or full-title match first; else a case-insensitive
    # full-title prefix that covers at least the first token (the guide
    # names sections like "Asset assertions" whose heading carries a
    # trailing qualifier; a partial token like `checker` stays a near miss).
    exact = [
        n for n, (_f, _i, _l, title) in enumerate(headings)
        if names[n].lower() == q or title.lower() == q
    ]
    prefix = [
        n for n, (_f, _i, _l, title) in enumerate(headings)
        if title.lower().startswith(q) and q.startswith(names[n].lower())
    ]
    hit = exact[:1] or (prefix if len(prefix) == 1 else [])
    if hit:
        doc, i, level, title = headings[hit[0]]
        lines = docs[doc]
        end = len(lines)
        under: list[str] = []
        for n, (f, j, lvl, _t) in enumerate(headings[hit[0] + 1:], hit[0] + 1):
            if f != doc:
                break  # a section never crosses its file
            if end == len(lines):
                end = j  # the section's own text stops at the next heading
            if lvl <= level:
                break
            under.append(_t.split(" (")[0])
        text = "\n".join(lines[i:end]).rstrip()
        tail = []
        held = _number_types(names[hit[0]])
        if held:
            # The number type each param holds (DESIGN 4.8), from the
            # declarations the build reads, so it cannot lag them.
            tail.append("numbers: " + ", ".join(f"{p} {k}" for p, k in held.items()))
        if under:
            tail.append("sections: " + ", ".join(under) + "  (moraine docs <section>)")
        cited = _cited_examples(text)
        if cited:
            tail.append("examples: " + ", ".join(cited) + "  (moraine docs examples/<name>)")
        _print_text(text + ("\n\n" + "\n".join(tail) if tail else ""))
        return 0
    # A prefix or substring of a section name or title is the common miss
    # (`checker` for `checker_texture`, `assertions` for `Asset
    # assertions`): name the candidates instead of sending the reader back
    # to the full listing — the first token when it carries the query,
    # else the full title (both resolve as a section argument).
    near = sorted({
        names[n] if q in names[n].lower() else title
        for n, (_f, _i, _l, title) in enumerate(headings)
        if q in names[n].lower() or q in title.lower()
    })
    hint = f"; did you mean: {', '.join(near)}" if near else ""
    raise MoraineError(
        "unknown_section",
        f"no section '{args.section}' in the shipped reference{hint}; "
        f"run 'moraine docs' to list sections",
        **({"did_you_mean": near} if near else {}),
    )


def _number_types(section: str) -> dict:
    """The number params of the operator a section documents, each
    with its type; empty for any other section."""
    from .. import ops
    from ..core import numbers

    if section not in ops.names():
        return {}
    return {
        param: kind for param, kind in ops.get(section).PARAM_KINDS.items()
        if kind != numbers.OTHER
    }


def _cmd_cache_stats(args) -> int:
    _print(cache_mod.stats())
    return 0


def _cmd_cache_clear(args) -> int:
    _print(cache_mod.clear())
    return 0


def _cmd_cache_evict(args) -> int:
    _print(cache_mod.evict())
    return 0


def _cmd_mcp(args) -> int:
    from .mcp import Server

    return Server(invoke).serve_stdio()


def invoke(argv: list[str], *, via: str) -> tuple[int, dict]:
    """Run one command in-process for another entry point (the MCP
    server, DESIGN 8.4): the same parser, handler, output contract and
    session-log ``command`` event as a shell invocation, marked ``via``
    (DESIGN 6) - but the printed object (a reference page: ``{"text":
    ...}``) is returned, not written, and
    anything else a handler prints is diverted to stderr, because the
    caller's stdout may be a transport. A usage error is an error object
    too (``bad_arguments``), never argparse's process exit."""
    session.reset()
    _out["bytes"], _out["last"], _out["quiet"] = 0, None, True
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                args = _parse_quietly(argv)
            except (_UsageError, SystemExit) as e:
                # SystemExit: a value spelled like '--help' reached
                # argparse's own exit; its text went to stderr.
                detail = e.message if isinstance(e, _UsageError) else "usage error"
                _print({"error": {
                    "code": "bad_arguments",
                    "message": f"{detail}; see the command's arguments",
                }})
                return 1, _out["last"]
            code = _run(args)
        _record_command(args, code, t0, via=via)
        _housekeep()
        return code, _out["last"] or {}
    finally:
        _out["quiet"] = False


class _UsageError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class _QuietParser(argparse.ArgumentParser):
    """The parser without its process exit on a usage error."""

    def error(self, message):
        raise _UsageError(message)


def _parse_quietly(argv: list[str]):
    parser = build_parser()
    parser.__class__ = _QuietParser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                sub.__class__ = _QuietParser
    return parser.parse_args(argv)


def _cmd_sessions(args) -> int:
    """Summary report over the session log (DESIGN 6): sessions and
    per-project rounds under a root, names only."""
    root = Path(args.path).resolve()
    if not root.is_dir():
        raise MoraineError(
            "no_such_directory",
            f"'{args.path}' is not a directory; pass a workspace root, a "
            "project directory, or a directory above them",
        )
    if args.gap <= 0:
        raise MoraineError("bad_gap", f"--gap must be positive minutes, got {args.gap}")
    if not args.all and args.last < 1:
        raise MoraineError("bad_last", f"--last must be at least 1, got {args.last}")
    # A workspace member's events are framed by the workspace root, so
    # its project directory narrows the workspace's sessions to that
    # project rather than selecting by path (which would find nothing).
    project = None
    if workspace_mod.is_project(root):
        enclosing = workspace_mod.enclosing_root(root)
        if enclosing != root:
            project = root.name
            root = enclosing
    d = session.log_dir()
    if args.all:
        events = session.read_events(d) if d is not None else []
        report = session.summarize(
            events, root, gap_minutes=args.gap, last=None, project=project,
        )
        if not report["sessions"]:
            report = None
    else:
        report = session.latest_report(
            root, gap_minutes=args.gap, last=args.last, project=project,
        )
    if report is None:
        where = f"log dir '{d}'" if d is not None else f"logging is off ({session.LOG_DIR_ENV} is empty)"
        raise MoraineError(
            "no_sessions",
            f"no session events under '{root}' ({where}); run moraine "
            "commands on a project there first, or pass a directory above it",
        )
    _print(report)
    return 0


def _housekeep() -> None:
    """Post-growth eviction check (DESIGN 4.5): runs after the command's
    output is flushed, only when this session wrote cache bytes and a cap
    is active. Housekeeping must never change the command's stdout or
    exit code — failures go to stderr as one line."""
    try:
        cache_mod.maybe_evict()
    except Exception as e:
        # Broad on purpose: whatever housekeeping raises, the command's
        # result already went to stdout and its exit code must stand.
        print(f"cache eviction skipped: {e}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moraine")
    # Same terse-JSON contract as every command (DESIGN 2 P2); the version
    # is the stamp journals carry (DESIGN 4.7), so it must be the first
    # flag that works.
    parser.add_argument(
        "--version",
        action="version",
        version=json.dumps({"version": dist_version()}, separators=(",", ":")),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    project_help = "project directory holding one root asset (<name>.star), or an asset file path (default: cwd)"
    sweep_help = project_help + "; a workspace root sweeps every project into one report"
    server_port_help = ("port of the running serve process (default: "
                        "%(default)s); must match its --port")

    p = sub.add_parser("init",
                       help="scaffold a workspace: AGENTS.md created once, "
                            "guide synced to the installed version")
    p.add_argument("dir", nargs="?", default=".",
                   help="target workspace directory (default: cwd)")
    p.set_defaults(fn=_cmd_init)

    p = sub.add_parser("build",
                       help="build a recipe and its reference closure into "
                            "IR under each project's build/")
    p.add_argument("project", nargs="?", default=".", help=project_help)
    p.add_argument("--diff", nargs="?", const="", default=None, metavar="PATH",
                   help="report the graph change since the last build: "
                        "nodes and asserts added, removed, and changed, and "
                        "the outputs made dirty; PATH names the IR file or "
                        "the build/ directory to compare against instead of "
                        "the IR on disk")
    p.add_argument("--check", action="store_true",
                   help="build to memory and compare with the IR on disk, "
                        "byte for byte; writes nothing, and a difference "
                        "exits 1")
    p.set_defaults(fn=_cmd_build)

    recompute_help = ("recompute every node ignoring cached results and compare "
                      "result hashes with the records on disk (the determinism "
                      "check); the cache is never cleared or overwritten, and a "
                      "mismatch exits 1")
    p = sub.add_parser("eval", help="evaluate asset, populate cache")
    p.add_argument("--recompute", action="store_true", help=recompute_help)
    p.add_argument("project", nargs="?", default=".", help=project_help)
    p.add_argument("--output", default=None,
                   help="evaluate only the subgraph of one named output; "
                        "the report is then flagged partial")
    p.set_defaults(fn=_cmd_eval)

    # Every runtime string in the package stays pure ASCII, help strings
    # included: argparse writes help raw to stdout, and a legacy-codepage
    # Windows console (e.g. cp932) hard-crashes on anything wider; JSON
    # output is escaped (json.dumps), but its consumers re-print the text
    # on the same console. Pinned by test_every_help_page_is_ascii
    # (tests/test_cli.py) and test_runtime_strings_are_ascii
    # (tests/test_hardening.py).
    status_port_help = ("port of a running serve process whose file-watch "
                        "liveness the report carries under 'serve' (default: "
                        "%(default)s); no server on the port, no key")
    p = sub.add_parser("stats", help="tri/vert counts, bounds - compact JSON")
    p.add_argument("project", nargs="?", default=".", help=sweep_help)
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=status_port_help)
    p.set_defaults(fn=_cmd_stats)

    p = sub.add_parser("validate", help="deterministic geometry checks - compact JSON")
    p.add_argument("project", nargs="?", default=".", help=sweep_help)
    # No argparse choices: export.PROFILES is the roster's single source,
    # and its unknown_profile error already lists it — a hardcoded copy
    # here silently strands any profile added there.
    p.add_argument("--profile", default=None,
                   help="also run export-profile checks")
    p.add_argument("--failures-only", action="store_true",
                   help="collapse healthy outputs to a count; detail only on failure")
    p.add_argument("--param", action="append", metavar="ID=VALUE",
                   help="judge the asset under this value of a declared knob, "
                        "over an accepted one, writing no file (repeatable; "
                        "VALUE is JSON, e.g. 2.4 or [1,2,3], and for a string "
                        "knob the text itself); one project only")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=status_port_help)
    p.set_defaults(fn=_cmd_validate)

    p = sub.add_parser(
        "query",
        help="read-only face-predicate query over evaluated outputs - compact JSON "
        "(same vocabulary as the tag op)",
    )
    p.add_argument("project", nargs="?", default=".", help=project_help)
    p.add_argument(
        "--select", default=None,
        help='face predicate JSON, e.g. {"normal_y": {"gt": 0.9}}',
    )
    p.add_argument("--tag", default=None, metavar="NAME",
                   help="the faces a tag caught, by its name; with --select, "
                        "those of them the predicate matches")
    p.add_argument("--instance", type=int, default=None, metavar="I",
                   help="read copy I of an instanced output where it stands "
                        "in the world (default: the prototype, local frame)")
    p.add_argument("--output", default=None, help="restrict to one named output")
    p.set_defaults(fn=_cmd_query)

    p = sub.add_parser(
        "export",
        help="write glTF (GLB) + manifest for an export profile; "
        "requires a green validate (waivers included)",
    )
    p.add_argument("project", nargs="?", default=".", help=project_help)
    # Roster deliberately not copied into help (as for validate --profile):
    # the unknown_profile error is its single source of truth.
    p.add_argument("--profile", default="gltf",
                   help="export profile (default: gltf)")
    p.add_argument("-o", "--out", default="exports",
                   help="output directory; a relative path resolves "
                        "against the project (default: exports)")
    p.add_argument("--bake-instances", action="store_true",
                   help="project instanced outputs to plain meshes at the "
                        "export edge (compatibility path for importers "
                        "without EXT_mesh_gpu_instancing); default keeps "
                        "instancing")
    p.add_argument("--recompute", action="store_true", help=recompute_help)
    p.set_defaults(fn=_cmd_export)

    p = sub.add_parser("serve",
                       help="HTTP+WS server + viewer, watches asset files; "
                            "a workspace root serves all projects with a "
                            "viewer picker")
    p.add_argument("project", nargs="?", default=".",
                   help=project_help + "; a workspace root serves all its "
                        "projects (switch in the viewer)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="listen port (default: %(default)s); commands that "
                        "talk to this server pass the same --port")
    p.add_argument("--no-open", action="store_true",
                   help="do not open the viewer in a browser (by default it "
                        "opens once serving, unless a viewer tab is already "
                        "connected)")
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("screenshot",
                       help="render a project's current file state via the running server")
    p.add_argument("project", nargs="?", default=None,
                   help="project to shoot (default: the current directory's; "
                        "outside any project, the one the viewer displays)")
    p.add_argument("--view", default=None, choices=list(camera_mod.VIEWS),
                   help="named view preset (default: iso); exclusive with --azimuth/--elevation")
    p.add_argument("--azimuth", type=float, default=None,
                   help="free camera: degrees, 0 looks from +Z (front), 90 from +X (right)")
    p.add_argument("--elevation", type=float, default=None,
                   help="free camera: degrees above horizontal, -90..90 (90 = top)")
    p.add_argument("--distance", type=float, default=None,
                   help="camera distance in meters (default: auto-fit); combines with any view")
    p.add_argument("--target", default=None,
                   help="look-at point as X,Y,Z meters (default: asset bounds center); write --target=X,Y,Z when X is negative")
    p.add_argument("--size", default="512x512",
                   help="pixels as WIDTHxHEIGHT, e.g. 900x900 (default 512x512)")
    p.add_argument("-o", "--out", default="shot.png")
    p.add_argument("--param", action="append", metavar="ID=VALUE",
                   help="shoot the asset under this value of a declared knob, "
                        "over an accepted one, writing no file and moving no "
                        "viewer (repeatable; VALUE as for validate --param)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=server_port_help)
    p.set_defaults(fn=_cmd_screenshot)

    p = sub.add_parser("docs",
                       help="the shipped reference: the listing is compact JSON, "
                            "a section prints as text")
    p.add_argument("section", nargs="?", default=None,
                   help="section name, e.g. 'place' (omit to list sections); "
                        "'examples' lists the shipped example assets, "
                        "'examples/<name>' prints one")
    p.set_defaults(fn=_cmd_docs)

    p = sub.add_parser(
        "mcp",
        help="MCP server over stdio: the read/verify commands "
             "(stats, validate, query, export, screenshot, "
             "docs) as tools; configure a client with "
             "command 'moraine', args ['mcp']",
        # This help page is the MCP contract's in-workspace carrier
        # (DESIGN 8.4; the guide says nothing about MCP by design).
        description=(
            "MCP server over stdio (newline-delimited JSON-RPC, tools only).\n"
            "Tools: stats, validate, query, export, screenshot, docs - each is\n"
            "the CLI command of the same name run in-process: the command's\n"
            "arguments as a JSON object, the same compact report as text;\n"
            "screenshot also returns the PNG as image content. Use whichever\n"
            "surface the harness gives you, not both. Assets are still edited\n"
            "as files."
        ),
        epilog=(
            "Configure a client with the installed command and the argument mcp:\n"
            "  claude mcp add moraine -- moraine mcp\n"
            '  {"mcpServers": {"moraine": {"command": "moraine", "args": ["mcp"]}}}'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(fn=_cmd_mcp)

    p = sub.add_parser("sessions",
                       help="session-log summary: sessions, rounds, command counts "
                            "under a directory - compact JSON")
    p.add_argument("path", nargs="?", default=".",
                   help="workspace root, project directory, or a directory above "
                        "them (default: cwd)")
    p.add_argument("--gap", type=float, default=session.DEFAULT_GAP_MINUTES,
                   help="idle minutes that end a session (default: %(default)s)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--last", type=int, default=1,
                       help="report the N most recent sessions (default: 1)")
    group.add_argument("--all", action="store_true", help="report every session")
    p.set_defaults(fn=_cmd_sessions)

    addressed_help = ("project the pins belong to (default: the current "
                      "directory's); refused unless the viewer displays it")
    p = sub.add_parser("pins",
                       help="deixis pins on the running server: read the "
                            "human's, place the agent's - compact JSON")
    pps = p.add_subparsers(dest="pins_command", required=True)
    pp = pps.add_parser("list",
                        help="the project's pins with the current revision hash "
                             "(empty unless the viewer displays it)")
    pp.add_argument("project", nargs="?", default=None, help=addressed_help)
    pp.add_argument("--port", type=int, default=DEFAULT_PORT, help=server_port_help)
    pp.set_defaults(fn=_cmd_pins_list)
    pp = pps.add_parser("add", help="place a pin by face predicate or face index")
    pp.add_argument("project", nargs="?", default=None, help=addressed_help)
    pp.add_argument("--output", required=True,
                    help="output name the pin points into")
    sel = pp.add_mutually_exclusive_group(required=True)
    sel.add_argument("--select", default=None,
                     help='face predicate JSON, e.g. {"normal_y": {"gt": 0.9}}; '
                          "the pin lands on the matched faces' centroid")
    sel.add_argument("--face", type=int, default=None,
                     help="face index on the output")
    pp.add_argument("--instance", type=int, default=None,
                    help="copy index where the output is instanced")
    pp.add_argument("--port", type=int, default=DEFAULT_PORT, help=server_port_help)
    pp.set_defaults(fn=_cmd_pins_add)
    pp = pps.add_parser("clear", help="remove one pin (--id) or all")
    pp.add_argument("project", nargs="?", default=None, help=addressed_help)
    pp.add_argument("--id", type=int, default=None,
                    help="pin id to remove (omit to clear all)")
    pp.add_argument("--port", type=int, default=DEFAULT_PORT, help=server_port_help)
    pp.set_defaults(fn=_cmd_pins_clear)

    p = sub.add_parser("cache",
                       help="evaluation cache maintenance - compact JSON")
    pcs = p.add_subparsers(dest="cache_command", required=True)
    pc = pcs.add_parser("stats", help="cache size, file and key counts, cap")
    pc.set_defaults(fn=_cmd_cache_stats)
    pc = pcs.add_parser("clear", help="delete all cache records (everything re-derives)")
    pc.set_defaults(fn=_cmd_cache_clear)
    pc = pcs.add_parser("evict", help="run LRU eviction down to the size cap now")
    pc.set_defaults(fn=_cmd_cache_evict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session.reset()
    _out["bytes"], _out["last"] = 0, None
    t0 = time.perf_counter()
    code = _run(args)
    # After output is flushed, and on error paths too (bytes may have been
    # written before the failure). serve disarms the counter on return
    # (its debounced check owns eviction), so its exit stays instant.
    try:
        sys.stdout.flush()
    except OSError:
        pass  # reader closed the pipe early; the command's work is done
    if args.command not in ("serve", "mcp"):
        # serve's record is its server's own events (start, reloads,
        # stop); a command event stamped at its shutdown would misplace
        # hours of wall time. mcp records one event per tool call for
        # the same reason. Everything else: one event, every exit path.
        _record_command(args, code, t0)
    _housekeep()
    return code


def _run(args) -> int:
    """One parsed command to its exit code, under the output contract:
    every failure is one printed error object and code 1, never a
    traceback on stdout (DESIGN 2 P2). Shared by main() and the MCP
    server, which relays the printed object instead of stdout."""
    try:
        code = args.fn(args)
    except MoraineError as e:
        _print({"error": e.to_dict()})
        code = 1
    except MemoryError:
        # A huge count/segments/resolution exhausted memory inside a
        # kernel: still the terse-JSON contract and exit 1, never a raw
        # numpy traceback that exits 0.
        _print({"error": {
            "code": "out_of_memory",
            "message": "evaluation ran out of memory; reduce "
                       "counts/segments/resolution in the asset",
        }})
        code = 1
    except Exception as e:
        # Last resort: any other exception is a Moraine bug. The JSON
        # contract and exit code still hold; the traceback goes to stderr
        # so the report is not lost.
        traceback.print_exc()
        _print({"error": {
            "code": "internal_error",
            "message": f"{type(e).__name__}: {e}; this is a Moraine bug "
                       f"- report it with the asset",
        }})
        code = 1
    return code


def _record_command(args, code: int, t0: float, **extra) -> None:
    cmd = args.command
    if cmd == "cache":
        cmd = f"cache.{args.cache_command}"
    elif cmd == "pins":
        cmd = f"pins.{args.pins_command}"
    last = _out["last"] or {}
    err = last.get("error") if isinstance(last.get("error"), dict) else None
    notes = session.take_annotations()
    projects = notes.pop("projects", None)
    sweep = notes.pop("sweep", False)
    if projects and not sweep and len(projects) == 1:
        # One project evaluated: flatten to the single-project shape.
        (name, fields), = projects.items()
        notes.update({"project": name, **fields})
    elif projects:
        notes["projects"] = projects
    session.record(
        "command", cmd=cmd, ok=code == 0,
        code=err.get("code") if err else None,
        ms=round((time.perf_counter() - t0) * 1000, 1),
        out_bytes=_out["bytes"], **extra, **notes,
    )


if __name__ == "__main__":
    sys.exit(main())
