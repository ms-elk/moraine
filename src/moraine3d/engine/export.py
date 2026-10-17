"""glTF 2.0 export with per-target profiles (DESIGN 9).

One asset exports to one GLB: each named output is one part — a glTF
root node named after the output (Day-One Invariants 7, 8; v1 hierarchy
is exactly the named-output level, every part's pivot is the asset
origin, so part node transforms are identity). Instanced outputs keep
instancing via ``EXT_mesh_gpu_instancing`` (Invariant 6). The GLB writer
itself lives in ``glb.py`` and is shared with the viewer.

Materials and textures ride in the same GLB (see glb.py for
the encoding conventions). The ``gltf`` profile emits glTF-native
metallic-roughness only — no separate MetallicSmoothness conversion:
the verified import path is Unity/URP via glTFast, which performs that
conversion itself at import; a sidecar texture set would be dead weight
on that path. A profile is a measured delta, named after whatever the
delta is specific to (DESIGN 9): a non-glTFast Unity path is what
would give a ``unity`` profile its first content.

Alongside the GLB goes a manifest recording the moraine version, every
op impl version used, and kernel versions — the determinism boundary of
DESIGN 4.4 made explicit per export — plus the GLB's content hash,
binding the artifact pair (DESIGN 9). zlib counts as a kernel:
PNG bytes (not pixel values) depend on its version.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

from .. import dist_version, ops
from ..core.atomic import ReplaceError, atomic_replace
from ..core.glb import build_glb, decompose_trs
from ..core.mesh import Image, Instances, Value, proto_of
from ..core.asset import Asset
from ..errors import MoraineError
from ..verify.validate import add_by_site
from . import validation
from .evaluator import Evaluation, evaluate_op_value

PROFILES = ("gltf",)


# -- profile validation ------------------------------------------------------


def profile_checks(
    profile: str,
    values: dict[str, Value],
    report: dict,
    bake_instances: bool = False,
) -> dict:
    """Append per-output profile checks to a tier-1 report (DESIGN 9.1).

    Blocking: image outputs (not exportable parts), missing normals
    (the importer shades them flat), undecomposable instance transforms
    (skipped under ``bake_instances`` — the export projects instances to
    plain meshes, so TRS-representability is irrelevant), missing UVs on
    a *textured* output. Advisory (``level: "warn"``, does not gate):
    missing UVs on untextured outputs. Material identity (name
    uniqueness, the declared material budget) is profile-independent and
    lives in plain ``validate`` (validate_outputs, DESIGN 9.1).
    """
    if profile not in PROFILES:
        raise MoraineError(
            "unknown_profile", f"unknown profile '{profile}'; known: {list(PROFILES)}"
        )
    for name, value in values.items():
        out = report["outputs"][name]
        checks = out["checks"]

        if isinstance(value, Image):
            checks.append(
                {
                    "check": "mesh_output",
                    "ok": False,
                    "fix": "output is an image, and GLB parts are meshes; "
                    "feed it into a material node and remove it from the "
                    "outputs (keep image outputs only while iterating on a "
                    "texture)",
                }
            )
            out["ok"] = False
            continue

        proto = proto_of(value)
        mat = proto.material

        has_normals = proto.normals is not None
        checks.append(
            {
                "check": "normals_present",
                "ok": has_normals,
                **(
                    {}
                    if has_normals
                    else {"fix": "output has no vertex normals; add smooth_normals"}
                ),
            }
        )
        if isinstance(value, Instances) and not bake_instances:
            try:
                decompose_trs(value.transforms, name)
                checks.append({"check": "instances_trs", "ok": True})
            except MoraineError as e:
                checks.append({"check": "instances_trs", "ok": False, "fix": e.message})
        if proto.uv is None:
            textured = mat is not None and bool(mat.textures)
            checks.append(
                {
                    "check": "uv_present",
                    "ok": False,
                    **({} if textured else {"level": "warn"}),
                    "fix": (
                        "textured output has no UVs; add unwrap before the "
                        "material node"
                        if textured
                        else "no UVs; fine untextured - unwrap before texturing"
                    ),
                }
            )
        out["ok"] = all(c["ok"] for c in checks if c.get("level") != "warn")

    # Profile checks only add failures: AND them into the incoming ok so
    # the asserts and budgets gates from validate_outputs survive —
    # asserts are documented as gating and unwaivable, and export is
    # exactly the path that ships the asset.
    report["ok"] = report["ok"] and all(o["ok"] for o in report["outputs"].values())
    report["profile"] = profile
    return report


# -- manifest ----------------------------------------------------------------


def build_manifest(
    ev: Evaluation, profile: str, glb_file: str, glb_hash: str, info: dict,
    *, bake_instances: bool = False,
) -> dict:
    """The determinism boundary of DESIGN 4.4, explicit per export.
    ``glb_hash`` binds the manifest to the exact GLB bytes it describes
    (the artifact-pair contract, DESIGN 9)."""
    # Evaluated nodes only: an unreachable node (which may name an
    # unknown op) contributes nothing to the artifact and must not be
    # able to fail the manifest.
    versions = dict(ev.ops_used())
    if bake_instances:
        # The projection shapes the exported bytes, so its op version
        # joins the recorded boundary even with no graph node using it.
        versions["bake_instances"] = ops.bake_instances.OP_IMPL_VERSION
    return {
        "moraine_version": dist_version("moraine3d"),
        # The build that wrote the IR the export evaluated (DESIGN 4.4).
        "build_version": ev.root.build_version,
        "profile": profile,
        "glb": glb_file,
        # 24-hex truncated sha256 of the GLB bytes, the repo's identity
        # format: a pair whose hash does not match the GLB on disk is an
        # interrupted export — re-run to heal (DESIGN 9).
        "glb_hash": glb_hash,
        # Export-edge projection flag (DESIGN 9): byte-identical re-export
        # holds per flag value, so the flag is part of the recorded identity.
        "bake_instances": bake_instances,
        "outputs": info,
        "op_impl_versions": versions,
        "kernel_versions": {
            "numpy": dist_version("numpy"),
            "trimesh": dist_version("trimesh"),
            "manifold3d": dist_version("manifold3d"),
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
        },
    }


# -- entry point -------------------------------------------------------------



def _write_out(path: Path, data: bytes) -> None:
    """Write one export file atomically (``core.atomic``), with the export
    directory's failure policy. Export names are not content-addressed
    (same name, new bytes), so a destination another program holds open
    (the engine importer) is a loud ``export_locked`` failure, never a
    benign lost race; an unwritable directory is ``export_unwritable``.
    Either way the in-flight temp file must not linger beside the user's
    files — it is Moraine's own file outside the cache, so it is unlinked
    directly (the cache choke point would refuse it)."""
    try:
        atomic_replace(path, data)
    except ReplaceError as err:
        err.tmp.unlink(missing_ok=True)
        e = err.cause
        if err.step == "write":
            raise MoraineError(
                "export_unwritable",
                f"cannot write '{path.name}' in '{path.parent}' "
                f"({e.__class__.__name__}); pass -o <a writable directory>",
            ) from e
        raise MoraineError(
            "export_locked",
            f"cannot replace '{path.name}' ({e.__class__.__name__}: held "
            f"open by another program, e.g. the engine importer); close it "
            f"and re-run export",
        ) from e


def in_export_order(values: dict[str, Value]) -> dict[str, Value]:
    """The outputs in the order an export writes them (DESIGN 9.1): by
    name, case folded, which is a total order since names are unique
    case-insensitively (DESIGN 4.1). The order a recipe declares its
    outputs in moves when the recipe is refactored, and the bytes of an
    export must not move with it."""
    return {name: values[name] for name in sorted(values, key=str.lower)}


def _report_path(p: Path) -> str:
    """Report paths cwd-relative when the target is inside cwd - the
    in-project and workspace-root flows then quote journal-safely
    (DESIGN 6); a target outside cwd is echoed as resolved (the author
    pointed there). Both sides are resolved before comparing - the
    target comes resolved via project_root, and a logical cwd (subst
    drive, junction) would otherwise never match it (the workspace
    discovery precedent)."""
    cwd = Path.cwd().resolve()
    return str(p.relative_to(cwd)) if p.is_relative_to(cwd) else str(p)


def export_asset(
    sc: Asset, profile: str, out_dir: str | Path, *,
    evaluation: Evaluation, bake_instances: bool = False,
) -> dict:
    """Validate against the profile, then write ``<stem>.glb`` + manifest.

    Validation failure writes nothing and returns the failing report
    (``ok: false``) — the error channel is the report, the last exported
    files stay good (DESIGN 8.2).

    ``bake_instances`` (DESIGN 9) projects every Instances output to a
    plain mesh at the export edge — tiled per the ``bake_instances`` op's
    semantics with the material re-bound as in-graph evaluation would,
    manifest-recorded (flag and op version), cache keys untouched — the
    compatibility path for importers without ``EXT_mesh_gpu_instancing``.
    The default stays keep (Invariant 6 governs the default path).
    Validation and asserts run on the authored values either way: the
    projection changes delivery format, never exportability, so only the
    TRS-representability check (which baking makes irrelevant) is
    skipped; per-output info reports what the GLB actually holds.

    ``evaluation``: the caller's ``Evaluation`` of ``sc`` to evaluate
    through, so the caller can read what it reached (the asset files,
    for the revision record).
    """
    if profile not in PROFILES:
        raise MoraineError(
            "unknown_profile", f"unknown profile '{profile}'; known: {list(PROFILES)}"
        )
    ev = evaluation
    records = ev.records()
    report, values = validation.report(sc, ev)

    baked = (
        {n for n, v in values.items() if isinstance(v, Instances)}
        if bake_instances else set()
    )

    report = profile_checks(profile, values, report, bake_instances=bake_instances)
    if not report["ok"]:
        return {"ok": False, "validation": add_by_site(report)}

    stem = sc.path.stem if sc.path is not None else "asset"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if baked:
        for name in baked:
            # The same evaluate-and-carry step the in-graph bake_instances
            # node runs (material re-bound per its preserved policy), so
            # the export edge can never drift from the op's semantics.
            values[name] = evaluate_op_value(
                ops.bake_instances, [values[name]], {}, sc.outputs[name]
            )

    generator = f"moraine {dist_version('moraine3d')}"
    glb, info = build_glb(in_export_order(values), generator)
    for name, rec in records.items():
        stats = rec.stats
        if name in baked:
            # Report what ships: the cached record keeps instanced stats
            # (baking never touches the cache), but the GLB holds the
            # merged mesh — its own stats, plus the flag.
            stats = {**values[name].stats(), "baked": True}
        info[name] = {**stats, **info[name], "node_id": sc.outputs[name]}
    # The artifact-pair contract (DESIGN 9): both payloads are built
    # before either write (a build failure writes nothing); each write is
    # atomic but the pair is not — the filesystem has no two-file
    # transaction — so the GLB goes first and the manifest last as the
    # pair's commit marker, with glb_hash binding it to the exact GLB
    # bytes. An export interrupted between the writes leaves a
    # detectably stale pair (hash mismatch), never a silent one.
    glb_path = out / f"{stem}.glb"
    manifest = build_manifest(
        ev, profile, glb_path.name,
        hashlib.sha256(glb).hexdigest()[:24], info,
        bake_instances=bake_instances,
    )
    manifest_bytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    _write_out(glb_path, glb)
    manifest_path = out / f"{stem}.manifest.json"
    try:
        _write_out(manifest_path, manifest_bytes)
    except MoraineError as e:
        raise MoraineError(
            e.code,
            f"{e.message}; the GLB was already replaced, so the manifest "
            f"on disk is stale (its glb_hash will not match) until a "
            f"re-run lands both files",
            **e.detail,
        ) from e

    result: dict = {
        "ok": True,
        "glb": _report_path(glb_path),
        "manifest": _report_path(manifest_path),
        "outputs": info,
    }
    warns = [
        {**c, "output": name}
        for name, o in report["outputs"].items()
        for c in o["checks"]
        if c.get("level") == "warn" or c.get("waived")
    ]
    # Asset-level waivers surface like per-output ones (no output to
    # name): an export must not be quieter than the validate it gates on.
    warns += [c for c in report.get("asset_checks") or [] if c.get("waived")]
    if report.get("warnings"):
        warns += report["warnings"]
    if warns:
        result["warnings"] = warns
    return result
