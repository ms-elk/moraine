"""asset_ref: place a named output of another asset in this graph (DESIGN 4.6).

A zero-input node whose params name another asset and one of its named
outputs. Its result is that output's value, Mesh or Instances alike —
instancing passes through untouched. References are opaque (the
referenced asset evaluates in its own namespace) and never leave the
workspace. Two mutually exclusive location forms:

- ``asset`` — an in-project path, exactly ``subassets/<name>.mora``
  in the IR, which the build writes from the recipe's
  ``subassets/<name>.star`` (the layout convention of DESIGN 4.2 is
  enforced here).
- ``project`` — a sibling project's directory name (its discovery
  identity, never a path); the target is a named output of that
  project's *root* asset. ``subassets/`` stay project-private. In the
  IR the build writes that asset's IR beside it as ``asset``, a bare
  file name ``<name>.mora``, so the reference resolves from the IR
  alone.

``overrides`` optionally maps the referenced asset's declared param ids
to per-use values (one interface, two consumers — the same declarations
drive the viewer panel). Only the shape is validated here; values are
checked against the referenced asset's declarations by the build.

``material_rename`` optionally maps material names found on the
received output to per-use names (DESIGN 4.6): the consumer's
explicit per-use identity — the Material-Instance shape — for the case
where overrides reach a material and the shared name would otherwise
bind two contents. It is applied to the *received value* at the
reference boundary (``rename_materials``), never inside the referenced
asset and never as part of its override identity: a renamed ref is a
one-input derived record over the referenced output's record, so
un-renamed refs keep pure aliasing. Every key must name a material
present on the received output (a Mesh or Instances carries one, so
the map is effectively singleton today; unknown keys fail loud).

The recursion into the referenced asset lives in the evaluator (it owns
asset loading, the cross-file memo, and cycle detection); this module
owns the parameter contract, path resolution, and the rename transform,
so the rules stay testable without an evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from typing import NamedTuple

from ..core import numbers
from ..core.mesh import Image, Value, bind_material, proto_of
from ..core.asset import (
    BUILD_DIR, MORAINE_DIR, RESERVED_DIRS, ASSET_EXT, SUBASSETS_DIR,
    ir_home, project_root,
)
from ..errors import OpError, AssetError
from . import common

# A plain reference never enters a cache key (it aliases the referenced
# record), so this version keys only the material_rename derived record
# (evaluator._eval_ref): bump it when rename_materials' output changes.
OP_IMPL_VERSION = 2
KERNELS = ()
INPUT_ARITY = (0, 0)
PARAM_KINDS = {
    "asset": numbers.OTHER,
    "project": numbers.OTHER,
    "output": numbers.OTHER,
    "overrides": numbers.OTHER,
    "material_rename": numbers.OTHER,
}
PARAMS = frozenset(PARAM_KINDS)
ATTRIBUTE_POLICY = "preserved"  # identity over the referenced output
# The value originates in another asset: the evaluator routes the node
# through its reference path, and graph walks / the material-origin
# walk (Evaluation.material_origins) treat it as a file boundary.
IS_REFERENCE = True


def authors_material(params: dict) -> bool:
    """A renamed ref binds a material identity in the *consuming* asset —
    the consumer named it, so no other asset audits it (DESIGN 9.1)."""
    return bool(params.get("material_rename"))


class RefSpec(NamedTuple):
    """Parsed asset_ref location params: a sub-asset's path, or a
    sibling project and, where the build has resolved it, the file
    name of its root asset's IR."""

    asset: str | None
    project: str | None
    output: str
    overrides: dict
    material_rename: dict


ONE_LOCATION = (
    "asset_ref needs exactly one of 'asset' (a subassets/ path) or "
    "'project' (a sibling project's directory name)"
)


def ref_params(params: dict) -> RefSpec:
    """Validate the location params and return the parsed reference spec."""
    asset = params.get("asset")
    project = params.get("project")
    output = params.get("output")
    if asset is None and project is None:
        raise OpError("bad_param", ONE_LOCATION, param="asset")
    if not isinstance(output, str) or not output:
        raise OpError(
            "bad_param",
            "asset_ref 'output' must name an output of the referenced asset",
            param="output",
        )
    if project is None:
        if not isinstance(asset, str) or not asset:
            raise OpError(
                "bad_param",
                "asset_ref 'asset' must be a project-relative path like 'subassets/part.mora'",
                param="asset",
            )
        if "\\" in asset:
            raise OpError(
                "bad_param",
                "asset_ref 'asset' must use forward slashes (portable across platforms)",
                param="asset",
            )
        if not asset.endswith(ASSET_EXT):
            raise OpError(
                "bad_param",
                f"asset_ref 'asset' must point at a {ASSET_EXT} file",
                param="asset",
            )
    else:
        if not isinstance(project, str) or not project:
            raise OpError(
                "bad_param",
                "asset_ref 'project' must name a sibling project by its "
                "directory name (its workspace discovery identity)",
                param="project",
            )
        if "/" in project or "\\" in project:
            raise OpError(
                "bad_param",
                "asset_ref 'project' names a sibling project by its directory "
                "name, never a path; the target is a named output of that "
                "project's root asset",
                param="project",
            )
        if asset is not None and (
            not isinstance(asset, str) or "/" in asset or "\\" in asset
            or not asset.endswith(ASSET_EXT) or asset == ASSET_EXT
        ):
            raise OpError(
                "bad_param",
                f"{ONE_LOCATION}; beside 'project' the IR carries the file "
                f"name of that project's root IR, '<name>{ASSET_EXT}', "
                f"which the build writes",
                param="asset",
            )
    overrides = params.get("overrides", {})
    if not isinstance(overrides, dict) or not all(isinstance(k, str) for k in overrides):
        raise OpError(
            "bad_param",
            "asset_ref 'overrides' must be an object mapping the referenced "
            "asset's declared param ids to values",
            param="overrides",
        )
    rename = params.get("material_rename", {})
    if not isinstance(rename, dict) or not all(
        isinstance(k, str) and k for k in rename
    ):
        raise OpError(
            "bad_param",
            "asset_ref 'material_rename' must be an object mapping a material "
            "name on the received output to its per-use name, like "
            '{"hull_panel": "hull_panel_b"}',
            param="material_rename",
        )
    for old, new in rename.items():
        common.material_name(
            "asset_ref", new, "material_rename",
            what=f"'material_rename' target for '{old}'",
        )
        if new == old:
            raise OpError(
                "bad_param",
                f"asset_ref 'material_rename' maps '{old}' to itself; drop the "
                f"entry (a rename that changes nothing is not a per-use identity)",
                param="material_rename",
            )
    if len(set(rename.values())) != len(rename):
        raise OpError(
            "bad_param",
            "asset_ref 'material_rename' targets must be distinct - two "
            "materials renamed to one name would collide as one identity",
            param="material_rename",
        )
    return RefSpec(asset, project, output, overrides, rename)


def rename_materials(value: Value, mapping: dict) -> Value:
    """Apply a per-use material rename to a received reference value.

    Only the material ``name`` changes — factors and textures are
    carried verbatim, so the renamed identity is the same content under
    a name the consuming asset owns. A fresh ``Material`` is built
    (``replace``), never an in-place edit: ``content_hash`` is memoized
    on the instance and the source value may be shared with other
    consumers of the same record. Every key must name the material
    present on the value; an Image, an unmaterialed output, or a key
    naming an absent material fails loud (``material_rename_unknown``) —
    a rename that silently does nothing would be the field-test's
    overrides-not-reaching-output class all over again.
    """
    if not mapping:
        return value
    proto = None if isinstance(value, Image) else proto_of(value)
    present = [] if proto is None or proto.material is None else [proto.material.name]
    unknown = sorted(k for k in mapping if k not in present)
    if unknown:
        raise OpError(
            "material_rename_unknown",
            f"material_rename names {unknown}, but the referenced output "
            f"carries {present or 'no material'}; rename a material the "
            f"output actually binds (check `moraine stats` of the referenced "
            f"asset for its material names)",
            param="material_rename",
            present=present,
            unknown=unknown,
        )
    return bind_material(value, replace(proto.material, name=mapping[proto.material.name]))


def override_token(overrides: dict) -> str:
    """Canonical evaluation-identity token for a per-use override set.

    Empty overrides normalize to "" — the plain identity, so an
    un-overridden reference keeps pure record aliasing (DESIGN 4.6).
    Values are deliberately not numerically folded ({"w": 2} and
    {"w": 2.0} are distinct identities): the coercion to the declared
    number type depends on the referenced asset's declaration,
    unknowable here. The build normalizes the overrides it writes into
    the IR (DESIGN 4.6), so the tokens of a built IR are folded already.
    """
    if not overrides:
        return ""
    return json.dumps(overrides, sort_keys=True, separators=(",", ":"))


def override_identity(overrides: dict) -> str:
    """The override identity as it names a file (DESIGN 4.6): the
    24-hex hash of the normalized overrides, "" for none. The build
    writes a variant IR under it and the evaluator opens that file by
    deriving the same name."""
    token = override_token(overrides)
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def _resolve_project_ref(asset_path: Path, spec: RefSpec) -> Path:
    """Resolve the project form to the IR it names (DESIGN 4.6): the
    sibling root's own IR under the sibling's ``build/``, or beside it
    the variant the overrides define.

    The project is a directory directly under the workspace root and
    the asset a bare file name, so neither can spell a path. The target
    is resolved, so a link-spelled sibling and its real directory are
    one file to every reader. The sibling and the file the reference
    opens must both lie inside the resolved workspace root, so a link
    anywhere on the way (the sibling, its ``build/``, the IR itself)
    cannot lead out.
    """
    name = spec.project
    consumer = project_root(asset_path)
    ws_root = consumer.parent
    if not (ws_root / MORAINE_DIR).is_dir():
        raise AssetError(
            "ref_no_workspace",
            f"asset_ref 'project' targets a sibling project, but "
            f"'{ws_root.name}' is not a workspace root (no {MORAINE_DIR}/ "
            f"marker); run 'moraine init' there - or vendor the asset "
            f"under {SUBASSETS_DIR}/ instead",
            project=name,
        )
    inside = ws_root.resolve()

    def held(path: Path) -> Path:
        path = path.resolve()
        if not path.is_relative_to(inside):
            raise AssetError(
                "project_outside_workspace",
                f"project '{name}' resolves outside the workspace (a link "
                f"under the workspace root points elsewhere); references "
                f"stay workspace-internal - move the target into the "
                f"workspace, or vendor the asset under {SUBASSETS_DIR}/",
                project=name,
            )
        return path

    def layout(directory: str) -> bool:
        return directory.startswith(".") or directory.lower() in RESERVED_DIRS

    sibling = ws_root / name if layout(name) else held(ws_root / name)
    if layout(name) or not sibling.is_dir():
        built = sorted(
            c.name for c in ws_root.iterdir()
            if not layout(c.name) and (c / BUILD_DIR).is_dir()
        )
        raise AssetError(
            "unknown_project",
            f"no project '{name}' in this workspace; the projects built "
            f"here: {', '.join(built) or 'none'}",
            project=name, projects=built,
        )
    if spec.asset is None:
        raise AssetError(
            "ref_unresolved",
            f"asset_ref names the project '{name}' and not the IR of its "
            f"root asset, which the build writes beside it as 'asset'; "
            f"run 'moraine build' on the project that holds this "
            f"reference",
            project=name,
        )
    return held(sibling / BUILD_DIR / _variant_name(spec.asset, spec.overrides))


def _variant_name(asset: str, overrides: dict) -> str:
    """The file name of the IR a reference opens (DESIGN 4.6): the
    asset's own, or the variant's, which carries the override identity
    before the extension."""
    identity = override_identity(overrides)
    if not identity:
        return asset
    return f"{asset[: -len(ASSET_EXT)]}.{identity}{ASSET_EXT}"


def resolve_ref(asset_path: Path, spec: RefSpec) -> Path:
    """Resolve a reference to the IR it names: a sub-asset's own IR or
    a sibling root's, or the variant its overrides define (DESIGN 4.6).

    Returns the absolute target path. For the in-project form, rejects
    references that escape the project directory, then references that
    break the layout grammar ``subassets/<name>.mora`` (DESIGN 4.2,
    4.6) — escape is the sharper diagnosis, so it wins when both apply.
    Deliberately does NOT check existence: the evaluator does that
    after registering the target in its reached-file set.
    """
    if spec.project is not None:
        return _resolve_project_ref(asset_path, spec)
    rel = spec.asset
    root = ir_home(asset_path)
    # PureWindowsPath.is_absolute catches both '/x' and 'C:/x' forms.
    if PureWindowsPath(rel).is_absolute() or rel.startswith("/"):
        raise AssetError(
            "ref_escape",
            f"asset_ref path '{rel}' is absolute; references must stay "
            f"project-relative (DESIGN 4.6) - vendor the asset under subassets/",
            path=rel,
        )
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise AssetError(
            "ref_escape",
            f"asset_ref path '{rel}' escapes the project directory; a "
            f"reference path never leaves the project (DESIGN 4.6) - vendor "
            f"the asset under subassets/, or name a sibling project with "
            f"'project' instead",
            path=rel,
        )
    parts = rel.split("/")
    if len(parts) != 2 or parts[0] != SUBASSETS_DIR or parts[1] in ("", ".", ".."):
        raise AssetError(
            "ref_layout",
            f"asset_ref path '{rel}' must be '{SUBASSETS_DIR}/<name>{ASSET_EXT}': "
            f"sub-assets live directly under {SUBASSETS_DIR}/, one level deep "
            f"(DESIGN 4.2) - move the asset there",
            path=rel,
        )
    if spec.overrides:
        target = target.with_name(
            _variant_name(target.name, spec.overrides)
        ).resolve()
        if not target.is_relative_to(root):
            raise AssetError(
                "ref_escape",
                f"the variant of '{rel}' this reference opens is a link "
                f"out of the project directory; a reference never leaves "
                f"the project (DESIGN 4.6) - remove the link and run "
                f"'moraine build'",
                path=rel,
            )
    return target


def evaluate(inputs, params, ctx):
    # Never reached: the evaluator intercepts asset_ref nodes (it owns the
    # recursion). Kept so the registry contract stays uniform.
    raise OpError(
        "internal",
        "asset_ref must be evaluated by the graph evaluator, not directly",
    )
