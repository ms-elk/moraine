"""Face-predicate vocabulary (DESIGN 5.1, 6).

One vocabulary shared by the write side (``tag``, anchored placement)
and the read side (``moraine query``): the same expression an agent uses
to inspect an asset is the one it fixes into the graph, so priors
transfer (P6 rubric 3).

A predicate is a JSON object mapping face fields to comparator objects,
e.g. ``{"normal_y": {"gt": 0.9}, "center_y": {"ge": 1.0}}``. All
comparisons AND together (across fields and within a field). Fields are
per-face scalars: ``normal_x/y/z`` (unit face normal; degenerate faces
get a zero normal and simply never match a normal test), ``center_x/y/z``
(face centroid, meters), ``area`` (m^2). Comparators: ``gt`` / ``ge`` /
``lt`` / ``le``, and ``eq`` with an explicit ``tol`` (default
``EQ_TOL_DEFAULT``) — |value - eq| <= tol.

Determinism: fields derive from evaluated geometry (deterministic,
DESIGN 4.4) and selections are reported in ascending face-index order,
so a selection never flickers between runs. Comparisons are exact float
thresholds — robustness near a boundary is the author's job (prefer
``{"gt": 0.9}`` over ``{"gt": 0.99999}``).

Cardinality (fixed at design time, DESIGN 5.1): ``one_or_more``
(default; empty errors), ``exactly_one`` (empty and ambiguous error),
``any`` (empty is valid — a consumer for which an empty selection is
meaningless refuses it itself, e.g. ``empty_group`` on a frame lookup).
A selection outside its declared cardinality is a structured error,
never a silent pick.
"""

from __future__ import annotations

import math

import numpy as np

from dataclasses import dataclass

from ..errors import MoraineError
from .mesh import Image, Instances, Mesh, proto_of, stat_num, stat_vec, tri_cross

# Shared-kernel version (DESIGN 4.5): ops that consume this module declare
# it in KERNELS and this integer joins their cache keys. Bump on ANY
# output-semantics change of a function an op consumes (field
# derivation, comparison, selection order, face-group resolution), bug
# fixes included — like OP_IMPL_VERSION.
KERNEL_VERSION = 1

FIELDS = (
    "normal_x", "normal_y", "normal_z",
    "center_x", "center_y", "center_z",
    "area",
)
COMPARATORS = ("gt", "ge", "lt", "le", "eq")
CARDINALITIES = ("one_or_more", "exactly_one", "any")
EQ_TOL_DEFAULT = 1e-6


def validate_predicate(pred, *, where: str = "select") -> None:
    """Reject a malformed predicate with an actionable error (DESIGN 8.2)."""

    def bad(msg: str) -> MoraineError:
        return MoraineError("bad_predicate", f"{where}: {msg}", predicate=pred)

    if not isinstance(pred, dict) or not pred:
        raise bad(
            "predicate must be a non-empty object mapping face fields to "
            'comparators, e.g. {"normal_y": {"gt": 0.9}}'
        )
    for field, comp in pred.items():
        if field not in FIELDS:
            raise bad(f"unknown field '{field}'; known fields: {list(FIELDS)}")
        if not isinstance(comp, dict) or not comp:
            raise bad(
                f"'{field}' must map to a comparator object like "
                f'{{"gt": 0.9}}; comparators: {list(COMPARATORS)}'
            )
        for key, val in comp.items():
            if key == "tol":
                if "eq" not in comp:
                    raise bad(f"'{field}' has 'tol' without 'eq'; tol only qualifies eq")
                if (
                    not isinstance(val, (int, float)) or isinstance(val, bool)
                    or not math.isfinite(val) or val < 0
                ):
                    raise bad(f"'{field}' 'tol' must be a finite number >= 0")
                continue
            if key not in COMPARATORS:
                raise bad(
                    f"'{field}' has unknown comparator '{key}'; "
                    f"known: {list(COMPARATORS)} (+ 'tol' with eq)"
                )
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise bad(f"'{field}' '{key}' must be a finite number")


def face_fields(mesh: Mesh) -> dict[str, np.ndarray]:
    """Per-face field values, ascending face-index order."""
    v = mesh.vertices.astype(np.float64)
    tri = v[mesh.faces.astype(np.int64)]  # (m, 3, 3)
    cross = tri_cross(tri)
    lens = np.linalg.norm(cross, axis=1, keepdims=True)
    safe = np.where(lens == 0, 1.0, lens)
    normal = cross / safe  # degenerate faces keep a zero normal
    center = tri.mean(axis=1)
    return {
        "normal_x": normal[:, 0], "normal_y": normal[:, 1], "normal_z": normal[:, 2],
        "center_x": center[:, 0], "center_y": center[:, 1], "center_z": center[:, 2],
        "area": lens[:, 0] / 2.0,
    }


def evaluate_predicate(mesh: Mesh, pred: dict, fields: dict | None = None) -> np.ndarray:
    """Boolean face mask for a validated predicate. ``fields`` hands
    over ``face_fields(mesh)`` where the caller already holds it."""
    if fields is None:
        fields = face_fields(mesh)
    mask = np.ones(len(mesh.faces), dtype=bool)
    for field, comp in pred.items():
        vals = fields[field]
        if field.startswith("normal_"):
            # Degenerate faces keep a zero normal and never match a
            # normal test (module contract above): unmasked, the zero
            # would satisfy any lt/le of a non-negative threshold and
            # eq 0, silently joining selections.
            mask &= fields["area"] > 0.0
        for key, ref in comp.items():
            if key == "gt":
                mask &= vals > ref
            elif key == "ge":
                mask &= vals >= ref
            elif key == "lt":
                mask &= vals < ref
            elif key == "le":
                mask &= vals <= ref
            elif key == "eq":
                mask &= np.abs(vals - ref) <= comp.get("tol", EQ_TOL_DEFAULT)
    return mask


def check_cardinality(count: int, expect: str, pred: dict, *, where: str = "select") -> None:
    """Enforce a selection's declared cardinality; structured errors only."""
    if expect == "one_or_more" and count == 0:
        raise MoraineError(
            "empty_selection",
            f"{where}: predicate matched 0 faces; loosen a threshold, or set "
            f'"expect": "any" if an empty selection is acceptable',
            predicate=pred,
        )
    if expect == "exactly_one":
        if count == 0:
            raise MoraineError(
                "empty_selection",
                f"{where}: predicate matched 0 faces but expects exactly one; "
                f"loosen a threshold",
                predicate=pred,
            )
        if count > 1:
            raise MoraineError(
                "ambiguous_selection",
                f"{where}: predicate matched {count} faces but expects exactly "
                f"one; tighten it (add a center_* or area bound)",
                predicate=pred, matched=count,
            )


def summarize(mesh: Mesh, mask: np.ndarray, fields: dict | None = None) -> dict:
    """Compact numeric summary of a selection (token cost scales with the
    answer, DESIGN 6): match count, total area, bounds of matched faces."""
    out: dict = {"faces": int(len(mesh.faces)), "matched": int(mask.sum())}
    if out["matched"]:
        if fields is None:
            fields = face_fields(mesh)
        out["area"] = stat_num(fields["area"][mask].sum())
        verts = mesh.vertices.astype(np.float64)[
            np.unique(mesh.faces.astype(np.int64)[mask])
        ]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        out["bounds"] = {"min": stat_vec(lo), "max": stat_vec(hi)}
    return out


def area_centroid(tri: np.ndarray) -> np.ndarray | None:
    """The area-weighted centroid of triangles ``(m, 3, 3)``: the origin
    ``place`` anchors on (DESIGN 5.1) and the one ``query`` reports, so
    the two cannot drift. None where the triangles hold no area."""
    areas = np.linalg.norm(tri_cross(tri), axis=1)
    total = areas.sum()
    if not total > 0:
        return None
    return (tri.mean(axis=1) * areas[:, None]).sum(axis=0) / total


def output_query(value, pred: dict | None, tag: str | None = None,
                 instance: int | None = None) -> dict:
    """One output's answer to a validated predicate, a tag, or both (the
    tag's faces the predicate matches), dispatched on the value's kind
    (Face predicates contract): an image output is skipped (face
    predicates apply to meshes), an instanced output is read on its
    prototype in its local frame — the same frame ``tag`` uses — and
    says so, a plain mesh is summarized directly. ``instance`` reads
    one copy where it stands in the world, the frame ``place`` and
    ``mate`` take an arrayed anchor in. An output without the tag is
    skipped with the tags it carries. The answer holds the matched
    faces' ``centroid`` (``area_centroid``)."""
    if isinstance(value, Image):
        return {"skipped": "image output; face predicates apply to meshes"}
    mesh = proto_of(value)
    try:
        matrix, _copies = (
            _instance_frame(value, instance, subject="the output")
            if instance is not None else (None, None)
        )
    except FaceGroupError as e:
        return {"skipped": e.message}
    fields = face_fields(mesh)
    mask = evaluate_predicate(mesh, pred or {}, fields)
    if tag is not None:
        groups = mesh.face_groups or {}
        if tag not in groups:
            carried = (
                f"it carries {sorted(groups)}" if groups
                else "it carries no tags (a tag node goes after the last "
                     "topology rebuild)"
            )
            return {"skipped": f"no tag '{tag}'; {carried}", "tags": sorted(groups)}
        member = np.zeros(len(mesh.faces), dtype=bool)
        member[np.asarray(groups[tag], dtype=np.int64)] = True
        mask &= member
    summary = summarize(mesh, mask, fields)
    if summary["matched"]:
        tri = mesh.vertices.astype(np.float64)[mesh.faces.astype(np.int64)[mask]]
        if matrix is not None:
            tri = tri @ matrix[:3, :3].T + matrix[:3, 3]
            flat = tri.reshape(-1, 3)
            summary["area"] = stat_num(np.linalg.norm(tri_cross(tri), axis=1).sum() / 2.0)
            summary["bounds"] = {
                "min": stat_vec(flat.min(axis=0)), "max": stat_vec(flat.max(axis=0)),
            }
        centroid = area_centroid(tri)
        if centroid is not None:
            summary["centroid"] = stat_vec(centroid)
    if isinstance(value, Instances):
        summary["instanced"] = len(value.transforms)
        summary["frame"] = (
            "prototype (local; copies are transforms applied afterwards)"
            if matrix is None else f"copy {instance} (world)"
        )
    return summary


# -- addressing a tagged face set on a value (DESIGN 5.1) ---------------------


class FaceGroupError(MoraineError):
    """A named face group (or a pointed-at face, ``resolve_pin``) that
    cannot be resolved on a value; ``code`` names why: ``image``,
    ``unknown_group``, ``empty_group``, ``missing_instance``,
    ``bad_instance``, ``bad_face``. Messages are factual and
    subject-prefixed; the caller maps the code into its own vocabulary
    (an op's error codes, an assertion's unevaluatable reason)."""


@dataclass(frozen=True)
class TaggedFaces:
    mesh: Mesh  # the mesh carrying the group (an Instances value's prototype)
    faces: np.ndarray  # the group's face indices
    matrix: np.ndarray | None  # the chosen copy's world matrix, None for a plain mesh
    copies: int | None  # copy count when instanced


def _instance_frame(value, instance, *, subject: str):
    """Validate a copy index against a value's kind and return the chosen
    copy's ``(matrix, copies)`` — ``(None, None)`` for a plain mesh. The
    shared half of ``resolve_face_group`` and ``resolve_pin``: a copy
    index is required on an instanced value (a frame is never guessed)
    and refused on a plain mesh; out-of-range errors instead of going
    stale."""
    if not isinstance(value, Instances):
        if instance is not None:
            raise FaceGroupError(
                "bad_instance", f"{subject} is a plain mesh; drop the copy index"
            )
        return None, None
    copies = len(value.transforms)
    if instance is None:
        raise FaceGroupError(
            "missing_instance",
            f"{subject} is instanced ({copies} copies); add the copy index "
            f"<0..{copies - 1}> - a frame is never guessed",
        )
    if not isinstance(instance, int) or isinstance(instance, bool):
        raise FaceGroupError(
            "bad_instance",
            f"{subject} is instanced ({copies} copies); the copy index must be "
            f"an integer 0..{copies - 1}",
        )
    if instance < 0 or instance >= copies:
        raise FaceGroupError(
            "bad_instance",
            f"{subject} has {copies} copies; copy index {instance} is out of "
            f"range (0..{copies - 1})",
        )
    return np.asarray(value.transforms[instance], dtype=np.float64), copies


def resolve_face_group(value, name, instance, *, subject: str) -> TaggedFaces:
    """The one resolver of "this tag on this value, this copy" shared by
    ``place``, ``mate``, and the ``mated`` assertion: tags ride the
    prototype, so an instanced value needs a copy index (required, never
    guessed; out-of-range errors instead of going stale) and a plain mesh
    refuses one; unknown tags list what is available; an empty group has
    no frame."""
    if isinstance(value, Image):
        raise FaceGroupError("image", f"{subject} is an image (texture); tags live on geometry")
    proto = proto_of(value)
    groups = proto.face_groups or {}
    if not isinstance(name, str) or name not in groups:
        available = (
            f"available tags: {sorted(groups)}"
            if groups
            else "it carries no tags - add a tag node upstream"
        )
        raise FaceGroupError(
            "unknown_group", f"{subject} has no face group '{name}'; {available}"
        )
    matrix, copies = _instance_frame(value, instance, subject=subject)
    idx = groups[name]
    if len(idx) == 0:
        raise FaceGroupError(
            "empty_group",
            f"face group '{name}' on {subject} selects 0 faces (its tag matched "
            f"nothing); loosen the tag's predicate - an empty face set has no frame",
        )
    return TaggedFaces(proto, idx, matrix, copies)


# Matched-face lists above this size are omitted from a predicate pin
# (display concern: the viewer highlights the matched polygons; a
# selection this large reads as "everything" anyway — marker only).
PIN_FACES_MAX = 20000


def resolve_predicate_pin(value, pred, instance, *, subject: str) -> dict:
    """Resolve a predicate into a pin (DESIGN 7.2): ``position`` is the
    area-weighted centroid of the matched faces, placed in world space
    when instanced, and the record carries the compact selection
    summary plus ``faces`` — the matched face indices, for display
    clients to highlight (omitted past ``PIN_FACES_MAX``). An empty
    match is the standard ``empty_selection`` error — a pin is never a
    silent pick. Kind and copy-index failures reuse the
    ``FaceGroupError`` codes."""
    if isinstance(value, Image):
        raise FaceGroupError(
            "image", f"{subject} is an image (texture); pins live on geometry"
        )
    matrix, copies = _instance_frame(value, instance, subject=subject)
    proto = proto_of(value)
    mask = evaluate_predicate(proto, pred)
    check_cardinality(int(mask.sum()), "one_or_more", pred, where=subject)
    fields = face_fields(proto)
    areas = fields["area"][mask]
    centers = np.stack(
        [fields["center_x"][mask], fields["center_y"][mask], fields["center_z"][mask]],
        axis=1,
    )
    total = float(areas.sum())
    if total > 0.0:
        center = (centers * (areas / total)[:, None]).sum(axis=0)
    else:  # all matched faces degenerate: fall back to the plain mean
        center = centers.mean(axis=0)
    position = center if matrix is None else (matrix @ np.append(center, 1.0))[:3]
    out = {
        "instance": instance if copies is not None else None,
        "summary": summarize(proto, mask),
        "position": stat_vec(position),
    }
    matched = np.flatnonzero(mask)
    if len(matched) <= PIN_FACES_MAX:
        out["faces"] = matched.tolist()
    return out


def resolve_pin(value, face, instance, *, subject: str, position=None) -> dict:
    """Resolve a pointed-at face into structural vocabulary (a pin,
    DESIGN 7.2): the face's tag-group membership plus its
    predicate-vocabulary fields. ``face_center`` and ``face_normal``
    are in the prototype's local frame — the frame ``tag`` and
    ``query`` read — while ``position`` is the face centroid placed in
    world space (the chosen copy's transform applied when instanced).
    A world-space click ``position``, when given, additionally yields
    ``bary`` — the click's barycentric coordinates in the face, clamped
    into the triangle — and the reported ``position`` becomes the bary
    point re-expressed on this (file-state) face: the click may have
    been cast against a client's preview shape, and the record must
    name a point on the geometry it describes, never the preview's
    (DESIGN 7.2 — a preview overlay never skews the record). Kind,
    copy-index and face-range failures reuse the ``FaceGroupError``
    codes."""
    if isinstance(value, Image):
        raise FaceGroupError(
            "image", f"{subject} is an image (texture); pins live on geometry"
        )
    matrix, copies = _instance_frame(value, instance, subject=subject)
    proto = proto_of(value)
    n = len(proto.faces)
    if not isinstance(face, int) or isinstance(face, bool) or face < 0 or face >= n:
        raise FaceGroupError(
            "bad_face",
            f"{subject} has {n} faces; the face index must be an integer "
            f"0..{n - 1}",
        )
    fields = face_fields(proto)
    center = np.array(
        [fields["center_x"][face], fields["center_y"][face], fields["center_z"][face]]
    )
    normal = np.array(
        [fields["normal_x"][face], fields["normal_y"][face], fields["normal_z"][face]]
    )
    world_center = center if matrix is None else (matrix @ np.append(center, 1.0))[:3]

    out = {
        "face": face,
        "instance": instance if copies is not None else None,
        "groups": sorted(
            g for g, idx in (proto.face_groups or {}).items()
            if bool((np.asarray(idx) == face).any())
        ),
        "face_center": stat_vec(center),
        "face_normal": stat_vec(normal),
        "area": stat_num(fields["area"][face]),
        "position": stat_vec(world_center),
    }
    if position is not None:
        tri = proto.vertices.astype(np.float64)[
            proto.faces.astype(np.int64)[face]
        ]
        p = np.asarray(position, dtype=np.float64)
        if matrix is not None:
            # The click is world-space; the triangle is prototype-local.
            p = (np.linalg.inv(matrix) @ np.append(p, 1.0))[:3]
        bary = _barycentric(tri, p)
        out["bary"] = stat_vec(bary)
        spot = bary @ tri
        out["position"] = stat_vec(
            spot if matrix is None else (matrix @ np.append(spot, 1.0))[:3]
        )
    return out


def _barycentric(tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of ``p`` against triangle ``tri`` (3x3),
    clamped into the triangle and renormalized — a raycast hit a hair
    outside the face must still name a point on it. A degenerate
    triangle answers the centroid."""
    e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
    w = p - tri[0]
    d00, d01, d11 = e0 @ e0, e0 @ e1, e1 @ e1
    denom = d00 * d11 - d01 * d01
    if denom <= 0.0:
        return np.full(3, 1.0 / 3.0)
    v = (d11 * (w @ e0) - d01 * (w @ e1)) / denom
    u = (d00 * (w @ e1) - d01 * (w @ e0)) / denom
    bary = np.clip(np.array([1.0 - v - u, v, u]), 0.0, None)
    return bary / bary.sum()
