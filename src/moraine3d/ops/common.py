"""Shared parameter and input validation for operators.

All validation raises ``OpError`` with an actionable message (DESIGN 8.2).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.mesh import NEUTRAL_MATERIAL_NAME, Image, Instances, Mesh
from ..core.paths import PORTABLE_NAME_FRAGMENT, PORTABLE_NAME_RE
from ..core.predicates import FaceGroupError, TaggedFaces, resolve_face_group
from ..errors import OpError


def material_name(op: str, value, param: str, *, what: str = "name") -> str:
    """Validate an exported material identity: a portable identifier (it
    becomes an engine object name), never the neutral default's reserved
    name. Shared by ``material`` and ``asset_ref`` ``material_rename`` so
    "what is a legal material name" has one spelling."""
    if not isinstance(value, str) or not PORTABLE_NAME_RE.fullmatch(value):
        raise OpError(
            "bad_param",
            f"{op} {what} must match {PORTABLE_NAME_FRAGMENT} - it is the "
            f"exported material identity",
            param=param,
        )
    if value == NEUTRAL_MATERIAL_NAME:
        raise OpError(
            "bad_param",
            f"{op} {what} '{value}' is reserved for the neutral default "
            f"material; pick another name",
            param=param,
        )
    return value


def is_num(v) -> bool:
    """Is ``v`` a finite non-bool number? The one spelling of the
    judgment (the ``num_pair`` rule: a hardening of what counts as a
    number — bools, numpy scalars — must land everywhere at once);
    the camera, ramp, and server input gates alias it."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def num_pair(v, *, integral: bool = False) -> bool:
    """Is ``v`` a 2-list of finite non-bool numbers (ints if integral)?

    The shared type/shape core of every pair-valued param (grid size and
    segments, scatter scale_jitter, texture [w, h]) — bounds, ordering,
    and the actionable message stay with each param, but a hardening of
    what counts as "a pair of numbers" (bools, numpy scalars, tuples)
    must land everywhere at once.
    """
    return (
        isinstance(v, (list, tuple))
        and len(v) == 2
        and all(
            isinstance(c, int if integral else (int, float))
            and not isinstance(c, bool)
            and (integral or math.isfinite(c))
            for c in v
        )
    )


def num(
    op: str, params: dict, name: str, default: float, *, positive: bool = False,
    minimum: float | None = None, maximum: float | None = None,
) -> float:
    v = params.get(name, default)
    if not is_num(v):
        raise OpError("bad_param", f"{op} '{name}' must be a finite number", param=name)
    if positive and v <= 0:
        raise OpError("bad_param", f"{op} '{name}' must be > 0", param=name)
    if minimum is not None and v < minimum:
        raise OpError("bad_param", f"{op} '{name}' must be >= {minimum}", param=name)
    if maximum is not None and v > maximum:
        raise OpError("bad_param", f"{op} '{name}' must be <= {maximum}", param=name)
    return float(v)


def boolean(op: str, params: dict, name: str, default: bool) -> bool:
    v = params.get(name, default)
    if not isinstance(v, bool):
        raise OpError(
            "bad_param", f"{op} '{name}' must be a JSON boolean (true/false)", param=name
        )
    return v


def integer(
    op: str, params: dict, name: str, default: int, *, minimum: int = 0,
    maximum: int | None = None,
) -> int:
    v = params.get(name, default)
    if not isinstance(v, int) or isinstance(v, bool):
        raise OpError("bad_param", f"{op} '{name}' must be an integer", param=name)
    if v < minimum:
        raise OpError("bad_param", f"{op} '{name}' must be >= {minimum}", param=name)
    if maximum is not None and v > maximum:
        raise OpError("bad_param", f"{op} '{name}' must be <= {maximum}", param=name)
    return v


# Seed ceiling for stochastic params: keeps every derived stream (octave
# strides included) inside uint64 so rand.py never overflows into a raw
# (non-actionable) exception.
SEED_MAX = 2**63 - 1

# Shared crease-angle default (degrees): the smooth_normals judgment -
# a bend at or under it still reads smooth. Ops that classify creases
# (smooth_normals, revolve) share it so their documented equivalence
# cannot drift.
CREASE_ANGLE_DEG_DEFAULT = 30.0

# Shared tessellation cell budget: bounds the *product* of a generator's
# two tessellation dimensions where the per-dimension 4096 ceilings
# alone would not (an absurdity backstop like the ceilings themselves -
# asset-quality bars are the asset's own budgets). The value is the
# shipped generators' maximum product (grid's 4096 x 4096): grid and
# sphere satisfy it structurally, so only an op whose tessellation can
# multiply past it (box, over its three segment counts; revolve, via
# hard-joint ring duplication) checks it at runtime. Pinned against
# the per-op ceilings by a hardening test.
TESSELLATION_CELLS_MAX = 4096 * 4096


def seed(op: str, params: dict, name: str = "seed", default: int = 0) -> int:
    return integer(op, params, name, default, minimum=0, maximum=SEED_MAX)


def vec3(
    op: str, params: dict, name: str, default: list[float], *,
    positive: bool = False, scalar_ok: bool = False,
) -> np.ndarray:
    """3-vector param: 3 finite numbers. ``scalar_ok`` lets a bare number
    broadcast to ``[v, v, v]`` — only for scale-like params (``scale``,
    ``step_scale``, box ``size``), where a uniform value is the common
    authoring intent; for a translation, rotation, offset, or direction
    a bare number is a typo class that would evaluate wrong rather than
    fail loud, so it is rejected."""
    v = params.get(name, default)
    if scalar_ok and isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v, v, v]
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 3
        or not all(is_num(c) for c in v)
    ):
        raise OpError("bad_param", f"{op} '{name}' must be 3 finite numbers", param=name)
    if positive and not all(c > 0 for c in v):
        raise OpError("bad_param", f"{op} '{name}' components must be > 0", param=name)
    return np.asarray(v, dtype=np.float64)


def choice(op: str, params: dict, name: str, default: str, allowed: tuple[str, ...]) -> str:
    v = params.get(name, default)
    if v not in allowed:
        raise OpError(
            "bad_param", f"{op} '{name}' must be one of {list(allowed)}", param=name
        )
    return v


# Remedy for an instanced value on a mesh-only input that consumes the
# geometry as a whole (a boolean cutter, a scatter target, a placed part):
# baking is the sanctioned path there, where the default prototype advice
# would rebuild the wrong thing (one bolt instead of the bolt circle).
BAKE_HINT = "insert a bake_instances node between the array/scatter node and this input"


def mesh_input(op: str, inputs: list, index: int, *, hint: str | None = None) -> Mesh:
    """Mesh input; rejects instances and images with the standard
    guidance. ``hint`` replaces the default instanced-input remedy (apply
    the op to the prototype) where baking is the right path
    (``BAKE_HINT``)."""
    v = inputs[index]
    if isinstance(v, Instances):
        remedy = hint or f"apply {op} to the prototype before the array/scatter node"
        raise OpError(
            "bad_input",
            f"{op} input {index} is instanced; {op} needs a mesh - {remedy}",
        )
    if isinstance(v, Image):
        raise OpError(
            "bad_input",
            f"{op} input {index} is an image (texture); {op} needs a mesh - "
            f"images feed material or texture nodes, not geometry ops",
        )
    return v


def geometry_input(op: str, inputs: list, index: int) -> Mesh | Instances:
    """Mesh or Instances input; rejects images with the standard guidance."""
    v = inputs[index]
    if isinstance(v, Image):
        raise OpError(
            "bad_input",
            f"{op} input {index} is an image (texture); {op} needs geometry - "
            f"images feed material or texture nodes, not geometry ops",
        )
    return v


def image_input(op: str, inputs: list, index: int, *, channels: int | None = None) -> Image:
    v = inputs[index]
    if not isinstance(v, Image):
        raise OpError(
            "bad_input",
            f"{op} input {index} must be an image (a texture node output), "
            f"got {'instances' if isinstance(v, Instances) else 'a mesh'}",
        )
    if channels is not None and v.channels != channels:
        raise OpError(
            "bad_input",
            f"{op} input {index} must be a {channels}-channel image, "
            f"got {v.channels} channel(s)",
        )
    return v


def rgb(op: str, params: dict, name: str, default: list[float]) -> np.ndarray:
    """Linear-RGB color param: 3 finite numbers >= 0 (glTF factor semantics)."""
    v = params.get(name, default)
    if (
        not isinstance(v, (list, tuple))
        or len(v) != 3
        or not all(is_num(c) and c >= 0 for c in v)
    ):
        raise OpError(
            "bad_param",
            f"{op} '{name}' must be a linear-RGB color: 3 finite numbers >= 0",
            param=name,
        )
    return np.asarray(v, dtype=np.float64)


def texture_size(op: str, params: dict, name: str = "size", default: int = 256) -> tuple[int, int]:
    """Texture size param: int (square) or [w, h], each in 4..2048."""
    v = params.get(name, default)
    if isinstance(v, int) and not isinstance(v, bool):
        v = [v, v]
    if not num_pair(v, integral=True) or not all(4 <= c <= 2048 for c in v):
        raise OpError(
            "bad_param",
            f"{op} '{name}' must be an integer or [w, h], each in 4..2048",
            param=name,
        )
    return int(v[0]), int(v[1])


def face_group(
    op: str, value, name, instance, *, index: int, role: str, param: str, inst_param: str
) -> TaggedFaces:
    """Resolve a tagged face set on an op input (``predicates.resolve_face_group``)
    in the op's error vocabulary: an image is ``bad_input``; an unknown or
    empty tag is ``unknown_{role}`` / ``empty_{role}`` naming ``param``; a
    missing copy index on an instanced input is ``missing_{inst_param}``;
    a wrong one is ``bad_param`` on ``inst_param``."""
    subject = f"input {index} (the {role})"
    try:
        return resolve_face_group(value, name, instance, subject=subject)
    except FaceGroupError as e:
        if e.code == "image":
            raise OpError("bad_input", f"{op} {e.message}") from None
        if e.code == "unknown_group":
            raise OpError(
                f"unknown_{role}", f"{op} '{param}' must name a face group: {e.message}",
                param=param,
            ) from None
        if e.code == "empty_group":
            raise OpError(f"empty_{role}", f"{op}: {e.message}", param=param) from None
        if e.code == "missing_instance":
            raise OpError(
                f"missing_{inst_param}", f"{op}: {e.message} (pass '{inst_param}')",
                param=inst_param,
            ) from None
        raise OpError(
            "bad_param", f"{op} '{inst_param}': {e.message}", param=inst_param
        ) from None
