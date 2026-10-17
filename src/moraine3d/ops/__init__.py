"""Operator registry.

Each operator module exposes:
- ``OP_IMPL_VERSION``: int, bumped on ANY output-semantics change (DESIGN 4.5).
- ``INPUT_ARITY``: (min, max) input count; max ``None`` = unbounded.
- ``INPUT_NAMES`` (optional): the role of each input, by position, for
  an op whose inputs play different roles (place's base and part). A
  recipe may pass such an input by its name (DESIGN 4.8); the IR keeps
  the positions. A name is a keyword of the builtin, so none is a
  param's or ``id``.
- ``SLOT_INPUTS`` and ``SLOT_PARAM`` (optional, together): the optional
  inputs a recipe passes by keyword alone, each keyword with the slot
  it fills, and the param the build writes the filled slots to, in
  input order (material's textures and ``slots``).
- ``PARAM_KINDS``: the accepted params, each with the number type it
  holds (``core.numbers``, DESIGN 4.8): ``FLOAT`` for a param read as
  floats, ``INT`` for one read as integers, ``OTHER`` for a string, a
  boolean, or a structure with rules of its own. The build writes a
  number in its param's type (``typed_params``).
- ``PARAMS``: the accepted param names, the keys of ``PARAM_KINDS``; the
  evaluator rejects unknown keys so a typo'd param errors instead of
  silently evaluating with the default.
- ``USES_NODE_ID`` (optional): True when node id salts stochastic streams,
  so the evaluator adds it to the cache key (see rand.py).
- ``KERNELS``: tuple of the kernels the output depends on, whose resolved
  versions join the cache key (DESIGN 4.5) — an installed distribution
  by name (``"manifold3d"``, ``"trimesh"``), or a shared in-package
  kernel module by its dotted path (``"core.mating"``,
  ``"core.predicates"``, ``"core.rand"``, ``"core.mesh"``), resolved to
  its ``KERNEL_VERSION``. numpy is never declared: its arithmetic is
  bit-tolerant (DESIGN 4.4). Empty for a pure op. A test pins the
  declaration to the module's imports in both directions.
- ``ATTRIBUTE_POLICY``: ``"preserved"`` or ``"dropped"`` — what happens to
  input vertex-attribute slots beyond POSITION/NORMAL (glTF vocabulary,
  see mesh.py). Part of the semantics contract; never silent (P6):
  ``validate`` warns when a graph feeds attributes into a dropping op.
- ``MATERIAL_POLICY`` (optional): same values, for a bound material;
  defaults to ATTRIBUTE_POLICY — an op that rebuilds topology
  drops the material for the same reason. The evaluator re-binds input
  0's material onto single-input "preserved" results centrally, so ops
  keep constructing bare geometry (see evaluator._propagate_material).
- ``SHADING_POLICY`` (optional): ``"preserved"`` (default) when input
  shading flows to the output, or ``"replaced"`` when the output's
  shading is independent of the input's (smooth_normals rebuilds it,
  boolean/bevel emit flat-shaded regardless of input). Consumed by the
  flat_shading graph walk (evaluator.shading_warnings), which stops at
  replaced-shading ops — new ops join by declaration, not walker edits.
- ``FLAT_SHADING_SOURCE`` (optional, default False): True when the
  flat_shading warning should name this op's own flat output as its
  source (boolean declares it; bevel deliberately does not). Consumed
  by the same walk — a new flat-emitting op joins by declaration.
- ``CARRIED_INPUTS`` (optional): indices of the inputs whose surface
  actually flows to the output; default = all. Reference-only inputs
  (scatter's target, place's anchor base) are excluded so graph walks
  about the output's surface do not follow them.
- ``PLACEMENT`` (optional; absent = the op reshapes geometry): how the
  output's vertex positions relate to the carried input's —
  ``"identity"`` (untouched: tag, material, unwrap, smooth_normals),
  ``"affine"`` (the module's ``matrix(params)`` applied — transform),
  or ``"rigid"`` (the whole part moved as one body by a motion computed
  from the data — place, mate). Consumed by the embedded-cede walk
  (Evaluation.embedded_via_refs, DESIGN 5.2), which crosses declaring
  ops and stops at any other — a new position-preserving op joins by
  declaration, never by a walker edit.
- ``IS_REFERENCE`` (optional): True when the value originates in another
  asset (asset_ref) — the evaluator routes the node through its
  reference path (the module supplies ref_params / resolve_ref /
  override_token / rename_materials), graph walks follow the reference
  alias there, and the material-origin walk
  (Evaluation.material_origins) treats it as a file boundary.
- ``authors_material(params) -> bool`` (optional module function; absent
  = never): True when the node binds a material identity the enclosing
  asset owns — ``material`` always, ``asset_ref`` when it carries a
  ``material_rename``. A predicate, not a constant, because the answer
  can depend on params. Consumed by Evaluation.material_origins (DESIGN 9.1).
- ``SEED_PARAMS`` (optional): frozenset of params that select an
  unordered stochastic stream (the ones validated by ``common.seed``).
  A public knob bound to one classifies as the ``seed`` interaction
  tier (DESIGN 7.2): a reroll control, never a slider.
- ``UNIFORM_PARAMS`` (optional): dict mapping params that affect ONLY
  the bound material's factor uniforms of this op's own output — never
  geometry, topology, or texture bytes — to their glTF channel (a
  ``UNIFORM_CHANNELS`` member). A semantics promise consumed by the
  interaction-tier classifier (engine.tiers, DESIGN 7.2): a knob bound
  to one may drive viewer material uniforms live, with no evaluation.
- ``evaluate(inputs, params, ctx) -> Mesh | Instances | Image``.
Operator semantics are documented in the shipped reference
(``moraine3d/docs/reference/``; ``moraine docs``) and pinned by
golden-mesh tests (DESIGN 5, 10).
"""

from __future__ import annotations

from ..core import numbers
from ..errors import OpError
from . import (
    array,
    bake_instances,
    bevel,
    boolean,
    box,
    brick_texture,
    checker_texture,
    colorize,
    cylinder,
    extrude,
    grid,
    material,
    mate,
    noise_displace,
    noise_texture,
    normal_map,
    place,
    ramp,
    revolve,
    scatter,
    asset_ref,
    smooth_normals,
    sphere,
    streaks,
    tag,
    transform,
    unwrap,
)

# The glTF metallic-roughness factor channels a live viewer preview can
# drive, each with the value range every op's own validation accepts for
# it (DESIGN 7.2). The range is part of the vocabulary because the
# factor tier skips evaluation: the classifier grants ``factor`` only to
# knobs whose declared [min, max] sits inside the channel range, so a
# value the declaration admits can never be one the op would reject —
# otherwise an unevaluated scrub could poison the overlay and Accept
# could write an unevaluable value into the asset file.
UNIFORM_RANGES = {
    "base_color": (0.0, 1.0),
    "metallic": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "emissive": (0.0, 1.0),
    "normal_scale": (0.0, float("inf")),
}
# ``UNIFORM_PARAMS`` values must come from this set — it is the whole
# vocabulary the viewer's uniform mapping implements, so an op inventing
# a channel would silently render as a dead knob.
UNIFORM_CHANNELS = frozenset(UNIFORM_RANGES)

_REGISTRY = {
    "array": array,
    "bake_instances": bake_instances,
    "bevel": bevel,
    "boolean": boolean,
    "box": box,
    "brick_texture": brick_texture,
    "checker_texture": checker_texture,
    "colorize": colorize,
    "cylinder": cylinder,
    "extrude": extrude,
    "grid": grid,
    "material": material,
    "mate": mate,
    "noise_displace": noise_displace,
    "noise_texture": noise_texture,
    "normal_map": normal_map,
    "place": place,
    "ramp": ramp,
    "revolve": revolve,
    "scatter": scatter,
    "asset_ref": asset_ref,
    "smooth_normals": smooth_normals,
    "sphere": sphere,
    "streaks": streaks,
    "tag": tag,
    "transform": transform,
    "unwrap": unwrap,
}


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str):
    mod = _REGISTRY.get(name)
    if mod is None:
        raise OpError("unknown_op", f"unknown operator '{name}'", known=sorted(_REGISTRY))
    return mod


def check_arity(name: str, n_inputs: int) -> None:
    lo, hi = get(name).INPUT_ARITY
    if n_inputs < lo or (hi is not None and n_inputs > hi):
        expect = str(lo) if hi == lo else (f">= {lo}" if hi is None else f"{lo}-{hi}")
        raise OpError(
            "bad_inputs",
            f"'{name}' takes {expect} input(s), got {n_inputs}",
            op=name,
        )


def typed_params(name: str, params: dict) -> dict:
    """``params`` with every number in the type its param declares
    (DESIGN 4.8): an integer given to a float param is held as a float,
    and a float given to an integer param is refused."""
    kinds = get(name).PARAM_KINDS
    out = {}
    for key, value in params.items():
        try:
            out[key] = numbers.typed(value, kinds.get(key, numbers.OTHER))
        except numbers.FloatForInteger as e:
            raise OpError(
                "bad_param", numbers.integer_fix(f"{name} '{key}'", e),
                op=name, param=key,
            ) from None
    return out
