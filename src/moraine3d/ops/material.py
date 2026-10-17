"""``material`` operator.

Semantics: binds a glTF 2.0 metallic-roughness material to its mesh (or
instanced) input 0 — geometry is unchanged; the result carries the
material through cache, viewer, and export. Texture inputs follow as
inputs 1..n, with their roles named positionally by the ``slots`` param
(e.g. ``"slots": ["base_color", "normal"]`` for a 3-input node). Slots:
``base_color`` (3-channel, sRGB-encoded on export), ``roughness`` /
``metallic`` (1-channel each; packed into the glTF metallicRoughness
texture at the GLB edge, and both must be the same size), ``normal``
(3-channel, +Y-up encoded normals, e.g. from ``normal_map``). Texture
values multiply factors, exactly as in glTF.

Params: ``name`` (required — the exported material identity; identical
content dedups across parts, one name with two contents is an export
error); ``base_color`` — linear RGB or RGBA factor, components 0..1
(default [1, 1, 1, 1]); ``metallic`` (default 0.0) and ``roughness``
(default 0.8), 0..1 — authoring defaults match the neutral material,
not glTF's fully-metallic default, which renders black without an
environment; ``emissive`` — linear RGB factor, >= 0 (default
[0, 0, 0]); ``normal_scale`` (default 1.0, >= 0); ``slots`` (default
[]). Deterministic and pure.

Downstream, single-input geometry ops with ``MATERIAL_POLICY``
"preserved" (the default when their ATTRIBUTE_POLICY preserves slots)
carry the material; topology-rebuilding ops drop it with a warning —
prefer binding the material as the last node of a part.
"""

from __future__ import annotations

import math

from ..core import numbers
from ..core.mesh import MATERIAL_TEXTURE_SLOTS, Instances, Material, Mesh, bind_material
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 1
KERNELS = ()
INPUT_ARITY = (1, 1 + len(MATERIAL_TEXTURE_SLOTS))
# A recipe names each texture input by its slot's keyword, and the
# build writes the inputs in this order with ``slots`` beside them.
SLOT_INPUTS = {f"{slot}_texture": slot for slot in MATERIAL_TEXTURE_SLOTS}
SLOT_PARAM = "slots"
PARAM_KINDS = {
    "name": numbers.OTHER,
    "slots": numbers.OTHER,
    "base_color": numbers.FLOAT,
    "metallic": numbers.FLOAT,
    "roughness": numbers.FLOAT,
    "emissive": numbers.FLOAT,
    "normal_scale": numbers.FLOAT,
}
PARAMS = frozenset(PARAM_KINDS)
# The factor params affect only the bound material's glTF factor uniforms
# (texture values multiply factors, so the promise holds with textures
# bound); "name"/"slots" are structural. Interaction-tier contract
# (DESIGN 7.2): a public knob on one of these may drive the viewer live.
UNIFORM_PARAMS = {
    "base_color": "base_color",
    "metallic": "metallic",
    "roughness": "roughness",
    "emissive": "emissive",
    "normal_scale": "normal_scale",
}
ATTRIBUTE_POLICY = "preserved"
PLACEMENT = "identity"  # binds a material; positions untouched


def authors_material(params: dict) -> bool:
    """Every material node binds an identity the enclosing asset owns
    (Evaluation.material_origins; the knob panel's authored-identity
    scope, DESIGN 7.2)."""
    return True

_SLOT_CHANNELS = {"base_color": 3, "roughness": 1, "metallic": 1, "normal": 3}


def _base_color(op: str, params: dict) -> tuple:
    v = params.get("base_color", [1.0, 1.0, 1.0, 1.0])
    if (
        not isinstance(v, (list, tuple))
        or len(v) not in (3, 4)
        or not all(
            isinstance(c, (int, float)) and not isinstance(c, bool)
            and math.isfinite(c) and 0.0 <= c <= 1.0
            for c in v
        )
    ):
        raise OpError(
            "bad_param",
            f"{op} 'base_color' must be a linear RGB(A) factor: 3 or 4 "
            f"numbers in 0..1",
            param="base_color",
        )
    return tuple(float(c) for c in v) + ((1.0,) if len(v) == 3 else ())


def evaluate(inputs: list, params: dict, ctx) -> Mesh | Instances:
    op = "material"
    name = common.material_name(op, params.get("name"), "name", what="'name' (required)")

    slots = params.get("slots", [])
    if (
        not isinstance(slots, list)
        or not all(isinstance(s, str) for s in slots)
        or len(set(slots)) != len(slots)
        or not all(s in MATERIAL_TEXTURE_SLOTS for s in slots)
    ):
        raise OpError(
            "bad_param",
            f"{op} 'slots' must be a list of distinct slot names from "
            f"{list(MATERIAL_TEXTURE_SLOTS)}, one per texture input in order",
            param="slots",
        )
    if len(slots) != len(inputs) - 1:
        raise OpError(
            "bad_param",
            f"{op} has {len(inputs) - 1} texture input(s) but 'slots' names "
            f"{len(slots)}; list one slot per input after the mesh, in order",
            param="slots",
        )

    textures = {
        slot: common.image_input(op, inputs, 1 + i, channels=_SLOT_CHANNELS[slot])
        for i, slot in enumerate(slots)
    }
    rt, mt = textures.get("roughness"), textures.get("metallic")
    if rt is not None and mt is not None and rt.pixels.shape[:2] != mt.pixels.shape[:2]:
        raise OpError(
            "bad_input",
            f"{op} roughness ({rt.width}x{rt.height}) and metallic "
            f"({mt.width}x{mt.height}) textures must be the same size - they "
            f"pack into one metallicRoughness texture on export",
        )

    normal_scale = common.num(op, params, "normal_scale", 1.0)
    if normal_scale < 0:
        raise OpError("bad_param", f"{op} 'normal_scale' must be >= 0", param="normal_scale")
    mat = Material(
        name=name,
        base_color_factor=_base_color(op, params),
        metallic_factor=common.num(op, params, "metallic", 0.0, minimum=0.0, maximum=1.0),
        roughness_factor=common.num(op, params, "roughness", 0.8, minimum=0.0, maximum=1.0),
        emissive_factor=tuple(common.rgb(op, params, "emissive", [0.0, 0.0, 0.0])),
        normal_scale=normal_scale,
        textures=textures,
    )

    src = inputs[0]
    if not isinstance(src, Instances):
        common.mesh_input(op, inputs, 0)  # type check; the bind is shared
    return bind_material(src, mat)
