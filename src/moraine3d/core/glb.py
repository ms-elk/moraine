"""Deterministic glTF 2.0 (GLB) writer — shared by export and the viewer.

Hand-rolled (not trimesh): trimesh's exporter cannot emit
``EXT_mesh_gpu_instancing``, and deterministic byte-identical output
(re-export must not break engine references — DESIGN 9.1 naming
determinism) is easier to guarantee than to retrofit. The viewer GLBs come
from this same writer, so what the viewer shows and what the export
contains are structurally the same payload.

Materials: each mesh's bound ``Material`` becomes a glTF metallic-roughness
material. Identity is content: identical content dedups into one glTF
material (one draw state per identity, DESIGN 9.1); one name with two
different contents is an error, never a silent split.
Emissive factors above 1 are normalized to the unit range and carried by
``KHR_materials_emissive_strength`` (glTF caps ``emissiveFactor`` at 1).
Meshes without a material share the neutral default. Texture encoding
(the only place pixels leave linear float, DESIGN 4.3):

- ``base_color`` -> PNG, sRGB-encoded;
- ``roughness`` / ``metallic`` (grayscale) -> packed into one linear PNG
  as glTF's metallicRoughnessTexture (G = roughness, B = metallic,
  R = 1); a missing map packs as white so the factor alone rules;
- ``normal`` -> linear PNG (+Y up, the glTF-required OpenGL convention).

Tangents are deliberately not written: the glTF spec requires importers
to generate MikkTSpace tangents when a normal-textured primitive has
none, which is exactly the MikkTSpace convention of DESIGN 4.3 —
delegating to the consumer beats shipping a non-MikkTSpace
approximation. Revisit only if in-engine verification shows seams.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from ..errors import MoraineError
from .mesh import NEUTRAL_MATERIAL_NAME, Image, Instances, Material, Mesh, Value
from .png import encode_png, quantize8, srgb_encode

# Writer version: bumped on any change to GLB generation. The cache
# embeds it in every cached GLB's filename (cache.records.glb_path), so
# GLBs from an older writer are orphaned, never served.
# 2: unified export and viewer on this writer — materials/textures in
# the viewer payload, instanced outputs via EXT_mesh_gpu_instancing.
# 3: non-opaque materials (alpha factor < 1) now emit alphaMode BLEND.
# 4: emissive factors above 1 are normalized and carried by
# KHR_materials_emissive_strength (the raw write was out of spec).
GLB_VERSION = 4

_GLTF_FLOAT = 5126
_GLTF_UINT32 = 5125

_ATYPE_BY_WIDTH = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}

# bufferView.target values (glTF): vertex attributes / indices.
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

# Neutral shared PBR material for meshes with no bound material: one
# material (one draw state) across every such part (DESIGN 9.1); factors
# match the viewer's historic neutral look.
NEUTRAL_MATERIAL = {
    "name": NEUTRAL_MATERIAL_NAME,
    "pbrMetallicRoughness": {
        "baseColorFactor": [0.75, 0.76, 0.8, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.8,
    },
}

# One shared sampler: repeat wrap (procedural textures tile), trilinear.
_SAMPLER = {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}


# -- TRS decomposition -------------------------------------------------------


def _quat_from_mat(r: np.ndarray) -> np.ndarray:
    """Rotation matrices (k, 3, 3) -> unit quaternions (k, 4) xyzw.

    Shepperd's method, vectorized: per matrix, branch on the largest of
    (trace, m00, m11, m22) for numerical stability.
    """
    k = len(r)
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((k, 4))
    choice = np.where(
        tr > np.maximum(np.maximum(m00, m11), m22),
        3,
        np.argmax(np.stack([m00, m11, m22], axis=1), axis=1),
    )

    m = choice == 3
    if m.any():
        s = np.sqrt(tr[m] + 1.0) * 2.0  # 4w
        q[m, 3] = 0.25 * s
        q[m, 0] = (r[m, 2, 1] - r[m, 1, 2]) / s
        q[m, 1] = (r[m, 0, 2] - r[m, 2, 0]) / s
        q[m, 2] = (r[m, 1, 0] - r[m, 0, 1]) / s
    for i, j, l in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        m = choice == i
        if not m.any():
            continue
        s = np.sqrt(1.0 + r[m, i, i] - r[m, j, j] - r[m, l, l]) * 2.0
        q[m, i] = 0.25 * s
        q[m, j] = (r[m, j, i] + r[m, i, j]) / s
        q[m, l] = (r[m, l, i] + r[m, i, l]) / s
        q[m, 3] = (r[m, l, j] - r[m, j, l]) / s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def decompose_trs(
    transforms: np.ndarray, output: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instance matrices (k, 4, 4) -> (translation, quaternion xyzw, scale).

    ``EXT_mesh_gpu_instancing`` carries TRS only; a matrix with shear (a
    rotate-then-nonuniform-scale composition, e.g. nested arrays) cannot be
    represented and errors — restructure, never silently merge (Invariant 6).
    """
    xf = np.asarray(transforms, dtype=np.float64)
    if not np.isfinite(xf).all():
        # Must be first: every later gate is a comparison that NaN would
        # silently pass (never-silent, P6).
        raise MoraineError(
            "nonfinite_instance",
            f"output '{output}' has NaN/inf in its instance transforms; "
            f"fix the producing array/scatter/transform node params",
        )
    t = xf[:, :3, 3].copy()
    a = xf[:, :3, :3]
    s = np.linalg.norm(a, axis=1)  # per-column norms: (k, 3) scale
    if np.any(s <= 1e-12):
        raise MoraineError(
            "degenerate_instance",
            f"output '{output}' has an instance transform with a zero scale "
            f"axis; fix the producing array/scatter/transform node",
        )
    r = a / s[:, None, :]
    det = np.linalg.det(r)
    if np.any(det < 0):
        raise MoraineError(
            "mirrored_instance",
            f"output '{output}' has mirrored (negative-determinant) instance "
            f"transforms; mirror the prototype mesh before the array/scatter "
            f"node instead",
        )
    # Orthonormality residual: nonzero means shear, which TRS cannot carry.
    rtr = r.transpose(0, 2, 1) @ r
    residual = np.abs(rtr - np.eye(3)).max()
    if residual > 1e-6:
        raise MoraineError(
            "sheared_instance",
            f"output '{output}' has instance transforms with shear (residual "
            f"{residual:.3g}), which EXT_mesh_gpu_instancing cannot represent; "
            f"add an explicit bake_instances node (works on every surface, "
            f"including serve), export with --bake-instances, or avoid "
            f"combining step_rotate_deg with non-uniform step_scale across "
            f"nested arrays",
        )
    return t, _quat_from_mat(r), s


# -- GLB writing -------------------------------------------------------------


class _Buffer:
    """The single GLB binary buffer; every view is 4-byte aligned."""

    def __init__(self):
        self.chunks: list[bytes] = []
        self.offset = 0
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def raw_view(self, raw: bytes) -> int:
        """Accessor-less bufferView (image payloads reference views directly)."""
        pad = (-len(raw)) % 4
        self.views.append({"buffer": 0, "byteOffset": self.offset, "byteLength": len(raw)})
        self.chunks.append(raw + b"\x00" * pad)
        self.offset += len(raw) + pad
        return len(self.views) - 1

    def accessor(
        self, data: np.ndarray, component: int, atype: str, *, minmax: bool = False,
        target: int | None = None,
    ) -> int:
        raw = np.ascontiguousarray(data).tobytes()
        view_index = self.raw_view(raw)
        if target is not None:
            self.views[view_index]["target"] = target
        acc = {
            "bufferView": view_index,
            "componentType": component,
            "count": len(data),
            "type": atype,
        }
        if minmax:  # required for POSITION
            acc["min"] = [float(v) for v in data.min(axis=0)]
            acc["max"] = [float(v) for v in data.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def binary(self) -> bytes:
        return b"".join(self.chunks)


class _MaterialTable:
    """Dedup materials by content and textures by pixels+encoding.

    glTF array order follows first use across outputs (outputs are
    ordered in the scene, so the result is deterministic).
    """

    def __init__(self, buf: _Buffer):
        self.buf = buf
        self.materials: list[dict] = []
        self.textures: list[dict] = []
        self.images: list[dict] = []
        self._mat_index: dict[str, int] = {}  # content hash -> index
        self._mat_by_name: dict[str, str] = {}  # name -> content hash
        self._img_index: dict[tuple, int] = {}  # (kind, hashes...) -> texture index
        self._neutral: int | None = None
        # glTF extensions the material table emitted (extensionsUsed).
        self.extensions_used: set[str] = set()

    def neutral_index(self) -> int:
        if self._neutral is None:
            self.materials.append(NEUTRAL_MATERIAL)
            self._neutral = len(self.materials) - 1
        return self._neutral

    def _texture(self, key: tuple, name: str, encode) -> int:
        # ``encode`` is a thunk: dedup hit must not pay for a PNG encode.
        idx = self._img_index.get(key)
        if idx is not None:
            return idx
        self.images.append(
            {
                "bufferView": self.buf.raw_view(encode()),
                "mimeType": "image/png",
                "name": name,
            }
        )
        self.textures.append({"sampler": 0, "source": len(self.images) - 1})
        idx = len(self.textures) - 1
        self._img_index[key] = idx
        return idx

    def _base_color_texture(self, img: Image) -> int:
        # Image names are content-derived, not material-derived: a shared
        # image dedups across materials, so naming it after the first user
        # would mislabel the others.
        key = ("srgb", img.result_hash())
        return self._texture(
            key,
            f"basecolor_{img.result_hash()[:8]}",
            lambda: encode_png(quantize8(srgb_encode(img.pixels))),
        )

    def _normal_texture(self, img: Image) -> int:
        key = ("linear", img.result_hash())
        return self._texture(
            key,
            f"normal_{img.result_hash()[:8]}",
            lambda: encode_png(quantize8(img.pixels)),
        )

    def _mr_texture(self, rough: Image | None, metal: Image | None, mat_name: str) -> int:
        # glTF metallicRoughnessTexture: G = roughness, B = metallic,
        # R unused (left white, occlusion-compatible). A missing map packs
        # as white so the corresponding factor alone rules (texture values
        # multiply factors in glTF).
        present = rough if rough is not None else metal
        assert present is not None
        h, w = present.pixels.shape[:2]
        for img in (rough, metal):
            if img is not None and img.pixels.shape[:2] != (h, w):
                raise MoraineError(
                    "material_texture_mismatch",
                    f"material '{mat_name}' roughness and metallic textures have "
                    f"different sizes; resize one so they can pack into one "
                    f"metallicRoughness texture",
                )
        key = (
            "mr",
            "" if rough is None else rough.result_hash(),
            "" if metal is None else metal.result_hash(),
        )

        def encode() -> bytes:
            packed = np.ones((h, w, 3), dtype=np.float32)
            if rough is not None:
                packed[..., 1] = rough.pixels[..., 0]
            if metal is not None:
                packed[..., 2] = metal.pixels[..., 0]
            return encode_png(quantize8(packed))

        name_hash = key[1] or key[2]
        return self._texture(key, f"mr_{name_hash[:8]}", encode)

    def index_for(self, mat: Material | None) -> int:
        if mat is None:
            return self.neutral_index()
        chash = mat.content_hash()
        idx = self._mat_index.get(chash)
        if idx is not None:
            return idx
        seen = self._mat_by_name.get(mat.name)
        if seen is not None and seen != chash:
            raise MoraineError(
                "material_name_collision",
                f"two different materials share the name '{mat.name}'; material "
                f"names are export identity - rename one, or make their params "
                f"identical so they dedup into one material",
            )
        if mat.name == NEUTRAL_MATERIAL_NAME:
            raise MoraineError(
                "material_name_reserved",
                f"material name '{mat.name}' is reserved for the neutral "
                f"default; pick another name",
            )
        tex = mat.textures
        pbr: dict = {
            "baseColorFactor": [float(v) for v in mat.base_color_factor],
            "metallicFactor": float(mat.metallic_factor),
            "roughnessFactor": float(mat.roughness_factor),
        }
        if "base_color" in tex:
            pbr["baseColorTexture"] = {
                "index": self._base_color_texture(tex["base_color"])
            }
        if "roughness" in tex or "metallic" in tex:
            pbr["metallicRoughnessTexture"] = {
                "index": self._mr_texture(
                    tex.get("roughness"), tex.get("metallic"), mat.name
                )
            }
        doc: dict = {"name": mat.name, "pbrMetallicRoughness": pbr}
        # glTF defaults to OPAQUE, which would ignore the alpha this
        # material carries — keep the exported meaning aligned with the
        # validate-side transparency test (validate._opaque). Alpha lives
        # in the factor only: Image is 1- or 3-channel, so no texture
        # carries it.
        if float(mat.base_color_factor[3]) < 1.0:
            doc["alphaMode"] = "BLEND"
        if "normal" in tex:
            nt: dict = {"index": self._normal_texture(tex["normal"])}
            if float(mat.normal_scale) != 1.0:
                nt["scale"] = float(mat.normal_scale)
            doc["normalTexture"] = nt
        if mat.is_emissive():
            emissive = [float(v) for v in mat.emissive_factor]
            strength = max(emissive)
            if strength > 1.0:
                # glTF caps emissiveFactor at 1 per component; intensity
                # above that is KHR_materials_emissive_strength (used, never
                # required: a non-supporting importer reads the unit-range
                # factor and loses only the strength).
                emissive = [v / strength for v in emissive]
                doc["extensions"] = {
                    "KHR_materials_emissive_strength": {"emissiveStrength": strength}
                }
                self.extensions_used.add("KHR_materials_emissive_strength")
            doc["emissiveFactor"] = emissive
        self.materials.append(doc)
        idx = len(self.materials) - 1
        self._mat_index[chash] = idx
        self._mat_by_name[mat.name] = chash
        return idx


def _add_mesh(buf: _Buffer, mesh: Mesh, name: str, material_index: int) -> dict:
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise MoraineError(
            "empty_mesh",
            f"output '{name}' has no triangles; nothing to export - check "
            f"the producing node",
        )
    if not np.isfinite(mesh.vertices).all():
        # Mirrors decompose_trs: an inf/NaN position would land in the
        # POSITION accessor min/max as a token no glTF reader parses.
        raise MoraineError(
            "nonfinite_mesh",
            f"output '{name}' has NaN/inf vertex data; fix the producing "
            f"node params",
        )
    attributes = {
        "POSITION": buf.accessor(
            mesh.vertices.astype(np.float32), _GLTF_FLOAT, "VEC3",
            minmax=True, target=_ARRAY_BUFFER,
        )
    }
    for semantic, arr in mesh.gltf_attributes().items():
        if semantic == "TANGENT":
            continue
        a = arr.astype(np.float32)
        attributes[semantic] = buf.accessor(
            a, _GLTF_FLOAT, _ATYPE_BY_WIDTH[a.shape[1]], target=_ARRAY_BUFFER
        )
    indices = buf.accessor(
        mesh.faces.astype(np.uint32).reshape(-1), _GLTF_UINT32, "SCALAR",
        target=_ELEMENT_ARRAY_BUFFER,
    )
    return {
        "name": name,
        "primitives": [
            {
                "attributes": attributes,
                "indices": indices,
                "material": material_index,
                "mode": 4,
            }
        ],
    }


def build_glb(values: dict[str, Value], generator: str) -> tuple[bytes, dict]:
    """Serialize named outputs to GLB bytes; also return export info.

    ``Image`` values are not parts and cannot appear here — the caller
    (export profile validation / the viewer's results path) routes them
    to PNG previews instead.
    """
    buf = _Buffer()
    mats = _MaterialTable(buf)
    meshes: list[dict] = []
    nodes: list[dict] = []
    info: dict[str, dict] = {}
    # Extensions the document relies on; instancing is the one a reader
    # cannot ignore (the copies would vanish), so it alone is required.
    used: set[str] = set()
    required: set[str] = set()

    for name, value in values.items():
        if isinstance(value, Image):
            raise MoraineError(
                "image_not_a_part",
                f"output '{name}' is an image; GLB parts are meshes - remove "
                f"it from the outputs or feed it into a material node",
            )
        if isinstance(value, Instances):
            mat_index = mats.index_for(value.prototype.material)
            meshes.append(_add_mesh(buf, value.prototype, name, mat_index))
            t, q, s = decompose_trs(value.transforms, name)
            node = {
                "name": name,
                "mesh": len(meshes) - 1,
                "extensions": {
                    "EXT_mesh_gpu_instancing": {
                        "attributes": {
                            "TRANSLATION": buf.accessor(
                                t.astype(np.float32), _GLTF_FLOAT, "VEC3"
                            ),
                            "ROTATION": buf.accessor(
                                q.astype(np.float32), _GLTF_FLOAT, "VEC4"
                            ),
                            "SCALE": buf.accessor(
                                s.astype(np.float32), _GLTF_FLOAT, "VEC3"
                            ),
                        }
                    }
                },
            }
            used.add("EXT_mesh_gpu_instancing")
            required.add("EXT_mesh_gpu_instancing")
            info[name] = {"instances": int(len(value.transforms))}
        else:
            mat_index = mats.index_for(value.material)
            meshes.append(_add_mesh(buf, value, name, mat_index))
            node = {"name": name, "mesh": len(meshes) - 1}
            info[name] = {}
        nodes.append(node)

    doc: dict = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": mats.materials,
        "accessors": buf.accessors,
        "bufferViews": buf.views,
    }
    if mats.textures:
        doc["samplers"] = [_SAMPLER]
        doc["textures"] = mats.textures
        doc["images"] = mats.images
    used |= mats.extensions_used
    if used:
        doc["extensionsUsed"] = sorted(used)
    if required:
        doc["extensionsRequired"] = sorted(required)

    binary = buf.binary()
    doc["buffers"] = [{"byteLength": len(binary)}]
    # sort_keys for byte-determinism: same values -> same GLB (9.1 naming
    # determinism extends to bytes, so re-export never dirties engine refs).
    # allow_nan=False is the backstop behind the per-mesh gates: a GLB
    # must never carry an 'Infinity' token.
    json_bytes = json.dumps(
        doc, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)

    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    glb = b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total),
            struct.pack("<II", len(json_bytes), 0x4E4F534A),  # JSON
            json_bytes,
            struct.pack("<II", len(binary), 0x004E4942),  # BIN
            binary,
        ]
    )
    return glb, info
