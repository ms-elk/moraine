"""In-memory evaluation values and conversions.

Three value kinds flow through the graph (DESIGN 4.1, 4.5):
- ``Mesh``: vertices/faces/normals and optional UVs.
- ``Instances``: a prototype mesh plus per-instance 4x4 transforms.
  Instancing stays instancing end to end (Day-One Invariant 6); merging
  would be an explicit operator, never implicit.
- ``Image``: a procedural texture buffer — (h, w, 1|3) float32,
  LINEAR values (DESIGN 4.3: color spaces are an encoding concern; sRGB
  happens only at PNG-encode time, per material slot). Cached uniformly
  with meshes (DESIGN 4.5).

A ``Material`` is not a graph value of its own: the ``material``
operator binds one to a Mesh (glTF 2.0 metallic-roughness vocabulary,
mirroring how the attribute model uses glTF semantics verbatim). It rides
``Mesh.material``, participates in ``result_hash``/cache/stats, and is
propagated by the evaluator per each op's MATERIAL_POLICY.

Conventions (DESIGN 4.3, normative): meters, right-handed, +Y up, +Z
forward (asset front faces +Z), CCW front faces.

Attribute model (DESIGN 4.1): the
vertex attribute vocabulary is glTF's attribute semantics verbatim.
``normals`` and ``uv`` are the materialized ``NORMAL`` and ``TEXCOORD_0``
slots; every further per-vertex attribute (``COLOR_n``, ``TEXCOORD_n``
for n >= 1, ``TANGENT``) lives in ``attributes`` under its glTF semantic
name. All slots participate in ``result_hash``, the cache record, and
``stats``; the exporter carries them through. Each operator's semantics
contract states per attribute whether it is preserved or dropped
(``ATTRIBUTE_POLICY``) — never silent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace

import numpy as np
import trimesh

from .paths import PORTABLE_NAME_FRAGMENT

# Shared-kernel version (DESIGN 4.5): an op that consumes one of the
# geometry helpers below declares ``core.mesh`` in KERNELS and this
# integer joins its cache key. Bump on ANY output-semantics change of a
# helper in KERNEL_HELPERS, bug fixes included — like OP_IMPL_VERSION.
# The value classes, their conversions, and the stats helpers are not
# kernels: they carry data, they do not derive geometry.
KERNEL_VERSION = 1
KERNEL_HELPERS = frozenset({
    "tri_cross", "degenerate_tri_count", "vertex_normals",
    "divergent_duplicates", "flat_shaded",
})

# Extra attribute slots allowed in ``Mesh.attributes`` — glTF semantics
# verbatim. POSITION/NORMAL/TEXCOORD_0 are the dedicated fields; skinning
# slots (JOINTS/WEIGHTS) are out of scope (hard-surface, no armatures).
GLTF_EXTRA_ATTR_RE = re.compile(r"^(COLOR_\d+|TEXCOORD_[1-9]\d*|TANGENT)$")

# Face-group names: same portable identifier form as
# output names — they appear in stats, errors, and `place` anchors.
FACE_GROUP_NAME_RE = re.compile(rf"^{PORTABLE_NAME_FRAGMENT}$")


def _neutral_visual(uv: np.ndarray | None = None) -> trimesh.visual.TextureVisuals:
    # Neutral PBR material so viewers don't fall back to the glTF
    # default (fully metallic renders black without an environment map).
    return trimesh.visual.TextureVisuals(
        uv=None if uv is None else uv.astype(np.float64),
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.75, 0.76, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.8,
        ),
    )


@dataclass
class Image:
    """Procedural texture buffer: (h, w, 1|3) float32, linear values.

    1 channel = data map (height, roughness, metallic); 3 channels =
    color or normal map. Alpha is out of scope for v1 (hard-surface
    opaque materials; ``base_color`` alpha comes from the factor).
    """

    pixels: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.pixels, dtype=np.float32)
        if p.ndim == 2:
            p = p[..., None]
        if p.ndim != 3 or p.shape[2] not in (1, 3) or p.shape[0] < 1 or p.shape[1] < 1:
            raise ValueError(
                f"Image pixels must be (h, w, 1|3), got shape {np.shape(self.pixels)}"
            )
        self.pixels = np.ascontiguousarray(p)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def channels(self) -> int:
        return int(self.pixels.shape[2])

    def result_hash(self) -> str:
        # Memoized on first call: Image is immutable after construction by
        # convention (ops build fresh buffers, nothing mutates pixels), and
        # one texture is re-hashed by every downstream node's result_hash,
        # the profile checks, and each GLB write — a full-buffer sha256
        # per call, ~0.3 s each at 4k. Not a dataclass field: the digest
        # must stay out of eq/repr.
        cached = self.__dict__.get("_result_hash")
        if cached is None:
            h = hashlib.sha256()
            h.update(b"img")
            h.update(np.asarray(self.pixels.shape, dtype=np.int64).tobytes())
            h.update(self.pixels.tobytes())
            cached = self.__dict__["_result_hash"] = h.hexdigest()
        return cached

    def stats(self) -> dict:
        return {
            "kind": "image",
            "size": [self.width, self.height],
            "channels": self.channels,
            "range": [stat_num(self.pixels.min()), stat_num(self.pixels.max())],
        }


def stat_num(x) -> float | None:
    """A stats number: 6-decimal rounded (stable cross-platform digits),
    and JSON-safe — a non-finite value (float32 overflow in a transform,
    a NaN pixel) becomes null rather than an 'Infinity'/'NaN' token that
    no standard JSON reader accepts. validate's ``finite`` check is the
    reporter of the condition; stats only stay parseable. The one home
    of the convention: camera poses and predicate reports round through
    here too, so the digits and the null policy cannot drift."""
    f = float(x)
    return round(f, 6) if math.isfinite(f) else None


def stat_vec(vec) -> list[float | None]:
    """A vector of stats numbers (see ``stat_num``)."""
    return [stat_num(x) for x in vec]


def _stat_bounds(lo, hi) -> dict:
    return {"min": stat_vec(lo), "max": stat_vec(hi)}


# Texture slots a Material can carry (glTF metallic-roughness vocabulary;
# roughness/metallic are authored as separate grayscale maps and packed
# into the glTF metallicRoughnessTexture G/B channels at GLB-write time —
# packing is an encoding concern, so it happens at the edge, not in the IR).
MATERIAL_TEXTURE_SLOTS = ("base_color", "roughness", "metallic", "normal")

# The neutral default material's exported name: what an unmaterialed part
# ships as (glb.NEUTRAL_MATERIAL), hence reserved for authored materials
# and a counted identity in the material budget (DESIGN 9.1).
NEUTRAL_MATERIAL_NAME = "moraine_default"


@dataclass
class Material:
    """glTF 2.0 metallic-roughness material bound to a Mesh.

    ``name`` is the exported material identity: same name must mean same
    content (validate gates it, export enforces it), and shared content
    dedups into one glTF material — one draw state shared across the kit
    pieces that bind it (DESIGN 9.1). Factors are linear, glTF semantics;
    textures multiply factors exactly as in glTF.
    """

    name: str
    base_color_factor: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 0.0
    roughness_factor: float = 0.8
    emissive_factor: tuple = (0.0, 0.0, 0.0)
    normal_scale: float = 1.0
    textures: dict[str, Image] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize factors to the canonical 9-decimal form ON CONSTRUCTION,
        # not just in factors(): a freshly evaluated Material and one
        # reloaded from cache must be indistinguishable everywhere —
        # including the raw fields the GLB writer serializes — or a warm
        # cache would change export bytes (byte-identical re-export, 9.1).
        self.base_color_factor = tuple(round(float(v), 9) for v in self.base_color_factor)
        self.metallic_factor = round(float(self.metallic_factor), 9)
        self.roughness_factor = round(float(self.roughness_factor), 9)
        self.emissive_factor = tuple(round(float(v), 9) for v in self.emissive_factor)
        self.normal_scale = round(float(self.normal_scale), 9)
        for slot in self.textures:
            if slot not in MATERIAL_TEXTURE_SLOTS:
                raise ValueError(
                    f"unknown material texture slot '{slot}'; "
                    f"known: {list(MATERIAL_TEXTURE_SLOTS)}"
                )

    def factors(self) -> dict:
        """Factor payload (no pixel data), canonical for hashing/serialization."""
        return {
            "name": self.name,
            "base_color_factor": list(self.base_color_factor),
            "metallic_factor": self.metallic_factor,
            "roughness_factor": self.roughness_factor,
            "emissive_factor": list(self.emissive_factor),
            "normal_scale": self.normal_scale,
        }

    def is_emissive(self) -> bool:
        """Any non-zero emissive component — the one predicate shared by
        the GLB writer (emits ``emissiveFactor``) and the material budget
        (an emissive material is structurally its own identity)."""
        return any(v != 0.0 for v in self.emissive_factor)

    def content_hash(self) -> str:
        # Memoized on first call — same immutability convention and
        # re-hash pressure as Image.result_hash (a bound material is
        # hashed once per downstream node, plus export and GLB writes).
        cached = self.__dict__.get("_content_hash")
        if cached is None:
            h = hashlib.sha256()
            h.update(json.dumps(self.factors(), sort_keys=True).encode("utf-8"))
            for slot in sorted(self.textures):
                h.update(f"tex:{slot}".encode("ascii"))
                h.update(self.textures[slot].result_hash().encode("ascii"))
            cached = self.__dict__["_content_hash"] = h.hexdigest()
        return cached


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) float32
    faces: np.ndarray  # (m, 3) int32, CCW front faces
    normals: np.ndarray | None = None  # (n, 3) float32 — glTF NORMAL
    uv: np.ndarray | None = None  # (n, 2) float32 — glTF TEXCOORD_0
    # Extra per-vertex slots keyed by glTF semantic (GLTF_EXTRA_ATTR_RE):
    # COLOR_n (n, 3|4) float32, TEXCOORD_n>=1 (n, 2) float32,
    # TANGENT (n, 4) float32.
    attributes: dict[str, np.ndarray] | None = None
    # Bound material; set by the material op, propagated by the
    # evaluator per MATERIAL_POLICY. None renders/exports as the neutral
    # shared default.
    material: Material | None = None
    # Named face groups: tag fixes a predicate selection
    # here as name -> sorted unique int32 face indices. Groups ride
    # result_hash / cache / stats (a tag is a semantic change downstream
    # must see) but never the GLB — they are authoring-side addressing,
    # not engine data. Propagation follows ATTRIBUTE_POLICY: same-topology
    # ops carry them (evaluator._propagate_face_groups), rebuilds drop
    # them with a warning.
    face_groups: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        if self.face_groups:
            for name in self.face_groups:
                if not FACE_GROUP_NAME_RE.fullmatch(name):
                    raise ValueError(
                        f"face group name '{name}' must match [A-Za-z][A-Za-z0-9_-]*"
                    )
                idx = np.unique(np.asarray(self.face_groups[name], dtype=np.int32))
                if len(idx) and (idx[0] < 0 or idx[-1] >= len(self.faces)):
                    raise ValueError(
                        f"face group '{name}' has indices outside 0..{len(self.faces) - 1}"
                    )
                self.face_groups[name] = idx  # canonical: sorted unique
        if not self.attributes:
            return
        for name, arr in self.attributes.items():
            if not GLTF_EXTRA_ATTR_RE.fullmatch(name):
                raise ValueError(
                    f"mesh attribute '{name}' is not a glTF extra-slot semantic "
                    f"(COLOR_n / TEXCOORD_n>=1 / TANGENT); NORMAL and TEXCOORD_0 "
                    f"are the dedicated normals/uv fields"
                )
            if len(arr) != len(self.vertices):
                raise ValueError(
                    f"mesh attribute '{name}' has {len(arr)} rows for "
                    f"{len(self.vertices)} vertices"
                )

    def result_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"v")
        h.update(np.ascontiguousarray(self.vertices, dtype=np.float32).tobytes())
        h.update(b"f")
        h.update(np.ascontiguousarray(self.faces, dtype=np.int32).tobytes())
        if self.normals is not None:
            h.update(b"n")
            h.update(np.ascontiguousarray(self.normals, dtype=np.float32).tobytes())
        if self.uv is not None:
            h.update(b"t")
            h.update(np.ascontiguousarray(self.uv, dtype=np.float32).tobytes())
        if self.attributes:
            for name in sorted(self.attributes):
                h.update(f"a:{name}".encode("ascii"))
                h.update(
                    np.ascontiguousarray(self.attributes[name], dtype=np.float32).tobytes()
                )
        if self.material is not None:
            h.update(b"m")
            h.update(self.material.content_hash().encode("ascii"))
        if self.face_groups:
            for name in sorted(self.face_groups):
                h.update(f"g:{name}".encode("ascii"))
                h.update(np.ascontiguousarray(self.face_groups[name], dtype=np.int32).tobytes())
        return h.hexdigest()

    def attribute_names(self) -> list[str]:
        """Present attribute slots beyond POSITION/NORMAL, glTF vocabulary."""
        names = [] if self.uv is None else ["TEXCOORD_0"]
        if self.attributes:
            names += self.attributes
        return sorted(names)

    def gltf_attributes(self) -> dict[str, np.ndarray]:
        """Present per-vertex slots (except POSITION) keyed by glTF semantic."""
        out: dict[str, np.ndarray] = {}
        if self.normals is not None:
            out["NORMAL"] = self.normals
        if self.uv is not None:
            out["TEXCOORD_0"] = self.uv
        if self.attributes:
            out.update(self.attributes)
        return out

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def stats(self) -> dict:
        lo, hi = self.bounds()
        s = {
            "verts": int(len(self.vertices)),
            "tris": int(len(self.faces)),
            "bounds": _stat_bounds(lo, hi),
        }
        if self.uv is not None:
            s["uv"] = True
        attrs = self.attribute_names()
        if attrs:
            s["attributes"] = attrs
        if self.material is not None:
            s["material"] = self.material.name
        if self.face_groups:
            # Face counts per group: a tag's selection is numerically
            # verifiable from stats alone (P6 rubric 4).
            s["groups"] = {n: int(len(g)) for n, g in sorted(self.face_groups.items())}
        return s

    def to_trimesh(self) -> trimesh.Trimesh:
        tm = trimesh.Trimesh(
            vertices=self.vertices.astype(np.float64),
            faces=self.faces.astype(np.int64),
            process=False,
        )
        if self.normals is not None:
            tm.vertex_normals = self.normals.astype(np.float64)
        tm.visual = _neutral_visual(self.uv)
        return tm


@dataclass
class Instances:
    prototype: Mesh
    transforms: np.ndarray  # (k, 4, 4) float64, row-major, applied as T @ [x y z 1]^T

    def result_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"proto")
        h.update(self.prototype.result_hash().encode("ascii"))
        h.update(b"xf")
        h.update(np.ascontiguousarray(self.transforms, dtype=np.float64).tobytes())
        return h.hexdigest()

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.prototype.bounds()
        corners = np.array(
            [[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )  # (8, 4)
        world = np.einsum("kij,cj->kci", self.transforms, corners)[..., :3].reshape(-1, 3)
        return world.min(axis=0), world.max(axis=0)

    def stats(self) -> dict:
        lo, hi = self.bounds()
        s = {
            "verts": int(len(self.prototype.vertices)),
            "tris": int(len(self.prototype.faces)),
            "instances": int(len(self.transforms)),
            "bounds": _stat_bounds(lo, hi),
        }
        attrs = self.prototype.attribute_names()
        if attrs:
            s["attributes"] = attrs
        if self.prototype.material is not None:
            s["material"] = self.prototype.material.name
        if self.prototype.face_groups:
            s["groups"] = {
                n: int(len(g)) for n, g in sorted(self.prototype.face_groups.items())
            }
        return s


Value = Mesh | Instances | Image


def bind_material(value: "Mesh | Instances", mat: Material) -> "Mesh | Instances":
    """Return ``value`` with ``mat`` bound — the one place that knows how to
    re-wrap a value around a new material: geometry arrays are shared
    (values are immutable by convention), the ``attributes`` and
    ``face_groups`` dicts are shallow-copied so ``Mesh.__post_init__``'s
    canonicalization never writes into the source value's dicts, and
    instancing passes through (the prototype is re-bound, transforms
    untouched). Used by the ``material`` op and ``asset_ref``'s
    ``material_rename``."""
    def bind(mesh: Mesh) -> Mesh:
        return replace(
            mesh,
            material=mat,
            attributes=None if not mesh.attributes else dict(mesh.attributes),
            face_groups=None if not mesh.face_groups else dict(mesh.face_groups),
        )

    if isinstance(value, Instances):
        return Instances(prototype=bind(value.prototype), transforms=value.transforms)
    return bind(value)


def proto_of(value: Mesh | Instances) -> Mesh:
    """The prototype Mesh of an Instances value; a Mesh is its own."""
    return value.prototype if isinstance(value, Instances) else value


def tri_cross(tri: np.ndarray) -> np.ndarray:
    """Edge cross products (v1-v0) x (v2-v0) of (m, 3, 3) corner triples.

    Direction = the unnormalized outward face normal, length = 2x face
    area, for CCW-wound faces — the single home of the winding-dependent
    edge order (DESIGN 4.3). Callers own their degenerate-face policy
    (clamp, mask, or reject): zero-area triples give zero vectors here.
    """
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


# Near-degeneracy bar for triangles, meters: height over the longest edge.
# Relative to the triangle's own shape, not an absolute area, and set four
# orders of magnitude above the coordinate perturbation a baked transform
# can introduce (O(scale * float64 eps) ~ 1e-13 m at the 1000 m scale cap):
# a triangle clearing this bar keeps clearing it when an asset_ref consumer
# transforms the mesh, so producing-asset green survives reuse (DESIGN 4.6).
DEGEN_HEIGHT_TOL = 1e-9


def degenerate_tri_count(vertices: np.ndarray, faces: np.ndarray) -> int:
    """Triangles thinner than DEGEN_HEIGHT_TOL (height over longest edge).

    The single home of the near-degeneracy criterion: `validate` reports
    it per output and op post-conditions (bevel) gate on it — both must
    judge a sliver identically or an op could emit meshes its own
    validation then rejects.
    """
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    area2 = np.linalg.norm(tri_cross(tri), axis=1)
    longest = np.sqrt(
        np.maximum.reduce(
            [
                ((tri[:, 1] - tri[:, 0]) ** 2).sum(axis=1),
                ((tri[:, 2] - tri[:, 1]) ** 2).sum(axis=1),
                ((tri[:, 0] - tri[:, 2]) ** 2).sum(axis=1),
            ]
        )
    )
    # area2 / longest edge = the triangle's height over that edge.
    height = np.divide(area2, longest, out=np.zeros_like(area2), where=longest > 0.0)
    return int((height <= DEGEN_HEIGHT_TOL).sum())


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Per-vertex normals: stored ones, else area-weighted from faces."""
    if mesh.normals is not None:
        return mesh.normals.astype(np.float64)
    v = mesh.vertices.astype(np.float64)
    f = mesh.faces
    fn = tri_cross(v[f.astype(np.int64)])  # area-weighted
    n = np.zeros_like(v)
    np.add.at(n, f.ravel(), np.repeat(fn, 3, axis=0))
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return n / lens


# Normal-agreement tolerance for coincident duplicated vertices: unit
# normals, so 1e-6 admits float32 storage noise, never a hard edge.
NORMAL_SPREAD_TOL = 1e-6


def divergent_duplicates(mesh: Mesh, welded=None) -> bool:
    """Do coincident duplicated vertices carry divergent normals?

    The flat-shaded signature (``box``, ``boolean``/``bevel`` output):
    identical positions whose normals disagree. UV-seam duplicates
    (normals agree) are not flat-shaded. The single home of the
    criterion — every op that refuses flat-shaded input must judge it
    identically. Non-finite normals count as divergent: a guard keyed
    on them must fail loud, never pass on a NaN comparison.
    ``welded``: optional precomputed ``(unique_count, inverse)`` of an
    exact-position weld, so a caller that already welded (extrude's
    closed-surface check) does not weld twice.
    """
    v = mesh.vertices.astype(np.float64)
    if welded is None:
        uniq, inv = np.unique(v, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        n_uniq = len(uniq)
    else:
        n_uniq, inv = welded
    if n_uniq == len(v):
        return False
    normals = vertex_normals(mesh)
    nmin = np.full((n_uniq, 3), np.inf)
    nmax = np.full((n_uniq, 3), -np.inf)
    np.minimum.at(nmin, inv, normals)
    np.maximum.at(nmax, inv, normals)
    # not-all-within (rather than any-above), so a NaN spread reads as
    # divergent instead of comparing False and slipping through.
    return not bool(np.all(nmax - nmin <= NORMAL_SPREAD_TOL))


def flat_shaded(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Rebuild with per-face duplicated vertices and flat normals."""
    v = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64).ravel()]
    m = len(v) // 3
    tri = v.reshape(m, 3, 3)
    fn = tri_cross(tri)
    lens = np.linalg.norm(fn, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    fn = fn / lens
    return Mesh(
        vertices=v.astype(np.float32),
        faces=np.arange(m * 3, dtype=np.int32).reshape(m, 3),
        normals=np.repeat(fn, 3, axis=0).astype(np.float32),
    )
