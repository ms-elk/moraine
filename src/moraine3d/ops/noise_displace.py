"""``noise_displace`` operator.

Semantics: displaces each vertex of its single mesh input along its
vertex normal by an fBm value-noise field sampled at the vertex position.
Params: ``amplitude`` meters (default 0.05) — peak displacement is about
+-amplitude; ``frequency`` cycles per meter (default 1); ``octaves``
(int, 1..10, default 4); ``lacunarity`` (default 2); ``gain`` (default
0.5); ``seed`` (int, default 0). The field is a function of position and
``seed`` only (not the node id), so two nodes with equal params produce
the identical field — decorrelate via ``seed``. A flat-shaded input
(coincident duplicated vertices with divergent normals — ``box``,
``boolean``/``bevel`` output) would displace coincident vertices apart
and tear the surface along every hard edge, so it is refused; weld the
shading first with ``smooth_normals``, raising ``angle_max_deg`` above
the hardest edge the displacement must cross (its default 30 keeps
hard edges split, which stay refused). Coincident duplicates whose
normals agree (UV seams) displace identically and pass. Topology and
UVs are unchanged; stored normals are kept as-is (recompute downstream
if the displacement is strong). Deterministic across runs and
platforms.
"""

from __future__ import annotations

import numpy as np

from ..core import numbers
from ..core import rand
from ..core.mesh import Mesh, divergent_duplicates, vertex_normals
from ..errors import OpError
from . import common

OP_IMPL_VERSION = 3
KERNELS = ("core.mesh", "core.rand")
INPUT_ARITY = (1, 1)
PARAM_KINDS = {
    "amplitude": numbers.FLOAT,
    "frequency": numbers.FLOAT,
    "octaves": numbers.INT,
    "lacunarity": numbers.FLOAT,
    "gain": numbers.FLOAT,
    "seed": numbers.INT,
}
PARAMS = frozenset(PARAM_KINDS)
SEED_PARAMS = frozenset({"seed"})  # reroll-with-history tier (DESIGN 7.2)
ATTRIBUTE_POLICY = "preserved"

# Fixed per-octave seed derivation (part of the output semantics; changing
# it is an OP_IMPL_VERSION bump). The stride keeps adjacent seeds off each
# other's octave lattice fields — with stride 1, seed s octave o+1 and
# seed s+1 octave o would sample the identical field, correlating the
# "decorrelated" surfaces the docstring promises.
_OCTAVE_SEED_STRIDE = 1000003


def _value_noise(p: np.ndarray, seed: int) -> np.ndarray:
    """3D value noise in [-1, 1] at points p (n, 3)."""
    p0 = np.floor(p).astype(np.int64)
    f = rand.fade(p - p0)
    n = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                c = rand.lattice_values(seed, p0[:, 0] + dx, p0[:, 1] + dy, p0[:, 2] + dz)
                n = n + c * wx * wy * wz
    return n


def evaluate(inputs: list, params: dict, ctx) -> Mesh:
    src = common.mesh_input("noise_displace", inputs, 0)
    amplitude = common.num("noise_displace", params, "amplitude", 0.05)
    frequency = common.num("noise_displace", params, "frequency", 1.0, positive=True)
    octaves = common.integer("noise_displace", params, "octaves", 4, minimum=1, maximum=10)
    lacunarity = common.num("noise_displace", params, "lacunarity", 2.0, positive=True)
    gain = common.num("noise_displace", params, "gain", 0.5, positive=True)
    seed = common.seed("noise_displace", params)

    v = src.vertices.astype(np.float64)
    # The int64 lattice bounds the sample domain: beyond it the floor
    # cast is platform-dependent (the overflow noise_texture's scale cap
    # prevents on its side). Guard on the actual extent, so every
    # param combination either works or fails loud.
    top_freq = frequency * lacunarity ** max(octaves - 1, 0)
    extent = float(np.abs(v).max()) if len(v) else 0.0
    if not np.isfinite(top_freq) or extent * top_freq >= 2.0**62:
        raise OpError(
            "bad_param",
            "noise_displace 'frequency' * 'lacunarity'^('octaves'-1) is too "
            "high for this mesh's coordinate range (int64 lattice overflow); "
            "lower 'frequency', 'lacunarity', or 'octaves'",
        )
    # The noise value is a function of position, so coincident duplicated
    # vertices displace identically — unless their normals diverge, which
    # moves them apart and tears the surface along every hard edge.
    if divergent_duplicates(src):
        raise OpError(
            "bad_input",
            "noise_displace input has duplicated (flat-shaded) vertices "
            "with divergent normals; displacing along them tears the "
            "surface at every hard edge - weld the shading first: "
            "smooth_normals with angle_max_deg above the hardest edge you "
            "mean to displace across (its default 30 keeps hard edges "
            "split, which stay refused here)",
        )
    normals = vertex_normals(src)

    total = np.zeros(len(v))
    amp, freq, norm = 1.0, frequency, 0.0
    for octave in range(octaves):
        total += amp * _value_noise(v * freq, seed + _OCTAVE_SEED_STRIDE * octave)
        norm += amp
        amp *= gain
        freq *= lacunarity
    total *= amplitude / norm

    verts = v + normals * total[:, None]
    return Mesh(
        vertices=verts.astype(np.float32),
        faces=src.faces,
        normals=src.normals,
        uv=src.uv,
        attributes=src.attributes,
    )
