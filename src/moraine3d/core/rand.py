"""Deterministic stochastic streams (DESIGN 4.4).

No global RNG anywhere. Every stochastic operator takes an explicit
``seed`` parameter and derives child streams as ``hash(seed, node_id,
index)``. The hash is a fixed integer mix (splitmix64 finalizer), so
results are identical across runs, platforms, and numpy versions.

Because ``node_id`` participates in the stream, an operator that uses
these helpers must declare ``USES_NODE_ID = True`` so the evaluator can
include the node id in its cache key (otherwise two nodes with identical
params would collide on one cache entry while producing different
results).
"""

from __future__ import annotations

import hashlib

import numpy as np

# Shared-kernel version (DESIGN 4.5): ops that consume this module declare
# it in KERNELS and this integer joins their cache keys. Bump on ANY
# output-semantics change of a function an op consumes (the stream mix,
# the lattice hash, the fade curve), bug fixes included — like
# OP_IMPL_VERSION.
KERNEL_VERSION = 1

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, vectorized over uint64 arrays (wrapping)."""
    with np.errstate(over="ignore"):  # uint64 wraparound is the algorithm
        x = np.asarray(x, dtype=np.uint64)
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def _node_salt(node_id: str) -> np.uint64:
    digest = hashlib.sha256(node_id.encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], "little"))


def stream_u64(seed: int, node_id: str, index) -> np.ndarray:
    """hash(seed, node_id, index) -> uint64 array, shape of ``index``."""
    with np.errstate(over="ignore"):
        base = _mix(np.uint64(seed) + _GOLDEN) ^ _node_salt(node_id)
        idx = np.asarray(index, dtype=np.uint64)
        return _mix(base + (idx + np.uint64(1)) * _GOLDEN)


def uniform(seed: int, node_id: str, index, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Deterministic uniforms in [lo, hi) indexed by ``index``."""
    bits = stream_u64(seed, node_id, index)
    u = (bits >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return lo + u * (hi - lo)


def lattice_values(seed: int, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Deterministic values in [-1, 1) on integer lattice points.

    Used by value-noise; no node_id salt so identical noise fields are
    shareable across nodes (single-stream noise decorrelates via ``seed``).

    Same-seed consumers share this one hash space. The isolation model
    has two tiers, because plane reservation only contains consumers
    whose ``iz`` is bounded:

    - Bounded consumers reserve ``iz`` planes among themselves:
      noise_texture octaves sample iz=0 only, streaks hashes on iz=1
      and meanders on iz=2, brick_texture varies on iz=3. Colliding
      planes make two same-seed maps visibly track each other, so a
      new bounded consumer claims an unused plane and records it here.
    - The 3D field (noise_displace) samples every plane its mesh
      extent covers, so no plane is isolated from it: a same-seed
      bounded consumer and the 3D field read identical values at the
      isolated lattice corners the bounded consumer touches. That
      overlap is accepted — it is corner-point sharing, not map-level
      tracking (and the noise family shares its field deliberately,
      octave strides included). A consumer that needs hard isolation
      from the 3D field decorrelates in seed space instead: a large
      fixed offset on its seed (the ``_OCTAVE_SEED_STRIDE`` precedent),
      recorded here.
    """
    p1, p2, p3 = np.uint64(0x8DA6B343), np.uint64(0xD8163841), np.uint64(0xCB1AB31F)
    with np.errstate(over="ignore"):
        h = (
            np.asarray(ix, np.int64).astype(np.uint64) * p1
            ^ np.asarray(iy, np.int64).astype(np.uint64) * p2
            ^ np.asarray(iz, np.int64).astype(np.uint64) * p3
            ^ _mix(np.uint64(seed) + _GOLDEN)
        )
    u = (_mix(h) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return u * 2.0 - 1.0


def fade(t: np.ndarray) -> np.ndarray:
    """Quintic smoothstep for lattice-noise interpolation.

    Part of the output semantics of every consumer (noise_texture,
    streaks, noise_displace): changing this curve is a
    ``KERNEL_VERSION`` bump on this module, which rekeys them all.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
