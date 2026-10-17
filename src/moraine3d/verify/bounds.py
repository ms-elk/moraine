"""Exact world-space bounds of evaluated values (verify kernel): the one
AABB source the checks and every assert kind share — sound for
disjointness culling and for the distance path's gap/cull boxes alike.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Instances, Value


def world_bounds(value: Value) -> tuple[np.ndarray, np.ndarray]:
    """Exact world-space AABB over actual vertices, not transformed corners.

    Instances reduce axis by axis: peak memory is one (n, k) product
    at a time, never the full (k, n, 3) world-vertex array — this runs
    on every validate of every instanced output, so the transient
    matters for scatter-heavy assets.
    """
    if isinstance(value, Instances):
        v = value.prototype.vertices.astype(np.float64)
        xf = value.transforms
        lo, hi = np.empty(3), np.empty(3)
        for axis in range(3):
            w = v @ xf[:, axis, :3].T + xf[:, axis, 3]  # (n, k)
            lo[axis], hi[axis] = float(w.min()), float(w.max())
        return lo, hi
    v = value.vertices.astype(np.float64)
    return v.min(axis=0), v.max(axis=0)


def copy_world_bounds(value: Instances) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-copy world AABBs ``(lo (k, 3), hi (k, 3))`` over actual
    vertices — the per-copy form of ``world_bounds``."""
    return transformed_bounds(value.prototype.vertices, value.transforms)


def transformed_bounds(points, transforms) -> tuple[np.ndarray, np.ndarray]:
    """Exact AABB of ``points`` ``(n, 3)`` under each of ``transforms``
    ``(k, 4, 4)``: ``(lo (k, 3), hi (k, 3))``, axis by axis (one (n, k)
    product at a time)."""
    v = np.asarray(points, dtype=np.float64)
    xf = np.asarray(transforms, dtype=np.float64)
    k = len(xf)
    lo, hi = np.empty((k, 3)), np.empty((k, 3))
    for axis in range(3):
        w = v @ xf[:, axis, :3].T + xf[:, axis, 3]  # (n, k)
        lo[:, axis], hi[:, axis] = w.min(axis=0), w.max(axis=0)
    return lo, hi
