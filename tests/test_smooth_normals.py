"""Golden-mesh tests for smooth_normals (angle-threshold auto-smooth).

The contract: welded fans across edges no sharper than the threshold
share smoothed vertices; sharper edges keep hard corners; geometry
(faces, winding, volume) is untouched.
"""

import numpy as np
import pytest
import trimesh

from moraine3d.core.mesh import flat_shaded
from moraine3d.ops import boolean, box, cylinder, smooth_normals, transform


def volume(mesh) -> float:
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.merge_vertices()
    return float(tm.volume)


def test_smooth_normals_restores_cylinder_side():
    cyl = cylinder.evaluate([], {"radius": 0.5, "height": 2.0, "segments": 32}, None)
    flat = flat_shaded(cyl.vertices, cyl.faces)
    m = smooth_normals.evaluate([flat], {"angle_max_deg": 30.0}, None)
    # Fans regroup into the primitive's own sharing: smooth side rings,
    # hard cap rims -> the same vertex count as the cylinder itself.
    assert len(m.vertices) == 2 * 32 + 2 * (32 + 1)
    assert len(m.faces) == len(flat.faces)
    assert volume(m) == pytest.approx(volume(cyl), rel=1e-6)
    cap = np.abs(m.normals[:, 1]) > 0.99
    assert cap.sum() == 2 * (32 + 1)
    # Side normals are exactly radial: corner-angle weighting is symmetric
    # about the vertex regardless of where the quad diagonals fall.
    side_n = m.normals[~cap]
    side_v = m.vertices[~cap]
    assert np.abs(side_n[:, 1]).max() < 1e-6
    radial = side_v * [1.0, 0.0, 1.0]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", side_n, radial).min() > 1.0 - 1e-6


def test_smooth_normals_keeps_box_hard():
    b = box.evaluate([], {}, None)
    m = smooth_normals.evaluate([b], {"angle_max_deg": 30.0}, None)
    # 8 corners x 3 hard fans (the two triangles of each face are coplanar
    # and merge); every normal stays a face normal.
    assert len(m.vertices) == 24
    assert np.allclose(np.abs(m.normals).max(axis=1), 1.0, atol=1e-6)
    assert volume(m) == pytest.approx(1.0, rel=1e-6)


def test_smooth_normals_threshold_admits_all():
    b = box.evaluate([], {}, None)
    m = smooth_normals.evaluate([b], {"angle_max_deg": 100.0}, None)
    # 90-degree edges now smooth: full weld, one averaged normal per corner.
    assert len(m.vertices) == 8
    n = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    assert np.allclose(n, m.normals, atol=1e-6)


def test_smooth_normals_after_boolean_is_deterministic():
    a = box.evaluate([], {}, None)
    b = transform.evaluate(
        [box.evaluate([], {}, None)], {"translate": [0.5, 0.0, 0.0]}, None
    )
    u = boolean.evaluate([a, b], {"mode": "union"}, None)
    m1 = smooth_normals.evaluate([u], {}, None)
    m2 = smooth_normals.evaluate([u], {}, None)
    assert m1.result_hash() == m2.result_hash()
    assert len(m1.faces) == len(u.faces)
    assert volume(m1) == pytest.approx(1.5, rel=1e-5)
