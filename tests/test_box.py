"""Golden-mesh tests for the box operator (DESIGN 10).

These pin the operator's semantics contract: counts, bounds, volume,
winding. Implementation swaps must keep them green.
"""

import numpy as np
import pytest
import trimesh

from moraine3d.errors import OpError
from moraine3d.ops import box


def test_box_golden():
    mesh = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    assert mesh.vertices.shape == (24, 3)
    assert mesh.faces.shape == (12, 3)
    s = mesh.stats()
    assert s["verts"] == 24
    assert s["tris"] == 12
    assert s["bounds"]["min"] == [-0.5, -1.0, -1.5]
    assert s["bounds"]["max"] == [0.5, 1.0, 1.5]


def test_box_volume_and_winding():
    mesh = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    # Signed volume is positive only if triangles wind CCW seen from outside.
    assert tm.volume == pytest.approx(6.0, abs=1e-6)


def test_box_normals_outward():
    mesh = box.evaluate([], {"size": [2.0, 2.0, 2.0]}, None)
    # Each vertex normal must point away from the origin (outward).
    dots = np.sum(mesh.vertices * mesh.normals, axis=1)
    assert np.all(dots > 0)


def test_box_default_size():
    s = box.evaluate([], {}, None).stats()
    assert s["bounds"]["min"] == [-0.5, -0.5, -0.5]
    assert s["bounds"]["max"] == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("size", [[0, 1, 1], [1, 1], "big", [1, 1, -2], None])
def test_box_rejects_bad_size(size):
    with pytest.raises(OpError):
        box.evaluate([], {"size": size}, None)


def test_box_default_result_hash_frozen():
    """`segments` is additive: the default path is bit-identical to the
    pre-param output, so op_impl_version stays 1 (DESIGN 4.5) and this
    hash pins that identity. Same platform caveat as the scatter hash:
    IEEE float arithmetic only, reproduced bit-for-bit on any conformant
    platform."""
    assert box.OP_IMPL_VERSION == 1
    mesh = box.evaluate([], {"size": [1.0, 2.0, 3.0]}, None)
    assert mesh.result_hash() == (
        "87504ac3a6f0c0f3469902e60bc05bf0b4b9ec91150e1747ebecbb6b5b2af588"
    )
    same = box.evaluate([], {"size": [1.0, 2.0, 3.0], "segments": [1, 1, 1]}, None)
    assert same.result_hash() == mesh.result_hash()


def test_box_segmented_golden():
    """Each face is its own (nu + 1) x (nv + 1) lattice with hard edges
    and per-face outward normals; segments change the tessellation, not
    the shape (box semantics)."""
    mesh = box.evaluate([], {"size": [1.0, 2.0, 3.0], "segments": [2, 3, 4]}, None)
    # 2 faces each of 3x4, 4x5, 5x3 lattices; 2 tris per quad.
    assert mesh.vertices.shape == (2 * (3 * 4 + 4 * 5 + 5 * 3), 3)
    assert mesh.faces.shape == (2 * 2 * (2 * 3 + 3 * 4 + 4 * 2), 3)
    s = mesh.stats()
    assert s["bounds"]["min"] == [-0.5, -1.0, -1.5]
    assert s["bounds"]["max"] == [0.5, 1.0, 1.5]
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    assert tm.volume == pytest.approx(6.0, abs=1e-6)
    assert np.allclose(tm.face_normals, mesh.normals[mesh.faces[:, 0]], atol=1e-6)
    assert np.all(np.sum(mesh.vertices * mesh.normals, axis=1) > 0)
    welded = tm.copy()
    welded.merge_vertices()
    assert welded.is_watertight


@pytest.mark.parametrize(
    "segments", [[0, 1, 1], [1, 1], [1, 1.5, 1], [True, 1, 1], [4097, 1, 1], "many"]
)
def test_box_rejects_bad_segments(segments):
    with pytest.raises(OpError) as exc:
        box.evaluate([], {"segments": segments}, None)
    assert exc.value.detail["param"] == "segments"


def test_box_segments_within_shared_cell_budget():
    """The per-axis ceilings multiply across three face pairs, so the
    shared tessellation cell budget is enforced at runtime (the revolve
    precedent), before allocation, as an actionable refusal."""
    with pytest.raises(OpError) as exc:
        box.evaluate([], {"segments": [4096, 4096, 1]}, None)
    assert exc.value.detail["param"] == "segments" and "budget" in exc.value.message
    box.evaluate([], {"segments": [2048, 1024, 1]}, None)  # 2*(2M+1024+2048) fits
