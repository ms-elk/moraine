"""Golden-mesh tests for ``bake_instances`` (DESIGN 4.1, 5)."""

import numpy as np
import pytest

from moraine3d.core.mesh import Mesh
from moraine3d.errors import OpError
from moraine3d.ops import (
    array,
    bake_instances,
    boolean,
    box,
    cylinder,
    transform,
    unwrap,
)
from tests.test_ops_v1 import volume, watertight


def _row_of_boxes(count=3, step=2.0):
    return array.evaluate(
        [box.evaluate([], {}, None)],
        {"count": count, "step_translate": [step, 0.0, 0.0]},
        None,
    )


def test_bake_golden_counts_bounds_volume():
    baked = bake_instances.evaluate([_row_of_boxes()], {}, None)
    assert isinstance(baked, Mesh)
    assert len(baked.vertices) == 3 * 24
    assert len(baked.faces) == 3 * 12
    s = baked.stats()
    assert s["bounds"]["min"] == [-0.5, -0.5, -0.5]
    assert s["bounds"]["max"] == [4.5, 0.5, 0.5]
    # Disjoint closed shells: union volume is the sum, winding stays outward.
    assert volume(baked) == pytest.approx(3.0, rel=1e-6)
    assert watertight(baked)


def test_bake_rotated_instances_transforms_normals():
    inst = array.evaluate(
        [box.evaluate([], {}, None)],
        {"count": 2, "step_translate": [2.0, 0.0, 0.0], "step_rotate_deg": [0.0, 90.0, 0.0]},
        None,
    )
    baked = bake_instances.evaluate([inst], {}, None)
    # Normals stay unit-length under the rotation.
    lens = np.linalg.norm(baked.normals, axis=1)
    assert np.allclose(lens, 1.0, atol=1e-6)
    assert volume(baked) == pytest.approx(2.0, rel=1e-6)


def test_bake_preserves_uv_and_attributes_tiled():
    proto = unwrap.evaluate([box.evaluate([], {}, None)], {"mode": "box"}, None)
    proto.attributes = {"COLOR_0": np.ones((len(proto.vertices), 4), dtype=np.float32)}
    inst = array.evaluate([proto], {"count": 2, "step_translate": [2.0, 0.0, 0.0]}, None)
    baked = bake_instances.evaluate([inst], {}, None)
    n = len(proto.vertices)
    assert len(baked.uv) == 2 * n
    assert np.array_equal(baked.uv[:n], proto.uv)
    assert np.array_equal(baked.uv[n:], proto.uv)
    assert baked.attributes["COLOR_0"].shape == (2 * n, 4)


def test_bake_unblocks_boolean_with_repeated_cutters():
    # A bolt circle as repeated cutters.
    plate = transform.evaluate(
        [cylinder.evaluate([], {"radius": 0.25, "height": 0.04, "segments": 48}, None)],
        {"translate": [0.0, 0.02, 0.0]},
        None,
    )
    bolt = transform.evaluate(
        [cylinder.evaluate([], {"radius": 0.02, "height": 0.08, "segments": 16}, None)],
        {"translate": [0.18, 0.02, 0.0]},
        None,
    )
    circle = array.evaluate([bolt], {"count": 8, "step_rotate_deg": [0.0, 45.0, 0.0]}, None)
    baked = bake_instances.evaluate([circle], {}, None)
    out = boolean.evaluate([plate, baked], {"mode": "difference"}, None)
    assert watertight(out)
    # Plate volume minus eight through-holes of the plate's thickness;
    # each hole is the inscribed 16-gon prism, so the difference is exact
    # to kernel precision (one missing hole is a 0.7% deviation).
    plate_vol = volume(plate)
    hole = 0.5 * 16 * np.sin(2.0 * np.pi / 16) * 0.02**2 * 0.04
    assert volume(out) == pytest.approx(plate_vol - 8 * hole, rel=1e-4)
    assert out.stats()["bounds"]["min"][1] == pytest.approx(0.0, abs=1e-6)


def test_bake_rejects_mesh_input():
    with pytest.raises(OpError) as e:
        bake_instances.evaluate([box.evaluate([], {}, None)], {}, None)
    assert "already a mesh" in str(e.value)


def test_bake_rejects_negative_determinant():
    inst = _row_of_boxes(count=2)
    inst.transforms[1][:3, :3] = np.diag([-1.0, 1.0, 1.0]) @ inst.transforms[1][:3, :3]
    with pytest.raises(OpError) as e:
        bake_instances.evaluate([inst], {}, None)
    assert "determinant" in str(e.value)


def test_bake_rotates_tangents_per_copy():
    """TANGENT directions map through each instance's linear part like
    the geometry they frame (bake_instances semantics); plain tiling
    would keep the prototype-space frame on rotated copies."""
    proto = box.evaluate([], {}, None)
    n = len(proto.vertices)
    tan = np.zeros((n, 4), np.float32)
    tan[:, 0] = 1.0
    tan[:, 3] = 1.0
    proto.attributes = {"TANGENT": tan}
    inst = array.evaluate(
        [proto], {"count": 2, "step_rotate_deg": [0.0, 90.0, 0.0]}, None
    )
    baked = bake_instances.evaluate([inst], {}, None)
    got = baked.attributes["TANGENT"]
    assert np.allclose(got[:n, :3], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(got[n:, :3], [0.0, 0.0, -1.0], atol=1e-6)  # rotated copy
    assert np.allclose(got[:, 3], 1.0)
