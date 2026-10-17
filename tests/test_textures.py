"""Golden-image tests for the texture operators (DESIGN 5, 10):
shape/range/pattern assertions plus determinism, mirroring the
golden-mesh pattern (counts/bounds/volume) at the image level."""

import numpy as np
import pytest

from moraine3d.core.mesh import Image
from moraine3d.errors import OpError
from moraine3d.ops import (
    box,
    brick_texture,
    checker_texture,
    colorize,
    noise_texture,
    normal_map,
    ramp,
    streaks,
    unwrap,
)


def _assert_tileable(p):
    """Seam-vs-interior check in both axes: the wrap-around gradient must
    look like an interior gradient."""
    interior = np.abs(np.diff(p, axis=1)).max()
    assert np.abs(p[:, 0] - p[:, -1]).max() <= interior * 1.5
    interior_v = np.abs(np.diff(p, axis=0)).max()
    assert np.abs(p[0, :] - p[-1, :]).max() <= interior_v * 1.5


# -- noise_texture -----------------------------------------------------------


def test_noise_golden():
    img = noise_texture.evaluate([], {"size": 64, "scale": 4, "seed": 7}, None)
    assert isinstance(img, Image)
    assert img.pixels.shape == (64, 64, 1)
    assert img.pixels.dtype == np.float32
    p = img.pixels[..., 0]
    assert 0.0 <= p.min() and p.max() <= 1.0
    # fBm over a few octaves: neither flat nor saturated.
    assert 0.3 < p.mean() < 0.7
    assert p.std() > 0.02


def test_noise_deterministic_and_seeded():
    a = noise_texture.evaluate([], {"size": 32, "seed": 1}, None)
    b = noise_texture.evaluate([], {"size": 32, "seed": 1}, None)
    c = noise_texture.evaluate([], {"size": 32, "seed": 2}, None)
    assert a.result_hash() == b.result_hash()
    assert a.result_hash() != c.result_hash()


def test_noise_range_mapping():
    img = noise_texture.evaluate([], {"size": 16, "low": 0.25, "high": 0.5}, None)
    p = img.pixels
    assert p.min() >= 0.25 - 1e-6 and p.max() <= 0.5 + 1e-6


def test_noise_tileable():
    # The lattice wraps per octave, so the seam gradient must look like an
    # interior gradient — compare wrap-around diffs against interior diffs.
    img = noise_texture.evaluate([], {"size": 128, "scale": 4, "seed": 3}, None)
    p = img.pixels[..., 0]
    interior = np.abs(np.diff(p, axis=1)).max()
    seam = np.abs(p[:, 0] - p[:, -1]).max()
    assert seam <= interior * 1.5
    seam_v = np.abs(p[0, :] - p[-1, :]).max()
    assert seam_v <= np.abs(np.diff(p, axis=0)).max() * 1.5


def test_noise_octaves_capped():
    with pytest.raises(OpError):
        noise_texture.evaluate([], {"octaves": 11}, None)


def test_noise_mode_default_is_fbm():
    """Pins the semantics claim that mode "fbm" is the default shaping:
    an authored {"mode": "fbm"} and an absent mode produce the identical
    output (the additive-param / no-impl-version-bump contract)."""
    absent = noise_texture.evaluate([], {"size": 64, "seed": 7}, None)
    explicit = noise_texture.evaluate([], {"size": 64, "seed": 7, "mode": "fbm"}, None)
    assert absent.result_hash() == explicit.result_hash()


def test_noise_default_output_frozen():
    """Pins the semantics claim that the default output is bit-identical
    to this op before `mode` existed (additive param — same
    op_impl_version): the frozen hash below was captured from the
    pre-`mode` implementation and must never change at op_impl_version 1
    (DESIGN 4.5 — a bit change without a bump serves stale cache
    records)."""
    img = noise_texture.evaluate([], {"size": 64, "scale": 4, "seed": 7}, None)
    assert img.result_hash() == (
        "af5fc731048e990ca500040becbe640576472bb6e224ede496fc281b58dcecef"
    )


def test_noise_mode_stats():
    """Pins the per-octave shaping semantics: "ridged" (1 - 2|n|) biases
    bright, "billow" (2|n| - 1) biases dark, all modes are distinct, and
    every mode keeps the exact [low, high] bounds."""
    base = {"size": 64, "seed": 7}
    fbm = noise_texture.evaluate([], base, None)
    ridged = noise_texture.evaluate([], {**base, "mode": "ridged"}, None)
    billow = noise_texture.evaluate([], {**base, "mode": "billow"}, None)
    assert ridged.pixels.mean() > 0.55
    assert billow.pixels.mean() < 0.45
    hashes = {fbm.result_hash(), ridged.result_hash(), billow.result_hash()}
    assert len(hashes) == 3
    for img in (ridged, billow):
        assert 0.0 <= img.pixels.min() and img.pixels.max() <= 1.0


def test_noise_mode_tileable():
    """Pins the claim that the pointwise mode transform leaves tiling
    unchanged — "ridged" (the |n| fold) at the seam-vs-interior test."""
    img = noise_texture.evaluate(
        [], {"size": 128, "scale": 4, "seed": 3, "mode": "ridged"}, None
    )
    _assert_tileable(img.pixels[..., 0])


def test_noise_mode_rejected():
    with pytest.raises(OpError):
        noise_texture.evaluate([], {"mode": "turbulence"}, None)


# -- checker_texture ---------------------------------------------------------


def test_checker_golden():
    img = checker_texture.evaluate(
        [], {"size": 8, "tiles": 2, "color_a": [1.0, 0.0, 0.0], "color_b": [0.0, 0.0, 1.0]}, None
    )
    assert img.pixels.shape == (8, 8, 3)
    # Top-left square (u = v = 0) is color_a; adjacent square is color_b.
    assert np.allclose(img.pixels[0, 0], [1, 0, 0])
    assert np.allclose(img.pixels[0, 7], [0, 0, 1])
    assert np.allclose(img.pixels[7, 0], [0, 0, 1])
    assert np.allclose(img.pixels[7, 7], [1, 0, 0])


# -- brick_texture -----------------------------------------------------------


def test_brick_height_golden():
    """Height is 1 on brick faces and 0 in the joints; `mortar` is the
    joint thickness as a fraction of a course height, equally thick in
    both directions on a square tile — so the joint area is the analytic
    union of `rows` horizontal and `columns` vertical joints."""
    rows, columns, mortar = 4, 2, 0.08
    img = brick_texture.evaluate(
        [],
        {"size": 100, "rows": rows, "columns": columns, "mortar": mortar,
         "chamfer": 0.0, "map": "height"},
        None,
    )
    assert img.pixels.shape == (100, 100, 1)
    p = img.pixels[..., 0]
    assert p.min() == 0.0 and p.max() == 1.0
    mortar_frac = (p == 0.0).mean()
    analytic = mortar + columns * mortar / rows - columns * mortar**2 / rows
    assert abs(mortar_frac / analytic - 1.0) < 0.03


def test_brick_running_bond_for_any_columns():
    """The default `offset` 0.5 is a running bond for every `columns`:
    each course's vertical joints sit half a brick from the course
    below's, and courses two apart coincide."""
    for columns in (2, 3, 4):
        img = brick_texture.evaluate(
            [],
            {"size": 96, "rows": 4, "columns": columns, "chamfer": 0.0,
             "map": "height"},
            None,
        )
        mortar = img.pixels[..., 0] == 0.0
        # Rows through the middle of courses 0, 1 and 2 (24 px each).
        course0, course1, course2 = mortar[12], mortar[36], mortar[60]
        # One vertical joint per brick: count joint runs around the row.
        assert np.count_nonzero(course0 & ~np.roll(course0, 1)) == columns
        half_brick = 96 // (2 * columns)
        assert np.array_equal(course1, np.roll(course0, half_brick))
        assert not np.array_equal(course1, course0)
        assert np.array_equal(course2, course0)


def test_brick_odd_rows_seam_repeats_phase():
    """With odd `rows` the top course meets course 0 across the v seam in
    the same phase (an odd cycle cannot alternate); the tile still
    wraps without a discontinuity."""
    img = brick_texture.evaluate(
        [], {"size": 96, "rows": 3, "columns": 2, "chamfer": 0.0, "map": "height"},
        None,
    )
    mortar = img.pixels[..., 0] == 0.0
    course0, course1, course2 = mortar[16], mortar[48], mortar[80]
    assert not np.array_equal(course1, course0)
    assert np.array_equal(course2, course0)
    _assert_tileable(img.pixels[..., 0])


def test_brick_color_variation_and_determinism():
    params = {"size": 64, "seed": 5, "variation": 0.4}
    a = brick_texture.evaluate([], params, None)
    b = brick_texture.evaluate([], params, None)
    assert a.result_hash() == b.result_hash()
    assert a.pixels.shape == (64, 64, 3)
    # Distinct bricks got distinct value multipliers: the brick-face red
    # channel is not constant.
    red = a.pixels[..., 0]
    face = red[red > 0.2]  # brick faces (red-ish), not mortar-only pixels
    assert face.std() > 0.01
    # The seed drives the variation stream: another seed rerolls it.
    c = brick_texture.evaluate([], {**params, "seed": 6}, None)
    assert c.result_hash() != a.result_hash()
    assert not np.allclose(c.pixels, a.pixels, atol=1e-3)


def test_brick_color_height_alignment():
    base = {"size": 64, "rows": 4, "columns": 2, "chamfer": 0.0}
    color = brick_texture.evaluate([], {**base, "variation": 0.0}, None)
    height = brick_texture.evaluate([], {**base, "map": "height"}, None)
    mortar_mask = height.pixels[..., 0] == 0.0
    mortar_px = color.pixels[mortar_mask]
    assert np.allclose(mortar_px, [0.78, 0.76, 0.72], atol=1e-5)


def test_brick_tileable():
    img = brick_texture.evaluate([], {"size": 64, "map": "height"}, None)
    p = img.pixels[..., 0]
    assert np.abs(p[:, 0] - p[:, -1]).max() <= np.abs(np.diff(p, axis=1)).max() + 1e-6
    assert np.abs(p[0, :] - p[-1, :]).max() <= np.abs(np.diff(p, axis=0)).max() + 1e-6


# -- streaks -----------------------------------------------------------------


def test_streaks_golden():
    """Pins the stats contract: background pixels are exactly ``low``,
    streak cores approach ``high``, and default coverage is sparse
    (streaks are trails, not bands)."""
    img = streaks.evaluate(
        [], {"seed": 3, "size": 128, "count": 16, "width": 0.04, "length": 0.6}, None
    )
    assert isinstance(img, Image)
    assert img.pixels.shape == (128, 128, 1)
    assert img.pixels.dtype == np.float32
    p = img.pixels[..., 0]
    assert p.min() == 0.0
    assert p.max() > 0.5
    assert 0.01 < p.mean() < 0.3
    assert p.std() > 0.02


def test_streaks_deterministic_and_seeded():
    """Pins "deterministic and pure": equal params share one output,
    decorrelation comes from ``seed`` alone (no node-id salt)."""
    a = streaks.evaluate([], {"size": 64, "seed": 1}, None)
    b = streaks.evaluate([], {"size": 64, "seed": 1}, None)
    c = streaks.evaluate([], {"size": 64, "seed": 2}, None)
    assert a.result_hash() == b.result_hash()
    assert a.result_hash() != c.result_hash()


def test_streaks_tileable():
    """Pins "seamlessly tileable in both u and v" — the v seam is the
    risky side (head fade-in / tail fade-out must meet at zero)."""
    img = streaks.evaluate(
        [], {"size": 128, "seed": 5, "count": 24, "wiggle": 0.02}, None
    )
    _assert_tileable(img.pixels[..., 0])


def test_streaks_low_high():
    """Pins the mapping contract: swapped low/high makes dark streaks on
    a light map, background exactly ``low``, all pixels within the
    [min(low, high), max(low, high)] bounds."""
    img = streaks.evaluate([], {"size": 64, "low": 1.0, "high": 0.3}, None)
    p = img.pixels[..., 0]
    assert p.min() >= 0.3 - 1e-6
    assert p.max() == 1.0
    assert p.mean() > 0.7


def test_streaks_count_scales_coverage():
    """Pins the ``count`` semantics: more streaks, more coverage."""
    lo = streaks.evaluate([], {"size": 64, "count": 4, "seed": 9}, None)
    hi = streaks.evaluate([], {"size": 64, "count": 48, "seed": 9}, None)
    assert lo.pixels.mean() < hi.pixels.mean()


def test_streaks_param_errors():
    for bad in (
        {"count": 0},
        {"count": 65},
        {"length": 0.0},
        {"length": 1.5},
        {"width": 0.0},
        {"width": 0.6},
        {"wiggle": -0.1},
        {"wiggle": 0.3},
        {"seed": 2**63},
    ):
        with pytest.raises(OpError):
            streaks.evaluate([], bad, None)


# -- colorize ----------------------------------------------------------------


def test_colorize_golden():
    grad = Image(pixels=np.linspace(0, 1, 16, dtype=np.float32).reshape(1, 16, 1))
    out = colorize.evaluate(
        [grad], {"color_low": [0.0, 0.0, 0.0], "color_high": [1.0, 0.5, 0.0]}, None
    )
    assert out.pixels.shape == (1, 16, 3)
    assert np.allclose(out.pixels[0, 0], [0, 0, 0], atol=1e-6)
    assert np.allclose(out.pixels[0, -1], [1, 0.5, 0], atol=1e-6)
    mid = out.pixels[0, 8]
    assert np.allclose(mid, np.array([1, 0.5, 0]) * grad.pixels[0, 8, 0], atol=1e-6)


def test_colorize_rejects_color_input():
    rgbimg = Image(pixels=np.zeros((4, 4, 3), np.float32))
    with pytest.raises(OpError):
        colorize.evaluate([rgbimg], {}, None)


# -- ramp --------------------------------------------------------------------


def test_ramp_golden_rgb():
    """Pins the multi-stop semantics: between stops, linear
    interpolation; at a stop, exactly that stop's value; outside the
    end stops, the end values."""
    grad = Image(pixels=np.linspace(0, 1, 16, dtype=np.float32).reshape(1, 16, 1))
    out = ramp.evaluate(
        [grad],
        {
            "stops": [0.0, 0.5, 1.0],
            "values": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        },
        None,
    )
    assert out.pixels.shape == (1, 16, 3)
    assert np.allclose(out.pixels[0, 0], [0, 0, 0], atol=1e-6)
    assert np.allclose(out.pixels[0, -1], [1, 1, 1], atol=1e-6)
    # p = 0.4: 80% of the way from stop 0.0 to stop 0.5.
    p = grad.pixels[0, 6, 0]
    assert np.allclose(out.pixels[0, 6], [p / 0.5, 0, 0], atol=1e-6)
    # p = 0.8: 60% of the way from stop 0.5 to stop 1.0.
    p = grad.pixels[0, 12, 0]
    t = (p - 0.5) / 0.5
    assert np.allclose(out.pixels[0, 12], [1, t, t], atol=1e-6)


def test_ramp_scalar_mode():
    """Pins the claim that all-number values emit a 1-channel image
    (grayscale contrast shaping for height / roughness maps)."""
    grad = Image(pixels=np.linspace(0, 1, 16, dtype=np.float32).reshape(1, 16, 1))
    out = ramp.evaluate(
        [grad], {"stops": [0.0, 0.6, 1.0], "values": [0.45, 0.6, 0.9]}, None
    )
    assert out.pixels.shape == (1, 16, 1)
    assert np.allclose(out.pixels[0, 0, 0], 0.45, atol=1e-6)
    assert np.allclose(out.pixels[0, -1, 0], 0.9, atol=1e-6)
    p = float(grad.pixels[0, 4, 0])  # p < 0.6: inside the first segment
    assert np.allclose(out.pixels[0, 4, 0], 0.45 + (p / 0.6) * 0.15, atol=1e-6)


def test_ramp_default_identity():
    """Pins the default contract: no params is the identity grayscale
    ramp — output equals clamp(input, 0, 1), 1-channel."""
    grad = Image(pixels=np.linspace(0, 1, 8, dtype=np.float32).reshape(1, 8, 1))
    out = ramp.evaluate([grad], {}, None)
    assert out.pixels.shape == (1, 8, 1)
    assert np.allclose(out.pixels, grad.pixels, atol=1e-6)


def test_ramp_clamps_input():
    """Pins the end-clamp semantics: input at or below the first stop
    takes the first value, at or above the last stop the last value."""
    src = Image(pixels=np.array([[[-0.5], [1.5]]], dtype=np.float32))
    out = ramp.evaluate(
        [src], {"stops": [0.2, 0.8], "values": [0.3, 0.7]}, None
    )
    assert np.allclose(out.pixels[0, 0, 0], 0.3, atol=1e-6)
    assert np.allclose(out.pixels[0, 1, 0], 0.7, atol=1e-6)


def test_ramp_param_errors():
    grad = Image(pixels=np.zeros((4, 4, 1), np.float32))
    with pytest.raises(OpError):
        ramp.evaluate([grad], {"stops": [0.5], "values": [0.5]}, None)
    with pytest.raises(OpError):
        ramp.evaluate([grad], {"stops": [0.5, 0.5], "values": [0, 1]}, None)
    with pytest.raises(OpError):
        ramp.evaluate([grad], {"stops": [0.0, 1.2], "values": [0, 1]}, None)
    with pytest.raises(OpError):
        ramp.evaluate([grad], {"stops": [0.0, 0.5, 1.0], "values": [0, 1]}, None)
    with pytest.raises(OpError):
        ramp.evaluate(
            [grad], {"stops": [0.0, 1.0], "values": [0.5, [1, 0, 0]]}, None
        )
    rgbimg = Image(pixels=np.zeros((4, 4, 3), np.float32))
    with pytest.raises(OpError):
        ramp.evaluate([rgbimg], {}, None)


# -- normal_map --------------------------------------------------------------


def test_normal_map_flat():
    flat = Image(pixels=np.full((8, 8, 1), 0.5, np.float32))
    out = normal_map.evaluate([flat], {}, None)
    assert out.pixels.shape == (8, 8, 3)
    # Encoded straight-up normal (0, 0, 1) -> (0.5, 0.5, 1.0).
    assert np.allclose(out.pixels, [0.5, 0.5, 1.0], atol=1e-6)


def test_normal_map_convention_and_unit_length():
    # Height rising with v (image-down): the normal tilts toward image-up,
    # so green > 0.5 (the +Y "OpenGL" convention glTF requires). Wrap rows
    # at the seam skew; check an interior row.
    h, w = 16, 16
    ramp_v = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None, None], (1, w, 1))
    out = normal_map.evaluate([Image(pixels=ramp_v)], {"strength": 0.5}, None)
    assert out.pixels[8, 8, 1] > 0.5
    n = out.pixels * 2.0 - 1.0
    lengths = np.linalg.norm(n[1:-1, 1:-1], axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-3)

    # Height rising with u: normal tilts toward -u, so red < 0.5.
    ramp_u = np.tile(np.linspace(0, 1, w, dtype=np.float32)[None, :, None], (h, 1, 1))
    out_u = normal_map.evaluate([Image(pixels=ramp_u)], {"strength": 0.5}, None)
    assert out_u.pixels[8, 8, 0] < 0.5


def test_normal_map_tilts_a_face_like_real_geometry():
    """A height field encoded as a normal map, read through the UVs
    ``unwrap`` assigns, tilts a face exactly as displacing that face by
    the same height field would (normal_map semantics; the UV and
    normal-map conventions of DESIGN 4.3)."""
    n, strength = 128, 0.2
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    u, v = (j + 0.5) / n, (i + 0.5) / n
    h = 0.5 + 0.25 * np.sin(2 * np.pi * u) * np.sin(2 * np.pi * v)
    nmap = normal_map.evaluate(
        [Image(pixels=h.astype(np.float32)[..., None])], {"strength": strength}, None
    )

    # The tangent frame a glTF renderer builds from one 1 m face's UVs:
    # u to the right, and the map's +Y toward the image's top (-v).
    face = unwrap.evaluate(
        [box.evaluate([], {"size": [1.0, 1.0, 1.0]}, None)], {"mode": "box"}, None
    )
    front = np.abs(face.vertices[face.faces][..., 2].mean(axis=1) - 0.5) < 1e-6
    tri = face.faces[front][0]
    p3, uv3 = face.vertices[tri].astype(np.float64), face.uv[tri].astype(np.float64)
    dp_du, dp_dv = np.linalg.solve(uv3[1:] - uv3[0], p3[1:] - p3[0])
    up = np.array([0.0, 0.0, 1.0])  # the face normal

    def unit(a):
        return a / np.linalg.norm(a, axis=-1, keepdims=True)

    # Ground truth: the analytic normal of that face displaced by strength * h.
    dh_du = 0.25 * 2 * np.pi * np.cos(2 * np.pi * u) * np.sin(2 * np.pi * v)
    dh_dv = 0.25 * 2 * np.pi * np.sin(2 * np.pi * u) * np.cos(2 * np.pi * v)
    su = dp_du + strength * dh_du[..., None] * up
    sv = dp_dv + strength * dh_dv[..., None] * up
    want = unit(np.cross(su, sv))
    want *= np.sign((want * up).sum(-1))[..., None]

    decoded = nmap.pixels.astype(np.float64) * 2.0 - 1.0
    got = (
        decoded[..., 0:1] * unit(dp_du)
        + decoded[..., 1:2] * unit(-dp_dv)
        + decoded[..., 2:3] * up
    )
    cos = np.clip((unit(got) * want).sum(-1), -1.0, 1.0)
    assert np.degrees(np.arccos(cos)).max() < 0.5


def test_normal_map_rejects_negative_strength():
    flat = Image(pixels=np.zeros((4, 4, 1), np.float32))
    with pytest.raises(OpError):
        normal_map.evaluate([flat], {"strength": -1.0}, None)


def test_noise_scale_capped():
    with pytest.raises(OpError):
        noise_texture.evaluate([], {"scale": 4096}, None)
    with pytest.raises(OpError):
        noise_texture.evaluate([], {"seed": 2**63}, None)


def test_brick_offset_periodic():
    # offset is in brick widths and taken modulo one brick; out-of-range
    # values reduce instead of overflowing into platform-dependent
    # lattice casts.
    a = brick_texture.evaluate([], {"size": 32, "offset": 0.3}, None)
    b = brick_texture.evaluate([], {"size": 32, "offset": 7.3}, None)
    assert np.allclose(a.pixels, b.pixels, atol=1e-6)


def test_brick_variation_off_noise_lattice_plane():
    """Same-seed maps stay independent: brick variation hashes (seed,
    brick row, brick column) on the reserved iz=3 lattice plane (the
    plane registry in rand.lattice_values) — on iz=0 it would read the
    very corner values a same-seed noise_texture octave interpolates
    through, so brick brightness would track the noise field."""
    from moraine3d.core.rand import lattice_values

    seed, rows, columns, w = 7, 4, 4, 64
    img = brick_texture.evaluate(
        [],
        {
            "seed": seed, "size": w, "rows": rows, "columns": columns,
            "offset": 0.0, "variation": 1.0, "chamfer": 0.0,
            "brick_color": [1.0, 1.0, 1.0], "mortar_color": [0.0, 0.0, 0.0],
        },
        None,
    )
    # Face value at each brick center is 1 - variation * u01(brick).
    centers = (np.arange(rows) + 0.5) / rows * w
    px = img.pixels[np.ix_(centers.astype(int), centers.astype(int), [0])][..., 0]
    u01 = 1.0 - px
    ci = np.arange(columns)[None, :].repeat(rows, axis=0)
    rvi = np.arange(rows)[:, None].repeat(columns, axis=1)
    reserved = lattice_values(seed, ci, rvi, 3) * 0.5 + 0.5
    colliding = lattice_values(seed, ci, rvi, 0) * 0.5 + 0.5
    assert np.allclose(u01, reserved, atol=1e-5)
    assert not np.allclose(u01, colliding, atol=1e-2)


def test_noise_texture_frozen_hashes():
    """The lattice-table gather in fbm_grid is an evaluation strategy,
    not an output-semantics change: outputs are pinned bit-identical to
    the pre-table implementation (hashes captured before the change),
    per the impl-version rule's bit-identity requirement."""
    img = noise_texture.evaluate(
        [], {"seed": 5, "size": [128, 64], "scale": 8, "octaves": 5, "mode": "ridged"},
        None,
    )
    assert img.result_hash() == (
        "8cec47e20a4c53be5ce847ec1f88f4f367aa8c84fc97e90baa90ac268e6ca59e"
    )
    img2 = noise_texture.evaluate([], {"seed": 5, "size": 256, "scale": 4}, None)
    assert img2.result_hash() == (
        "27b8928255e53ddbc29547e3d8203784206fec7242f0e84a1d5e09736ce066a5"
    )


def test_checker_tiles_capped():
    """`tiles` is bounded at 2048 (checker_texture semantics — beyond one
    tile per pixel nothing is resolved): a huge count refuses with
    bad_param instead of overflowing the int cast platform-dependently
    into a silently uniform image (DESIGN 4.4)."""
    with pytest.raises(OpError) as e:
        checker_texture.evaluate([], {"tiles": 2049}, None)
    assert e.value.code == "bad_param" and "<= 2048" in e.value.message
    img = checker_texture.evaluate([], {"size": 8, "tiles": 2048}, None)
    assert img.pixels.shape == (8, 8, 3)


def test_brick_and_streaks_default_hashes_frozen():
    """Frozen at brick_texture op_impl_version 4 and streaks
    op_impl_version 1 — a change here means bump (DESIGN 4.5): the
    default outputs are pinned across runs by their result hashes, not
    only in-process. Captured on one x86-64 platform; the images are IEEE
    float arithmetic over hashed lattices, so any conformant platform
    must reproduce them bit-for-bit — a mismatch is a semantics change or
    a determinism gap (DESIGN 4.4), never noise."""
    assert brick_texture.OP_IMPL_VERSION == 4
    assert streaks.OP_IMPL_VERSION == 1
    assert brick_texture.evaluate([], {}, None).result_hash() == (
        "e1f7b1ba834f1c500aaee79b17e06b59d23de469d996904e66409b1763a3036d"
    )
    assert streaks.evaluate([], {}, None).result_hash() == (
        "8db6a25328200faa4ba82c74cc87b5fab38e03cf8d9c822c312ee39dae49b6ce"
    )
