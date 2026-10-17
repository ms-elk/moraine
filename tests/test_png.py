"""The hand-rolled PNG encoder: spec-level chunk structure, exact pixel
round-trip through our own decode, sRGB transfer golden values."""

import struct
import zlib

import numpy as np
import pytest

from moraine3d.core.png import encode_png, quantize8, srgb_encode


def decode_png(data: bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + payload) & 0xFFFFFFFF
        chunks[tag] = payload
        pos += 12 + length
    w, h, depth, color_type, comp, filt, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert (depth, comp, filt, interlace) == (8, 0, 0, 0)
    c = {0: 1, 2: 3}[color_type]
    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, 1 + w * c)
    assert (raw[:, 0] == 0).all()  # filter type 0 rows
    return raw[:, 1:].reshape(h, w, c)


def test_png_roundtrip_rgb():
    rng = np.random.default_rng(0)
    px = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(encode_png(px)), px)


def test_png_roundtrip_gray():
    px = np.arange(16, dtype=np.uint8).reshape(4, 4, 1) * 16
    assert np.array_equal(decode_png(encode_png(px)), px)


def test_png_deterministic():
    px = np.full((8, 8, 3), 127, np.uint8)
    assert encode_png(px) == encode_png(px)


def test_png_rejects_bad_shapes():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 2), np.uint8))
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 3), np.float32))


def test_srgb_golden_values():
    # IEC 61966-2-1 anchor points: 0 -> 0, 1 -> 1, 0.5 linear -> ~188/255.
    assert srgb_encode(np.array(0.0)) == 0.0
    assert srgb_encode(np.array(1.0)) == pytest.approx(1.0)
    assert quantize8(srgb_encode(np.array([[0.5]]))).item() == 188
    assert quantize8(srgb_encode(np.array([[0.0031308]]))).item() == 10
