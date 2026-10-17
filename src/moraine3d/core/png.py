"""Minimal deterministic PNG encoder.

Hand-rolled on stdlib ``zlib`` so texture export adds no dependency (the
Windows+macOS-wheel rule, DESIGN 3). Output is 8-bit grayscale or RGB,
filter type 0 on every row — byte-identical for identical pixels under
one zlib version (the manifest records ``zlib`` alongside the kernels,
the determinism boundary of DESIGN 4.4).

Transfer functions live here too: IR images are linear float32 (DESIGN
4.3 — albedo/emissive get sRGB-encoded at the edge, data maps stay
linear). glTF has no colorspace tag; consumers assume by material slot,
so applying the transfer at encode time is what makes the convention
hold by construction.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear [0, 1] -> sRGB [0, 1] (IEC 61966-2-1)."""
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def quantize8(x: np.ndarray) -> np.ndarray:
    """[0, 1] float -> uint8 with round-half-away rounding via +0.5 floor.

    NaN maps to 0 deterministically (a NaN->uint8 cast is platform-
    dependent in numpy). ``validate`` is the layer that *reports* NaN
    pixels; this guard only keeps un-validated preview encodes
    byte-deterministic across platforms.
    """
    return np.clip(np.nan_to_num(np.floor(x * 255.0 + 0.5), nan=0.0), 0, 255).astype(
        np.uint8
    )


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return b"".join(
        [
            struct.pack(">I", len(payload)),
            tag,
            payload,
            struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF),
        ]
    )


def encode_png(pixels8: np.ndarray) -> bytes:
    """uint8 pixels (h, w, 1|3) -> PNG bytes (grayscale / truecolor)."""
    p = np.asarray(pixels8)
    if p.dtype != np.uint8 or p.ndim != 3 or p.shape[2] not in (1, 3):
        raise ValueError("encode_png expects uint8 pixels shaped (h, w, 1|3)")
    h, w, c = p.shape
    color_type = 0 if c == 1 else 2
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    # Filter byte 0 prepended to each row; fixed compression level for
    # deterministic bytes under one zlib version.
    raw = np.concatenate([np.zeros((h, 1), np.uint8), p.reshape(h, w * c)], axis=1)
    idat = zlib.compress(raw.tobytes(), 9)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", idat),
            _chunk(b"IEND", b""),
        ]
    )
