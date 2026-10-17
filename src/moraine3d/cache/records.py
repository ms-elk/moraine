"""Cache records on disk (DESIGN 4.5): the write side of the cache grammar.

A record is ``<key>.npz`` (the value: mesh / instances / image, with
material and face groups) plus ``<key>.meta.json`` (result hash and
stats); derived artifacts are ``<key>-g<GLB_VERSION>-<output>.glb`` and
``<key>-i<IMG_VERSION>-<output>.png``. Every name is composed in this
module (``record_paths`` / ``glb_path`` / ``png_path``) — the write side
of the grammar ``cache.store._NAME_RE`` / ``app.server._CACHE_FILE_RE``
recognize; change them together. ``evaluator.CARRY_VERSION`` folds the
npz layout into every cache key: bump it when ``save_value`` /
``load_value`` change what a record holds.

Every file goes through ``write_atomic`` (temp name, then rename —
``core.atomic``): the cache is shared across concurrent processes, so a
reader sees an older record or none, never a partial one. Concurrent
writers of one key produce identical bytes (determinism, DESIGN 4.4),
so the last rename winning is benign.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

from ..core.atomic import ReplaceError, atomic_replace
from ..core.glb import GLB_VERSION, build_glb
from ..core.mesh import Image, Instances, Material, Mesh, Value, proto_of
from ..core.paths import PREVIEW_DIR_NAME, cache_dir, preview_dir
from ..core.png import encode_png, quantize8, srgb_encode
from ..errors import MoraineError
from . import store as cache_mod


def write_atomic(path: Path, data: bytes) -> None:
    """Write one content-addressed cache file atomically, with the cache's
    failure policy: a leftover temp file goes through the deletion choke
    point; a destination another process holds open (Windows refuses the
    replace — the viewer streaming a GLB, ``np.load`` on a record) is the
    benign lost race of the module contract, since the bytes derive from
    the name and the existing file already holds them (the write-side
    counterpart of ``cache.store.unlink_cache_file``'s lock skip); a temp
    file that vanished under us is a concurrent cache clear
    (``cache_missing``); anything else is ``cache_unwritable``."""
    try:
        atomic_replace(path, data)
    except ReplaceError as err:
        cache_mod.unlink_cache_file(err.tmp)
        e = err.cause
        if err.step == "replace":
            if isinstance(e, PermissionError) and path.exists():
                return
            if isinstance(e, FileNotFoundError):
                raise MoraineError(
                    "cache_missing",
                    f"cache record {path.name} disappeared mid-write (a "
                    "concurrent cache clear or eviction); re-run the "
                    "command to re-evaluate",
                ) from e
        raise _cache_unwritable(path, e) from e


def _cache_unwritable(path: Path, e: OSError) -> MoraineError:
    # Same code and fix as paths.cache_dir's mkdir probe: a dir that
    # exists but refuses writes (ACL, read-only mount, disk full).
    return MoraineError(
        "cache_unwritable",
        f"cache dir '{path.parent}' is not writable "
        f"({e.__class__.__name__}: {e}); set MORAINE_CACHE_DIR to a "
        "writable directory",
    )


def save_value(value: Value, npz_path: Path) -> int:
    arrays: dict[str, np.ndarray] = {}
    if isinstance(value, Image):
        # Texture buffers cache uniformly with meshes (DESIGN 4.5); the
        # ``pixels`` array is the record-kind discriminator.
        arrays["pixels"] = value.pixels
    else:
        proto = proto_of(value)
        arrays["vertices"] = proto.vertices
        arrays["faces"] = proto.faces
        if proto.normals is not None:
            arrays["normals"] = proto.normals
        if proto.uv is not None:
            arrays["uv"] = proto.uv
        if proto.attributes:
            # Extra glTF slots (mesh.py attribute model), namespaced so their
            # semantic names can never collide with the core arrays above.
            for name, arr in proto.attributes.items():
                arrays[f"attr:{name}"] = arr
        if proto.face_groups:
            for name, idx in proto.face_groups.items():
                arrays[f"grp:{name}"] = idx
        if proto.material is not None:
            mat = proto.material
            arrays["mat:json"] = np.frombuffer(
                json.dumps(mat.factors(), sort_keys=True).encode("utf-8"), dtype=np.uint8
            )
            for slot, img in mat.textures.items():
                arrays[f"mat:tex:{slot}"] = img.pixels
        if isinstance(value, Instances):
            arrays["transforms"] = value.transforms
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    write_atomic(npz_path, data)
    return len(data)


def _load_material(z) -> Material | None:
    if "mat:json" not in z.files:
        return None
    fac = json.loads(bytes(z["mat:json"]).decode("utf-8"))
    textures = {
        k.removeprefix("mat:tex:"): Image(pixels=z[k])
        for k in z.files
        if k.startswith("mat:tex:")
    }
    return Material(
        name=fac["name"],
        base_color_factor=tuple(fac["base_color_factor"]),
        metallic_factor=fac["metallic_factor"],
        roughness_factor=fac["roughness_factor"],
        emissive_factor=tuple(fac["emissive_factor"]),
        normal_scale=fac["normal_scale"],
        textures=textures,
    )


def load_value(npz_path: Path) -> Value:
    try:
        with np.load(npz_path) as z:
            if "pixels" in z.files:
                return Image(pixels=z["pixels"])
            attributes = {
                k.removeprefix("attr:"): z[k] for k in z.files if k.startswith("attr:")
            }
            face_groups = {
                k.removeprefix("grp:"): z[k] for k in z.files if k.startswith("grp:")
            }
            mesh = Mesh(
                vertices=z["vertices"],
                faces=z["faces"],
                normals=z["normals"] if "normals" in z else None,
                uv=z["uv"] if "uv" in z else None,
                attributes=attributes or None,
                material=_load_material(z),
                face_groups=face_groups or None,
            )
            if "transforms" in z:
                return Instances(prototype=mesh, transforms=z["transforms"])
        return mesh
    except FileNotFoundError as e:
        # The record vanished between the hit check and this lazy load —
        # a concurrent `cache clear` (which has no freshness guard), not
        # corruption. Name the actual cause; a "corrupt" report would
        # send the user chasing a defect that never existed.
        raise MoraineError(
            "cache_missing",
            f"cache record {npz_path.name.removesuffix('.npz')} disappeared "
            "mid-run (a concurrent cache clear or eviction); re-run the "
            "command to re-evaluate",
        ) from e
    except Exception as e:
        # Corrupted record (bad zip, missing arrays, truncated file):
        # remove it so the next run re-evaluates instead of failing again.
        # Through the tool-wide choke point, meta first (4.5).
        key = npz_path.name.removesuffix(".npz")
        cache_mod.unlink_cache_file(npz_path.with_name(f"{key}.meta.json"))
        cache_mod.unlink_cache_file(npz_path)
        # The choke point swallows deletion failures (locks); the error
        # must not promise a removal that did not happen, or the fix
        # path loops forever.
        if npz_path.exists():
            raise MoraineError(
                "cache_corrupt",
                f"cache record {key} is unreadable and could not be removed "
                "(held open by another process?); delete "
                f"'{npz_path.name}' from the cache dir manually, then re-run",
            ) from e
        raise MoraineError(
            "cache_corrupt",
            f"cache record {key} was unreadable and has been removed; "
            "re-run the command to re-evaluate it",
        ) from e


def _vec3(v: object) -> bool:
    # null components are the JSON-safe form of non-finite bounds
    # (mesh.stats), so a record of overflowed geometry is a valid shape.
    return (
        isinstance(v, list)
        and len(v) == 3
        and all(x is None or isinstance(x, (int, float)) for x in v)
    )


def meta_intact(meta: object) -> bool:
    """The record-shape contract a cached meta must satisfy before the
    hit path serves it: the fields the tool later subscripts must carry
    the types they are subscripted with — result_hash, and the stats
    keys consumed by the CLI asset rollup, the attribute-drop warnings,
    and export. Valid JSON failing this is corruption (4.5): serving
    it as a hit would keep a poisoned record alive and surface later as
    a raw traceback (8.2) instead of self-healing here."""
    if not isinstance(meta, dict) or not isinstance(meta.get("result_hash"), str):
        return False
    stats = meta.get("stats")
    if not isinstance(stats, dict):
        return False
    if stats.get("kind") == "image":
        return True  # image stats are never subscripted for geometry
    bounds = stats.get("bounds")
    return (
        isinstance(stats.get("verts"), int)
        and isinstance(stats.get("tris"), int)
        and isinstance(bounds, dict)
        and _vec3(bounds.get("min"))
        and _vec3(bounds.get("max"))
        and isinstance(stats.get("instances", 0), int)
        and isinstance(stats.get("attributes", []), list)
        and isinstance(stats.get("groups", {}), dict)
        and isinstance(stats.get("material", ""), str)
    )


def stats_attributes(stats: dict) -> list[str]:
    """Attribute slots recorded in cached stats. Records written before the
    attribute model carry only the legacy ``uv`` flag; derive TEXCOORD_0
    from it so old cache metas keep working without invalidation."""
    attrs = stats.get("attributes")
    if attrs is not None:
        return list(attrs)
    return ["TEXCOORD_0"] if stats.get("uv") else []


class Record:
    """Per-node evaluation record; loads the cached value lazily.
    ``preview`` marks a record living in the ephemeral preview bucket
    (DESIGN 4.5) rather than the persistent store: its derived artifacts
    follow it there, and its bytes never count toward the session write
    accounting that triggers persistent eviction."""

    def __init__(
        self,
        key: str,
        result_hash: str,
        stats: dict,
        value: Value | None,
        preview: bool = False,
    ):
        self.key = key
        self.result_hash = result_hash
        self.stats = stats
        self.preview = preview
        self._value = value

    def value(self) -> Value:
        if self._value is None:
            self._value = load_value(record_paths(self.key, self.preview)[0])
        return self._value


# -- record protocol ----------------------------------------------------------


def record_paths(key: str, preview: bool = False) -> tuple[Path, Path]:
    """``(npz, meta)`` for a key — the one place record names are spelled.
    ``preview`` roots the pair in the ephemeral bucket instead of the
    persistent store; the names themselves are identical (one grammar).
    Names only — the bucket dir is never created here: reads (lookup,
    lazy value loads) must not create it, that is ``store``'s moment
    (the bucket appears on the first preview write, DESIGN 4.5)."""
    cdir = cache_dir() / PREVIEW_DIR_NAME if preview else cache_dir()
    return cdir / f"{key}.npz", cdir / f"{key}.meta.json"


def lookup(key: str, preview: bool = False) -> Record | None:
    """The on-disk record for ``key`` with a lazy value, or None when there
    is none. A record exists only when both files do. A corrupt or
    truncated meta — including valid JSON that fails the record-shape
    contract (``meta_intact``) — is dropped through the choke point, meta
    first (DESIGN 4.5), and reads as absent: nothing has been computed
    yet, so the caller simply re-evaluates (unlike ``load_value``'s
    recovery, which must raise). Touch-on-hit IS the hit count (4.5: the
    hit path that touches is the hit path that counts); only the meta is
    touched — npz mtimes stay dirty-subgraph evidence."""
    npz_path, meta_path = record_paths(key, preview)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta_intact(meta):
            raise TypeError("meta does not carry the record shape")
        result_hash, stats = meta["result_hash"], meta["stats"]
    except (OSError, ValueError, KeyError, TypeError):
        cache_mod.unlink_cache_file(meta_path)
        cache_mod.unlink_cache_file(npz_path)
        return None
    try:
        os.utime(meta_path)
    except OSError:
        pass  # racing eviction: the record re-derives next pass
    return Record(key, result_hash, stats, None, preview)


def store(key: str, value: Value, preview: bool = False) -> Record:
    """Persist a freshly computed value under ``key`` and return its
    record (value attached). npz first, meta second: ``lookup`` requires
    both, so a record becomes visible only once its value file is
    complete. ``allow_nan=False``: stats are JSON-safe by construction
    (non-finite bounds are null); a NaN reaching here is a stats bug, and
    an 'Infinity' token in the meta would poison every later reader.
    ``preview`` lands the record in the ephemeral bucket and skips the
    session write accounting: bucket bytes sit outside the persistent cap
    (DESIGN 4.5), so they must not arm the post-growth eviction check."""
    if preview:
        preview_dir()  # first preview write creates the bucket (DESIGN 4.5)
    npz_path, meta_path = record_paths(key, preview)
    wrote = save_value(value, npz_path)
    meta = {"result_hash": value.result_hash(), "stats": value.stats()}
    meta_bytes = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode("utf-8")
    write_atomic(meta_path, meta_bytes)
    if not preview:
        cache_mod.note_write(wrote + len(meta_bytes), npz_path.parent)
    return Record(key, meta["result_hash"], meta["stats"], value, preview)


# -- derived artifacts: glb / preview png ---------------------------------------

# Version of the cached image-output previews (``<key>-i<ver>-<name>.png``),
# the role GLB_VERSION (core.glb) plays for the GLBs: part of the filename,
# so stale previews from an older writer are orphaned, never served.
IMG_VERSION = 1


def glb_path(key: str, output_name: str, preview: bool = False) -> Path:
    # Names only, like record_paths: a preview record's bucket already
    # exists (its own store created it), so the derived-artifact writers
    # need no mkdir and the read paths must not get one.
    d = cache_dir() / PREVIEW_DIR_NAME if preview else cache_dir()
    return d / f"{key}-g{GLB_VERSION}-{output_name}.glb"


def png_path(key: str, output_name: str, preview: bool = False) -> Path:
    d = cache_dir() / PREVIEW_DIR_NAME if preview else cache_dir()
    return d / f"{key}-i{IMG_VERSION}-{output_name}.png"


def ensure_glb(rec: Record, output_name: str) -> Path:
    """The cached viewer GLB for an output, written on first use.
    Deterministic naming: glTF nodes/meshes named after the asset output
    (Day-One Invariant 7). Fixed generator string: the file is keyed by
    content + GLB_VERSION, so a version string here would only churn
    bytes."""
    path = glb_path(rec.key, output_name, rec.preview)
    if not path.exists():
        data, _info = build_glb({output_name: rec.value()}, generator="moraine viewer")
        write_atomic(path, data)
        if not rec.preview:
            cache_mod.note_write(len(data), path.parent)
    return path


def ensure_preview_png(rec: Record, output_name: str) -> Path:
    """The cached viewer preview of an image output, written on first use —
    sRGB-encoded for display regardless of slot (export encodes per
    material slot; these bytes never leave the cache)."""
    path = png_path(rec.key, output_name, rec.preview)
    if not path.exists():
        data = encode_png(quantize8(srgb_encode(rec.value().pixels)))
        write_atomic(path, data)
        if not rec.preview:
            cache_mod.note_write(len(data), path.parent)
    return path
