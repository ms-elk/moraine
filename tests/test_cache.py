"""Cache size management: LRU eviction, freshness guard, deletion safety
(DESIGN 4.5). Instrumentation counters are covered in
test_asset_eval / test_cli; this file pins the eviction engine."""

import os
import re
import threading
import time
from pathlib import Path

import pytest

from conftest import SRC_ROOT, package_sources
from moraine3d.cache import store as cache
from moraine3d.core import atomic, paths
from moraine3d.errors import MoraineError

NOW = time.time()


@pytest.fixture
def cdir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(d))
    d.mkdir()
    # A legitimate Moraine cache carries the tag the write path stamps;
    # the untagged case is exercised explicitly by the gate tests.
    (d / cache.TAG_NAME).write_bytes(cache._TAG_BODY)
    return d


def _key(i: int) -> str:
    return f"{i:024x}"


def _mk_group(cdir: Path, i: int, *, npz=1000, meta=100, glb=0, age=0.0) -> str:
    """Write one record group; mtimes set to NOW - age seconds."""
    key = _key(i)
    names = [(f"{key}.npz", npz), (f"{key}.meta.json", meta)]
    if glb:
        names.append((f"{key}-g3-main.glb", glb))
    for name, size in names:
        p = cdir / name
        p.write_bytes(b"x" * size)
        os.utime(p, (NOW - age, NOW - age))
    return key


def _group_files(cdir: Path, key: str) -> list[str]:
    return sorted(p.name for p in cdir.iterdir() if p.name.startswith(key))


# A stand-in for an outgoing grammar a future rename would register:
# fullmatch-anchored, key captured as group 1, covering the in-flight
# .tmp variants (the _LEGACY_NAME_RES contract).
_OLD_GLB = re.compile(r"^([0-9a-f]{24})-old\.glb(\.tmp\d+)?\Z")


# -- LRU / hysteresis / freshness --------------------------------------------


def test_evict_lru_order_and_hysteresis(cdir):
    """Eviction deletes oldest-mtime groups first and runs down to 90% of
    the cap, not merely under it (DESIGN 4.5: hysteresis, so consecutive
    sessions don't each trip it)."""
    for i in range(10):  # group i is older for lower i: age 20h .. 11h
        _mk_group(cdir, i, npz=1000, meta=100, glb=0, age=(20 - i) * 3600)
    # total = 11000 bytes; cap 8000 -> target 7200 -> free >= 3800 -> 4 groups
    report = cache.evict(cap=8000, now=NOW)
    assert report["bytes"] == 11000
    assert report["evicted"] == 4
    assert report["after"] <= int(8000 * cache.EVICT_TO)
    for i in range(4):  # the four oldest are gone, whole-group
        assert _group_files(cdir, _key(i)) == []
    for i in range(4, 10):  # the rest untouched
        assert len(_group_files(cdir, _key(i))) == 2


def test_evict_noop_under_cap(cdir):
    """No deletion and no hysteresis run while total <= cap."""
    _mk_group(cdir, 1, age=10 * 3600)
    report = cache.evict(cap=10_000, now=NOW)
    assert report["evicted"] == 0 and report["freed"] == 0
    assert len(_group_files(cdir, _key(1))) == 2


def test_freshness_guard_protects_recent_groups(cdir):
    """Groups touched within the last hour survive even over cap — the
    guard protects lazily-loaded records held by running evaluations
    (DESIGN 4.5)."""
    _mk_group(cdir, 1, age=120)  # fresh
    _mk_group(cdir, 2, age=90)  # fresh
    report = cache.evict(cap=1000, now=NOW)
    assert report["evicted"] == 0
    assert len(list(cdir.iterdir())) == 5  # 2 groups + the dir tag


def test_touch_on_hit_resets_lru_position(cdir):
    """A touched meta moves its whole group to the young end of the LRU
    order: recency is the meta mtime (DESIGN 4.5 — LRU via
    touch-on-hit, no index database)."""
    _mk_group(cdir, 1, age=20 * 3600)
    _mk_group(cdir, 2, age=10 * 3600)
    # "Hit" group 1: touch only its meta, as the evaluator does.
    os.utime(cdir / f"{_key(1)}.meta.json", (NOW, NOW))
    report = cache.evict(cap=1500, now=NOW)  # must evict exactly one group
    assert report["evicted"] == 1
    assert _group_files(cdir, _key(2)) == []  # untouched group went first
    assert len(_group_files(cdir, _key(1))) == 2


def test_meta_deleted_first_within_group(cdir, monkeypatch):
    """Within a group the meta goes first, then npz, then derived — the
    reverse of write-visibility order, so no half-visible record state
    can exist (DESIGN 4.5)."""
    _mk_group(cdir, 1, glb=50, age=20 * 3600)
    deleted: list[str] = []
    real_unlink = os.unlink

    def recording_unlink(path, *a, **kw):
        deleted.append(Path(path).name)
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(os, "unlink", recording_unlink)
    cache.evict(cap=1, now=NOW)
    key = _key(1)
    assert deleted == [f"{key}.meta.json", f"{key}.npz", f"{key}-g3-main.glb"]


def test_orphan_group_without_meta_is_evictable(cdir):
    """A meta-less orphan group (crash between npz and meta writes) still
    scans and evicts, aged by its newest member."""
    key = _key(1)
    p = cdir / f"{key}.npz"
    p.write_bytes(b"x" * 1000)
    os.utime(p, (NOW - 20 * 3600, NOW - 20 * 3600))
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 1
    assert not p.exists()


# -- cap semantics -------------------------------------------------------------


def test_cap_zero_skips_all_checks(cdir, monkeypatch):
    """cap 0 = unlimited and skips all checks: neither evict nor
    maybe_evict may even scan the dir (DESIGN 4.5)."""

    def boom():
        raise AssertionError("scan() must not run when cap is 0")

    monkeypatch.setattr(cache, "scan", boom)
    assert cache.evict(cap=0)["evicted"] == 0
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "0")
    cache.note_write(1234)
    assert cache.maybe_evict() is None


def test_maybe_evict_skips_zero_write_sessions(cdir, monkeypatch):
    """A session that wrote no cache bytes never scans: touch-on-hit
    changes mtimes, not size, so warm-cache and read-only commands pay
    nothing (DESIGN 4.5)."""
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "1000")

    def boom():
        raise AssertionError("scan() must not run for a zero-write session")

    monkeypatch.setattr(cache, "scan", boom)
    assert cache.take_wrote() == 0
    assert cache.maybe_evict() is None


def test_maybe_evict_runs_after_writes_and_resets_counter(cdir, monkeypatch):
    """The process that grew the cache checks after growing; the check
    consumes the session counter so it fires once per growth."""
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "1000")
    _mk_group(cdir, 1, age=20 * 3600)
    cache.note_write(1100)
    report = cache.maybe_evict()
    assert report is not None and report["evicted"] == 1
    assert cache.maybe_evict() is None  # counter consumed


def test_bad_cap_is_actionable(cdir, monkeypatch):
    """An unparseable or negative cap fails with a machine-readable code,
    not a traceback (DESIGN P2)."""
    for bad in ("banana", "-1"):
        monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", bad)
        with pytest.raises(MoraineError) as exc:
            cache.max_bytes()
        assert exc.value.code == "bad_cache_cap"


# -- races ---------------------------------------------------------------------


def test_concurrent_deletion_race_is_benign(cdir, monkeypatch):
    """An evictor losing a file to a concurrent evictor swallows the miss
    and continues (DESIGN 4.5: losers swallow FileNotFoundError)."""
    _mk_group(cdir, 1, age=20 * 3600)
    _mk_group(cdir, 2, age=19 * 3600)
    real_unlink = os.unlink
    stolen = f"{_key(1)}.npz"

    def racing_unlink(path, *a, **kw):
        if Path(path).name == stolen:
            real_unlink(path)  # the other evictor got there first
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(os, "unlink", racing_unlink)
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 2
    assert _group_files(cdir, _key(1)) == []
    assert _group_files(cdir, _key(2)) == []


# -- tmp files -------------------------------------------------------------------


def test_stale_tmp_evicted_fresh_tmp_kept(cdir):
    """In-flight ``.tmp<pid>-<n>`` files older than the freshness window are
    crash leftovers and evict; fresh ones may belong to a live writer and
    stay."""
    _mk_group(cdir, 1, age=20 * 3600)
    stale = cdir / f"{_key(8)}.npz.tmp123"
    stale.write_bytes(b"x" * 10)
    os.utime(stale, (NOW - 2 * 3600, NOW - 2 * 3600))
    fresh = cdir / f"{_key(9)}.npz.tmp456"
    fresh.write_bytes(b"x" * 10)
    cache.evict(cap=1, now=NOW)
    assert not stale.exists()
    assert fresh.exists()


def test_current_and_pre_counter_tmp_forms_both_recognized(cdir):
    """The in-flight name is ``.tmp<pid>-<n>`` (core.atomic); the older
    ``.tmp<pid>`` form still matches the same grammar, so leftovers from
    either era are reaped as stale without a legacy registration."""
    for name in (f"{_key(8)}.npz.tmp123-4", f"{_key(9)}.meta.json.tmp123"):
        key, _kind, is_tmp = cache._classify(name)
        assert is_tmp and key == name[:24]
        p = cdir / name
        p.write_bytes(b"x" * 10)
        os.utime(p, (NOW - 2 * 3600, NOW - 2 * 3600))
    cache.evict(cap=10_000, now=NOW)
    assert [p.name for p in cdir.iterdir()] == [cache.TAG_NAME]


def test_tmp_names_unique_per_call_and_recognized(cdir):
    """Two writers of one path in one process never share a temp file:
    the temp name carries a per-call counter beside the pid, and every
    name it yields is the cache grammar's in-flight form."""
    path = cdir / f"{_key(1)}.npz"
    a, b = atomic.tmp_name(path), atomic.tmp_name(path)
    assert a != b
    for tmp in (a, b):
        assert tmp.parent == cdir
        assert cache._classify(tmp.name) == (_key(1), ".npz", True)


def test_concurrent_same_record_writes_never_lose_a_file(cdir):
    """A loop-side reload and a worker-thread pass may write the same
    content-addressed record at once (the server's concurrency); neither
    writer sees its temp file vanish under the other, and the record
    holds the bytes."""
    from moraine3d.cache.records import write_atomic

    path = cdir / f"{_key(2)}.npz"
    data = b"record" * 1000
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def writer():
        try:
            start.wait()
            for _ in range(200):
                write_atomic(path, data)
        except BaseException as e:  # noqa: BLE001 - the test collects
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert path.read_bytes() == data
    assert [p.name for p in cdir.iterdir() if p.name != cache.TAG_NAME] == [path.name]


def test_tmp_reaped_even_under_cap(cdir):
    """Stale in-flight files are garbage at any cache size: an under-cap
    `cache evict` still reaps them (they must not count unreclaimable
    bytes against the cap forever)."""
    stale = cdir / f"{_key(8)}.npz.tmp123"
    stale.write_bytes(b"x" * 10)
    os.utime(stale, (NOW - 2 * 3600, NOW - 2 * 3600))
    report = cache.evict(cap=10_000, now=NOW)
    assert report["evicted"] == 0 and report["freed"] == 10
    assert not stale.exists()


def test_future_mtime_tmp_is_not_immortal(cdir):
    """The tmp staleness test uses the same symmetric window as groups: a
    future mtime (clock skew) must not make a crash leftover
    unreclaimable."""
    skewed = cdir / f"{_key(8)}.npz.tmp123"
    skewed.write_bytes(b"x" * 10)
    os.utime(skewed, (NOW + 2 * 3600, NOW + 2 * 3600))
    cache.evict(cap=1, now=NOW)
    assert not skewed.exists()


def test_clear_removes_all_recognized_tmp(cdir):
    """``clear`` removes recognized in-flight files regardless of age —
    a full clear leaves no Moraine-owned bytes behind."""
    tmp = cdir / f"{_key(9)}.meta.json.tmp456"
    tmp.write_bytes(b"x" * 10)
    _mk_group(cdir, 1)
    report = cache.clear()
    assert report["cleared"] == 1
    assert not tmp.exists()
    # Only the directory tag survives a full clear.
    assert [p.name for p in cdir.iterdir()] == [cache.TAG_NAME]


# -- deletion safety (user requirement: never delete non-moraine data) -----------

FOREIGN = (
    "README.txt",  # arbitrary user file
    "notes.tmp123",  # bare tmp, no cache-name prefix
    "deadbeef.npz",  # hex but not 24 chars
    _key(1) + "ab.npz",  # 26 hex chars
    f"{_key(2)}.npz.bak",  # unrecognized suffix
    f"{_key(3)}.meta.json.tmpX",  # tmp suffix without pid digits
    f"{_key(3)}.meta.json.tmp123-",  # counter separator without digits
    f"{_key(3)}.meta.json.tmp123-4x",  # trailing junk after the counter
    f"{_key(4)}-gX-main.glb",  # malformed derived name
)


def _seed_foreign(cdir: Path) -> dict[str, bytes]:
    files = {}
    for name in FOREIGN:
        data = f"foreign:{name}".encode()
        (cdir / name).write_bytes(data)
        files[name] = data
    sub = cdir / "subdir"
    sub.mkdir()
    (sub / f"{_key(5)}.npz").write_bytes(b"nested cache-looking file")
    files["subdir/" + f"{_key(5)}.npz"] = b"nested cache-looking file"
    return files


def test_foreign_files_survive_evict_and_clear(cdir):
    """Anything the writer grammar does not recognize survives both a
    maximal eviction and a full clear, byte-identical — Moraine never
    deletes what it cannot attribute to itself (deletion is
    allowlist-only)."""
    foreign = _seed_foreign(cdir)
    _mk_group(cdir, 6, age=20 * 3600)
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 1
    report = cache.clear()
    assert report["cleared"] == 0
    assert report["other"] == len(FOREIGN)
    for rel, data in foreign.items():
        assert (cdir / rel).read_bytes() == data


def test_unlink_choke_point_rejects_out_of_scope_paths(cdir, tmp_path):
    """The tool-wide deletion choke point refuses paths outside the cache
    dir and names outside the writer grammar, deleting nothing."""
    outside = tmp_path / f"{_key(1)}.npz"
    outside.write_bytes(b"x")
    assert cache.unlink_cache_file(outside) == 0
    assert outside.exists()
    inside_bad = cdir / "precious.db"
    inside_bad.write_bytes(b"x")
    assert cache.unlink_cache_file(inside_bad) == 0
    assert inside_bad.exists()
    inside_ok = cdir / f"{_key(1)}.npz"
    inside_ok.write_bytes(b"xy")
    assert cache.unlink_cache_file(inside_ok) == 2
    assert not inside_ok.exists()


def test_every_deletion_matches_the_allowlist(cdir, monkeypatch):
    """Structural guarantee: every path evict/clear unlink, across all
    code paths, is a direct cache-dir child the recognition seam
    attributes to Moraine — current or registered legacy grammar, the
    same predicate the choke point enforces (catches any future
    deletion path added outside the choke point)."""
    monkeypatch.setattr(cache, "_LEGACY_NAME_RES", (_OLD_GLB,))
    _seed_foreign(cdir)
    for i in range(10, 16):
        _mk_group(cdir, i, glb=30, age=20 * 3600)
    pdir = cdir / paths.PREVIEW_DIR_NAME
    pdir.mkdir()
    (pdir / f"{_key(40)}.npz").write_bytes(b"x")
    (pdir / "stray.txt").write_bytes(b"keep")
    (cdir / f"{_key(20)}.npz.tmp1").write_bytes(b"x")
    os.utime(cdir / f"{_key(20)}.npz.tmp1", (NOW - 2 * 3600, NOW - 2 * 3600))
    legacy = cdir / f"{_key(30)}-old.glb"
    legacy.write_bytes(b"x" * 5)
    os.utime(legacy, (NOW - 20 * 3600, NOW - 20 * 3600))
    legacy_tmp = cdir / f"{_key(31)}-old.glb.tmp7"
    legacy_tmp.write_bytes(b"x")
    os.utime(legacy_tmp, (NOW - 2 * 3600, NOW - 2 * 3600))

    deleted: list[Path] = []
    real_unlink = os.unlink

    def recording_unlink(path, *a, **kw):
        deleted.append(Path(path))
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(os, "unlink", recording_unlink)
    cache.evict(cap=1, now=NOW)
    cache.clear()
    assert deleted  # the harness saw the deletions it is auditing
    # The widened (legacy and preview-bucket) halves of the allowlist are
    # exercised, not idle.
    assert not legacy.exists() and not legacy_tmp.exists()
    assert not (pdir / f"{_key(40)}.npz").exists()
    assert (pdir / "stray.txt").exists()
    for p in deleted:
        assert p.parent in (cdir, pdir)
        assert cache._classify(p.name) is not None, p.name


# -- deletion gate: CACHEDIR.TAG + path sanity ------------------------------------


def test_deletion_refuses_untagged_dir_with_foreign_content(cdir):
    """Eviction and clear refuse an untagged directory holding files
    Moraine does not recognize — the firewall for a misconfigured
    MORAINE_CACHE_DIR pointed at a directory that is not purely
    Moraine's cache."""
    (cdir / cache.TAG_NAME).unlink()
    (cdir / "user-notes.txt").write_bytes(b"foreign")
    _mk_group(cdir, 1, age=20 * 3600)
    with pytest.raises(MoraineError) as exc:
        cache.evict(cap=1, now=NOW)
    assert exc.value.code == "cache_untagged"
    with pytest.raises(MoraineError) as exc:
        cache.clear()
    assert exc.value.code == "cache_untagged"
    assert len(_group_files(cdir, _key(1))) == 2  # nothing was deleted


def test_deletion_allowed_on_untagged_pure_moraine_dir(cdir):
    """An untagged dir holding nothing but Moraine's own files passes the
    gate by content provenance: a plain `cache clear` on a fresh or
    empty cache dir must work, not error."""
    (cdir / cache.TAG_NAME).unlink()
    report = cache.clear()  # empty dir: nothing foreign to lose
    assert report["cleared"] == 0
    _mk_group(cdir, 1, age=20 * 3600)
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 1


def _foreign_tag_bytes() -> bytes:
    """A spec-valid tag another tool could have stamped: the standard
    signature (derived from the constant so this test keeps pinning the
    signature-vs-creator-line boundary) with a non-Moraine creator."""
    return paths._TAG_SIGNATURE + b"\n# This file is a cache directory tag created by uv.\n"


def test_deletion_refuses_foreign_apps_tag(cdir):
    """The deletion gate matches Moraine's own tag body (the standard
    signature plus the "created by Moraine" line), not the bare
    signature: a CACHEDIR.TAG stamped by another tool marks the dir as
    that tool's cache — it neither opens Moraine's deletion gate nor
    gets overwritten by the write path — and the refusal names the tag
    itself as the blocker (the actionable-errors rule: "remove the
    foreign files" alone cannot be acted on when the only foreign file
    looks like expected infrastructure)."""
    (cdir / cache.TAG_NAME).write_bytes(_foreign_tag_bytes())
    _mk_group(cdir, 1, age=20 * 3600)
    with pytest.raises(MoraineError) as exc:
        cache.evict(cap=1, now=NOW)
    assert exc.value.code == "cache_untagged"
    assert "written by another tool" in str(exc.value)
    with pytest.raises(MoraineError) as exc:
        cache.clear()
    assert exc.value.code == "cache_untagged"
    assert len(_group_files(cdir, _key(1))) == 2  # nothing was deleted
    # The gate-closing file must be visible in the tool's own reporting:
    # stats counts a non-Moraine tag as "other", never skips it.
    assert cache.stats()["other"] == 1
    cache.note_write(1)
    assert (cdir / cache.TAG_NAME).read_bytes() == _foreign_tag_bytes()


def test_torn_own_tag_self_heals(cdir):
    """A truncated write of Moraine's own tag body (the stamp is
    non-atomic) is 'torn', not foreign: the next cache write re-stamps
    the full body and the deletion gate stays open — a crash mid-stamp
    must never permanently refuse Moraine's own cache."""
    (cdir / cache.TAG_NAME).write_bytes(cache._TAG_BODY[:17])
    _mk_group(cdir, 1, age=20 * 3600)
    report = cache.evict(cap=1, now=NOW)  # content provenance holds
    assert report["evicted"] == 1
    cache.note_write(1)
    assert (cdir / cache.TAG_NAME).read_bytes() == cache._TAG_BODY


def test_unreadable_tag_is_transient_not_foreign(cdir, monkeypatch):
    """A tag-read failure (e.g. a backup/AV agent briefly holding the
    file on Windows) yields no provenance verdict: the write path does
    not memoize the dir as foreign, and the next write re-reads and
    stamps normally."""
    (cdir / cache.TAG_NAME).unlink()
    real = cache._tag_state
    flaky_calls: list[int] = []

    def flaky(d):
        if not flaky_calls:
            flaky_calls.append(1)
            return "unreadable"
        return real(d)

    monkeypatch.setattr(cache, "_tag_state", flaky)
    cache.note_write(1)  # unreadable: no stamp, and crucially no memo
    assert cdir not in cache._judged_dirs
    assert not (cdir / cache.TAG_NAME).exists()
    cache.note_write(1)  # lock released: this write stamps
    assert (cdir / cache.TAG_NAME).read_bytes() == cache._TAG_BODY


def test_transient_scan_failure_is_not_memoized(cdir, monkeypatch):
    """A failed content scan is no evidence: only a positive observation
    of foreign content may enter the judged-dir memo. A transient
    OSError mid-scan (a backup/AV agent holding the directory) must not
    permanently disable stamping for the process — a verdict minted
    from a failed scan would record no misconfiguration, breaking the
    premise the no-revalidation decision rests on."""
    (cdir / cache.TAG_NAME).unlink()
    real_scandir = os.scandir
    failed: list[int] = []

    def flaky(path):
        if not failed:
            failed.append(1)
            raise OSError("transient scan failure")
        return real_scandir(path)

    monkeypatch.setattr(cache.os, "scandir", flaky)
    cache.note_write(1)  # scan fails: no stamp, and crucially no memo
    assert cdir not in cache._judged_dirs
    assert not (cdir / cache.TAG_NAME).exists()
    cache.note_write(1)  # scan works again: this write stamps
    assert (cdir / cache.TAG_NAME).read_bytes() == cache._TAG_BODY


def test_foreign_verdict_is_per_process(cdir):
    """The foreign-dir memo is per-process and never revalidated: after
    the user removes the foreign content, the running process keeps the
    verdict — accepted staleness, because the deletion gate reads the
    directory fresh (content provenance admits the cleaned dir at once)
    and the next process re-judges and stamps, exactly what the
    cache_untagged message promises of "the next evaluating command"."""
    (cdir / cache.TAG_NAME).unlink()
    (cdir / "user-notes.txt").write_bytes(b"foreign")
    cache.note_write(1)
    assert cdir in cache._judged_dirs
    assert not (cdir / cache.TAG_NAME).exists()
    (cdir / "user-notes.txt").unlink()
    cache.note_write(1)  # same process: memo holds, no stamp
    assert not (cdir / cache.TAG_NAME).exists()
    cache.clear()  # the gate never consults the memo: already open
    cache._judged_dirs.clear()  # what a fresh process starts with
    cache.note_write(1)
    assert (cdir / cache.TAG_NAME).read_bytes() == cache._TAG_BODY


def test_stale_ours_memo_loses_no_safety(cdir):
    """The recoverability half of the memo invariant: a tag deleted
    mid-process leaves an untagged pure-Moraine dir that the gate's
    fresh content-provenance read still admits — deletion keeps working
    despite the stale memo (the refusal half is pinned by
    test_gate_ignores_stale_ours_memo)."""
    cache.note_write(1)
    assert cdir in cache._judged_dirs
    (cdir / cache.TAG_NAME).unlink()
    cache.note_write(1)  # memo holds: no re-stamp, by design
    assert not (cdir / cache.TAG_NAME).exists()
    _mk_group(cdir, 1, age=20 * 3600)
    report = cache.evict(cap=1, now=NOW)  # gate: content provenance
    assert report["evicted"] == 1


def test_gate_ignores_stale_ours_memo(cdir):
    """The refusal half of the memo invariant: the deletion gate never
    consults the judged-dir memo — it reads the directory fresh on
    every call — so a dir judged "ours" that another tool has since
    claimed is still refused; a stale verdict can never make foreign
    data deletable."""
    cache.note_write(1)  # this process judges the dir ours
    assert cdir in cache._judged_dirs
    # Another tool claims the dir: its tag replaces Moraine's.
    (cdir / cache.TAG_NAME).write_bytes(_foreign_tag_bytes())
    _mk_group(cdir, 1, age=20 * 3600)
    with pytest.raises(MoraineError) as exc:
        cache.evict(cap=1, now=NOW)
    assert exc.value.code == "cache_untagged"
    with pytest.raises(MoraineError) as exc:
        cache.clear()
    assert exc.value.code == "cache_untagged"
    assert len(_group_files(cdir, _key(1))) == 2  # nothing was deleted


def test_tag_prefix_bytes_are_frozen():
    """_TAG_MORAINE_PREFIX is a frozen on-disk format: the deletion gate
    compares it byte-exact against tags already stamped on user
    machines, so rewording it would orphan every existing cache (gate
    refused, never re-stamped). Any change must ship a migration that
    still accepts these bytes — this pin makes that decision loud."""
    assert paths._TAG_MORAINE_PREFIX == (
        b"Signature: 8a477f597d28d172789f06886806bc55\n"
        b"# This file is a cache directory tag created by Moraine.\n"
    )


def test_deletion_refuses_cwd_and_home(cdir, monkeypatch):
    """Even a tagged directory is refused when it is the current working
    directory or the home directory — deletion never runs where a path
    mixup would be catastrophic."""
    _mk_group(cdir, 1, age=20 * 3600)
    monkeypatch.chdir(cdir)
    with pytest.raises(MoraineError) as exc:
        cache.clear()
    assert exc.value.code == "cache_unsafe_dir"

    monkeypatch.chdir(cdir.parent)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: cdir))
    with pytest.raises(MoraineError) as exc:
        cache.evict(cap=1, now=NOW)
    assert exc.value.code == "cache_unsafe_dir"
    assert len(_group_files(cdir, _key(1))) == 2  # nothing was deleted


def test_write_path_stamps_the_tag(cdir):
    """The write path stamps the standard CACHEDIR.TAG signature: a
    directory moraine caches into becomes deletable (and tag-honoring
    backup tools skip it)."""
    (cdir / cache.TAG_NAME).unlink()
    cache.note_write(1)
    tag = (cdir / cache.TAG_NAME).read_bytes()
    assert tag.startswith(b"Signature: 8a477f597d28d172789f06886806bc55")


def test_tag_follows_cache_dir_changes(cdir, tmp_path, monkeypatch):
    """The tag memo is per-directory, not a process flag: a mid-process
    MORAINE_CACHE_DIR change re-stamps the new dir on its first write,
    so deletion never gets stuck refusing a dir the process wrote to."""
    cache.note_write(1)  # memoizes cdir as tagged
    other = tmp_path / "other-cache"
    other.mkdir()
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(other))
    cache.note_write(1)
    assert (other / cache.TAG_NAME).exists()


def test_allowlist_rejects_trailing_newline_names():
    """The name grammar anchors with \\Z: '$' would also match before a
    trailing newline, admitting a foreign POSIX filename like
    '<key>.npz\\n' into the deletion allowlist."""
    good = "0" * 24 + ".npz"
    assert cache._NAME_RE.match(good)
    assert cache._NAME_RE.match(good + "\n") is None
    assert cache._NAME_RE.match("0" * 24 + ".meta.json.tmp1\n") is None


def test_future_mtimes_do_not_block_eviction(cdir):
    """The freshness window is symmetric: a far-future mtime (clock skew)
    is not 'fresh forever' — the cap stays enforceable — while an mtime
    slightly ahead of now (timestamp granularity) is still protected."""
    _mk_group(cdir, 1, age=-2 * 20 * 3600)  # 40h in the future: skew
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 1
    _mk_group(cdir, 2, age=-10)  # 10s ahead: granularity, protected
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 0


def test_evict_recheck_protects_groups_touched_after_scan(cdir, monkeypatch):
    """A group hit (meta-touched) after the scan captured its mtime is
    skipped at delete time: the freshness guard protects records held
    lazily by a running evaluation, and the scan snapshot alone cannot
    see a touch that happens mid-eviction."""
    _mk_group(cdir, 1, age=0)  # on-disk meta is fresh
    s = cache.scan()
    for g in s.groups.values():
        g.mtime = NOW - 20 * 3600  # but the scan snapshot says stale
    monkeypatch.setattr(cache, "scan", lambda: s)
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 0
    assert len(_group_files(cdir, _key(1))) == 2


def test_tag_never_stamped_into_preexisting_foreign_dir(tmp_path, monkeypatch):
    """Stamping requires provenance: a pre-existing directory holding
    foreign files never acquires CACHEDIR.TAG (backup tools honor the
    tag as do-not-back-up, so a misconfigured MORAINE_CACHE_DIR must
    not amplify into excluding user data from backups) — and deletion
    stays refused there."""
    foreign = tmp_path / "documents"
    foreign.mkdir()
    (foreign / "notes.txt").write_bytes(b"user data")
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(foreign))
    cache.note_write(1)
    assert not (foreign / cache.TAG_NAME).exists()
    with pytest.raises(MoraineError) as exc:
        cache.clear()
    assert exc.value.code == "cache_untagged"
    assert (foreign / "notes.txt").read_bytes() == b"user data"


def test_tag_stamped_when_moraine_created_the_dir(tmp_path, monkeypatch):
    """A directory Moraine itself created is stampable even after foreign
    files appear in it later — creation is the provenance proof."""
    d = tmp_path / "fresh-cache"
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(d))
    from moraine3d.core.paths import cache_dir

    cache_dir()  # Moraine creates the directory
    (d / "stray.txt").write_bytes(b"x")
    cache.note_write(1)
    assert (d / cache.TAG_NAME).exists()


def test_tag_restamped_after_dir_recreation(cdir, monkeypatch):
    """An externally deleted-and-recreated cache dir heals: creation-time
    stamping in paths.cache_dir re-tags it at mkdir — the sole
    intra-process mechanism once the judged-dir memo holds (the write
    path's re-stamp is memo-gated) — so a long-lived serve never gets
    stuck with cache_untagged on a dir Moraine itself just recreated."""
    import shutil

    cache.note_write(1)
    shutil.rmtree(cdir)  # external cleanup tool wipes the cache dir
    cache.note_write(1)  # next write: cache_dir() recreates and stamps
    assert (cdir / cache.TAG_NAME).exists()
    report = cache.evict(cap=1000, now=NOW)  # gate passes again
    assert report["evicted"] == 0


def test_locked_corrupt_record_error_is_honest(cdir, monkeypatch):
    """When a corrupt record cannot be removed (file locked — the choke
    point swallows the failure), the cache_corrupt error must say so and
    give the manual fix, not promise a removal that did not happen."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation
    from moraine3d.errors import MoraineError as ME

    Evaluation(asset_mod.loads(DIAMOND)).records()
    npzs = sorted(cdir.glob("*.npz"))
    for p in npzs:
        p.write_bytes(b"not a zipfile")
    monkeypatch.setattr(cache, "unlink_cache_file", lambda p: 0)  # lock simulation
    ev = Evaluation(asset_mod.loads(DIAMOND))
    recs = ev.records()  # hits: metas are intact
    with pytest.raises(ME) as exc:
        for rec in recs.values():
            rec.value()
    assert exc.value.code == "cache_corrupt"
    assert "could not be removed" in exc.value.message
    assert all(p.exists() for p in npzs)  # and indeed nothing was removed


def test_tag_is_never_cache_content(cdir):
    """The tag is infrastructure: excluded from stats accounting and
    untouched by clear."""
    _mk_group(cdir, 1)
    s = cache.stats()
    assert s["files"] == 2 and "other" not in s
    cache.clear()
    assert (cdir / cache.TAG_NAME).exists()


# -- instrumentation (DESIGN 6, bundled with touch-on-hit) -----------

DIAMOND = """{
"build_version": "fixture",
"nodes": [
{"id": "bx01", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "tr01", "op": "transform", "inputs": ["bx01"], "params": {"translate": [2.0, 0.0, 0.0]}},
{"id": "tr02", "op": "transform", "inputs": ["bx01"], "params": {"translate": [-2.0, 0.0, 0.0]}},
{"id": "un01", "op": "boolean", "inputs": ["tr01", "tr02"], "params": {"mode": "union"}}
],
"outputs": {"main": "un01"}
}
"""


def test_touch_on_hit_bumps_meta_and_leaves_npz(cdir):
    """A cache hit utimes the record's .meta.json and nothing else: the
    meta mtime is the LRU recency signal (DESIGN 4.5), while npz mtimes
    stay dirty-subgraph evidence (test_graph pins that side)."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation

    Evaluation(asset_mod.loads(DIAMOND)).records()
    before = {p.name: p.stat().st_mtime_ns for p in cdir.iterdir()}
    time.sleep(0.01)  # ensure mtime resolution can observe the touch
    Evaluation(asset_mod.loads(DIAMOND)).records()
    after = {p.name: p.stat().st_mtime_ns for p in cdir.iterdir()}
    assert set(after) == set(before)
    for name in before:
        if name.endswith(".meta.json"):
            assert after[name] > before[name], name
        else:
            assert after[name] == before[name], name


def test_counters_match_hand_trace(cdir):
    """hits + misses equals the unique record keys looked up on disk:
    cold diamond = 4 misses (memo dedups the shared box, not the disk
    count), warm rerun = 4 hits, downstream edit = 3 hits + 1 miss."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation

    ev1 = Evaluation(asset_mod.loads(DIAMOND))
    ev1.records()
    p1 = ev1.profile()
    assert (p1["hits"], p1["misses"]) == (0, 4)
    assert {e["node"] for e in p1["slowest"]} <= {"bx01", "tr01", "tr02", "un01"}

    ev2 = Evaluation(asset_mod.loads(DIAMOND))
    ev2.records()
    p2 = ev2.profile()
    assert (p2["hits"], p2["misses"]) == (4, 0)
    assert "slowest" not in p2  # all-hit pass: nothing to optimize
    assert p2["ms"] >= 0

    import json as json_mod

    doc = json_mod.loads(DIAMOND)
    doc["nodes"][3]["params"]["mode"] = "difference"
    ev3 = Evaluation(asset_mod.loads(json_mod.dumps(doc)))
    ev3.records()
    p3 = ev3.profile()
    assert (p3["hits"], p3["misses"]) == (3, 1)
    assert p3["slowest"][0]["node"] == "un01"


def test_asset_ref_alias_contributes_no_count(cdir, tmp_path):
    """An asset_ref record is an alias of the referenced output's record
    (DESIGN 4.6): it performs no disk lookup of its own, so it adds
    nothing to hits or misses."""
    import json as json_mod

    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation

    proj = tmp_path / "proj"
    (proj / "subassets").mkdir(parents=True)
    part = {
        "build_version": "fixture",
        "nodes": [{"id": "bx01", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}}],
        "outputs": {"part": "bx01"},
    }
    root = {
        "build_version": "fixture",
        "nodes": [{"id": "rf01", "op": "asset_ref",
                   "params": {"asset": "subassets/part.mora", "output": "part"}}],
        "outputs": {"main": "rf01"},
    }
    (proj / "subassets" / "part.mora").write_text(json_mod.dumps(part), encoding="utf-8")
    (proj / "root.mora").write_text(json_mod.dumps(root), encoding="utf-8")

    ev = Evaluation(asset_mod.load(proj / "root.mora"))
    ev.records()
    p = ev.profile()
    assert (p["hits"], p["misses"]) == (0, 1)  # the box only; rf01 is free


def test_wrong_shape_meta_self_heals(cdir):
    """A meta that is valid JSON but not the record shape (null, [], {})
    is corruption like any other: the pair is dropped and the node
    re-evaluated — never an uncaught KeyError/TypeError (the recovery
    guard covers the subscripts, not just the parse)."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation

    Evaluation(asset_mod.loads(DIAMOND)).records()
    metas = sorted(cdir.glob("*.meta.json"))
    metas[0].write_text("{}", encoding="utf-8")
    metas[1].write_text("null", encoding="utf-8")
    metas[2].write_text("[1,2]", encoding="utf-8")
    ev = Evaluation(asset_mod.loads(DIAMOND))
    ev.records()  # must not raise
    assert ev.profile()["misses"] >= 3  # the three dropped records re-derived

    # The same claim one level down: correct keys with the wrong internal
    # types (a non-str result_hash, a non-dict stats) is corruption too —
    # a non-dict stats accepted as a hit would break the stats rollup
    # downstream instead of self-healing here.
    metas = sorted(cdir.glob("*.meta.json"))
    metas[0].write_text('{"result_hash": "x", "stats": []}', encoding="utf-8")
    metas[1].write_text('{"result_hash": 123, "stats": {}}', encoding="utf-8")
    ev = Evaluation(asset_mod.loads(DIAMOND))
    ev.records()  # must not raise
    assert ev.profile()["misses"] >= 2  # both dropped records re-derived


def test_wrong_typed_nested_meta_self_heals(cdir):
    """The hit path validates the record shape the tool consumes, not
    just top-level types: a stats dict whose consumed fields carry the
    wrong types (missing/non-int verts, malformed bounds, non-mapping
    groups) is corruption — dropped and re-derived, never served as a
    hit that would traceback later in the stats rollup or the
    attribute-drop warnings and leave a poisoned record behind."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation

    Evaluation(asset_mod.loads(DIAMOND)).records()
    metas = sorted(cdir.glob("*.meta.json"))
    metas[0].write_text('{"result_hash": "x", "stats": {"groups": 3}}', encoding="utf-8")
    metas[1].write_text(
        '{"result_hash": "x", "stats": {"verts": "a", "tris": 1,'
        ' "bounds": {"min": [0, 0, 0], "max": [0, 0, 0]}}}',
        encoding="utf-8",
    )
    metas[2].write_text(
        '{"result_hash": "x", "stats": {"verts": 1, "tris": 1,'
        ' "bounds": {"min": [0, 0], "max": [0, 0, 0]}}}',
        encoding="utf-8",
    )
    ev = Evaluation(asset_mod.loads(DIAMOND))
    ev.records()  # must not raise
    assert ev.profile()["misses"] >= 3  # all three dropped and re-derived


def test_evaluation_notes_written_bytes(cdir):
    """Cache writes (records and derived GLBs) feed the session counter
    that arms the post-growth eviction check; a warm rerun of records()
    adds nothing (touch-on-hit changes mtimes, not size)."""
    from moraine3d.core import asset as asset_mod
    from support.engine import Evaluation

    cache.take_wrote()
    Evaluation(asset_mod.loads(DIAMOND)).results()
    assert cache.take_wrote() > 0
    Evaluation(asset_mod.loads(DIAMOND)).records()
    assert cache.take_wrote() == 0


# -- stats -----------------------------------------------------------------------


def test_stats_reports_owned_vs_other(cdir):
    """``cache stats`` counts Moraine-owned files/bytes and reports
    unrecognized names separately (DESIGN §4.5; terse JSON vocabulary:
    files / bytes / keys)."""
    _mk_group(cdir, 1, npz=1000, meta=100)
    _mk_group(cdir, 2, npz=500, meta=50, glb=25)
    (cdir / "foreign.bin").write_bytes(b"x" * 7)
    s = cache.stats()
    assert s["keys"] == 2
    assert s["files"] == 5
    assert s["bytes"] == 1675
    assert s["other"] == 1
    assert s["cap"] == 0  # conftest pins MORAINE_CACHE_MAX_BYTES=0


# -- legacy grammar migration (frozen on-disk format) -------------------------


def test_legacy_named_files_stay_moraine_content(cdir, monkeypatch):
    """A filename-grammar change ships its outgoing pattern in
    _LEGACY_NAME_RES, in-flight .tmp variants included: legacy names
    count as Moraine content for provenance — the tag still gets
    stamped and the deletion gate still opens — final forms join their
    key's record group, tmp leftovers join the stale-tmp reap, and
    clear/evict reclaim them instead of leaving the dir permanently
    foreign (DESIGN 4.5)."""
    monkeypatch.setattr(cache, "_LEGACY_NAME_RES", (_OLD_GLB,))
    (cdir / cache.TAG_NAME).unlink()
    legacy = cdir / f"{_key(1)}-old.glb"
    legacy.write_bytes(b"x" * 10)
    os.utime(legacy, (NOW - 20 * 3600, NOW - 20 * 3600))
    stale_tmp = cdir / f"{_key(2)}-old.glb.tmp99"
    stale_tmp.write_bytes(b"x")
    os.utime(stale_tmp, (NOW - 2 * 3600, NOW - 2 * 3600))
    _mk_group(cdir, 1, age=20 * 3600)
    assert cache._content_provenance(cdir) == "pure"
    cache.note_write(1, cdir)  # the write path stamps despite legacy names
    assert (cdir / cache.TAG_NAME).read_bytes() == cache._TAG_BODY
    s = cache.scan()
    assert s.other_files == 0
    assert legacy.name in [name for name, _size, _kind in s.groups[_key(1)].files]
    assert [t[0] for t in s.tmp] == [stale_tmp.name]
    report = cache.clear()
    assert report["cleared"] == 1
    assert not legacy.exists() and not stale_tmp.exists()


def test_choke_point_accepts_registered_legacy_names(cdir, monkeypatch):
    """unlink_cache_file admits a registered legacy grammar (tmp forms
    included) — without this, clear could not reclaim the very files
    the legacy list exists to attribute to Moraine — while names
    outside both grammars stay refused."""
    monkeypatch.setattr(cache, "_LEGACY_NAME_RES", (_OLD_GLB,))
    p = cdir / f"{_key(2)}-old.glb"
    p.write_bytes(b"xy")
    assert cache.unlink_cache_file(p) == 2
    assert not p.exists()
    tmp = cdir / f"{_key(2)}-old.glb.tmp5"
    tmp.write_bytes(b"x")
    assert cache.unlink_cache_file(tmp) == 1
    assert not tmp.exists()
    bad = cdir / f"{_key(2)}-older.glb"
    bad.write_bytes(b"x")
    assert cache.unlink_cache_file(bad) == 0
    assert bad.exists()


def test_unregistered_old_grammar_is_foreign(cdir):
    """The flip side pin: with no legacy entry, an old-grammar name (here
    the pre-generation-prefix GLB naming) is foreign and closes the
    gate — the deliberate non-grandfather of the pre-release g<N> rename
    (DESIGN 4.5)."""
    (cdir / cache.TAG_NAME).unlink()
    (cdir / f"{_key(1)}-main.glb").write_bytes(b"x")  # no g<N>- prefix
    assert cache._content_provenance(cdir) == "foreign"
    with pytest.raises(MoraineError) as exc:
        cache.clear()
    assert exc.value.code == "cache_untagged"


def test_filename_grammar_is_frozen():
    """The cache filename grammar is FROZEN on-disk format like the tag
    body (DESIGN 4.5): it is matched against files already on user
    machines. If this pin fails you are changing the grammar — register
    the outgoing pattern (with its .tmp<pid>-<n> variants) in
    cache._LEGACY_NAME_RES before repinning, or every pre-change cache
    turns permanently foreign. The one change so far widened the
    in-flight suffix from ``.tmp<pid>`` to ``.tmp<pid>-<n>`` with the
    old form still matching, so it registered nothing."""
    assert cache._NAME_RE.pattern == (
        r"^([0-9a-f]{24})"
        r"(\.npz|\.meta\.json"
        r"|-g\d+-[A-Za-z][A-Za-z0-9_-]*\.glb"
        r"|-i\d+-[A-Za-z][A-Za-z0-9_-]*\.png)"
        r"(\.tmp\d+(?:-\d+)?)?\Z"
    )


def test_legacy_pattern_must_capture_the_key(monkeypatch):
    """The registration-time check fails loudly on a pattern that cannot
    yield the 24-hex key as group 1, instead of an IndexError surfacing
    deep in the deletion machinery on the first legacy-named file."""
    monkeypatch.setattr(
        cache, "_LEGACY_NAME_RES", (re.compile(r"[0-9a-f]{24}-old\.glb\Z"),)
    )
    with pytest.raises(AssertionError):
        cache._check_legacy_patterns()


def test_legacy_matching_is_fullmatch_anchored(monkeypatch):
    """Legacy recognition is fullmatch, so a registered pattern missing
    its \\Z cannot widen the deletion allowlist to prefix matches."""
    monkeypatch.setattr(
        cache, "_LEGACY_NAME_RES", (re.compile(r"^([0-9a-f]{24})-old\.glb"),)
    )
    assert cache._classify(f"{_key(1)}-old.glb") is not None
    assert cache._classify(f"{_key(1)}-old.glb.bak") is None
    assert cache._classify(f"{_key(1)}-old.glb\n") is None


def test_clear_reports_locked_groups_honestly(cdir, monkeypatch):
    """``clear`` counts a group as cleared only when every file of it is
    gone; a group with a file another process holds open (the choke
    point skips the lock) is reported under ``locked``, so the report
    never claims a group removed that ``stats`` still lists."""
    k1 = _mk_group(cdir, 1)
    _mk_group(cdir, 2)
    real_unlink = os.unlink

    def lock_npz(path, *a, **kw):
        if Path(path).name == f"{k1}.npz":
            raise PermissionError(13, "held open")
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(os, "unlink", lock_npz)
    report = cache.clear()
    assert report["cleared"] == 1 and report["locked"] == 1
    assert (cdir / f"{k1}.npz").exists()
    assert cache.stats()["keys"] == 1


# -- ephemeral preview bucket -------------------------------------------------


def _pdir(cdir: Path) -> Path:
    d = cdir / paths.PREVIEW_DIR_NAME
    d.mkdir(exist_ok=True)
    return d


def test_choke_point_accepts_preview_bucket_children(cdir):
    """The deletion choke point accepts direct children of the preview
    bucket under the same name grammar (DESIGN 4.5: the bucket purge is
    choke-point traffic, not a deletion site of its own) — and nothing
    deeper, nothing off-grammar, and no directory merely named like the
    bucket elsewhere."""
    pdir = _pdir(cdir)
    ok = pdir / f"{_key(1)}.npz"
    ok.write_bytes(b"xy")
    assert cache.unlink_cache_file(ok) == 2
    assert not ok.exists()
    bad_name = pdir / "precious.db"
    bad_name.write_bytes(b"x")
    assert cache.unlink_cache_file(bad_name) == 0
    assert bad_name.exists()
    nested = pdir / "sub"
    nested.mkdir()
    deep = nested / f"{_key(2)}.npz"
    deep.write_bytes(b"x")
    assert cache.unlink_cache_file(deep) == 0
    assert deep.exists()
    elsewhere = cdir / "previews"
    elsewhere.mkdir()
    off = elsewhere / f"{_key(3)}.npz"
    off.write_bytes(b"x")
    assert cache.unlink_cache_file(off) == 0
    assert off.exists()


def test_choke_point_refuses_symlinked_preview_dir(cdir, tmp_path):
    """A symlink planted under the bucket's name must not redirect the
    choke point into an arbitrary tree (DESIGN 4.5: the bucket is the
    actual directory, never a symlink named after it)."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    victim = target / f"{_key(1)}.npz"
    victim.write_bytes(b"x")
    try:
        os.symlink(target, cdir / paths.PREVIEW_DIR_NAME, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable (Windows without privilege)")
    assert cache.unlink_cache_file(cdir / paths.PREVIEW_DIR_NAME / victim.name) == 0
    assert victim.exists()
    # Provenance agrees: the symlink is foreign content, not the bucket.
    assert cache._content_provenance(cdir) == "foreign"
    _assert_preview_write_refused(target, victim)


def _assert_preview_write_refused(target: Path, victim: Path) -> None:
    """The write side of the same identity rule (DESIGN 4.5: the bucket
    is the actual directory, never a link named after it): the first
    preview write, which creates the bucket, refuses the link with the
    fix named, and nothing lands in the link's target."""
    import numpy as np

    from moraine3d.cache import records
    from moraine3d.core.mesh import Mesh

    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    with pytest.raises(MoraineError) as e:
        records.store(_key(7), mesh, preview=True)
    assert e.value.code == "cache_unwritable"
    assert "remove the link" in e.value.message
    assert sorted(p.name for p in target.iterdir()) == [victim.name]


def test_choke_point_refuses_junction_preview_dir(cdir, tmp_path):
    """An NTFS junction planted under the bucket's name must not carry
    the purge into its target tree (DESIGN 4.5: the bucket is the actual
    directory, never a symlink or junction named after it) — junctions
    are the redirection is_symlink misses and need no privilege to
    create, so every bucket-identity probe checks for both."""
    try:
        import _winapi
    except ImportError:
        pytest.skip("junctions are Windows-only")
    target = tmp_path / "elsewhere"
    target.mkdir()
    victim = target / f"{_key(1)}.npz"
    victim.write_bytes(b"x")
    _winapi.CreateJunction(str(target), str(cdir / paths.PREVIEW_DIR_NAME))
    assert cache.unlink_cache_file(cdir / paths.PREVIEW_DIR_NAME / victim.name) == 0
    assert victim.exists()
    # The scan refuses to look through the junction, so the purge and
    # the stats section see an empty bucket, not the target's files.
    assert cache.scan_preview().names == []
    report = cache.purge_preview()
    assert report["freed"] == 0
    assert victim.exists()
    # Provenance agrees: the junction is foreign content, not the bucket.
    assert cache._content_provenance(cdir) == "foreign"
    _assert_preview_write_refused(target, victim)


def test_preview_lookup_does_not_create_bucket(cdir):
    """DESIGN 4.5: the bucket appears on the first preview write — a
    preview-side record lookup (the read-through fallback of a preview
    pass) must not create the directory as a side effect."""
    from moraine3d.cache import records

    assert records.lookup("0" * 24, preview=True) is None
    assert not (cdir / paths.PREVIEW_DIR_NAME).exists()


def test_purge_preview_removes_only_recognized_files(cdir):
    """``purge_preview`` empties the bucket of grammar-matching files —
    in-flight tmp forms included, no freshness guard: a single-session
    store holds only garbage at purge time — while unrecognized names
    and deeper entries survive and the empty directory stays for the
    next session (DESIGN 4.5)."""
    pdir = _pdir(cdir)
    (pdir / f"{_key(1)}.npz").write_bytes(b"x" * 10)
    (pdir / f"{_key(1)}.meta.json").write_bytes(b"x" * 5)
    (pdir / f"{_key(2)}.npz.tmp7").write_bytes(b"x")  # fresh tmp: still purged
    foreign = pdir / "keep.txt"
    foreign.write_bytes(b"keep")
    (pdir / "nested").mkdir()
    report = cache.purge_preview()
    assert report == {"freed": 16, "locked": 0, "other": 2}
    assert foreign.read_bytes() == b"keep"
    assert (pdir / "nested").is_dir()
    assert pdir.is_dir()
    assert sorted(p.name for p in pdir.iterdir()) == ["keep.txt", "nested"]


def test_purge_preview_without_bucket_is_noop(cdir):
    """A missing bucket purges as empty and is not created by the purge —
    the directory appears only on the first preview write (DESIGN 4.5)."""
    report = cache.purge_preview()
    assert report == {"freed": 0, "locked": 0}
    assert not (cdir / paths.PREVIEW_DIR_NAME).exists()


def test_purge_preview_respects_deletion_gate(cdir):
    """The bucket purge is a bulk deletion and sits behind the same
    provenance gate as evict/clear (DESIGN 4.5): a directory that cannot
    prove itself Moraine's cache is refused wholesale."""
    (cdir / cache.TAG_NAME).unlink()
    (cdir / "user-notes.txt").write_bytes(b"foreign")
    p = _pdir(cdir) / f"{_key(1)}.npz"
    p.write_bytes(b"x")
    with pytest.raises(MoraineError) as exc:
        cache.purge_preview()
    assert exc.value.code == "cache_untagged"
    assert p.exists()


def test_preview_bucket_keeps_untagged_dir_pure(cdir):
    """Content provenance counts the preview bucket as Moraine's own
    infrastructure (DESIGN 4.5): an untagged cache dir holding records
    plus the bucket still judges pure — the write path stamps the tag
    and the deletion gate opens — instead of turning permanently
    foreign."""
    (cdir / cache.TAG_NAME).unlink()
    _mk_group(cdir, 1, age=20 * 3600)
    (_pdir(cdir) / f"{_key(2)}.npz").write_bytes(b"x")
    assert cache._content_provenance(cdir) == "pure"
    cache.note_write(1, cdir)
    assert (cdir / cache.TAG_NAME).read_bytes() == cache._TAG_BODY
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 1


def test_clear_purges_preview_bucket(cdir):
    """``cache clear`` empties the bucket too, attributed separately in
    its report — a full clear that left the bucket behind would be
    dishonest (DESIGN 4.5) — and the persistent numbers stay about the
    persistent store alone."""
    _mk_group(cdir, 1, npz=1000, meta=100)
    pdir = _pdir(cdir)
    (pdir / f"{_key(2)}.npz").write_bytes(b"x" * 30)
    (pdir / f"{_key(2)}.meta.json").write_bytes(b"x" * 10)
    report = cache.clear()
    assert report["cleared"] == 1 and report["freed"] == 1100
    assert report["preview_freed"] == 40
    assert "preview_locked" not in report and "other" not in report
    assert list(pdir.iterdir()) == []
    report = cache.clear()  # empty bucket: no preview attribution at all
    assert "preview_freed" not in report


def test_clear_reports_bucket_survivors(cdir, monkeypatch):
    """DESIGN 4.5: clear attributes bucket survivors like the persistent
    side's — a file outliving the purge under an open-file lock counts
    as preview_locked, and unrecognized bucket names join other — a
    clean-sweep claim over a bucket still holding bytes would be the
    same dishonest reporting one level down."""
    pdir = _pdir(cdir)
    held = pdir / f"{_key(1)}.npz"
    held.write_bytes(b"x" * 30)
    (pdir / f"{_key(2)}.npz").write_bytes(b"x" * 10)
    (pdir / "stray.txt").write_bytes(b"x")
    real_unlink = os.unlink

    def locked_unlink(p, *a, **k):
        # A viewer tab still streaming the GLB: Windows refuses the
        # unlink; the choke point swallows it as benign housekeeping.
        if Path(p) == held:
            raise PermissionError(13, "held open", str(p))
        return real_unlink(p, *a, **k)

    monkeypatch.setattr(cache.os, "unlink", locked_unlink)
    report = cache.clear()
    assert report["preview_freed"] == 10
    assert report["preview_locked"] == 1
    assert report["other"] == 1
    assert held.exists()


def test_stats_reports_preview_section(cdir):
    """``cache stats`` reports the bucket as its own section — its bytes
    sit outside the cap, so folding them into the persistent numbers
    would misstate how full the capped store is (DESIGN 4.5) — and omits
    the section entirely while no bucket content exists."""
    _mk_group(cdir, 1, npz=1000, meta=100)
    assert "preview" not in cache.stats()
    pdir = _pdir(cdir)
    (pdir / f"{_key(2)}.npz").write_bytes(b"x" * 30)
    (pdir / f"{_key(2)}.meta.json").write_bytes(b"x" * 10)
    (pdir / "stray.txt").write_bytes(b"x")
    s = cache.stats()
    assert s["preview"] == {"files": 2, "bytes": 40, "other": 1}
    assert s["bytes"] == 1100  # persistent numbers exclude the bucket


def test_evict_never_touches_preview_bucket(cdir):
    """Eviction is structurally blind to the bucket (DESIGN 4.5: the scan
    ignores directories, and no cap applies inside) — bucket files
    survive any over-cap pass regardless of age."""
    _mk_group(cdir, 1, npz=1000, meta=100, age=20 * 3600)
    old = _pdir(cdir) / f"{_key(2)}.npz"
    old.write_bytes(b"x" * 500)
    os.utime(old, (NOW - 40 * 3600, NOW - 40 * 3600))
    report = cache.evict(cap=1, now=NOW)
    assert report["evicted"] == 1
    assert old.exists()


def test_only_the_choke_point_deletes_files():
    """DESIGN 4.5 made structural: every deletion site under
    src/moraine3d is pinned here — ``cache.store.unlink_cache_file``
    (the cache choke point), ``engine.export._write_out`` removing its
    own in-flight ``.tmp<pid>-<n>`` file next to an export target (Moraine's
    own file outside the cache, which the cache choke point would
    refuse), and ``app.session._retire`` dropping expired session-log
    day files behind its own name + first-line-provenance gate (§6 —
    log files are not cache data, so the cache choke point cannot own
    them), and ``core.asset.write_text`` removing its own
    in-flight ``.tmp<pid>-<n>`` beside the IR it writes (the build,
    §4.8 — Moraine's own file outside the cache, same as the export
    case), and ``recipe.clean.unlink_stale_ir`` removing from a
    project's own ``build/`` the IR no root of the workspace reaches,
    behind its name + first-bytes gate (§4.8), and
    ``core.accepted.write`` removing its own in-flight temp beside the
    ``params.json`` it writes (§7.2). The ephemeral preview bucket
    added no site: its purge is
    choke-point traffic (DESIGN 4.5). Any other unlink / remove /
    rmtree / rmdir under the package (vendored code excluded) fails
    this test."""
    import re

    pattern = re.compile(r"os\.unlink\(|os\.remove\(|\.unlink\(|shutil\.rmtree\(|os\.rmdir\(")
    allowed = {
        ("cache/store.py", "unlink_cache_file"): 1,
        ("engine/export.py", "_write_out"): 1,
        ("app/session.py", "_retire"): 1,
        ("core/asset.py", "write_text"): 1,
        ("recipe/clean.py", "unlink_stale_ir"): 1,
        ("core/accepted.py", "write"): 1,
    }
    found: dict[tuple[str, str], int] = {}
    for py in package_sources():
        func = "<module>"
        for line in py.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*def (\w+)\(", line)
            if m:
                func = m.group(1)
            if pattern.search(line):
                key = (py.relative_to(SRC_ROOT).as_posix(), func)
                found[key] = found.get(key, 0) + 1
    assert found == allowed


# -- location and key composition (DESIGN 4.2 / 4.5 / 4.4) --------------------


def test_default_cache_dir_is_the_os_cache_directory(monkeypatch):
    """DESIGN 4.2: the cache is not part of the project — without
    ``MORAINE_CACHE_DIR`` it lives in the OS cache directory
    (``%LOCALAPPDATA%\\moraine\\Cache``, ``~/Library/Caches/moraine``),
    machine-level. The accessor creates the directory it resolves, so
    the filesystem step is stubbed to report it present: the test pins
    the resolved location, never creates it on the machine."""
    import sys

    import platformdirs

    monkeypatch.delenv("MORAINE_CACHE_DIR", raising=False)

    def _exists(self, *args, **kwargs):
        raise FileExistsError(str(self))

    monkeypatch.setattr(Path, "mkdir", _exists)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    d = paths.cache_dir()
    assert d == platformdirs.user_cache_path("moraine", appauthor=False)
    assert d.is_absolute() and "moraine" in d.parts
    if sys.platform == "win32":
        assert d.parts[-2:] == ("moraine", "Cache")
        assert d.parent.parent == Path(os.environ["LOCALAPPDATA"])
    elif sys.platform == "darwin":
        assert d == Path.home() / "Library" / "Caches" / "moraine"


def test_cache_key_is_truncated_sha256_of_the_canonical_payload():
    """DESIGN 4.5: ``key = hash(op_name, op_impl_version, params,
    upstream_result_hashes)`` — sha256 over the canonical JSON of that
    payload (sorted keys, compact separators), truncated to 24 hex
    chars, carrying the evaluator-level carry version, the node id when
    the op declares ``USES_NODE_ID`` and the declared kernel versions;
    a pure-numpy op's key carries neither field."""
    import hashlib
    import json

    from moraine3d.core import mesh, rand
    from moraine3d.engine import evaluator

    def canonical(payload: dict) -> str:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    params = {"count": 4, "seed": 7, "size": [1.0, 2.0, 3.0]}
    upstream = ["a" * 24, "b" * 24]
    full = evaluator._cache_key(
        "scatter", 3, params, upstream, "sc01", ("core.rand", "core.mesh"),
    )
    assert full == canonical({
        "op": "scatter",
        "impl": 3,
        "carry": evaluator.CARRY_VERSION,
        "params": params,
        "upstream": upstream,
        "node_id": "sc01",
        "kernels": [["core.mesh", mesh.KERNEL_VERSION], ["core.rand", rand.KERNEL_VERSION]],
    })
    assert len(full) == 24 and int(full, 16) >= 0
    bare = evaluator._cache_key("box", 1, {"size": [1.0, 1.0, 1.0]}, [])
    assert bare == canonical({
        "op": "box",
        "impl": 1,
        "carry": evaluator.CARRY_VERSION,
        "params": {"size": [1.0, 1.0, 1.0]},
        "upstream": [],
    })
    # Every named part shapes the key: the same payload with one part
    # changed is a different record.
    assert evaluator._cache_key("box", 2, {"size": [1.0, 1.0, 1.0]}, []) != bare
    assert evaluator._cache_key("box", 1, {"size": [1.0, 1.0, 2.0]}, []) != bare
    assert evaluator._cache_key("box", 1, {"size": [1.0, 1.0, 1.0]}, ["c" * 24]) != bare
    assert evaluator._cache_key("box", 1, {"size": [1.0, 1.0, 1.0]}, [], "bx01") != bare


def test_stream_function_is_pinned():
    """DESIGN 4.4: the stream function is pinned (``rand.py``: sha256 of
    the node id, folded with seed and index through a splitmix64
    finalizer), and ``rand.py`` is a declared kernel. These golden
    values freeze the function: a change that moves them is an
    output-semantics change and a ``KERNEL_VERSION`` bump."""
    import numpy as np

    from moraine3d.core import rand

    got = rand.stream_u64(7, "sc01", [0, 1, 2])
    assert got.dtype == np.uint64 and got.shape == (3,)
    assert [int(x) for x in got] == [
        0xD22E9E8577993929,
        0xBA462112F4E83B2B,
        0xC59D66BE1C8B583D,
    ]
    assert int(rand.stream_u64(0, "n", 0)) == 3738056435564720421
    # Each of seed, node id and index moves the stream.
    assert int(rand.stream_u64(8, "sc01", 0)) != int(got[0])
    assert int(rand.stream_u64(7, "sc02", 0)) != int(got[0])
