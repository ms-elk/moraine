"""The non-destructive determinism check (DESIGN 4.4): ``--recompute``
recomputes every node, compares result hashes with the records on disk,
and overwrites nothing."""

import json

from moraine3d.app.cli import main
from moraine3d.cache import records
from support import projects
from support.engine import Evaluation, export_asset

_ASSET = """{
"build_version": "fixture",
"nodes": [
{"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
{"id": "t1", "op": "transform", "inputs": ["b1"], "params": {"translate": [0.0, 0.5, 0.0]}}
],
"outputs": {"main": "t1"}
}
"""


def _project(tmp_path, monkeypatch):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    d = tmp_path / "asset"
    projects.write_project(d, _ASSET, name="asset")
    return d


def _meta_files(tmp_path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (tmp_path / "cache").glob("*.meta.json")}


def _tamper(tmp_path, key: str) -> None:
    """Rewrite one record's result_hash on disk: the bytes a
    re-evaluation must be compared against, made to disagree."""
    _npz, meta = records.record_paths(key)
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["result_hash"] = "0" * len(doc["result_hash"])
    meta.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")


def test_recompute_matches_a_warm_cache_and_leaves_records_alone(tmp_path, monkeypatch):
    """Determinism (DESIGN 4.4): a recompute over a warm cache matches
    every record and rewrites none of them — identical bytes need no
    rewrite."""
    d = _project(tmp_path, monkeypatch)
    sc = projects.load(d)
    Evaluation(sc).records()
    before = _meta_files(tmp_path)
    ev = Evaluation(sc, recompute=True)
    recs = ev.records()
    assert recs["main"].value() is not None
    assert ev.recompute_report() == {"nodes": 2, "matched": 2, "unrecorded": 0}
    assert ev.profile()["hits"] == 0 and ev.profile()["misses"] == 2
    assert _meta_files(tmp_path) == before


def test_recompute_reports_a_mismatch_and_keeps_the_evidence(tmp_path, monkeypatch):
    """A record whose bytes disagree with the recomputation is reported
    as mismatched, by node and op, and left in place (DESIGN 4.4: the
    difference is the evidence). A tampered meta leaves the real bytes
    unchanged, so the downstream key still resolves and matches — only
    a genuinely different recomputation re-keys its downstream."""
    d = _project(tmp_path, monkeypatch)
    sc = projects.load(d)
    ev0 = Evaluation(sc)
    ev0.records()
    box_key = ev0._eval_node(sc, "b1").key
    _tamper(tmp_path, box_key)
    tampered = _meta_files(tmp_path)
    ev = Evaluation(sc, recompute=True)
    ev.records()
    rep = ev.recompute_report()
    assert rep["nodes"] == 2 and rep["matched"] == 1 and rep["unrecorded"] == 0
    assert rep["mismatched"] == [{"node": "b1", "op": "box", "site": "asset.star:1"}]
    assert rep["mismatched_total"] == 1
    assert "'b1' (box)" in rep["fix"] and "left in place" in rep["fix"]
    after = _meta_files(tmp_path)
    assert after[f"{box_key}.meta.json"] == tampered[f"{box_key}.meta.json"]


def test_recompute_mismatch_at_an_output_materializes_no_artifact(tmp_path, monkeypatch):
    """A mismatched output node keeps its record's key, so a GLB under it
    would be the stale one or overwrite the evidence: the entry carries
    `mismatched: true` in place of its artifact, and no GLB is written
    (DESIGN 4.4/6: the cache is never overwritten by the check)."""
    d = _project(tmp_path, monkeypatch)
    sc = projects.load(d)
    ev0 = Evaluation(sc)
    ev0.records()  # records only: no GLB exists yet under the output's key
    out_key = ev0._eval_node(sc, "t1").key
    _tamper(tmp_path, out_key)
    ev = Evaluation(sc, recompute=True)
    entry = ev.results()["main"]
    assert entry["mismatched"] is True and "glb" not in entry and "glb_file" not in entry
    assert not list((tmp_path / "cache").glob(f"{out_key}-*.glb"))
    assert ev.recompute_report()["mismatched"] == [
        {"node": "t1", "op": "transform", "site": "asset.star:2"}
    ]


def test_export_recompute_mismatch_writes_nothing(tmp_path, monkeypatch, capsys):
    """`export --recompute` judges before it writes: a mismatch is a red
    report with nothing written, and the last good export stays in place
    (DESIGN 6 — the validation-failure rule applied to the check)."""
    d = _project(tmp_path, monkeypatch)
    sc = projects.load(d)
    assert export_asset(sc, profile="gltf", out_dir=d / "exports")["ok"] is True
    before = {p.name: p.read_bytes() for p in (d / "exports").iterdir()}
    ev = Evaluation(sc)
    ev.records()
    _tamper(tmp_path, ev._eval_node(sc, "b1").key)
    assert main(["export", str(d), "-o", "exports", "--recompute"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["recompute"]["mismatched_total"] == 1
    assert "fix" in out and out["outputs"] == {}
    assert {p.name: p.read_bytes() for p in (d / "exports").iterdir()} == before


def test_recompute_cli_exit_codes(tmp_path, monkeypatch, capsys):
    """`eval --recompute` prints the verdict under `recompute` and exits 1
    on a mismatch with the report intact (DESIGN 6)."""
    d = _project(tmp_path, monkeypatch)
    assert main(["eval", str(d)]) == 0
    capsys.readouterr()
    assert main(["eval", str(d), "--recompute"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recompute"] == {"nodes": 2, "matched": 2, "unrecorded": 0}
    sc = projects.load(d)
    ev = Evaluation(sc)
    ev.records()
    _tamper(tmp_path, ev._eval_node(sc, "b1").key)
    assert main(["eval", str(d), "--recompute"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["recompute"]["mismatched"][0]["node"] == "b1"
    assert "outputs" in out  # the report, not an error object


def test_export_recompute_writes_identical_bytes(tmp_path, monkeypatch, capsys):
    """`export --recompute` writes the GLB from recomputed values: the
    byte-identical re-export proof without a cache clear (DESIGN 4.4);
    a mismatch turns the export report red."""
    d = _project(tmp_path, monkeypatch)
    sc = projects.load(d)
    assert export_asset(sc, profile="gltf", out_dir=d / "exports")["ok"] is True
    first = (d / "exports" / "asset.glb").read_bytes()
    assert main(["export", str(d), "-o", "again", "--recompute"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["recompute"]["matched"] == 2
    assert (d / "again" / "asset.glb").read_bytes() == first
    ev = Evaluation(sc)
    ev.records()
    _tamper(tmp_path, ev._eval_node(sc, "b1").key)
    assert main(["export", str(d), "-o", "third", "--recompute"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["recompute"]["mismatched"][0]["node"] == "b1"
