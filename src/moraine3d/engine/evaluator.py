"""Graph evaluation with content-addressed cache (DESIGN 4.5).

Cache key = hash(op_name, op_impl_version, params, upstream_result_hashes)
— plus the node id for operators that declare ``USES_NODE_ID`` (their
stochastic streams are salted with it, DESIGN 4.4), plus the resolved
version of every kernel the operator declares in ``KERNELS`` (an
installed distribution's version, or a shared in-package kernel
module's ``KERNEL_VERSION``), so a dependency upgrade or a shared-kernel
fix can never serve records written under the old implementation.
``op_impl_version`` is inside every key (Day-One Invariant 5).

Every node's value is cached (``<key>.npz`` + ``<key>.meta.json``), so a
parameter change re-evaluates only the dirty downstream subgraph:
untouched upstream nodes hit the cache by key and are loaded — or, when
only their hash is needed, not even loaded. Output GLBs are cached as
``<key>-g<glb_version>-<output_name>.glb`` (the name participates because
glTF nodes are named after asset outputs — Day-One Invariant 7; the
writer version participates so a change to GLB generation orphans stale
GLBs instead of serving them).

The cache is machine-level (see ``paths.cache_dir``): content-addressed
keys are location-independent, so results are shared across projects.
How a record lives on disk — names, the meta contract, atomic writes
under concurrent processes — is ``cache.records``; this module only
decides which key a node has and when to compute.

Asset references (DESIGN 4.6): a ``asset_ref`` node is evaluated here,
not in its op module — the evaluator owns asset loading, the per-file
record namespaces, and reference-cycle detection. An un-overridden
asset_ref is an identity over the referenced output, so its record IS
the referenced output's record: location params (path, output name)
never touch a cache key, and dirty propagation across files plus
content-level dedup follow from the shared key. Per-use ``overrides``
are resolved by the build, which writes one IR per override identity;
the evaluator derives the identity from the node's ``overrides`` to
open that IR and never applies them, so cache keys follow content as
always. What the overrides change is *evaluation identity* — the
record memo and the asset registry carry a normalized override token,
giving each differently-overridden use of one file its own instance
(and un-overridden uses the shared plain one). A
``material_rename`` does not join that identity: it is a
one-input derived record over the referenced output's record — keyed
by the referenced result hash plus the mapping, materialized and cached
exactly like an op result — so the rename stays the consumer's and the
sub-asset instance (and its asserts) is shared with un-renamed uses.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import ops
from ..cache import records
from ..cache.records import Record, stats_attributes
from ..core import params as params_mod
from ..core import asset as asset_mod
from ..core.mesh import Image, Mesh, Value, proto_of
from ..core.asset import Node, Asset
from ..core.sites import where
from ..errors import MoraineError, OpError, AssetError
from ..ops import smooth_normals


@dataclass
class OpContext:
    node_id: str


# Evaluator-level output semantics that no op_impl_version covers, folded
# into every cache key (DESIGN 4.5). Bump when ``evaluate_op_value``'s
# carry rules (material / face-group propagation) or the npz record
# layout (``records.save_value`` / ``records.load_value``) change —
# otherwise records written under the old rules keep hitting under
# unchanged keys and a warm cache answers differently from a cold one.
CARRY_VERSION = 1


@functools.cache
def _kernel_version(name: str) -> str | int:
    """The installed version of a kernel an op declares in ``KERNELS``
    (DESIGN 4.5): a dotted name is a shared in-package kernel module
    (``core.mating``) and resolves to its ``KERNEL_VERSION``; a bare name
    is an installed distribution (``manifold3d``) and resolves to its
    distribution version. Memoized: versions cannot change within a
    process, and key computation runs per node per evaluation."""
    if "." in name:
        return int(importlib.import_module(f"..{name}", __package__).KERNEL_VERSION)
    return importlib.metadata.version(name)


def _cache_key(
    op_name: str, impl_version: int, params: dict, upstream: list[str],
    node_id: str | None = None, kernels: tuple[str, ...] = (),
) -> str:
    """Content-addressed record key: explicit parts, so a derived record
    (a renamed reference) keys exactly the params that shape it. The
    kernel field is omitted when the op declares none, so a pure-numpy
    op's key is exactly the pre-declaration key."""
    payload: dict = {
        "op": op_name,
        "impl": impl_version,
        "carry": CARRY_VERSION,
        "params": params,
        "upstream": upstream,
    }
    if node_id is not None:
        payload["node_id"] = node_id
    if kernels:
        payload["kernels"] = [[k, _kernel_version(k)] for k in sorted(kernels)]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


# -- evaluation -------------------------------------------------------------


def _material_policy(op) -> str:
    """``MATERIAL_POLICY`` with ``ATTRIBUTE_POLICY`` as its default: an op
    that rebuilds topology (drops attribute slots) drops the bound material
    for the same reason, unless it declares otherwise."""
    return getattr(op, "MATERIAL_POLICY", getattr(op, "ATTRIBUTE_POLICY", "preserved"))


def _propagate_material(op, inputs: list[Value], value: Value) -> None:
    """Carry a bound material through single-input material-preserving ops.

    Centralized here (not per-op) so the contract cannot be forgotten:
    ops keep constructing bare geometry; the evaluator re-binds input 0's
    material onto the fresh result. Multi-input ops never inherit
    implicitly — instancing ops carry the material through prototype
    object identity, and dropping ops warn (attribute_drop_warnings).
    """
    if _material_policy(op) != "preserved" or len(inputs) != 1:
        return
    proto_out = proto_of(value)
    if not isinstance(proto_out, Mesh) or proto_out.material is not None:
        return
    src = inputs[0]
    if isinstance(src, Image):
        return
    src_proto = proto_of(src)
    if src_proto.material is not None:
        proto_out.material = src_proto.material


def _propagate_face_groups(op, inputs: list[Value], value: Value) -> None:
    """Carry face groups through single-input topology-preserving ops.

    Same centralization rationale as ``_propagate_material``; groups
    follow ATTRIBUTE_POLICY (a topology rebuild invalidates face indices
    for the same reason it invalidates vertex slots). The face-count
    guard makes a policy misdeclaration fail safe: indices are copied
    only onto an identical face layout. Ops that reshape faces but keep
    a meaningful mapping (``bake_instances`` tiling) handle groups
    themselves.
    """
    if getattr(op, "ATTRIBUTE_POLICY", "preserved") != "preserved" or len(inputs) != 1:
        return
    proto_out = proto_of(value)
    if not isinstance(proto_out, Mesh) or proto_out.face_groups is not None:
        return
    src = inputs[0]
    if isinstance(src, Image):
        return
    src_proto = proto_of(src)
    if src_proto.face_groups and len(src_proto.faces) == len(proto_out.faces):
        proto_out.face_groups = dict(src_proto.face_groups)


def evaluate_op_value(op, inputs: list[Value], params: dict, node_id: str) -> Value:
    """Evaluate one op over materialized inputs and apply the carry
    contracts (material, face groups) — the single step both the in-graph
    walk (``_eval_op``) and the export edge's ``--bake-instances``
    projection run, so the two can never drift apart (DESIGN 9)."""
    value = op.evaluate(inputs, params, OpContext(node_id))
    _propagate_material(op, inputs, value)
    _propagate_face_groups(op, inputs, value)
    return value


class Evaluation:
    """One evaluation pass over a root asset and its reference DAG (4.6).

    Records are memoized per (asset file, override token, node id) so a
    diamond of references evaluates each sub-asset instance once — one
    instance per distinct override set, the plain instance shared by all
    un-overridden uses. ``asset_files`` collects every referenced file
    (including ones that then fail to load) so ``serve`` knows what to
    watch.
    """

    def __init__(self, sc: Asset, preview: bool = False, recompute: bool = False,
                 irs: dict[Path, Asset] | None = None, *,
                 source_of: Callable[[Path], Path | None],
                 stamp: str):
        self.root = sc
        # The IRs a build just made, by the path each is written at
        # (DESIGN 4.8): a reference opens its target here first, so an
        # evaluation that follows a build reads what that build built,
        # written or not. Anything else is opened from disk.
        self._irs = irs or {}
        # The file an IR was written from, by the IR's path, or None
        # where it has none (DESIGN 6): what a report names in the IR's
        # place. The evaluator reads nothing from the answer but where
        # it sits.
        self._source_of = source_of
        # The stamp an IR opened from disk must carry (DESIGN 4.1), the
        # running build's. The root's own is checked where the root is
        # opened.
        self._stamp = stamp
        # Reference -> the IR it names: resolving touches the disk, and
        # a kit is referenced once per output it lends.
        self._targets: dict[tuple, Path] = {}
        # IR path -> report name: the name is asked once per report row.
        self._rel_memo: dict[Path, str] = {}
        # Preview pass (DESIGN 4.5): reads hit the persistent store first
        # (clean upstream records are file history and stay warm), but
        # every miss this pass computes lands in the ephemeral preview
        # bucket — exploration must never grow the persistent cache.
        self.preview = preview
        # Determinism check (DESIGN 4.4): every node is recomputed, the
        # disk record read only to compare result hashes, and nothing on
        # disk overwritten — a mismatch is evidence, not something to
        # replace. The CLI's proof of byte-identical re-evaluation without
        # a machine-wide cache clear; never combined with a preview pass.
        if preview and recompute:
            raise ValueError("recompute is a file-state check; a preview pass cannot run it")
        self.recompute = recompute
        self._recompute: list[dict] = []
        self._mismatched_keys: set[str] = set()  # records whose fresh bytes differ
        self.asset_files: set[Path] = set()
        # (resolved path, override token) -> loaded asset instance
        self._assets: dict[tuple[Path, str], Asset] = {}
        # (asset path | None, override token, node id)
        self._records: dict[tuple, Record] = {}
        # asset_ref record key -> the referenced output's record key, so
        # graph walks (shading_warnings) can cross reference boundaries.
        self._ref_aliases: dict[tuple, tuple] = {}
        # record key -> whether its flat-shaded value has curved surface
        # (the flat_shading gate; one mesh load per boolean, not per walk).
        self._curved_memo: dict[tuple, bool] = {}
        # (resolved path, override token) -> the landed overrides, for
        # ref_asserts report stamping (token alone is not human-readable).
        self._ref_overrides: dict[tuple[Path, str], dict] = {}
        # Chain of asset files from the root down to the asset currently
        # evaluating, for reference-cycle reporting.
        self._stack: list[Path] = [sc.path] if sc.path is not None else []
        self._proj_root = asset_mod.project_root(sc.path) if sc.path is not None else None
        # Profiling (DESIGN 6): one entry per memo entry —
        # (path, token, node) — evaluated this pass. Plain asset_ref
        # aliases and in-process memo returns contribute nothing (a
        # renamed ref is a materialized record and counts like an op);
        # hits + misses counts disk lookups. Two override identities whose landed params
        # coincide (e.g. an override restating the declared value) look
        # one disk key up twice, so the sum can exceed unique keys.
        self._prof: list[dict] = []
        self._hits = 0
        self._misses = 0

    # -- public --

    def where_of(self, node_id: object, asset: str | None = None) -> dict:
        """The recipe position of a node (DESIGN 4.1) as a report row
        carries it, by the node's id and, for a node of a referenced
        asset, that asset's report name. Empty for a node without one."""
        if asset is None:
            pool = [self.root]
        else:
            pool = [
                sc for sc in self._assets.values()
                if sc.path is not None and self._rel(sc.path) == asset
            ]
        for sc in pool:
            node = sc.nodes.get(node_id) if isinstance(node_id, str) else None
            if node is not None and node.site is not None:
                return where(node)
        return {}

    def _sited(self, rows: list[dict]) -> list[dict]:
        """``rows`` with each row that names a node carrying its site,
        so a report never makes its reader search the recipe for it."""
        for row in rows:
            found = self.where_of(row.get("node"), row.get("asset"))
            if "site" not in row:
                row.update(found)
        return rows

    def output_sites(self) -> dict[str, dict]:
        """Output name -> the recipe position of its node
        (``asset.where``), for the outputs whose node carries one."""
        out = {}
        for name, node_id in self.root.outputs.items():
            found = where(self.root.nodes[node_id])
            if found:
                out[name] = found
        return out

    def records(self, only: str | None = None) -> dict[str, Record]:
        """Evaluate named root outputs (all, or just ``only``); populate cache."""
        try:
            return self._records_of(only)
        except MoraineError as e:
            self._site_error(e)
            raise

    def _site_error(self, e: MoraineError) -> None:
        """An error that names a node names its recipe line too."""
        found = self.where_of(e.detail.get("node"), e.detail.get("asset"))
        if "site" not in e.detail:
            e.detail.update(found)

    def _records_of(self, only: str | None = None) -> dict[str, Record]:
        outputs = self.root.outputs
        if only is not None:
            if only not in outputs:
                raise AssetError(
                    "unknown_output",
                    f"no output named '{only}'; asset outputs: {sorted(outputs)}",
                )
            outputs = {only: outputs[only]}
        return {
            name: self._eval_chain(self.root, node_id)
            for name, node_id in outputs.items()
        }

    def results(self, only: str | None = None) -> dict:
        """records() plus output GLBs (meshes) / preview PNGs (images);
        per-output info dicts."""
        results: dict[str, dict] = {}
        for name, rec in self.records(only).items():
            node = self.root.nodes[self.root.outputs[name]]
            entry = {"node_id": node.id}
            entry.update(where(node))
            entry["key"] = rec.key
            entry["stats"] = rec.stats
            if rec.key in self._mismatched_keys:
                # The determinism check found this node's fresh bytes
                # disagree with its disk record, and its key is the
                # record's: a derived artifact under that key would
                # either be the stale one or overwrite the evidence
                # (DESIGN 4.4) - the report carries the verdict instead.
                entry["mismatched"] = True
            elif rec.stats.get("kind") == "image":
                png_path = records.ensure_preview_png(rec, name)
                entry["png"] = str(png_path)
                entry["png_file"] = png_path.name
            else:
                glb_path = records.ensure_glb(rec, name)
                entry["glb"] = str(glb_path)
                entry["glb_file"] = glb_path.name
            if rec.preview:
                # The record lives in the ephemeral bucket: consumers that
                # address artifacts by location (the server's URL routes)
                # must not point at the persistent store.
                entry["bucket"] = True
            results[name] = entry
        return results

    def assets(self) -> list[Asset]:
        """The root asset plus every referenced asset instance loaded so
        far — complete after ``records()``/``results()`` has walked the
        graph. One file may contribute several instances, one per
        override identity (DESIGN 4.6)."""
        return [self.root, *self._assets.values()]

    def profile(self) -> dict:
        """Cache hit/miss counts and per-node wall time for this pass
        (DESIGN 6), token-terse per 8.2: a rollup plus the
        slowest misses only — misses are the only optimization targets,
        and a large asset must not dump one line per node. Only
        meaningful after ``records()``/``results()``.
        """
        out = {
            "hits": self._hits,
            "misses": self._misses,
            "ms": round(sum(e["ms"] for e in self._prof), 1),
        }
        slowest = sorted(
            (e for e in self._prof if not e["hit"]),
            key=lambda e: e["ms"], reverse=True,
        )[:5]
        if slowest:
            out["slowest"] = self._sited([
                {
                    "node": e["node"], "op": e["op"], "ms": round(e["ms"], 1),
                    **({"asset": e["asset"]} if "asset" in e else {}),
                }
                for e in slowest
            ])
        return out

    def ops_used(self) -> dict[str, int]:
        """Op name -> impl version over nodes actually evaluated — the ops
        whose semantics the produced values depend on. Unreachable nodes
        (never evaluated, possibly unknown ops) are excluded by
        construction. Only meaningful after ``records()``/``results()``.
        """
        used: set[str] = set()
        for path, token, node_id in self._records:
            sc = self._instance(path, token)
            if sc is not None:
                used.add(sc.nodes[node_id].op)
        return {name: ops.get(name).OP_IMPL_VERSION for name in sorted(used)}

    # -- advisory warnings (DESIGN 4.1) --
    #
    # Root-asset nodes only: a warning names an edit site, and a node
    # inside a referenced asset is one the consumer cannot edit — the
    # producer's own validate reports it. The walks still cross
    # references where the root's surface arrives through one.

    def _root_record(self, node_id: str):
        return self._records.get((self.root.path, "", node_id))

    def _carried_refs(self, node: Node) -> list[str]:
        op = ops.get(node.op)
        carried = getattr(op, "CARRIED_INPUTS", None)
        if carried is None:
            return list(node.inputs)
        return [node.inputs[i] for i in carried if i < len(node.inputs)]

    def _downstream_maps(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Root-asset carried adjacency (input -> evaluated consumers) and
        output node -> output names — built once per warning pass."""
        down: dict[str, list[str]] = {}
        for nid, node in self.root.nodes.items():
            if self._root_record(nid) is None:
                continue
            for ref in self._carried_refs(node):
                down.setdefault(ref, []).append(nid)
        by_node: dict[str, list[str]] = {}
        for name, nid in self.root.outputs.items():
            by_node.setdefault(nid, []).append(name)
        return down, by_node

    @staticmethod
    def _reached_outputs(node_id: str, down: dict, by_node: dict) -> list[str]:
        """Root output names whose surface the node's value flows into
        (downstream along carried inputs); empty for a value that only
        anchors, targets, or cuts and never itself ships."""
        reached: list[str] = []
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            reached.extend(by_node.get(nid, []))
            stack.extend(down.get(nid, []))
        return sorted(reached)

    def attribute_drop_warnings(self) -> list[dict]:
        """Advisory findings: root-asset nodes whose op drops attribute
        slots, face groups, or a bound material present on an input
        (``ATTRIBUTE_POLICY`` / ``MATERIAL_POLICY`` contracts — never
        silent, DESIGN P6), restricted to losses that reach an output:
        the dropped slot, group, or material is still absent at some
        root output the node's value flows into. A loss re-authored
        downstream (``unwrap`` after the boolean, the material re-bound
        as the part's last node) never ships, and a value that only cuts
        or anchors reaches no output — neither is reported. Only
        meaningful after ``records()``/``results()``.
        """
        warnings: list[dict] = []
        down, by_node = self._downstream_maps()
        for node_id, node in self.root.nodes.items():
            if self._root_record(node_id) is None:
                continue
            op = ops.get(node.op)
            attr_drops = getattr(op, "ATTRIBUTE_POLICY", "preserved") == "dropped"
            mat_drops = _material_policy(op) == "dropped"
            if not attr_drops and not mat_drops:
                continue
            dropped: set[str] = set()
            materials: set[str] = set()
            groups: set[str] = set()
            for ref in node.inputs:
                up = self._root_record(ref)
                if up is None:
                    continue
                if attr_drops:
                    dropped.update(stats_attributes(up.stats))
                    groups.update(up.stats.get("groups", {}))
                if mat_drops and "material" in up.stats:
                    materials.add(up.stats["material"])
            if not (dropped or groups or materials):
                continue
            reached = self._reached_outputs(node_id, down, by_node)
            if not reached:
                continue
            at_output = {
                name: self._root_record(self.root.outputs[name]).stats for name in reached
            }
            lost_attrs = sorted(
                s for s in dropped
                if any(s not in stats_attributes(st) for st in at_output.values())
            )
            lost_groups = sorted(
                g for g in groups
                if any(g not in st.get("groups", {}) for st in at_output.values())
            )
            bare = sorted(name for name, st in at_output.items() if "material" not in st)
            if lost_attrs:
                outs = sorted(
                    name for name, st in at_output.items()
                    if any(s not in stats_attributes(st) for s in lost_attrs)
                )
                warnings.append(
                    {
                        "warning": "attributes_dropped",
                        "node": node_id,
                        "op": node.op,
                        "dropped": lost_attrs,
                        "outputs": outs,
                        "note": f"{node.op} rebuilds topology and drops these input "
                        f"slots (semantics contract), and nothing downstream "
                        f"re-authors them before the outputs named; unwrap after "
                        f"{node.op} if they are needed",
                    }
                )
            if lost_groups:
                outs = sorted(
                    name for name, st in at_output.items()
                    if any(g not in st.get("groups", {}) for g in lost_groups)
                )
                warnings.append(
                    {
                        "warning": "groups_dropped",
                        "node": node_id,
                        "op": node.op,
                        "groups": lost_groups,
                        "outputs": outs,
                        "note": f"{node.op} rebuilds topology and drops these face "
                        f"groups (semantics contract), and the outputs named "
                        f"ship without them; re-apply tag after {node.op} if a "
                        f"consumer places against them",
                    }
                )
            if materials and bare:
                warnings.append(
                    {
                        "warning": "material_dropped",
                        "node": node_id,
                        "op": node.op,
                        "materials": sorted(materials),
                        "outputs": bare,
                        "note": f"{node.op} rebuilds its result and drops the input "
                        f"material (semantics contract), and the outputs named "
                        f"ship with no material; re-apply the material node "
                        f"after {node.op}",
                    }
                )
        return self._sited(warnings)

    def _curved(self, key: tuple) -> bool:
        """Whether the flat-shaded value at ``key`` has curved surface to
        restore (``smooth_normals.would_smooth`` at the default threshold);
        memoized per record — several materials may reach one boolean."""
        if key not in self._curved_memo:
            value = self._records[key].value()
            mesh = proto_of(value) if not isinstance(value, Image) else None
            self._curved_memo[key] = mesh is not None and smooth_normals.would_smooth(mesh)
        return self._curved_memo[key]

    def shading_warnings(self) -> list[dict]:
        """Advisory findings: a ``boolean`` output with curved surface
        reaching a root-asset ``material`` node with its flat shading
        intact — no shading-replacing op in between — so the bound
        surface loses curved shading. Fires per material node (the edit
        site is that node's input chain), crosses ``asset_ref``
        boundaries via the reference aliases (a boolean from another
        asset is labeled ``asset:id`` — node ids are per-file), follows
        only surface-carrying inputs (``CARRIED_INPUTS``), and stops at
        ``SHADING_POLICY: "replaced"`` ops, whose output shading is
        independent of the input's. Gated on face-normal spread: an
        all-planar boolean output (a box minus a box) shades identically
        flat or smooth, so ``smooth_normals`` would change nothing and
        nothing is reported. Only meaningful after
        ``records()``/``results()``.
        """
        warnings: list[dict] = []
        path, token = self.root.path, ""
        for node_id, node in self.root.nodes.items():
            if node.op != "material" or self._root_record(node_id) is None:
                continue
            booleans: set[str] = set()
            seen_keys: set[tuple] = set()
            stack = [(path, token, ref) for ref in node.inputs]
            while stack:
                cur = stack.pop()
                if cur in seen_keys or cur not in self._records:
                    continue
                seen_keys.add(cur)
                cpath, ctoken, cid = cur
                csc = self._instance(cpath, ctoken)
                if csc is None:
                    continue
                cnode = csc.nodes[cid]
                cop = ops.get(cnode.op)
                if getattr(cop, "SHADING_POLICY", "preserved") == "replaced":
                    # A shading-replacing op ends the causal chain:
                    # nothing upstream reaches the material's surface.
                    # Whether its own flat output warrants the warning
                    # is the op's declaration (FLAT_SHADING_SOURCE),
                    # like every other fact this walk consumes.
                    if getattr(cop, "FLAT_SHADING_SOURCE", False) and self._curved(cur):
                        booleans.add(
                            cid if cpath == path
                            else f"{self._rel(cpath)}:{cid}"
                        )
                    continue
                if getattr(cop, "IS_REFERENCE", False):
                    alias = self._ref_aliases.get(cur)
                    if alias is not None:
                        stack.append(alias)
                    continue
                stack.extend((cpath, ctoken, r) for r in self._carried_refs(cnode))
            if booleans:
                warnings.append(
                    {
                        "warning": "flat_shading",
                        "node": node_id,
                        "op": "material",
                        "booleans": sorted(booleans),
                        "note": "a boolean output with curved surface reaches "
                        "this material without smooth_normals in between; "
                        "boolean output is flat-shaded - insert smooth_normals "
                        "after the boolean to restore curved-surface shading",
                    }
                )
        return self._sited(warnings)

    def mate_frames(self) -> list[dict]:
        """Frame provenance of every evaluated root-asset node whose op
        declares ``frame_provenance`` (``mate``): one row per node with
        the origin word per role (``opening`` / ``centroid``, DESIGN
        5.1). The op's own report of a fact its execution never
        surfaces — the area-centroid fallback is silent, and the paired
        ``mated`` shares the frame and passes — carried by ``validate``
        under ``mates``. Only meaningful after ``records()``/``results()``.
        """
        rows: list[dict] = []
        for node_id, node in self.root.nodes.items():
            if self._root_record(node_id) is None:
                continue
            provenance = getattr(ops.get(node.op), "frame_provenance", None)
            if provenance is None:
                continue
            inputs = [self._root_record(ref).value() for ref in node.inputs]
            rows.append({"node": node_id, **provenance(inputs, node.params)})
        return self._sited(rows)

    def material_origins(self) -> dict[str, dict]:
        """Which references each root output's material arrives through
        (DESIGN 9.1).

        Per output: ``refs`` — the reference node ids the output's
        material arrives through when nothing local authors an identity
        (the fix target for a name collision the consumer cannot
        otherwise repair: ``material_rename`` on that node); empty when
        this asset authors the identity (``authors_material`` — a local
        ``material`` node, or a ``asset_ref`` carrying
        ``material_rename``: the consumer named it) or the carried
        closure crosses no reference. The walk follows
        ``CARRIED_INPUTS`` only (a scatter target or a place anchor
        base never reaches the output's surface, so neither its
        material nor its references count — the same rule the shading
        walk uses) and is op-trait-driven (``IS_REFERENCE`` /
        ``authors_material``), never module identity. Asset structure
        only — no values are loaded.
        """
        sc = self.root
        origins: dict[str, dict] = {}
        for name, root_id in sc.outputs.items():
            local = False
            ref_ids: list[str] = []
            seen: set[str] = set()
            stack = [root_id]
            while stack:
                nid = stack.pop()
                if nid in seen:
                    continue
                seen.add(nid)
                node = sc.nodes[nid]
                op = ops.get(node.op)
                if getattr(op, "IS_REFERENCE", False):
                    ref_ids.append(nid)
                authors = getattr(op, "authors_material", None)
                if authors is not None and authors(node.params):
                    local = True
                stack.extend(self._carried_refs(node))
            origins[name] = {"refs": [] if local else sorted(ref_ids)}
        return origins

    def embedded_via_refs(self, asset: Asset | None = None, token: str = "") -> dict:
        """Output coverage of embeds declared inside the assets that
        ``asset`` (default the root; ``token`` its override identity)
        references (DESIGN 5.2 — the guard's cede travels with the
        parts).

        An output that reaches a ``asset_ref`` through ops declaring a
        ``PLACEMENT`` (``ops``) maps into that reference's (asset,
        override identity): ``"identity"`` ops leave vertex positions
        untouched, ``"affine"`` ops compose their matrix — a scaled one
        breaks the trace, since embed depths change — and ``"rigid"``
        ops move their part as one body by a data-dependent motion. Two
        outputs mapping into the same identity under the same composed
        placement, with no data-dependent motion on either trace,
        inherit the sub-asset's declared pair embeds (``pairs``, in
        this asset's output names); an output mapping onto a sub-output
        with a declared copy-form embed inherits it for its own copies
        (``self``) — any rigid motion of the whole set keeps every copy
        pair's depth. A different override identity may change the
        geometry, so it never maps — the consumer declares those pairs
        itself. Op-trait-driven, never module identity. Asset structure
        plus the reference registry only (no values loaded); meaningful
        after ``records()``/``results()``.
        """
        import numpy as np

        sc = self.root if asset is None else asset
        traces: dict[str, tuple] = {}
        for name, root_id in sc.outputs.items():
            m = np.eye(4)
            data_placed = False
            nid = root_id
            while True:  # the loader guarantees an acyclic graph
                node = sc.nodes[nid]
                op = ops.get(node.op)
                if getattr(op, "IS_REFERENCE", False):
                    alias = self._ref_aliases.get((sc.path, token, node.id))
                    if alias is not None:
                        target, sub_token, _ = alias
                        traces[name] = (
                            target, sub_token, node.params.get("output"), m,
                            data_placed,
                        )
                    break
                placement = getattr(op, "PLACEMENT", None)
                if placement == "affine":
                    tm = op.matrix(node.params)
                    lin = tm[:3, :3]
                    if not np.allclose(lin @ lin.T, np.eye(3), rtol=0.0, atol=1e-12):
                        break  # scaled placement: embed depths change
                    m = m @ tm
                elif placement == "rigid":
                    data_placed = True
                elif placement != "identity":
                    break
                inputs = self._carried_refs(node)
                if len(inputs) != 1:
                    break
                nid = inputs[0]

        sub_memo: dict[tuple, tuple[set, set]] = {}

        def sub_embeds(target, token) -> tuple[set, set]:
            key = (target, token)
            if key not in sub_memo:
                sub = self._assets.get(key)
                pairs: set[frozenset] = set()
                selfs: set[str] = set()
                sub_groups = (sub.groups or {}) if sub is not None else {}
                for a in (sub.asserts or []) if sub is not None else []:
                    if a["kind"] == "embedded":
                        if len(a["outputs"]) == 2:
                            # A group pair is its member cross pairs
                            # (check_asserts cedes the same set).
                            x, y = a["outputs"]
                            pairs.update(
                                frozenset((p, q))
                                for p in sub_groups.get(x, [x])
                                for q in sub_groups.get(y, [y])
                                if p != q
                            )
                        else:
                            selfs.add(a["outputs"][0])
                sub_memo[key] = (pairs, selfs)
            return sub_memo[key]

        names = sorted(traces)
        covered_pairs: list[list[str]] = []
        covered_self: list[str] = []
        for name in names:
            target, sub_token, out, _, _ = traces[name]
            if out in sub_embeds(target, sub_token)[1]:
                covered_self.append(name)
        for i, na in enumerate(names):
            ta = traces[na]
            for nb in names[i + 1:]:
                tb = traces[nb]
                if (
                    ta[0] == tb[0] and ta[1] == tb[1]
                    and not ta[4] and not tb[4]
                    and np.array_equal(ta[3], tb[3])
                    and frozenset({ta[2], tb[2]}) in sub_embeds(ta[0], ta[1])[0]
                ):
                    covered_pairs.append([na, nb])
        return {"pairs": covered_pairs, "self": covered_self}

    def ref_asserts(self) -> list[dict]:
        """Sub-asset assert bundles (DESIGN 4.6): for each asset
        instance loaded through ``asset_ref``, its asserts plus every
        evaluated output — in the sub-asset's own namespace, one bundle
        per (asset, override identity), exactly the identity under which
        overrides change the geometry. Every instance bundles, asserts
        or not: the built-in overlap guard runs in each (the validator
        injects it), over all of the instance's outputs, so outputs no
        consumer evaluates are evaluated here — cached records make
        repeat passes cheap — and ``ref_embeds`` carries the cede the
        instance's own references contribute. Nested references are
        covered transitively (every loaded instance appears in the
        memo): evaluating an unconsumed output can itself load a
        sub-asset no consumer reached, so bundling repeats until the
        memo stops growing. Only meaningful after
        ``records()``/``results()``.
        """
        bundles: list[dict] = []
        values_by_instance: dict[tuple[Path, str], dict] = {}
        # Evaluating an instance's outputs may load further instances
        # (an asset_ref reachable only through an unconsumed output), so
        # a single snapshot of the memo would miss them: evaluate to a
        # fixpoint first, then bundle, so every cede sees every instance.
        while True:
            pending = [
                key for key in self._assets if key not in values_by_instance
            ]
            if not pending:
                break
            for path, token in sorted(
                pending, key=lambda key: (self._rel(key[0]), key[1])
            ):
                sub = self._assets[(path, token)]
                try:
                    values_by_instance[(path, token)] = {
                        name: self._eval_chain(sub, sub.outputs[name], token).value()
                        for name in sorted(sub.outputs)
                    }
                except MoraineError as e:
                    self._site_error(e)
                    raise
        instances = sorted(
            self._assets.items(), key=lambda kv: (self._rel(kv[0][0]), kv[0][1])
        )
        for (path, token), sub in instances:
            bundle = {
                "asset": self._rel(path),
                "asserts": list(sub.asserts or []),
                "values": values_by_instance[(path, token)],
                "ref_embeds": self.embedded_via_refs(sub, token),
            }
            if sub.groups:
                bundle["groups"] = sub.groups
            knobs = params_mod.range_entries(sub)
            if knobs:
                bundle["params"] = knobs
            overrides = self._ref_overrides.get((path, token)) or {}
            if overrides:
                bundle["overrides"] = overrides
            bundles.append(bundle)
        return bundles

    # -- internals --

    def rel(self, path: Path) -> str:
        """An asset file's report name: project-relative, or
        workspace-relative for a sibling project's file (DESIGN 4.6)."""
        return self._rel(path)

    def _rel(self, path: Path) -> str:
        name = self._rel_memo.get(path)
        if name is None:
            name = self._rel_memo[path] = self._report_name(path)
        return name

    def _report_name(self, path: Path) -> str:
        # A report names the file its reader edits.
        path = self._source_of(path) or path
        if self._proj_root is not None:
            if path.is_relative_to(self._proj_root):
                return path.relative_to(self._proj_root).as_posix()
            # Sibling-project files (cross-project refs, DESIGN 4.6) render
            # workspace-relative: 'water_tower/water_tower.mora'.
            if path.is_relative_to(self._proj_root.parent):
                return path.relative_to(self._proj_root.parent).as_posix()
        return path.name

    def _instance(self, path: Path | None, token: str) -> Asset | None:
        if path == self.root.path and token == "":
            return self.root
        return self._assets.get((path, token))

    def _eval_chain(self, sc: Asset, node_id: str, token: str = "") -> Record:
        """Memo-fill the subgraph upstream-first, without Python recursion.

        ``_eval_node`` recurses per input, so a deep-but-valid linear
        chain would hit the interpreter recursion limit — a raw
        RecursionError, not a MoraineError — at author-controlled depth.
        Walking iteratively first keeps ``_eval_node``'s recursion depth
        at one regardless of chain length; recursion across ``asset_ref``
        boundaries stays, bounded by reference nesting (``ref_cycle``
        caps it). The loader guarantees acyclicity per asset, so the
        walk terminates.
        """
        stack = [node_id]
        while stack:
            nid = stack[-1]
            if (sc.path, token, nid) in self._records:
                stack.pop()
                continue
            pending = [
                i for i in sc.nodes[nid].inputs
                if (sc.path, token, i) not in self._records
            ]
            if pending:
                stack.extend(pending)
            else:
                self._eval_node(sc, nid, token)
                stack.pop()
        return self._records[(sc.path, token, node_id)]

    def _eval_node(self, sc: Asset, node_id: str, token: str = "") -> Record:
        memo_key = (sc.path, token, node_id)
        rec = self._records.get(memo_key)
        if rec is not None:
            return rec
        node = sc.nodes[node_id]
        try:
            op = ops.get(node.op)
            ops.check_arity(node.op, len(node.inputs))
        except MoraineError as e:
            e.detail.setdefault("node", node.id)
            for k, v in self._ref_site(sc).items():
                e.detail.setdefault(k, v)
            raise
        unknown = set(node.params) - op.PARAMS
        if unknown:
            raise OpError(
                "unknown_param",
                f"{node.op} got unknown param(s) {sorted(unknown)}; "
                f"accepted: {sorted(op.PARAMS)}",
                node=node.id,
                **self._ref_site(sc),
            )
        if getattr(op, "IS_REFERENCE", False):
            rec = self._eval_ref(sc, node, op, token)
        else:
            rec = self._eval_op(sc, node, op, token)
        self._records[memo_key] = rec
        return rec

    def _eval_op(self, sc: Asset, node: Node, op, token: str) -> Record:
        ups = [self._eval_node(sc, i, token) for i in node.inputs]
        key = _cache_key(
            node.op, op.OP_IMPL_VERSION, node.params, [u.result_hash for u in ups],
            node.id if getattr(op, "USES_NODE_ID", False) else None,
            tuple(getattr(op, "KERNELS", ())),
        )

        def compute() -> Value:
            try:
                return evaluate_op_value(
                    op, [u.value() for u in ups], node.params, node.id
                )
            except MoraineError as e:
                e.detail.setdefault("node", node.id)
                if sc.path is not None and sc.path != self.root.path:
                    e.detail.setdefault("asset", self._rel(sc.path))
                raise

        return self._materialize(sc, node, key, compute)

    def _materialize(self, sc: Asset, node: Node, key: str, compute) -> Record:
        """Resolve one cache key to a record: disk hit (lazy value) or
        ``compute()`` then save. Shared by op nodes and renamed
        references — every materialized record goes through here. A
        preview pass reads through to the persistent store, then falls
        back to the ephemeral bucket, and stores misses aside into the
        bucket (DESIGN 4.5): the persistent cache indexes file history,
        the bucket indexes exploration."""
        t0 = time.perf_counter()  # after upstream recursion: own time only
        if self.recompute:
            return self._recompute_record(sc, node, key, compute, t0)
        rec = records.lookup(key)
        if rec is None and self.preview:
            rec = records.lookup(key, preview=True)
        if rec is not None:
            self._note(sc, node, hit=True, ms=(time.perf_counter() - t0) * 1000)
            return rec
        rec = records.store(key, compute(), preview=self.preview)
        self._note(sc, node, hit=False, ms=(time.perf_counter() - t0) * 1000)
        return rec

    def _recompute_record(self, sc: Asset, node: Node, key: str, compute, t0: float) -> Record:
        """The determinism check's materialization (DESIGN 4.4): compute
        regardless of the cache, compare the fresh result hash with the
        record on disk, and leave the disk record untouched either way —
        identical bytes need no rewrite, and differing bytes are the
        evidence the check exists to surface. A key with no record is
        stored as an ordinary miss would be (a mismatched node's
        downstream keys, built from its fresh hash, land here — the
        report names the first mismatch, not its shadow)."""
        on_disk = records.lookup(key)
        value = compute()
        row: dict = {"node": node.id, "op": node.op}
        if sc.path is not None and sc.path != self.root.path:
            row["asset"] = self._rel(sc.path)
        if on_disk is None:
            rec = records.store(key, value)
        else:
            rec = Record(key, value.result_hash(), value.stats(), value, False)
            row["matched"] = rec.result_hash == on_disk.result_hash
            if not row["matched"]:
                self._mismatched_keys.add(key)
        self._recompute.append(row)
        self._note(sc, node, hit=False, ms=(time.perf_counter() - t0) * 1000)
        return rec

    def recompute_report(self) -> dict:
        """The determinism check's verdict (DESIGN 4.4/6): nodes
        recomputed, how many matched their disk record, how many had
        none, and on a mismatch the first few offending nodes with the
        fix — the shortest path is naming the op, since a node that
        recomputes to different bytes is a dependency upgraded since the
        record was written, an op reading something other than its
        inputs and seed, or a non-deterministic kernel."""
        rows = self._recompute
        out = {
            "nodes": len(rows),
            "matched": sum(1 for r in rows if r.get("matched") is True),
            "unrecorded": sum(1 for r in rows if "matched" not in r),
        }
        bad = [r for r in rows if r.get("matched") is False]
        if bad:
            out["mismatched_total"] = len(bad)
            out["mismatched"] = self._sited([
                {k: r[k] for k in ("node", "op", "asset") if k in r} for r in bad[:5]
            ])
            first = bad[0]
            out["fix"] = (
                f"{len(bad)} node(s) recomputed to different bytes than their "
                f"cached records; the first is '{first['node']}' ({first['op']}) - "
                f"a dependency upgraded since the record was written (compare the "
                f"export manifest's kernel_versions with the installed ones), an op "
                f"reading anything but its inputs and explicit seed, or a "
                f"non-deterministic kernel: report the op with the asset. Its "
                f"downstream nodes take new keys and read as unrecorded, not "
                f"mismatched; the cached records were left in place as evidence"
            )
        return out

    def _note(self, sc: Asset, node: Node, *, hit: bool, ms: float) -> None:
        if hit:
            self._hits += 1
        else:
            self._misses += 1
        entry = {"node": node.id, "op": node.op, "ms": ms, "hit": hit}
        if sc.path is not None and sc.path != self.root.path:
            entry["asset"] = self._rel(sc.path)
        self._prof.append(entry)

    def _ref_site(self, sc: Asset) -> dict:
        """Detail locating a failing ref node's own asset: node ids are
        per-file, so nested references make bare `node` ambiguous."""
        if sc.path is not None and sc.path != self.root.path:
            return {"asset": self._rel(sc.path)}
        return {}

    def _decorate_ref_error(
        self, e: MoraineError, sc: Asset, node_id: str, **extra
    ) -> None:
        """Attach the failing ref node's site to an in-flight error
        (setdefault throughout: detail set closer to the failure wins)."""
        e.detail.setdefault("node", node_id)
        for k, v in extra.items():
            e.detail.setdefault(k, v)
        for k, v in self._ref_site(sc).items():
            e.detail.setdefault(k, v)

    def _eval_ref(self, sc: Asset, node: Node, op, cur_token: str = "") -> Record:
        """The reference path (ops declaring ``IS_REFERENCE``; the op module
        supplies ``ref_params`` / ``resolve_ref`` / ``override_token`` /
        ``rename_materials``, which today means ``asset_ref``)."""
        if sc.path is None:
            raise AssetError(
                "ref_no_project",
                "asset_ref needs an asset loaded from a file to resolve "
                "project-relative paths; save the asset first",
                node=node.id,
            )
        try:
            spec = op.ref_params(node.params)
        except MoraineError as e:
            self._decorate_ref_error(e, sc, node.id)
            raise
        token = op.override_token(spec.overrides)
        named = (sc.path, spec.asset, spec.project, token)
        target = self._targets.get(named)
        if target is None:
            try:
                target = op.resolve_ref(sc.path, spec)
            except MoraineError as e:
                self._decorate_ref_error(e, sc, node.id)
                raise
            self._targets[named] = target
        # Register before the existence check: a missing target must still
        # be watched, so creating the file triggers the next hot reload.
        self.asset_files.add(target)
        if target not in self._irs and not target.is_file():
            shown = target.relative_to(self._proj_root.parent).as_posix() if (
                self._proj_root is not None
                and target.is_relative_to(self._proj_root.parent)
            ) else target.name
            raise AssetError(
                "ref_not_found",
                f"no IR at '{shown}' for the asset this node references; "
                f"'moraine build' on the project writes it",
                node=node.id,
                path=shown,
                **self._ref_site(sc),
            )
        own = (target, token)
        if own not in self._assets and target not in self._stack:
            # The overrides only named the file (DESIGN 4.6): the
            # variant's own params hold the overridden values, and
            # applying them here would resolve the knob a second time.
            opened = self._irs.get(target)
            if opened is None:
                try:
                    opened = asset_mod.load(target)
                    asset_mod.check_stamp(
                        opened, self._stamp, self._rel(target),
                        source=lambda: self._source_of(target),
                    )
                except MoraineError as e:
                    self._decorate_ref_error(
                        e, sc, node.id, file=self._rel(target),
                    )
                    raise
            self._assets[own] = opened
            self._ref_overrides[own] = spec.overrides or {}
        if target in self._stack:
            chain = [self._rel(p) for p in self._stack] + [self._rel(target)]
            raise AssetError(
                "ref_cycle",
                f"asset references form a cycle: {' -> '.join(chain)}; "
                f"break the cycle (the reference graph must be a DAG)",
                chain=chain,
            )
        sub = self._assets[(target, token)]
        if spec.output not in sub.outputs:
            raise AssetError(
                "ref_unknown_output",
                f"asset '{self._rel(target)}' has no output '{spec.output}'; "
                f"available: {sorted(sub.outputs)}",
                node=node.id,
                **self._ref_site(sc),
            )
        self._ref_aliases[(sc.path, cur_token, node.id)] = (
            target, token, sub.outputs[spec.output],
        )
        self._stack.append(target)
        try:
            # Identity over the referenced output (plain refs alias its
            # record outright); location params never enter a cache key,
            # and overrides reach keys only as landed content (DESIGN 4.6).
            rec = self._eval_chain(sub, sub.outputs[spec.output], token)
        finally:
            self._stack.pop()
        if not spec.material_rename:
            return rec
        # A renamed ref is a one-input derived record over the referenced
        # output: keyed by its result hash plus the mapping (location and
        # override identity stay out, as for every ref), materialized and
        # cached like an op result. The referenced record itself is never
        # touched — other consumers of that output keep its name.
        key = _cache_key(
            node.op, op.OP_IMPL_VERSION, {"material_rename": spec.material_rename},
            [rec.result_hash],
        )

        def compute() -> Value:
            try:
                return op.rename_materials(rec.value(), spec.material_rename)
            except MoraineError as e:
                self._decorate_ref_error(
                    e, sc, node.id, ref=self._rel(target), output=spec.output
                )
                raise

        return self._materialize(sc, node, key, compute)


def evaluate_records(
    sc: Asset, only: str | None = None, *,
    source_of: Callable[[Path], Path | None], stamp: str,
) -> dict[str, Record]:
    """Evaluate named outputs; populate cache; return per-output records."""
    return Evaluation(sc, source_of=source_of, stamp=stamp).records(only)


def evaluate_asset(
    sc: Asset, only: str | None = None, *,
    source_of: Callable[[Path], Path | None], stamp: str,
) -> dict:
    """Evaluate named outputs; write output GLBs; return per-output info."""
    return Evaluation(sc, source_of=source_of, stamp=stamp).results(only)
