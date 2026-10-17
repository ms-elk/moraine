"""What every kind's evaluation shares (DESIGN 5.2): one validate's
``Context`` (the memos every kind reads an output through, the group
resolution, the geometry roster), the entry conventions (honesty keys,
owner-ruled fix advice, the unchecked roster, the storage-noise floor),
the face-set readers, the dimension judgment, and the pair scope the
distance kinds start from.
"""

from __future__ import annotations

from ...core import mating
from ...core.asserts.common import NOISE_FLOOR
from ...core.mesh import Image, Value, proto_of
from ...core.predicates import FaceGroupError, resolve_face_group
from .. import bounds, depth, solids
from ..distance import (
    aabb_gap,
    exact_surface_distance,
    source_why,
    surface_distance,
    world_tri_source,
)

# A key a kind echoes into its entry header with no default: the parser
# guarantees it (KeyError = unparsed).
REQUIRED = object()

# The honesty keys: what a green entry still tells the reader — a scope
# it did not quantify (``unchecked``), pairs it ceded to a declared
# embed (``ceded``), a stronger mode that also holds (``stronger``).
# One roster: the sub-asset fold keeps entries carrying any of them and
# ``failures_only`` rolls each up by name, so a key added here is never
# silently folded away by one consumer and kept by another.
HONESTY_KEYS = ("unchecked", "ceded", "stronger")

# Failing items named in a fix sentence before it points at `detail`,
# which lists them all (token economy — the sentence says what failed,
# the data says all of it); `listed` applies it.
LISTED_COPIES = 10


def listed(items: list, render) -> str:
    """The fix sentence's roster: the first LISTED_COPIES ``items``
    rendered and joined, then "and N more (see detail)"."""
    shown = items[:LISTED_COPIES]
    text = "; ".join(render(x) for x in shown)
    if len(items) > len(shown):
        text += f"; and {len(items) - len(shown)} more (see detail)"
    return text


def refuse(entry: dict, why: str) -> None:
    """Fail an entry that cannot be evaluated, with the reason as its fix
    (DESIGN 5.2: an unevaluatable subject fails with the reason)."""
    entry["ok"] = False
    entry["fix"] = why


def owner_advice(entry: dict, repair: str, weaken: str, knob: str = "tol") -> str:
    """The fix advice's owner rule, one home for every assert kind
    (DESIGN 5.2): weakening the constraint (``weaken``) is offered only
    on an agent-owned entry; a failing human-owned assertion means fix
    the geometry (``repair``), never the ``knob`` — the error channel is
    the shortest path to a fix and must not steer the reader toward the
    forbidden edit."""
    if entry["owner"] == "agent":
        return f"{repair} or {weaken}"
    return (
        f"{repair} - a failing human-owned assertion means fix the geometry, "
        f"not the {knob}"
    )


def unchecked_instanced(entry: dict, key: str, names) -> None:
    """Name the instanced outputs an entry did not quantify over — the
    report says what it skipped, never a silent gap (one home for the
    three opt-in quantifiers and the enclosure check)."""
    names = list(names)
    if names:
        entry["unchecked"] = {key: names}


def misses(entry: dict, shortfall: float) -> bool:
    """Judge one length comparison under the storage-noise floor (DESIGN
    5.2): ``shortfall`` is how far the measurement misses its authored
    bound (positive = misses, zero or negative = meets). A miss under
    NOISE_FLOOR passes, and the entry records ``noise_floor`` so
    the pass is legible as one the floor decided — ``settle_floor``
    drops the key from an entry that fails on some other comparison."""
    if shortfall <= 0:
        return False
    if shortfall <= NOISE_FLOOR:
        entry["noise_floor"] = NOISE_FLOOR
        return False
    return True


def settle_floor(entry: dict) -> None:
    """Drop ``noise_floor`` from a failing entry (DESIGN 5.2): the floor
    only ever turns a miss into a pass, so a fail was decided elsewhere.
    A one-vs-rest clearance or an ``every`` contact makes many
    comparisons, and one waived partner must not read as the reason a
    failure meters wide is reported."""
    if "noise_floor" in entry and not entry.get("ok"):
        del entry["noise_floor"]


def face_label(spec) -> str:
    return f"{spec[0]}.{spec[1]}" + (f"[{spec[2]}]" if len(spec) > 2 else "")


def face_triangles(values: dict[str, Value], spec):
    """World triangles of one mated face ``[output, tag, instance?]``, or
    ``(None, error)`` — the shared resolver's refusal, whose message is
    the unevaluatable reason and whose code names it (tag presence, copy
    index, emptiness are evaluation facts)."""
    name, tag = spec[0], spec[1]
    inst = spec[2] if len(spec) > 2 else None
    try:
        tf = resolve_face_group(values[name], tag, inst, subject=f"output '{name}'")
    except FaceGroupError as e:
        return None, e
    return mating.group_triangles(tf.mesh, tf.faces, tf.matrix), None


def dimension_requirement(entry: dict, a: dict, default_tol) -> None:
    """Echo the judgment a dimension entry states (DESIGN 5.2): the
    resolved ``value``/``tol``, or the band with the omitted edge as
    ``null`` — the requirement rides every entry, evaluable or not, the
    way embedded's resolved band does."""
    if "value" in a:
        entry["value"] = a["value"]
        if default_tol is not None:
            tol = a.get("tol")
            entry["tol"] = default_tol if tol is None else tol
    else:
        entry["min"] = a.get("min")
        entry["max"] = a.get("max")


def dimension_verdict(entry: dict, measured: float, unit: str = "") -> str | None:
    """Judge ``measured`` against the entry's echoed requirement; the
    returned string names the miss (``None`` = within), in the entry's
    own numbers so the fix text and the report agree. A length (unit
    given) is judged under the storage-noise floor; a count is exact."""
    def under(shortfall: float) -> bool:
        return misses(entry, shortfall) if unit else shortfall > 0

    if "value" in entry:
        value, tol = entry["value"], entry.get("tol", 0)
        if not under(abs(measured - value) - tol):
            return None
        allowance = f" +/- {tol:.6g}{unit}" if tol else ""
        return (
            f"against value {value:.6g}{unit}{allowance} "
            f"({measured - value:+.4g}{unit} off)"
        )
    lo, hi = entry["min"], entry["max"]
    if not (lo is not None and under(lo - measured)) and not (
        hi is not None and under(measured - hi)
    ):
        return None
    if lo is None:
        return f"above max {hi:.6g}{unit}"
    if hi is None:
        return f"below min {lo:.6g}{unit}"
    return f"outside the band [{lo:.6g}, {hi:.6g}]{unit}"


def conclude(entry: dict, fix: list[str], detail: dict | None = None) -> None:
    """Settle an entry from its collected fix sentences: green when there
    are none, else the sentences joined and the detail rows attached."""
    entry["ok"] = not fix
    if fix:
        entry["fix"] = "; ".join(fix)
        if detail:
            entry["detail"] = detail


class Context:
    """One validate's shared state for every assert kind (DESIGN 5.2):
    the evaluated ``values``, the root output ``groups``, what declared
    kinds cede from the guard (``ceded``, collected by the dispatcher
    from the kinds' ``cedes`` hooks), and one memo per output for the
    refusal reason, the solid, the exact world bounds (united and per
    copy) and the triangle source, so a kind never rebuilds what
    another already built. ``proto_manifold`` and ``copy_bounds`` are
    the caller's memos shared with the enclosure check.
    """

    def __init__(
        self, values: dict[str, Value], *, proto_manifold=None, copy_bounds=None,
        groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.values = values
        self.proto_manifold = proto_manifold
        self._copy_bounds_of = copy_bounds
        self.groups = groups or {}
        # (cross-output pairs, outputs whose copy pairs) declared kinds
        # police instead of the guard — filled before any check runs.
        self.ceded: tuple[set[frozenset], set[str]] = (set(), set())
        self._refusals: dict[str, str | None] = {}
        self._manifolds: dict[str, tuple] = {}
        self._tri_sources: dict[str, tuple] = {}
        self._bounds: dict[str, tuple] = {}
        self._copy_bounds: dict[str, tuple] = {}
        # The asset's geometry outputs: the guard's pair universe and the
        # partner set of a one-vs-rest clearance (images are not parts).
        self.geometry = sorted(n for n, v in values.items() if not isinstance(v, Image))

    def members(self, name: str) -> list[str]:
        """A group's members; an output is its own one member."""
        return self.groups.get(name, [name])

    def refusal(self, name: str) -> str | None:
        """Why an output cannot be measured (image, empty, non-finite),
        or None — the predicates alone, no triangle gather."""
        if name not in self._refusals:
            self._refusals[name] = source_why(self.values[name], name)
        return self._refusals[name]

    def solid(self, name: str):
        """World-space solid of an output or a group, or ``(None, why)``."""
        if name not in self._manifolds:
            if name in self.groups:
                # A group's solid is its members' union; the first member
                # that cannot be built is the group's reason.
                parts = []
                for m in self.groups[name]:
                    s, why = self.solid(m)
                    if why is not None:
                        self._manifolds[name] = (None, why)
                        break
                    parts.append(s)
                else:
                    import manifold3d

                    self._manifolds[name] = (
                        manifold3d.Manifold.batch_boolean(parts, manifold3d.OpType.Add),
                        None,
                    )
                return self._manifolds[name]
            v = self.values[name]
            if self.proto_manifold is not None and not isinstance(v, Image):
                self._manifolds[name] = solids.world_manifold(v, name, self.proto_manifold(name))
            else:
                self._manifolds[name] = solids.world_manifold(v, name)
        return self._manifolds[name]

    def world_bounds(self, name: str):
        # One bounds memo and one bounds source for every assert kind:
        # exact world AABBs, sound for disjointness culling and for the
        # distance path's gap/cull boxes alike.
        if name not in self._bounds:
            self._bounds[name] = bounds.world_bounds(self.values[name])
        return self._bounds[name]

    def copy_bounds(self, name: str):
        """Exact per-copy world AABBs ``(lo (k, 3), hi (k, 3))`` of an
        instanced output."""
        if name not in self._copy_bounds:
            if self._copy_bounds_of is not None:
                self._copy_bounds[name] = self._copy_bounds_of(name)
            else:
                self._copy_bounds[name] = bounds.copy_world_bounds(self.values[name])
        return self._copy_bounds[name]

    def under_bar(self, na: str, nb: str) -> bool:
        (lo_a, hi_a), (lo_b, hi_b) = self.world_bounds(na), self.world_bounds(nb)
        return bool(depth.under_depth_bar(lo_a, hi_a, lo_b, hi_b))

    def tri_source(self, name: str):
        if name not in self._tri_sources:
            self._tri_sources[name] = world_tri_source(self.values[name], name)
        return self._tri_sources[name]

    def proto_solid(self, name: str):
        if self.proto_manifold is not None:
            return self.proto_manifold(name)
        mesh = proto_of(self.values[name])
        return solids.to_manifold(mesh.vertices, mesh.faces)


class PairScope:
    """The subjects a distance kind measures between, resolved once
    (DESIGN 5.2): ``outputs[0]`` against its partners — the one other
    name, every other geometry output in the one-vs-rest form
    (``others``, echoed with the partner count), or none in clearance's
    copy form — with a name standing for the union of its group
    members. ``srcs`` and ``bounds`` are the
    members' triangle sources and exact boxes; ``why`` is set when a
    member has no source (the entry fails with that reason). ``scope``
    is the ``self`` quantifier's range — both names of a pair, the
    subject alone in the one-vs-rest form — and ``inst`` its instanced
    outputs. The pair helpers measure between two names member-pair by
    member-pair, each pair under its own proven bound (a union's bound
    is not a member's). ``tol`` is the distance the kind judges by:
    the entry's ``key``, or ``default_tol`` where the kind has one.
    """

    def __init__(
        self, ctx: Context, entry: dict, a: dict, default_tol: float | None,
        key: str = "tol",
    ) -> None:
        self.ctx = ctx
        # An absent tol is the kind's default (the parser admits null too).
        tol = a.get(key)
        self.tol = default_tol if tol is None else tol
        self.why: str | None = None
        if self.tol is None:
            # Refused by the parser; an entry that reaches evaluation
            # without it still fails as one entry, never as a traceback.
            self.why = f"'{key}' is missing: the parser requires it on this kind"
        outs = entry["outputs"]
        self.name_a = outs[0]
        self.mem_a = ctx.members(self.name_a)
        # The one-vs-rest form (clearance only, DESIGN 5.2): the partners are
        # every other geometry output the asset has now, so an output added
        # later is covered instead of being a pair nobody wrote; a group's
        # own members are the piece, never its partners.
        self.others_mode = bool(a.get("others"))
        self.partners = (
            [n for n in ctx.geometry if n not in self.mem_a] if self.others_mode else outs[1:]
        )
        if self.others_mode:
            entry["others"] = True
            entry["pairs"] = len(self.partners)
        names = [self.name_a, *self.partners]
        self.flat = list(dict.fromkeys(m for n in names for m in ctx.members(n)))
        self.srcs: dict = {}
        if self.why is not None:
            return
        for n in self.flat:
            src, why = ctx.tri_source(n)
            if why is not None:
                self.why = why
                return
            self.srcs[n] = src
        self.bounds = {n: ctx.world_bounds(n) for n in self.flat}
        # `self`'s scope: both outputs of a pair; the named output alone in
        # the one-vs-rest form (its partners are the rest of the asset, and
        # clearance against them already holds per copy).
        self.scope = self.mem_a if self.others_mode else self.flat
        self.inst = [n for n in self.scope if self.srcs[n][1] is not None]
        self.name_b = self.partners[0] if self.partners else None

    def pair_broad(self, x: str, y: str) -> dict[tuple[str, str], float | None]:
        """The broad-phase result per member pair of two names."""
        return {
            (p, q): surface_distance(self.srcs[p], self.bounds[p], self.srcs[q], self.bounds[q], self.tol)
            for p in self.ctx.members(x) for q in self.ctx.members(y)
        }

    @staticmethod
    def pair_best(broad: dict) -> float | None:
        found = [d for d in broad.values() if d is not None]
        return min(found) if found else None

    def pair_gap(self, x: str, y: str) -> float:
        return min(
            aabb_gap(*self.bounds[p], *self.bounds[q])
            for p in self.ctx.members(x) for q in self.ctx.members(y)
        )

    def pair_exact(self, broad: dict) -> float:
        """The exact closest approach over member pairs, each pair under
        its own proven bound (a union's bound is not a member's)."""
        return min(
            exact_surface_distance(self.srcs[p], self.bounds[p], self.srcs[q], self.bounds[q], upper=up)
            for (p, q), up in broad.items()
        )
