"""The assert vocabulary's two registries (DESIGN 3): one kind, one
module on each side of the layer line — ``core/asserts`` (parse-time
shape and defaults) and ``verify/asserts`` (evaluation) — kept equal
to each other and to the files in their packages by these tests, and
each kind held to the contract its registry's docstring states.
"""

from __future__ import annotations

import json

import pytest

from conftest import SRC_ROOT
from moraine3d.core import asserts as parse_side
from moraine3d.core import asset as asset_mod
from moraine3d.errors import AssetError
from moraine3d.verify import asserts as check_side


def _kind_modules(package: str) -> set[str]:
    """The kind modules in a package: every module but the registry
    root and the shared helpers."""
    return {
        p.stem for p in (SRC_ROOT / package).glob("*.py")
        if p.stem not in ("__init__", "common")
    }


def _authorable(mod) -> bool:
    return getattr(mod, "AUTHORABLE", True)


def test_the_two_registries_mirror_each_other_and_their_files():
    """DESIGN 3: a kind is added by two files and two registry lines —
    the parse side and the evaluation side name the same kinds, each
    names exactly the kind modules its package holds, and a kind's
    module is the one named after it."""
    assert set(parse_side.KINDS) == set(check_side.KINDS)
    assert set(parse_side.KINDS) == _kind_modules("core/asserts")
    assert set(check_side.KINDS) == _kind_modules("verify/asserts")
    for kind, mod in parse_side.KINDS.items():
        assert mod.__name__ == f"moraine3d.core.asserts.{kind}"
    for kind, mod in check_side.KINDS.items():
        assert mod.__name__ == f"moraine3d.verify.asserts.{kind}"


def test_every_kind_declares_its_contract():
    """The kind-module contract each registry's docstring states: on the
    parse side ``KEYS`` drawn from the entry keys with a ``KEY_MEANING``
    for each, a ``SUBJECT`` phrase, ``REFUSALS`` only for keys the kind
    refuses, ``validate_entry`` and ``TOL_UNIT`` wherever ``tol`` is
    accepted (the unit the parser's number message names — a kind must
    not admit ``tol`` without one) on every authorable kind, and on a
    built-in kind ``ENTRY`` (its reserved id, its own kind, marked
    ``builtin``) and ``REFUSAL``; on the evaluation side ``check``, and
    an ``ECHOED`` key is one the parser accepts on that kind."""
    entry_keys = parse_side.KEYS - {"id", "kind", "owner"}
    for kind, mod in parse_side.KINDS.items():
        if not _authorable(mod):
            assert isinstance(mod.REFUSAL, str) and mod.REFUSAL, kind
            assert mod.ENTRY["kind"] == kind and mod.ENTRY["builtin"] is True, kind
            assert mod.ENTRY["id"] in parse_side.RESERVED_IDS, kind
            continue
        assert isinstance(mod.KEYS, frozenset) and mod.KEYS <= entry_keys, kind
        assert set(mod.KEY_MEANING) == set(mod.KEYS), kind
        assert isinstance(mod.SUBJECT, str) and mod.SUBJECT, kind
        assert not set(getattr(mod, "REFUSALS", {})) & mod.KEYS, kind
        assert callable(mod.validate_entry), kind
        if "tol" in mod.KEYS:
            assert isinstance(mod.TOL_UNIT, str) and mod.TOL_UNIT, kind
    assert parse_side.AUTHORABLE_KINDS == {
        k for k, m in parse_side.KINDS.items() if _authorable(m)
    }
    assert len(parse_side.BUILTIN_ENTRIES) == len(parse_side.RESERVED_IDS) >= 1
    for kind, mod in check_side.KINDS.items():
        assert callable(mod.check), kind
        if _authorable(parse_side.KINDS[kind]):
            assert set(getattr(mod, "ECHOED", {})) <= parse_side.KINDS[kind].KEYS, kind


# One well-formed entry per authorable kind over two declared, tagged
# outputs — the base the per-kind probes below vary one key of.
_MINIMAL = {
    "contact": {"outputs": ["a", "b"]},
    "clearance": {"outputs": ["a", "b"], "min": 0.01},
    "mated": {"faces": [["a", "top"], ["b", "top"]]},
    "embedded": {"outputs": ["a", "b"]},
    "extent": {"outputs": ["a"], "measure": "size_x", "value": 1.0},
    "count": {"outputs": ["a"], "value": 1},
}


def _asset(entry: dict) -> str:
    return json.dumps({
        "build_version": "fixture",
        "nodes": [
            {"id": "b1", "op": "box", "params": {"size": [1.0, 1.0, 1.0]}},
            {"id": "g1", "op": "tag", "inputs": ["b1"],
             "params": {"name": "top", "select": {"normal_y": {"gt": 0.9}}}},
            {"id": "t1", "op": "transform", "inputs": ["g1"],
             "params": {"translate": [3.0, 0.0, 0.0]}},
        ],
        "outputs": {"a": "g1", "b": "t1"},
        "asserts": [entry],
    })


def test_every_kind_gates_its_number_and_flag_keys():
    """DESIGN 5.2: on every kind, a ``tol`` is a finite number >= 0,
    clearance's ``min`` a finite number > 0, and a switch key (``others``, ``self``, ``connected``, ``every``) is
    refused when written as anything but its one spelling — the gates
    the old central parse loop applied by key presence, now each
    kind's own call, so a kind that admits a key without gating it is
    caught here rather than at evaluation."""
    assert set(_MINIMAL) == parse_side.AUTHORABLE_KINDS
    for kind, base in _MINIMAL.items():
        keys = parse_side.KINDS[kind].KEYS
        entry = {"id": "x", "kind": kind, "owner": "agent", **base}
        asset_mod.loads(_asset(entry))  # the base parses
        if "tol" in keys:
            with pytest.raises(AssetError) as exc:
                asset_mod.loads(_asset({**entry, "tol": -1}))
            assert "finite number >= 0" in str(exc.value), kind
        if kind == "clearance":
            for gap in (0, -1):
                with pytest.raises(AssetError) as exc:
                    asset_mod.loads(_asset({**entry, "min": gap}))
                assert "finite number > 0" in str(exc.value)
        for key in keys & {"others", "self", "connected", "every"}:
            with pytest.raises(AssetError) as exc:
                asset_mod.loads(_asset({**entry, key: "yes"}))
            assert exc.value.code == "bad_assert", (kind, key)


def test_key_refusals_name_the_owners_from_the_modules():
    """A key refused on one kind names the kinds that take it, with what
    it means on each, from the modules' own KEY_MEANING — so the error
    channel cannot lag the vocabulary when a kind starts accepting a
    key (DESIGN 8.2: errors state the shortest path to a fix)."""
    assert parse_side.key_refusal("count", "every") == (
        "'every' is only for contact (every copy of the named instanced "
        "outputs must touch the other output), mated (every copy of the "
        "named instanced outputs, paired by copy index), and extent (each "
        "copy measured on its own)"
    )
    assert parse_side.key_refusal("contact", "min") == (
        "'min' is a bound for clearance (the least surface distance, m), "
        "embedded (penetration depth, m), extent (m), and count (copies); "
        "contact takes 'tol'"
    )
    for kind, mod in parse_side.KINDS.items():
        if not _authorable(mod):
            continue
        for key in parse_side.KEYS - {"id", "kind", "owner"} - mod.KEYS:
            text = parse_side.key_refusal(kind, key)
            owners = [k for k, m in parse_side.KINDS.items() if key in getattr(m, "KEYS", ())]
            own = getattr(mod, "REFUSALS", {})
            if key not in own and kind != "mated":
                assert all(o in text for o in owners), (kind, key, text)


def test_clearance_refuses_every_with_its_own_reason():
    """DESIGN 5.2: on clearance `every` is rejected and `self` is the
    copy-pair form — the refusal is the kind's own explanation (the
    union minimum already is the per-copy minimum), not the generic
    list of the kinds that take the key."""
    entry = {"id": "x", "kind": "clearance", "outputs": ["a", "b"],
             "every": ["a"], "owner": "agent", "min": 0.001}
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_asset(entry))
    assert exc.value.code == "bad_assert"
    msg = str(exc.value)
    assert "'every' does not apply to clearance" in msg
    assert "per-copy minimum" in msg and '"self": true' in msg
    assert "only for" not in msg


def test_contact_every_names_are_drawn_from_the_entry_outputs():
    """DESIGN 5.2: contact's `every` names are drawn from the entry's
    `outputs`, so the key says whose copies — a name outside them is
    refused with the outputs it had to choose from."""
    entry = {"id": "x", "kind": "contact", "outputs": ["a", "b"],
             "every": ["not_an_output"], "owner": "agent"}
    with pytest.raises(AssetError) as exc:
        asset_mod.loads(_asset(entry))
    assert exc.value.code == "bad_assert"
    msg = str(exc.value)
    assert "drawn from its 'outputs'" in msg
    assert "['a', 'b']" in msg
