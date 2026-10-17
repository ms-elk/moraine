"""IR fixtures written by hand (DESIGN 4.1)."""

from __future__ import annotations


def with_knob_values(doc: dict) -> dict:
    """``doc`` with each params entry that names a node param carrying
    that param's value, as the build records it (DESIGN 4.1). A fixture
    states the binding once and the value follows the node."""
    nodes = {n["id"]: n for n in doc.get("nodes", [])}
    for decl in doc.get("params") or []:
        if "value" not in decl and "node" in decl:
            decl["value"] = nodes[decl["node"]]["params"][decl["param"]]
    return doc
