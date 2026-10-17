"""The ``validate`` report of an evaluated asset (DESIGN 6), assembled
in one place: the command, the export gate, and the check an accept
runs read the same report.
"""

from __future__ import annotations

from ..core import params as params_mod
from ..core.asset import Asset
from ..verify.validate import (
    VERDICT_CHANNELS, append_ref_asserts, validate_outputs,
)


def report(sc: Asset, ev) -> tuple[dict, dict]:
    """The report of ``sc`` as ``ev`` evaluated it, before any export
    profile's checks, and the values it judged, by output."""
    values = {name: rec.value() for name, rec in ev.records().items()}
    out = validate_outputs(
        values,
        checks_policy=sc.checks,
        warnings=ev.attribute_drop_warnings() + ev.shading_warnings(),
        asserts=sc.asserts,
        asset_checks=sc.asset_checks,
        budgets=sc.budgets,
        origins=ev.material_origins(),
        ref_embeds=ev.embedded_via_refs(),
        groups=sc.groups,
        params=params_mod.range_entries(sc),
        sites=ev.output_sites(),
    )
    return append_ref_asserts(out, ev.ref_asserts()), values


def moves_no_entry(knob: dict) -> bool:
    """Whether a pick of this knob alone can change no entry of the
    report (DESIGN 7.2), read off its schema entry: a `factor`-tier
    knob on a channel no verdict reads. A knob whose channel is not
    named is taken to move one."""
    channel = (knob.get("uniform") or {}).get("channel")
    return (
        knob.get("tier") == "factor"
        and channel is not None and channel not in VERDICT_CHANNELS
    )


# Who owns a failing entry of each gating section that carries no
# owner of its own (DESIGN 5.2): a budget is the human's acceptance
# criterion, a knob's range the declaration the agent wrote, and a
# check one every asset is held to.
_OWNERS = {"budgets": "human", "params": "agent"}
BUILTIN = "builtin"


def failing(report: dict) -> list[dict]:
    """The failing gating entries of a report, each by its id and its
    owner, human-owned first (DESIGN 7.2): what an accept's check
    sends."""
    rows: list[dict] = []

    def row(section: str, entry: str, owner: str, source: dict, **more) -> None:
        found = {"section": section, "entry": entry, "owner": owner, **more}
        for key in ("asset", "site"):
            if key in source:
                found[key] = source[key]
        rows.append(found)

    for name, out in (report.get("outputs") or {}).items():
        for check in out.get("checks", []):
            if not check.get("ok") and check.get("level") != "warn":
                row("checks", check["check"], BUILTIN, check, output=name)
    for check in report.get("asset_checks") or []:
        if not check.get("ok"):
            row("asset_checks", check["check"], BUILTIN, check)
    for entry in report.get("asserts") or []:
        if not entry.get("ok"):
            row("asserts", entry["assert"], entry.get("owner", "agent"), entry)
    for entry in report.get("budgets") or []:
        if not entry.get("ok"):
            row("budgets", entry["budget"], _OWNERS["budgets"], entry)
    for entry in report.get("params") or []:
        if not entry.get("ok"):
            row("params", entry["param"], _OWNERS["params"], entry)
    order = {"human": 0, "agent": 1, BUILTIN: 2}
    return sorted(rows, key=lambda r: order[r["owner"]])
