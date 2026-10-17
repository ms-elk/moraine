"""The recipe position an IR entry carries (DESIGN 4.1): its ``site``,
the file and line of the call that made it, and for an entry made
inside a function its ``via``, the calls that were on the way.
Location, not content: neither joins a cache key.
"""

from __future__ import annotations


def where(holder) -> dict:
    """The position of a node or an entry of the IR, as the keys a
    report row takes: ``site``, and ``via`` where there is one."""
    if isinstance(holder, dict):
        site, via = holder.get("site"), holder.get("via")
    else:
        site, via = holder.site, holder.via
    out: dict = {}
    if site is not None:
        out["site"] = site
        if via:
            out["via"] = list(via)
    return out
