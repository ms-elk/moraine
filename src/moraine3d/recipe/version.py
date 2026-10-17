"""``build_version`` (DESIGN 4.1, 4.4): the stamp of the build's
contract, written at the root of every IR and refused on mismatch.

Three parts. The contract number moves by hand with the id
derivation, the resolution rules, and the IR's shape. The interpreter
distribution's version follows the pin. The signature is a hash of the
generated prelude, so a builtin added, removed, or renamed moves the
stamp without anyone remembering to. The tool's own version is not
part of it: a release that changes nothing a recipe means leaves every
IR current.
"""

from __future__ import annotations

import functools
import hashlib

from .. import dist_version
from ..core import asset as asset_mod
from . import dialect, prelude

# Bump on any change to what a recipe means that the prelude's text
# does not show: the id derivation, a knob's resolution order, the
# load rules, the IR's keys.
CONTRACT = 7


@functools.cache
def build_version() -> str:
    signed = "\n".join([
        prelude.source(),
        f"call_depth={dialect.CALL_DEPTH_CAP}",
        "ir=" + ",".join(asset_mod.ROOT_KEY_ORDER),
    ])
    signature = hashlib.sha256(signed.encode("utf-8")).hexdigest()[:8]
    return f"{CONTRACT}+{dialect.INTERPRETER}.{dist_version(dialect.INTERPRETER)}+{signature}"
