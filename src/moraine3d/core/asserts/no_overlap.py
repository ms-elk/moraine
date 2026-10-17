"""``no_overlap``, the built-in overlap guard (DESIGN 5.2): never
authored — every validate checks all output pairs and copy pairs for
material interpenetration against the depth bar (common.DEPTH_BAR),
minus the pairs a declared ``embedded`` cedes. The registry injects
``ENTRY`` into every validate under its reserved id.
"""

from __future__ import annotations

AUTHORABLE = False
# The parser's answer to an authored entry of this kind.
REFUSAL = (
    "the overlap guard is built in - every validate already checks all "
    "output pairs and copy pairs, and the guard holds this requirement at "
    "owner 'human', so removing the entry needs no approval round; remove "
    "it, and declare an intended overlap as an 'embedded' assert"
)
# The built-in overlap guard (DESIGN 5.2): every validate checks all
# output pairs and copy pairs for material interpenetration. Human-owned
# by spec assignment — "no part may interpenetrate another" is the
# standing human requirement the field states universally — so
# superseding it is a human-approval edit, while a declared `embedded`
# (an agent-writable substituted checkable claim) cedes its pairs
# without one. Its id is reserved; its entry carries `builtin: true`.
ENTRY = {
    "id": "no_overlap", "kind": "no_overlap", "outputs": "all",
    "self": True, "owner": "human", "builtin": True,
}
