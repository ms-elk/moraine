"""How much a report lists before it only counts (DESIGN 8.2): a
report's cost follows the answer, never the size of the asset.
"""

from __future__ import annotations

# Names, ids, or rows a list holds; what is past them is counted.
LISTED = 10
