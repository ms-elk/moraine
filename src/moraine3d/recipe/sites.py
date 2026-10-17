"""Call sites, read from the interpreter's call stack (DESIGN 4.8).

The interpreter returns the stack as text written for a human reader
and documents it as neither stable nor meant to be parsed. This module
is the one place that parses it, and the one place that knows its
shape; a test freezes that shape, so a change fails the suite when the
pin moves and never a build.

The shape, one frame per pair of lines under a heading line::

    Traceback (most recent call last):
      * <file>:<line>, in <function>
          <the source line>

A frame names the call it holds: the file and line where the call is
written, and the function whose body that line belongs to
(``<module>`` at a file's top level). The last frame is the innermost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..errors import BuildError

_FRAME = re.compile(r"^  \* (.+):(\d+), in (.+)$", re.M)
# The span line of an interpreter error: ``--> file:line:column``.
_SPAN = re.compile(r"^ *--> (.+):(\d+):(\d+)$", re.M)

# The function name of a frame at a file's top level.
MODULE = "<module>"


@dataclass(frozen=True)
class Frame:
    file: str
    line: int
    function: str

    @property
    def site(self) -> str:
        return f"{self.file}:{self.line}"

    @property
    def at_top_level(self) -> bool:
        return self.function == MODULE


def frames(stack: str) -> list[Frame]:
    """The recipe frames of a stack, outermost first. Frames of files
    the build names itself (``<...>``) are not recipe frames."""
    return [
        Frame(m[1], int(m[2]), m[3])
        for m in _FRAME.finditer(stack)
        if not m[1].startswith("<")
    ]


def read(stack: object, builtin: str) -> list[Frame]:
    """The frames of a builtin's call, or the build's failure: a node
    or an assert whose site cannot be read is never written."""
    found = frames(stack) if isinstance(stack, str) else []
    if not found:
        raise BuildError(
            "site_unreadable",
            f"mo.{builtin} could not read where it was called from; the "
            f"build writes no entry without its recipe site - report this "
            f"with the recipe line, it is a fault of the build",
            builtin=builtin,
        )
    return found


def span(text: str) -> str | None:
    """The ``file:line`` an interpreter error points at, if it names
    one."""
    m = _SPAN.search(text)
    return f"{m[1]}:{m[2]}" if m else None


class Where(str):
    """The site of a call, which also knows the calls it was reached
    through (DESIGN 4.8): ``via`` holds the sites of the outer frames,
    outermost first, and is empty for a call at a file's top level.
    A string, so whatever names a site names it unchanged."""

    via: tuple[str, ...] = ()

    def __new__(cls, site: str, via=()):
        self = super().__new__(cls, site)
        self.via = tuple(via)
        return self

    @property
    def keys(self) -> dict:
        """The position as an IR entry carries it."""
        out: dict = {"site": str(self)}
        if self.via:
            out["via"] = list(self.via)
        return out


def where(found: list[Frame], limit: int = 6) -> Where:
    """The position of a builtin's call: the innermost frame's site,
    and the outer frames' as ``via``, a long run cut in the middle as
    ``chain`` cuts it."""
    outer = [f.site for f in found[:-1]]
    if len(outer) > limit:
        half = limit // 2
        outer = outer[:half] + [f"... {len(outer) - limit} more"] + outer[-half:]
    return Where(found[-1].site, outer)


def chain(found: list[Frame], limit: int = 6) -> list[str]:
    """A call chain for a report: the sites outermost first, a long
    run cut in the middle so the two ends stay readable."""
    sites = [f"{f.site} in {f.function}" for f in found]
    if len(sites) <= limit:
        return sites
    half = limit // 2
    return sites[:half] + [f"... {len(sites) - limit} more"] + sites[-half:]
