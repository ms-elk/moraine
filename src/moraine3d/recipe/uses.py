"""How an asset file uses its names, read from its text (DESIGN 7.2,
5.2).

``mo.param`` returns a plain value, and a value carries no mark of
where it came from once the recipe computes with it. Two questions are
therefore asked of the text, of the names a value was bound to and of
every place the file writes them. Whether a knob's value reached one
operator param whole and nothing else (``direct``, DESIGN 7.2). And
whether a threshold of the agent's own assert is built from a constant
that also builds the geometry it judges (``threshold_tautology``,
DESIGN 5.2). Both readings leave out what they cannot follow: a knob
used in any way the text does not settle is rebuilt, never patched,
and a threshold that reaches an assert as a function's argument or a
loop's variable draws no warning.

The text is read with Python's parser: a recipe keeps Python's syntax
(DESIGN 4.8), and the reading asks only where names are written, which
the two languages agree on. A file that parser cannot read is left
unread.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field

from ..core import asserts as vocab
from ..core.sources import REQUIREMENTS_EXT
from . import prelude

HOST = "mo"
KNOB = "param"
LOAD = "load"


@dataclass(frozen=True)
class Use:
    """One knob's one use: the operator call that takes the value
    whole, by the line the call is written on."""

    line: int
    op: str
    param: str


@dataclass(frozen=True)
class Shared:
    """One constant of the asset that judges what it builds: its name,
    the line that binds it, and the lines of the assert calls whose
    threshold is computed from it."""

    name: str
    line: int
    judged_at: tuple[int, ...]


@dataclass
class Reading:
    # By param id: the use of each knob whose value the file hands to
    # one operator param whole and writes nowhere else.
    whole: dict[str, Use] = field(default_factory=dict)
    # The constants a threshold and the geometry share, in the order
    # the file binds them.
    shared: list[Shared] = field(default_factory=list)


def _builtin(node: ast.AST) -> str | None:
    """The builtin's name where ``node`` is the call ``mo.<name>(...)``."""
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == HOST
    ):
        return node.func.attr
    return None


def _is_assert(node: ast.AST) -> bool:
    return (_builtin(node) or "").startswith(prelude.ASSERT_PREFIX)


def _knob_id(call: ast.Call) -> str | None:
    """The id a ``mo.param`` call declares, where the text states it."""
    given = call.args[0] if call.args else next(
        (k.value for k in call.keywords if k.arg == "id"), None,
    )
    if isinstance(given, ast.Constant) and isinstance(given.value, str):
        return given.value
    return None


def _bound(statement: ast.stmt) -> str | None:
    """The name a top-level statement binds, where it binds one."""
    if (
        isinstance(statement, ast.Assign) and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    return None


def _calls_made_once(tree: ast.Module):
    """The operator calls a file's top level makes exactly once: the
    call a top-level statement is, and the calls written as its
    arguments. A call under a loop, a condition, a function, a
    comprehension, or a list is not among them."""
    operators = set(prelude.operator_names())
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.Expr)):
            continue
        pending = [statement.value]
        while pending:
            call = pending.pop()
            if not isinstance(call, ast.Call):
                continue
            if _builtin(call) in operators:
                yield call
            pending.extend(call.args)
            pending.extend(k.value for k in call.keywords)


def parse(source: str) -> ast.Module | None:
    """The file as Python's parser reads it, or None where it cannot."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


def read(source: str) -> Reading | None:
    """What an asset file's text says of its names, or None for a file
    Python's parser cannot read."""
    tree = parse(source)
    if tree is None:
        return None
    text = _Text(tree)
    return Reading(whole=_whole(tree, text), shared=_shared(tree, text))


_SCOPES = (
    ast.FunctionDef, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp,
    ast.GeneratorExp,
)


def _locals(scope: ast.AST) -> frozenset[str]:
    """The names a function or a comprehension binds itself: its
    arguments, and what its body assigns."""
    return frozenset(
        n.arg if isinstance(n, ast.arg) else n.id
        for n in ast.walk(scope)
        if isinstance(n, ast.arg)
        or (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    )


@dataclass(frozen=True)
class _Read:
    """One place the file reads a name of its top level."""

    name: str
    statement: int  # the top-level statement it is written in
    judging: bool  # inside an assert's call
    threshold: int | None  # the assert's line, inside one of its thresholds


class _Text:
    """One pass over a file: every place a name is written, every read
    of a top-level name with what it is read for, and the builtins
    each top-level statement calls. A name a function binds itself is
    that function's, and a function called is no value read."""

    def __init__(self, tree: ast.Module):
        self.written: dict[str, int] = {}
        self.reads: list[_Read] = []
        self.builtins: dict[int, set[str]] = {}
        for index, statement in enumerate(tree.body):
            self._statement = index
            held = frozenset()
            if isinstance(statement, ast.For):
                held = frozenset(
                    n.id for n in ast.walk(statement.target) if isinstance(n, ast.Name)
                )
            self._visit(statement, held, False, None)

    def _count(self, name: str) -> None:
        self.written[name] = self.written.get(name, 0) + 1

    def _visit(self, node: ast.AST, held: frozenset, judging: bool,
               threshold: int | None) -> None:
        if isinstance(node, ast.Name):
            self._count(node.id)
            if isinstance(node.ctx, ast.Load) and node.id not in held and node.id != HOST:
                self.reads.append(_Read(node.id, self._statement, judging, threshold))
            return
        if isinstance(node, ast.arg):
            self._count(node.arg)
        if isinstance(node, _SCOPES):
            held = held | _locals(node)
        if isinstance(node, ast.Call):
            name = _builtin(node)
            if name is not None:
                self.builtins.setdefault(self._statement, set()).add(name)
            if name is not None and name.startswith(prelude.ASSERT_PREFIX):
                # Nothing an assert names builds: its outputs and its
                # faces are what it judges.
                for value in node.args:
                    self._visit(value, held, True, None)
                for keyword in node.keywords:
                    at = node.lineno if keyword.arg in vocab.THRESHOLDS else None
                    self._visit(keyword.value, held, True, at)
                return
            if isinstance(node.func, ast.Name):
                self._count(node.func.id)
                for value in node.args:
                    self._visit(value, held, judging, threshold)
                for keyword in node.keywords:
                    self._visit(keyword.value, held, judging, threshold)
                return
        for child in ast.iter_child_nodes(node):
            self._visit(child, held, judging, threshold)


# -- a knob handed whole to one param -------------------------------------------


def _whole(tree: ast.Module, text: _Text) -> dict[str, Use]:
    # The name each knob's value is bound to, where a top-level
    # statement binds it to one name.
    bound: dict[str, str] = {}
    for statement in tree.body:
        name = _bound(statement)
        if name is not None and _builtin(statement.value) == KNOB:
            pid = _knob_id(statement.value)
            if pid is not None:
                bound[name] = pid
    taken: dict[str, list[Use]] = {}
    for call in _calls_made_once(tree):
        for keyword in call.keywords:
            value = keyword.value
            if keyword.arg is None:
                continue  # a ** expansion
            use = Use(call.lineno, _builtin(call), keyword.arg)
            if isinstance(value, ast.Name) and value.id in bound:
                taken.setdefault(bound[value.id], []).append(use)
            elif _builtin(value) == KNOB and _knob_id(value) is not None:
                # The knob declared where it is used, bound to no name.
                taken.setdefault(_knob_id(value), []).append(use)
    names = {pid: name for name, pid in bound.items()}
    whole: dict[str, Use] = {}
    for pid, found in taken.items():
        name = names.get(pid)
        # The binding and the one use, and the name written nowhere else.
        if len(found) == 1 and (name is None or text.written.get(name) == 2):
            whole[pid] = found[0]
    return whole


# -- a constant that judges what it builds ---------------------------------------


def _loaded(tree: ast.Module) -> tuple[set[str], dict[str, int]]:
    """The names a file loads: those of its requirements file, the
    human's constants, which an asset builds against by design (DESIGN
    5.2), and those of ``lib/``, the agent's, each with the line that
    loads it."""
    human: set[str] = set()
    agent: dict[str, int] = {}
    for statement in tree.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not (
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == LOAD and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            continue
        names = {
            a.value for a in call.args[1:]
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        } | {k.arg for k in call.keywords if k.arg}
        if str(call.args[0].value).endswith(REQUIREMENTS_EXT):
            human |= names
        else:
            agent.update({name: statement.lineno for name in names})
    return human, agent


def _shared(tree: ast.Module, text: _Text) -> list[Shared]:
    if not any(read.threshold for read in text.reads):
        return []
    human, loaded = _loaded(tree)
    read_in: dict[int, set[str]] = {}
    for read in text.reads:
        read_in.setdefault(read.statement, set()).add(read.name)
    # The file's own constants: each top-level name and the names its
    # value is computed from, where the value is arithmetic alone (no
    # builtin's call in it).
    computed_from: dict[str, set[str]] = {name: set() for name in loaded}
    bound_at: dict[str, int] = dict(loaded)
    knobs: set[str] = set()
    arithmetic: set[int] = set()
    for index, statement in enumerate(tree.body):
        name = _bound(statement)
        if name is None:
            continue
        if _builtin(statement.value) == KNOB:
            knobs.add(name)
        plain = not text.builtins.get(index)
        if plain:
            arithmetic.add(index)
        if name in computed_from:
            # Bound twice: which value a use reads is not the text's to say.
            computed_from[name] = {name}
            continue
        computed_from[name] = read_in.get(index, set()) if plain else set()
        bound_at[name] = statement.lineno

    known = set(computed_from) | human
    found: dict[str, frozenset[str]] = {}

    def sources(name: str, seen: frozenset = frozenset()) -> frozenset[str]:
        """The constants a name's value goes back to. A value computed
        from the human's constants goes back to those: the arithmetic
        is translation, not authorship (DESIGN 5.2)."""
        if name in found:
            return found[name]
        under = set() if name in human else computed_from.get(name, set()) & known
        if name in seen or not under:
            return frozenset({name})
        out = frozenset().union(*(sources(u, seen | {name}) for u in under))
        if not seen:
            found[name] = out
        return out

    judged: dict[str, list[int]] = {}
    building: set[str] = set()
    for read in text.reads:
        if read.name not in computed_from:
            continue
        if read.threshold is not None:
            for source in sources(read.name):
                if source not in human and source not in knobs:
                    judged.setdefault(source, []).append(read.threshold)
        elif not read.judging and read.statement not in arithmetic:
            # What builds: a read outside every assert and outside the
            # arithmetic that only names another constant.
            building |= sources(read.name) - human
    return [
        Shared(name, bound_at[name], tuple(sorted(set(lines))))
        for name, lines in sorted(judged.items(), key=lambda item: bound_at[item[0]])
        if name in building
    ]
