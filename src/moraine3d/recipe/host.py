"""The host side of ``mo`` (DESIGN 4.8): what each builtin records.

One ``Session`` holds one asset's declarations while its recipe runs.
The build is two-phase: a call records, and names resolve once every
declaration is in (``document``), so order inside a file matters to
values and not to names. Every entry carries the site of the call that
made it, and an error names that site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import starlark as sl

from .. import ops
from ..core import asserts as vocab
from ..core import numbers
from ..core import params as params_mod
from ..core.brief import LISTED
from ..core import asset as asset_mod
from ..errors import BuildError, MoraineError
from . import prelude, sites, uses
from .sites import Frame

# What a file is to the build, by where it sits (DESIGN 4.8, 5.2).
ASSET = "asset"
REQUIREMENTS = "requirements"
LIB = "lib"


def _same(a, b) -> bool:
    """Whether two values are one value in one number type, so that
    writing either writes the same IR."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class Handle:
    """What an operator call returns: the node, to the recipe an opaque
    value it can pass on and nothing else."""

    __slots__ = ("id", "named")

    def __init__(self, id: str, named: bool):
        self.id = id
        self.named = named


class KnobValue:
    """What ``mo.param_value`` returns (DESIGN 4.8): one of the
    asset's knobs, by its id, to stand whole as a threshold. It holds
    no number, since the knob is resolved only once the asset has
    declared it."""

    __slots__ = ("id", "site")

    def __init__(self, id: str, site: str):
        self.id = id
        self.site = site


WHOLE = (
    "it stands whole as a threshold of an assert in the requirements "
    f"file ({', '.join(vocab.THRESHOLDS[:-1])}, or {vocab.THRESHOLDS[-1]}) "
    "and nowhere else"
)


@dataclass
class Reference:
    """One ``asset_ref`` call, resolved by the build once the recipe
    has run (DESIGN 4.6)."""

    node: str
    site: str
    asset: object  # the recipe path as written, project-relative
    project: object
    output: object
    overrides: object


@dataclass
class Session:
    asset_file: str  # the asset file's name in sites
    requirements_file: str  # its requirements file's, whether or not it exists
    offered: dict = field(default_factory=dict)
    offered_site: str | None = None
    # The values the human accepted for this asset's knobs, from the
    # project's params.json (DESIGN 7.2); empty for a sub-asset, whose
    # knobs take a value from their consumer alone.
    accepted: dict = field(default_factory=dict)
    # Whether an offered value for a knob the asset does not declare
    # fails the build. A reference's override does; the viewer's
    # overlay does not, since the file wins over a value offered for a
    # knob an edit removed (DESIGN 7.2).
    strict_offers: bool = True
    # Whether an offered value outside the knob's declared range fails
    # the build. A consumer's does; a read command's does not, since
    # its report fails the knob's entry and still judges the shape
    # (DESIGN 6).
    ranged_offers: bool = True

    def __post_init__(self):
        self.roles: dict[str, str] = {}
        self.nodes: dict[str, dict] = {}
        self.asserts: list[dict] = []
        self.params: list[dict] = []
        self.outputs: list[tuple[str, Handle, str]] = []
        self.groups: list[tuple[str, object, str]] = []
        self.waivers: list[tuple[object, object, str]] = []
        self.asset_waivers: list[tuple[object, str]] = []
        self.budgets: list[tuple[str, object, str]] = []
        self.provenance: tuple[dict, str] | None = None
        self.references: list[Reference] = []
        self.accepted_offers: dict = {}
        # The thresholds that follow a knob, each with its entry: the
        # knob is resolved once every declaration is in.
        self.following: list[tuple[dict, dict[str, KnobValue], str]] = []
        # Every handle mo.param_value returned: one that reaches no
        # threshold was read some other way, which tells the file nothing.
        self.handles: list[KnobValue] = []
        self._unnamed: dict[str, int] = {}
        self._assert_ids: dict[str, str] = {}
        self._param_ids: set[str] = set()
        # The warnings the document raises (DESIGN 5.2), framed as its
        # entries are.
        self.warnings: list[dict] = []
        # The refusal a host function raised. The interpreter reports a
        # host exception as text, so the error itself is kept here and
        # the build raises it in place of the text.
        self.error: BuildError | None = None

    # -- errors ---------------------------------------------------

    def fail(self, code: str, message: str, at: str | None, **detail) -> BuildError:
        """The build's refusal, at a call's position: its site, and
        the calls it was reached through where it has any."""
        if at is not None:
            detail["site"] = str(at)
            if getattr(at, "via", ()):
                detail["via"] = list(at.via)
        self.error = BuildError(code, message, **detail)
        return self.error

    def _from(self, e: MoraineError, at: str | None) -> BuildError:
        detail = dict(e.detail)
        return self.fail(e.code, e.message, at, **detail)

    def _frames(self, stack: object, builtin: str) -> list[Frame]:
        """The frames of a builtin's call. The interpreter reports a
        host exception as text, so the refusal is kept like any other."""
        try:
            return sites.read(stack, builtin)
        except BuildError as e:
            self.error = e
            raise

    # -- fail() ---------------------------------------------------

    def failed(self, stack, /, *args, **kw):
        """The language's ``fail`` (DESIGN 4.8). At a file's top level
        it is a check script in one line, refused for the reason a
        check script was: it would stop the build with one message
        where a report carries every finding. Inside a function it is
        the function's own refusal of what it was handed, and fails
        the build with its message and the calls on the way."""
        found = self._frames(stack, prelude.FAIL)
        at = sites.where(found)
        kw = dict(kw)
        sep = kw.pop("sep", " ")
        words = [str(kw.pop("msg"))] if "msg" in kw else []
        words += [str(a) for a in args]
        words += [f"{k} = {v}" for k, v in kw.items()]
        said = (sep if isinstance(sep, str) else " ").join(words) or "no message"
        if found[-1].at_top_level:
            raise self.fail(
                "fail_at_top_level",
                f"fail() at a file's top level stops the build with one "
                f"message where the report carries every finding; state "
                f"the requirement as an assert, mo.assert_<kind>(...), "
                f"which the report judges beside every other entry (the "
                f"message was: {said})",
                at,
            )
        raise self.fail(
            "recipe_failed",
            f"'{found[-1].function}' refused what it was called with: {said}",
            at, function=found[-1].function,
        )

    # -- the calling file -----------------------------------------

    def _role(self, found: list[Frame]) -> str:
        """What the file is whose top level is running. A function
        called from an asset runs for the asset wherever it was
        defined, so the outermost frame decides."""
        return self.roles.get(found[0].file, ASSET)

    def _refuse_in_lib(self, found: list[Frame], builtin: str) -> None:
        if self._role(found) == LIB:
            raise self.fail(
                "lib_not_definitions",
                f"'{found[0].file}' calls mo.{builtin} at its top level; a "
                f"lib/ file holds definitions only - put the call inside a "
                f"function and call that function from the asset",
                sites.where(found),
            )

    def _refuse_in_requirements(self, found: list[Frame], builtin: str, why: str) -> None:
        if self._role(found) == REQUIREMENTS:
            raise self.fail(
                "requirements_builtin",
                f"mo.{builtin} is not available in a requirements file: "
                f"{why}; write it in '{self.asset_file}'",
                sites.where(found), builtin=builtin,
            )

    # -- values ---------------------------------------------------

    def _plain(self, value, what: str, at: str):
        """A value as the IR holds it: JSON, finite, no handle."""
        if value is None:
            raise self.fail(
                "bad_value",
                f"{what} is None; if it came from arithmetic the result "
                f"was not a finite number (an overflow), which "
                f"cannot cross into the IR - otherwise leave the keyword "
                f"out to take the default",
                at,
            )
        if isinstance(value, bool) or isinstance(value, (int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise self.fail(
                    "bad_value", f"{what} is not a finite number", at,
                )
            return value
        if isinstance(value, (list, tuple)):
            return [self._plain(v, what, at) for v in value]
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    raise self.fail(
                        "bad_value",
                        f"{what} has a key that is not a string ({k!r})", at,
                    )
                out[k] = self._plain(v, f"{what}.{k}", at)
            return out
        if isinstance(value, KnobValue):
            raise self.fail(
                "param_value_misplaced",
                f"{what} is the knob '{value.id}' as mo.param_value returned "
                f"it (at {value.site}); {WHOLE}",
                at, param=value.id,
            )
        if isinstance(value, Handle):
            raise self.fail(
                "bad_value",
                f"{what} is a node; a node is an input, passed by position "
                f"before the keywords or by the name the operator gives "
                f"that input ('moraine docs api' lists them)",
                at,
            )
        raise self.fail(
            "bad_value",
            f"{what} is a {type(value).__name__}, which the IR cannot "
            f"hold; a param is a number, a string, a boolean, a list, or "
            f"a dict of those",
            at,
        )

    # -- ids ------------------------------------------------------

    def _id(self, given, found: list[Frame], *, noun: str, stem: str,
            builtin: str, required: str | None):
        """The id of a node or an assert (DESIGN 4.8): the name the
        call gave, or at the root scope ``stem`` and its ordinal."""
        at = sites.where(found)
        if given is not None:
            if asset_mod.id_segments(given) is None:
                raise self.fail(
                    "bad_id",
                    f"'id' must be a string of {asset_mod.ID_GRAMMAR}; "
                    f"got {given!r}",
                    at,
                )
            return given, True
        if required is not None:
            raise self.fail("id_required", required, at)
        if not found[-1].at_top_level:
            raise self.fail(
                "id_required",
                f"mo.{builtin} is called inside '{found[-1].function}' "
                f"without an id; inside a function every {noun} is named - the "
                f"function takes 'id' as its first argument and names what "
                f"it creates under it, as in id = id + \"/<name>\"",
                at,
            )
        n = self._unnamed.get(builtin, 0) + 1
        self._unnamed[builtin] = n
        return f"{stem}{n}", False

    # -- operators ------------------------------------------------

    def node(self, op: str, stack, /, *args, **kw):
        found = self._frames(stack, op)
        at = sites.where(found)
        self._refuse_in_lib(found, op)
        self._refuse_in_requirements(
            found, op, "operators build geometry, which belongs to the asset",
        )
        module = ops.get(op)
        kw = dict(kw)
        given = kw.pop("id", None)
        if getattr(module, "IS_REFERENCE", False):
            inputs: list[str] = []
            params = self._reference(op, list(args), kw, at)
        else:
            inputs = self._inputs(op, module, args, kw, at)
            params = kw
        unknown = sorted(set(params) - module.PARAMS)
        if unknown:
            raise self.fail(
                "unknown_param",
                f"{op} got unknown param(s) {unknown}; accepted: "
                f"{sorted(module.PARAMS)}",
                at, op=op,
            )
        try:
            ops.check_arity(op, len(inputs))
        except MoraineError as e:
            raise self._from(e, at) from None
        params = {k: self._plain(v, f"{op} '{k}'", at) for k, v in params.items()}
        try:
            params = ops.typed_params(op, params)
        except MoraineError as e:
            raise self._from(e, at) from None
        required = None
        if getattr(module, "USES_NODE_ID", False):
            required = (
                f"mo.{op} needs an id: its random streams are drawn from "
                f"it, so the name is part of the result - write id = "
                f"\"<name>\""
            )
        node_id, named = self._id(
            given, found, noun="node", stem=op, builtin=op, required=required,
        )
        if node_id in self.nodes:
            raise self.fail(
                "duplicate_id",
                f"node id '{node_id}' is already taken by the node created "
                f"at {self.nodes[node_id]['site']}; ids are unique within "
                f"an asset",
                at, first=self.nodes[node_id]["site"],
            )
        node: dict = {"id": node_id, "op": op}
        if inputs:
            node["inputs"] = inputs
        node["params"] = params
        node.update(at.keys)
        self.nodes[node_id] = node
        if getattr(module, "IS_REFERENCE", False):
            self.references.append(Reference(
                node=node_id, site=at, asset=params.get("asset"),
                project=params.get("project"), output=params.get("output"),
                overrides=params.get("overrides", {}),
            ))
        return sl.OpaquePythonObject(Handle(node_id, named))

    def _inputs(self, op: str, module, args, kw: dict, at: str) -> list[str]:
        """The call's inputs in the IR's order (DESIGN 4.8): by
        position, or by the name the operator gives an input's role.
        Takes the input keywords out of ``kw``, and writes the slots
        the call filled where the operator has any."""
        names = getattr(module, "INPUT_NAMES", ())
        slotted = getattr(module, "SLOT_INPUTS", {})
        given = list(args)
        labels = [
            f"'{names[i]}'" if i < len(names) else str(i) for i in range(len(given))
        ]
        named = {n: kw.pop(n) for n in names if n in kw}
        if named:
            last = max(names.index(n) for n in named)
            for i, name in enumerate(names[: last + 1]):
                if i < len(args):
                    if name in named:
                        raise self.fail(
                            "bad_input",
                            f"{op} got its input '{name}' by position and by "
                            f"keyword",
                            at, op=op,
                        )
                    continue
                if name not in named:
                    raise self.fail(
                        "bad_input",
                        f"{op} is missing its input '{name}', which comes "
                        f"before '{names[last]}'; its inputs are "
                        f"{', '.join(names)}",
                        at, op=op,
                    )
                given.append(named[name])
                labels.append(f"'{name}'")
        if slotted:
            param = module.SLOT_PARAM
            keywords = ", ".join(f"{k} = <node>" for k in slotted)
            if param in kw:
                raise self.fail(
                    "bad_input",
                    f"{op} takes no '{param}': name each one by its keyword "
                    f"({keywords}), and the build writes '{param}'",
                    at, op=op,
                )
            if len(args) > module.INPUT_ARITY[0]:
                raise self.fail(
                    "bad_input",
                    f"{op} takes {module.INPUT_ARITY[0]} input by position; "
                    f"the others are named by keyword ({keywords})",
                    at, op=op,
                )
            filled = [k for k in slotted if k in kw]
            for key in filled:
                given.append(kw.pop(key))
                labels.append(f"'{key}'")
            if filled:
                kw[param] = [slotted[key] for key in filled]
        inputs = []
        for label, a in zip(labels, given):
            if not isinstance(a, Handle):
                raise self.fail(
                    "bad_input",
                    f"{op} input {label} is a {type(a).__name__}, not a node; "
                    f"inputs are the nodes other operators returned, passed "
                    f"by position or by the name of their role, and params "
                    f"are passed by keyword (a list of nodes is expanded "
                    f"after the keywords, as in mo.{op}(id = ..., *nodes))",
                    at, op=op,
                )
            if a.id not in self.nodes:
                raise self.fail(
                    "bad_input",
                    f"{op} input {label} is a node of another asset; an asset "
                    f"takes another asset's output through mo.asset_ref",
                    at, op=op,
                )
            inputs.append(a.id)
        return inputs

    def _reference(self, op: str, args: list, kw: dict, at: str) -> dict:
        """``asset_ref``'s params from its call (DESIGN 4.6): the
        recipe path and the output by position, or a sibling project
        and its output by keyword."""
        names = ("asset", "output")
        if len(args) > len(names):
            raise self.fail(
                "bad_input",
                f"{op} takes no inputs: write mo.{op}(\"subassets/<name>"
                f".star\", \"<output>\") or mo.{op}(project = "
                f"\"<sibling>\", output = \"<output>\")",
                at, op=op,
            )
        params = {}
        for name, value in zip(names, args):
            if name in kw:
                raise self.fail(
                    "bad_input",
                    f"{op} got '{name}' by position and by keyword", at, op=op,
                )
            params[name] = value
        params.update(kw)
        return params

    # -- asserts --------------------------------------------------

    def assertion(self, kind: str, stack, /, *args, **kw):
        word = prelude.ASSERT_PREFIX + kind
        found = self._frames(stack, word)
        at = sites.where(found)
        self._refuse_in_lib(found, word)
        if args:
            raise self.fail(
                "bad_assert",
                f"mo.{word} takes keywords only, with the IR's names "
                f"(outputs = [...], tol = ...)",
                at,
            )
        kw = dict(kw)
        if "owner" in kw:
            raise self.fail(
                "bad_assert",
                f"mo.{word} takes no 'owner': the file decides - an entry "
                f"in '{self.requirements_file}' is the human's, an entry "
                f"in an asset file is the agent's",
                at,
            )
        if "from_param" in kw:
            raise self.fail(
                "bad_assert",
                f"mo.{word} takes no 'from_param': the build records it "
                f"where a threshold is a knob's value - in "
                f"'{self.requirements_file}', write the threshold as "
                f"mo.param_value(\"<param_id>\")",
                at,
            )
        human = self._role(found) == REQUIREMENTS
        required = None
        if human:
            required = (
                f"mo.{word} in a requirements file needs an id: a "
                f"requirement is named, so that adding one never renames "
                f"another - write id = \"<name>\""
            )
        aid, _ = self._id(
            kw.pop("id", None), found, noun="assert", stem=kind, builtin=word,
            required=required,
        )
        if aid in self._assert_ids:
            raise self.fail(
                "duplicate_assert_id",
                f"assert id '{aid}' is already taken by the assert "
                f"declared at {self._assert_ids[aid]}",
                at, first=self._assert_ids[aid],
            )
        self._assert_ids[aid] = at
        entry = {"id": aid, "kind": kind}
        follows: dict[str, KnobValue] = {}
        for k, v in kw.items():
            if isinstance(v, KnobValue) and k in vocab.THRESHOLDS and human:
                follows[k] = v
                continue
            entry[k] = self._plain(v, f"{word} '{k}'", at)
        if follows:
            self.following.append((entry, follows, at))
        try:
            vocab.typed_entry(entry)
        except MoraineError as e:
            raise self._from(e, at) from None
        entry["owner"] = "human" if human else "agent"
        entry.update(at.keys)
        self.asserts.append(entry)
        return None

    # -- handles --------------------------------------------------

    def handle(self, what: str, stack, /, *args, **kw):
        """``mo.param_value(id)`` (DESIGN 4.8): an opaque handle to one
        of the asset's knobs, for a requirements file."""
        found = self._frames(stack, what)
        at = sites.where(found)
        self._refuse_in_lib(found, what)
        if self._role(found) != REQUIREMENTS:
            raise self.fail(
                "param_value_outside_requirements",
                f"mo.{what} is for a requirements file, where no knob's "
                f"value is in reach; in '{self.asset_file}' the value is "
                f"what mo.param returned, so bind it to a name and use that",
                at,
            )
        given = self._arguments(what, ("id",), list(args), dict(kw), at)
        if not isinstance(given["id"], str):
            raise self.fail(
                "bad_declaration",
                f"mo.{what} names a knob by its id, a string, as in "
                f"mo.{what}(\"<param_id>\")",
                at,
            )
        knob = KnobValue(given["id"], at)
        self.handles.append(knob)
        return sl.OpaquePythonObject(knob)

    def on_a_handle(self, error: BuildError, text: str) -> BuildError:
        """The refusal of arithmetic on a knob's handle (DESIGN 4.8),
        read from the interpreter's own: it names the value by its
        type, and in a requirements file that type is a handle's
        alone, since no operator runs there."""
        at = error.detail.get("site", "")
        file = at.rsplit(":", 1)[0]
        if (
            self.roles.get(file) != REQUIREMENTS
            or "opaque_python_object" not in text
        ):
            return error
        return BuildError(
            "param_value_arithmetic",
            f"a knob's value from mo.param_value is computed with "
            f"({error.message}); {WHOLE}, since the knob is resolved only "
            f"after the asset has declared it - write the derived quantity "
            f"as a constant of this file",
            **error.detail,
        )

    # -- declarations ---------------------------------------------

    def declare(self, what: str, stack, /, *args, **kw):
        found = self._frames(stack, what)
        self._refuse_in_lib(found, what)
        return getattr(self, f"_declare_{what}")(found, list(args), dict(kw))

    def _arguments(self, what: str, names: tuple[str, ...], args: list, kw: dict,
                   at: str, optional: tuple[str, ...] = ()) -> dict:
        """The call's arguments by name: ``names`` may be passed by
        position in that order, ``optional`` by keyword only."""
        shape = f"mo.{what}({', '.join(names + tuple(f'{o} = ...' for o in optional))})"
        if len(args) > len(names):
            raise self.fail(
                "bad_declaration", f"mo.{what} got too many values; write {shape}", at,
            )
        given = dict(zip(names, args))
        for k, v in kw.items():
            if k in given or k not in names + optional:
                raise self.fail(
                    "bad_declaration",
                    f"mo.{what} got an unexpected '{k}'; write {shape}", at,
                )
            given[k] = v
        missing = [n for n in names if n not in given]
        if missing:
            raise self.fail(
                "bad_declaration",
                f"mo.{what} is missing {missing}; write {shape}", at,
            )
        return given

    def _declare_output(self, found, args, kw):
        at = sites.where(found)
        self._refuse_in_requirements(
            found, "output", "what ships is declared where it is built",
        )
        given = self._arguments("output", ("name", "node"), args, kw, at)
        node = given["node"]
        if not isinstance(node, Handle) or node.id not in self.nodes:
            raise self.fail(
                "bad_declaration",
                f"mo.output('{given['name']}', ...) needs the node that is "
                f"the output, as an operator returned it",
                at,
            )
        if not node.named:
            raise self.fail(
                "id_required",
                f"output '{given['name']}' is the node '{node.id}', which "
                f"was created without an id at "
                f"{self.nodes[node.id]['site']}; an output's node is named, "
                f"since reports and pins point at it - give that call an id",
                at,
            )
        self.outputs.append((given["name"], node, at))

    def _declare_group(self, found, args, kw):
        """In an asset or in its requirements file (DESIGN 5.2): a group
        a requirement names belongs in the file the requirement is in."""
        at = sites.where(found)
        given = self._arguments("group", ("name", "outputs"), args, kw, at)
        members = self._plain(given["outputs"], "mo.group 'outputs'", at)
        self.groups.append((given["name"], members, at))

    def _declare_waive(self, found, args, kw):
        at = sites.where(found)
        self._refuse_in_requirements(
            found, "waive", "a waiver documents the asset's own intent",
        )
        given = self._arguments("waive", ("output", "check"), args, kw, at)
        self.waivers.append((given["output"], given["check"], at))

    def _declare_waive_asset(self, found, args, kw):
        at = sites.where(found)
        self._refuse_in_requirements(
            found, "waive_asset", "a waiver documents the asset's own intent",
        )
        given = self._arguments("waive_asset", ("check",), args, kw, at)
        self.asset_waivers.append((given["check"], at))

    def _declare_vendored(self, found, args, kw):
        at = sites.where(found)
        self._refuse_in_requirements(
            found, "vendored", "provenance is the asset's own record",
        )
        given = self._arguments("vendored", ("source", "hash"), args, kw, at)
        if self.provenance is not None:
            raise self.fail(
                "bad_declaration",
                f"mo.vendored is declared twice; the first is at "
                f"{self.provenance[1]}",
                at,
            )
        self.provenance = ({
            "source": self._plain(given["source"], "mo.vendored 'source'", at),
            "content_hash": self._plain(given["hash"], "mo.vendored 'hash'", at),
        }, at)

    def _declare_budget(self, found, args, kw):
        at = sites.where(found)
        if self._role(found) != REQUIREMENTS:
            raise self.fail(
                "budget_outside_requirements",
                f"mo.budget is declared in an asset file; a budget is an "
                f"acceptance criterion the human owns - write it in "
                f"'{self.requirements_file}'",
                at,
            )
        if args or not kw:
            raise self.fail(
                "bad_declaration",
                f"mo.budget takes the budgets by keyword, as in "
                f"mo.budget(materials = 4); known: "
                f"{sorted(asset_mod.KNOWN_BUDGETS)}",
                at,
            )
        for key, value in kw.items():
            self.budgets.append((key, self._plain(value, f"mo.budget '{key}'", at), at))

    def _declare_param(self, found, args, kw):
        at = sites.where(found)
        self._refuse_in_requirements(
            found, "param", "a knob is declared by the asset that consumes it",
        )
        given = self._arguments(
            "param", ("id", "default"), args, kw, at,
            optional=("min", "max", "label", "step"),
        )
        decl = {"id": given["id"], "value": self._plain(given["default"], "mo.param 'default'", at)}
        for key in ("label", "min", "max", "step"):
            if key in given:
                decl[key] = self._plain(given[key], f"mo.param '{key}'", at)
        decl.update(at.keys)
        try:
            asset_mod.check_param(
                f"mo.param at {at}", decl, {}, self._param_ids, set(),
            )
        except MoraineError as e:
            raise self._from(e, at) from None
        pid = decl["id"]
        # The recipe's default, then the accepted value, then a
        # consumer's (DESIGN 4.8); later values take precedence, and
        # the entry keeps the default where another value took its
        # place (DESIGN 4.1).
        default = decl["value"]
        if pid in self.accepted:
            try:
                decl["value"] = params_mod.typed_value(decl, self.accepted[pid])
                decl["from"] = "accepted"
                decl["default"] = default
            except MoraineError:
                # A finding about the human's file, which the report
                # carries beside every other entry (DESIGN 8.2).
                decl["mismatch"] = self.accepted[pid]
        if pid in self.offered:
            if self.offered[pid] is None and not self.strict_offers:
                # The overlay's unset (DESIGN 7.2): the state without
                # the human's line, which is the recipe's default.
                decl["value"] = default
                for key in ("from", "default", "mismatch"):
                    decl.pop(key, None)
                self.accepted_offers[pid] = None
            else:
                judge = params_mod.offered_value if self.ranged_offers else params_mod.typed_value
                try:
                    decl["value"] = judge(decl, self.offered[pid])
                except MoraineError as e:
                    raise self._from(e, self.offered_site or at) from None
                decl["from"] = "override"
                decl["default"] = default
                decl.pop("mismatch", None)
                self.accepted_offers[pid] = decl["value"]
        self.params.append(decl)
        return decl["value"]

    def record_direct(self, reading: uses.Reading) -> None:
        """Record the knobs whose value reached one node param whole
        (DESIGN 7.2): ``node`` and ``param`` on the entry. The text
        says which call took the value, and the entry is written only
        where the graph agrees: one node made by that call, holding
        the knob's value in its own number type. A reference is left
        out, since the build resolves what it names, and so is a knob
        a threshold follows (``mo.param_value``): its value is read in
        a second place, which a patch of the one param would leave
        behind."""
        followed = {
            knob.id for _, follows, _ in self.following for knob in follows.values()
        }
        for decl in self.params:
            use = reading.whole.get(decl["id"])
            if use is None or "via" in decl or decl["id"] in followed:
                continue
            if decl["site"].rsplit(":", 1)[0] != self.asset_file:
                continue
            at = f"{self.asset_file}:{use.line}"
            made = [
                node for node in self.nodes.values()
                if node["site"] == at and "via" not in node
                and node["op"] == use.op
            ]
            if len(made) != 1 or use.param not in made[0]["params"]:
                continue
            if getattr(ops.get(use.op), "IS_REFERENCE", False):
                continue
            if not _same(made[0]["params"][use.param], decl["value"]):
                continue
            decl["node"] = made[0]["id"]
            decl["param"] = use.param

    # -- resolution -----------------------------------------------

    def _checked(self, at: str, check, *args) -> None:
        try:
            check(*args)
        except MoraineError as e:
            raise self._from(e, at) from None

    def _names(self, entry: dict):
        try:
            return vocab.output_names(entry)
        except (KeyError, IndexError, TypeError):
            return set()  # the entry's shape is refused by its own check

    def _resolve_following(self) -> None:
        """Give each threshold that follows a knob the knob's resolved
        value (DESIGN 4.8), in the number type its key declares, and
        record the knob on the entry. A handle no threshold took is
        refused: the language lets a file test one (`if`, `==`), and
        the test would read the same at every value of the knob."""
        taken = {id(knob) for _, follows, _ in self.following for knob in follows.values()}
        for knob in self.handles:
            if id(knob) not in taken:
                raise self.fail(
                    "param_value_unused",
                    f"mo.param_value(\"{knob.id}\") became no assert's "
                    f"threshold; {WHOLE}. If it was read in a condition "
                    f"('if', '==', 'and', 'not'): it holds no number while "
                    f"the file runs, so the file cannot branch on a knob's "
                    f"value - state requirements that hold at every value "
                    f"the knob takes. Otherwise pass it as the threshold "
                    f"that moves with the knob, or remove the call",
                    knob.site, param=knob.id,
                )
        knobs = {decl["id"]: decl for decl in self.params}
        for entry, follows, at in self.following:
            kinds = getattr(vocab.KINDS.get(entry["kind"]), "KEY_KINDS", {})
            for key, knob in follows.items():
                decl = knobs.get(knob.id)
                if decl is None:
                    raise self._from(params_mod.unknown_param(knob.id, knobs), at)
                value = decl["value"]
                if asset_mod.param_kind(value) != "number":
                    raise self.fail(
                        "param_value_kind",
                        f"assert '{entry['id']}' takes '{key}' from the knob "
                        f"'{knob.id}', which is a "
                        f"{asset_mod.param_kind(value)}; a threshold is one "
                        f"number, so name a number knob",
                        at, param=knob.id,
                    )
                try:
                    entry[key] = numbers.typed(value, kinds.get(key, numbers.OTHER))
                except numbers.FloatForInteger:
                    raise self.fail(
                        "bad_assert",
                        f"assert '{entry['id']}' '{key}' takes an integer, "
                        f"and the knob '{knob.id}' holds the float "
                        f"{value!r}; declare the knob with an integer "
                        f"default, as in mo.param(\"{knob.id}\", "
                        f"{int(value)}, ...)",
                        at, param=knob.id,
                    ) from None
            entry["from_param"] = {key: knob.id for key, knob in follows.items()}

    def _unowned_groups(self, declared_at: dict[str, str]) -> list[dict]:
        """The ``group_not_owned`` warnings (DESIGN 5.2): a human-owned
        entry names a group the asset file declares, so the agent can
        narrow the requirement's reach with no diff in the human's
        file."""
        named: dict[str, list[str]] = {}
        for entry in self.asserts:
            if entry["owner"] != "human":
                continue
            names = self._names(entry)
            for n in sorted(names if isinstance(names, set) else ()):
                if n in declared_at:
                    named.setdefault(n, []).append(entry["id"])
        out = []
        for name, ids in named.items():
            at = declared_at[name]
            if self.roles.get(str(at).rsplit(":", 1)[0]) == REQUIREMENTS:
                continue
            out.append({
                "warning": "group_not_owned",
                "group": name,
                "site": str(at),
                "count": len(ids),
                "asserts": ids[:LISTED],
                "fix": (
                    f"declare '{name}' in '{self.requirements_file}', beside "
                    f"the requirement that names it, so its members change "
                    f"only with the human's file"
                ),
            })
        return out

    def document(self, stamp: str) -> dict:
        """The IR document, every name resolved (DESIGN 4.8). Runs once
        the recipe and its requirements file have run."""
        unused = sorted(set(self.offered) - set(self.accepted_offers))
        if unused and self.strict_offers:
            e = params_mod.unknown_param(unused[0], [d["id"] for d in self.params])
            raise self._from(e, self.offered_site)
        if not self.outputs:
            raise self.fail(
                "no_outputs",
                f"'{self.asset_file}' declares no output; what ships is "
                f"declared with mo.output(\"<name>\", <node>)",
                self.asset_file,
            )
        outputs: dict[str, str] = {}
        folded: dict[str, str] = {}
        for name, node, at in self.outputs:
            if isinstance(name, str) and name in outputs:
                raise self.fail(
                    "duplicate_output", f"output '{name}' is declared twice", at,
                )
            self._checked(at, asset_mod.check_output, name, node.id, self.nodes, folded)
            outputs[name] = node.id
        groups: dict[str, list] = {}
        declared_at: dict[str, str] = {}
        for name, members, at in self.groups:
            if isinstance(name, str) and name in groups:
                raise self.fail(
                    "bad_group",
                    f"group '{name}' is declared twice; the first is at "
                    f"{declared_at[name]}",
                    at,
                )
            self._checked(at, asset_mod.check_group, name, members, outputs, folded)
            groups[name] = members
            declared_at[name] = at
        checks: dict[str, dict] = {}
        for output, check, at in self.waivers:
            if not isinstance(output, str) or not isinstance(check, str):
                raise self.fail(
                    "bad_declaration",
                    "mo.waive takes the output's name and the check's name", at,
                )
            self._checked(at, asset_mod.check_waiver, output, {check: False}, outputs)
            checks.setdefault(output, {})[check] = False
        asset_checks: dict[str, bool] = {}
        for check, at in self.asset_waivers:
            self._checked(at, asset_mod.check_asset_check, check, False)
            asset_checks[check] = False
        budgets: dict[str, int] = {}
        for key, value, at in self.budgets:
            if key in budgets:
                raise self.fail(
                    "bad_declaration", f"budget '{key}' is declared twice", at,
                )
            self._checked(at, asset_mod.check_budget, key, value)
            budgets[key] = value
        self._resolve_following()
        seen: set[str] = set()
        for entry in self.asserts:
            at = sites.Where(entry["site"], entry.get("via", ()))
            names = self._names(entry)
            missing = sorted(
                n for n in (names if isinstance(names, set) else ())
                if isinstance(n, str) and n not in outputs and n not in groups
            )
            if missing:
                raise self.fail(
                    "unknown_output",
                    f"assert '{entry['id']}' names '{missing[0]}', which no "
                    f"mo.output or mo.group declares; declared outputs: "
                    f"{sorted(outputs)}"
                    + (f"; groups: {sorted(groups)}" if groups else ""),
                    at, assertion=entry["id"], output=missing[0],
                )
            self._checked(
                at, asset_mod.check_assert, f"assert at {at}", entry,
                outputs, groups, seen,
            )

        self.warnings = self._unowned_groups(declared_at)

        doc: dict = {"build_version": stamp, "nodes": list(self.nodes.values())}
        if self.params:
            doc["params"] = self.params
        doc["outputs"] = outputs
        if groups:
            doc["groups"] = groups
        if checks:
            doc["checks"] = checks
        if asset_checks:
            doc["asset_checks"] = asset_checks
        if self.asserts:
            doc["asserts"] = self.asserts
        if budgets:
            doc["budgets"] = budgets
        if self.provenance is not None:
            doc["provenance"] = self.provenance[0]
        return doc
