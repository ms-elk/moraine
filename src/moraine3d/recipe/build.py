"""``moraine build`` (DESIGN 4.8, 6): recipes in, IR out.

The build resolves the target's reference closure, runs each asset's
recipe and its requirements file, resolves every name, and writes the
IR in canonical form under each project's ``build/``. It is a pure
function of the recipe files and the values a consumer offers, so the
same inputs give the same bytes (DESIGN 4.4). Nothing here evaluates
geometry.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import starlark as sl

from ..core import accepted as accepted_mod
from ..core import asset as asset_mod
from ..core import sources as sources_mod
from ..core import workspace as workspace_mod
from ..core.brief import LISTED
from ..core.asset import (
    BUILD_DIR, LIB_DIR, RESERVED_DIRS, ASSET_EXT, SUBASSETS_DIR,
)
from ..core.sources import RECIPE_EXT, REQUIREMENTS_EXT
from ..errors import BuildError, MoraineError
from ..ops import asset_ref as ref_op
from . import clean, dialect, host, prelude, refusals, repetition, uses, version


@dataclass
class BuiltAsset:
    """One asset under one override identity, built."""

    recipe: Path  # the asset file
    identity: str  # "" for the asset's own values
    ir_path: Path  # where its IR is written
    text: str  # the IR, canonical
    asset: asset_mod.Asset  # the IR, loaded, ``path`` at ``ir_path``
    overrides: dict  # the offered values it was built under, normalized
    sources: frozenset[Path]  # the recipe files that contributed
    counts: dict = field(default_factory=dict)
    # The ids the project's params.json holds that the asset declares
    # no knob for (DESIGN 7.2).
    unused: tuple[str, ...] = ()
    # How the asset file uses its names; None for a file left unread.
    reading: uses.Reading | None = None
    # The warnings the document raised (``group_not_owned``).
    warnings: tuple[dict, ...] = ()


@dataclass
class BuildResult:
    target: BuiltAsset
    assets: dict[Path, BuiltAsset]  # by IR path: the closure, variants included
    root: Path  # the root sites and reports are framed from
    # Every recipe file the build named, present or not (DESIGN 4.6):
    # what a watcher follows, so a requirements file created later is
    # seen. ``sources`` holds the ones read, whose bytes are the revision.
    named: frozenset[Path] = frozenset()
    # What the build found at every file it named, by resolved path
    # (DESIGN 6): the bytes it read, None for a file that was not
    # there. The revision is their hash, whatever the files hold by the
    # time it is named.
    contents: dict[Path, bytes | None] = field(default_factory=dict)
    written: list[Path] = field(default_factory=list)
    unchanged: int = 0
    ms: float = 0.0
    # Built under values offered for the target's own knobs: never written.
    preview: bool = False
    # What the build found in the target project's ``build/`` and
    # removed from it, and who reads the project (``clean.sweep``,
    # DESIGN 4.8, 6). Empty for a build that did not look.
    swept: clean.Swept = field(default_factory=clean.Swept)
    # Advisory findings about the target's recipe (DESIGN 4.8); they
    # never fail a build.
    warnings: list[dict] = field(default_factory=list)
    # The asset files whose text the build could run and not read for
    # how it uses its names (``uses``).
    unread: list[Path] = field(default_factory=list)

    def _blocked(self) -> dict:
        """The report's ``blocked`` (DESIGN 4.8): why nothing was
        removed from ``build/``, and what lifts it."""
        out: dict = {}
        fixes = []
        if self.swept.rootless:
            names = [p.name for p in self.swept.rootless]
            out["assets"] = names
            fixes.append(
                f"the project holds {len(names)} top-level asset files "
                f"and a build/ is read from one root - move definitions "
                f"under {LIB_DIR}/ and parts under {SUBASSETS_DIR}/"
            )
        if self.swept.blocked:
            out["files"] = [_relative(p, self.root) for p in self.swept.blocked[:LISTED]]
            fixes.append(
                "the files named are not IR as this build writes it, so "
                "what they reference is unknown - 'moraine build' on the "
                "project of each writes it again"
            )
        out["fix"] = "nothing was removed from build/: " + "; ".join(fixes)
        return out

    @property
    def sources(self) -> frozenset[Path]:
        """Every recipe file the build read (DESIGN 6): the assets, their
        requirements files, and the ``lib/`` files they loaded."""
        out: set[Path] = set()
        for built in self.assets.values():
            out |= built.sources
        return frozenset(out)

    def report(self, *, wrote: bool = True) -> dict:
        """The ``build`` report (DESIGN 6). ``wrote`` False is a build
        kept in memory, which has no files to name."""
        assets = {}
        for built in sorted(self.assets.values(), key=lambda b: b.ir_path.as_posix()):
            assets[_relative(built.ir_path, self.root)] = {
                "recipe": _relative(built.recipe, self.root), **built.counts,
            }
        out: dict = {"ok": True, "build_version": version.build_version()}
        if wrote:
            out["written"] = [_relative(p, self.root) for p in self.written]
            out["unchanged"] = self.unchanged
            if self.swept.removed:
                out["removed"] = [_relative(p, self.root) for p in self.swept.removed]
        if self.swept.unrecognized:
            names = [_relative(p, self.root) for p in self.swept.unrecognized]
            out["unrecognized"] = {"count": len(names), "names": names[:LISTED]}
        if self.swept.blocked or self.swept.rootless:
            out["blocked"] = self._blocked()
        out["assets"] = assets
        if self.swept.readers is not None:
            out["consumers"] = self.swept.readers
        if self.warnings:
            out["warnings"] = self.warnings
        if self.unread:
            out["unread"] = {
                "files": [_relative(p, self.root) for p in self.unread[:LISTED]],
                "note": (
                    "the build ran these files and could not read how "
                    "they use their names, so their knobs are rebuilt on "
                    "every change (no live slider, no reroll)"
                ),
            }
        out["ms"] = self.ms
        return out


def _relative(path: Path, root: Path) -> str:
    """``path`` as reports show it, from ``root``; both resolved by
    the caller."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# The builtins are evaluated once per thread and reach the session of
# the asset being run through it: the prelude costs as much as a small
# recipe, and a build runs one per asset of the closure. Per thread,
# since the interpreter's objects stay with the thread that made them.
_local = threading.local()


def _node(op, stack, /, *args, **kw):
    return _local.session.node(op, stack, *args, **kw)


def _assert(kind, stack, /, *args, **kw):
    return _local.session.assertion(kind, stack, *args, **kw)


def _declare(what, stack, /, *args, **kw):
    return _local.session.declare(what, stack, *args, **kw)


def _handle(what, stack, /, *args, **kw):
    return _local.session.handle(what, stack, *args, **kw)


def _fail(stack, /, *args, **kw):
    return _local.session.failed(stack, *args, **kw)


def _builtins() -> sl.FrozenModule:
    """The prelude, evaluated: ``mo`` and ``math``."""
    frozen = getattr(_local, "builtins", None)
    if frozen is None:
        module = sl.Module()
        module.add_callable("_node", _node)
        module.add_callable("_assert", _assert)
        module.add_callable("_declare", _declare)
        module.add_callable("_handle", _handle)
        module.add_callable("_fail", _fail)
        for name in dialect.MATH_CONSTANTS:
            module[f"_math_{name}"] = getattr(math, name)
        for name in dialect.MATH_FUNCTIONS:
            module.add_callable(f"_math_{name}", dialect.math_function(name))
        ast = sl.parse("<prelude>", prelude.source(), dialect.dialect())
        sl.eval(module, ast, dialect.prelude_globals())
        frozen = _local.builtins = module.freeze()
    return frozen


class _Files:
    """The recipe files of one build, each looked at once (DESIGN 6):
    every asset and variant of the build runs the same bytes, a file
    that was not there stays absent, and the revision names that
    state."""

    def __init__(self):
        # By resolved path: the bytes read, None for a file not there.
        self.held: dict[Path, bytes | None] = {}

    def read(self, path: Path) -> bytes:
        key = path.resolve()
        if key not in self.held:
            try:
                self.held[key] = path.read_bytes()
            except FileNotFoundError:
                self.held[key] = None
        data = self.held[key]
        if data is None:
            raise FileNotFoundError(str(path))
        return data

    def is_file(self, path: Path) -> bool:
        key = path.resolve()
        if key not in self.held and not path.is_file():
            self.held[key] = None
        return self.held.get(key, b"") is not None

    def found(self, paths) -> dict[Path, bytes | None]:
        """What the build found at each of ``paths`` it looked at, by
        resolved path."""
        keys = {p.resolve() for p in paths}
        return {key: self.held[key] for key in keys if key in self.held}


def _read(files: _Files, path: Path, shown: str, at: str | None) -> str:
    detail = {"site": at} if at else {}
    try:
        return files.read(path).decode("utf-8")
    except FileNotFoundError:
        raise BuildError(
            "recipe_not_found", f"no recipe file '{shown}'", file=shown, **detail,
        ) from None
    except UnicodeDecodeError as e:
        raise BuildError(
            "bad_encoding",
            f"recipe '{shown}' is not valid UTF-8 (byte {e.start}); "
            f"re-save it as UTF-8",
            file=shown, **detail,
        ) from e
    except OSError as e:
        raise BuildError(
            "recipe_unreadable",
            f"cannot read recipe '{shown}' ({e.__class__.__name__}: {e})",
            file=shown, **detail,
        ) from e


class _Run:
    """One asset's recipe files, run into one session."""

    def __init__(self, session: host.Session, project: Path, root: Path,
                 files: _Files, named_at: str | None = None,
                 read: set[Path] | None = None):
        self.read = read if read is not None else set()
        self.files = files
        self.session = session
        self.project = project
        self.root = root
        self.named_at = named_at  # the site of the reference that named the asset
        self.loading: list[str] = []  # project-relative names, outermost first
        self.loaded: dict[str, sl.FrozenModule] = {}
        self.sources: set[Path] = set()
        self.globals = dialect.recipe_globals()
        self.loader = sl.FileLoader(self._load)
        self.builtins = _builtins()
        self.in_bootstrap = False

    def _shown(self, name: str) -> str:
        """A project-relative name as sites and reports show it."""
        return _relative(self.project / name, self.root)

    def _raise(self, error: Exception) -> BuildError:
        """The build's error for whatever stopped an evaluation: the
        refusal a host function kept, or the interpreter's own, read
        from its text."""
        if isinstance(error, BuildError):
            self.session.error = error
        elif self.session.error is None:
            found = refusals.from_interpreter(str(error))
            self.session.error = self.session.on_a_handle(found, str(error))
        return self.session.error

    def run(self, name: str, role: str) -> sl.Module:
        """Run one recipe file, named project-relative."""
        shown = self._shown(name)
        path = self.project / name
        by = self._shown(self.loading[-1]) if self.loading else self.named_at
        self.read.add(path)
        try:
            self._check_own(path, name, shown)
            source = _read(self.files, path, shown, by)
        except BuildError as e:
            raise self._raise(e) from None
        self.sources.add(path)
        self.session.roles[shown] = role
        module = sl.Module()
        self.loading.append(name)
        outer = getattr(_local, "session", None)
        _local.session = self.session
        try:
            self.in_bootstrap = True
            try:
                boot = sl.parse(
                    "<builtins>",
                    f'load("{dialect.BUILTINS}", "mo", "math", "{prelude.FAIL}")',
                    dialect.dialect(),
                )
                sl.eval(module, boot, self.globals, self.loader)
            finally:
                self.in_bootstrap = False
            ast = sl.parse(shown, source, dialect.dialect(load=role != host.REQUIREMENTS))
            sl.eval_with(dialect.options(), module, ast, self.globals, self.loader)
        except (sl.StarlarkError, BuildError) as e:
            raise self._raise(e) from None
        finally:
            self.loading.pop()
            _local.session = outer
        return module

    def _check_own(self, path: Path, name: str, shown: str) -> None:
        """Refuse a recipe file reached through a link (DESIGN 4.8):
        the rules judge the path a recipe spells, and a link under it
        would run another file in that one's place. Case is the
        filesystem's to fold."""
        if os.path.normcase(path.resolve()) == os.path.normcase(path):
            return
        why = (
            "the file is reached through a link; a recipe's files are "
            "its project's own - replace the link with the file"
        )
        if self.loading:
            raise self._refuse(name, why)
        raise BuildError("recipe_is_link", f"recipe '{shown}': {why}", file=shown)

    def _load(self, name: str) -> sl.FrozenModule:
        """The interpreter's ``load()`` (DESIGN 4.8)."""
        if name == dialect.BUILTINS:
            if self.in_bootstrap:
                return self.builtins
            raise self._refuse(
                name,
                "'mo' and 'math' are in scope already; a recipe loads only "
                "its project's own files",
            )
        by = self.loading[-1]
        self._check_load(name, by)
        if name in self.loading:
            chain = self.loading[self.loading.index(name):] + [name]
            raise self._refuse(
                name, f"load cycle: {' -> '.join(chain)}", chain=chain,
            )
        if name not in self.loaded:
            role = host.REQUIREMENTS if sources_mod.is_requirements_file(name) else host.LIB
            self.loaded[name] = self.run(name, role).freeze()
        return self.loaded[name]

    def _refuse(self, name: str, why: str, **detail) -> BuildError:
        by = self._shown(self.loading[-1]) if self.loading else None
        return self.session.fail(
            "load_refused", f"load(\"{name}\"): {why}", by, load=name, **detail,
        )

    def _check_load(self, name: str, by: str) -> None:
        """The three rules of ``load()`` (DESIGN 4.8), and the path
        grammar under them: project-relative, forward slashes, no way
        out of the project."""
        if (
            "\\" in name or name.startswith("/")
            or PureWindowsPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise self._refuse(
                name,
                f"a load path is project-relative with forward slashes, as "
                f"in \"{LIB_DIR}/<name>{RECIPE_EXT}\"; it never leaves the "
                f"project",
            )
        if not name.endswith(RECIPE_EXT):
            raise self._refuse(name, f"a recipe file ends in {RECIPE_EXT}")
        parts = name.split("/")
        by_lib = by.split("/")[0] == LIB_DIR
        if sources_mod.is_requirements_file(name):
            own = self.session.requirements_file
            if by_lib:
                raise self._refuse(
                    name,
                    f"a {LIB_DIR}/ file loads no requirements file; a helper "
                    f"that needs a requirement constant takes it as an "
                    f"argument",
                )
            if self._shown(name) != own:
                raise self._refuse(
                    name,
                    f"an asset loads only its own requirements file, "
                    f"'{own}'; another asset's requirements are checked "
                    f"with that asset",
                )
            return
        if parts[0] == LIB_DIR and len(parts) == 2:
            return
        raise self._refuse(
            name,
            f"an asset file is referenced, never loaded - write "
            f"mo.asset_ref(\"{name}\", \"<output>\"); definitions shared "
            f"between assets live in {LIB_DIR}/<name>{RECIPE_EXT}",
        )


class _Build:
    def __init__(self, root: Path, strict_offers: bool = True, ranged_offers: bool = True):
        self.root = root
        self.strict_offers = strict_offers
        self.ranged_offers = ranged_offers
        # Every recipe file the build named, read or not: what to watch
        # when the build fails (DESIGN 4.6).
        self.read: set[Path] = set()
        self.files = _Files()
        self.assets: dict[Path, BuiltAsset] = {}
        self.unread: set[Path] = set()
        # How each asset file uses its names, read once however many
        # variants of the asset the build makes.
        self.readings: dict[Path, uses.Reading | None] = {}
        self.memo: dict[tuple[Path, str], BuiltAsset] = {}
        self.building: list[Path] = []

    # -- one asset ------------------------------------------------

    def asset(self, recipe: Path, offered: dict, offered_site: str | None,
              consumer: Path | None) -> BuiltAsset:
        recipe = recipe.resolve()
        key = (recipe, ref_op.override_token(offered))
        if key in self.memo:
            return self.memo[key]
        shown = _relative(recipe, self.root)
        if recipe in self.building:
            chain = [_relative(p, self.root) for p in self.building] + [shown]
            raise BuildError(
                "ref_cycle",
                f"asset references form a cycle: {' -> '.join(chain)}; "
                f"break the cycle (the reference graph must be a DAG)",
                chain=chain, **({"site": offered_site} if offered_site else {}),
            )
        if asset_mod.VARIANT_STEM.search(recipe.name[: -len(RECIPE_EXT)]):
            raise BuildError(
                "bad_asset_name",
                f"asset file '{shown}' ends its name like a variant IR "
                f"(<name>.<24 hex digits>); rename the file",
                file=shown,
            )
        project = asset_mod.project_root(recipe)
        accepted, accepted_file = self._accepted(recipe, project)
        session = host.Session(
            asset_file=shown,
            requirements_file=_relative(sources_mod.requirements_file(recipe), self.root),
            offered=dict(offered), offered_site=offered_site,
            strict_offers=self.strict_offers or consumer is not None,
            ranged_offers=self.ranged_offers or consumer is not None,
            accepted=accepted,
        )
        run = _Run(
            session, project, self.root, self.files,
            named_at=offered_site, read=self.read,
        )
        name = recipe.relative_to(project).as_posix()
        self.building.append(recipe)
        try:
            run.run(name, host.ASSET)
            requirements = sources_mod.requirements_file(recipe)
            # Named whether or not it exists, so creating it later is
            # an edit the watcher sees.
            self.read.add(requirements)
            req_name = requirements.relative_to(project).as_posix()
            if req_name not in run.loaded and self.files.is_file(requirements):
                run.loaded[req_name] = run.run(req_name, host.REQUIREMENTS).freeze()
            sources = set(run.sources)
            if accepted_file is not None:
                sources.add(accepted_file)
            for ref in session.references:
                target = self._reference(session, ref, recipe, project)
                sources |= target.sources
            if recipe not in self.readings:
                self.readings[recipe] = uses.read(
                    self.files.read(recipe).decode("utf-8"),
                )
            reading = self.readings[recipe]
            if reading is None:
                self.unread.add(recipe)
            else:
                session.record_direct(reading)
            doc = session.document(version.build_version())
        finally:
            self.building.pop()
        try:
            loaded = asset_mod.loads(json.dumps(doc, allow_nan=False))
            text = asset_mod.dumps(loaded)
        except MoraineError as e:
            raise BuildError(e.code, e.message, file=shown, **e.detail) from None
        identity = ref_op.override_identity(session.accepted_offers)
        ir_path = self._ir_path(
            recipe, project, identity if consumer is not None else "",
        )
        loaded.path = ir_path
        built = BuiltAsset(
            recipe=recipe, identity=identity, ir_path=ir_path, text=text,
            asset=loaded, overrides=dict(session.accepted_offers),
            sources=frozenset(sources),
            counts={
                "nodes": len(loaded.nodes),
                "asserts": len(loaded.asserts or []),
                "outputs": len(loaded.outputs),
            },
            unused=tuple(sorted(
                set(accepted) - {decl["id"] for decl in session.params}
            )),
            reading=reading,
            warnings=tuple(session.warnings),
        )
        self.memo[key] = built
        self.assets[ir_path] = built
        return built

    def _accepted(self, recipe: Path, project: Path) -> tuple[dict, Path | None]:
        """The values the human accepted for an asset's knobs (DESIGN
        7.2): the project's ``params.json``, read for the project's
        root whoever builds it. A sub-asset's knobs take their values
        from the asset that references it (DESIGN 4.6). With them, the file
        where there is one: its bytes are part of the revision."""
        if recipe.parent != project:
            return {}, None
        path = project / accepted_mod.PARAMS_FILE
        # Named whether or not it exists, so the first accept is an
        # edit the watcher sees.
        self.read.add(path)
        if not self.files.is_file(path):
            return {}, None
        shown = _relative(path, self.root)
        try:
            return accepted_mod.loads(self.files.read(path), shown), path
        except MoraineError as e:
            raise BuildError(e.code, e.message, **e.detail) from None
        except OSError as e:
            raise BuildError(
                "params_unreadable",
                f"cannot read '{shown}' ({e.__class__.__name__}: {e}); "
                f"check that the file is readable",
                file=shown,
            ) from e

    def _ir_path(self, recipe: Path, project: Path, identity: str) -> Path:
        """Where an asset's IR lives (DESIGN 4, 4.6): under its
        project's ``build/``, a variant beside the asset's own under
        the override identity."""
        stem = recipe.name[: -len(RECIPE_EXT)]
        where = project / BUILD_DIR
        if recipe.parent != project:
            where = where / SUBASSETS_DIR
        name = f"{stem}.{identity}" if identity else stem
        return where / f"{name}{ASSET_EXT}"

    # -- references -----------------------------------------------

    def _reference(self, session: host.Session, ref: host.Reference,
                   recipe: Path, project: Path) -> BuiltAsset:
        """Resolve one ``asset_ref`` (DESIGN 4.6): build what it names
        under the values it offers, and rewrite the node to name the
        IR. A failure that belongs to the reference names its node."""
        try:
            return self._resolve(session, ref, project)
        except BuildError as e:
            if e.detail.get("site") == ref.site:
                e.detail.setdefault("node", ref.node)
            raise

    def _resolve(self, session: host.Session, ref: host.Reference,
                 project: Path) -> BuiltAsset:
        node = session.nodes[ref.node]
        params = dict(node["params"])
        if ref.asset is not None and ref.project is not None:
            raise session.fail(
                "bad_param", ref_op.ONE_LOCATION, ref.site, param="asset",
            )
        if ref.asset is not None and not isinstance(ref.asset, str):
            raise session.fail(
                "bad_param",
                f"asset_ref names its asset by a project-relative path, as "
                f"in \"{SUBASSETS_DIR}/<name>{RECIPE_EXT}\"; got "
                f"{ref.asset!r}",
                ref.site, param="asset",
            )
        if isinstance(ref.asset, str):
            target = self._sub_asset(session, ref, project)
            params["asset"] = ref.asset[: -len(RECIPE_EXT)] + ASSET_EXT
        try:
            spec = ref_op.ref_params(params)
        except MoraineError as e:
            raise session.fail(e.code, e.message, ref.site, **e.detail) from None
        if spec.project is not None:
            target = self._sibling(session, ref, spec.project, project)
        try:
            built = self.asset(target, spec.overrides, ref.site, project)
        except BuildError as e:
            if e.code.startswith("param_") and e.detail.get("site") == ref.site:
                # Where the declarations live: the message's "this asset".
                e.detail.setdefault("asset", _relative(target, self.root))
            session.error = e
            raise
        if spec.output not in built.asset.outputs:
            raise session.fail(
                "ref_unknown_output",
                f"'{_relative(target, self.root)}' declares no output "
                f"'{spec.output}'; its outputs: {sorted(built.asset.outputs)}",
                ref.site, outputs=sorted(built.asset.outputs),
            )
        if built.overrides:
            params["overrides"] = built.overrides
        else:
            params.pop("overrides", None)
        if spec.project is not None:
            # The sibling root's IR by file name, so the reference
            # resolves from the IR alone (DESIGN 4.6).
            own = self._ir_path(
                built.recipe, asset_mod.project_root(built.recipe), "",
            )
            sibling = params.pop("project")
            params = {"project": sibling, "asset": own.name, **params}
        node["params"] = params
        return built

    def _sub_asset(self, session: host.Session, ref: host.Reference,
                   project: Path) -> Path:
        rel = ref.asset
        absolute = PureWindowsPath(rel).is_absolute() or rel.startswith("/")
        if absolute or not (project / rel).resolve().is_relative_to(project):
            raise session.fail(
                "ref_escape",
                f"asset_ref path '{rel}' leaves the project directory; a "
                f"reference path never leaves the project - vendor the "
                f"asset under {SUBASSETS_DIR}/, or name a sibling project "
                f"with project = \"<name>\" instead",
                ref.site, path=rel,
            )
        parts = rel.split("/")
        if (
            len(parts) != 2 or parts[0] != SUBASSETS_DIR
            or not sources_mod.is_asset_file(parts[1])
            or parts[1] in (RECIPE_EXT, "")
        ):
            raise session.fail(
                "ref_layout",
                f"asset_ref path '{ref.asset}' must be "
                f"'{SUBASSETS_DIR}/<name>{RECIPE_EXT}': sub-assets live "
                f"directly under {SUBASSETS_DIR}/, one level deep, and a "
                f"sibling project is named with project = \"<name>\"",
                ref.site, path=ref.asset,
            )
        return project / SUBASSETS_DIR / parts[1]

    def _sibling(self, session: host.Session, ref: host.Reference, name: str,
                 project: Path) -> Path:
        """A sibling project's root asset, by the project's discovery
        identity (DESIGN 4.6): a directory directly under the
        workspace root, resolved inside it."""
        ws_root = project.parent
        if not (ws_root / workspace_mod.MORAINE_DIR).is_dir():
            raise session.fail(
                "ref_no_workspace",
                f"asset_ref names the project '{name}', and "
                f"'{project.name}' is not a member of a workspace; sibling "
                f"projects are the directories beside it under a workspace "
                f"root ('moraine init' scaffolds one) - or vendor the "
                f"asset under {SUBASSETS_DIR}/",
                ref.site, project=name,
            )
        candidate = ws_root / name
        if (
            name.startswith(".") or name.lower() in RESERVED_DIRS
            or not candidate.is_dir()
            or not sources_mod.asset_files(candidate)
        ):
            known = sorted(
                c.name for c in ws_root.iterdir()
                if c.is_dir() and not c.name.startswith(".")
                and c.name.lower() not in RESERVED_DIRS
                and sources_mod.asset_files(c)
            )
            raise session.fail(
                "unknown_project",
                f"no project '{name}' in this workspace; one of: "
                f"{', '.join(known)}",
                ref.site, project=name, projects=known,
            )
        found = sources_mod.asset_files(candidate)
        if len(found) != 1:
            raise session.fail(
                "ambiguous_asset",
                f"referenced project '{name}' holds {len(found)} top-level "
                f"asset files ({', '.join(p.name for p in found)}); a "
                f"project is referenceable when it holds exactly one root "
                f"asset - move definitions under {LIB_DIR}/ and parts "
                f"under {SUBASSETS_DIR}/",
                ref.site, project=name, assets=[p.name for p in found],
            )
        target = found[0]
        if not target.resolve().is_relative_to(ws_root.resolve()):
            raise session.fail(
                "project_outside_workspace",
                f"project '{name}' resolves outside the workspace (a link "
                f"under the workspace root points elsewhere); references "
                f"stay workspace-internal - move the target into the "
                f"workspace, or vendor the asset under {SUBASSETS_DIR}/",
                ref.site, project=name,
            )
        return target


def check_target(recipe: Path) -> None:
    """A build's target is an asset file (DESIGN 4.8): the root or a
    sub-asset, never a requirements file or a ``lib/`` file."""
    name = recipe.name
    if sources_mod.is_requirements_file(name):
        asset = name[: -len(REQUIREMENTS_EXT)] + RECIPE_EXT
        raise BuildError(
            "not_an_asset",
            f"'{name}' is a requirements file, built with its asset; "
            f"name '{asset}' or the project",
            file=name,
        )
    if recipe.parent.name.lower() == LIB_DIR:
        raise BuildError(
            "not_an_asset",
            f"'{LIB_DIR}/{name}' holds definitions, built with the assets "
            f"that load it; name an asset file or the project",
            file=name,
        )
    if not name.lower().endswith(RECIPE_EXT):
        raise BuildError(
            "not_an_asset",
            f"'{name}' is not a recipe; an asset file is <name>{RECIPE_EXT}",
            file=name,
        )


def build(recipe: str | Path, *, offered: dict | None = None,
          write: bool = True, strict_offers: bool = True,
          ranged_offers: bool = True) -> BuildResult:
    """Build an asset file and its reference closure.

    ``offered`` holds the values a consumer offers for the target's own
    knobs, the viewer's overlay; such a build is a preview and is never
    written. ``ranged_offers`` False takes an offered value outside its
    declared range, which the report then fails on the knob's entry
    (a read command's ``--param``). ``write`` False builds to memory. A
    failure carries the recipe files the build had named as ``sources``.
    """
    t0 = time.perf_counter()
    recipe = Path(recipe).resolve()
    check_target(recipe)
    project = asset_mod.project_root(recipe)
    root = workspace_mod.enclosing_root(project)
    build_ = _Build(root, strict_offers, ranged_offers)
    try:
        target = build_.asset(recipe, dict(offered or {}), None, None)
    except BuildError as e:
        e.sources = frozenset(build_.read | {recipe})
        e.contents = build_.files.found(e.sources)
        raise
    result = BuildResult(
        target=target, assets=build_.assets, root=root,
        named=frozenset(build_.read | {recipe}),
    )
    result.contents = build_.files.found(result.named)
    result.unread = sorted(build_.unread)
    result.preview = bool(offered)
    # The target's own recipe alone: a referenced asset is its
    # producer's to be warned about (DESIGN 4.1).
    result.warnings = (
        repetition.find(target.asset) + _tautologies(target, root)
        + _unused(target, root) + list(target.warnings)
    )
    if write:
        write_out(result)
    result.ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def _tautologies(target: BuiltAsset, root: Path) -> list[dict]:
    """The ``threshold_tautology`` warnings (DESIGN 5.2), one per
    constant of the asset file that both builds the geometry and sets
    a threshold of the agent's own assert: the entry moves with the
    constant, so it cannot fail."""
    if target.reading is None:
        return []
    shown = _relative(target.recipe, root)
    by_line: dict[str, list[str]] = {}
    for entry in target.asset.asserts or []:
        if entry.get("owner") == "agent":
            by_line.setdefault(entry.get("site"), []).append(entry["id"])
    requirements = _relative(sources_mod.requirements_file(target.recipe), root)
    out = []
    for shared in target.reading.shared:
        ids = [
            aid for line in shared.judged_at
            for aid in by_line.get(f"{shown}:{line}", [])
        ]
        if not ids:
            continue  # an assert the recipe writes and never reaches
        out.append({
            "warning": "threshold_tautology",
            "constant": shared.name,
            "site": f"{shown}:{shared.line}",
            "count": len(ids),
            "asserts": ids[:LISTED],
            "fix": (
                f"write the number the requirement states in the assert, "
                f"or as a constant of '{requirements}' that the asset loads"
            ),
        })
    return out


def _unused(target: BuiltAsset, root: Path) -> list[dict]:
    """The ``accepted_value_unused`` warning (DESIGN 7.2): the ids the
    project's ``params.json`` holds that its root asset declares no
    knob for. The agent does not edit the human's file, so the line
    stays until the human accepts again."""
    if not target.unused:
        return []
    project = asset_mod.project_root(target.recipe)
    shown = _relative(project / accepted_mod.PARAMS_FILE, root)
    declared = [decl["id"] for decl in target.asset.params or []]
    return [{
        "warning": "accepted_value_unused",
        "file": shown,
        "params": list(target.unused[:LISTED]),
        "count": len(target.unused),
        "declared": declared[:LISTED],
        "fix": (
            f"'{shown}' holds a value for {len(target.unused)} id(s) no "
            f"mo.param of '{_relative(target.recipe, root)}' declares; if "
            f"a knob was renamed, declare it under its old id so the "
            f"accepted value reaches it - the file is the human's, and "
            f"the next accept in the viewer drops the lines"
        ),
    }]


def write_out(result: BuildResult) -> None:
    """Write what a build made under each project's ``build/`` (DESIGN
    4.8), and after the build of a project's root remove from that
    project's ``build/`` what no root of the workspace reaches (DESIGN
    4.5). A preview, built under values a consumer offered for the
    target's own knobs, is never written."""
    if result.preview:
        return
    ordered = sorted(result.assets.values(), key=lambda b: b.ir_path.as_posix())
    # Every path is judged before any file is written, so a build
    # that refuses one has written none.
    for built in ordered:
        _check_inside(built.ir_path)
    for built in ordered:
        if _write(built):
            result.written.append(built.ir_path)
        else:
            result.unchanged += 1
    look(result, remove=_is_root(result.target.recipe))


def _is_root(recipe: Path) -> bool:
    """Whether an asset file is its project's root: only the build of
    the root says what the project's ``build/`` holds."""
    return recipe.parent == asset_mod.project_root(recipe)


def look(result: BuildResult, *, remove: bool = False) -> None:
    """Read the target project's ``build/`` and the roots that reach
    it (``clean.sweep``) into the result. ``remove`` False changes
    nothing on disk."""
    if result.preview:
        return
    result.swept = clean.sweep(
        asset_mod.project_root(result.target.recipe), set(result.assets),
        remove=remove,
    )


# How much of a line a report quotes (DESIGN 8.2): a node's line can
# hold a long list, and the reader needs where it parts, not all of it.
_QUOTED = 400


def _quote(line: str | None) -> str | None:
    if line is None or len(line) <= _QUOTED:
        return line
    return line[:_QUOTED] + f"... ({len(line) - _QUOTED} more characters)"


def compare(result: BuildResult) -> dict:
    """``build --check`` (DESIGN 4.4, 6): the IR a build made in memory
    against the IR on disk, byte for byte. Each file that differs is
    named with its first differing line quoted from both, and each file
    the disk does not hold is named. ``stale`` names the IR of the
    project that no root reaches. Nothing is written or removed."""
    same = 0
    differs: list[dict] = []
    missing: list[str] = []
    for built in sorted(result.assets.values(), key=lambda b: b.ir_path.as_posix()):
        shown = _relative(built.ir_path, result.root)
        try:
            on_disk = built.ir_path.read_bytes()
        except FileNotFoundError:
            missing.append(shown)
            continue
        except OSError as e:
            raise BuildError(
                "asset_unreadable",
                f"cannot read '{shown}' ({e.__class__.__name__}: {e}); "
                f"check that the file is readable",
                file=shown,
            ) from e
        if on_disk == built.text.encode("utf-8"):
            same += 1
            continue
        ours = built.text.split("\n")
        theirs = on_disk.decode("utf-8", "replace").split("\n")
        at = next(
            (i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b),
            min(len(ours), len(theirs)),
        )
        differs.append({
            "file": shown,
            "recipe": _relative(built.recipe, result.root),
            "line": at + 1,
            "built": _quote(ours[at] if at < len(ours) else None),
            "on_disk": _quote(theirs[at] if at < len(theirs) else None),
        })
    out: dict = {"same": same}
    if differs:
        out["differs"] = differs
    if missing:
        out["missing"] = missing
    if result.swept.stale:
        # What the next written build of the root removes (DESIGN 4.8):
        # beside the comparison, not part of it.
        out["stale"] = [_relative(p, result.root) for p in result.swept.stale]
    if differs or missing:
        out["fix"] = (
            "the IR on disk is not what the recipe builds to now; "
            "'moraine build' writes it again"
        )
    return out


def _check_inside(ir_path: Path) -> None:
    """Refuse an IR path a link leads out of the project (DESIGN 4.5):
    the build writes inside the project and nowhere a link leads. The
    nearest directory that exists decides, since the ones below it are
    the build's to create and a link above them carries them along."""
    where = ir_path.parent
    while not where.exists() and where != where.parent:
        where = where.parent
    if where.resolve() != where:
        raise BuildError(
            "build_is_link",
            f"'{where.name}/' is reached through a link, so writing "
            f"'{ir_path.name}' there would leave the project; remove the "
            f"link and let the build create the directory",
            file=ir_path.name,
        )


def _write(built: BuiltAsset) -> bool:
    """Write one IR unless the file holds these bytes already, so a
    build that changes nothing touches nothing."""
    data = built.text.encode("utf-8")
    try:
        if built.ir_path.read_bytes() == data:
            return False
    except OSError:
        pass
    try:
        built.ir_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise BuildError(
            "build_unwritable",
            f"cannot create '{built.ir_path.parent}' "
            f"({e.__class__.__name__}: {e}); check directory permissions",
        ) from e
    asset_mod.write_text(built.ir_path, built.text)
    return True
