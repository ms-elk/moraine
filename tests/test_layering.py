"""Package layering (DESIGN 3): folders are layers, and a layer imports
only itself and the layers below it.

The folder layout under ``src/moraine3d`` is the dependency structure
made visible (the stack diagram in DESIGN 3); these tests are what keep
the two the same. Lazy (function-level) imports count: they defer the
cost of an edge, not its direction.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from conftest import SRC_ROOT, package_sources
from moraine3d.recipe.build import build
from support.recipes import write_files

PKG = SRC_ROOT.name
# The import name of the recipe language's interpreter.
INTERPRETER = "starlark"

# Bottom-up, as DESIGN 3 draws them. A module's layer is its folder;
# the root-level modules are placed by what they are (``errors`` is the
# floor; the package root carries only ``dist_version`` and sits on the
# floor with it; ``__main__`` is the CLI entry, an app concern).
LAYERS = ["errors", "core", "cache", "ops", "verify", "engine", "recipe", "app"]
ROOT_MODULE_LAYER = {"__init__": "errors", "errors": "errors", "__main__": "app"}


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PKG, *parts])


MODULES = {_module_name(p): p for p in package_sources()}


def _layer_of(module: str) -> str:
    path = MODULES[module]
    if path.parent == SRC_ROOT:
        return ROOT_MODULE_LAYER[path.stem]
    return module.split(".")[1]


def _imports(module: str) -> list[tuple[int, str]]:
    """(line, absolute imported module) for every intra-package import,
    at any depth (lazy imports included)."""
    path = MODULES[module]
    is_pkg = path.name == "__init__.py"
    pkg_parts = module.split(".") if is_pkg else module.split(".")[:-1]
    out: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == PKG:
                    out.append((node.lineno, a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod.split(".")[0] != PKG:
                continue
            for a in node.names:
                # "from .x import y" may name a module or a symbol; the
                # layer is the same either way, so resolve to the deepest
                # module that exists.
                cand = f"{mod}.{a.name}"
                out.append((node.lineno, cand if cand in MODULES else mod))
    return out


EDGES = {m: _imports(m) for m in MODULES}


def _edge(module: str, line: int, target: str) -> str:
    return f"{MODULES[module].relative_to(SRC_ROOT).as_posix()}:{line}: {module} -> {target}"


def test_every_module_sits_in_a_declared_layer():
    """DESIGN 3: each layer is a folder of the same name, except ``errors``,
    a single module at the package root — so every module resolves to a
    declared layer and no stray root-level module appears."""
    undeclared = []
    for module, path in sorted(MODULES.items()):
        if path.parent == SRC_ROOT:
            if path.stem not in ROOT_MODULE_LAYER:
                undeclared.append(module)
        elif module.split(".")[1] not in LAYERS:
            undeclared.append(module)
    assert undeclared == [], undeclared


def test_layers_import_only_downward():
    """DESIGN 3: a layer imports only itself and what lies below it, lazy
    imports included."""
    violations = [
        _edge(module, line, target)
        for module, edges in sorted(EDGES.items())
        for line, target in edges
        if LAYERS.index(_layer_of(target)) > LAYERS.index(_layer_of(module))
    ]
    assert violations == [], "\n".join(violations)


def test_the_build_knows_only_the_floor_the_model_and_the_operators():
    """DESIGN 3: ``recipe`` imports only ``errors``, ``core``, and
    ``ops`` — a build never evaluates, so the layers between the
    operators and the build are closed to it though they lie below."""
    violations = [
        _edge(module, line, target)
        for module, edges in sorted(EDGES.items())
        if _layer_of(module) == "recipe"
        for line, target in edges
        if _layer_of(target) not in ("errors", "core", "ops", "recipe")
    ]
    assert violations == [], "\n".join(violations)


def test_the_runtime_past_the_build_names_no_recipe_file():
    """DESIGN 3: which file is a recipe and which projects a workspace
    holds are read by the build and the entry points; the layers that
    read the IR (``cache``, ``ops``, ``verify``, ``engine``) import
    neither module."""
    closed = {f"{PKG}.core.sources", f"{PKG}.core.workspace"}
    violations = [
        _edge(module, line, target)
        for module, edges in sorted(EDGES.items())
        if _layer_of(module) in ("cache", "ops", "verify", "engine")
        for line, target in edges
        if target in closed
    ]
    assert violations == [], "\n".join(violations)


def test_the_interpreter_is_imported_by_the_build_alone():
    """DESIGN 3: the recipe language's interpreter is a dependency of
    ``recipe`` and of nothing else, lazy imports included."""
    violations = []
    for module, path in sorted(MODULES.items()):
        if _layer_of(module) == "recipe":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == INTERPRETER for name in names):
                violations.append(f"{path.relative_to(SRC_ROOT).as_posix()}:{node.lineno}")
    assert violations == [], "\n".join(violations)


def test_an_ir_evaluates_where_the_interpreter_cannot_be_imported(tmp_path):
    """DESIGN 3, 4: nothing past the build knows how the graph was
    written, so an IR and the IRs it references evaluate in a process
    that cannot import the interpreter."""
    project = write_files(tmp_path / "yard", {
        "yard.star": 'mo.output("p", mo.asset_ref("subassets/part.star", "p", id = "p"))\n',
        "subassets/part.star": 'mo.output("p", mo.box(id = "p", size = [1, 1, 1]))\n',
    })
    ir = build(project / "yard.star").target.ir_path
    script = (
        "import sys\n"
        f"sys.modules[{INTERPRETER!r}] = None\n"
        "from moraine3d.core import asset\n"
        "from moraine3d.engine.evaluator import evaluate_asset\n"
        "from moraine3d.engine.export import export_asset\n"
        "from moraine3d.verify import validate\n"
        "sc = asset.load(sys.argv[1])\n"
        "out = evaluate_asset(sc, source_of=lambda ir: None, stamp=sc.build_version)\n"
        "assert out['p']['stats']['bounds']['max'] == [0.5, 0.5, 0.5], out\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(ir)], capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr


def test_ops_know_only_core_and_their_shared_helpers():
    """DESIGN 3: an operator knows only ``core`` and the ``ops`` package
    below its root (the shared helpers and its siblings) — never the
    registry or the engine — so adding an operator cannot break another
    and every operator is testable alone. The
    registry (``ops/__init__``) imports every operator; the reverse edge
    would be a cycle. ``cache`` sits below ``ops`` in the stack but is
    the evaluator's concern, not an operator's."""
    registry = f"{PKG}.ops"
    violations = [
        _edge(module, line, target)
        for module, edges in sorted(EDGES.items())
        if module.startswith(registry + ".")
        for line, target in edges
        if not (
            _layer_of(target) in ("errors", "core")
            or (target.startswith(registry + ".") and target != registry)
        )
    ]
    assert violations == [], "\n".join(violations)


def test_assert_kinds_know_only_kernels_and_their_shared_helpers():
    """DESIGN 3: an assert kind module, on either side of the layer line,
    knows the kernels and its side's shared helpers (``common``) — never
    a registry (its package root), a sibling kind, or the module that
    dispatches to it (``core.asset``, ``verify.validate``) — so a kind
    is added by two files and two registry lines and is testable alone.
    Of the parse-side package the evaluation side reads its own twin
    and the shared ``common``, nothing else."""
    parse_side, check_side = f"{PKG}.core.asserts", f"{PKG}.verify.asserts"
    dispatchers = {f"{PKG}.core.asset", f"{PKG}.verify.validate"}
    violations = []
    for package in (parse_side, check_side):
        for module, edges in sorted(EDGES.items()):
            if not module.startswith(package + "."):
                continue
            stem = module.rsplit(".", 1)[1]
            twins = {f"{parse_side}.common", f"{parse_side}.{stem}"}
            for line, target in edges:
                closed = (
                    target == package
                    or target in dispatchers
                    or (target.startswith(package + ".") and target != f"{package}.common")
                    or (
                        package == check_side
                        and (target == parse_side or target.startswith(parse_side + "."))
                        and target not in twins
                    )
                )
                if closed:
                    violations.append(_edge(module, line, target))
    assert violations == [], "\n".join(violations)


def test_no_import_cycles():
    """DESIGN 3: the import graph is acyclic."""
    graph = {m: {t for _, t in edges if t in MODULES} for m, edges in EDGES.items()}
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(m: str) -> None:
        state[m] = 1
        stack.append(m)
        for t in sorted(graph[m]):
            s = state.get(t, 0)
            if s == 0:
                visit(t)
            elif s == 1:
                cycles.append(stack[stack.index(t):] + [t])
        stack.pop()
        state[m] = 2

    for m in sorted(graph):
        if state.get(m, 0) == 0:
            visit(m)
    assert cycles == [], cycles
