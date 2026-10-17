"""The build: recipe to IR (DESIGN 4.8).

A recipe is a Starlark file; the build runs it and writes the IR the
rest of the runtime reads. The layer sits directly below the entry
points, which alone import it, and imports only the floor, the data
model, and the operators, whose registries it generates its builtins
from: a build never evaluates, an evaluation never builds (DESIGN 3).
"""
