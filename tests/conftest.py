"""Shared fixtures.

Every test runs with ``MORAINE_CACHE_DIR`` pointed at a per-test
temporary directory (tests that need their own still override it), so
no test can ever write into the developer's real user cache, and with
``MORAINE_CACHE_MAX_BYTES=0`` (eviction disabled) so an ambient cap in
the developer's environment can never evict the per-test caches
mid-run — existing tests assume records persist for their duration.
Tests that exercise the cap set their own value. The cache module's
session write counter is process-global, so it is reset around each
test to keep the eviction trigger deterministic. The server channels'
loopback pre-check reads "no server" for every test, so no test waits
on a real port (a firewalled host makes a closed port a timeout, not a
refusal); tests that stand a server in through a urlopen mock let the
check through, and the one test of the real socket imports the
original function.

Also the one source walk the structural tests share: ``SRC_ROOT`` is the
package directory and ``package_sources()`` yields its Python files
(vendored code excluded), so a second vendored tree is excluded in one
place.
"""

from pathlib import Path

import pytest

from moraine3d.app import cli as cli_mod
from moraine3d.app import session as session_mod
from moraine3d.cache import store as cache_mod

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "moraine3d"
# The IR corpus of the build tests (DESIGN 10): one project per
# directory, the IR the shipped example of the same name was written
# from, kept as the fixture the translator and the equivalence read.
IR_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ir"


def package_sources():
    """Every Python file under the package, vendored code excluded, in a
    stable order."""
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "vendor" not in path.parts:
            yield path


@pytest.fixture(autouse=True)
def _fixture_stamp(request, monkeypatch):
    """The tests' build stamps its IR ``fixture``, the stamp the IR
    fixtures carry (DESIGN 4.1): a fixture written by hand then reads
    as an IR this build wrote, and none has to be rewritten when the
    vocabulary moves the real stamp. A test about the stamp itself
    asks for the real one with ``@pytest.mark.real_stamp``."""
    if request.node.get_closest_marker("real_stamp") is None:
        from moraine3d.recipe import version

        monkeypatch.setattr(version, "build_version", lambda: "fixture")


@pytest.fixture(autouse=True)
def _cache_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("MORAINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MORAINE_CACHE_MAX_BYTES", "0")
    cache_mod.take_wrote()
    # The judged-dir memo is keyed by directory, so per-test dir swaps
    # re-judge naturally; the reset just keeps tests order-independent.
    monkeypatch.setattr(cache_mod, "_judged_dirs", set())
    # Session log isolation (DESIGN 6): the suite never appends to the
    # machine-level log, and per-process annotation state never leaks
    # between tests that call main() in-process.
    monkeypatch.setenv(session_mod.LOG_DIR_ENV, str(tmp_path / "log"))
    monkeypatch.setattr(session_mod, "_warned", False)
    monkeypatch.setattr(session_mod, "_parse_cache", {})
    session_mod.reset()
    # No server on the port, at once (DESIGN 6 pre-check): the suite
    # owns no socket, and a probe must never wait on a real one.
    monkeypatch.setattr(cli_mod, "_reachable", lambda port: False)
    yield
    cache_mod.take_wrote()
    session_mod.reset()
