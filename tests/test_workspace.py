"""Workspace detection and discovery (DESIGN 4.2): root-direct project
sweep with deterministic ordering, and the layout-conflict refusal."""

import os
import sys

import pytest

from moraine3d.core import workspace
from moraine3d.errors import MoraineError
from support.links import link_dir as _link_dir


def _mk_project(root, name):
    d = root / name
    d.mkdir()
    (d / f"{name}.star").write_text("", encoding="utf-8")
    return d


def test_workspace_asset_conflict_is_refused(tmp_path):
    """A directory holding both .moraine/ and a top-level asset file is a
    diagnosed layout conflict at every classification site, never a
    classification (DESIGN 4.2)."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    assert workspace.is_workspace(ws)
    assert not workspace.is_project(ws)

    (ws / "asset.star").write_text("", encoding="utf-8")
    assert workspace.is_project(ws)
    for classify in (workspace.is_workspace, workspace.discover,
                     workspace.enclosing_root):
        with pytest.raises(MoraineError) as e:
            classify(ws)
        assert e.value.code == "workspace_asset_conflict"
        assert "child project directory" in e.value.message

    # No .moraine/, no asset file: neither — resolution falls through to the
    # ordinary asset_not_found path.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert not workspace.is_workspace(bare)
    assert not workspace.is_project(bare)


def test_conflicted_parent_fails_member_framing(tmp_path):
    """A workspace member's records are framed from the workspace root
    (DESIGN 4.2), so a conflicted parent is a refusal, not a silent
    fall-back to project-directory framing."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    proj = _mk_project(ws, "alpha")
    assert workspace.enclosing_root(proj) == ws.resolve()

    (ws / "stray.star").write_text("", encoding="utf-8")
    with pytest.raises(MoraineError) as e:
        workspace.enclosing_root(proj)
    assert e.value.code == "workspace_asset_conflict"


def test_discovery_skips_dot_and_reserved_dirs(tmp_path):
    """Discovery examines only immediate child directories, skipping
    dot-directories and the reserved layout names (subassets, lib,
    build, exports, external, custom), and orders projects
    lexicographically (DESIGN 4.2)."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _mk_project(ws, "beta")
    _mk_project(ws, "alpha")
    for name in ("subassets", "lib", "build", "exports", "external", "custom"):
        (ws / name / "inner").mkdir(parents=True)
        (ws / name / "inner" / "decoy.star").write_text("", encoding="utf-8")
    (ws / ".git").mkdir()
    (ws / ".git" / "decoy.star").write_text("", encoding="utf-8")
    # A mora-less child dir and a plain file are not projects.
    (ws / "notes").mkdir()
    (ws / "README.md").write_text("", encoding="utf-8")

    found = workspace.discover(ws)
    assert found.projects == ("alpha", "beta")
    assert found.project_dir("alpha") == ws / "alpha"


def test_asset_dirs_are_not_projects(tmp_path):
    """A *directory* named `*.star` is not an asset (DESIGN 4.2: a project
    holds a top-level asset file)."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    # A stray directory named like an asset must not make the root a
    # project (which would hide the workspace entirely).
    (ws / "old.star").mkdir()
    assert not workspace.is_project(ws)
    assert workspace.is_workspace(ws)
    # A child whose only ".star" entry is a directory is not a project.
    (ws / "notes").mkdir()
    (ws / "notes" / "x.star").mkdir()

    assert workspace.discover(ws).projects == ()


@pytest.mark.parametrize(
    "name", ["subassets", "lib", "build", "exports", "external", "custom", "Exports"],
)
def test_a_asset_under_a_reserved_name_is_refused(tmp_path, name):
    """DESIGN 4.2: a directory of a reserved name that holds a top-level
    asset file is refused wherever a project is classified, the sweep
    and the project addressed alike, with the rename named. The names
    match case-insensitively."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _mk_project(ws, "alpha")
    taken = _mk_project(ws, name)

    for classify, target in (
        (workspace.discover, ws), (workspace.enclosing_root, taken),
    ):
        with pytest.raises(MoraineError) as e:
            classify(target)
        assert e.value.code == "reserved_project_name"
        assert e.value.detail["directory"] == name
        assert "rename the directory" in e.value.message
    assert workspace.enclosing_root(ws / "alpha") == ws.resolve()


def test_a_standalone_project_under_a_reserved_name_is_refused(tmp_path):
    """DESIGN 4.2: the reserved names are names a project directory may
    contain, never be, inside a workspace or outside one."""
    taken = _mk_project(tmp_path, "external")
    with pytest.raises(MoraineError) as e:
        workspace.enclosing_root(taken)
    assert e.value.code == "reserved_project_name"


def test_resolve_asset_validates_membership(tmp_path):
    """Workspace.resolve_asset validates roster membership before any
    filesystem access, so an externally supplied project name can never
    reach the filesystem as a path (DESIGN 7.2 target restriction)."""
    import pytest

    from moraine3d.errors import MoraineError

    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _mk_project(ws, "alpha")
    found = workspace.discover(ws)

    assert found.resolve_asset("alpha") == ws / "alpha" / "alpha.star"
    with pytest.raises(MoraineError) as e:
        found.resolve_asset("../escape")
    assert e.value.code == "unknown_project"
    assert e.value.detail["projects"] == ["alpha"]


def test_resolve_asset_enforces_workspace_containment(tmp_path):
    """DESIGN 4.6: the resolved asset file must lie inside the resolved
    workspace root, checked at the choke point every consumer of a
    project name shares — a link-spelled sibling is legal while its
    target stays internal, one pointing outside the workspace is
    refused."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ext.star").write_text("", encoding="utf-8")
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _mk_project(ws, "alpha")
    try:
        _link_dir(outside, ws / "gamma")
        _link_dir(ws / "alpha", ws / "delta")
    except OSError as e:  # e.g. symlink-restricted environment
        pytest.skip(f"cannot create a directory link here: {e}")

    found = workspace.discover(ws)
    assert found.projects == ("alpha", "delta", "gamma")
    # Internal link: inside the resolved root, one namespace — legal.
    assert found.resolve_asset("delta").resolve() == (
        ws / "alpha" / "alpha.star"
    ).resolve()
    # External link: discovery lists the name, use refuses it.
    with pytest.raises(MoraineError) as e:
        found.resolve_asset("gamma")
    assert e.value.code == "project_outside_workspace"
    assert "workspace-internal" in e.value.message


def test_discovery_is_root_direct_only(tmp_path):
    """Projects live directly under the workspace root — a project nested
    one level deeper is not discovered (DESIGN 4.2: arbitrary-depth
    discovery is future work)."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    group = ws / "group"
    group.mkdir()
    _mk_project(group, "nested")

    assert workspace.discover(ws).projects == ()


def test_unswept_assets_names_what_discovery_misses(tmp_path):
    """The sweep names the asset files below the project level it
    did not reach (DESIGN 4.2 `not_swept`): a nested project's root
    asset and an asset under a plain child directory, root-relative and
    sorted; roster projects (and their sub-assets), a nested project's
    own sub-assets, dot-directories, and the root's reserved names are
    not listed."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    alpha = _mk_project(ws, "alpha")
    (alpha / "subassets").mkdir()
    (alpha / "subassets" / "kit.star").write_text("", encoding="utf-8")
    (ws / "group").mkdir()
    nested = _mk_project(ws / "group", "nested")
    (nested / "subassets").mkdir()
    (nested / "subassets" / "own.star").write_text("", encoding="utf-8")
    (ws / "docs" / "old").mkdir(parents=True)
    (ws / "docs" / "old" / "thing.star").write_text("", encoding="utf-8")
    (ws / ".hidden").mkdir()
    (ws / ".hidden" / "h.star").write_text("", encoding="utf-8")
    (ws / "exports" / "old").mkdir(parents=True)
    (ws / "exports" / "old" / "e.star").write_text("", encoding="utf-8")

    disc = workspace.discover(ws)
    assert disc.projects == ("alpha",)
    assert workspace.unswept_assets(disc) == [
        "docs/old/thing.star", "group/nested/nested.star",
    ]


def test_unswept_assets_does_not_follow_links(tmp_path):
    """A link back into the workspace cannot manufacture `not_swept` rows
    (DESIGN 4.2): links are not followed and no directory is entered
    twice, so the empty list stays the positive statement."""
    import os
    import sys

    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _mk_project(ws, "alpha")
    (ws / "group").mkdir()
    link = ws / "group" / "loop"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(ws), str(link))
    else:
        os.symlink(ws, link, target_is_directory=True)
    disc = workspace.discover(ws)
    assert disc.projects == ("alpha",)
    assert workspace.unswept_assets(disc) == []


def test_the_sweep_names_assets_and_never_definitions(tmp_path):
    """DESIGN 4.2: the sweep names the asset files below the project
    level it did not reach. A `lib/` file holds definitions and a
    `build/` file is IR, so neither is an asset anywhere, and an asset
    file is one whatever the case of its extension."""
    ws = tmp_path / "ws"
    (ws / ".moraine").mkdir(parents=True)
    _mk_project(ws, "alpha")
    for rel in ("group/lib/parts.star", "group/build/old.star",
                "group/nested/Nested.STAR", "group/nested/lib/more.star"):
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    found = workspace.discover(ws)
    assert workspace.unswept_assets(found) == ["group/nested/Nested.STAR"]
