"""The usage snippet a recipe declares is published verbatim.

Which makes it the one piece of card metadata that can be wrong in ways nobody
notices: it is copied out of a project's README into a Python literal, rendered
through str.format, and fenced as bash on a public page. Each of those steps has
already mangled one.
"""

from __future__ import annotations

import importlib

import pytest

from mlx_forge.recipes import AVAILABLE_RECIPES


def _metadata(recipe: str):
    return importlib.import_module(AVAILABLE_RECIPES[recipe]).METADATA


DECLARING = sorted(r for r in AVAILABLE_RECIPES if _metadata(r).cli_snippet)


def test_snippets_are_declared_where_they_can_be():
    """A guard on the guard: this file proves nothing if the list is empty."""
    assert len(DECLARING) >= 8, f"only {len(DECLARING)} recipes declare a snippet"


def _declared_literal(recipe: str) -> str:
    """The cli_snippet literal exactly as written in the recipe's source."""
    import ast
    import inspect

    module = importlib.import_module(AVAILABLE_RECIPES[recipe])
    source = inspect.getsource(module)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "cli_snippet":
            segment = ast.get_source_segment(source, node.value)
            assert segment is not None, f"{recipe}: cannot read the cli_snippet literal"
            return segment
    raise AssertionError(f"{recipe}: no cli_snippet keyword found")


@pytest.mark.parametrize("recipe", DECLARING)
def test_a_line_continuation_survives_the_python_literal(recipe):
    """In a non-raw triple-quoted string, a trailing backslash eats its newline.

    `python3 generate.py \\` + newline came out as one long line, silently
    collapsing every multi-line shell command into one. Comparing the literal as
    written against the value Python built is what catches it: a raw string
    keeps both counts equal, a plain one drops them all.
    """
    written = _declared_literal(recipe).count("\\\n")
    built = _metadata(recipe).cli_snippet.count("\\\n")

    # Only a LOSS is a defect. A declaration written with explicit "\\\n"
    # escapes, as the ernie recipes are, holds no real newline after a
    # backslash and legitimately counts zero in the source.
    assert built >= written, (
        f"{recipe}: {written - built} line continuation(s) were swallowed by the "
        'Python literal — declare the snippet as a raw string (r"""...""")'
    )


@pytest.mark.parametrize("recipe", DECLARING)
def test_no_brace_the_renderer_would_choke_on(recipe):
    """The snippet goes through .format(repo_id=...), which any brace reaches.

    A stray `{` — a shell parameter expansion, a dict literal in an embedded
    Python block — raises at render time and takes the whole card down.
    """
    snippet = _metadata(recipe).cli_snippet

    snippet.format(repo_id="acme/demo-mlx")  # must not raise

    braced = {seg.split("}")[0] for seg in snippet.split("{")[1:] if "}" in seg}
    assert braced <= {"repo_id"}, f"{recipe}: unsupported placeholders {braced - {'repo_id'}}"


@pytest.mark.parametrize("recipe", DECLARING)
def test_the_snippet_names_the_repo_it_documents(recipe):
    """Otherwise the same text is published on a model's bf16, q8 and q4 repos.

    `{repo_id}` is the whole reason one declaration can serve every build.
    """
    metadata = _metadata(recipe)
    snippet = metadata.cli_snippet
    if metadata.name == "void-model":
        pytest.skip("void's snippet differs per build and is supplied per repo")

    assert "{repo_id}" in snippet, f"{recipe}: snippet is identical across builds"


@pytest.mark.parametrize("recipe", DECLARING)
def test_the_snippet_points_at_our_own_project(recipe):
    """These weights run through our ports, not through a hypothetical package.

    Only ernie-image-mlx is released on PyPI; everything else is installed from
    its repository and driven by that repository's CLI. A snippet claiming
    otherwise sends a reader to something that does not exist.
    """
    snippet = _metadata(recipe).cli_snippet
    usage_url = _metadata(recipe).usage_url or ""
    project = usage_url.rstrip("/").split("/")[-1]

    assert project, f"{recipe}: declares a snippet but no usage_url to install from"
    assert project in snippet, f"{recipe}: snippet never mentions {project}"


#: Ours, and actually released on PyPI (checked against pypi.org, Aug 2026).
#: Everything else of ours is installed from its repository, so `pip install
#: <name>` would send a reader to a package that does not exist. Third-party
#: dependencies are not this test's business.
OUR_PYPI_PACKAGES = {"ernie-image-mlx", "mlx-forge", "mlx-arsenal"}


@pytest.mark.parametrize("recipe", DECLARING)
def test_our_own_project_is_installed_from_where_it_exists(recipe):
    """`pip install <name>` is a claim that a package is published under it.

    An invented `pip install matrix-game-mlx` reached a draft card once. Only
    three of our projects are on PyPI; the rest are cloned and run from the
    repository, which is what these snippets do.
    """
    metadata = _metadata(recipe)
    project = (metadata.usage_url or "").rstrip("/").split("/")[-1]

    for line in metadata.cli_snippet.splitlines():
        stripped = line.strip()
        if not stripped.startswith("pip install "):
            continue
        installed = {
            token
            for token in stripped[len("pip install ") :].split()
            if not token.startswith(("-", "git+", "http", "#"))
        }
        offending = installed & {project} - OUR_PYPI_PACKAGES
        assert not offending, (
            f"{recipe}: `pip install {project}` — that project is not on PyPI; "
            "install it from its repository, as its README does"
        )
