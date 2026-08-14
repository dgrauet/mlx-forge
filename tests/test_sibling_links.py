"""Per-repo links add to the recipe's, they do not replace them.

A card needs two kinds of link: the ones true of every build of a model, which
belong in the recipe, and the ones true of one repo alone — its sibling builds,
"q8 variant: ...". `--link` used to replace the whole list, so adding a sibling
meant retyping the declaration and freezing a copy of it in the manifest: a link
later added to the recipe would never reach the repos that once passed --link.
"""

from __future__ import annotations

import json

from mlx_forge.upload import card_links, persist_card_metadata, sibling_links

DECLARED = ["Code: https://github.com/acme/tool", "Docs: https://acme.invalid/docs"]
SIBLING = "q8 variant: https://huggingface.co/acme/demo-mlx-q8"


def _info(**over) -> dict:
    return {"recipe": "demo", "source": "acme/Demo", "links": list(DECLARED), **over}


def test_the_card_shows_the_declaration_then_the_repos_own():
    """Declared first, so the common links read alike across bf16/q8/q4."""
    assert card_links(_info(extra_links=[SIBLING])) == [*DECLARED, SIBLING]


def test_a_supplied_link_is_added_not_substituted():
    assert card_links(_info(), [SIBLING]) == [*DECLARED, SIBLING]


def test_the_recipe_stays_live_after_a_sibling_is_recorded(tmp_path):
    """The point of the change: a link added to the recipe later must arrive.

    Storing the operator's links in `links` froze a copy of the declaration, so
    a repo that once took --link stopped tracking the recipe forever.
    """
    info = _info()
    (tmp_path / "split_model.json").write_text(json.dumps(info))

    stored = persist_card_metadata(
        tmp_path, info, usage_url=None, links=[SIBLING], cli_snippet=None
    )

    assert stored["extra_links"] == [SIBLING]
    assert stored["links"] == DECLARED, "the declaration must not be copied into the manifest"

    # The recipe gains a link; the repo picks it up without being touched.
    grown = {**stored, "links": [*DECLARED, "Paper: https://arxiv.invalid/1"]}
    assert card_links(grown) == [*DECLARED, "Paper: https://arxiv.invalid/1", SIBLING]


def test_a_link_the_recipe_already_declares_is_not_recorded_twice(tmp_path):
    info = _info()
    (tmp_path / "split_model.json").write_text(json.dumps(info))

    stored = persist_card_metadata(
        tmp_path, info, usage_url=None, links=[DECLARED[0], SIBLING], cli_snippet=None
    )

    assert stored["extra_links"] == [SIBLING]
    assert card_links(stored).count(DECLARED[0]) == 1


def test_repeating_a_refresh_changes_nothing(tmp_path):
    """--card-only is documented idempotent; siblings must not accumulate."""
    info = _info()
    (tmp_path / "split_model.json").write_text(json.dumps(info))

    once = persist_card_metadata(tmp_path, info, usage_url=None, links=[SIBLING], cli_snippet=None)
    twice = persist_card_metadata(tmp_path, once, usage_url=None, links=[SIBLING], cli_snippet=None)

    assert once == twice
    assert card_links(twice) == [*DECLARED, SIBLING]


def test_nothing_supplied_leaves_the_manifest_alone(tmp_path):
    info = _info()

    assert persist_card_metadata(tmp_path, info, usage_url=None, links=None, cli_snippet=None) is (
        info
    )
    assert not (tmp_path / "split_model.json").exists()


def test_sibling_links_preserves_order_and_drops_duplicates():
    supplied = ["b: 2", "a: 1", "b: 2", DECLARED[0]]
    assert sibling_links(_info(), supplied) == ["b: 2", "a: 1"]


def test_a_manifest_without_a_declaration_still_shows_its_links():
    """A directory converted before the recipe declared any links."""
    assert card_links({"extra_links": [SIBLING]}) == [SIBLING]
    assert card_links({}) == []
