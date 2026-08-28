"""The quantized presentation of a model card.

A q8 repo has more to say than "MLX format conversion of X": at which width and
group size, over which weights, and from which bf16 build. The void-model cards
said all of it by hand, which is why they were the last two the template could
not reproduce.

It is opt-in per recipe, through `quantization_scope`. Nine quantized repos are
already published with the plain presentation and regenerate losslessly;
switching them is a decision to take repo by repo, not a side effect of adding
a template branch.
"""

from __future__ import annotations

import json

import pytest

from mlx_forge.upload import generate_model_card, quantize_command, unquantized_repo

SCOPE = "transformer Linear weights only"


def _card(tmp_path, **split) -> str:
    """A rendered card. usage_url/cli_snippet are call arguments, not manifest keys."""
    repo = split.pop("_repo", "acme/demo-mlx-q8")
    usage_url = split.pop("usage_url", None)
    cli_snippet = split.pop("cli_snippet", None)
    info = {"recipe": "demo", "source": "acme/Demo", "base_model": "acme/Demo", **split}
    return generate_model_card(
        tmp_path,
        split_info=info,
        config={},
        repo_id=repo,
        usage_url=usage_url,
        cli_snippet=cli_snippet,
        file_listing={"model.safetensors": 10},
    )


def _quantized(tmp_path, *, bits=8, **over) -> str:
    return _card(
        tmp_path,
        quantized=True,
        quantization_bits=bits,
        quantization_group_size=64,
        quantization_scope=SCOPE,
        **over,
    )


# ---------------------------------------------------------------------------
# Derived, not declared
# ---------------------------------------------------------------------------


def test_the_bf16_repo_is_derived_from_the_name():
    """The tool owns the naming convention; declaring it again would drift."""
    assert unquantized_repo("dgrauet/void-model-mlx-q8", 8) == "dgrauet/void-model-mlx"
    assert unquantized_repo("dgrauet/void-model-mlx-q4", 4) == "dgrauet/void-model-mlx"


def test_an_unquantized_repo_has_no_bf16_counterpart():
    assert unquantized_repo("dgrauet/void-model-mlx", None) is None
    assert unquantized_repo("dgrauet/void-model-mlx", 8) is None


def test_the_quantize_command_names_the_variant_when_there_is_one():
    base = {"recipe": "ernie-image"}
    assert quantize_command(base, 8) == "mlx-forge convert ernie-image --quantize --bits 8"
    assert quantize_command({**base, "variant": "sft"}, 8) == (
        "mlx-forge convert ernie-image --variant sft --quantize --bits 8"
    )
    assert quantize_command(base, None) is None
    assert quantize_command({}, 8) is None


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_the_opening_states_width_group_size_and_scope(tmp_path):
    card = _quantized(tmp_path)

    assert "Int8 quantization (group_size 64, transformer Linear weights only) of" in card
    assert "[acme/demo-mlx](https://huggingface.co/acme/demo-mlx)" in card
    assert "conversion of [acme/Demo](https://huggingface.co/acme/Demo)" in card
    assert "MLX format conversion of" not in card, "the plain opening must be replaced"


def test_the_command_that_produced_it_is_shown(tmp_path):
    card = _quantized(tmp_path)
    assert "`mlx-forge convert demo --quantize --bits 8`" in card


def test_the_front_matter_is_tagged(tmp_path):
    card = _quantized(tmp_path, bits=4)
    front = card.split("---\n", 2)[1]

    assert "  - quantized\n" in front
    assert "  - int4\n" in front


def test_a_per_build_note_is_published_verbatim(tmp_path):
    note = "**This is the 32 GB configuration**: peaks at ~23.7 GB.\nPSNR = 35.5 dB."
    card = _quantized(tmp_path, build_note=note)

    assert note in card


def test_the_legacy_notes_table_is_not_mistaken_for_prose(tmp_path):
    """Six published manifests use `notes` for a {component: explanation} map.

    Rendering one as a paragraph raised UndefinedError and took the card down —
    hence `build_note` for the prose, and a guard for the day both appear.
    """
    card = _quantized(
        tmp_path,
        notes={"vocoder": "Also contains BWE generator weights."},
        build_note="Peaks at ~23.7 GB.",
    )

    assert "Peaks at ~23.7 GB." in card
    assert "vocoder" not in card


def test_a_non_string_build_note_is_ignored_rather_than_fatal(tmp_path):
    assert "## Files" in _quantized(tmp_path, build_note={"unexpected": "shape"})


def test_the_quantize_config_note_follows_the_snippet(tmp_path):
    card = _quantized(tmp_path, usage_url="https://github.com/acme/run", cli_snippet="run it")

    assert "Keep `quantize_config.json` next to the weights" in card
    assert card.index("run it") < card.index("Keep `quantize_config.json`")


def test_a_snippet_turns_the_usage_sentence_into_a_lead_in(tmp_path):
    """It introduces the block that follows, so it ends in a colon."""
    with_snippet = _quantized(tmp_path, usage_url="https://github.com/acme/run", cli_snippet="x")
    without = _quantized(tmp_path, usage_url="https://github.com/acme/run")

    assert "[run](https://github.com/acme/run):" in with_snippet
    assert "[run](https://github.com/acme/run)." in without
    # The blank line before the fence must survive Jinja's trim_blocks.
    assert "):\n\n```bash" in with_snippet


def test_the_redundant_quantization_bullet_is_dropped(tmp_path):
    """The width is already in the opening sentence; twice is noise."""
    assert "- **Quantization:** int8" not in _quantized(tmp_path)


# ---------------------------------------------------------------------------
# Opt-in: the nine published plain cards must not move
# ---------------------------------------------------------------------------


def test_without_a_declared_scope_the_card_keeps_its_plain_form(tmp_path):
    """Nine quantized repos regenerate losslessly today; they must keep doing so."""
    card = _card(tmp_path, quantized=True, quantization_bits=8, quantization_group_size=64)

    assert "MLX format conversion of [acme/Demo]" in card
    assert "Converted with [mlx-forge]" in card
    assert "- **Quantization:** int8" in card
    assert "Int8 quantization (" not in card
    assert "mlx-forge convert demo --quantize" not in card


def test_an_unquantized_card_is_untouched(tmp_path):
    card = _card(tmp_path, quantization_scope=SCOPE)

    assert "MLX format conversion of [acme/Demo]" in card
    assert "Int8 quantization" not in card
    assert "  - quantized\n" not in card


# ---------------------------------------------------------------------------
# Where the numbers come from
# ---------------------------------------------------------------------------


def test_width_is_read_from_quantize_config_when_the_manifest_is_silent(tmp_path):
    """void published three repos whose manifest predates any quantize flag.

    quantize_config.json is written by the quantizer itself, so it is the
    authority — and the group size lives nowhere else regardless.
    """
    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 32}})
    )
    card = _card(tmp_path, quantization_scope=SCOPE, _repo="acme/demo-mlx-q4")

    assert "Int4 quantization (group_size 32, transformer Linear weights only)" in card
    assert "  - int4\n" in card.split("---\n", 2)[1]


def test_the_quantization_record_is_recovered_into_the_manifest(tmp_path):
    """A --card-only refresh works from metadata alone, so the manifest must say it.

    Nine published quantized repos recorded the width but never the group size,
    which lived only in quantize_config.json next to the weights.
    """
    from mlx_forge.upload import backfill_quantization

    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"quantization": {"bits": 8, "group_size": 64}})
    )
    info = {"recipe": "demo", "source": "acme/Demo", "quantized": True, "quantization_bits": 8}
    (tmp_path / "split_model.json").write_text(json.dumps(info))

    merged = backfill_quantization(tmp_path, info)

    assert merged["quantization_group_size"] == 64
    stored = json.loads((tmp_path / "split_model.json").read_text())
    assert stored["quantization_group_size"] == 64


def test_the_recovery_never_overrules_the_manifest(tmp_path):
    """The manifest describes this build; quantize_config only fills its gaps."""
    from mlx_forge.upload import backfill_quantization

    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 32}})
    )
    info = {"quantized": True, "quantization_bits": 8, "quantization_group_size": 64}

    assert backfill_quantization(tmp_path, info) == info
    assert not (tmp_path / "split_model.json").exists()


def test_an_unquantized_model_is_left_alone(tmp_path):
    from mlx_forge.upload import backfill_quantization

    info = {"recipe": "demo"}
    assert backfill_quantization(tmp_path, info) is info
    assert not (tmp_path / "split_model.json").exists()


def test_the_manifest_wins_over_quantize_config_for_the_width(tmp_path):
    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"quantization": {"bits": 4, "group_size": 32}})
    )
    card = _card(
        tmp_path,
        quantized=True,
        quantization_bits=8,
        quantization_group_size=64,
        quantization_scope=SCOPE,
    )

    assert "Int8 quantization (group_size 64," in card


@pytest.mark.parametrize("recipe", ["void-model"])
def test_void_declares_what_its_quantizer_touches(recipe):
    import importlib

    from mlx_forge.recipes import AVAILABLE_RECIPES

    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe]).METADATA
    assert metadata.quantization_scope == SCOPE
    assert metadata.as_split_fields()["quantization_scope"] == SCOPE


#: Recipes whose --quantize does something, and must therefore be able to say
#: what. A recipe that never quantizes is not required to declare a scope.
QUANTIZING_RECIPES = [
    "cogvideox-fun-v1.5-5b-inp",
    "ernie-image",
    "ernie-image-pe",
    "ltx-2.3",
    "void-model",
]


@pytest.mark.parametrize("recipe", QUANTIZING_RECIPES)
def test_a_quantizing_recipe_says_what_it_quantizes(recipe):
    """Otherwise its q4/q8 cards can only say "MLX format conversion of X".

    Nine published quantized repos carried exactly that: a card that never
    mentioned the width it was quantized at, let alone over which weights.
    """
    import importlib

    from mlx_forge.recipes import AVAILABLE_RECIPES

    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe]).METADATA

    scope = metadata.quantization_scope
    assert scope, f"{recipe} quantizes but does not declare a quantization_scope"
    assert not scope.endswith("."), "it is spliced into a sentence, so it carries no full stop"
    assert scope[0].islower(), "likewise it starts mid-sentence"


@pytest.mark.parametrize("recipe", QUANTIZING_RECIPES)
def test_the_declared_scope_reads_as_a_clause(recipe, tmp_path):
    """It is spliced into "Int8 quantization (group_size 64, <scope>) of ..."."""
    import importlib

    from mlx_forge.recipes import AVAILABLE_RECIPES

    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe]).METADATA
    card = generate_model_card(
        tmp_path,
        split_info={
            **metadata.as_split_fields(),
            "quantized": True,
            "quantization_bits": 8,
            "quantization_group_size": 64,
        },
        config={},
        repo_id=f"acme/{metadata.name}-mlx-q8",
        file_listing={},
    )

    assert f"Int8 quantization (group_size 64, {metadata.quantization_scope}) of" in card


def test_the_base_model_link_is_not_declared_by_the_void_recipe():
    """It differs per build: bf16 pairs with bf16, q4 with the q8 base.

    Declared, it put a base model on every quantized card that that card does
    not use, next to the right one.
    """
    import importlib

    from mlx_forge.recipes import AVAILABLE_RECIPES

    metadata = importlib.import_module(AVAILABLE_RECIPES["void-model"]).METADATA
    assert not any("Base model weights" in link for link in metadata.links)


def test_every_quantizing_recipe_declares_its_scope():
    """A q4/q8 card must say which layers were quantized. Recipes that
    support --quantize but declare no quantization_scope shipped cards
    that named the bit width and nothing else."""
    import argparse
    import importlib

    from mlx_forge.recipes import AVAILABLE_RECIPES

    missing = []
    for name in AVAILABLE_RECIPES:
        # AVAILABLE_RECIPES maps a recipe name to its module path (the same
        # helper tests/test_license_compliance.py defines locally).
        recipe = importlib.import_module(AVAILABLE_RECIPES[name])
        parser = argparse.ArgumentParser()
        recipe.add_convert_args(parser)
        supports_quantize = any("--quantize" in a.option_strings for a in parser._actions)
        metadata = getattr(recipe, "METADATA", None)
        if supports_quantize and metadata is not None and not metadata.quantization_scope:
            missing.append(name)
    assert missing == [], f"quantizing recipes without quantization_scope: {missing}"
