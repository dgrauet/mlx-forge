"""quantize_config.json: the record of how a model was quantized.

`validate` decides whether to run the scales/biases checks by looking for this
file. matrix-game-3.0 recorded quantization only in split_model.json while its
validate gated on quantize_config.json — so `is_quantized` was always False and
`validate_quantization()` never ran, even on a quantized model. A recipe must
not gate on a file it never writes.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from mlx_forge.quantize import (
    QUANTIZE_CONFIG_FILENAME,
    read_quantize_config,
    write_quantize_config,
)
from mlx_forge.recipes import AVAILABLE_RECIPES

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)


class TestRoundTrip:
    def test_write_then_read(self, tmp_path):
        write_quantize_config(tmp_path, bits=4, group_size=32)
        assert read_quantize_config(tmp_path) == {"bits": 4, "group_size": 32}

    def test_absent_file_reads_as_none(self, tmp_path):
        assert read_quantize_config(tmp_path) is None

    def test_extra_fields_are_preserved(self, tmp_path):
        write_quantize_config(
            tmp_path,
            bits=8,
            group_size=64,
            skip_components=["vae"],
            only_transformer_blocks=True,
        )
        qconfig = read_quantize_config(tmp_path)
        assert qconfig is not None
        assert qconfig["skip_components"] == ["vae"]
        assert qconfig["only_transformer_blocks"] is True

    def test_file_shape_is_stable(self, tmp_path):
        """Downstream runtimes read this file — keep the nesting as published."""
        path = write_quantize_config(tmp_path, bits=8, group_size=64)
        assert path.name == QUANTIZE_CONFIG_FILENAME
        assert json.loads(path.read_text()) == {"quantization": {"bits": 8, "group_size": 64}}

    def test_written_next_to_the_weights(self, tmp_path):
        assert write_quantize_config(tmp_path, bits=8, group_size=64).parent == tmp_path


def _source_of(module) -> str:
    """A recipe module's source text."""
    path = module.__file__
    assert path is not None, f"{module.__name__} has no file on disk"
    return Path(path).read_text()


def _called_names(module) -> set[str]:
    """Every function name called anywhere in a recipe module's source."""
    tree = ast.parse(_source_of(module))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_a_recipe_that_reads_the_config_also_writes_it(recipe_name: str):
    """Gating validation on a file the recipe never produces disables the check.

    A recipe may legitimately use neither (hunyuan3d-2.1 records quantization
    inside split_model.json and reads it back from there). What it may not do
    is read without writing.
    """
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])
    called = _called_names(module)

    if "read_quantize_config" in called:
        assert "write_quantize_config" in called, (
            f"{recipe_name}: validate() gates on quantize_config.json but convert() "
            "never writes it — the quantization checks can never run"
        )


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_no_recipe_hand_rolls_the_file(recipe_name: str):
    """The filename must not be re-derived: that is how the two sides drifted."""
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])
    source = _source_of(module)

    offenders = [
        line.strip()
        for line in source.splitlines()
        if QUANTIZE_CONFIG_FILENAME in line
        and ("open(" in line or "/ " in line)
        and not line.strip().startswith("#")
    ]
    assert offenders == [], f"{recipe_name}: use write/read_quantize_config instead"


def test_quantize_file_writes_the_shared_record(tmp_path):
    """The generic quantizer must write the same file the helper writes —
    hand-written JSON here is the drift the helper was introduced to remove."""
    import mlx.core as mx

    from mlx_forge.quantize import QUANTIZE_CONFIG_FILENAME, quantize_file, read_quantize_config

    src = tmp_path / "m.safetensors"
    mx.save_safetensors(str(src), {"layer.weight": mx.ones((64, 64))})
    quantize_file(src, bits=4, group_size=32, config_path=tmp_path / QUANTIZE_CONFIG_FILENAME)

    assert read_quantize_config(tmp_path) == {"bits": 4, "group_size": 32}
