"""Tests for split_model utility."""

import json

import mlx.core as mx
import pytest

from mlx_forge.split import split_model


class TestSplitModel:
    def _create_unified(self, tmp_path, weights, filename="model.safetensors"):
        mx.save_safetensors(str(tmp_path / filename), weights)

    def test_basic_split(self, tmp_path):
        weights = {
            "transformer.block.0.weight": mx.zeros((4, 4)),
            "transformer.block.1.weight": mx.ones((4, 4)),
            "vae.conv.weight": mx.zeros((2, 2)),
        }
        self._create_unified(tmp_path, weights)

        component_map = {
            "transformer": "transformer.safetensors",
            "vae": "vae.safetensors",
        }
        result = split_model(tmp_path, component_map)

        assert result["transformer.safetensors"] == 2
        assert result["vae.safetensors"] == 1

        # Verify files exist and can be loaded
        t_weights = mx.load(str(tmp_path / "transformer.safetensors"))
        assert len(t_weights) == 2
        v_weights = mx.load(str(tmp_path / "vae.safetensors"))
        assert len(v_weights) == 1

    def test_unmatched_keys_go_to_fallback(self, tmp_path):
        weights = {
            "transformer.weight": mx.zeros((2, 2)),
            "unknown.weight": mx.zeros((2, 2)),
        }
        self._create_unified(tmp_path, weights)

        component_map = {"transformer": "transformer.safetensors"}
        result = split_model(tmp_path, component_map, fallback_filename="other.safetensors")

        assert result["transformer.safetensors"] == 1
        assert result["other.safetensors"] == 1

    def test_unmatched_keys_skipped_when_no_fallback(self, tmp_path):
        weights = {
            "transformer.weight": mx.zeros((2, 2)),
            "unknown.weight": mx.zeros((2, 2)),
        }
        self._create_unified(tmp_path, weights)

        component_map = {"transformer": "transformer.safetensors"}
        result = split_model(tmp_path, component_map, fallback_filename=None)

        assert result["transformer.safetensors"] == 1
        assert "unknown" not in str(result)

    def test_marker_file_written(self, tmp_path):
        weights = {"transformer.weight": mx.zeros((2, 2))}
        self._create_unified(tmp_path, weights)

        split_model(tmp_path, {"transformer": "transformer.safetensors"})

        marker = tmp_path / "split_model.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["split"] is True
        assert "transformer.safetensors" in data["files"]

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            split_model(tmp_path, {"a": "a.safetensors"})

    def test_custom_source_filename(self, tmp_path):
        weights = {"comp.weight": mx.zeros((2, 2))}
        self._create_unified(tmp_path, weights, filename="unified.safetensors")

        result = split_model(
            tmp_path,
            {"comp": "comp.safetensors"},
            source_filename="unified.safetensors",
        )
        assert result["comp.safetensors"] == 1

    def test_empty_model(self, tmp_path):
        self._create_unified(tmp_path, {})
        result = split_model(tmp_path, {"transformer": "transformer.safetensors"})
        assert result == {}

    def test_multiple_keys_same_component(self, tmp_path):
        weights = {
            "encoder.layer1.weight": mx.zeros((3, 3)),
            "encoder.layer2.weight": mx.zeros((3, 3)),
            "encoder.layer3.bias": mx.zeros((3,)),
        }
        self._create_unified(tmp_path, weights)

        result = split_model(tmp_path, {"encoder": "encoder.safetensors"})
        assert result["encoder.safetensors"] == 3
