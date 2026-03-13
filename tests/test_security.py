"""Security tests for path traversal and safe deserialization."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from mlx_forge.convert import _validate_path_within, load_weights


class TestValidatePathWithin:
    def test_valid_child(self, tmp_path):
        child = tmp_path / "file.safetensors"
        assert _validate_path_within(child, tmp_path) == child.resolve()

    def test_valid_nested_child(self, tmp_path):
        child = tmp_path / "sub" / "file.safetensors"
        assert _validate_path_within(child, tmp_path) == child.resolve()

    def test_traversal_blocked(self, tmp_path):
        malicious = tmp_path / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_path_within(malicious, tmp_path)

    def test_dot_dot_in_middle(self, tmp_path):
        malicious = tmp_path / "sub" / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="Path traversal detected"):
            _validate_path_within(malicious, tmp_path)

    def test_parent_itself_is_valid(self, tmp_path):
        assert _validate_path_within(tmp_path, tmp_path) == tmp_path.resolve()


class TestLoadWeightsPathTraversal:
    def test_malicious_index_blocked(self, tmp_path):
        """A malicious index.json with path traversal in weight_map should be rejected."""
        # Create a malicious index.json
        index = {
            "weight_map": {
                "layer.weight": "../../malicious.safetensors",
            }
        }
        index_path = tmp_path / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f)

        with pytest.raises(ValueError, match="Path traversal detected"):
            load_weights(tmp_path)

    def test_valid_index_loads(self, tmp_path):
        """A valid index.json with normal shard names should load fine."""
        # Create a valid shard
        weights = {"layer.weight": mx.zeros((4, 4))}
        mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), weights)

        index = {
            "weight_map": {
                "layer.weight": "model-00001-of-00001.safetensors",
            }
        }
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        result = load_weights(tmp_path)
        assert "layer.weight" in result
