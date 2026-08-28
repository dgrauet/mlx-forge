"""Tests for split_model utility."""

import gc
import json
import weakref

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

    def test_marker_merges_existing_manifest(self, tmp_path):
        """split must not clobber convert's manifest (recipe, gating, licence)."""
        weights = {"transformer.weight": mx.zeros((2, 2))}
        self._create_unified(tmp_path, weights)
        marker = tmp_path / "split_model.json"
        marker.write_text(json.dumps({"recipe": "ltx-2.3", "gated": True}))

        split_model(tmp_path, {"transformer": "transformer.safetensors"})

        data = json.loads(marker.read_text())
        assert data["recipe"] == "ltx-2.3"
        assert data["gated"] is True
        assert data["split"] is True
        assert data["files"]["transformer.safetensors"] == 1

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(SystemExit):
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

    def test_every_component_is_materialized_before_save(self, tmp_path, monkeypatch):
        """CLAUDE.md: always materialize before mx.save_safetensors. Lazy
        tensors save as zeros; split must not be the one path that skips it."""
        import mlx_forge.split as split_mod

        seen: list[int] = []
        real = split_mod._materialize

        def spy(*tensors):
            seen.append(len(tensors))
            real(*tensors)

        monkeypatch.setattr(split_mod, "_materialize", spy)
        weights = {"a.w": mx.ones((2, 2)), "b.w": mx.ones((2, 2)), "b.v": mx.ones((2,))}
        self._create_unified(tmp_path, weights)

        split_model(tmp_path, {"a": "a.safetensors", "b": "b.safetensors"})

        assert seen == [1, 2]  # one call per component, all its tensors at once

    def test_each_component_is_actually_released_after_saving(self, tmp_path, monkeypatch):
        """Regression test for the "free between components" claim.

        Before the fix, `all_weights` (the `mx.load` dict) and `sorted_items`
        kept a reference to every array for the whole save loop, so
        `weights.clear()` + `gc.collect()` + `mx.clear_cache()` had nothing
        left to reclaim. This proves an array is actually collected right
        after its component is saved: a weakref taken on one tensor of a
        component dies by the time that component's `gc.collect()` runs, which
        is only possible if no other object (all_weights, sorted_items, or a
        lingering loop-local) still holds it.

        This does NOT prove peak process memory stays low, or that MLX's own
        GPU-side buffers are freed (that's `mx.clear_cache()`'s job, not
        Python's) — only that the Python-level reference chain is severed
        component by component, as the fix and comment claim.
        """
        import mlx_forge.split as split_mod

        weights = {"a.w": mx.ones((4, 4)), "b.w": mx.ones((4, 4))}
        self._create_unified(tmp_path, weights)

        refs: list[weakref.ReferenceType] = []
        real_save = split_mod.mx.save_safetensors

        def spy_save(path, arrs):
            # Snapshot a weakref to this component's tensor before it is
            # saved and cleared, so we can check it dies at the next
            # gc.collect() — i.e. this component only, not the whole run.
            refs.append(weakref.ref(next(iter(arrs.values()))))
            real_save(path, arrs)

        monkeypatch.setattr(split_mod.mx, "save_safetensors", spy_save)

        collect_calls = []
        real_collect = gc.collect

        def spy_collect():
            real_collect()
            collect_calls.append(refs[-1]() is None if refs else None)

        monkeypatch.setattr(split_mod.gc, "collect", spy_collect)

        split_model(tmp_path, {"a": "a.safetensors", "b": "b.safetensors"})

        # One gc.collect() per component, and each component's tensor is
        # already gone by the time its own collect() runs.
        assert collect_calls == [True, True]
