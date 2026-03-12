"""Integration tests for the full conversion pipeline using synthetic tensor data.

Tests end-to-end: load -> classify -> sanitize -> process -> save -> reload.
No network access required -- everything uses fake checkpoints with realistic key names.
"""

from __future__ import annotations

import json

import mlx.core as mx

from mlx_forge.convert import classify_keys, process_component
from mlx_forge.quantize import quantize_weights
from mlx_forge.recipes.fish_s2 import (
    COMPONENT_PREFIX as FISH_COMPONENT_PREFIX,
)
from mlx_forge.recipes.fish_s2 import (
    COMPONENTS as FISH_COMPONENTS,
)
from mlx_forge.recipes.fish_s2 import (
    FISH_S2_SPLIT_MAP,
    fish_s2_should_quantize,
)
from mlx_forge.recipes.fish_s2 import (
    SANITIZERS as FISH_SANITIZERS,
)
from mlx_forge.recipes.fish_s2 import (
    classify_key as fish_classify_key,
)
from mlx_forge.recipes.ltx_23 import (
    COMPONENT_PREFIX as LTX_COMPONENT_PREFIX,
)
from mlx_forge.recipes.ltx_23 import (
    COMPONENTS as LTX_COMPONENTS,
)
from mlx_forge.recipes.ltx_23 import (
    LTX23_SPLIT_MAP,
    ltx23_should_quantize,
    maybe_transpose,
)
from mlx_forge.recipes.ltx_23 import (
    SANITIZERS as LTX_SANITIZERS,
)
from mlx_forge.recipes.ltx_23 import (
    classify_key as ltx_classify_key,
)
from mlx_forge.split import split_model

# ---------------------------------------------------------------------------
# Fake checkpoint builders
# ---------------------------------------------------------------------------


def _make_ltx23_checkpoint() -> dict[str, mx.array]:
    """Create a minimal fake LTX-2.3 checkpoint with realistic keys."""
    weights: dict[str, mx.array] = {}

    # Transformer keys
    weights["model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"] = (
        mx.random.normal((256, 128))
    )
    weights["model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias"] = (
        mx.random.normal((256,))
    )
    weights["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"] = mx.random.normal(
        (128, 128)
    )
    weights["model.diffusion_model.transformer_blocks.0.attn1.to_k.weight"] = mx.random.normal(
        (128, 128)
    )
    weights["model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"] = mx.random.normal(
        (128, 128)
    )
    weights["model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"] = mx.random.normal(
        (256, 128)
    )
    weights["model.diffusion_model.transformer_blocks.0.ff.net.2.weight"] = mx.random.normal(
        (128, 256)
    )
    weights["model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight"] = (
        mx.random.normal((128, 128))
    )

    # Connector keys
    weights["model.diffusion_model.video_embeddings_connector.linear.weight"] = mx.random.normal(
        (128, 64)
    )
    weights["model.diffusion_model.audio_embeddings_connector.linear.weight"] = mx.random.normal(
        (128, 64)
    )
    weights["text_embedding_projection.aggregate_embed.weight"] = mx.random.normal((128, 64))

    # VAE decoder (conv weights need transposition)
    weights["vae.decoder.conv_in.weight"] = mx.random.normal((64, 32, 3, 3, 3))
    weights["vae.decoder.mid_block.resnets.0.norm1.weight"] = mx.random.normal((64,))

    # VAE encoder
    weights["vae.encoder.conv_in.weight"] = mx.random.normal((64, 32, 3, 3, 3))
    weights["vae.encoder.down_blocks.0.resnets.0.norm1.weight"] = mx.random.normal((64,))

    # Shared VAE stats
    weights["vae.per_channel_statistics.mean-of-means"] = mx.random.normal((128,))
    weights["vae.per_channel_statistics.std-of-means"] = mx.random.normal((128,))

    # Audio VAE
    weights["audio_vae.decoder.conv_in.weight"] = mx.random.normal((64, 32, 3))
    weights["audio_vae.per_channel_statistics.mean-of-means"] = mx.random.normal((64,))
    weights["audio_vae.per_channel_statistics.std-of-means"] = mx.random.normal((64,))

    # Vocoder (conv1d weights)
    weights["vocoder.conv_pre.weight"] = mx.random.normal((512, 128, 7))
    weights["vocoder.ups.0.weight"] = mx.random.normal((256, 512, 16))
    weights["vocoder.conv_post.weight"] = mx.random.normal((1, 512, 7))

    return weights


def _make_fish_s2_checkpoint() -> dict[str, mx.array]:
    """Create a minimal fake Fish S2 Pro checkpoint with realistic keys."""
    weights: dict[str, mx.array] = {}

    # Text model keys (Qwen3-style)
    weights["text_model.model.embeddings.weight"] = mx.random.normal((1024, 256))
    weights["text_model.model.norm.weight"] = mx.random.normal((256,))
    for i in range(2):
        prefix = f"text_model.model.layers.{i}"
        weights[f"{prefix}.attention.wqkv.weight"] = mx.random.normal((768, 256))
        weights[f"{prefix}.attention.wo.weight"] = mx.random.normal((256, 256))
        weights[f"{prefix}.attention.q_norm.weight"] = mx.random.normal((128,))
        weights[f"{prefix}.attention.k_norm.weight"] = mx.random.normal((128,))
        weights[f"{prefix}.attention_norm.weight"] = mx.random.normal((256,))
        weights[f"{prefix}.feed_forward.w1.weight"] = mx.random.normal((512, 256))
        weights[f"{prefix}.feed_forward.w2.weight"] = mx.random.normal((256, 512))

    # Audio decoder keys
    weights["audio_decoder.codebook_embeddings.weight"] = mx.random.normal((10, 256))
    weights["audio_decoder.output.weight"] = mx.random.normal((4096, 256))
    for i in range(2):
        prefix = f"audio_decoder.layers.{i}"
        weights[f"{prefix}.attention.wqkv.weight"] = mx.random.normal((768, 256))
        weights[f"{prefix}.attention.wo.weight"] = mx.random.normal((256, 256))
        weights[f"{prefix}.feed_forward.w1.weight"] = mx.random.normal((512, 256))

    return weights


# ---------------------------------------------------------------------------
# LTX-2.3 integration tests
# ---------------------------------------------------------------------------


class TestLtx23Pipeline:
    """End-to-end tests for the LTX-2.3 conversion pipeline."""

    def test_classify_all_keys(self):
        """All fake checkpoint keys are classified into known components."""
        checkpoint = _make_ltx23_checkpoint()
        by_component = classify_keys(checkpoint, ltx_classify_key)

        assert "transformer" in by_component
        assert "connector" in by_component
        assert "vae_decoder" in by_component
        assert "vae_encoder" in by_component
        assert "audio_vae" in by_component
        assert "vocoder" in by_component
        assert "vae_shared_stats" in by_component

        # Every key should be classified (none left over)
        total_classified = sum(len(v) for v in by_component.values())
        assert total_classified == len(checkpoint)

    def test_full_convert_pipeline(self, tmp_path):
        """Convert fake LTX-2.3 checkpoint end-to-end and verify output files."""
        checkpoint = _make_ltx23_checkpoint()
        output_dir = tmp_path / "ltx23-mlx"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, ltx_classify_key)

        total = 0
        for comp_name in LTX_COMPONENTS:
            keys = by_component.get(comp_name, [])
            if not keys:
                continue
            count = process_component(
                checkpoint,
                comp_name,
                keys,
                output_dir,
                LTX_COMPONENT_PREFIX[comp_name],
                sanitizer=LTX_SANITIZERS[comp_name],
                transform=maybe_transpose,
            )
            total += count

        assert total > 0

        # Verify output files exist and are loadable
        for comp_name in ["transformer", "connector", "vae_decoder", "vae_encoder", "vocoder"]:
            out_file = output_dir / f"{comp_name}.safetensors"
            assert out_file.exists(), f"{comp_name}.safetensors missing"
            loaded = mx.load(str(out_file))
            assert len(loaded) > 0

    def test_key_sanitization_in_output(self, tmp_path):
        """Verify keys are properly sanitized in the output files."""
        checkpoint = _make_ltx23_checkpoint()
        output_dir = tmp_path / "ltx23-sanitized"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, ltx_classify_key)

        # Process transformer
        keys = by_component["transformer"]
        process_component(
            checkpoint,
            "transformer",
            keys,
            output_dir,
            LTX_COMPONENT_PREFIX["transformer"],
            sanitizer=LTX_SANITIZERS["transformer"],
            transform=maybe_transpose,
        )

        loaded = mx.load(str(output_dir / "transformer.safetensors"))
        output_keys = set(loaded.keys())

        # No PyTorch prefix should remain
        assert not any("model.diffusion_model." in k for k in output_keys)
        # Keys should have component prefix
        assert all(k.startswith("transformer.") for k in output_keys)
        # Sanitization rules applied
        assert not any(".to_out.0." in k for k in output_keys)
        assert not any(".ff.net." in k for k in output_keys)
        assert not any(".linear_1." in k for k in output_keys)
        # Sanitized forms present
        assert any(".to_out." in k for k in output_keys)
        assert any(".ff.proj_in." in k for k in output_keys)
        assert any(".ff.proj_out." in k for k in output_keys)
        assert any(".linear1." in k for k in output_keys)

    def test_conv_transposition(self, tmp_path):
        """Conv weights in VAE components are transposed to MLX layout."""
        checkpoint = _make_ltx23_checkpoint()
        output_dir = tmp_path / "ltx23-conv"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, ltx_classify_key)

        # Process vae_decoder which has conv weights
        process_component(
            checkpoint,
            "vae_decoder",
            by_component["vae_decoder"],
            output_dir,
            LTX_COMPONENT_PREFIX["vae_decoder"],
            sanitizer=LTX_SANITIZERS["vae_decoder"],
            transform=maybe_transpose,
        )

        loaded = mx.load(str(output_dir / "vae_decoder.safetensors"))
        conv_key = "vae_decoder.conv_in.weight"
        assert conv_key in loaded

        # Original: (O, I, D, H, W) = (64, 32, 3, 3, 3)
        # Transposed to MLX: (O, D, H, W, I) = (64, 3, 3, 3, 32)
        assert loaded[conv_key].shape == (64, 3, 3, 3, 32)

    def test_transformer_not_transposed(self, tmp_path):
        """Transformer Linear weights should NOT be transposed."""
        checkpoint = _make_ltx23_checkpoint()
        output_dir = tmp_path / "ltx23-notrans"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, ltx_classify_key)

        process_component(
            checkpoint,
            "transformer",
            by_component["transformer"],
            output_dir,
            LTX_COMPONENT_PREFIX["transformer"],
            sanitizer=LTX_SANITIZERS["transformer"],
            transform=maybe_transpose,
        )

        loaded = mx.load(str(output_dir / "transformer.safetensors"))
        q_key = "transformer.transformer_blocks.0.attn1.to_q.weight"
        assert q_key in loaded
        # Shape should remain (128, 128) -- no transposition
        assert loaded[q_key].shape == (128, 128)

    def test_convert_then_quantize(self, tmp_path):
        """Convert and then quantize transformer weights; verify scales/biases."""
        checkpoint = _make_ltx23_checkpoint()
        output_dir = tmp_path / "ltx23-quant"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, ltx_classify_key)

        # Convert transformer component
        process_component(
            checkpoint,
            "transformer",
            by_component["transformer"],
            output_dir,
            LTX_COMPONENT_PREFIX["transformer"],
            sanitizer=LTX_SANITIZERS["transformer"],
            transform=maybe_transpose,
        )

        # Load and quantize
        tf_path = output_dir / "transformer.safetensors"
        weights = mx.load(str(tf_path))
        quantized = quantize_weights(
            weights, bits=8, group_size=64, should_quantize=ltx23_should_quantize
        )

        mx.save_safetensors(str(tf_path), quantized)

        # Reload and verify quantization artifacts
        reloaded = mx.load(str(tf_path))
        q_keys = [k for k in reloaded if k.endswith(".scales")]
        b_keys = [k for k in reloaded if k.endswith(".biases")]

        assert len(q_keys) > 0, "No .scales keys found after quantization"
        assert len(b_keys) > 0, "No .biases keys found after quantization"
        assert len(q_keys) == len(b_keys), "Mismatch between .scales and .biases counts"

        # Original weight keys should still exist (now as uint32 quantized)
        block_weight_key = "transformer.transformer_blocks.0.attn1.to_q.weight"
        assert block_weight_key in reloaded
        assert reloaded[block_weight_key].dtype == mx.uint32

    def test_empty_component_skipped(self, tmp_path):
        """A component with no keys produces no output file."""
        checkpoint = {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": mx.random.normal(
                (128, 128)
            ),
        }
        output_dir = tmp_path / "ltx23-empty"
        output_dir.mkdir()

        count = process_component(
            checkpoint,
            "vocoder",
            [],
            output_dir,
            LTX_COMPONENT_PREFIX["vocoder"],
            sanitizer=LTX_SANITIZERS["vocoder"],
            transform=maybe_transpose,
        )

        assert count == 0
        assert not (output_dir / "vocoder.safetensors").exists()

    def test_round_trip_values(self, tmp_path):
        """Save fake weights, reload, and verify values match exactly."""
        checkpoint = _make_ltx23_checkpoint()
        output_dir = tmp_path / "ltx23-roundtrip"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, ltx_classify_key)

        process_component(
            checkpoint,
            "connector",
            by_component["connector"],
            output_dir,
            LTX_COMPONENT_PREFIX["connector"],
            sanitizer=LTX_SANITIZERS["connector"],
            transform=maybe_transpose,
        )

        loaded = mx.load(str(output_dir / "connector.safetensors"))

        # text_embedding_projection key is kept as-is by sanitizer
        orig_key = "text_embedding_projection.aggregate_embed.weight"
        out_key = f"connector.{orig_key}"
        assert out_key in loaded

        orig_val = checkpoint[orig_key]
        loaded_val = loaded[out_key]
        assert mx.allclose(orig_val, loaded_val).item()


# ---------------------------------------------------------------------------
# Fish S2 Pro integration tests
# ---------------------------------------------------------------------------


class TestFishS2Pipeline:
    """End-to-end tests for the Fish S2 Pro conversion pipeline."""

    def test_classify_all_keys(self):
        """All fake checkpoint keys are classified into known components."""
        checkpoint = _make_fish_s2_checkpoint()
        by_component = classify_keys(checkpoint, fish_classify_key)

        assert "text_model" in by_component
        assert "audio_decoder" in by_component

        total_classified = sum(len(v) for v in by_component.values())
        assert total_classified == len(checkpoint)

    def test_full_convert_pipeline(self, tmp_path):
        """Convert fake Fish S2 checkpoint end-to-end."""
        checkpoint = _make_fish_s2_checkpoint()
        output_dir = tmp_path / "fish-s2-mlx"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, fish_classify_key)

        total = 0
        for comp_name in FISH_COMPONENTS:
            keys = by_component.get(comp_name, [])
            if not keys:
                continue
            count = process_component(
                checkpoint,
                comp_name,
                keys,
                output_dir,
                FISH_COMPONENT_PREFIX[comp_name],
                sanitizer=FISH_SANITIZERS[comp_name],
            )
            total += count

        assert total > 0

        # Only text_model and audio_decoder have keys in the fake checkpoint (no codec)
        for comp_name in ["text_model", "audio_decoder"]:
            out_file = output_dir / f"{comp_name}.safetensors"
            assert out_file.exists(), f"{comp_name}.safetensors missing"
            loaded = mx.load(str(out_file))
            assert len(loaded) > 0

    def test_key_sanitization_text_model(self, tmp_path):
        """Text model keys have text_model.model. prefix stripped."""
        checkpoint = _make_fish_s2_checkpoint()
        output_dir = tmp_path / "fish-s2-sanitized"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, fish_classify_key)

        process_component(
            checkpoint,
            "text_model",
            by_component["text_model"],
            output_dir,
            FISH_COMPONENT_PREFIX["text_model"],
            sanitizer=FISH_SANITIZERS["text_model"],
        )

        loaded = mx.load(str(output_dir / "text_model.safetensors"))
        output_keys = set(loaded.keys())

        # No "text_model.model." prefix should remain
        assert not any("text_model.model." in k for k in output_keys)
        # All keys should have the component prefix
        assert all(k.startswith("text_model.") for k in output_keys)
        # Specific checks
        assert "text_model.embeddings.weight" in output_keys
        assert "text_model.layers.0.attention.wqkv.weight" in output_keys

    def test_key_sanitization_audio_decoder(self, tmp_path):
        """Audio decoder keys have audio_decoder. prefix stripped and re-added."""
        checkpoint = _make_fish_s2_checkpoint()
        output_dir = tmp_path / "fish-s2-audio"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, fish_classify_key)

        process_component(
            checkpoint,
            "audio_decoder",
            by_component["audio_decoder"],
            output_dir,
            FISH_COMPONENT_PREFIX["audio_decoder"],
            sanitizer=FISH_SANITIZERS["audio_decoder"],
        )

        loaded = mx.load(str(output_dir / "audio_decoder.safetensors"))
        output_keys = set(loaded.keys())

        assert all(k.startswith("audio_decoder.") for k in output_keys)
        assert "audio_decoder.codebook_embeddings.weight" in output_keys
        assert "audio_decoder.output.weight" in output_keys

    def test_convert_then_quantize(self, tmp_path):
        """Convert and quantize Fish S2 text_model; verify scales/biases."""
        checkpoint = _make_fish_s2_checkpoint()
        output_dir = tmp_path / "fish-s2-quant"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, fish_classify_key)

        process_component(
            checkpoint,
            "text_model",
            by_component["text_model"],
            output_dir,
            FISH_COMPONENT_PREFIX["text_model"],
            sanitizer=FISH_SANITIZERS["text_model"],
        )

        tm_path = output_dir / "text_model.safetensors"
        weights = mx.load(str(tm_path))
        quantized = quantize_weights(
            weights, bits=8, group_size=64, should_quantize=fish_s2_should_quantize
        )
        mx.save_safetensors(str(tm_path), quantized)

        reloaded = mx.load(str(tm_path))
        scales = [k for k in reloaded if k.endswith(".scales")]
        biases = [k for k in reloaded if k.endswith(".biases")]

        assert len(scales) > 0
        assert len(biases) > 0

        # Embeddings should NOT be quantized
        emb_key = "text_model.embeddings.weight"
        assert emb_key in reloaded
        assert reloaded[emb_key].dtype != mx.uint32

        # Linear weights should be quantized
        linear_key = "text_model.layers.0.attention.wqkv.weight"
        assert linear_key in reloaded
        assert reloaded[linear_key].dtype == mx.uint32

    def test_round_trip_values(self, tmp_path):
        """Save and reload Fish S2 weights; verify values match."""
        checkpoint = _make_fish_s2_checkpoint()
        output_dir = tmp_path / "fish-s2-roundtrip"
        output_dir.mkdir()

        by_component = classify_keys(checkpoint, fish_classify_key)

        process_component(
            checkpoint,
            "audio_decoder",
            by_component["audio_decoder"],
            output_dir,
            FISH_COMPONENT_PREFIX["audio_decoder"],
            sanitizer=FISH_SANITIZERS["audio_decoder"],
        )

        loaded = mx.load(str(output_dir / "audio_decoder.safetensors"))

        orig_key = "audio_decoder.codebook_embeddings.weight"
        out_key = "audio_decoder.codebook_embeddings.weight"
        assert out_key in loaded

        orig_val = checkpoint[orig_key]
        loaded_val = loaded[out_key]
        assert mx.allclose(orig_val, loaded_val).item()


# ---------------------------------------------------------------------------
# Split pipeline integration tests
# ---------------------------------------------------------------------------


class TestSplitPipeline:
    """Integration tests for splitting a unified model into components."""

    def test_split_ltx23_unified(self, tmp_path):
        """Split a fake unified LTX-2.3 model and verify output."""
        # Build a unified model with component-prefixed keys
        unified = {
            "transformer.block.0.weight": mx.random.normal((128, 128)),
            "transformer.block.1.weight": mx.random.normal((128, 128)),
            "connector.linear.weight": mx.random.normal((64, 64)),
            "text_embedding_projection.weight": mx.random.normal((64, 64)),
            "vae_decoder.conv.weight": mx.random.normal((32, 32)),
            "vae_encoder.conv.weight": mx.random.normal((32, 32)),
            "vocoder.conv.weight": mx.random.normal((16, 16)),
            "audio_vae.conv.weight": mx.random.normal((16, 16)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        result = split_model(tmp_path, LTX23_SPLIT_MAP)

        assert "transformer.safetensors" in result
        assert "connector.safetensors" in result
        assert "vae_decoder.safetensors" in result
        assert "vae_encoder.safetensors" in result
        assert "vocoder.safetensors" in result
        assert "audio_vae.safetensors" in result

        # connector.safetensors should have both connector and text_embedding_projection
        conn_weights = mx.load(str(tmp_path / "connector.safetensors"))
        assert "connector.linear.weight" in conn_weights
        assert "text_embedding_projection.weight" in conn_weights

        # split_model.json should be written
        marker = tmp_path / "split_model.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["split"] is True
        assert "transformer.safetensors" in data["files"]

    def test_split_fish_s2_unified(self, tmp_path):
        """Split a fake unified Fish S2 model and verify output."""
        unified = {
            "text_model.embeddings.weight": mx.random.normal((256, 128)),
            "text_model.layers.0.weight": mx.random.normal((128, 128)),
            "audio_decoder.output.weight": mx.random.normal((256, 128)),
            "audio_decoder.layers.0.weight": mx.random.normal((128, 128)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        result = split_model(tmp_path, FISH_S2_SPLIT_MAP)

        assert "text_model.safetensors" in result
        assert "audio_decoder.safetensors" in result

        # Verify loadable
        tm = mx.load(str(tmp_path / "text_model.safetensors"))
        assert len(tm) == 2
        ad = mx.load(str(tmp_path / "audio_decoder.safetensors"))
        assert len(ad) == 2

    def test_split_model_json_content(self, tmp_path):
        """Verify split_model.json has correct structure and counts."""
        unified = {
            "text_model.weight": mx.random.normal((64, 64)),
            "audio_decoder.weight": mx.random.normal((64, 64)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        split_model(tmp_path, FISH_S2_SPLIT_MAP)

        data = json.loads((tmp_path / "split_model.json").read_text())
        assert data["split"] is True
        assert data["files"]["text_model.safetensors"] == 1
        assert data["files"]["audio_decoder.safetensors"] == 1

    def test_split_round_trip_values(self, tmp_path):
        """Values survive the split pipeline unchanged."""
        original_tensor = mx.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        unified = {"transformer.weight": original_tensor}
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        split_model(tmp_path, LTX23_SPLIT_MAP)

        loaded = mx.load(str(tmp_path / "transformer.safetensors"))
        assert mx.allclose(loaded["transformer.weight"], original_tensor).item()


# ---------------------------------------------------------------------------
# Cross-recipe quantization tests
# ---------------------------------------------------------------------------


class TestQuantizationIntegration:
    """Quantization integration tests across recipes."""

    def test_quantize_preserves_non_quantizable(self, tmp_path):
        """Non-quantizable weights keep their original dtype and values."""
        weights = {
            "transformer.transformer_blocks.0.attn1.to_q.weight": mx.random.normal((128, 128)),
            "transformer.adaln_single.emb.timestep_embedder.linear1.bias": mx.random.normal((256,)),
            "transformer.scale_shift_table": mx.random.normal((5, 128)),
        }

        # Save the bias value before quantization
        bias_key = "transformer.adaln_single.emb.timestep_embedder.linear1.bias"
        original_bias = weights[bias_key]
        mx.eval(original_bias)  # noqa: S307 -- mlx.core.eval, not builtins.eval

        quantized = quantize_weights(
            weights, bits=8, group_size=64, should_quantize=ltx23_should_quantize
        )

        # Bias should be unchanged
        assert bias_key in quantized
        assert mx.allclose(quantized[bias_key], original_bias).item()

        # scale_shift_table should be unchanged
        sst_key = "transformer.scale_shift_table"
        assert sst_key in quantized

    def test_4bit_quantization(self, tmp_path):
        """4-bit quantization produces valid output."""
        weights = {
            "layers.0.attention.wqkv.weight": mx.random.normal((256, 256)),
            "layers.0.attention_norm.weight": mx.random.normal((256,)),
        }

        quantized = quantize_weights(
            weights, bits=4, group_size=64, should_quantize=fish_s2_should_quantize
        )

        assert "layers.0.attention.wqkv.scales" in quantized
        assert "layers.0.attention.wqkv.biases" in quantized
        assert quantized["layers.0.attention.wqkv.weight"].dtype == mx.uint32

        # norm weight should be untouched
        assert quantized["layers.0.attention_norm.weight"].dtype != mx.uint32

    def test_quantize_skips_incompatible_shapes(self):
        """Weights with last dim not divisible by group_size are skipped."""
        weights = {
            "layers.0.attention.wqkv.weight": mx.random.normal((256, 100)),
        }

        quantized = quantize_weights(
            weights, bits=8, group_size=64, should_quantize=fish_s2_should_quantize
        )

        # Should be kept as-is (not quantized) because 100 % 64 != 0
        assert "layers.0.attention.wqkv.weight" in quantized
        assert "layers.0.attention.wqkv.scales" not in quantized
