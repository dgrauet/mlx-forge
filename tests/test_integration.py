"""Integration tests for the full conversion pipeline using synthetic tensor data.

Tests end-to-end: load -> classify -> sanitize -> process -> save -> reload.
No network access required -- everything uses fake checkpoints with realistic key names.
"""

from __future__ import annotations

import json

import mlx.core as mx

from mlx_forge.cli import main
from mlx_forge.convert import classify_keys, load_safetensors, process_component
from mlx_forge.quantize import quantize_weights
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
            loaded = load_safetensors(out_file)
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

        loaded = load_safetensors(output_dir / "transformer.safetensors")
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

        loaded = load_safetensors(output_dir / "vae_decoder.safetensors")
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

        loaded = load_safetensors(output_dir / "transformer.safetensors")
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
        weights = load_safetensors(tf_path)
        quantized = quantize_weights(
            weights, bits=8, group_size=64, should_quantize=ltx23_should_quantize
        )

        mx.save_safetensors(str(tf_path), quantized)

        # Reload and verify quantization artifacts
        reloaded = load_safetensors(tf_path)
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

        loaded = load_safetensors(output_dir / "connector.safetensors")

        # text_embedding_projection key is kept as-is by sanitizer
        orig_key = "text_embedding_projection.aggregate_embed.weight"
        out_key = f"connector.{orig_key}"
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
        conn_weights = load_safetensors(tmp_path / "connector.safetensors")
        assert "connector.linear.weight" in conn_weights
        assert "text_embedding_projection.weight" in conn_weights

        # split_model.json should be written
        marker = tmp_path / "split_model.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["split"] is True
        assert "transformer.safetensors" in data["files"]

    def test_split_multi_component_unified(self, tmp_path):
        """Split a unified model whose keys span several components."""
        unified = {
            "vae_encoder.conv_in.weight": mx.random.normal((256, 128)),
            "vae_encoder.blocks.0.weight": mx.random.normal((128, 128)),
            "vae_decoder.conv_out.weight": mx.random.normal((256, 128)),
            "vae_decoder.blocks.0.weight": mx.random.normal((128, 128)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        result = split_model(tmp_path, LTX23_SPLIT_MAP)

        assert "vae_encoder.safetensors" in result
        assert "vae_decoder.safetensors" in result

        # Verify loadable
        enc = load_safetensors(tmp_path / "vae_encoder.safetensors")
        assert len(enc) == 2
        dec = load_safetensors(tmp_path / "vae_decoder.safetensors")
        assert len(dec) == 2

    def test_split_merges_components_sharing_one_file(self, tmp_path):
        """Two key prefixes mapped to the same output land in one file."""
        unified = {
            "connector.weight": mx.random.normal((64, 64)),
            "text_embedding_projection.weight": mx.random.normal((64, 64)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        result = split_model(tmp_path, LTX23_SPLIT_MAP)

        assert result["connector.safetensors"] == 2

    def test_split_model_json_content(self, tmp_path):
        """Verify split_model.json has correct structure and counts."""
        unified = {
            "vae_encoder.weight": mx.random.normal((64, 64)),
            "vae_decoder.weight": mx.random.normal((64, 64)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        split_model(tmp_path, LTX23_SPLIT_MAP)

        data = json.loads((tmp_path / "split_model.json").read_text())
        assert data["split"] is True
        assert data["files"]["vae_encoder.safetensors"] == 1
        assert data["files"]["vae_decoder.safetensors"] == 1

    def test_split_round_trip_values(self, tmp_path):
        """Values survive the split pipeline unchanged."""
        original_tensor = mx.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
        unified = {"transformer.weight": original_tensor}
        mx.save_safetensors(str(tmp_path / "model.safetensors"), unified)

        split_model(tmp_path, LTX23_SPLIT_MAP)

        loaded = load_safetensors(tmp_path / "transformer.safetensors")
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
            "transformer_blocks.0.attn1.to_qkv.weight": mx.random.normal((256, 256)),
            "transformer_blocks.0.norm1.weight": mx.random.normal((256,)),
        }

        quantized = quantize_weights(
            weights, bits=4, group_size=64, should_quantize=ltx23_should_quantize
        )

        assert "transformer_blocks.0.attn1.to_qkv.scales" in quantized
        assert "transformer_blocks.0.attn1.to_qkv.biases" in quantized
        assert quantized["transformer_blocks.0.attn1.to_qkv.weight"].dtype == mx.uint32

        # norm weight should be untouched
        assert quantized["transformer_blocks.0.norm1.weight"].dtype != mx.uint32

    def test_quantize_skips_incompatible_shapes(self):
        """Weights with last dim not divisible by group_size are skipped."""
        weights = {
            "transformer_blocks.0.attn1.to_qkv.weight": mx.random.normal((256, 100)),
        }

        quantized = quantize_weights(
            weights, bits=8, group_size=64, should_quantize=ltx23_should_quantize
        )

        # Should be kept as-is (not quantized) because 100 % 64 != 0
        assert "transformer_blocks.0.attn1.to_qkv.weight" in quantized
        assert "transformer_blocks.0.attn1.to_qkv.scales" not in quantized


# ---------------------------------------------------------------------------
# Delta workflow end-to-end integration tests
# ---------------------------------------------------------------------------


class TestDeltaWorkflowEndToEnd:
    """Wire test: convert delta → validate → upload --add-only → upload --card-only."""

    def test_delta_workflow_glue(self, tmp_path, capsys):
        import argparse
        import json
        from unittest.mock import MagicMock, patch

        from mlx_forge.recipes import ltx_23
        from mlx_forge.upload import upload_model

        # Stage 1: simulate convert --skip-shared output (don't actually run convert —
        # T1/T2 already test that). Just write the artifacts a delta convert would produce.
        # Write a minimal valid safetensors file (validate calls mx.load on it).
        mx.save_safetensors(
            str(tmp_path / "transformer-distilled-1.1.safetensors"),
            {"transformer.transformer_blocks.0.attn1.to_gate_logits.weight": mx.zeros((4, 4))},
        )
        (tmp_path / "split_model.json").write_text(
            json.dumps(
                {
                    "format": "split",
                    "model_version": "2.3.0",
                    "components": [],
                    "transformer_variants": ["distilled-1.1"],
                    "lora": [],
                    "source": "Lightricks/LTX-2.3",
                    "delta": True,
                }
            )
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_version": "2.3.0",
                    "is_v2": True,
                    "apply_gated_attention": True,
                    "caption_channels": None,
                    "num_layers": 48,
                    "num_attention_heads": 32,
                    "attention_head_dim": 128,
                    "connector_positional_embedding_max_pos": [4096],
                    "connector_rope_type": "SPLIT",
                    "variants": {"distilled-1.1": {"cross_attention_adaln": True}},
                }
            )
        )

        # Stage 2: validate auto-detects delta
        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "Delta mode" in out

        # Stage 3: upload --add-only with mocked api — only new files get uploaded
        api = MagicMock()
        info = MagicMock()
        info.siblings = [
            MagicMock(rfilename="config.json"),
            MagicMock(rfilename="transformer-distilled.safetensors"),
        ]
        api.model_info.return_value = info

        upload_model(tmp_path, api=api, repo_id="user/repo", add_only=True)

        uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert "transformer-distilled-1.1.safetensors" in uploaded
        assert "config.json" not in uploaded  # already on remote

        # Stage 4: `upload --card-only` refreshes the card from the remote.
        # The card is assembled by the CLI, which owns every input; upload_model
        # only pushes what is on disk, so this drives the CLI.
        api2 = MagicMock()
        info2 = MagicMock()
        info2.siblings = [
            MagicMock(rfilename="transformer-distilled.safetensors", size=1),
            MagicMock(rfilename="transformer-distilled-1.1.safetensors", size=1),
        ]
        api2.model_info.return_value = info2
        api2.create_repo.return_value = "https://huggingface.co/user/repo"

        with (
            patch(
                "sys.argv",
                ["mlx-forge", "upload", str(tmp_path), "--repo-id", "user/repo", "--card-only"],
            ),
            patch("huggingface_hub.HfApi", return_value=api2),
        ):
            main()

        readme_text = (tmp_path / "README.md").read_text()
        # Both remote variants must appear in the regenerated card
        assert "distilled" in readme_text
        assert "distilled-1.1" in readme_text
