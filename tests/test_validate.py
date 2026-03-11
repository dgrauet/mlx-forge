"""Tests for the validation framework."""

import mlx.core as mx

from mlx_forge.validate import (
    ValidationResult,
    validate_conv_layout,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)


class TestValidationResult:
    def test_initial_state(self):
        vr = ValidationResult()
        assert vr.errors == 0
        assert vr.warnings == 0
        assert vr.checks == []
        assert vr.passed is True

    def test_passing_check(self):
        vr = ValidationResult()
        result = vr.check(True, "all good")
        assert result is True
        assert vr.errors == 0
        assert vr.warnings == 0
        assert len(vr.checks) == 1
        assert vr.checks[0] == (True, "all good")

    def test_failing_check(self):
        vr = ValidationResult()
        result = vr.check(False, "something broke")
        assert result is False
        assert vr.errors == 1
        assert vr.passed is False

    def test_warn_only_check(self):
        vr = ValidationResult()
        result = vr.check(False, "minor issue", warn_only=True)
        assert result is False
        assert vr.warnings == 1
        assert vr.errors == 0
        assert vr.passed is True

    def test_mixed_checks(self):
        vr = ValidationResult()
        vr.check(True, "ok 1")
        vr.check(False, "fail 1")
        vr.check(False, "warn 1", warn_only=True)
        vr.check(True, "ok 2")
        assert vr.errors == 1
        assert vr.warnings == 1
        assert len(vr.checks) == 4
        assert vr.passed is False

    def test_summary_pass(self, capsys):
        vr = ValidationResult()
        vr.check(True, "ok")
        vr.summary()
        captured = capsys.readouterr()
        assert "passed" in captured.out

    def test_summary_fail(self, capsys):
        vr = ValidationResult()
        vr.check(False, "bad")
        vr.summary()
        captured = capsys.readouterr()
        assert "1 checks failed" in captured.out


class TestValidateFileExists:
    def test_file_present(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
        vr = ValidationResult()
        result = validate_file_exists(tmp_path, "model.safetensors", vr)
        assert result is True
        assert vr.errors == 0

    def test_file_missing(self, tmp_path):
        vr = ValidationResult()
        result = validate_file_exists(tmp_path, "missing.safetensors", vr)
        assert result is False
        assert vr.errors == 1


class TestValidateNoPytorchPrefix:
    def test_no_prefix_found(self):
        weights = {
            "block.0.weight": mx.zeros((2, 2)),
            "block.1.bias": mx.zeros((2,)),
        }
        vr = ValidationResult()
        validate_no_pytorch_prefix(weights, "model.diffusion_model.", vr)
        assert vr.errors == 0

    def test_prefix_found(self):
        weights = {
            "model.diffusion_model.block.0.weight": mx.zeros((2, 2)),
            "block.1.bias": mx.zeros((2,)),
        }
        vr = ValidationResult()
        validate_no_pytorch_prefix(weights, "model.diffusion_model.", vr)
        assert vr.errors == 1

    def test_partial_match(self):
        weights = {"model.diffusion_model.layer.weight": mx.zeros((2, 2))}
        vr = ValidationResult()
        validate_no_pytorch_prefix(weights, "model.diffusion_model.", vr)
        assert vr.errors == 1


class TestValidateConvLayout:
    def test_correct_mlx_layout_5d(self):
        # MLX Conv3d: (O, D, H, W, I) with small spatial dims
        weights = {"conv.weight": mx.zeros((16, 3, 3, 3, 8))}
        vr = ValidationResult()
        validate_conv_layout(weights, vr, ndim=5)
        assert vr.errors == 0

    def test_wrong_layout_5d(self):
        # PyTorch Conv3d: (O, I, D, H, W) — I=128 in spatial position looks suspect
        weights = {"conv.weight": mx.zeros((16, 128, 3, 3, 3))}
        vr = ValidationResult()
        validate_conv_layout(weights, vr, ndim=5)
        assert vr.errors == 1

    def test_correct_mlx_layout_4d(self):
        # MLX Conv2d: (O, H, W, I)
        weights = {"conv.weight": mx.zeros((32, 3, 3, 16))}
        vr = ValidationResult()
        validate_conv_layout(weights, vr, ndim=4)
        assert vr.errors == 0

    def test_non_conv_keys_ignored(self):
        weights = {"linear.weight": mx.zeros((16, 128, 3, 3, 3))}
        vr = ValidationResult()
        validate_conv_layout(weights, vr, ndim=5)
        assert vr.errors == 0

    def test_no_conv_weights(self):
        weights = {"linear.weight": mx.zeros((64, 32))}
        vr = ValidationResult()
        validate_conv_layout(weights, vr, ndim=5)
        assert vr.errors == 0


class TestValidateQuantization:
    def test_valid_quantization(self):
        weights = {
            "transformer_blocks.0.attn.weight": mx.zeros((64, 8)),
            "transformer_blocks.0.attn.scales": mx.zeros((64, 1)),
            "transformer_blocks.0.attn.biases": mx.zeros((64, 1)),
        }
        vr = ValidationResult()
        validate_quantization(weights, vr, block_key="transformer_blocks")
        assert vr.errors == 0

    def test_missing_biases(self):
        weights = {
            "transformer_blocks.0.attn.weight": mx.zeros((64, 8)),
            "transformer_blocks.0.attn.scales": mx.zeros((64, 1)),
        }
        vr = ValidationResult()
        validate_quantization(weights, vr, block_key="transformer_blocks")
        # .biases count is 0, so "Equal .scales and .biases count" fails
        assert vr.errors >= 1

    def test_no_scales_at_all(self):
        weights = {"layer.weight": mx.zeros((64, 32))}
        vr = ValidationResult()
        validate_quantization(weights, vr, block_key="transformer_blocks")
        assert vr.errors >= 1

    def test_non_block_scales_warn(self):
        weights = {
            "other_layer.scales": mx.zeros((64, 1)),
            "other_layer.biases": mx.zeros((64, 1)),
        }
        vr = ValidationResult()
        validate_quantization(weights, vr, block_key="transformer_blocks")
        assert vr.warnings >= 1
