"""Generic validation framework for converted MLX models.

Provides colored pass/fail/warn output and a check() helper.
Model-specific validation logic lives in recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

PASS = "\033[92m\u2713\033[0m"
FAIL = "\033[91m\u2717\033[0m"
WARN = "\033[93m\u26a0\033[0m"


@dataclass
class ValidationResult:
    """Accumulates validation results."""

    errors: int = 0
    warnings: int = 0
    checks: list[tuple[bool, str]] = field(default_factory=list)

    def check(self, condition: bool, msg: str, *, warn_only: bool = False) -> bool:
        """Record a check result.

        Args:
            condition: True if the check passed.
            msg: Description of the check.
            warn_only: If True, failure is a warning instead of an error.

        Returns:
            The condition value.
        """
        if condition:
            print(f"  {PASS} {msg}")
        elif warn_only:
            print(f"  {WARN} {msg}")
            self.warnings += 1
        else:
            print(f"  {FAIL} {msg}")
            self.errors += 1
        self.checks.append((condition, msg))
        return condition

    @property
    def passed(self) -> bool:
        return self.errors == 0

    def summary(self) -> None:
        """Print summary and exit with appropriate code."""
        print(f"\n{'=' * 60}")
        if self.passed:
            print(f"{PASS} All checks passed! ({self.warnings} warnings)")
        else:
            print(f"{FAIL} {self.errors} checks failed, {self.warnings} warnings")


def validate_file_exists(model_dir: Path, filename: str, result: ValidationResult) -> bool:
    """Check that a file exists and report its size.

    Args:
        model_dir: Directory to check in.
        filename: Expected filename.
        result: ValidationResult to record into.

    Returns:
        True if the file exists.
    """
    path = model_dir / filename
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        result.check(True, f"{filename} exists ({size_mb:.1f} MB)")
        return True
    result.check(False, f"{filename} missing")
    return False


def count_layer_indices(keys: set[str], block_key: str = "layers") -> set[int]:
    """Extract unique integer layer indices from weight keys."""
    indices = set()
    marker = f"{block_key}."
    for k in keys:
        if marker in k:
            parts = k.split(marker)
            if len(parts) > 1:
                idx = parts[1].split(".")[0]
                if idx.isdigit():
                    indices.add(int(idx))
    return indices


def validate_no_pytorch_prefix(
    weights: dict[str, mx.array], prefix: str, result: ValidationResult
) -> None:
    """Check that no keys still have a PyTorch-style prefix.

    Args:
        weights: Dict of weight keys.
        prefix: PyTorch prefix to check for (e.g., "model.diffusion_model.").
        result: ValidationResult to record into.
    """
    bad_keys = [k for k in weights if prefix in k]
    result.check(
        len(bad_keys) == 0,
        f"No PyTorch prefix '{prefix}' remaining (found {len(bad_keys)})",
    )
    for k in bad_keys[:5]:
        print(f"    Bad key: {k}")


def validate_conv_layout(
    weights: dict[str, mx.array], result: ValidationResult, *, ndim: int = 5
) -> None:
    """Check that conv weights appear to be in MLX channels-last layout.

    For Conv3d (ndim=5), MLX layout is (O, D, H, W, I) where D/H/W are small spatial dims.

    Args:
        weights: Dict of weight keys -> tensors.
        result: ValidationResult to record into.
        ndim: Expected number of dimensions for conv weights.
    """
    conv_weights = [
        (k, v)
        for k, v in weights.items()
        if "conv" in k.lower() and "weight" in k and v.ndim == ndim
    ]
    mlx_layout = True
    for k, v in conv_weights:
        spatial = v.shape[1 : ndim - 1]
        if any(s > 16 for s in spatial):
            mlx_layout = False
            print(f"    Suspect layout: {k} shape={v.shape}")

    result.check(
        mlx_layout,
        f"Conv{ndim - 2}d weights in MLX channels-last layout ({len(conv_weights)} checked)",
    )


def validate_quantization(
    weights: dict[str, mx.array], result: ValidationResult, *, block_key: str
) -> None:
    """Check quantized weights have matching .scales/.biases pairs.

    Args:
        weights: Dict of weight keys -> tensors.
        result: ValidationResult to record into.
        block_key: Key substring that identifies the quantized layer group
            (e.g. "transformer_blocks" for LTX-2.3).
    """
    scale_keys = [k for k in weights if k.endswith(".scales")]
    bias_keys = [k for k in weights if k.endswith(".biases")]

    result.check(len(scale_keys) > 0, f"Quantized: {len(scale_keys)} .scales keys")
    result.check(len(bias_keys) > 0, f"Quantized: {len(bias_keys)} .biases keys")
    result.check(len(scale_keys) == len(bias_keys), "Equal .scales and .biases count")

    non_block = [k for k in scale_keys if block_key not in k]
    result.check(
        len(non_block) == 0,
        f"Quantization only in {block_key} (non-block scales: {len(non_block)})",
        warn_only=True,
    )
