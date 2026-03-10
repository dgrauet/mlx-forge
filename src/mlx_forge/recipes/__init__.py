"""Model-specific conversion recipes.

Each recipe defines:
- Key classification (which component a weight belongs to)
- Key sanitization (PyTorch names -> MLX names)
- Conv transposition rules
- Config extraction
- Validation checks
"""

AVAILABLE_RECIPES = {
    "ltx23": "mlx_forge.recipes.ltx23",
}
