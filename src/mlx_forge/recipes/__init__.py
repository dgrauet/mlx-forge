"""Model-specific conversion recipes.

Each recipe defines:
- Key classification (which component a weight belongs to)
- Key sanitization (PyTorch names -> MLX names)
- Conv transposition rules
- Config extraction
- Validation checks
"""

AVAILABLE_RECIPES = {
    "ltx-2.3": "mlx_forge.recipes.ltx_23",
    "fish-s2-pro": "mlx_forge.recipes.fish_s2",
    "matrix-game-3.0": "mlx_forge.recipes.matrix_game_3_0",
    "mistral-small-3.1": "mlx_forge.recipes.mistral_small_31",
    "qwen-image-2512": "mlx_forge.recipes.qwen_image_2512",
    "cogvideox-fun-v1.5-5b-inp": "mlx_forge.recipes.cogvideox_fun_v1_5_5b_inp",
    "void-model": "mlx_forge.recipes.void_model",
    "hunyuan3d-2.1": "mlx_forge.recipes.hunyuan3d_21",
    "ernie-image": "mlx_forge.recipes.ernie_image",
}
