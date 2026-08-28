import pytest

from mlx_forge.recipes import matrix_game_3_0 as mg
from tests.parity._harness import ComponentParity, cast_all_to, cast_f32_to_bf16, check_parity

PUB = "published/matrix-game-3.0-mlx.json"

# --- DiT complete_upstream_keys: reconstruct the harvest-filtered Sequential siblings ---
#
# scripts/harvest_keys.py's keep() drops any key with an all-digit path segment > 1.
# _DIT_SEQUENTIAL_MAP (matrix_game_3_0.py:114-131) renames PyTorch nn.Sequential
# indices to named Linear layers, e.g. ".ffn.2." -> ".ffn_linear2.". A raw upstream
# key like "...ffn.2.weight" has digit segment "2" > 1, so it never survives the
# upstream fixture's own harvest. But the PUBLISHED fixture was harvested from the
# already-converted file, where the key is "...ffn_linear2.weight" — no digit segment
# at all, so it trivially survives that harvest. The result: for every Sequential
# group with a kept "index 0" sibling (survives upstream reduction) and a dropped
# "index 2/3" sibling, our raw-key set is missing the high-index sibling purely
# because of this fixture-harvest asymmetry, not because the sanitizer disagrees with
# the published pack.
#
# Reconstruct the missing sibling on the RAW/upstream side — before sanitize_dit_key
# runs — from its "index 0" kept counterpart, using the same raw index patterns
# _DIT_SEQUENTIAL_MAP itself matches on. This way the renamed name in the emitted set
# is produced by the real sanitizer, not hardcoded here: mutating or deleting one of
# these _DIT_SEQUENTIAL_MAP rules changes what the sanitizer emits for the
# reconstructed key too, so the row stays a live gate on the rename rule instead of
# going blind to it (a post-hoc rewrite of the *sanitizer's output* names, as an
# earlier version of this test did, would keep passing even if the rename rule
# powering it were deleted).
_DIT_UPSTREAM_SIBLING_PAIRS = [
    (".ffn.0.", ".ffn.2."),
    ("text_embedding.0.", "text_embedding.2."),
    ("time_embedding.0.", "time_embedding.2."),
    (".action_model.keyboard_embed.0.", ".action_model.keyboard_embed.2."),
    (".action_model.mouse_mlp.0.", ".action_model.mouse_mlp.2."),
    (".action_model.mouse_mlp.0.", ".action_model.mouse_mlp.3."),
]


def complete_dit_upstream_keys(keys: set[str]) -> set[str]:
    out = set(keys)
    for key in keys:
        for low, high in _DIT_UPSTREAM_SIBLING_PAIRS:
            if low in key:
                out.add(key.replace(low, high))
    return out


# --- LightVAE finalize_keys: mirror the fixture harvest's own reduction rule ---
#
# The opposite asymmetry from the DiT case: _map_resnet_tail (matrix_game_3_0.py:
# 207-219) renames plain names like "conv1"/"norm2" (no digit segment, so the raw
# key trivially survives the upstream harvest) into "residual.2."/"residual.3."
# (now WITH a digit segment > 1). The published fixture was harvested from the
# converted file, so keep() excludes these renamed keys from ITS reduced list —
# but our emitted set (sanitized straight from the upstream reduced list) still has
# them, since the digit only appears after renaming. Apply the identical keep()
# rule (scripts/harvest_keys.py:32-34) to the emitted set to reproduce exactly what
# the published fixture's own harvest did.
def _keep(key: str) -> bool:
    return all(int(seg) <= 1 for seg in key.split(".") if seg.isdigit())


def finalize_lightvae_keys(keys: set[str]) -> set[str]:
    return {k for k in keys if _keep(k)}


TABLE = [
    ComponentParity(
        "matrix-game",
        "dit",
        "upstream/matrix-game-dit.json",
        "base_model/diffusion_pytorch_model.safetensors",
        PUB,
        "dit.safetensors",
        mg.sanitize_dit_key,
        "dit",
        dtype_map=cast_f32_to_bf16,
        # sanitize_dit_key never returns None (matrix_game_3_0.py:169-180) — the key
        # count gap is entirely the harvest-reduction asymmetry above, handled by
        # complete_upstream_keys, not a genuine drop.
        sanitizer_drops_keys=False,
        complete_upstream_keys=complete_dit_upstream_keys,
    ),
    pytest.param(
        ComponentParity(
            "matrix-game",
            "dit_distilled",
            "upstream/matrix-game-dit-distilled.json",
            "base_distilled_model/diffusion_pytorch_model.safetensors",
            PUB,
            "dit_distilled.safetensors",
            mg.sanitize_dit_key,
            "dit_distilled",
            dtype_map=cast_f32_to_bf16,
            sanitizer_drops_keys=False,
            complete_upstream_keys=complete_dit_upstream_keys,
        ),
        marks=pytest.mark.xfail(
            reason=(
                "genuine drift, not a fixture artifact — matrix-game-3.0/dit_distilled: "
                "published pack (dit_distilled.safetensors) holds F32 throughout "
                "(1356 tensors); current code would emit BF16 throughout, because "
                "_convert_dit (matrix_game_3_0.py:466-468) casts every F32 weight to "
                "BF16 unconditionally and the distilled upstream checkpoint is F32 "
                "throughout. See task-4-report.md."
            ),
            strict=True,
        ),
        id="dit_distilled",
    ),
    ComponentParity(
        "matrix-game",
        "t5_encoder",
        "upstream/matrix-game-t5.json",
        "models_t5_umt5-xxl-enc-bf16.pth",
        PUB,
        "t5_encoder.safetensors",
        mg.sanitize_t5_key,
        "t5_encoder",
        dtype_map=cast_all_to("BF16"),
        # sanitize_t5_key is `return key` unconditionally (matrix_game_3_0.py:188-195).
        sanitizer_drops_keys=False,
    ),
    ComponentParity(
        "matrix-game",
        "vae",
        "upstream/matrix-game-vae.json",
        "Wan2.2_VAE.pth",
        PUB,
        "vae.safetensors",
        mg.sanitize_vae_key,
        "vae",
        dtype_map=cast_all_to("F32"),
        # _normalize_vae_key returns None for dynamic_feature_projection_heads.*
        # (matrix_game_3_0.py:227-229) — a genuine drop (198 -> 196 tensors).
        sanitizer_drops_keys=True,
    ),
    ComponentParity(
        "matrix-game",
        "vae_lightvae",
        "upstream/matrix-game-vae-lightvae.json",
        "MG-LightVAE.pth",
        PUB,
        "vae_lightvae.safetensors",
        mg.sanitize_vae_key,
        "vae_lightvae",
        dtype_map=cast_all_to("F32"),
        sanitizer_drops_keys=True,
        finalize_keys=finalize_lightvae_keys,
    ),
    ComponentParity(
        "matrix-game",
        "vae_lightvae_v2",
        "upstream/matrix-game-vae-lightvae-v2.json",
        "MG-LightVAE_v2.pth",
        PUB,
        "vae_lightvae_v2.safetensors",
        mg.sanitize_vae_key,
        "vae_lightvae_v2",
        dtype_map=cast_all_to("F32"),
        sanitizer_drops_keys=True,
        finalize_keys=finalize_lightvae_keys,
    ),
]


@pytest.mark.parametrize("spec", TABLE, ids=lambda s: s.component)
def test_parity(spec):
    check_parity(spec)
