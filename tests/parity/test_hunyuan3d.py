import pytest

from mlx_forge.recipes import hunyuan3d_21 as h
from tests.parity._harness import ComponentParity, cast_all_to, check_parity

PUB = "published/hunyuan3d-2.1-mlx.json"
F16 = cast_all_to("F16")


def fuse_qkv_keys(keys: set[str]) -> set[str]:
    """Name-level image of fuse_dino_qkv: query/key/value triplets become one qkv."""
    out = set()
    for key in keys:
        for proj in ("query", "key", "value"):
            marker = f".attn.{proj}."
            if marker in key:
                group, suffix = key.split(marker)[0], key.split(marker)[-1]
                out.add(f"{group}.attn.qkv.{suffix}")
                break
        else:
            out.add(key)
    return out


# paint_unet finalize_keys: reconstruct the harvest-filtered ".ff.net.2." -> ".ff.proj_out."
# siblings, the same fixture-harvest asymmetry documented in test_matrix_game.py's
# finalize_dit_keys (a raw "...ff.net.2...." key has digit segment "2" > 1 so it never
# survives the upstream fixture's own harvest; the renamed "...ff.proj_out...." has no
# digit segment, so it trivially survives the published fixture's harvest). Reconstruct
# it from its "ff.proj_in" sibling (sanitize_paint_unet_key's ".ff.net.0.proj." ->
# ".ff.proj_in." rule, hunyuan3d_21.py:201).
def finalize_paint_unet_keys(keys: set[str]) -> set[str]:
    out = set(keys)
    for key in keys:
        if ".ff.proj_in." in key:
            out.add(key.replace(".ff.proj_in.", ".ff.proj_out."))
    return out


TABLE = [
    ComponentParity(
        "hunyuan3d",
        "dit",
        "upstream/hunyuan3d-shape-model.json",
        "model.fp16.ckpt",
        PUB,
        "dit.safetensors",
        h.SHAPE_SANITIZERS["dit"],
        "dit",
        F16,
        # sanitize_dit_key is `return key` unconditionally (hunyuan3d_21.py:118-119).
        sanitizer_drops_keys=False,
    ),
    ComponentParity(
        "hunyuan3d",
        "vae",
        "upstream/hunyuan3d-shape-vae.json",
        "model.fp16.ckpt",
        PUB,
        "vae.safetensors",
        h.SHAPE_SANITIZERS["vae"],
        "vae",
        F16,
        # sanitize_shape_vae_key is `return key` unconditionally (hunyuan3d_21.py:122-123).
        sanitizer_drops_keys=False,
    ),
    ComponentParity(
        "hunyuan3d",
        "image_encoder",
        "upstream/hunyuan3d-shape-conditioner.json",
        "model.fp16.ckpt",
        PUB,
        "image_encoder.safetensors",
        h.SHAPE_SANITIZERS["image_encoder"],
        "image_encoder",
        F16,
        # sanitize_shape_image_encoder_key returns None for any key that doesn't start
        # with "main_image_encoder.model." (hunyuan3d_21.py:126-130) — a genuine drop.
        sanitizer_drops_keys=True,
    ),
    pytest.param(
        ComponentParity(
            "hunyuan3d",
            "paint_unet",
            "upstream/hunyuan3d-paint-unet.json",
            "diffusion_pytorch_model.bin",
            PUB,
            "paint_unet.safetensors",
            h.sanitize_paint_unet_key,
            None,
            F16,
            # ".to_out.1." -> None (dropout) is a genuine drop (hunyuan3d_21.py:194-195).
            sanitizer_drops_keys=True,
            finalize_keys=finalize_paint_unet_keys,
        ),
        marks=pytest.mark.xfail(
            reason=(
                "genuine drift, not a fixture artifact — hunyuan3d-2.1/paint_unet: "
                "published pack (paint_unet.safetensors) still holds 28 un-renamed "
                "'.processor.to_out_mr.0.*' keys (attn1/attn_refview, bias+weight, "
                "across the down/mid/up blocks); current code would rename them away, "
                "because sanitize_paint_unet_key does "
                "'.processor.to_out_mr.0.' -> '.processor.to_out_mr.' "
                "(hunyuan3d_21.py:199). See task-4-report.md."
            ),
            strict=True,
        ),
        id="paint_unet",
    ),
    ComponentParity(
        "hunyuan3d",
        "paint_vae",
        "upstream/hunyuan3d-paint-vae.json",
        "diffusion_pytorch_model.bin",
        PUB,
        "paint_vae.safetensors",
        h.sanitize_paint_vae_key,
        None,
        F16,
        # sanitize_paint_vae_key always returns key (hunyuan3d_21.py:216-231) — the
        # replacements dict only renames a fixed set of substrings, never drops.
        sanitizer_drops_keys=False,
    ),
    ComponentParity(
        "hunyuan3d",
        "paint_clip",
        "upstream/hunyuan3d-paint-clip.json",
        "hunyuan3d-paintpbr-v2-1/image_encoder/model.safetensors",
        PUB,
        "paint_clip.safetensors",
        h.sanitize_paint_clip_key,
        None,
        F16,
        # sanitize_paint_clip_key returns key on every branch (hunyuan3d_21.py:264-273).
        sanitizer_drops_keys=False,
    ),
    ComponentParity(
        "hunyuan3d",
        "paint_dino",
        "upstream/hunyuan3d-dino.json",
        "model.safetensors",
        PUB,
        "paint_dino.safetensors",
        h.sanitize_paint_dino_key,
        None,
        F16,
        finalize_keys=fuse_qkv_keys,
        # sanitize_paint_dino_key returns None for "mask_token" keys (hunyuan3d_21.py:
        # 246-248) — a genuine drop.
        sanitizer_drops_keys=True,
    ),
]


@pytest.mark.parametrize("spec", TABLE, ids=lambda s: s.component)
def test_parity(spec):
    check_parity(spec)
