import pytest

from mlx_forge.recipes import cogvideox_fun_v1_5_5b_inp as cog
from tests.parity._harness import ComponentParity, check_parity

UP = "upstream/cogvideox-fun.json"
PUB = "published/cogvideox-fun-v1.5-5b-inp-mlx.json"

TABLE = [
    # sanitize_transformer_key / sanitize_vae_key are `return key` unconditionally
    # (cogvideox_fun_v1_5_5b_inp.py:159-164, 175-180) — never drop a key, so
    # sanitizer_drops_keys stays False and the harness runs the exact-count check.
    ComponentParity(
        "cogvideox",
        "transformer",
        UP,
        "transformer/diffusion_pytorch_model.safetensors",
        PUB,
        "transformer.safetensors",
        cog.sanitize_transformer_key,
        "transformer",
        sanitizer_drops_keys=False,
    ),
    ComponentParity(
        "cogvideox",
        "vae",
        UP,
        "vae/diffusion_pytorch_model.safetensors",
        PUB,
        "vae.safetensors",
        cog.sanitize_vae_key,
        "vae",
        sanitizer_drops_keys=False,
    ),
]


@pytest.mark.parametrize("spec", TABLE, ids=lambda s: s.component)
def test_parity(spec):
    check_parity(spec)


def test_text_encoder_parity_across_shards():
    """The T5 upstream is sharded; the published file is one. Union the shard records."""
    import json

    from tests.parity._harness import FIXTURES

    up = json.loads((FIXTURES / UP).read_text())
    shards = [v for k, v in up.items() if k.startswith("text_encoder/")]
    assert shards, "no text_encoder shard records in the upstream fixture"
    emitted = set()
    for record in shards:
        for key in record["keys"]:
            new_key = cog.sanitize_text_encoder_key(key)
            if new_key is not None:
                emitted.add(f"text_encoder.{new_key}")
    published = json.loads((FIXTURES / PUB).read_text())["text_encoder.safetensors"]
    assert emitted == set(published["keys"])
    assert sum(r["tensor_count"] for r in shards) >= published["tensor_count"]
