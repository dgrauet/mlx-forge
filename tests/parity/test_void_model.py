import pytest

from mlx_forge.recipes import void_model
from tests.parity._harness import ComponentParity, check_parity

TABLE = [
    ComponentParity(
        recipe="void-model",
        component=name,
        upstream_fixture="upstream/void-model.json",
        upstream_file=f"{name}.safetensors",
        published_fixture="published/void-model-mlx.json",
        published_file=f"{name}.safetensors",
        sanitizer=void_model.sanitize_key,
        prefix=None,  # _convert_pass stores bare keys (void_model.py:172)
        # sanitize_key is `return key` unconditionally (void_model.py:84-90) — never
        # drops a key, so this stays False and the harness runs the exact-count check.
        sanitizer_drops_keys=False,
    )
    for name in ("void_pass1", "void_pass2")
]


@pytest.mark.parametrize("spec", TABLE, ids=lambda s: s.component)
def test_parity(spec):
    check_parity(spec)
