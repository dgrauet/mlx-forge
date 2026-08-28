import pytest

from mlx_forge.recipes import vjepa_2_1_vitl as v21
from tests.parity._harness import ComponentParity, check_parity

PUB = "published/vjepa-2.1-vitl-mlx.json"

# The fixtures hold the RAW checkpoint keys, before the recipe's unwrap step runs
# (harvested with --section ema_encoder / --section predictor, which pulls the raw
# section straight out of the .pt, before _unwrap_encoder / _unwrap_predictor strip
# any prefix). Both raw sections carry a shared "module.backbone." prefix:
#   - _unwrap_predictor (vjepa_2_1_vitl.py:207-224) strips exactly "module.backbone."
#     from every predictor key.
#   - _unwrap_encoder (vjepa_2_1_vitl.py:182-204) strips prefixes one at a time from
#     _KNOWN_PREFIXES = ("module.", "encoder.", "target_encoder.", "backbone."); since
#     every encoder key starts "module.backbone....", the first pass strips "module."
#     and the (now all-"backbone."-prefixed) second pass strips "backbone." too — net
#     effect identical to stripping "module.backbone." in one shot.
# sanitize_key itself is `return key` (vjepa_2_1_vitl.py:232-239), so the sanitizer
# under test composes the unwrap-prefix-strip with that identity.
_STRIP = "module.backbone."


def _strip_and_sanitize(key: str) -> str | None:
    return v21.sanitize_key(key.removeprefix(_STRIP))


TABLE = [
    ComponentParity(
        "vjepa-2.1",
        "encoder",
        "upstream/vjepa-2.1-encoder.json",
        "vjepa2_1_vitl.pt",
        PUB,
        "encoder.safetensors",
        _strip_and_sanitize,
        None,  # bare keys (vjepa_2_1_vitl.py:355)
    ),
    ComponentParity(
        "vjepa-2.1",
        "predictor",
        "upstream/vjepa-2.1-predictor.json",
        "vjepa2_1_vitl.pt",
        PUB,
        "predictor.safetensors",
        lambda k: k.removeprefix(_STRIP),
        None,
    ),
]


@pytest.mark.parametrize("spec", TABLE, ids=lambda s: s.component)
def test_parity(spec):
    check_parity(spec)
