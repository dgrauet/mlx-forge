import pytest

from mlx_forge.recipes import vjepa_2_0_vitl as v20
from tests.parity._harness import ComponentParity, check_parity

PUB = "published/vjepa-2.0-vitl-mlx.json"

TABLE = [
    ComponentParity(
        "vjepa-2.0",
        "encoder",
        "upstream/vjepa-2.0-encoder.json",
        "vitl.pt",
        PUB,
        "encoder.safetensors",
        v20._sanitize_encoder_key,
        "encoder",
    ),
    ComponentParity(
        "vjepa-2.0",
        "predictor",
        "upstream/vjepa-2.0-predictor.json",
        "vitl.pt",
        PUB,
        "predictor.safetensors",
        v20._sanitize_encoder_key,
        "predictor",
    ),
    *[
        ComponentParity(
            "vjepa-2.0",
            f"{name}_probe",
            f"upstream/vjepa-2.0-probe-{name}.json",
            basename,
            PUB,
            f"{name}_probe.safetensors",
            v20._sanitize_probe_key,
            f"{name}_probe",
        )
        for name, basename in (
            ("ssv2", "ssv2-vitl.pt"),
            ("diving48", "diving48-vitl.pt"),
            ("ek100", "ek100-vitl.pt"),
        )
    ],
]


@pytest.mark.parametrize("spec", TABLE, ids=lambda s: s.component)
def test_parity(spec):
    check_parity(spec)
