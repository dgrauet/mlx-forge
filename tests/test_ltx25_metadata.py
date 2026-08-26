"""Metadata propagation into converted safetensors headers.

The consuming ecosystem gates behaviour on `model_version` in the file
headers (e.g. ANCESTRAL_SAMPLER_SINCE_VERSION) and reads the DiT config from
`embedded_config.json`; the packs originally shipped with
`__metadata__ = None` everywhere. These tests pin the three mechanisms that
now carry it: process_component writes it, quantize_component preserves it,
and _embedded_config_payload surfaces it (with the one casing normalisation
the consumer asked for).
"""

from __future__ import annotations

import argparse
import json
import struct

import mlx.core as mx

from mlx_forge.convert import load_safetensors, process_component, quantize_component
from mlx_forge.recipes import ltx_25


def _header_metadata(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__")


class TestProcessComponentMetadata:
    def test_metadata_lands_in_the_header(self, tmp_path):
        weights = {"comp.w.weight": mx.zeros((2, 2))}
        process_component(
            weights,
            "comp",
            list(weights),
            tmp_path,
            component_prefix="comp",
            sanitizer=lambda k: k.removeprefix("comp."),
            metadata={"model_version": "2.5.0", "config": "{}"},
        )
        md = _header_metadata(tmp_path / "comp.safetensors")
        assert md == {"model_version": "2.5.0", "config": "{}"}

    def test_no_metadata_stays_clean(self, tmp_path):
        weights = {"comp.w.weight": mx.zeros((2, 2))}
        process_component(
            weights,
            "comp",
            list(weights),
            tmp_path,
            component_prefix="comp",
            sanitizer=lambda k: k.removeprefix("comp."),
        )
        assert not _header_metadata(tmp_path / "comp.safetensors")


class TestQuantizePreservesMetadata:
    def test_quantize_component_keeps_the_header(self, tmp_path):
        mx.save_safetensors(
            str(tmp_path / "comp.safetensors"),
            {"comp.w.weight": mx.zeros((64, 64))},
            metadata={"model_version": "2.5.0"},
        )
        quantize_component(
            tmp_path,
            "comp",
            bits=8,
            group_size=64,
            should_quantize=lambda k, w: True,
        )
        md = _header_metadata(tmp_path / "comp.safetensors")
        assert md == {"model_version": "2.5.0"}
        # And it actually quantized: scales present.
        weights = load_safetensors(tmp_path / "comp.safetensors")
        assert any(k.endswith(".scales") for k in weights)


class TestEmbeddedConfigPayload:
    RAW = json.dumps(
        {
            "transformer": {"num_layers": 48, "text_encoder_norm_type": "PER_TOKEN_RMS"},
            "scheduler": {"sampler": "LinearQuadratic"},
        }
    )

    def test_adds_model_version_and_lowercases_norm_type(self):
        payload = ltx_25._embedded_config_payload(self.RAW, "2.5.0")
        assert payload["model_version"] == "2.5.0"
        assert payload["transformer"]["text_encoder_norm_type"] == "per_token_rms"
        # Everything else verbatim.
        assert payload["transformer"]["num_layers"] == 48
        assert payload["scheduler"] == {"sampler": "LinearQuadratic"}

    def test_without_model_version_nothing_is_invented(self):
        payload = ltx_25._embedded_config_payload(self.RAW, None)
        assert "model_version" not in payload


class TestValidateGatesOnHeaderVersion:
    def test_a_transformer_without_model_version_fails(self, tmp_path):
        import pytest

        from tests.test_ltx25_sources import _full_pack

        _full_pack(tmp_path)
        # Rewrite the transformer without header metadata — the pre-fix shape.
        weights = dict(load_safetensors(tmp_path / ltx_25.VARIANT_FILENAMES["dev"]))
        mx.save_safetensors(str(tmp_path / ltx_25.VARIANT_FILENAMES["dev"]), weights)
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))
