"""vjepa split_model.json must be readable by vjepa2-core-mlx.

The consumer resolves component files through exactly one expression
(`vjepa2_core_mlx/utils/weights.py::_find_safetensors`):

    components = manifest.get("components", {})
    if component in components: ...

It reads a NESTED `components` mapping and never iterates the top-level keys.
vjepa-2.0 wrote a flat {component: filename} table, so `.get("components")`
returned {} and the manifest was ignored entirely — asking for the predictor or
a probe fell through to the canonical encoder filename. Verified by running the
real `_find_safetensors` against both shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_forge.recipes import vjepa_2_0_vitl as v20

torch = pytest.importorskip("torch")


def _resolve(manifest: dict, component: str) -> str | None:
    """Replicate the consumer's lookup, so this suite pins the same contract."""
    return manifest.get("components", {}).get(component)


def _converted(tmp_path) -> Path:
    """Run the real convert() against a miniature encoder checkpoint."""
    src = tmp_path / "vitl.pt"
    torch.save({"target_encoder": {"blocks.0.norm1.weight": torch.ones(4)}}, str(src))
    out = tmp_path / "out"
    v20.convert(
        argparse.Namespace(
            source=str(src),
            ssv2_source=None,
            diving48_source=None,
            ek100_source=None,
            output=str(out),
            quantize=False,
            bits=8,
            group_size=64,
            dry_run=False,
        )
    )
    return out


class TestManifestShape:
    def test_components_are_nested(self, tmp_path):
        manifest = json.loads((_converted(tmp_path) / "split_model.json").read_text())
        assert isinstance(manifest.get("components"), dict), (
            "a flat {component: filename} table is invisible to the consumer"
        )

    def test_every_component_resolves(self, tmp_path):
        out = _converted(tmp_path)
        manifest = json.loads((out / "split_model.json").read_text())

        for component, filename in manifest["components"].items():
            assert _resolve(manifest, component) == filename
            assert (out / filename).exists(), f"{component} listed but absent"

    def test_encoder_resolves_to_its_own_file(self, tmp_path):
        """The flat shape resolved every component to encoder.safetensors."""
        manifest = json.loads((_converted(tmp_path) / "split_model.json").read_text())
        assert _resolve(manifest, "encoder") == "encoder.safetensors"

    def test_source_is_declared(self, tmp_path):
        """Safe to add: the consumer looks under `components`, never at the root."""
        manifest = json.loads((_converted(tmp_path) / "split_model.json").read_text())
        assert manifest["source"] == v20.METADATA.source
        assert _resolve(manifest, "source") is None, "must not look like a component"


class TestFlatShapeWasBroken:
    """Pins why the change was made, against the shape vjepa-2.0 used to write."""

    def test_flat_manifest_resolves_nothing(self):
        flat = {"encoder": "encoder.safetensors", "predictor": "predictor.safetensors"}
        assert _resolve(flat, "encoder") is None
        assert _resolve(flat, "predictor") is None

    def test_nested_manifest_resolves_each_component(self):
        nested = {
            "components": {"encoder": "encoder.safetensors", "predictor": "predictor.safetensors"}
        }
        assert _resolve(nested, "predictor") == "predictor.safetensors"


def test_probes_are_addressable(tmp_path):
    """A probe must resolve to its own file, not to the encoder's."""
    src = tmp_path / "vitl.pt"
    torch.save({"target_encoder": {"blocks.0.norm1.weight": torch.ones(4)}}, str(src))
    probe = tmp_path / "ssv2.pt"
    torch.save({"classifiers": [{"linear.weight": torch.ones(174, 4)}]}, str(probe))
    out = tmp_path / "out"

    v20.convert(
        argparse.Namespace(
            source=str(src),
            ssv2_source=str(probe),
            diving48_source=None,
            ek100_source=None,
            output=str(out),
            quantize=False,
            bits=8,
            group_size=64,
            dry_run=False,
        )
    )

    manifest = json.loads((out / "split_model.json").read_text())
    probe_components = [c for c in manifest["components"] if "probe" in c]
    assert probe_components, "the probe was converted but is not in the manifest"
    for c in probe_components:
        assert _resolve(manifest, c) != "encoder.safetensors"
        assert (out / _resolve(manifest, c)).exists()


def test_mx_is_importable():
    assert mx is not None
