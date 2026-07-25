"""End-to-end checks on the recipe functions that read PyTorch checkpoints.

These call sites had no coverage at all before the shared `load_torch_state_dict`
helper landed, so a mechanical substitution there would have gone unnoticed.
Each test writes a synthetic checkpoint shaped like the real one and drives the
recipe function against it.
"""

from __future__ import annotations

import pytest

from mlx_forge.recipes import hunyuan3d_21, matrix_game_3_0, vjepa_2_0_vitl

torch = pytest.importorskip("torch")


def _save(tmp_path, obj, name):
    path = tmp_path / name
    torch.save(obj, str(path))
    return path


class TestVjepa20Loaders:
    def test_encoder(self, tmp_path):
        path = _save(tmp_path, {"target_encoder": {"blocks.0.norm1.weight": torch.ones(4)}}, "e.pt")
        raw = vjepa_2_0_vitl._load_pt_encoder(path)
        assert raw["blocks.0.norm1.weight"].shape == (4,)

    def test_encoder_without_target_encoder_key_exits(self, tmp_path):
        path = _save(tmp_path, {"something_else": {}}, "e.pt")
        with pytest.raises(SystemExit) as exc_info:
            vjepa_2_0_vitl._load_pt_encoder(path)
        assert "target_encoder" in str(exc_info.value)

    def test_predictor_uses_mmap_and_unwraps(self, tmp_path):
        path = _save(tmp_path, {"predictor": {"module.backbone.w": torch.ones(2)}}, "p.pt")
        raw = vjepa_2_0_vitl._load_pt_predictor(path)
        assert raw is not None
        assert raw["module.backbone.w"].shape == (2,)

    def test_predictor_absent_degrades_to_none(self, tmp_path, capsys):
        """A checkpoint with no predictor must warn, not crash."""
        path = _save(tmp_path, {"target_encoder": {}}, "p.pt")
        assert vjepa_2_0_vitl._load_pt_predictor(path) is None
        assert "no 'predictor' key" in capsys.readouterr().out

    def test_probe_takes_first_classifier(self, tmp_path):
        path = _save(
            tmp_path, {"classifiers": [{"w": torch.ones(3)}, {"w": torch.zeros(3)}]}, "c.pt"
        )
        raw = vjepa_2_0_vitl._load_pt_probe(path)
        assert raw["w"].sum().item() == 3.0

    def test_probe_empty_classifiers_exits(self, tmp_path):
        path = _save(tmp_path, {"classifiers": []}, "c.pt")
        with pytest.raises(SystemExit):
            vjepa_2_0_vitl._load_pt_probe(path)


class TestHunyuanTorchBin:
    def test_converts_to_mlx_arrays(self, tmp_path):
        path = _save(tmp_path, {"conv.weight": torch.ones(2, 2)}, "model.bin")
        state = hunyuan3d_21._load_torch_bin(path)
        assert state["conv.weight"].shape == (2, 2)
        assert state["conv.weight"].dtype.size == 4  # float32


class TestMatrixGamePthConversion:
    def test_t5_pth_written_as_safetensors(self, tmp_path):
        path = _save(tmp_path, {"token_embedding.weight": torch.ones(4, 8)}, "t5.pth")
        count = matrix_game_3_0._convert_t5_pth(str(path), tmp_path)
        assert count == 1
        assert (tmp_path / "t5_encoder.safetensors").exists()

    def test_t5_pth_unwraps_state_dict_wrapper(self, tmp_path):
        path = _save(
            tmp_path, {"state_dict": {"token_embedding.weight": torch.ones(2, 2)}}, "t.pth"
        )
        assert matrix_game_3_0._convert_t5_pth(str(path), tmp_path) == 1

    def test_vae_pth_written_with_prefix(self, tmp_path):
        from mlx_forge.convert import load_safetensors

        path = _save(tmp_path, {"encoder.conv_in.weight": torch.ones(2, 2, 3, 3, 3)}, "vae.pth")
        out = tmp_path / "vae.safetensors"
        assert matrix_game_3_0._convert_vae_pth(str(path), out, "vae") == 1
        assert all(k.startswith("vae.") for k in load_safetensors(out))
