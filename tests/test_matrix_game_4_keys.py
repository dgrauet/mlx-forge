"""Tests for the Matrix-Game-4.0 recipe's key handling, quantization scope and config.

DiT keys come from tests/fixtures/upstream/matrix-game-4-dit.json, harvested
from the real fps-castle/model.safetensors header (scripts/harvest_keys.py),
so the sanitizer is checked against upstream's names rather than invented ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_forge.recipes import matrix_game_4_0 as mg4

FIXTURE = Path(__file__).parent / "fixtures" / "upstream" / "matrix-game-4-dit.json"


@pytest.fixture(scope="module")
def upstream() -> dict:
    return json.loads(FIXTURE.read_text())["fps-castle/model.safetensors"]


class TestDitSanitizer:
    def test_fixture_matches_upstream_model_json(self, upstream):
        # Each scene's model.json declares 825 tensors; validate() checks against it.
        assert upstream["tensor_count"] == mg4.EXPECTED_DIT_TENSORS

    def test_wrapper_prefix_detected(self, upstream):
        assert mg4.strip_checkpoint_prefix(upstream["keys"]) == "model."

    def test_partial_prefix_not_stripped(self):
        # Mirrors the reference loader: strip only when every key carries it.
        assert mg4.strip_checkpoint_prefix(["model.a", "b"]) == ""

    def test_every_key_sanitized_without_prefix_or_sequential_index(self, upstream):
        sanitized = [mg4.sanitize_dit_key(k) for k in upstream["keys"]]
        out = [k for k in sanitized if k is not None]
        assert len(out) == len(sanitized)
        assert len(set(out)) == len(out)
        assert not any(k.startswith("model.") for k in out)
        assert not any(".ffn.0." in k or "_embedding.0." in k for k in out)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("model.blocks.0.ffn.0.weight", "blocks.0.ffn_linear1.weight"),
            ("model.blocks.3.ffn.2.bias", "blocks.3.ffn_linear2.bias"),
            ("model.text_embedding.0.weight", "text_embedding_linear1.weight"),
            ("model.text_embedding.2.weight", "text_embedding_linear2.weight"),
            ("model.time_embedding.2.bias", "time_embedding_linear2.bias"),
            ("model.time_projection.1.weight", "time_projection_linear1.weight"),
            ("model.head.head.weight", "head.head.weight"),
            ("model.blocks.0.self_attn.q.weight", "blocks.0.self_attn.q.weight"),
        ],
    )
    def test_mapping(self, raw, expected):
        assert mg4.sanitize_dit_key(raw) == expected

    def test_ref_conv_branch_dropped(self):
        assert mg4.sanitize_dit_key("model.ref_conv.weight") is None

    def test_patch_embedding_becomes_linear(self):
        w = mx.zeros((3072, 148, 1, 2, 2), dtype=mx.bfloat16)
        out = mg4.transform_dit("patch_embedding.weight", w, "dit_castle")
        assert out.shape == (3072, 148 * 4)

    def test_block_linear_untouched(self):
        w = mx.zeros((3072, 3072))
        assert mg4.transform_dit("blocks.0.self_attn.q.weight", w, "dit_castle") is w


class TestShouldQuantize:
    def _key(self, bare: str) -> str:
        return f"dit_ancient_egypt.{bare}"

    @pytest.mark.parametrize(
        "bare",
        [
            "blocks.0.self_attn.q.weight",
            "blocks.0.cross_attn.o.weight",
            "blocks.29.ffn_linear1.weight",
            "blocks.29.ffn_linear2.weight",
        ],
    )
    def test_block_linears_quantized(self, bare):
        assert mg4.should_quantize(self._key(bare), mx.zeros((64, 64)))

    @pytest.mark.parametrize(
        "bare",
        [
            # The output head: MG3's predicate strips a literal "dit." and would
            # quantize this under a scene prefix.
            "head.head.weight",
            "patch_embedding.weight",
            "text_embedding_linear1.weight",
            "time_embedding_linear1.weight",
            "time_projection_linear1.weight",
            "blocks.0.self_attn.norm_q.weight",
            "blocks.0.norm3.weight",
        ],
    )
    def test_sensitive_layers_kept(self, bare):
        assert not mg4.should_quantize(self._key(bare), mx.zeros((64, 64)))

    def test_non_matrix_kept(self):
        assert not mg4.should_quantize(self._key("blocks.0.modulation"), mx.zeros((1, 6, 64)))
        assert not mg4.should_quantize(self._key("blocks.0.self_attn.q.bias"), mx.zeros((64,)))

    def test_quantized_set_over_real_keys(self, upstream):
        """Over the harvested keys: exactly the per-block attention/ffn Linears."""
        picked = set()
        for raw in upstream["keys"]:
            new = mg4.sanitize_dit_key(raw)
            assert new is not None
            ndim = 2 if raw.endswith(".weight") and "norm" not in raw else 1
            if mg4.should_quantize(f"dit_castle.{new}", mx.zeros((8,) * ndim)):
                picked.add(new)
        expected = {
            f"blocks.{i}.{attn}.{p}.weight"
            for i in (0, 1)
            for attn in ("self_attn", "cross_attn")
            for p in "qkvo"
        } | {f"blocks.{i}.ffn_linear1.weight" for i in (0, 1)}
        assert picked == expected


class TestTaehv:
    def test_keys_verbatim(self):
        assert mg4.sanitize_taehv_key("decoder.7.conv.weight") == "decoder.7.conv.weight"

    def test_conv2d_channels_last(self):
        w = mx.zeros((64, 12, 3, 3))
        assert mg4.transform_taehv("encoder.0.weight", w, "taehv").shape == (64, 3, 3, 12)

    def test_bias_untouched(self):
        b = mx.zeros((64,))
        assert mg4.transform_taehv("encoder.0.bias", b, "taehv") is b


class TestLayout:
    def test_component_names(self):
        assert mg4.DIT_COMPONENTS == ["dit_castle", "dit_ancient_egypt", "dit_racing"]

    def test_scene_files_cover_every_scene(self):
        for folder in mg4.SCENES.values():
            assert f"{folder}/inference.json" in mg4.SCENE_FILES
            assert f"{folder}/sink.png" in mg4.SCENE_FILES

    def test_config_prefers_upstream_values_and_lists_scenes(self):
        cfg = mg4.build_config(
            {"in_dim": 148, "num_layers": 30}, {"racing": {"name": "City Racing"}}
        )
        assert cfg["in_dim"] == 148
        assert cfg["scenes"]["racing"] == {
            "component": "dit_racing",
            "inference": "driving_game/inference.json",
            "sink_image": "driving_game/sink.png",
            "name": "City Racing",
        }
        assert cfg["codecs"] == {"quality": "vae", "realtime": "taehv"}
        assert cfg["is_action_model"] is False


class TestPinnedDownload:
    """The base config.json comes from the revision upstream pins and sha-verifies."""

    def test_revision_reaches_the_hub(self, tmp_path, monkeypatch):
        from mlx_forge import convert as convert_mod

        calls = []
        monkeypatch.setattr(convert_mod, "hf_hub_download", lambda **kw: calls.append(kw))
        convert_mod.download_hf_files(
            mg4.BASE_REPO_ID, [mg4.BASE_CONFIG_FILE], tmp_path, revision=mg4.BASE_REVISION
        )
        assert calls[0]["repo_id"] == "alibaba-pai/Wan2.2-Fun-5B-Control"
        assert calls[0]["revision"] == "b8bc1a65ab71d054ba4636dc0dac104aa4df2686"

    def test_default_follows_main(self, tmp_path, monkeypatch):
        from mlx_forge import convert as convert_mod

        calls = []
        monkeypatch.setattr(convert_mod, "hf_hub_download", lambda **kw: calls.append(kw))
        convert_mod.download_hf_files(mg4.REPO_ID, ["README.md"], tmp_path)
        assert calls[0]["revision"] is None


class TestSharedReference:
    """T5 / VAE / tokenizer live in matrix-game-3.0-mlx, as upstream references its base."""

    def test_pack_holds_only_scene_dits_and_taehv(self):
        assert mg4.COMPONENTS == [*mg4.DIT_COMPONENTS, "taehv"]

    def test_manifest_block(self):
        shared = mg4.shared_manifest()
        assert shared["repo_id"] == "dgrauet/matrix-game-3.0-mlx"
        assert len(shared["revision"]) == 40
        assert shared["components"] == {
            "t5_encoder": "t5_encoder.safetensors",
            "vae": "vae.safetensors",
        }
        paths = {f["path"] for f in shared["files"]}
        assert {"t5_encoder.safetensors", "vae.safetensors"} <= paths
        assert {f"google/umt5-xxl/{n}" for n in ("spiece.model", "tokenizer.json")} <= paths
        assert all(len(f["sha256"]) == 64 for f in shared["files"])

    def _api(self, lfs_sha: dict[str, str | None]):
        from types import SimpleNamespace

        siblings = [
            SimpleNamespace(rfilename=p, lfs=SimpleNamespace(sha256=s) if s else None)
            for p, s in lfs_sha.items()
        ]

        class Api:
            def model_info(self, repo_id, revision, files_metadata):
                assert (repo_id, revision) == (mg4.SHARED_REPO_ID, mg4.SHARED_REVISION)
                return SimpleNamespace(siblings=siblings)

        return Api()

    def _stub_small_files(self, monkeypatch, content: dict[str, bytes]):
        def fake_download(repo_id, filenames, download_dir, *, revision=None):
            assert revision == mg4.SHARED_REVISION
            for name in filenames:
                (download_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (download_dir / name).write_bytes(content[name])

        monkeypatch.setattr(mg4, "download_hf_files", fake_download)

    def test_accepts_matching_hub(self, tmp_path, monkeypatch):
        import hashlib

        small = {"google/umt5-xxl/tokenizer_config.json": b"{}"}
        monkeypatch.setattr(
            mg4,
            "SHARED_FILES",
            {
                "vae.safetensors": "a" * 64,
                **{p: hashlib.sha256(c).hexdigest() for p, c in small.items()},
            },
        )
        self._stub_small_files(monkeypatch, small)
        api = self._api({"vae.safetensors": "a" * 64, **dict.fromkeys(small)})
        mg4.verify_shared_reference(tmp_path, api=api)

    def test_rejects_changed_lfs_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mg4, "SHARED_FILES", {"vae.safetensors": "a" * 64})
        with pytest.raises(SystemExit, match="vae.safetensors: sha256"):
            mg4.verify_shared_reference(tmp_path, api=self._api({"vae.safetensors": "b" * 64}))

    def test_rejects_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mg4, "SHARED_FILES", {"t5_encoder.safetensors": "a" * 64})
        with pytest.raises(SystemExit, match="t5_encoder.safetensors: missing"):
            mg4.verify_shared_reference(tmp_path, api=self._api({}))

    def test_rejects_changed_small_file(self, tmp_path, monkeypatch):
        path = "google/umt5-xxl/tokenizer_config.json"
        monkeypatch.setattr(mg4, "SHARED_FILES", {path: "0" * 64})
        self._stub_small_files(monkeypatch, {path: b"{}"})
        with pytest.raises(SystemExit, match="sha256 differs"):
            mg4.verify_shared_reference(tmp_path, api=self._api({path: None}))
