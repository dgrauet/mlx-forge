"""The lock on a duplication we chose to accept — and one we chose not to.

ltx-2.5 is a standalone recipe, which recreates the duplication that
docs/recipe-anatomy.md identifies as the source of three latent defects already
found in this repo. The conv video VAE and the audio VAE are byte-for-byte the
same architecture as LTX-2.3, verified key by key against the upstream
headers. This test does not undo that duplication; it makes it unable to
drift in silence.

The one difference for those two is the source prefix: 2.3's monolith names a
VAE weight `vae.decoder.…`, while 2.5's per-role file names it `decoder.…`.
The test bridges that and asserts everything downstream of it is identical.

The vocoder is the deliberate exception:
`test_vocoder_deliberately_diverges_from_2_3` below asserts a *divergence*
from 2.3, not parity with it — see `sanitize_vocoder_key`'s docstring for why.
"""

import json
from pathlib import Path

from mlx_forge.recipes import ltx_23, ltx_25

FIXTURE = Path(__file__).parent / "fixtures" / "ltx_25_keys.json"
VAE_CONV = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO = "vae/ltx-2.5-audio-vae-bf16.safetensors"


def keys(filename: str) -> list[str]:
    with open(FIXTURE) as handle:
        return json.load(handle)[filename]["keys"]


class TestSanitizerParity:
    def test_vae_decoder(self):
        for key in keys(VAE_CONV):
            if not key.startswith(("decoder.", "per_channel_statistics.")):
                continue
            assert ltx_25.sanitize_vae_decoder_key(key) == ltx_23.sanitize_vae_decoder_key(
                f"vae.{key}"
            ), key

    def test_vae_encoder(self):
        for key in keys(VAE_CONV):
            if not key.startswith(("encoder.", "per_channel_statistics.")):
                continue
            assert ltx_25.sanitize_vae_encoder_key(key) == ltx_23.sanitize_vae_encoder_key(
                f"vae.{key}"
            ), key

    def test_audio_vae(self):
        for key in keys(AUDIO):
            if not key.startswith("audio_vae."):
                continue
            assert ltx_25.sanitize_audio_vae_key(key) == ltx_23.sanitize_audio_vae_key(key), key

    def test_vocoder_deliberately_diverges_from_2_3(self):
        # 2.3's sanitizer uses key.replace("vocoder.", "") — it strips every
        # occurrence, not just the leading one, so the vocoder file's two
        # sibling generators (the main one, itself named "vocoder", and
        # "bwe_generator") land at different depths: the main one flattens to
        # the component root ("act_post..."), its sibling keeps one level
        # ("bwe_generator.act_post..."). That is a known latent defect, kept
        # in ltx_23.py because its published packs and the ComfyUI nodes that
        # load them already depend on the flattened names — changing it there
        # is a breaking change to a shipped artefact.
        #
        # 2.5 has no runtime yet, so nothing depends on the flattened form,
        # and sanitize_vocoder_key strips only the leading "vocoder." — both
        # generators keep their name at the same depth. This test pins that
        # difference explicitly rather than letting a future "fix" for 2.5
        # silently re-import 2.3's bug via a parity assertion like the ones
        # above.
        key = "vocoder.vocoder.act_post.act.alpha"
        assert key in keys(AUDIO)
        assert ltx_25.sanitize_vocoder_key(key) == "vocoder.act_post.act.alpha"
        assert ltx_23.sanitize_vocoder_key(key) == "act_post.act.alpha"

    def test_the_parity_claim_is_not_vacuous(self):
        # A parity test over an empty key set passes and proves nothing. The
        # fixture is reduced to keys whose numeric path components are all 0
        # or 1 (tests/test_ltx25_keys.py), so these thresholds are sized to
        # that reduced set, not the full ~170/1329-tensor upstream files.
        assert len([k for k in keys(VAE_CONV) if k.startswith("decoder.")]) > 5
        assert len([k for k in keys(AUDIO) if k.startswith("vocoder.")]) > 20
