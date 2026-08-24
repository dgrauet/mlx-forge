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

The vocoder was briefly a deliberate exception (the 2.5 sanitizer kept the
main generator nested one level deeper than 2.3's flattened layout); the
ltx-2-mlx loader crashed on the 667 renamed parameters, settling the
question: the 2.3 emitted layout is the published contract, and the vocoder
is under the same parity lock as everything else.
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

    def test_vocoder(self):
        # The regression that shipped: the main generator (itself named
        # "vocoder") must flatten to the component root exactly as 2.3 emits
        # it — the loader crashed on the nested form. Architecture identical,
        # so ANY naming divergence from 2.3 here is a recipe bug.
        for key in keys(AUDIO):
            if not key.startswith("vocoder."):
                continue
            assert ltx_25.sanitize_vocoder_key(key) == ltx_23.sanitize_vocoder_key(key), key

    def test_vocoder_sanitized_keys_do_not_collide(self):
        # Flattening the main generator moves 667 keys to the component
        # root; none may collide with bwe_generator.* or mel_stft.*.
        sanitized = [
            ltx_25.sanitize_vocoder_key(k) for k in keys(AUDIO) if k.startswith("vocoder.")
        ]
        assert len(sanitized) == len(set(sanitized))

    def test_the_parity_claim_is_not_vacuous(self):
        # A parity test over an empty key set passes and proves nothing. The
        # fixture is reduced to keys whose numeric path components are all 0
        # or 1 (tests/test_ltx25_keys.py), so these thresholds are sized to
        # that reduced set, not the full ~170/1329-tensor upstream files.
        assert len([k for k in keys(VAE_CONV) if k.startswith("decoder.")]) > 5
        assert len([k for k in keys(AUDIO) if k.startswith("vocoder.")]) > 20
