# Fixtures

- `ltx_25_keys.json` — LTX-2.5 upstream keys, `scripts/harvest_ltx25_keys.py`.
- `published/<repo>.json` — golden records of OUR published packs (Tier 1 of the
  factorisation spec): reduced keys, dtype histogram, tensor_count per file. Harvested
  2026-08-28 with `scripts/harvest_keys.py --repo dgrauet/<repo> --file ...`.
- `upstream/<name>.json` — the same record for the UPSTREAM sources each component is
  converted from (Tier 2). Safetensors sources via `--repo`; pickled sources via
  `--torch <file> [--section KEY]` on a local download, then deleted. Sections used:
  - hunyuan3d shape `.ckpt` (`model.fp16.ckpt`) → `model` / `vae` / `conditioner`
    (`hunyuan3d-shape-model.json` / `hunyuan3d-shape-vae.json` /
    `hunyuan3d-shape-conditioner.json`).
  - hunyuan3d paint `unet`/`vae` `.bin` → no section (flat state dict).
  - matrix-game `models_t5_umt5-xxl-enc-bf16.pth` → no section (flat state dict).
  - matrix-game `Wan2.2_VAE.pth` → no section (flat state dict).
  - matrix-game `MG-LightVAE.pth` / `MG-LightVAE_v2.pth` → wrapped two levels deep,
    `state_dict.gen_model` (the harvester's single-level `--section` can't reach this;
    both were harvested with a 3-line one-off that drills `ckpt["state_dict"]["gen_model"]`
    and feeds it to `summarise()` — matches `_extract_state_dict`'s recursive unwrap in
    `matrix_game_3_0.py`).
  - vjepa 2.0 `vitl.pt` → `target_encoder` / `predictor`.
  - vjepa 2.1 `vjepa2_1_vitl_dist_vitG_384.pt` → `ema_encoder` / `predictor`.
  - vjepa 2.0 probes (`ssv2-vitl.pt`, `diving48-vitl.pt`, `ek100-vitl.pt`) → no
    `--section` support for a list index, harvested with a 3-line one-off:
    `torch.load(p, weights_only=True)["classifiers"][0]` fed to `summarise()`.
- No records were built without their upstream source (`"source":
  "derived-from-published"`): the V-JEPA 2.0/2.1 encoder, predictor, and probe
  checkpoints are not on the Hub, but the operator downloaded them from the public
  `dl.fbaipublicfiles.com/vjepa2/...` URLs listed in the facebookresearch/vjepa2
  GitHub README (2.0 ViT-L: `vitl.pt`; 2.1 ViT-L: `vjepa2_1_vitl_dist_vitG_384.pt`;
  the three ViT-L attentive probes under `evals/`), so every V-JEPA component has a
  real upstream record.

Regenerate a record only when the upstream file or our published pack changes; a
diff in `published/` means a published pack changed, which the factorisation must never
cause.

## `keep()` reduction rule

`scripts/harvest_keys.py`'s `keep()` treats a key as carrying a repeat-block index only
when a full dot-separated path segment is all digits (e.g. `blocks.0.attn.weight` keeps
segment `0`). It does NOT treat a digit embedded in a component name as an index — e.g.
`t5_encoder.block.0.w`, `vae_lightvae_v2.decoder.1.w`, and the vjepa `ssv2_probe.*` /
`ek100_probe.*` / `diving48_probe.*` keys keep all their non-block-index segments. An
earlier version matched any digit anywhere in the key, which zeroed out every kept key
for `t5_encoder.safetensors`, `vae_lightvae_v2.safetensors`, and all three V-JEPA 2.0
probe files; fixed in commit 879a72a alongside `tests/test_harvest_keys.py` coverage.
