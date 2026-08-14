# Anatomy of a recipe — what varies, and why

A reader opening two recipes side by side sees a lot of difference and has no
way to tell which of it is *forced* and which is *drift*. This document draws
that line.

**The rule:** a recipe should only differ from its neighbours along the six
axes in [Intrinsic variation](#intrinsic-variation-imposed-by-upstream). Every
other difference is either already factored into the shared layer, or is a
known gap listed in [Not yet declarative](#not-yet-declarative).

---

## Intrinsic variation (imposed by upstream)

These differ because the upstream PyTorch repositories genuinely differ. They
are model facts, not style. Unifying them would create false equivalences.

| Axis | Why it cannot be shared |
|---|---|
| **Key classification / sanitization** | Every checkpoint names its weights differently. `sanitize_*` encodes one upstream's naming, verified key by key. |
| **Quantization scope** (`should_quantize`) | Which layers survive int4/int8 depends on the architecture — an embedding table is fine to quantize in one model and destroys quality in another. |
| **Conv transposition** | PyTorch is channels-second, MLX channels-last, but *which* tensors are convs (and of what rank) is model-specific. `attn.proj.weight` is a Linear despite matching a patch-embed's name. |
| **Source format** | safetensors, `.pt`/`.pth` (5 recipes), fp8 with per-row scales (ideogram-4), sharded or single, sometimes nested inside a pickled container. |
| **Component decomposition** | 1 output file (ernie-image-pe) to 6 (LTX-2.3, matrix-game), or two independent stages (hunyuan3d). |
| **Pipeline files** | Which tokenizer/scheduler files the runtime needs, and whether they keep their directory. |

Everything below this line should look the same across recipes.

---

## Shared layer (already factored)

Reach for these before writing anything new. A recipe that reimplements one of
them is drifting.

| Concern | Helper |
|---|---|
| Common convert flags | `convert.add_common_convert_args` |
| Default output dir `models/<name>-mlx[-q<bits>]` | `convert.default_output_dir` |
| Download from the Hub | `convert.download_hf_files` |
| Load `.pt`/`.pth`/`.ckpt` | `convert.load_torch_state_dict` |
| Load safetensors (sharded or not) | `convert.load_weights` / `load_safetensors` |
| Per-component sanitize → transform → materialize → save | `convert.process_component` |
| Quantize a component in place | `convert.quantize_component` |
| Copy pipeline files, strictly | `convert.copy_required_files` |
| Write `split_model.json` | `convert.write_split_model` |
| End-of-conversion listing | `convert.print_output_summary` |
| Record / read quantization | `quantize.write_quantize_config` / `read_quantize_config` |
| Open / close a validation run | `validate.start_validation` / `finish_validation` |
| Conv layout | `transpose.transpose_conv` |
| Publication metadata | `metadata.RecipeMetadata` |
| Persist operator-supplied card metadata | `upload.persist_card_metadata` |

Each of these replaced between 2 and 17 hand-written copies. Three latent
defects were found in the process — the copies had drifted:

- two recipes' `validate()` reported failures and exited **0**;
- matrix-game gated its quantization checks on a file it never wrote, so they
  **never ran**;
- `void-model --dry-run` **downloaded** several GB before honouring the flag.

That is the argument for using the shared layer: not line count, but that
independent copies diverge silently.

---

## Recipe map

| Recipe | Upstream | Source format | Components | Recipe-specific flags |
|---|---|---|---|---|
| `ltx-2.3` | Lightricks/LTX-2.3 | safetensors | 6 + upscalers + LoRAs | `--variant --skip-shared --spatial-upscaler --temporal-upscaler --lora` |
| `ideogram-4` | ideogram-ai/ideogram-4-fp8 | **fp8** + scales | 4 | `--source` |
| `matrix-game-3.0` | Skywork/Matrix-Game-3.0 | safetensors + `.pth` | 6 | `--dit/--t5/--vae-checkpoint --skip-tokenizer` |
| `cogvideox-fun-v1.5-5b-inp` | alibaba-pai/CogVideoX-Fun-V1.5-5b-InP | safetensors | 3 | `--source` |
| `void-model` | netflix/void-model | safetensors | 2 passes | `--source` |
| `hunyuan3d-2.1` | tencent/Hunyuan3D-2.1 | `.ckpt` + `.bin` + safetensors | 2 stages × 3 | `--stage --checkpoint --local-path --dino-path` |
| `ernie-image` | baidu/ERNIE-Image(-Turbo) | safetensors | 3 | `--variant --checkpoint` |
| `ernie-image-pe` | baidu/ERNIE-Image-Turbo `/pe` | safetensors | 1 | `--checkpoint` |
| `vjepa-2.1-vitl` | Meta CDN `.pt` (not on the Hub) | `.pt`, pickled wrapper | 2 | `--source` |
| `vjepa-2.0-vitl` | Meta CDN `.pt` (not on the Hub) | `.pt` | 2 + up to 3 probes | `--source --ssv2/--diving48/--ek100-source` |

Reading a recipe: the flags column tells you what the upstream forces on the
operator. Everything not in that column is the shared layer and behaves
identically everywhere.

---

## Declarative metadata

Each recipe declares a `RecipeMetadata` (`src/mlx_forge/metadata.py`):

```python
METADATA = RecipeMetadata(
    source="Skywork/Matrix-Game-3.0",
    links=["Code: https://github.com/dgrauet/Matrix-Game-mlx"],
)
```

`convert` persists it into `split_model.json`; `upload` reads it back for the
card. Operator flags still win for one-offs, and anything they pass
(`--usage-url`, `--link`, `--cli-snippet`) is written back so the next
`--card-only` refresh keeps it.

This replaced metadata that lived only in CLI flags, which is why the published
cards below track neither the recipe nor a rule.

### Licence

Converting and quantising produces a **derivative**, so what upstream attaches
to its weights travels with ours. Three fields mirror the upstream repo's own
declaration — never a paraphrase, never a guess:

```python
license="other",                    # SPDX id; "other" identifies nothing alone
license_name="ltx-2-community-license-agreement",
license_link="https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
license_file="LICENSE",             # or a tuple: ("LICENSE", "Notice.txt")
```

`license_file` is the one with teeth. The community licences (LTX-2.x §3.2,
Tencent Hunyuan) oblige whoever distributes a derivative to give the recipient
*a copy of the agreement*; a link in the front-matter does not discharge that.
Declaring it makes `convert` fetch the file verbatim from upstream and `upload`
**refuse to publish** without it. Leave it `None` for apache-2.0 or mit, which
the SPDX identifier satisfies on its own.

The fetch hangs off `write_split_model()`, the one function all ten recipes
call, so a recipe cannot forget it. It is best-effort there (a Hub hiccup must
not destroy a long conversion) and strict in the upload path, which is where the
obligation actually binds.

### Evidence from the published cards

Measured against the 21 repos under `dgrauet/` on the Hub (July 2026):

| Symptom | Repos affected |
|---|---|
| Files section omits at least one file the repo contains | **21 of 21** |
| No `base_model` in the front-matter | `vjepa-2.1-vitl-mlx` |
| No Usage section | `matrix-game-3.0-mlx` |

Concretely:

- **`matrix-game-3.0-mlx`** — the card omits all four `google/umt5-xxl/*`
  tokenizer files. They are in the repo; the card could not see into a
  subdirectory.
- **`CogVideoX-Fun-V1.5-5b-InP-mlx-q8`** — omits `spiece.model`, i.e. the very
  file whose absence caused two earlier incidents.
- **`ltx-2.3-mlx-q8`** — omits `transformer-distilled-1.1.safetensors` and its
  LoRA. Those were added by the documented delta workflow
  (`--skip-shared` + `--add-only`), and the card **never mentions them at
  all**.
- **The two vjepa repos are inverted relative to their code.**
  `vjepa-2.1-vitl-mlx` has **no** `base_model` even though its recipe *does*
  write `"source"` — the published `split_model.json` predates that field:

  ```json
  {"model_name": "vjepa2-vit-l-rope-mlx", "components": {...}, "quantized": false}
  ```

  `vjepa-2.0-vitl-mlx` **has** `base_model` even though its recipe writes no
  `source` at all (its file is the flat `{component: filename}` table) — the
  operator passed `--base-model` by hand that day.

  So the published metadata reflects neither the recipe nor a rule, but what
  was typed at upload time. That is the failure mode of metadata living in
  flags, and it is why a declarative source of truth is the fix rather than
  more flags.

All of the above is fixed for *future* uploads: the card lists what the upload
publishes, its file list is derived from the remote merged with what is about
to go up (so a delta upload no longer produces a card that contradicts its own
variant list), and `source` / `cli_snippet` now persist. Two caveats remain:

1. ~~`vjepa-2.0` still writes no `source`.~~ **Resolved.** The caution was
   based on a guessed consumer. `vjepa2-core-mlx` resolves components through
   `manifest.get("components", {})` and never iterates the top-level keys, so
   adding `source` is safe — and the flat table it used to write was being
   ignored outright, leaving its predictor and probes unaddressable. It now
   writes the same nested shape as vjepa-2.1. **Check the consumer before
   assuming a format is load-bearing.**
2. **Every already-published card stays stale** until its model is
   re-uploaded, or refreshed with `mlx-forge upload <dir> --card-only`.

`split_model.json` also has three incompatible schemas — most recipes write
`{format, components, source, …}`, vjepa-2.0 a flat `{component: filename}`
table, vjepa-2.1 a `{model_name, components, quantized}` record, and
void-model wrote none at all until it was given one (without it,
`mlx-forge upload` on a void model refused to run without `--repo-id`), and
vjepa-2.0 wrote a flat table no consumer could read until it was nested.

**Those schemas are deliberately not unified.** Their non-common fields have
no reader, and rewriting them would change metadata already published on the
Hub. `RecipeMetadata` adds the fields that *are* read, additively, so existing
consumers keep working.

One caution: **vjepa-2.0's top-level keys *are* its component table.** Adding
any key there changes what a downstream loader iterating the file sees. That
recipe needs its consumer checked before it changes.

---

## Adding a recipe

Implement the six functions the CLI dispatches to (`convert` / `validate` /
`split` plus their `add_*_args`), register it in `recipes/__init__.py`, and use
the shared layer for everything outside the six intrinsic axes. See the
"Adding a New Recipe" section of `CLAUDE.md` for the contract, which
`tests/test_recipe_contract.py` enforces across every registered recipe.
