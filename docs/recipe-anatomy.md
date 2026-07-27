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

## Not yet declarative

This is the remaining accidental variation, and where the "why is this
per-recipe?" feeling now comes from. Card and repo metadata are passed as CLI
flags at upload time instead of being declared once by the recipe.

| Metadata | Declared by the recipe? | Consequence |
|---|---|---|
| `source` (→ `base_model` on the Hub) | 9 of 10 (not vjepa-2.0) | the published cards do not track the code either way — see below |
| `links` | 1 of 10 (matrix-game) | operator must re-pass `--link` on every refresh |
| `usage_url` | 0 of 10 | idem |
| `cli_snippet` | **not persisted anywhere** | a `--card-only` refresh **silently drops the Usage section** |

That last row is a live defect: `CLAUDE.md` documents `--card-only` as
idempotent ("re-running always produces a card matching the current remote
state"), but `cli_snippet` is the one card variable with no fallback, so
regenerating without re-passing `--cli-snippet` publishes a card missing the
section.

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

The first row is largely addressed for *future* uploads (the card now lists
what the upload publishes), but two structural issues remain:

1. **Sections have inconsistent provenance.** `transformer_variants` is
   derived from the *remote* repo, `model_files` from the *local* directory,
   `cli_snippet` from the *command line*. A `--card-only` refresh after a delta
   upload therefore produces a card whose variant list and file list disagree,
   which is exactly what `ltx-2.3-mlx-q8` shows.
2. Every already-published card stays stale until its model is re-uploaded.

`split_model.json` also has three incompatible schemas — most recipes write
`{format, components, source, …}`, vjepa-2.0 a flat `{component: filename}`
table, vjepa-2.1 a `{model_name, components, quantized}` record, and
void-model writes none at all (so `mlx-forge upload` on it fails without
`--repo-id`).

**The fix is not to unify those schemas.** Their non-common fields have no
reader, and rewriting them would change metadata already published on the Hub.
The fix is a declarative `RecipeMetadata` per recipe — source, links, usage
URL, CLI snippet — persisted at convert time and read back at upload time, with
CLI flags still overriding for one-offs. That is additive, so existing
consumers keep working. Paired with it, the card's Files section should be
derived from the *remote* listing like `transformer_variants` already is, so
every section of a card describes the same repo.

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
