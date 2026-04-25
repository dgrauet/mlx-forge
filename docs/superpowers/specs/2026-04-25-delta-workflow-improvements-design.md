# Delta-Workflow Improvements — Design

**Date:** 2026-04-25
**Status:** Approved (awaiting user review of written spec)
**Scope:** Single implementation cycle covering 3 related features.

## Background

This session shipped a new LTX-2.3 transformer variant (`distilled-1.1`) and matching LoRA across three HF repos (`dgrauet/ltx-2.3-mlx`, `dgrauet/ltx-2.3-mlx-q8`, `dgrauet/ltx-2.3-mlx-q4`). The workflow surfaced three friction points in the current pipeline:

1. **`mlx-forge convert` regenerates all shared components** (connector, vae_*, audio_vae, vocoder ≈ 7 GB) even when adding a single new transformer variant. ~30% of conversion time and disk is wasted.
2. **`mlx-forge upload` has no delta mode** — uploading the entire model directory after a delta convert overwrites `split_model.json` and the model card with content that lists only the new variant (corrupts metadata for previously uploaded variants). Working around this required dropping into raw `huggingface_hub` calls.
3. **Model card refresh is impractical** — `--card-only` reads the local `split_model.json`, which after a delta convert lists only the new variant. There's no built-in way to regenerate the card with the union of remote and local variants.

These three features are technically independent but share the "delta workflow" theme and a single implementation cycle.

## Goals

- Reduce time/disk cost of adding a variant to an existing repo (convert).
- Make upload of a delta directory safe-by-default (upload).
- Allow refreshing the model card to reflect the actual remote state (card refresh).
- Document the resulting workflow so agents (and humans) can use it without rediscovery.
- Keep all existing default behaviors unchanged. New flags are opt-in.

## Non-Goals

- No retry logic / rate-limit handling (see "Why no retry" below).
- No SHA-based dedup in `--add-only`. Skip is by filename only.
- No per-recipe model-card template overrides. Single global template for v1.
- No multi-recipe rollout in this cycle. Pattern proven on `ltx-2.3` first; other recipes adopt opt-in later.

### Why no retry

Implementation considered and rejected. The two upload failures observed this session (q8 hang at 99%, q4 stall at 45%) didn't surface explicit 429s — they look like server-side bandwidth throttling that the user can recover from manually. Adding retry logic would add complexity for a problem that may not actually be a 429 / classic rate limit. Revisit if real 429s surface.

## Architecture

Three additive features extending the `convert → validate → upload` pipeline:

```
convert --skip-shared       →  delta directory (transformer + LoRA only;
                                 split_model.json includes "delta": true)
validate                     →  auto-detects "delta": true, skips shared-component checks
upload --add-only            →  skips files already present on remote (by name)
upload --card-only           →  regenerates card via Jinja2 template, variants
                                 derived from remote repo (not local manifest)
```

Compatibility: existing workflows (full convert, full upload, default card generation) are unchanged. All new flags are opt-in. The Jinja2 template replaces the existing `generate_model_card()` Python string-builder for ALL uploads, with output textually equivalent to the current implementation.

## Components

### 1. `convert --skip-shared`

**CLI surface** (recipe-level — added to `add_convert_args` in each recipe that opts into delta mode; LTX-2.3 first):

```bash
mlx-forge convert ltx-2.3 --variant distilled-1.1 --skip-shared --output models/ltx-2.3-mlx-delta-1.1
```

**Behavior**:

- Recipe receives `args.skip_shared = True`.
- In the recipe's variant-conversion loop, only the requested transformer variants and the LoRA-sync step run.
- Skipped components: `connector`, `vae_decoder`, `vae_encoder`, `audio_vae`, `vocoder`, `vae_shared_stats` (LTX-2.3 list — other recipes have analogous component sets).
- `split_model.json` written to the output dir gains two fields:
  - `"delta": true`
  - `"components": [...]` — sorted list of components actually written (e.g., `["transformer-distilled-1.1"]`).
- `embedded_config.json` and `config.json` are written normally (cheap, useful for downstream consumers).

**Implementation locus**: ~10 lines in `_convert_variant` (LTX-2.3) — guard around the shared-component loop. Plus the flag declaration in `add_convert_args`. Pattern is to be replicated in other recipes when they opt in.

### 2. Validate auto-detects delta mode

**Behavior**:

- `validate.py` reads `split_model.json` (already does).
- If `"delta": true`:
  - Logs `[INFO] Delta mode (skipping shared component checks)` at the top of the report.
  - Skips checks that assert presence/structure of shared components NOT listed in `"components"`.
  - Still runs all checks for the components that ARE listed (transformer, LoRA, etc.).
- If the `delta` key is absent or false: current strict behavior unchanged.
- If `"delta": true` but a declared component is missing (e.g., transformer file gone): FAIL with clear message.

**Implementation locus**: small refactor in the recipe's `validate()` (LTX-2.3) — wrap the shared-component checks in `if not delta:`. The framework helpers in `validate.py` are unchanged.

### 3. `upload --add-only`

**CLI surface**:

```bash
mlx-forge upload models/ltx-2.3-mlx-delta-1.1 --repo-id dgrauet/ltx-2.3-mlx --add-only
```

**Behavior**:

1. Calls `api.model_info(repo_id)` once.
2. Builds a set `remote_filenames = {s.rfilename for s in info.siblings}`.
3. Iterates over local files matching `allow_patterns=["*.safetensors", "*.json", "README.md"]`.
4. For each file whose name is in `remote_filenames`: print `Skipped (on remote): {name}`.
5. For each file whose name is NOT in `remote_filenames`: upload via `api.upload_file` (one commit per file — more resilient than `upload_folder` against the hangs we saw this session).
6. If no files are new: print `Nothing to upload (all N files already on remote)` and exit 0.

**Refusals**:

- Non-existent repo: SystemExit with message instructing to do a normal upload first.
- `model_info` failure (network/auth): SystemExit before any upload attempt.

**Per-file commits**: each upload is its own commit. The repo will show N commits instead of 1 — acceptable cosmetic tradeoff for resilience. Commit messages follow the existing `commit_message` argument; if multiple files, append the filename to the message.

### 4. Card template (Jinja2) + remote-aware refresh

**New runtime template**: `src/mlx_forge/templates/model-card.md.j2`

- Loaded via `importlib.resources` (proper packaging path, not relative-from-`__file__`).
- Equivalent in output to the current `generate_model_card()` for non-delta uploads.
- Conditional sections via `{% if %}` blocks (frontmatter base_model, details lines, usage URL, CLI snippet, related projects, file listing).

**Refonte of `generate_model_card`**:

- Becomes a thin wrapper: builds a context dict, calls `template.render(**ctx)`.
- Context keys mirror today's parameters: `repo_id`, `base_model`, `transformer_variants`, `quantized`, `bits`, `model_version`, `usage_url`, `cli_snippet`, `links`, `model_files`, `lora_files`, `license_id`.
- New keys for v1: `lora_files` (list), `transformer_variants` (list).

**`--card-only` remote-aware refresh**:

When `--card-only` is set AND the repo exists on HF:

1. Call `api.model_info(repo_id)`.
2. Derive `transformer_variants` from sibling filenames matching `transformer-*.safetensors` (strip prefix/suffix).
3. Derive `lora_files` from sibling filenames matching `*lora*.safetensors`.
4. Render template with these REMOTE-derived lists, NOT the local `split_model.json` lists.
5. Write `README.md` to the local model_dir.
6. Upload via `api.upload_file` as today.

This makes `--card-only` idempotent: re-running it always produces a card matching the current remote state, regardless of what the local model_dir contains.

**Edge cases**:

- Empty remote / no transformers detected: `transformer_variants=[]`, template's `{% if transformer_variants %}` omits the section.
- `--card-only` without `--repo-id` and without a local `split_model.json` to derive it from: existing error path.

## Data Flow — End-to-End Delta Workflow

Reproducing this session's actual scenario (adding `distilled-1.1` to three repos):

```bash
# 1. Convert delta — once per quantization level (fp16/q8/q4)
mlx-forge convert ltx-2.3 \
    --variant distilled-1.1 \
    --lora distilled-384-1.1 \
    --skip-shared \
    --output models/ltx-2.3-mlx-delta-1.1
# ≈3 min instead of ≈25 min for full convert

# 2. Validate (auto-detects delta)
mlx-forge validate ltx-2.3 models/ltx-2.3-mlx-delta-1.1
# [INFO] Delta mode (skipping shared component checks)
# == Transformer Weights (distilled-1.1) ==  ✓
# ...

# 3. Upload delta to each repo
mlx-forge upload models/ltx-2.3-mlx-delta-1.1 \
    --repo-id dgrauet/ltx-2.3-mlx --add-only
# Skipped (on remote): connector.safetensors, vae_decoder.safetensors, ...
# Uploading: transformer-distilled-1.1.safetensors
# Uploading: ltx-2.3-22b-distilled-lora-384-1.1.safetensors

# 4. Refresh card on each repo (idempotent)
mlx-forge upload models/ltx-2.3-mlx-delta-1.1 \
    --repo-id dgrauet/ltx-2.3-mlx --card-only
# Detected variants on remote: distilled, dev, distilled-1.1
# Detected LoRAs on remote: distilled-384, distilled-384-1.1
# Uploaded: README.md
```

Steps 1 and 2 run once per quantization level. Steps 3 and 4 run once per repo (3 times — fp16, q8, q4).

## Error Handling

| Surface | Case | Behavior |
|---------|------|----------|
| `--skip-shared` | Multiple variants | Allowed — process each transformer, skip shared once |
| `--skip-shared` | Combined with `--quantize` | Allowed — quantize transformer, no shared to quantize |
| `--add-only` | Repo doesn't exist | SystemExit with message ("use a normal upload to create the repo first") |
| `--add-only` | `model_info` fails | SystemExit before any upload attempt |
| `--add-only` | All files already on remote | Print message, exit 0, no `upload_file` calls |
| `--add-only` | File appears between `model_info` and `upload_file` | huggingface_hub error propagates as-is — no custom retry |
| `--card-only` | Repo empty / no transformers detected | Render with `transformer_variants=[]`; template omits the section |
| `--card-only` | No `--repo-id` and no local `split_model.json` | Existing error path unchanged |
| `--card-only` | Template file missing from package | `FileNotFoundError` propagates — install bug, not a user case |
| Validate delta | `delta: true` but transformer missing | FAIL with explicit message |
| Validate delta | `delta` key absent | Current strict behavior, unchanged |

## Testing

Tests added/extended in existing files:

- `tests/test_ltx23_keys.py` — argparse accepts `--skip-shared`, propagates to `args.skip_shared`. Convert with mocked recipe verifies output dir contents (only transformer + LoRA + manifest), `split_model.json` has `delta: true` and correct `components` list.
- `tests/test_validate.py` — three cases: `delta: true` + transformer present (PASS), `delta: true` + transformer missing (FAIL with message), no `delta` key (current strict behavior, regression check).
- `tests/test_upload.py`:
  - `--add-only` with mocked `api.model_info`/`api.upload_file`: assert `upload_file` called only for absent files.
  - `--add-only` against repo not found: SystemExit.
  - `--add-only` with all files present: no upload calls, exit 0.
  - Card template renders correctly for canonical inputs (snapshot test).
  - `--card-only` derives variants from mocked remote, not local.
- `tests/test_integration.py` — end-to-end golden flow: convert delta → validate → upload (mocked) → upload card (mocked).

## Documentation

### `CLAUDE.md` — new "Delta workflow" section (~25-30 lines)

Concise, agent-facing, explaining the four-step workflow with the actual CLI commands. Includes the refusal cases (e.g., `--add-only` on a non-existent repo).

### `.claude/skills/mlx-recipe/SKILL.md` — new "Model card template" section (~40 lines)

- Path: `src/mlx_forge/templates/model-card.md.j2`.
- List of available variables: `repo_id`, `base_model`, `transformer_variants`, `quantized`, `bits`, `model_version`, `usage_url`, `cli_snippet`, `links`, `model_files`, `lora_files`, `license_id`.
- Excerpt of the template's main blocks (frontmatter, body, conditional sections).
- Customization: edit the file directly. No per-recipe override (out of scope v1).

### Template file location

Template lives at `src/mlx_forge/templates/model-card.md.j2`. The skill **describes** and **points to** it; the package **owns** it. No file duplication.

## Surface Modified

- `src/mlx_forge/convert.py` — propagate `skip_shared` flag from CLI args.
- `src/mlx_forge/recipes/ltx_23.py` — honor `args.skip_shared`; emit `delta: true` in `split_model.json`; adapt `validate()`.
- `src/mlx_forge/upload.py` — `--add-only` flag; refonte of `generate_model_card` to load Jinja2 template; `--card-only` augmentation to derive variants from remote.
- `src/mlx_forge/cli.py` — wire `--add-only` into the upload subparser.
- New: `src/mlx_forge/templates/model-card.md.j2`.
- `CLAUDE.md` — new "Delta workflow" section.
- `.claude/skills/mlx-recipe/SKILL.md` — new "Model card template" section.
- `tests/` — extensions per Testing section.

## Out of Scope (deferred)

- Generalizing `--skip-shared` to recipes other than LTX-2.3. Pattern documented; other recipes opt in case-by-case.
- Per-recipe model-card template overrides.
- Retry/backoff for upload failures.
- SHA-based dedup in `--add-only`.
