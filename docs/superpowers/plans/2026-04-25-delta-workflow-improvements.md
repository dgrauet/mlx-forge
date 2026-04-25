# Delta-Workflow Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--skip-shared` to convert, auto-detect delta mode in validate, `--add-only` for upload, Jinja2 model card template with remote-aware refresh, and document the resulting delta workflow.

**Architecture:** Three additive features extending the convert/validate/upload pipeline (all opt-in, default behavior preserved). The Jinja2 template replaces the current Python string-builder for model cards (output textually equivalent for non-delta cases). Documentation lives in `CLAUDE.md` (workflow) and `.claude/skills/mlx-recipe/SKILL.md` (template internals).

**Tech Stack:** Python 3.11+, argparse, huggingface_hub (Jinja2 already a transitive dep), pytest, ruff.

**Reference spec:** `docs/superpowers/specs/2026-04-25-delta-workflow-improvements-design.md`

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `src/mlx_forge/recipes/ltx_23.py` | LTX-2.3 recipe — convert/validate | Modify |
| `src/mlx_forge/upload.py` | Upload + card generation | Modify (refonte `generate_model_card`, add `--add-only` logic, augment `--card-only`) |
| `src/mlx_forge/cli.py` | CLI subcommand wiring | Modify (add `--add-only` arg) |
| `src/mlx_forge/templates/model-card.md.j2` | Jinja2 template for model card | Create |
| `src/mlx_forge/templates/__init__.py` | Make templates a package resource | Create |
| `tests/test_ltx23_delta.py` | Tests for delta-mode convert + validate | Create |
| `tests/test_upload.py` | Tests for `--add-only`, template rendering, remote-aware card refresh | Modify |
| `tests/test_integration.py` | End-to-end delta workflow test | Modify |
| `CLAUDE.md` | Agent-facing delta workflow doc | Modify |
| `.claude/skills/mlx-recipe/SKILL.md` | Template-customization doc | Modify |

---

## Task 1: Add `--skip-shared` flag (LTX-2.3 argparse)

**Files:**
- Modify: `src/mlx_forge/recipes/ltx_23.py:1293` (function `add_convert_args`)
- Test: `tests/test_ltx23_delta.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ltx23_delta.py`:

```python
"""Tests for LTX-2.3 delta-mode convert/validate."""

import argparse

from mlx_forge.recipes.ltx_23 import add_convert_args


class TestSkipSharedFlag:
    def test_default_is_false(self):
        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        args = parser.parse_args([])
        assert args.skip_shared is False

    def test_flag_sets_true(self):
        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        args = parser.parse_args(["--skip-shared"])
        assert args.skip_shared is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ltx23_delta.py -v`
Expected: FAIL — `args.skip_shared` AttributeError or `--skip-shared` unknown argument.

- [ ] **Step 3: Add the flag**

In `src/mlx_forge/recipes/ltx_23.py`, locate `add_convert_args` (around line 1293). Add this argument near `--dry-run`:

```python
    parser.add_argument(
        "--skip-shared",
        action="store_true",
        help=(
            "Delta mode: convert only the requested transformer variant(s) and LoRAs. "
            "Skips connector, vae_*, audio_vae, vocoder, vae_shared_stats, and upscalers. "
            "Marks split_model.json with 'delta': true. "
            "Use when adding a variant to an existing repo."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ltx23_delta.py -v`
Expected: PASS (2/2)

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/mlx_forge/recipes/ltx_23.py tests/test_ltx23_delta.py`
Expected: All checks passed.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_forge/recipes/ltx_23.py tests/test_ltx23_delta.py
git commit -m "feat(ltx-2.3): add --skip-shared flag (argparse only)"
```

---

## Task 2: Honor `--skip-shared` in convert flow

**Files:**
- Modify: `src/mlx_forge/recipes/ltx_23.py:733-883` (function `convert`)
- Test: `tests/test_ltx23_delta.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ltx23_delta.py`:

```python
import json
import types
from pathlib import Path
from unittest.mock import patch


def _make_convert_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_convert_args(parser)
    args = parser.parse_args([])
    args.checkpoint = None
    args.variant = ["distilled-1.1"]
    args.output = str(tmp_path)
    args.spatial_upscaler = []
    args.temporal_upscaler = []
    args.spatial_upscaler_checkpoint = []
    args.temporal_upscaler_checkpoint = []
    args.lora = []
    args.quantize = False
    args.bits = 8
    args.group_size = 64
    args.dry_run = False
    args.skip_shared = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestSkipSharedConvertEffect:
    def test_skip_shared_forces_is_first_false(self, tmp_path):
        """When skip_shared is True, _convert_variant must be called with is_first=False."""
        from mlx_forge.recipes import ltx_23

        args = _make_convert_args(tmp_path, skip_shared=True)
        captured: list[bool] = []

        def fake_convert_variant(args, variant, output_dir, *, is_first):
            captured.append(is_first)
            return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)

        with patch.object(ltx_23, "_convert_variant", side_effect=fake_convert_variant):
            ltx_23.convert(args)

        assert captured == [False]  # is_first must be forced to False with skip_shared

    def test_split_model_json_has_delta_flag(self, tmp_path):
        from mlx_forge.recipes import ltx_23

        args = _make_convert_args(tmp_path, skip_shared=True)

        def fake_convert_variant(args, variant, output_dir, *, is_first):
            return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)

        with patch.object(ltx_23, "_convert_variant", side_effect=fake_convert_variant):
            ltx_23.convert(args)

        split = json.loads((tmp_path / "split_model.json").read_text())
        assert split["delta"] is True
        assert split["components"] == []  # no shared components in delta mode
        assert split["transformer_variants"] == ["distilled-1.1"]

    def test_normal_mode_no_delta_flag(self, tmp_path):
        from mlx_forge.recipes import ltx_23

        args = _make_convert_args(tmp_path, skip_shared=False)

        def fake_convert_variant(args, variant, output_dir, *, is_first):
            return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)

        with patch.object(ltx_23, "_convert_variant", side_effect=fake_convert_variant):
            ltx_23.convert(args)

        split = json.loads((tmp_path / "split_model.json").read_text())
        assert "delta" not in split or split["delta"] is False
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_ltx23_delta.py::TestSkipSharedConvertEffect -v`
Expected: 3 FAILs (no skip_shared logic in convert yet).

- [ ] **Step 3: Modify `convert()` in `src/mlx_forge/recipes/ltx_23.py`**

Two changes:

**Change A** (around line 760): force `is_first=False` when skip_shared:

```python
    skip_shared = getattr(args, "skip_shared", False)
    for i, variant in enumerate(variants):
        print(f"\n{'=' * 60}")
        print(f"Converting variant: {variant} ({i + 1}/{len(variants)})")
        print("=" * 60)

        cfg, cross_attn_adaln, count = _convert_variant(
            args, variant, output_dir, is_first=(i == 0 and not skip_shared)
        )
        total_weights += count
        variant_adaln[variant] = cross_attn_adaln
        if config is None:
            config = cfg
```

**Change B** (around line 779-818): wrap the upscaler loops so they're skipped under delta mode:

```python
    # Convert upscalers (skipped in delta mode)
    upscaler_components = []
    download_dir = Path("models") / "ltx-2.3-src"

    if not skip_shared:
        for i, scale in enumerate(args.spatial_upscaler):
            # ...existing spatial upscaler loop unchanged...

        for i, scale in enumerate(args.temporal_upscaler):
            # ...existing temporal upscaler loop unchanged...
```

**Change C** (around line 841-855): emit `delta` flag and `[]` components:

```python
    # Create split_model.json
    if skip_shared:
        components: list[str] = []
    else:
        components = [c for c in COMPONENTS if c != "transformer"] + upscaler_components
    split_info: dict = {
        "format": "split",
        "model_version": config["model_version"],
        "components": components,
        "transformer_variants": list(variant_adaln.keys()),
        "lora": lora_synced,
        "source": "Lightricks/LTX-2.3",
        "notes": {
            "vocoder": "Also contains BWE (bandwidth extension) generator weights"
            " — upsample layers [6,5,2,2,2] (240x) and mel_stft parameters.",
        },
    }
    if skip_shared:
        split_info["delta"] = True
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)
```

- [ ] **Step 4: Run tests to verify passing**

Run: `uv run pytest tests/test_ltx23_delta.py -v`
Expected: PASS (5/5 — 2 from Task 1 + 3 from this task).

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/mlx_forge/recipes/ltx_23.py tests/test_ltx23_delta.py`
Expected: All checks passed.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_forge/recipes/ltx_23.py tests/test_ltx23_delta.py
git commit -m "feat(ltx-2.3): honor --skip-shared in convert flow + emit delta in split_model.json"
```

---

## Task 3: Validate auto-detects delta mode

**Files:**
- Modify: `src/mlx_forge/recipes/ltx_23.py:958` (function `validate`)
- Test: `tests/test_ltx23_delta.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ltx23_delta.py`:

```python
class TestValidateDeltaMode:
    def _write_minimal_split(self, dir: Path, *, delta: bool, variants: list[str]) -> None:
        split = {
            "format": "split",
            "model_version": "2.3.0",
            "components": [],
            "transformer_variants": variants,
            "lora": [],
            "source": "Lightricks/LTX-2.3",
        }
        if delta:
            split["delta"] = True
        (dir / "split_model.json").write_text(json.dumps(split))

    def _write_minimal_config(self, dir: Path) -> None:
        cfg = {
            "model_version": "2.3.0",
            "is_v2": True,
            "apply_gated_attention": True,
            "caption_channels": None,
            "num_layers": 48,
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "connector_positional_embedding_max_pos": [4096],
            "connector_rope_type": "SPLIT",
            "variants": {"distilled-1.1": {"cross_attention_adaln": True}},
        }
        (dir / "config.json").write_text(json.dumps(cfg))

    def test_delta_skips_shared_checks(self, tmp_path, capsys):
        """validate with delta:true must not FAIL on missing shared components."""
        from mlx_forge.recipes import ltx_23

        self._write_minimal_split(tmp_path, delta=True, variants=["distilled-1.1"])
        self._write_minimal_config(tmp_path)

        # No shared component files. No transformer file either; just check
        # that the FAIL is about the transformer, not about connector/vae/etc.
        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "Delta mode" in out
        assert "connector.safetensors exists" not in out  # shared check skipped
        assert "vae_decoder.safetensors exists" not in out

    def test_normal_mode_still_strict(self, tmp_path, capsys):
        """validate without delta key must FAIL on missing shared components."""
        from mlx_forge.recipes import ltx_23

        self._write_minimal_split(tmp_path, delta=False, variants=["distilled-1.1"])
        self._write_minimal_config(tmp_path)

        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "Delta mode" not in out
        assert "connector.safetensors" in out  # shared check ran (and presumably failed)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_ltx23_delta.py::TestValidateDeltaMode -v`
Expected: FAIL — validate doesn't yet read the `delta` flag.

- [ ] **Step 3: Modify `validate()` in `src/mlx_forge/recipes/ltx_23.py`**

Locate `validate()` at line 958. Read `split_info` early in the function (it's likely already loaded). Add a `delta` extraction and gate the shared component checks. Find the section that does:

```python
for comp in [c for c in COMPONENTS if c != "transformer"]:
    validate_file_exists(model_dir, f"{comp}.safetensors", result)
```

(or analogous loop). Wrap it in `if not delta:` and add the announcement print at the top of the function:

```python
def validate(args) -> None:
    """Validate a converted LTX-2.3 model."""
    model_dir = Path(args.model_dir)
    print(f"Validating: {model_dir}")
    result = ValidationResult()

    split_info, config_data = load_model_metadata(model_dir)
    delta = bool(split_info.get("delta", False))
    if delta:
        print("\n[INFO] Delta mode (skipping shared component checks)")

    print("\n== File Structure ==")
    validate_file_exists(model_dir, "config.json", result)
    validate_file_exists(model_dir, "split_model.json", result)

    if not delta:
        for comp in [c for c in COMPONENTS if c != "transformer"]:
            validate_file_exists(model_dir, f"{comp}.safetensors", result)

    # ...rest of validate unchanged (transformer checks, etc.)
```

The exact wrap location may differ — find the loop that calls `validate_file_exists` for shared components and wrap it. Also wrap any later block that loads shared component weights for deeper checks (e.g., the `== Connector Weights ==`, `== VAE Decoder Weights ==`, `== Audio VAE Weights ==`, `== Vocoder Weights ==` sections) in `if not delta:`.

If `load_model_metadata` is not already imported, add: `from mlx_forge.upload import load_model_metadata`. (Or read split_model.json directly — your call. Prefer the existing helper if available.)

- [ ] **Step 4: Run tests to verify passing**

Run: `uv run pytest tests/test_ltx23_delta.py -v`
Expected: All PASS.

- [ ] **Step 5: Run ruff + format**

Run: `uv run ruff check src/mlx_forge/recipes/ltx_23.py && uv run ruff format src/mlx_forge/recipes/ltx_23.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_forge/recipes/ltx_23.py tests/test_ltx23_delta.py
git commit -m "feat(ltx-2.3): validate auto-detects delta mode (skips shared checks)"
```

---

## Task 4: Add `--add-only` flag (CLI wiring + skeleton)

**Files:**
- Modify: `src/mlx_forge/cli.py` (upload subparser)
- Modify: `src/mlx_forge/upload.py` (function `upload_model` signature)
- Test: `tests/test_upload.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_upload.py`:

```python
class TestAddOnlyArgparse:
    def test_default_is_false(self):
        from mlx_forge.cli import build_parser  # may already exist; otherwise inline

        parser = build_parser()
        args = parser.parse_args(["upload", "models/foo"])
        assert args.add_only is False

    def test_flag_sets_true(self):
        from mlx_forge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["upload", "models/foo", "--add-only"])
        assert args.add_only is True
```

If `cli.py` doesn't expose a `build_parser()` function, refactor: extract the parser construction into `def build_parser() -> argparse.ArgumentParser` and call it from `main()`. Keep behavior identical.

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_upload.py::TestAddOnlyArgparse -v`
Expected: FAIL — `--add-only` unknown argument.

- [ ] **Step 3: Refactor `cli.py` to expose `build_parser()` and add `--add-only`**

In `src/mlx_forge/cli.py`, locate the upload subparser. Add:

```python
upload_parser.add_argument(
    "--add-only",
    action="store_true",
    help=(
        "Delta upload: skip files whose names already exist on the remote repo. "
        "Useful after a `convert --skip-shared` to push only the new variant. "
        "Refuses to run if the repo doesn't exist."
    ),
)
```

Also extend `upload_model` signature in `src/mlx_forge/upload.py` to accept `add_only: bool = False` (no behavior change yet — just thread it through).

- [ ] **Step 4: Run tests to verify passing**

Run: `uv run pytest tests/test_upload.py::TestAddOnlyArgparse -v`
Expected: PASS.

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/mlx_forge/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_forge/cli.py src/mlx_forge/upload.py tests/test_upload.py
git commit -m "feat(upload): wire --add-only flag through CLI (no-op behavior)"
```

---

## Task 5: Implement `--add-only` filtering in `upload_model`

**Files:**
- Modify: `src/mlx_forge/upload.py` (function `upload_model`)
- Test: `tests/test_upload.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_upload.py`:

```python
import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from unittest.mock import MagicMock, call


class TestAddOnlyBehavior:
    def _setup_dir(self, tmp_path):
        # Two safetensors files in the local model dir
        (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors").write_bytes(b"y" * 50)
        (tmp_path / "vae_decoder.safetensors").write_bytes(b"z" * 200)
        (tmp_path / "split_model.json").write_text("{}")

    def _make_api(self, remote_files: list[str], repo_exists: bool = True) -> MagicMock:
        api = MagicMock()
        if repo_exists:
            info = MagicMock()
            info.siblings = [MagicMock(rfilename=f) for f in remote_files]
            api.model_info.return_value = info
        else:
            api.model_info.side_effect = RepositoryNotFoundError("not found")
        api.create_repo.return_value = "https://huggingface.co/test/repo"
        return api

    def test_uploads_only_new_files(self, tmp_path):
        from mlx_forge.upload import upload_model

        self._setup_dir(tmp_path)
        # Remote has vae_decoder already; transformer + lora are new
        api = self._make_api(
            remote_files=["vae_decoder.safetensors", "config.json"],
            repo_exists=True,
        )

        upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)

        uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert "transformer-distilled-1.1.safetensors" in uploaded
        assert "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" in uploaded
        assert "vae_decoder.safetensors" not in uploaded

    def test_refuses_when_repo_not_found(self, tmp_path, capsys):
        from mlx_forge.upload import upload_model

        self._setup_dir(tmp_path)
        api = self._make_api(remote_files=[], repo_exists=False)

        with pytest.raises(SystemExit):
            upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)
        api.upload_file.assert_not_called()

    def test_nothing_to_upload_exits_cleanly(self, tmp_path, capsys):
        from mlx_forge.upload import upload_model

        self._setup_dir(tmp_path)
        api = self._make_api(
            remote_files=[
                "transformer-distilled-1.1.safetensors",
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                "vae_decoder.safetensors",
                "split_model.json",
            ],
            repo_exists=True,
        )

        upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)

        api.upload_file.assert_not_called()
        out = capsys.readouterr().out
        assert "Nothing to upload" in out
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_upload.py::TestAddOnlyBehavior -v`
Expected: 3 FAILs.

- [ ] **Step 3: Implement `--add-only` branch in `upload_model`**

In `src/mlx_forge/upload.py`, modify `upload_model`. After the existing repo creation block, add:

```python
def upload_model(
    model_dir: Path,
    *,
    api: HfApi,
    repo_id: str,
    commit_message: str = "Upload MLX model via mlx-forge",
    private: bool = False,
    collection_title: str | None = None,
    card_only: bool = False,
    add_only: bool = False,
) -> str:
    if add_only:
        # Verify repo exists; refuse otherwise
        try:
            info = api.model_info(repo_id)
        except RepositoryNotFoundError:
            print(
                f"ERROR: --add-only refuses to run on non-existent repo '{repo_id}'. "
                "Use a normal upload to create the repo first."
            )
            raise SystemExit(1)
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            print(f"ERROR: Could not query repo '{repo_id}': {e}")
            raise SystemExit(1)

        remote = {s.rfilename for s in info.siblings}
        candidates = sorted(
            p for p in model_dir.iterdir()
            if p.is_file() and p.suffix in (".safetensors", ".json")
        )
        new_files = [p for p in candidates if p.name not in remote]

        if not new_files:
            print(f"Nothing to upload (all {len(candidates)} files already on remote)")
            return f"https://huggingface.co/{repo_id}"

        skipped = [p.name for p in candidates if p.name in remote]
        if skipped:
            print(f"Skipped (on remote): {', '.join(skipped)}")

        for p in new_files:
            msg = f"{commit_message}: {p.name}" if len(new_files) > 1 else commit_message
            print(f"Uploading: {p.name}")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=repo_id,
                commit_message=msg,
            )
        return f"https://huggingface.co/{repo_id}"

    # ...existing non-add-only branch unchanged...
```

Add the import at the top of `upload.py`:

```python
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
```

(`HfHubHTTPError` already imported; just add `RepositoryNotFoundError`.)

Also wire `add_only` from CLI args in `cli.py` upload handler (pass `add_only=args.add_only` into `upload_model`).

- [ ] **Step 4: Run tests to verify passing**

Run: `uv run pytest tests/test_upload.py::TestAddOnlyBehavior -v`
Expected: 3 PASS.

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/mlx_forge/upload.py tests/test_upload.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_forge/upload.py src/mlx_forge/cli.py tests/test_upload.py
git commit -m "feat(upload): implement --add-only (skip files present on remote)"
```

---

## Task 6: Jinja2 model card template + refactor `generate_model_card`

**Files:**
- Create: `src/mlx_forge/templates/__init__.py`
- Create: `src/mlx_forge/templates/model-card.md.j2`
- Modify: `src/mlx_forge/upload.py` (function `generate_model_card`)
- Test: `tests/test_upload.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_upload.py`:

```python
class TestModelCardTemplate:
    def test_renders_minimal_card(self):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            Path("/tmp/dummy"),
            split_info={"source": "Org/Model", "transformer_variants": ["dev"]},
            config={"model_version": "2.3.0"},
            repo_id="user/model-mlx",
        )
        assert "library_name: mlx" in card
        assert "user/model-mlx" in card
        assert "[Org/Model](https://huggingface.co/Org/Model)" in card
        assert "Transformer variants:" in card
        assert "dev" in card

    def test_renders_quantized(self):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            Path("/tmp/dummy"),
            split_info={
                "source": "Org/Model",
                "transformer_variants": ["dev"],
                "quantized": True,
                "quantization_bits": 8,
            },
            config={"model_version": "2.3.0"},
            repo_id="user/model-mlx-q8",
        )
        assert "Quantization:" in card
        assert "int8" in card

    def test_omits_optional_sections(self):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            Path("/tmp/dummy"),
            split_info={"source": "Org/Model"},
            config={},
            repo_id="user/model-mlx",
        )
        # No transformer_variants → no Transformer variants line
        assert "Transformer variants:" not in card
        # No usage_url and no cli_snippet → no Usage section
        assert "## Usage" not in card
        # No links → no Related Projects section
        assert "## Related Projects" not in card
```

- [ ] **Step 2: Run tests to verify they pass against current implementation**

Run: `uv run pytest tests/test_upload.py::TestModelCardTemplate -v`
Expected: PASS (current Python builder produces equivalent output). If any fail because the current builder differs textually, treat that as the snapshot we need to preserve in the template.

If a test FAILs against the current implementation, adjust the assertion to match what `generate_model_card` produces today (snapshot). Then proceed.

- [ ] **Step 3: Create the template file**

Create `src/mlx_forge/templates/__init__.py` (empty file — makes the dir a package resource).

Create `src/mlx_forge/templates/model-card.md.j2`:

```jinja
---
library_name: mlx
license: {{ license_id }}
{%- if base_model %}
base_model: {{ base_model }}
{%- endif %}
tags:
  - mlx
  - mlx-forge
  - apple-silicon
  - safetensors
---

# {{ repo_id }}

{% if base_model -%}
MLX format conversion of [{{ base_model }}](https://huggingface.co/{{ base_model }}).
{%- else -%}
MLX format model.
{%- endif %}

Converted with [mlx-forge](https://github.com/dgrauet/mlx-forge).

{% if transformer_variants -%}
- **Transformer variants:** {{ transformer_variants | join(", ") }}
{% endif -%}
{% if model_version -%}
- **Model version:** {{ model_version }}
{% endif -%}
{% if quantized and bits -%}
- **Quantization:** int{{ bits }}
{% endif %}
{% if usage_url or cli_snippet %}
## Usage

{% if usage_url -%}
These weights can be used with [{{ usage_url.rstrip("/").split("/")[-1] }}]({{ usage_url }}).
{% endif %}
{% if cli_snippet -%}
```bash
{{ cli_snippet.rstrip() }}
```
{% endif %}
{% endif %}
{% if links %}
## Related Projects

{% for link in links -%}
{% if ": " in link -%}
- **{{ link.split(": ", 1)[0] }}:** {{ link.split(": ", 1)[1] }}
{% else -%}
- {{ link }}
{% endif -%}
{% endfor %}
{% endif %}
{% if model_files %}
## Files

{% for f in model_files -%}
- `{{ f.name }}` ({{ f.size_str }})
{% endfor %}
{% endif %}
```

- [ ] **Step 4: Refactor `generate_model_card` to render the template**

In `src/mlx_forge/upload.py`, replace the body of `generate_model_card` with:

```python
def generate_model_card(
    model_dir: Path,
    *,
    split_info: dict,
    config: dict,
    repo_id: str,
    base_model: str | None = None,
    license_id: str = "other",
    usage_url: str | None = None,
    links: list[str] | None = None,
    cli_snippet: str | None = None,
    transformer_variants: list[str] | None = None,
    lora_files: list[str] | None = None,
) -> str:
    """Render the Jinja2 model card template."""
    from importlib.resources import files
    from jinja2 import Environment

    source = split_info.get("source", "")
    if base_model is None:
        base_model = source or None
    if transformer_variants is None:
        transformer_variants = split_info.get("transformer_variants", []) or []

    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits")
    model_version = config.get("model_version")

    # Build file listing from local dir
    model_files = []
    for p in sorted(model_dir.iterdir() if model_dir.exists() else []):
        if p.is_file() and p.suffix in (".safetensors", ".json"):
            model_files.append(
                type("F", (), {
                    "name": p.name,
                    "size_str": format_bytes(p.stat().st_size),
                })()
            )

    template_text = (
        files("mlx_forge.templates").joinpath("model-card.md.j2").read_text()
    )
    env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    template = env.from_string(template_text)

    return template.render(
        repo_id=repo_id,
        base_model=base_model,
        license_id=license_id,
        transformer_variants=transformer_variants,
        lora_files=lora_files or [],
        model_version=model_version,
        quantized=quantized,
        bits=bits,
        usage_url=usage_url,
        cli_snippet=cli_snippet,
        links=links or [],
        model_files=model_files,
    )
```

Verify `pyproject.toml` has `package_data` / `tool.setuptools.package-data` configured to include the `.j2` file. If using `hatchling`/`uv`, add to `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/mlx_forge"]
include = ["src/mlx_forge/templates/*.j2"]
```

(Adapt syntax to actual build backend — check existing `pyproject.toml`.)

- [ ] **Step 5: Run all upload tests**

Run: `uv run pytest tests/test_upload.py -v`
Expected: All PASS, including pre-existing card-related tests (which should still produce equivalent output).

If pre-existing tests fail because of small textual differences (e.g., trailing newlines, whitespace), tweak the template until output matches. The goal is byte-equivalence with the old builder for canonical inputs.

- [ ] **Step 6: Run ruff**

Run: `uv run ruff check src/mlx_forge/upload.py tests/test_upload.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_forge/templates/ src/mlx_forge/upload.py tests/test_upload.py pyproject.toml
git commit -m "refactor(upload): replace generate_model_card body with Jinja2 template"
```

---

## Task 7: `--card-only` derives variants from remote

**Files:**
- Modify: `src/mlx_forge/upload.py` (function `upload_model`, `card_only` branch)
- Test: `tests/test_upload.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_upload.py`:

```python
class TestCardOnlyRemoteRefresh:
    def test_card_only_uses_remote_variants(self, tmp_path):
        from mlx_forge.upload import upload_model

        # Local dir has only one variant (delta convert leftover)
        (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"x")
        (tmp_path / "split_model.json").write_text(
            json.dumps({"source": "Lightricks/LTX-2.3", "transformer_variants": ["distilled-1.1"]})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_version": "2.3.0"}))

        # Remote has all three transformer variants
        api = MagicMock()
        info = MagicMock()
        info.siblings = [
            MagicMock(rfilename="transformer-distilled.safetensors"),
            MagicMock(rfilename="transformer-dev.safetensors"),
            MagicMock(rfilename="transformer-distilled-1.1.safetensors"),
            MagicMock(rfilename="ltx-2.3-22b-distilled-lora-384.safetensors"),
            MagicMock(rfilename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
            MagicMock(rfilename="config.json"),
        ]
        api.model_info.return_value = info
        api.create_repo.return_value = "https://huggingface.co/test/repo"

        upload_model(
            tmp_path, api=api, repo_id="test/repo", card_only=True
        )

        # The README uploaded must mention all three variants
        readme_call = next(
            c for c in api.upload_file.call_args_list
            if c.kwargs["path_in_repo"] == "README.md"
        )
        readme_path = readme_call.kwargs["path_or_fileobj"]
        readme_text = Path(readme_path).read_text()
        assert "distilled" in readme_text
        assert "dev" in readme_text
        assert "distilled-1.1" in readme_text
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_upload.py::TestCardOnlyRemoteRefresh -v`
Expected: FAIL — README only mentions local variants.

- [ ] **Step 3: Augment `card_only` branch in `upload_model`**

In `src/mlx_forge/upload.py`, the existing `card_only` branch reads README.md from disk. Add a step that derives variants from remote and regenerates README before upload:

```python
    if card_only:
        # Derive transformer variants and LoRA files from remote
        try:
            info = api.model_info(repo_id)
            remote_files = [s.rfilename for s in info.siblings]
        except (HfHubHTTPError, OSError, ConnectionError):
            remote_files = []  # fall through with local data only

        if remote_files:
            transformer_variants = sorted(
                f.removeprefix("transformer-").removesuffix(".safetensors")
                for f in remote_files
                if f.startswith("transformer-") and f.endswith(".safetensors")
            )
            lora_files = sorted(
                f for f in remote_files
                if "lora" in f and f.endswith(".safetensors")
            )
            print(f"Detected variants on remote: {', '.join(transformer_variants) or '(none)'}")
            print(f"Detected LoRAs on remote: {', '.join(lora_files) or '(none)'}")
        else:
            transformer_variants = None  # generate_model_card falls back to split_info
            lora_files = None

        split_info, config_data = load_model_metadata(model_dir)
        readme_text = generate_model_card(
            model_dir,
            split_info=split_info,
            config=config_data,
            repo_id=repo_id,
            transformer_variants=transformer_variants,
            lora_files=lora_files,
        )
        readme_path = model_dir / "README.md"
        readme_path.write_text(readme_text)

        print(f"Uploading {readme_path.name} -> {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=commit_message,
        )
        return f"https://huggingface.co/{repo_id}"
```

The existing `--card-only` code path may already do part of this (load metadata, generate card). The change is: insert the remote-info fetch step and pass `transformer_variants`/`lora_files` to `generate_model_card`.

- [ ] **Step 4: Run all upload tests**

Run: `uv run pytest tests/test_upload.py -v`
Expected: All PASS.

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check src/mlx_forge/upload.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_forge/upload.py tests/test_upload.py
git commit -m "feat(upload): --card-only derives variants from remote repo"
```

---

## Task 8: End-to-end integration test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

Append to `tests/test_integration.py`:

```python
class TestDeltaWorkflowEndToEnd:
    """Wire test: convert delta → validate → upload --add-only → upload --card-only."""

    def test_delta_workflow_glue(self, tmp_path, capsys, monkeypatch):
        import json
        from unittest.mock import MagicMock, patch
        from mlx_forge.recipes import ltx_23
        from mlx_forge.upload import upload_model

        # 1. Mock convert: writes split_model.json with delta=true and a fake transformer file
        (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "split_model.json").write_text(json.dumps({
            "format": "split",
            "model_version": "2.3.0",
            "components": [],
            "transformer_variants": ["distilled-1.1"],
            "lora": [],
            "source": "Lightricks/LTX-2.3",
            "delta": True,
        }))
        (tmp_path / "config.json").write_text(json.dumps({
            "model_version": "2.3.0",
            "is_v2": True,
            "apply_gated_attention": True,
            "caption_channels": None,
            "num_layers": 48,
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "connector_positional_embedding_max_pos": [4096],
            "connector_rope_type": "SPLIT",
            "variants": {"distilled-1.1": {"cross_attention_adaln": True}},
        }))

        # 2. Validate auto-detects delta and skips shared checks
        import argparse
        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "Delta mode" in out

        # 3. upload --add-only with mocked api: only new file gets uploaded
        api = MagicMock()
        info = MagicMock()
        info.siblings = [
            MagicMock(rfilename="config.json"),
            MagicMock(rfilename="transformer-distilled.safetensors"),
        ]
        api.model_info.return_value = info

        upload_model(tmp_path, api=api, repo_id="user/repo", add_only=True)

        uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert "transformer-distilled-1.1.safetensors" in uploaded
        assert "config.json" not in uploaded  # already on remote

        # 4. upload --card-only refreshes card with remote-derived variants
        api2 = MagicMock()
        info2 = MagicMock()
        info2.siblings = [
            MagicMock(rfilename="transformer-distilled.safetensors"),
            MagicMock(rfilename="transformer-distilled-1.1.safetensors"),
        ]
        api2.model_info.return_value = info2

        upload_model(tmp_path, api=api2, repo_id="user/repo", card_only=True)

        readme_call = next(
            c for c in api2.upload_file.call_args_list
            if c.kwargs["path_in_repo"] == "README.md"
        )
        readme_text = Path(readme_call.kwargs["path_or_fileobj"]).read_text()
        assert "distilled" in readme_text
        assert "distilled-1.1" in readme_text
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_integration.py::TestDeltaWorkflowEndToEnd -v`
Expected: PASS (if Tasks 1-7 are correct).

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: ALL PASS.

- [ ] **Step 4: Run ruff**

Run: `uv run ruff check tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end delta workflow integration test"
```

---

## Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/skills/mlx-recipe/SKILL.md`

- [ ] **Step 1: Add "Delta workflow" section to `CLAUDE.md`**

Append after the existing "Conventions" section in `CLAUDE.md`:

```markdown
## Delta workflow (adding a variant to an existing repo)

When upstream publishes a new transformer variant or LoRA for a model that's
already converted and uploaded, use the delta workflow instead of regenerating
the full model:

1. **Convert delta** — only the new transformer + LoRAs:
   ```bash
   mlx-forge convert <recipe> --variant <new> --skip-shared --output models/<name>-delta
   ```
   Skips connector, vae_*, audio_vae, vocoder, vae_shared_stats, and upscalers.
   Writes `split_model.json` with `"delta": true`.

2. **Validate** — auto-detects delta mode:
   ```bash
   mlx-forge validate <recipe> models/<name>-delta
   ```
   Logs `[INFO] Delta mode (skipping shared component checks)` and verifies
   only the components present.

3. **Upload delta** — skip files already on remote:
   ```bash
   mlx-forge upload models/<name>-delta --repo-id <user/repo> --add-only
   ```
   Refuses if the repo doesn't exist (use a normal upload first to create it).
   Each new file gets its own commit (more resilient against transient HF
   upload hangs we've observed).

4. **Refresh card** — derive variants from remote, regenerate README:
   ```bash
   mlx-forge upload models/<name>-delta --repo-id <user/repo> --card-only
   ```
   Idempotent. Re-running always produces a card matching the current remote
   state, regardless of what the local model_dir contains.

Currently the only recipe that supports `--skip-shared` is `ltx-2.3`. Other
recipes can opt in by mirroring the LTX-2.3 implementation pattern (see
`src/mlx_forge/recipes/ltx_23.py`, search for `skip_shared`).
```

- [ ] **Step 2: Add "Model card template" section to `.claude/skills/mlx-recipe/SKILL.md`**

Append before the "## Phase 5: Verification" section (or wherever fits the existing flow):

```markdown
## Model card template

The HuggingFace model card uploaded with `mlx-forge upload` is rendered from a
single Jinja2 template:

**Path:** `src/mlx_forge/templates/model-card.md.j2`

**Available variables (rendered context):**

| Variable | Type | Source |
|----------|------|--------|
| `repo_id` | str | Derived or explicit (`--repo-id`) |
| `base_model` | str \| None | `split_info["source"]` or explicit (`--base-model`) |
| `license_id` | str | `--license` (default: `other`) |
| `transformer_variants` | list[str] | Remote-derived in `--card-only` mode, else `split_info` |
| `lora_files` | list[str] | Remote-derived in `--card-only` mode, else `[]` |
| `model_version` | str \| None | `config["model_version"]` |
| `quantized` | bool | `split_info["quantized"]` |
| `bits` | int \| None | `split_info["quantization_bits"]` |
| `usage_url` | str \| None | `--usage-url` |
| `cli_snippet` | str \| None | `--cli-snippet` |
| `links` | list[str] | `--link` (repeatable, "Label: URL") |
| `model_files` | list of `{name, size_str}` | Listed from local model_dir |

**Conditional sections** (`{% if %}`): frontmatter `base_model`, transformer
variants line, model version line, quantization line, Usage section, Related
Projects section, Files section.

**Customizing:** edit the template file directly. No per-recipe override
exists in v1 (deferred — `docs/superpowers/specs/2026-04-25-delta-workflow-improvements-design.md`).
Verify changes by running the upload tests:

```bash
uv run pytest tests/test_upload.py::TestModelCardTemplate -v
```
```

- [ ] **Step 3: Verify nothing else needs updating**

Run: `git status` — only `CLAUDE.md` and `.claude/skills/mlx-recipe/SKILL.md` should be modified.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md .claude/skills/mlx-recipe/SKILL.md
git commit -m "docs: delta workflow (CLAUDE.md) + model card template (mlx-recipe skill)"
```

---

## Self-Review Checklist (run before considering plan done)

- [ ] All 4 spec components covered: `--skip-shared` (Tasks 1-2), validate delta (Task 3), `--add-only` (Tasks 4-5), Jinja2 + remote-aware card (Tasks 6-7).
- [ ] All error-handling cases from spec table addressed: tests cover non-existent repo refusal, all-files-present exit, delta + missing transformer FAIL.
- [ ] Documentation tasks present: CLAUDE.md (delta workflow), mlx-recipe SKILL.md (template internals).
- [ ] No "TBD" / "implement later" / placeholder steps. Code blocks present where steps modify code.
- [ ] Type/name consistency: `add_only` (kwarg) and `--add-only` (CLI flag), `skip_shared` (attr) and `--skip-shared` (CLI flag), `transformer_variants` and `lora_files` consistent throughout.
- [ ] Frequent commits: one commit per task, never batched.

---

## Out of Scope (per spec)

- Generalizing `--skip-shared` to non-LTX-2.3 recipes.
- Per-recipe model card template overrides.
- Retry/backoff on upload failures.
- SHA-based dedup in `--add-only`.
