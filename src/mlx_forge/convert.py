"""Generic conversion utilities shared across recipes.

Provides common helpers for downloading, loading, processing, and saving
model components during PyTorch-to-MLX conversion.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx
from huggingface_hub import hf_hub_download

# Enable high-performance mode for hf-xet (saturates network bandwidth)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from tqdm import tqdm

from .quantize import _materialize, quantize_weights


def add_common_convert_args(
    parser,
    *,
    output_default: str,
    quantize_help: str,
    bits_default: int = 8,
    dry_run_help: str = "Preview conversion plan without writing anything",
) -> None:
    """Register the --output/--quantize/--bits/--group-size/--dry-run block.

    Every recipe takes these five arguments; only the wording and the default
    bit-width differ. Call this at the point where `--output` belongs so the
    order in `--help` is preserved, then add recipe-specific arguments.

    Args:
        parser: Parser to register on.
        output_default: Default output path, as shown in help
            (e.g. "./models/ltx-2.3-mlx[-q<bits>]").
        quantize_help: What --quantize acts on for this recipe.
        bits_default: Default bit-width (8 everywhere except ernie-image-pe).
        dry_run_help: Override when the recipe's dry run differs (e.g. recipes
            that would otherwise download).
    """
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory (default: {output_default})",
    )
    parser.add_argument("--quantize", action="store_true", help=quantize_help)
    parser.add_argument(
        "--bits",
        type=int,
        default=bits_default,
        choices=[4, 8],
        help=f"Quantization bits (default: {bits_default})",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument("--dry-run", action="store_true", help=dry_run_help)


def load_torch_state_dict(
    path: str | Path,
    *,
    label: str | None = None,
    mmap: bool = False,
    weights_only: bool = True,
) -> dict:
    """Load a PyTorch checkpoint, failing with an install hint if torch is absent.

    torch is an optional dependency — only recipes whose upstream ships .pt/.pth
    /.ckpt files need it — so the import is deferred to call time.

    Args:
        path: Checkpoint file to load.
        label: Name used in the progress line (defaults to the filename).
        mmap: Memory-map the checkpoint instead of reading it into RAM.
        weights_only: Keep True unless the checkpoint stores non-tensor objects
            that must be unpickled. False executes arbitrary code from the
            file — only pass it for a checkpoint you trust (vjepa-2.1 stores
            its encoder under a pickled wrapper).

    Returns:
        The raw object torch.load returned — usually a state dict, sometimes a
        wrapper the caller must unwrap.
    """
    try:
        import torch  # ty: ignore[unresolved-import]
    except ImportError:
        raise SystemExit(
            "ERROR: PyTorch is required to read this checkpoint.\nInstall it with: uv add torch"
        )

    path = Path(path)
    print(f"  Loading {label or path.name} (torch.load)...")
    return torch.load(str(path), map_location="cpu", mmap=mmap, weights_only=weights_only)


SPLIT_MODEL_FILENAME = "split_model.json"


def default_output_dir(model_name: str, *, quantize: bool, bits: int) -> Path:
    """`models/<model_name>-mlx[-q<bits>]` — the house naming convention.

    upload.derive_repo_id() parses this suffix back out to recover the bit
    width for recipes that record it nowhere else, so the shape matters.
    """
    suffix = f"-q{bits}" if quantize else ""
    return Path("models") / f"{model_name}-mlx{suffix}"


def write_split_model(output_dir: Path, info: dict) -> Path:
    """Write split_model.json.

    Centralises the filename only. The CONTENT stays per-recipe on purpose:
    upload.py and the model card read `source`, `quantized`,
    `quantization_bits` and `transformer_variants` out of it, and the recipes
    genuinely disagree on the rest (vjepa-2.0 writes a flat
    {component: filename} table, vjepa-2.1 a {model_name, components,
    quantized} record). Imposing one schema would rewrite published metadata,
    which is a behaviour change, not a refactor.
    """
    # Before the dump, not after: ensure_license_file records where the licence
    # copy came from into `info`, and that provenance belongs in the same write.
    # Best effort here so a licence server hiccup cannot destroy a conversion
    # that took twenty minutes. The obligation binds on distribution, so the
    # blocking check lives in the upload path, which calls this strictly.
    ensure_license_file(output_dir, info, strict=False)

    path = output_dir / SPLIT_MODEL_FILENAME
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return path


def _sha256(path: Path) -> str:
    """Content hash of a licence copy, which is what "verbatim" is checked on."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_license_source(spec: str | None) -> tuple[str, str] | None:
    """Split a "github:<owner>/<repo>/<path>" declaration, or None if it is not one.

    Returns None rather than raising for a plain Hub repo id, so the ten
    recipes that declare no source keep the existing code path untouched.

    Raises:
        ValueError: The spec claims the github scheme but names no file.
    """
    if not spec or not spec.startswith("github:"):
        return None
    parts = spec[len("github:") :].split("/")
    if len(parts) < 3 or not all(parts):
        raise ValueError(
            f"malformed licence source {spec!r}: expected github:<owner>/<repo>/<path>"
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def fetch_github_license(repo: str, path: str) -> tuple[bytes, str | None]:
    """The raw bytes of `path` on `repo`'s default branch, and the commit it came from.

    The commit is resolved separately and best-effort: provenance is better
    partial than absent, and a rate-limited API must not fail a conversion that
    already has the bytes it needs.
    """
    import json as _json
    import urllib.error
    import urllib.request

    raw = f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"
    with urllib.request.urlopen(raw) as response:
        content = response.read()

    revision: str | None = None
    api = f"https://api.github.com/repos/{repo}/commits?path={path}&per_page=1"
    try:
        with urllib.request.urlopen(api) as response:
            commits = _json.loads(response.read())
        if isinstance(commits, list) and commits and isinstance(commits[0].get("sha"), str):
            revision = commits[0]["sha"]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass

    return content, revision


def ensure_license_file(output_dir: Path, info: dict, *, strict: bool = True) -> list[Path]:
    """Place the upstream licence text next to the weights, and vouch for it.

    Converting and quantising a model produces a derivative, and the community
    licences these models ship under oblige whoever distributes one to hand the
    recipient a copy of the agreement — a `license_link` in the card front-matter
    does not discharge that. The recipe declares `license_file`; this fetches it
    verbatim from the upstream repo. Never rewrite or summarise the text: a
    paraphrase is not "a copy of this Agreement".

    A copy is only worth shipping if we can say where it came from, so each one
    is recorded in the manifest under `license_provenance`: the upstream repo,
    its revision, and the content hash. That record is what makes this
    idempotent rather than merely repeatable — a second run rehashes the local
    file and stops, without a network call, and a file that does not match what
    was recorded is reported instead of published. Before it existed, any file
    sitting at the right name was accepted unconditionally: a licence dropped in
    by hand, or left over from an upstream that has since revised its terms,
    shipped under a card claiming it was a verbatim copy.

    Called from `write_split_model`, which every recipe goes through, so no
    recipe can forget it, and again from the upload path where `strict` applies.

    Args:
        output_dir: The converted model directory.
        info: split_model.json contents; `license_file` names the path(s) inside
            the upstream repo, `base_model`/`source` identify that repo. Updated
            in place with `license_provenance`, which the caller persists — this
            runs before the manifest is written, so the record lands in it.
        strict: Abort on a mismatch or a failed fetch rather than warning. True
            when publishing, which is where the obligation binds.

    Returns:
        The local paths, empty when nothing is declared or when a fetch or a
        check failed in non-strict mode. Each lands at the repo root under its
        basename, which is what the card links to.
    """
    from .metadata import hub_repo_from_source, license_files

    declared = license_files(info.get("license_file"))
    if not declared:
        return []

    def refuse(message: str) -> list[Path]:
        if strict:
            raise SystemExit(
                f"ERROR: {message}\nThe licence obliges us to pass a copy on to "
                "whoever receives these weights; refusing to publish an "
                "unverified one."
            )
        print(f"  WARNING: {message}")
        return []

    try:
        github = parse_license_source(info.get("license_source"))
    except ValueError as e:
        return refuse(str(e))

    if github:
        upstream = f"github:{github[0]}"
    else:
        upstream = info.get("base_model") or hub_repo_from_source(info.get("source"))
    provenance = dict(info.get("license_provenance") or {})

    for filename in declared:
        name = Path(filename).name
        local = output_dir / name
        recorded = provenance.get(name) or {}

        if local.exists() and local.stat().st_size:
            digest = _sha256(local)
            if recorded.get("sha256") == digest:
                continue  # vouched for already: no network, no rewrite
            if recorded.get("sha256"):
                return refuse(
                    f"{local} does not match the copy recorded in the manifest "
                    f"(expected sha256 {recorded['sha256'][:16]}, found {digest[:16]}); "
                    "it was replaced from an undocumented source"
                )
            # No record yet: an older pack. Vouch for it against upstream rather
            # than trusting a file whose origin nobody wrote down.
        elif not upstream:
            return refuse(
                f"{output_dir} declares license_file={list(declared)} but names no "
                "upstream Hub repo to fetch it from (set base_model in the recipe)"
            )

        if not upstream:
            return refuse(
                f"{local} has no recorded provenance and no upstream repo to check "
                "it against (set base_model in the recipe)"
            )

        if github:
            repo, path = github
            if Path(path).name != name:
                # license_source names exactly one GitHub file. A declared
                # license_file whose basename differs from that file (the
                # multi-file case docs/recipe-anatomy.md documents for
                # Hunyuan3D's ("LICENSE", "Notice.txt")) would otherwise fetch
                # the same path again and write its bytes under the wrong
                # name — a silently wrong licence file recorded as verified.
                # Refuse loudly instead; see RecipeMetadata.license_source.
                return refuse(
                    f"license_source names {path}, which does not match the "
                    f"declared license_file entry {name!r}; a GitHub "
                    "license_source can only describe a single file"
                )
            try:
                content, revision = fetch_github_license(repo, path)
            except (OSError, ValueError) as e:
                return refuse(f"could not fetch {path} from github.com/{repo}: {e}")
            cached = output_dir / f".{name}.fetched"
            cached.write_bytes(content)
        else:
            try:
                cached = Path(hf_hub_download(repo_id=upstream, filename=filename))
                revision = _upstream_revision(upstream)
            except (HfHubHTTPError, OSError, ConnectionError, ValueError) as e:
                return refuse(f"could not fetch {filename} from {upstream}: {e}")

        try:
            digest = _sha256(cached)
            if local.exists() and local.stat().st_size and _sha256(local) != digest:
                return refuse(
                    f"{local} differs from {filename} in {upstream}. Either it came from "
                    "somewhere undocumented, or upstream has revised its terms since this "
                    "pack was built; both need a decision, not a silent overwrite"
                )

            if not local.exists() or not local.stat().st_size:
                shutil.copyfile(cached, local)
                print(f"  Licence: {filename} from {upstream} -> {name}")

            provenance[name] = {"repo": upstream, "revision": revision, "sha256": digest}
        finally:
            if github:
                # `cached` is a scratch copy for the GitHub path only (the Hub
                # path's `cached` is the shared hf_hub_download cache, which is
                # not ours to delete). Clean it up on every exit from this
                # block, including the mismatch refusal above.
                cached.unlink(missing_ok=True)

    if provenance != (info.get("license_provenance") or {}):
        info["license_provenance"] = provenance

    return [output_dir / Path(f).name for f in declared]


def _upstream_revision(repo_id: str) -> str | None:
    """The upstream repo revision a licence copy was taken at, if resolvable."""
    from huggingface_hub import HfApi

    try:
        revision = HfApi().model_info(repo_id).sha
    except Exception:  # noqa: BLE001 — provenance is better partial than absent
        return None
    # The manifest is JSON: anything that is not a plain string — a stubbed API
    # under test, a client that returns an object — must not reach the dump.
    return revision if isinstance(revision, str) else None


def print_output_summary(output_dir: Path, *, header: str | None = None) -> None:
    """List what a conversion produced, with sizes.

    Recurses: four recipes used iterdir() and missed files written into
    subdirectories (ideogram-4 writes tokenizer/ that way), three used rglob().
    """
    if header:
        print(f"\n{'=' * 60}")
        print(header)
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(output_dir)}: {size_mb:.1f} MB")


def fmt_size(mb: float) -> str:
    """Format a size in MB to a human-readable string."""
    if mb >= 1000:
        return f"{mb / 1000:.1f} GB"
    return f"{mb:.0f} MB"


def load_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load a safetensors file and return a typed dict of weight arrays.

    Wraps mx.load() (which returns a broad union type) with an explicit cast
    to dict[str, mx.array]. Safe to call only on .safetensors files.
    """
    return cast(dict[str, mx.array], mx.load(str(path)))


def _validate_path_within(filepath: Path, parent: Path) -> Path:
    """Ensure filepath is within parent directory (prevents path traversal).

    Uses lexical normalization (os.path.normpath) rather than symlink resolution
    so that HuggingFace cache layouts — which store shard files as symlinks to
    ../../blobs/ — pass the check while ``../`` traversal attacks are still caught.

    Threat-model note: this intentionally allows a symlink *inside* ``parent``
    that points outside ``parent`` (required for HF-cache ``--source``
    conversions). Do not "restore" ``Path.resolve()`` here to re-tighten the
    check — it re-breaks HF-cache conversions. The lexical check still rejects
    ``../`` traversal and absolute-path injection in the requested filepath.
    """
    norm = os.path.normpath(filepath)
    norm_parent = os.path.normpath(parent)
    sep = os.sep
    if not (str(norm).startswith(str(norm_parent) + sep) or norm == norm_parent):
        raise ValueError(f"Path traversal detected: '{filepath}' resolves outside '{parent}'")
    return filepath


def download_hf_files(
    repo_id: str,
    filenames: list[str],
    download_dir: Path,
) -> None:
    """Download files from HuggingFace Hub with error handling.

    Skips files already present in download_dir.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        target = download_dir / filename
        _validate_path_within(target, download_dir)
        if target.exists():
            print(f"  Already downloaded: {filename}")
            continue
        try:
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=download_dir,
            )
        except GatedRepoError:
            print(
                f"ERROR: '{repo_id}' is gated.\n"
                f"Accept the terms at https://huggingface.co/{repo_id} , then run: hf auth login"
            )
            raise SystemExit(1)
        except RepositoryNotFoundError:
            print(
                f"ERROR: Repository '{repo_id}' not found or access denied.\n"
                "If this is a gated repo, request access and run: huggingface-cli login"
            )
            raise SystemExit(1)
        except LocalEntryNotFoundError:
            print(
                f"ERROR: '{filename}' not in cache and network unavailable.\n"
                "Check your internet connection or download the file manually."
            )
            raise SystemExit(1)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 401:
                print("ERROR: Authentication required. Run: huggingface-cli login")
            elif status == 403:
                print(f"ERROR: Access denied to '{repo_id}'.")
            elif status == 404:
                print(f"ERROR: File '{filename}' not found in '{repo_id}'.")
            else:
                print(f"ERROR: HuggingFace Hub request failed: {e}")
            raise SystemExit(1)
        except (OSError, ConnectionError) as e:
            print(f"ERROR: Network error: {e}")
            raise SystemExit(1)


def copy_required_files(
    source_dir: Path,
    output_dir: Path,
    files: list[str],
    *,
    flatten: bool,
    optional: set[str] | None = None,
    keep_tree: set[str] | None = None,
) -> None:
    """Copy pipeline files, aborting loudly when a required one is missing.

    A silent `if src.exists()` skip shipped incomplete artifacts twice
    (cogvideox q8 without spiece.model, matrix-game-3.0-mlx without
    google/umt5-xxl/spiece.model): a recipe's output must be complete or
    the conversion must fail naming every missing file. Entries in
    `optional` only warn (e.g. chat_template.jinja, bypassed at runtime
    by the ernie-image port).

    flatten=True maps `a/b.ext` to `a_b.ext`; flatten=False preserves the
    source tree under output_dir. `keep_tree` lists top-level directories
    exempt from flattening — ideogram-4 keeps `tokenizer/` intact so
    AutoTokenizer loads it without path mapping, while flattening the rest.
    """
    optional = optional or set()
    keep_tree = keep_tree or set()
    missing = [f for f in files if not (source_dir / f).exists()]
    required_missing = [f for f in missing if f not in optional]
    if required_missing:
        raise SystemExit(
            "ERROR: required pipeline files missing from source: "
            + ", ".join(required_missing)
            + f" (looked in {source_dir})"
        )
    for f in missing:
        print(f"  WARNING: optional file missing, skipped: {f}")
    for f in files:
        if f in missing:
            continue
        src = source_dir / f
        top = f.split("/")[0] if "/" in f else ""
        if top and top in keep_tree:
            dest = output_dir / f
            dest.parent.mkdir(parents=True, exist_ok=True)
        elif flatten and top:
            dest = output_dir / f"{top}_{Path(f).name}"
        elif flatten:
            dest = output_dir / Path(f).name
        else:
            dest = output_dir / f
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        print(f"  Copied {f} -> {dest.relative_to(output_dir)}")


def load_weights(
    checkpoint_dir: Path,
    *,
    index_filename: str = "model.safetensors.index.json",
    single_filename: str = "model.safetensors",
) -> dict[str, mx.array]:
    """Load weights from sharded or single safetensors files.

    If an index file exists, loads shards. Otherwise loads a single file.
    All weights are loaded lazily via mx.load() (memory-mapped).
    """
    index_path = checkpoint_dir / index_filename
    if index_path.exists():
        print("\nLoading sharded weights lazily...")
        weights: dict[str, mx.array] = {}
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard in shard_files:
            shard_path = checkpoint_dir / shard
            _validate_path_within(shard_path, checkpoint_dir)
            print(f"  Loading {shard}...")
            shard_weights = load_safetensors(shard_path)
            weights.update(shard_weights)
        return weights

    single_path = checkpoint_dir / single_filename
    print(f"\nLoading weights lazily from {single_path.name}...")
    return load_safetensors(single_path)


def classify_keys(
    weights: dict[str, mx.array],
    classify_fn: Callable[[str], str | None],
) -> dict[str, list[str]]:
    """Group weight keys by component using a classification function.

    Keys for which classify_fn returns None are skipped.
    """
    keys_by_component: dict[str, list[str]] = {}
    for key in weights:
        comp = classify_fn(key)
        if comp:
            keys_by_component.setdefault(comp, []).append(key)
    return keys_by_component


# Type alias for the optional per-weight transform function.
# Signature: (sanitized_key, weight, component_name) -> transformed_weight
WeightTransform = Callable[[str, mx.array, str], mx.array]


def process_component(
    checkpoint_weights: dict,
    component_name: str,
    keys: list[str],
    output_dir: Path,
    component_prefix: str,
    *,
    sanitizer: Callable[[str], str | None],
    transform: WeightTransform | None = None,
    output_filename: str | None = None,
) -> int:
    """Process one component: sanitize keys, optionally transform, materialize, save.

    Args:
        checkpoint_weights: Full checkpoint weight dict.
        component_name: Name of the component (for display).
        keys: List of checkpoint keys belonging to this component.
        output_dir: Directory to write the output safetensors file.
        component_prefix: Prefix to prepend to sanitized keys in output.
        sanitizer: Function to rename keys. Returns None to skip a key.
        transform: Optional per-weight transform (e.g. conv transposition).
        output_filename: Override output filename (default: {component_name}.safetensors).

    Returns:
        Number of weights saved.
    """
    component_weights = {}

    for key in tqdm(keys, desc=f"  {component_name}", leave=False):
        new_key = sanitizer(key)
        if new_key is None:
            continue

        weight = checkpoint_weights[key]
        if transform is not None:
            weight = transform(new_key, weight, component_name)

        _materialize(weight)
        component_weights[f"{component_prefix}.{new_key}"] = weight

    if not component_weights:
        print(f"  No weights for {component_name}, skipping")
        return 0

    count = len(component_weights)
    fname = output_filename or f"{component_name}.safetensors"
    output_file = output_dir / fname
    print(f"  Saving {count} weights to {output_file.name}...")
    mx.save_safetensors(str(output_file), component_weights)

    del component_weights
    gc.collect()
    mx.clear_cache()
    return count


def quantize_component(
    output_dir: Path,
    component_name: str,
    *,
    bits: int = 8,
    group_size: int = 64,
    should_quantize: Callable[[str, mx.array], bool],
    filename: str | None = None,
) -> None:
    """Quantize a component's weights in-place.

    Args:
        output_dir: Directory containing the component safetensors file.
        component_name: Name of the component (e.g. "text_model"), used in the
            progress output and to derive the filename.
        bits: Quantization bits (4 or 8).
        group_size: Quantization group size.
        should_quantize: Predicate deciding which weights to quantize.
        filename: Override for `{component_name}.safetensors`, for recipes
            whose output files are not named after their component (e.g. VOID's
            `void_pass1.safetensors`, LTX's per-variant transformer files).
    """
    filepath = output_dir / (filename or f"{component_name}.safetensors")
    if not filepath.exists():
        print(f"  WARNING: {filepath.name} not found, skipping quantization")
        return

    print(f"\n  Quantizing {component_name} to int{bits} (group_size={group_size})...")
    weights = load_safetensors(filepath)

    # quantize_weights() empties `weights` as it runs (see its docstring) to keep
    # peak memory bounded; it is not touched again after this call.
    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize,
    )

    print(f"  Saving quantized {component_name} ({len(result)} keys)...")
    mx.save_safetensors(str(filepath), result)

    del result, weights
    gc.collect()
    mx.clear_cache()


def shard_filenames(n: int, prefix: str = "model") -> list[str]:
    """Generate shard filenames for n-shard models, plus the index file.

    No recipe calls this today (fish-s2-pro was the only caller). Kept because
    it is the counterpart of load_weights()'s index handling and any sharded
    upstream needs it — covered by tests so it cannot rot silently.
    """
    shards = [f"{prefix}-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    shards.append(f"{prefix}.safetensors.index.json")
    return shards
