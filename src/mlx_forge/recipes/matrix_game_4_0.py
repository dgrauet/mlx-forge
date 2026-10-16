"""Matrix-Game-4.0 conversion recipe.

Converts Skywork/Matrix-Game-4.0 to MLX split format, all scenes in one pass:
  - dit_castle         — Medieval Castle scene DiT (Hub folder fps-castle/)
  - dit_ancient_egypt  — Ancient Egypt scene DiT (Hub folder fps-egypt/)
  - dit_racing         — City Racing scene DiT (Hub folder driving_game/)
  - taehv              — TAEHV taew2_2 streaming codec (realtime preset codec)

Each scene DiT is a causal finetune of Wan2.2-Fun-5B-Control: the game engine
renders a low-poly control video and the model re-renders it, so there is no
action module (unlike Matrix-Game-3.0). Its 148 input channels are noisy
latent 48 | control latent 48 | zeros 52 (inpaint 48 + mask 4).

Shared files are referenced, not converted. Upstream does not duplicate T5,
the Wan2.2 VAE or the tokenizer: releases/models.json fetches them from
alibaba-pai/Wan2.2-Fun-5B-Control at a pinned revision. Those sources are
byte-identical to the ones Matrix-Game-3.0 ships, and our matrix-game-3.0-mlx
pack holds their MLX conversion, so this pack does the same: split_model.json
records a `shared` block pointing at t5_encoder.safetensors, vae.safetensors
and google/umt5-xxl/* in that repo, pinned by revision and sha256. `convert`
checks that reference against the Hub before writing it.

Not converted:
  - driving_game/vae_decoder/decoder.pth — a TAEHV with a finetuned decoder,
    absent from releases/models.json and loaded nowhere in the reference
    source (the racing scene uses base/taew2_2.pth like the others).
  - game/ and racing/assets/ — geometry, maps and audio for the game engine
    that renders the control video, not model weights.

Usage:
    mlx-forge convert matrix-game-4.0
    mlx-forge convert matrix-game-4.0 --quantize --bits 8
    mlx-forge validate matrix-game-4.0 models/matrix-game-4.0-mlx
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import HfApi

from ..convert import (
    add_common_convert_args,
    copy_required_files,
    default_output_dir,
    download_hf_files,
    fmt_size,
    load_safetensors,
    load_torch_state_dict,
    print_output_summary,
    process_component,
    quantization_manifest_fields,
    quantize_component,
    source_download_dir,
    write_split_model,
)
from ..metadata import RecipeMetadata
from ..quantize import read_quantize_config, write_quantize_config
from ..transpose import transpose_conv
from ..validate import (
    ValidationResult,
    count_layer_indices,
    finish_validation,
    start_validation,
    validate_file_exists,
    validate_quantization,
)
from . import matrix_game_3_0 as mg3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "Skywork/Matrix-Game-4.0"

# Upstream base model, pinned exactly as releases/models.json pins it. Only
# its config.json is read here (the DiT architecture).
BASE_REPO_ID = "alibaba-pai/Wan2.2-Fun-5B-Control"
BASE_REVISION = "b8bc1a65ab71d054ba4636dc0dac104aa4df2686"

# Where the converted T5 / VAE / tokenizer live. Its t5_encoder and vae files
# were converted from the same sources (identical sha256) by matrix-game-3.0,
# and re-converting them here reproduced both files byte for byte.
SHARED_REPO_ID = "dgrauet/matrix-game-3.0-mlx"
SHARED_REVISION = "ff0d4df16dde5c86fca8cbd934a41fdff5eec00c"
SHARED_COMPONENTS: dict[str, str] = {
    "t5_encoder": "t5_encoder.safetensors",
    "vae": "vae.safetensors",
}
SHARED_TOKENIZER_DIR = "google/umt5-xxl"
# Content sha256 of every referenced file at SHARED_REVISION.
SHARED_FILES: dict[str, str] = {
    "t5_encoder.safetensors": "f8770d43e586deb65541e1925d106ecf2901982c7e4b396409b86ea47f29e606",
    "vae.safetensors": "fee0cef1828cc1285fd2ae3cb6d85bc510dd8dc00f19fe5b357a065f3f4998a4",
    "google/umt5-xxl/special_tokens_map.json": (
        "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"
    ),
    "google/umt5-xxl/spiece.model": (
        "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"
    ),
    "google/umt5-xxl/tokenizer.json": (
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
    ),
    "google/umt5-xxl/tokenizer_config.json": (
        "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"
    ),
}

# Runtime scene name -> Hub folder. The upstream downloader maps these Hub
# folders to resources/models/<scene>/; the names are not interchangeable.
SCENES: dict[str, str] = {
    "castle": "fps-castle",
    "ancient-egypt": "fps-egypt",
    "racing": "driving_game",
}


def scene_component(scene: str) -> str:
    """Output component name for a runtime scene name ("ancient-egypt" -> "dit_ancient_egypt")."""
    return "dit_" + scene.replace("-", "_")


DIT_COMPONENTS = [scene_component(s) for s in SCENES]

# Components this pack holds; t5_encoder and vae come from SHARED_REPO_ID.
COMPONENTS = [*DIT_COMPONENTS, "taehv"]

# Output sizes in MB, measured on a real conversion
_COMPONENT_SIZE_MB: dict[str, int] = {
    **dict.fromkeys(DIT_COMPONENTS, 9_539),  # 5.0B BF16 params each
    "taehv": 22,  # FP16
}

TAEHV_FILE = "base/taew2_2.pth"
BASE_CONFIG_FILE = "config.json"

# Per-scene inference contract: inference.json carries the sink policy, RoPE
# mode, steps and prompt; sink.png is the identity sink (castle/egypt) or the
# supplied first frame (racing). The runtime cannot run a scene without them.
SCENE_FILES = [
    f"{folder}/{name}" for folder in SCENES.values() for name in ("inference.json", "sink.png")
]

# Copied with the weights: upstream's Apache-2.0 LICENSE (section 4(a) asks
# redistributors to pass a copy on; licenses/MG4CF-LICENSE upstream is the same
# text, byte for byte, so it is not duplicated) and TAEHV's MIT notice, which
# must travel with copies of those weights. Pipeline files rather than
# METADATA.license_file, which the project reserves for bespoke licences.
LICENSE_FILES = ["LICENSE", "licenses/TAEHV-LICENSE"]

# Upstream DiT checkpoints are saved from a wrapper; the reference loader
# (matrix_game/inference/loading.py) strips one of these whole-file prefixes.
_DIT_CHECKPOINT_PREFIXES = ("pipe.dit.", "dit.", "model.")

# Present in Fun-5B-Control checkpoints, never built by the reference model
# (wrappers.py IGNORED_KEY_PREFIXES). Skipped rather than shipped dead.
_DIT_IGNORED_PREFIXES = ("ref_conv.",)

# Wan2.2-Fun-5B-Control architecture, as the reference wrapper's
# FUN5B_CONTROL_CONFIG declares it (used when a scene dir has no config.json,
# which is every Matrix-Game-4.0 scene).
_FUN5B_CONTROL_ARCH: dict = {
    "model_type": "ti2v",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "in_dim": 148,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

# Upstream tensor count per scene checkpoint (each scene's model.json).
EXPECTED_DIT_TENSORS = 825

METADATA = RecipeMetadata(
    name="matrix-game-4.0",
    source=REPO_ID,
    # Skywork/Matrix-Game-4.0 declares apache-2.0 for the scene weights; the
    # shared Wan/umT5 files are apache-2.0 upstream too. The Apache text and
    # TAEHV's MIT notice ship as pipeline files (LICENSE_FILES).
    license="apache-2.0",
    quantization_scope=(
        "DiT transformer-block Linear weights only (all three scenes), leaving the "
        "patch/time/text embeddings, the output head, norms and modulation parameters "
        "in bf16; TAEHV untouched (T5 and the Wan2.2 VAE are not in this repo)"
    ),
    links=["Upstream code: https://github.com/SkyworkAI/Matrix-Game"],
)

# ---------------------------------------------------------------------------
# DiT key sanitization
# ---------------------------------------------------------------------------


def strip_checkpoint_prefix(keys: list[str]) -> str:
    """Return the wrapper prefix every key shares, or "" when there is none.

    Mirrors clean_state_dict_keys in the reference loader: a prefix is only
    stripped when *all* keys carry it.
    """
    for prefix in _DIT_CHECKPOINT_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            return prefix
    return ""


def sanitize_dit_key(key: str, prefix: str = "model.") -> str | None:
    """Convert an upstream scene-checkpoint key to MLX format.

    Strips the checkpoint wrapper prefix, drops the unused Fun reference-image
    branch, then applies Matrix-Game-3.0's Sequential -> named-Linear mapping
    (the same Wan2.2 backbone, without the action module).
    """
    key = key.removeprefix(prefix)
    if key.startswith(_DIT_IGNORED_PREFIXES):
        return None
    return mg3.sanitize_dit_key(key)


def transform_dit(key: str, value: mx.array, _component: str) -> mx.array:
    """patch_embedding Conv3d (O, I, 1, 2, 2) -> Linear (O, I*4); everything else is Linear."""
    return mg3.maybe_transpose(key, value, "dit")


# ---------------------------------------------------------------------------
# TAEHV
# ---------------------------------------------------------------------------


def sanitize_taehv_key(key: str) -> str | None:
    """TAEHV keys (encoder.N.*, decoder.N.*) index nn.Sequential slots; kept verbatim.

    The indices are load-bearing — they encode where the parameter-free
    ReLU/Upsample/PixelShuffle layers sit — so an MLX port must rebuild the
    same Sequential layout rather than rename them.
    """
    return key


def transform_taehv(key: str, value: mx.array, _component: str) -> mx.array:
    """Conv2d weights (O, I, H, W) -> MLX channels-last (O, H, W, I).

    Every 4D tensor in taew2_2 is a Conv2d (including the 1x1 TPool/TGrow and
    MemBlock skip convs). No TGrow kernel patching is needed: the reference
    patch_tgrow_layers only trims a decoder TGrow whose stride the checkpoint
    exceeds, and taew2_2's first TGrow (decoder.7) is stride 1 on both sides.
    """
    if key.endswith(".weight") and value.ndim == 4:
        return transpose_conv(value)
    return value


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def should_quantize(key: str, weight: mx.array) -> bool:
    """Quantize only Linear weights inside DiT transformer blocks.

    Keys arrive prefixed with their scene component ("dit_castle.blocks.0...").
    Matrix-Game-3.0's predicate strips a literal "dit." and so would quantize
    the output head here; this one matches on the bare key's structure instead.
    """
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    bare_key = key.split(".", 1)[1] if "." in key else key
    if not bare_key.startswith("blocks."):
        # patch_embedding, text/time embeddings, time_projection, head
        return False
    leaf = bare_key.split(".")[-2]
    if "norm" in leaf or "modulation" in bare_key:
        return False
    # blocks.N.{self_attn,cross_attn}.{q,k,v,o} and blocks.N.ffn_linear{1,2}
    return True


# ---------------------------------------------------------------------------
# Shared-component reference
# ---------------------------------------------------------------------------


def shared_manifest() -> dict:
    """The `shared` block of split_model.json: what to fetch from SHARED_REPO_ID.

    A consumer downloads `files` from `repo_id` at `revision`, checks each
    sha256, and loads `components` (name -> file) and the tokenizer directory.
    """
    return {
        "repo_id": SHARED_REPO_ID,
        "revision": SHARED_REVISION,
        "components": dict(SHARED_COMPONENTS),
        "tokenizer": SHARED_TOKENIZER_DIR,
        "files": [{"path": path, "sha256": sha} for path, sha in SHARED_FILES.items()],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shared_reference(download_dir: Path, api: HfApi | None = None) -> None:
    """Check SHARED_FILES against the Hub at SHARED_REVISION; abort on any mismatch.

    LFS files are checked from the Hub's own sha256 metadata (no download);
    the small non-LFS files are fetched (kilobytes) and hashed. A pack whose
    reference resolves to nothing — or to different bytes — would be
    unusable, so this fails the conversion rather than writing the manifest.
    """
    api = api or HfApi()
    info = api.model_info(SHARED_REPO_ID, revision=SHARED_REVISION, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings or []}
    problems = []
    small = []
    for path, sha in SHARED_FILES.items():
        sibling = remote.get(path)
        if sibling is None:
            problems.append(f"{path}: missing")
        elif sibling.lfs is not None:
            if sibling.lfs.sha256 != sha:
                problems.append(f"{path}: sha256 {sibling.lfs.sha256} != {sha}")
        else:
            small.append(path)
    if small:
        target = download_dir / "shared"
        download_hf_files(SHARED_REPO_ID, small, target, revision=SHARED_REVISION)
        problems += [
            f"{path}: sha256 differs"
            for path in small
            if _sha256(target / path) != SHARED_FILES[path]
        ]
    if problems:
        raise SystemExit(
            f"ERROR: shared reference {SHARED_REPO_ID}@{SHARED_REVISION[:12]} does not match: "
            + "; ".join(problems)
        )
    print(f"  Shared reference verified: {SHARED_REPO_ID}@{SHARED_REVISION[:12]}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_config(base_config: dict | None, scene_contracts: dict[str, dict]) -> dict:
    """Build output config.json: the DiT architecture plus the scene table.

    Args:
        base_config: alibaba-pai/Wan2.2-Fun-5B-Control config.json, or None
            when unavailable. Where present it wins over the built-in
            architecture, so the numbers come from upstream, not from us.
        scene_contracts: Runtime scene name -> that scene's inference.json.

    Returns:
        The config dict written to config.json.
    """
    arch = dict(_FUN5B_CONTROL_ARCH)
    if base_config:
        for key in arch:
            if key in base_config:
                arch[key] = base_config[key]
    scenes = {
        scene: {
            "component": scene_component(scene),
            "inference": f"{folder}/inference.json",
            "sink_image": f"{folder}/sink.png",
            "name": scene_contracts.get(scene, {}).get("name"),
        }
        for scene, folder in SCENES.items()
    }
    return {
        **arch,
        # 148 = noisy 48 | control 48 | zeros 52 (inpaint 48 + mask 4)
        "control_channels": 48,
        "is_causal": True,
        "is_action_model": False,
        # "vae" is a shared component: see split_model.json `shared`.
        "codecs": {"quality": "vae", "realtime": "taehv"},
        "scenes": scenes,
        "source": REPO_ID,
        "base_source": f"{BASE_REPO_ID}@{BASE_REVISION}",
    }


# ---------------------------------------------------------------------------
# Component conversion helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _convert_dit(scene: str, download_dir: Path, output_dir: Path) -> int:
    """Download, convert and save one scene's DiT. Returns weight count."""
    component = scene_component(scene)
    folder = SCENES[scene]
    _banner(f"Converting {component} ({folder}/)")

    download_hf_files(REPO_ID, [f"{folder}/model.safetensors"], download_dir)
    path = download_dir / folder / "model.safetensors"

    t0 = time.monotonic()
    weights = load_safetensors(path)
    keys = list(weights)
    prefix = strip_checkpoint_prefix(keys)
    print(f"  {len(keys)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s, prefix {prefix!r}")

    t0 = time.monotonic()
    count = process_component(
        weights,
        component,
        keys,
        output_dir,
        component,
        sanitizer=lambda key: sanitize_dit_key(key, prefix),
        transform=transform_dit,
        # Upstream is BF16 throughout; cast only if a future checkpoint is not.
        dtype=lambda _key, weight: mx.bfloat16 if weight.dtype == mx.float32 else None,
    )
    print(f"  Done: {count} weights saved in {time.monotonic() - t0:.1f}s")
    del weights
    gc.collect()
    mx.clear_cache()
    return count


def _convert_taehv(pth_path: Path, output_dir: Path) -> int:
    """Convert the TAEHV taew2_2 codec. Returns weight count."""
    _banner("Converting TAEHV (taew2_2)")
    raw = load_torch_state_dict(pth_path, label=f"TAEHV weights from {pth_path}")
    print(f"  {len(raw)} keys loaded")
    count = process_component(
        raw,
        "taehv",
        list(raw),
        output_dir,
        "taehv",
        sanitizer=sanitize_taehv_key,
        transform=transform_taehv,
        # fp16 upstream, which numpy holds natively; kept as fp16.
        load_weight=lambda value: mx.array(value.numpy()),
    )
    print(f"  Done: {count} weights saved")
    del raw
    gc.collect()
    mx.clear_cache()
    return count


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert every Matrix-Game-4.0 scene plus TAEHV to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = default_output_dir("matrix-game-4.0", quantize=args.quantize, bits=args.bits)

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = source_download_dir(output_dir)
    base_dir = download_dir / "wan2.2-fun-5b-control"

    total_weights = 0

    # 1. Scene DiTs
    for scene in SCENES:
        total_weights += _convert_dit(scene, download_dir, output_dir)

    # 2. TAEHV
    download_hf_files(REPO_ID, [TAEHV_FILE, *LICENSE_FILES], download_dir)
    total_weights += _convert_taehv(download_dir / TAEHV_FILE, output_dir)

    # 3. Pipeline files: scene contracts + TAEHV notice
    _banner("Syncing pipeline files")
    download_hf_files(REPO_ID, SCENE_FILES, download_dir)
    copy_required_files(download_dir, output_dir, [*SCENE_FILES, *LICENSE_FILES], flatten=False)

    # 4. Shared T5 / VAE / tokenizer: referenced, checked against the Hub
    _banner(f"Verifying shared components in {SHARED_REPO_ID}")
    verify_shared_reference(download_dir)

    # 5. config.json (architecture from the pinned base repo)
    download_hf_files(BASE_REPO_ID, [BASE_CONFIG_FILE], base_dir, revision=BASE_REVISION)
    base_config = json.loads((base_dir / BASE_CONFIG_FILE).read_text())
    scene_contracts = {
        scene: json.loads((download_dir / folder / "inference.json").read_text())
        for scene, folder in SCENES.items()
    }
    config = build_config(base_config, scene_contracts)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print("\nSaved config.json")

    # Manifest written once, after the optional quantization, so `quantized`
    # reflects what actually happened.
    split_info: dict = {
        "format": "split",
        "components": COMPONENTS,
        "shared": shared_manifest(),
        **METADATA.as_split_fields(),
    }

    # 6. Optional quantization (scene DiTs only)
    if args.quantize:
        for comp in DIT_COMPONENTS:
            print(f"\nQuantizing {comp} to int{args.bits} (group_size={args.group_size})...")
            quantize_component(
                output_dir,
                comp,
                bits=args.bits,
                group_size=args.group_size,
                should_quantize=should_quantize,
            )
        write_quantize_config(output_dir, bits=args.bits, group_size=args.group_size)

    split_info.update(
        quantization_manifest_fields(
            quantized=args.quantize, bits=args.bits, group_size=args.group_size
        )
    )
    write_split_model(output_dir, split_info)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print_output_summary(output_dir)
    print("\nDone!")


def _dry_run(args, output_dir: Path) -> None:
    """Print conversion plan without executing anything."""
    print("=" * 60)
    print("DRY RUN — no files will be downloaded or written")
    print("=" * 60)

    print(f"\nSource:     {REPO_ID} (HuggingFace)")
    print(f"Base:       {BASE_REPO_ID}@{BASE_REVISION[:12]} (config.json)")
    print(f"Output dir: {output_dir}")
    print("\nOutput files:")

    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = float(_COMPONENT_SIZE_MB[comp])
        if args.quantize and comp in DIT_COMPONENTS:
            size_mb = size_mb * args.bits / 16
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (int{args.bits})")
        else:
            dtype = "fp16" if comp == "taehv" else "bf16"
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} ({dtype})")
        total_mb += size_mb

    print("  config.json, split_model.json")
    for scene, folder in SCENES.items():
        print(f"  {folder}/inference.json, {folder}/sink.png ({scene} contract)")
    print(f"  {', '.join(LICENSE_FILES)}")

    print(f"\nReferenced, not copied ({SHARED_REPO_ID}@{SHARED_REVISION[:12]}):")
    for path in SHARED_FILES:
        print(f"  {path}")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: DiT block Linear weights only (all three scenes)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_dit(
    model_dir: Path, component: str, config: dict, result: ValidationResult, is_quantized: bool
) -> None:
    """Validate one scene DiT."""
    print(f"\n== {component} Weights ==")
    path = model_dir / f"{component}.safetensors"
    if not path.exists():
        return

    weights = load_safetensors(path)
    keys = set(weights)
    prefix = f"{component}."
    result.check(
        all(k.startswith(prefix) for k in keys), f"All keys have '{prefix}' prefix ({len(keys)})"
    )
    bare = {k.removeprefix(prefix) for k in keys}

    result.check(
        not any(k.startswith("model.") for k in bare), "Checkpoint 'model.' prefix stripped"
    )
    result.check(
        not any(".ffn.0." in k or ".ffn.2." in k for k in bare),
        "No un-sanitized ffn Sequential keys",
    )
    result.check(not any("action_model" in k for k in bare), "No action module (MG4 has none)")

    num_layers = config.get("num_layers", 30)
    blocks = count_layer_indices(bare, block_key="blocks")
    result.check(len(blocks) == num_layers, f"{num_layers} transformer blocks (got {len(blocks)})")

    pe = weights.get(f"{prefix}patch_embedding.weight")
    if pe is not None:
        patch = config.get("patch_size", [1, 2, 2])
        in_features = config.get("in_dim", 148) * patch[0] * patch[1] * patch[2]
        result.check(
            tuple(pe.shape) == (config.get("dim", 3072), in_features),
            f"patch_embedding is Linear (dim, in_dim*prod(patch)) (shape: {tuple(pe.shape)})",
        )

    if is_quantized:
        validate_quantization(weights, result, block_key="blocks")
    else:
        result.check(
            len(keys) == EXPECTED_DIT_TENSORS,
            f"{EXPECTED_DIT_TENSORS} tensors, as upstream model.json (got {len(keys)})",
        )

    print(f"  Total {component} parameters: {sum(v.size for v in weights.values()) / 1e9:.2f}B")
    del weights
    gc.collect()
    mx.clear_cache()


def _validate_taehv(model_dir: Path, result: ValidationResult) -> None:
    print("\n== TAEHV Weights ==")
    path = model_dir / "taehv.safetensors"
    if not path.exists():
        return
    weights = load_safetensors(path)
    result.check(all(k.startswith("taehv.") for k in weights), "All keys have 'taehv.' prefix")
    result.check(
        any(k.startswith("taehv.encoder.") for k in weights)
        and any(k.startswith("taehv.decoder.") for k in weights),
        "Encoder and decoder present",
    )
    convs = {k: v for k, v in weights.items() if v.ndim == 4}
    # channels-last: (O, kH, kW, I) with kernels of 1 or 3
    suspect = [k for k, v in convs.items() if v.shape[1] not in (1, 3) or v.shape[2] not in (1, 3)]
    for k in suspect:
        print(f"    Suspect layout: {k} shape={tuple(convs[k].shape)}")
    result.check(not suspect, f"Conv2d weights in MLX channels-last layout ({len(convs)} checked)")
    # Latent width must match the DiT's: encoder head emits 48 channels.
    head = weights.get("taehv.encoder.17.weight")
    result.check(
        head is not None and head.shape[0] == 48,
        f"Encoder emits 48 latent channels (Wan2.2 width) "
        f"(got {None if head is None else head.shape[0]})",
    )
    del weights
    gc.collect()
    mx.clear_cache()


def validate(args) -> None:
    """Validate a converted Matrix-Game-4.0 model."""
    model_dir, result = start_validation(args.model_dir)

    qconfig = read_quantize_config(model_dir)
    is_quantized = qconfig is not None
    if qconfig is not None:
        print(f"Model is quantized: int{qconfig.get('bits', '?')}")

    print("\n== File Structure ==")
    for comp in COMPONENTS:
        validate_file_exists(model_dir, f"{comp}.safetensors", result)
    for name in ["config.json", "split_model.json", *SCENE_FILES, *LICENSE_FILES]:
        validate_file_exists(model_dir, name, result)

    print("\n== Shared Components ==")
    split_path = model_dir / "split_model.json"
    shared = json.loads(split_path.read_text()).get("shared") if split_path.exists() else None
    result.check(
        shared == shared_manifest(),
        f"split_model.json references {SHARED_REPO_ID}@{SHARED_REVISION[:12]} "
        f"(t5_encoder, vae, tokenizer)",
    )

    print("\n== Config Validation ==")
    config: dict = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        for key in ("num_layers", "num_heads", "dim", "in_dim", "out_dim"):
            expected = _FUN5B_CONTROL_ARCH[key]
            result.check(
                config.get(key) == expected, f"{key} == {expected} (got: {config.get(key)})"
            )
        result.check(
            set(config.get("scenes", {})) == set(SCENES),
            f"Scene table lists {sorted(SCENES)} (got: {sorted(config.get('scenes', {}))})",
        )

    for comp in DIT_COMPONENTS:
        _validate_dit(model_dir, comp, config, result, is_quantized)
    _validate_taehv(model_dir, result)

    finish_validation(result)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add Matrix-Game-4.0 convert arguments to a parser."""
    add_common_convert_args(
        parser,
        output_default="./models/matrix-game-4.0-mlx[-q<bits>]",
        quantize_help="Quantize all three scene DiTs after conversion",
        dry_run_help="Preview conversion plan without downloading or writing anything",
    )


def add_validate_args(parser) -> None:
    """Add Matrix-Game-4.0 validate arguments to a parser."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    """Add Matrix-Game-4.0 split arguments to a parser."""
    parser.add_argument("model_dir", type=str, help="Model directory")


def split(args) -> None:
    """No-op: conversion already writes one file per component."""
    print("Matrix-Game 4.0 is already split by component during conversion.")
    print("No further splitting needed.")
