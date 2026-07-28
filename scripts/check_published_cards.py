"""Regenerate every published card and diff it against what is on the Hub.

Maintenance tool, not a test: it needs network access and a HF login, so CI
does not run it. Use it before refreshing cards in bulk.

    uv run python scripts/check_published_cards.py [-v] [--author NAME]

A card that regenerates with LOSSES must not be pushed: the loss is content
that exists only in the published README, typed once as a CLI flag. Either
declare it in the recipe (if it is true of every build) or pass the flag again
(if it is specific to that repo) — `persist_card_metadata` then keeps it.

Simulates `mlx-forge upload <dir> --card-only` for a directory that mirrors the
remote: manifest from the repo, file listing from the repo, no CLI flags. The
criterion is that regenerating must never LOSE content.
"""

import difflib
import json
import pathlib
import sys
import tempfile

from huggingface_hub import HfApi, hf_hub_download

from mlx_forge.upload import backfill_from_recipe, generate_model_card

api = HfApi()
pertes_totales = 0
resume = []
AUTHOR = sys.argv[sys.argv.index("--author") + 1] if "--author" in sys.argv else "dgrauet"

for m in sorted(api.list_models(author=AUTHOR), key=lambda x: x.id):
    repo = m.id
    try:
        publie = open(hf_hub_download(repo, "README.md")).read()
    except Exception:
        continue
    info = api.model_info(repo, files_metadata=True)
    listing = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    try:
        split_info = json.load(open(hf_hub_download(repo, "split_model.json")))
    except Exception:
        split_info = {}
    try:
        config = json.load(open(hf_hub_download(repo, "config.json")))
    except Exception:
        config = {}

    d = pathlib.Path(tempfile.mkdtemp())
    split_info = backfill_from_recipe(d, dict(split_info))
    genere = generate_model_card(
        d,
        split_info=split_info,
        config=config,
        repo_id=repo,
        file_listing=listing,
        usage_url=split_info.get("usage_url"),
        links=split_info.get("links"),
        cli_snippet=split_info.get("cli_snippet"),
    )
    diff = list(
        difflib.unified_diff(
            publie.splitlines(), genere.splitlines(), "publie", "genere", lineterm="", n=0
        )
    )
    pertes = [
        line
        for line in diff
        if line.startswith("-") and not line.startswith("---") and line[1:].strip()
    ]
    ajouts = [
        line
        for line in diff
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    pertes_totales += len(pertes)
    resume.append((repo.split("/")[-1], len(pertes), len(ajouts)))
    if pertes and "-v" in sys.argv:
        print(f"\n===== {repo}  ({len(pertes)} pertes)")
        for line in pertes[:14]:
            print("   ", line)

print(f"\n{'depot':40} {'pertes':>7} {'ajouts':>7}")
for r, p, a in resume:
    print(f"{r:40} {p:>7} {a:>7}" + ("   <-- PERTE" if p else ""))
print(f"\nTOTAL pertes: {pertes_totales}")
