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

from mlx_forge.cli import _remote_variants
from mlx_forge.upload import backfill_from_recipe, card_links, generate_model_card

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
    # Mirror the CLI exactly, including the remote-derived variant lists —
    # otherwise this tool reports losses a real refresh would not produce.
    variants, loras = _remote_variants(api, repo)
    genere = generate_model_card(
        d,
        split_info=split_info,
        config=config,
        repo_id=repo,
        file_listing=listing,
        usage_url=split_info.get("usage_url"),
        # card_links(), not split_info["links"]: the CLI renders the recipe's
        # links followed by the repo's own `extra_links`, so reading only the
        # declaration reported every sibling link as a loss the refresh would
        # not cause. Same failure as #72 — this tool must mirror the CLI.
        links=card_links(split_info),
        cli_snippet=split_info.get("cli_snippet"),
        transformer_variants=variants,
        lora_files=loras,
    )
    diff = list(
        difflib.unified_diff(
            publie.splitlines(), genere.splitlines(), "publie", "genere", lineterm="", n=0
        )
    )

    def contenu(prefixe, entete):
        return [
            line
            for line in diff
            if line.startswith(prefixe) and not line.startswith(entete) and line[1:].strip()
        ]

    def cle(line: str) -> str:
        # Same pairing the CLI uses: an entry whose size or value changed is a
        # modification, not content disappearing.
        body = line[1:]
        if body.startswith("- `"):
            return body.split("` (")[0]
        if body.startswith("- **") and ":**" in body:
            return body.split(":**", 1)[0]
        return body

    ajouts = contenu("+", "+++")
    remplaces = {cle(line) for line in ajouts}
    pertes = [line for line in contenu("-", "---") if cle(line) not in remplaces]
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
