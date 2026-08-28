"""Report what every published card COULD say from its metadata, and does not.

The companion to check_published_cards.py, which asks a different question.
That one asks whether regenerating a card loses anything; a card can be poor
and still regenerate perfectly, because the published version is equally poor.
This one asks whether the metadata behind a card is complete.

Maintenance tool, not a test: it needs network access and a HF login, so CI
does not run it.

    uv run python scripts/check_card_completeness.py [--author NAME]

A gap here is a card that could say more with no new work beyond a declaration:
a Usage section it lacks, a sibling build it does not point at, a quantization
it cannot describe. Some absences are legitimate and are reported as such —
V-JEPA 2.1 has no Hub base_model because Meta publishes it as a direct .pt
download, and inventing one would be worse than leaving it out.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download

from mlx_forge.recipes import resolve_recipe_metadata
from mlx_forge.upload import card_links

#: Absences that are correct, with the reason. Checked so that a gap which
#: later becomes fixable stops being silently excused.
EXPECTED_ABSENCES = {
    ("vjepa-2.1-vitl-mlx", "no base_model"): (
        "Meta publishes V-JEPA 2.1 as a direct .pt download, not a Hub repo"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--author", default="dgrauet", help="Hub namespace to scan")
    return parser.parse_args(argv)


def _json(repo: str, filename: str) -> dict:
    try:
        with open(hf_hub_download(repo, filename)) as f:
            return json.load(f)
    except Exception:
        return {}


def gaps_for(repo: str, data: dict, siblings: list[str]) -> list[str]:
    """Everything this repo's metadata could supply to its card and does not."""
    split_info, card_data, qconfig = data["split_info"], data["card_data"], data["qconfig"]
    metadata = resolve_recipe_metadata(split_info)

    out = []
    if metadata is None:
        out.append("recipe unidentified")
    if not card_data.get("base_model"):
        out.append("no base_model")
    if not split_info.get("usage_url"):
        out.append("no usage_url")
    if not card_links(split_info):
        out.append("no links")
    # Reported apart: every other gap closes with a declaration, but a usage
    # snippet is a command that has to actually work. Guessing one publishes a
    # lie — an invented `pip install` reached a draft card once — so this is
    # the operator's to supply, and the tool must not pretend otherwise.
    if not split_info.get("cli_snippet"):
        out.append("NEEDS-OPERATOR: no cli_snippet")

    if qconfig:
        # quantize_config.json is the quantizer's own record, so its presence
        # is what proves the build is quantized, whatever the manifest says.
        if not split_info.get("quantized"):
            out.append("manifest silent on quantized")
        if not split_info.get("quantization_bits"):
            out.append("manifest silent on bits")
        if not split_info.get("quantization_group_size"):
            out.append("manifest silent on group_size")
        if metadata and not metadata.quantization_scope:
            out.append("no quantization_scope")

    linked = " ".join(card_links(split_info))
    unlinked = [s.split("/")[1] for s in siblings if s.split("/")[1] not in linked]
    if unlinked:
        out.append("sibling(s) unlinked: " + ", ".join(sorted(unlinked)))
    return out


def main() -> None:
    args = parse_args()
    api = HfApi()

    repos: dict[str, dict] = {}
    for model in sorted(api.list_models(author=args.author), key=lambda m: m.id):
        info = api.model_info(model.id)
        repos[model.id] = {
            "card_data": info.cardData or {},
            "split_info": _json(model.id, "split_model.json"),
            "qconfig": _json(model.id, "quantize_config.json").get("quantization", {}),
        }

    # Builds of one model share a name up to the -q<bits> suffix.
    families: dict[str, list[str]] = defaultdict(list)
    for repo in repos:
        families[re.sub(r"-q\d+$", "", repo)].append(repo)

    rows = []
    for repo, data in repos.items():
        siblings = [s for s in families[re.sub(r"-q\d+$", "", repo)] if s != repo]
        name = repo.split("/")[1]
        found = gaps_for(repo, data, siblings)
        excused = [g for g in found if (name, g) in EXPECTED_ABSENCES]
        rows.append((name, [g for g in found if g not in excused], excused))

    width = max(len(name) for name, _, _ in rows)
    complete = 0
    derivable = 0
    for name, found, excused in rows:
        if not found:
            complete += 1
            suffix = f"  (expected: {', '.join(excused)})" if excused else ""
            print(f"   {name:<{width}}  complete{suffix}")
        else:
            if all(g.startswith("NEEDS-OPERATOR") for g in found):
                derivable += 1
            print(f"<< {name:<{width}}  " + "; ".join(found))

    print(f"\ncomplete: {complete}/{len(rows)}")
    print(f"complete but for a usage snippet only the operator can write: {complete + derivable}")
    for (name, gap), reason in sorted(EXPECTED_ABSENCES.items()):
        print(f"expected absence — {name}: {gap} ({reason})")

    # Only the declarative gaps fail the run: the rest waits on a human-verified
    # command, and a check that can never pass is one nobody runs.
    sys.exit(1 if complete + derivable < len(rows) else 0)


if __name__ == "__main__":
    main()
