"""Audit where every published licence copy came from, and whether it still holds.

`ensure_license_file` records a copy's origin and hash in the manifest, which
makes a local check cheap: rehash the file, compare, done. What that record
cannot do on its own is notice when the *upstream* text changes — the copy still
matches what we wrote down, because we wrote down the old one. "Upstream" here
may be a Hub repo or a GitHub repository: LTX-2.5's agreement is published only
on GitHub, so a Hub-only check would never see it at all.

That is not hypothetical. Lightricks published an "LTX-2.x" agreement on
2026-08-11, 30938 bytes against the 21399 the Hub still serves, adding §6 AI
Regulations obligations. Nothing in the portfolio would have said so.

This asks both questions of every published repo that ships a licence:

  * does the shipped file match the provenance recorded beside it?
  * does it still match what upstream publishes today?

A drift is not automatically a defect. The copy we ship is the agreement the
weights were obtained under, and replacing it is a decision, not a refresh — so
drift is reported, never acted on.

Maintenance tool, not a test: network access and a HF login, so CI skips it.

    uv run python scripts/check_license_provenance.py [--author NAME]
"""

import argparse
import hashlib
import json
import sys

from huggingface_hub import HfApi, hf_hub_download

from mlx_forge.convert import fetch_github_license, parse_license_source
from mlx_forge.metadata import hub_repo_from_source, license_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--author", default="dgrauet", help="Hub namespace to scan")
    return parser.parse_args(argv)


def digest(repo: str, filename: str) -> str | None:
    try:
        with open(hf_hub_download(repo, filename), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except Exception:  # noqa: BLE001 — an unreachable file is a missing one here
        return None


def main() -> None:
    args = parse_args()
    api = HfApi()

    rows: list[tuple[str, list[str], list[str]]] = []
    for model in sorted(api.list_models(author=args.author), key=lambda m: m.id):
        try:
            with open(hf_hub_download(model.id, "split_model.json")) as handle:
                split_info = json.load(handle)
        except Exception:  # noqa: BLE001
            continue

        declared = license_files(split_info.get("license_file"))
        if not declared:
            continue

        try:
            github = parse_license_source(split_info.get("license_source"))
        except ValueError as exc:
            rows.append((model.id.split("/")[1], [f"license_source: {exc}"], []))
            continue
        upstream = (
            f"github:{github[0]}"
            if github
            else (split_info.get("base_model") or hub_repo_from_source(split_info.get("source")))
        )
        provenance = split_info.get("license_provenance") or {}
        problems, drifts = [], []

        for path in declared:
            name = path.split("/")[-1]
            shipped = digest(model.id, name)
            if shipped is None:
                problems.append(f"{name}: declared but not in the repo")
                continue

            recorded = (provenance.get(name) or {}).get("sha256")
            if not recorded:
                problems.append(f"{name}: shipped with no recorded provenance")
            elif recorded != shipped:
                problems.append(f"{name}: does not match its recorded provenance")

            if github:
                try:
                    content, _ = fetch_github_license(github[0], path)
                    current = hashlib.sha256(content).hexdigest()
                except Exception:  # noqa: BLE001 — an unreachable text is an unknown one here
                    current = None
            else:
                current = digest(upstream, path) if upstream else None
            if current is None:
                problems.append(f"{name}: cannot read {path} from {upstream} to compare")
            elif current != shipped:
                drifts.append(f"{name}: upstream {upstream} now publishes a different text")

        rows.append((model.id.split("/")[1], problems, drifts))

    if not rows:
        print("no published repo ships a licence copy")
        sys.exit(0)

    width = max(len(name) for name, _, _ in rows)
    for name, problems, drifts in rows:
        if not problems and not drifts:
            print(f"   {name:<{width}}  vouched, and upstream agrees")
        for problem in problems:
            print(f"<< {name:<{width}}  {problem}")
        for drift in drifts:
            print(f" ~ {name:<{width}}  DRIFT {drift}")

    print(f"\nrepos shipping a licence: {len(rows)}")
    print(f"with a provenance problem: {sum(1 for _, p, _ in rows if p)}")
    print(f"where upstream has since changed: {sum(1 for _, _, d in rows if d)}")

    # Drift is upstream's doing and needs a decision; only our own record failing is
    # a defect this repo can fix.
    sys.exit(1 if any(p for _, p, _ in rows) else 0)


if __name__ == "__main__":
    main()
