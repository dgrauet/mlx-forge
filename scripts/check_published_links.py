"""Read the cards as PUBLISHED and check they hold together.

The third maintenance check, and the one that looks at the artefact itself.
check_published_cards.py compares a regeneration against the Hub;
check_card_completeness.py asks what the metadata could supply. Neither opens
the published page. This does: it fetches every card and verifies the things a
reader would hit — a fence that never closes, a `./LICENSE` that is not in the
repo, a Files section that disagrees with the repo, an unsubstituted
`{repo_id}`, and every link resolving.

It found a dead `license_link` mirrored verbatim from upstream: the file it
names is LICENSE.md, so the URL upstream declares has always been a 404.

Maintenance tool, not a test: network access and a HF login, so CI skips it.

    uv run python scripts/check_published_links.py [--author NAME] [--no-links]
"""

import re
import sys
import urllib.request

from huggingface_hub import HfApi, hf_hub_download

AUTHOR = sys.argv[sys.argv.index("--author") + 1] if "--author" in sys.argv else "dgrauet"
CHECK_LINKS = "--no-links" not in sys.argv

api = HfApi()


def structural_problems(repo: str, text: str, files: set[str]) -> list[str]:
    """Everything wrong with the card as a document, before any network call."""
    out = []

    if not text.startswith("---\n"):
        out.append("no YAML front-matter")
    else:
        for line in text.split("---\n", 2)[1].splitlines():
            if line and not line.startswith((" ", "-")) and ":" not in line:
                out.append(f"front-matter line is not key: value -> {line!r}")

    # str.format left a hole: the card advertises a placeholder as literal text.
    for placeholder in sorted(set(re.findall(r"\{[a-z_]+\}", text))):
        out.append(f"unsubstituted placeholder {placeholder}")

    if text.count("```") % 2:
        out.append("unbalanced ``` code fence")

    for target in sorted(set(re.findall(r"\]\(\./([^)]+)\)", text))):
        if target not in files:
            out.append(f"links ./{target}, which the repo does not contain")

    if "## Files" in text:
        listed = set(re.findall(r"^- `([^`]+)`", text.split("## Files", 1)[1], re.M))
        real = {
            f for f in files if f not in {"README.md", ".gitattributes"} and not f.startswith(".")
        }
        out += [f"repo has {f}, card omits it" for f in sorted(real - listed)]
        out += [f"card lists {f}, repo has not" for f in sorted(listed - real)]

    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        lines = block.splitlines()
        if lines and lines[-1].rstrip().endswith("\\"):
            out.append("a code block ends on a line continuation")
        # A swallowed backslash-newline collapses a whole command onto one line.
        if any(len(line) > 200 for line in lines):
            out.append("a code block line is over 200 chars (collapsed continuation?)")

    if f"# {repo}" not in text:
        out.append("title does not name the repo")
    for heading in set(re.findall(r"^## (.+)$", text, re.M)):
        if text.count(f"\n## {heading}\n") > 1:
            out.append(f"duplicate section {heading!r}")
    return out


def unreachable(url: str) -> str | None:
    """The failure for `url`, or None when it resolves."""
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
        status = urllib.request.urlopen(request, timeout=30).status
    except Exception as error:  # noqa: BLE001 — any failure is a failure to resolve
        status = getattr(error, "code", type(error).__name__)
    return None if status == 200 else str(status)


cards: dict[str, tuple[str, set[str]]] = {}
for model in sorted(api.list_models(author=AUTHOR), key=lambda m: m.id):
    info = api.model_info(model.id)
    try:
        with open(hf_hub_download(model.id, "README.md")) as handle:
            cards[model.id] = (handle.read(), {s.rfilename for s in (info.siblings or [])})
    except Exception as error:  # noqa: BLE001
        print(f"<< {model.id}: no README ({error})")

problems: list[tuple[str, str]] = []
for repo, (text, files) in cards.items():
    problems += [(repo.split("/")[1], p) for p in structural_problems(repo, text, files)]

if CHECK_LINKS:
    urls: dict[str, set[str]] = {}
    for repo, (text, _) in cards.items():
        for url in re.findall(r"https?://[^\s)\"'<>]+", text):
            urls.setdefault(url.rstrip(".,"), set()).add(repo.split("/")[1])
    print(f"checking {len(urls)} distinct URLs across {len(cards)} cards...")
    for url, on in sorted(urls.items()):
        failure = unreachable(url)
        if failure:
            problems.append((", ".join(sorted(on)), f"{failure} {url}"))

print(f"\ncards checked: {len(cards)}")
if problems:
    for where, problem in problems:
        print(f"<< {where}: {problem}")
    sys.exit(1)
print("no problem found")
