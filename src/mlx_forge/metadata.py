"""Publication metadata a recipe declares once, instead of the operator retyping it.

Card and repo metadata used to reach the model card only through CLI flags at
upload time. The result, measured across the 21 published repos: two vjepa
models inverted relative to their own code — one carries `base_model` because
`--base-model` happened to be typed that day, the other lacks it although its
recipe writes a source. Metadata that lives in flags tracks neither the recipe
nor a rule.

A recipe declares its metadata here; `convert` persists it into
split_model.json; `upload` reads it back. CLI flags still win, for one-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Keys carried through split_model.json. Additive: recipes keep whatever else
#: they already write, and downstream consumers of the old shape keep working.
SPLIT_MODEL_KEYS = (
    "recipe",
    "variant",
    "source",
    "base_model",
    "license",
    "links",
    "usage_url",
    "cli_snippet",
    "usage_note",
)


@dataclass(frozen=True)
class RecipeMetadata:
    """What a recipe knows about its own publication.

    Args:
        name: The recipe's registry key, e.g. "ernie-image-pe". Written into
            split_model.json as `recipe`, which is what lets `upload` find the
            declaration again for a directory converted long ago.
        variant: Which variant of the model this directory holds, when the
            recipe converts more than one — ernie-image publishes an "sft" and
            a "turbo" build from different upstream repos. Optional: most
            recipes have a single output and leave it None.
        known_sources: Every upstream this recipe converts from, when its
            variants come from different repos. Used to recognise a directory
            by its `source` when the manifest predates the `recipe` key —
            ernie-image writes the SFT repo for --variant sft, which is not the
            declaration's own source.
        source: Where the weights come from, as prose — it may name a
            subfolder or a non-Hub origin ("baidu/ERNIE-Image-Turbo/pe",
            "facebookresearch/vjepa2 (app/vjepa_2_1)"). Basis for the
            auto-derived repo name.
        base_model: Only when the Hub repo is NOT the one `source` names —
            vjepa-2.0 converts from Meta's source tree but its weights
            correspond to `facebook/vjepa2-vitl-fpc64-256`. Otherwise leave
            None: the repo is derived from `source`, dropping any subfolder or
            variant inside it.
        license: SPDX identifier for the card front-matter. Declared here
            because the CLI default ("other") silently downgraded it on every
            refresh — 13 of the 21 published repos carry apache-2.0 or mit.
        links: Related projects, each "Label: URL".
        usage_url: Inference project that consumes these weights.
        usage_note: Clause appended after the project link, before the period —
            "a native MLX inference pipeline for LTX-2.3 on Apple Silicon".
        cli_snippet: Bash shown in the card's Usage section. Published
            verbatim on the Hub — only reference a package or command that
            actually exists, and remember it now persists across refreshes.
            `{repo_id}` is substituted with the target repo at render time, so
            one declaration covers a model's bf16/q8/q4 repos.
    """

    name: str
    source: str
    variant: str | None = None
    known_sources: tuple[str, ...] = ()
    base_model: str | None = None
    license: str | None = None
    links: list[str] = field(default_factory=list)
    usage_url: str | None = None
    usage_note: str | None = None
    cli_snippet: str | None = None

    def for_variant(self, variant: str, source: str | None = None) -> RecipeMetadata:
        """This declaration as it applies to one variant of the model.

        ernie-image publishes an "sft" and a "turbo" build, each from its own
        upstream repo, so the variant usually comes with a source.
        """
        return replace(self, variant=variant, source=source or self.source)

    def with_source(self, source: str) -> RecipeMetadata:
        """Same declaration with a different origin, variant unchanged."""
        return replace(self, source=source)

    def as_split_fields(self) -> dict:
        """The subset to persist in split_model.json, omitting empty values."""
        out: dict = {"recipe": self.name, "source": self.source}
        if self.variant:
            out["variant"] = self.variant
        if self.base_model:
            out["base_model"] = self.base_model
        if self.license:
            out["license"] = self.license
        if self.links:
            out["links"] = list(self.links)
        if self.usage_url:
            out["usage_url"] = self.usage_url
        if self.usage_note:
            out["usage_note"] = self.usage_note
        if self.cli_snippet:
            out["cli_snippet"] = self.cli_snippet
        return out


def is_hub_repo_id(value: str | None) -> bool:
    """Whether `value` is exactly a Hub repo id, "owner/name"."""
    if not value:
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not any(c in value for c in " ()")


def hub_repo_from_source(source: str | None) -> str | None:
    """The Hub repo a `source` refers to, ignoring anything inside it.

    base_model names the remote repository, not the variant or subfolder taken
    from it: "baidu/ERNIE-Image-Turbo/pe" is published from
    "baidu/ERNIE-Image-Turbo". Returns None when `source` does not name a Hub
    repo at all — "facebookresearch/vjepa2 (app/vjepa_2_1)" is a source tree,
    and an unresolvable base_model is worse than none.
    """
    if not source:
        return None
    parts = source.split("/")
    if len(parts) < 2:
        return None
    candidate = "/".join(parts[:2])
    return candidate if is_hub_repo_id(candidate) else None
