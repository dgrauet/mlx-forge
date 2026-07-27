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
SPLIT_MODEL_KEYS = ("source", "links", "usage_url", "cli_snippet")


@dataclass(frozen=True)
class RecipeMetadata:
    """What a recipe knows about its own publication.

    Args:
        source: Upstream repo or origin, e.g. "Skywork/Matrix-Game-3.0".
            Becomes `base_model` in the model card front-matter, and the basis
            for the auto-derived repo name.
        links: Related projects, each "Label: URL".
        usage_url: Inference project that consumes these weights.
        cli_snippet: Bash shown in the card's Usage section. Published
            verbatim on the Hub — only reference a package or command that
            actually exists, and remember it now persists across refreshes.
    """

    source: str
    links: list[str] = field(default_factory=list)
    usage_url: str | None = None
    cli_snippet: str | None = None

    def with_source(self, source: str) -> RecipeMetadata:
        """Same declaration with a different origin.

        For recipes whose upstream depends on a flag — ernie-image publishes
        from ERNIE-Image-SFT or -Turbo depending on --variant.
        """
        return replace(self, source=source)

    def as_split_fields(self) -> dict:
        """The subset to persist in split_model.json, omitting empty values."""
        out: dict = {"source": self.source}
        if self.links:
            out["links"] = list(self.links)
        if self.usage_url:
            out["usage_url"] = self.usage_url
        if self.cli_snippet:
            out["cli_snippet"] = self.cli_snippet
        return out
