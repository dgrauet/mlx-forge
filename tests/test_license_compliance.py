"""Licence obligations a published pack must carry.

Converting and quantising a community-licensed model produces a derivative, and
those licences oblige the distributor to pass the agreement on to whoever
receives the weights. A `license_link` in the front-matter does not discharge
that, so these tests pin the two halves: the front-matter names the agreement,
and a verbatim copy of it ships with the weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from huggingface_hub import HfApi

from mlx_forge import convert
from mlx_forge.metadata import SPLIT_MODEL_KEYS, RecipeMetadata
from mlx_forge.recipes import AVAILABLE_RECIPES
from mlx_forge.upload import generate_model_card


def load_recipe(name: str):
    """The recipe module registered under `name`."""
    import importlib

    return importlib.import_module(AVAILABLE_RECIPES[name])


def _front_matter(card: str) -> dict[str, str]:
    """The card's YAML front-matter as a flat key -> value mapping."""
    assert card.startswith("---\n")
    body = card.split("---\n", 2)[1]
    out = {}
    for line in body.splitlines():
        if line.startswith((" ", "-")) or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_license_fields_are_carried_through_the_manifest():
    """A declared licence must survive into split_model.json.

    The card is rendered from the manifest, not from the recipe: a field the
    manifest drops is a field the published card cannot show.
    """
    metadata = RecipeMetadata(
        name="demo",
        source="acme/Demo",
        license="other",
        license_name="demo-community-license",
        license_link="https://example.invalid/LICENSE",
        license_file="LICENSE",
    )
    fields = metadata.as_split_fields()

    assert fields["license_name"] == "demo-community-license"
    assert fields["license_link"] == "https://example.invalid/LICENSE"
    assert fields["license_file"] == ["LICENSE"]
    for key in ("license_name", "license_link", "license_file"):
        assert key in SPLIT_MODEL_KEYS


def test_permissive_licenses_declare_no_license_file():
    """apache-2.0 and mit are identified by SPDX id alone.

    Shipping a copy is what a bespoke community licence requires; demanding one
    for every recipe would make `upload` refuse packs that owe nothing.
    """
    for name in AVAILABLE_RECIPES:
        metadata = getattr(load_recipe(name), "METADATA", None)
        if metadata is None or metadata.license in (None, "other"):
            continue
        assert metadata.license_file is None, (
            f"{name} declares license={metadata.license} with a license_file"
        )


def test_other_license_recipes_name_the_agreement():
    """ "other" is the SPDX escape hatch and identifies nothing on its own.

    A recipient reading `license: other` learns neither the terms nor where to
    find them, so any recipe using it must also declare license_name.
    """
    unnamed = [
        name
        for name in AVAILABLE_RECIPES
        if (m := getattr(load_recipe(name), "METADATA", None)) is not None
        and m.license == "other"
        and not m.license_name
    ]
    assert unnamed == [], f"license: other without license_name: {unnamed}"


def test_ltx_23_mirrors_its_upstream_declaration():
    """Pinned to what Lightricks/LTX-2.3 publishes, verbatim."""
    metadata = load_recipe("ltx-2.3").METADATA

    assert metadata.license == "other"
    assert metadata.license_name == "ltx-2-community-license-agreement"
    assert metadata.license_file == "LICENSE"
    assert metadata.as_split_fields()["license_file"] == ["LICENSE"]


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def _ltx_split_info() -> dict:
    return dict(load_recipe("ltx-2.3").METADATA.as_split_fields())


def test_the_licence_link_shows_the_text_we_ship():
    """The link and the shipped copy must be one document, not two.

    Upstream's card declares github.com/Lightricks/LTX-2/blob/main/LICENSE —
    a 404, because that file is LICENSE.md. And the .md is no longer the same
    agreement: on 2026-08-11 GitHub gained an "LTX-2.x" text dated August 2026,
    30938 bytes against the 21399 the Hub still publishes, with §6 AI
    Regulations obligations the shipped copy does not contain. A card saying
    "a verbatim copy ships with them, and the upstream text is at X" must not
    have X be a different document.
    """
    metadata = load_recipe("ltx-2.3").METADATA

    assert metadata.license_link == "https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE"
    assert "github.com/Lightricks/LTX-2/blob/main/LICENSE" not in (metadata.license_link or "")


def test_a_recipe_shipping_a_licence_links_where_that_licence_lives():
    """Whatever the URL, it must be the source of the file we distribute."""
    for name in AVAILABLE_RECIPES:
        metadata = getattr(load_recipe(name), "METADATA", None)
        if metadata is None or not metadata.license_file:
            continue
        assert metadata.license_link, f"{name} ships a licence copy but links nowhere"


def test_card_front_matter_names_and_links_the_agreement(tmp_path):
    card = generate_model_card(
        tmp_path,
        split_info=_ltx_split_info(),
        config={},
        repo_id="dgrauet/ltx-2.3-mlx",
        file_listing={"LICENSE": 10, "transformer.safetensors": 20},
    )
    front = _front_matter(card)

    assert front["license"] == "other"
    assert front["license_name"] == "ltx-2-community-license-agreement"
    assert front["license_link"] == "https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE"


def test_card_body_points_at_the_shipped_copy(tmp_path):
    """The body must not contradict the front-matter, and must be reachable."""
    card = generate_model_card(
        tmp_path,
        split_info=_ltx_split_info(),
        config={},
        repo_id="dgrauet/ltx-2.3-mlx",
        file_listing={"LICENSE": 10},
    )

    assert "## License" in card
    assert "(./LICENSE)" in card
    assert "https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE" in card
    # One licence, stated once: no competing SPDX id elsewhere in the body.
    body = card.split("---\n", 2)[2]
    assert "apache" not in body.lower() and "MIT" not in body


def test_card_omits_the_license_section_when_nothing_is_declared(tmp_path):
    """A permissive pack owes no copy; the section would link to a missing file."""
    card = generate_model_card(
        tmp_path,
        split_info={"recipe": "demo", "source": "acme/Demo", "license": "apache-2.0"},
        config={},
        repo_id="acme/demo-mlx",
        file_listing={},
    )
    front = _front_matter(card)

    assert front["license"] == "apache-2.0"
    assert "license_name" not in front
    assert "## License" not in card


def test_license_copy_is_listed_among_the_published_files(tmp_path):
    """It is a real file in the repo, not only a link."""
    card = generate_model_card(
        tmp_path,
        split_info=_ltx_split_info(),
        config={},
        repo_id="dgrauet/ltx-2.3-mlx",
        file_listing={"LICENSE": 12345, "transformer.safetensors": 20},
    )

    assert "- `LICENSE`" in card


# ---------------------------------------------------------------------------
# The copy itself
# ---------------------------------------------------------------------------


def test_ensure_license_file_fetches_verbatim(tmp_path, monkeypatch):
    """The text is copied byte for byte — a paraphrase is not "a copy"."""
    upstream = tmp_path / "cache" / "LICENSE"
    upstream.parent.mkdir()
    text = "LTX-2.x COMMUNITY LICENSE\n\n1.5 Derivatives ...\n"
    upstream.write_text(text)

    seen = {}

    def fake_download(*, repo_id, filename, **kwargs):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        return str(upstream)

    monkeypatch.setattr(convert, "hf_hub_download", fake_download)

    out = tmp_path / "model"
    out.mkdir()
    convert.ensure_license_file(out, _ltx_split_info())

    assert seen == {"repo_id": "Lightricks/LTX-2.3", "filename": "LICENSE"}
    assert (out / "LICENSE").read_text() == text


def test_ensure_license_file_is_a_no_op_when_none_is_declared(tmp_path, monkeypatch):
    def explode(**kwargs):  # pragma: no cover - must never run
        raise AssertionError("fetched a licence for a pack that declares none")

    monkeypatch.setattr(convert, "hf_hub_download", explode)

    assert convert.ensure_license_file(tmp_path, {"source": "acme/Demo"}) == []


def test_ensure_license_file_keeps_an_existing_copy(tmp_path, monkeypatch):
    """Re-uploading must not re-fetch, and must not clobber what is there."""

    def explode(**kwargs):  # pragma: no cover - must never run
        raise AssertionError("re-fetched a licence already present locally")

    monkeypatch.setattr(convert, "hf_hub_download", explode)

    (tmp_path / "LICENSE").write_text("already here\n")
    written = convert.ensure_license_file(tmp_path, _ltx_split_info())

    assert written == [tmp_path / "LICENSE"]
    assert (tmp_path / "LICENSE").read_text() == "already here\n"


def test_ensure_license_file_refuses_to_publish_without_one(tmp_path, monkeypatch):
    """Strict mode is what the upload path uses: no copy, no publication."""

    def fail(**kwargs):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(convert, "hf_hub_download", fail)

    with pytest.raises(SystemExit):
        convert.ensure_license_file(tmp_path, _ltx_split_info())
    assert not (tmp_path / "LICENSE").exists()


def test_ensure_license_file_does_not_destroy_a_conversion(tmp_path, monkeypatch, capsys):
    """Non-strict is what `convert` uses: warn, never lose hours of GPU work."""

    def fail(**kwargs):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(convert, "hf_hub_download", fail)

    assert convert.ensure_license_file(tmp_path, _ltx_split_info(), strict=False) == []
    assert "WARNING" in capsys.readouterr().out


def test_zero_byte_copy_is_treated_as_missing(tmp_path, monkeypatch):
    """An interrupted fetch must not pass for a licence."""
    good = tmp_path / "src"
    good.write_text("real text\n")
    monkeypatch.setattr(convert, "hf_hub_download", lambda **kw: str(good))

    (tmp_path / "LICENSE").write_bytes(b"")
    written = convert.ensure_license_file(tmp_path, _ltx_split_info())

    assert written == [tmp_path / "LICENSE"]
    assert (tmp_path / "LICENSE").read_text() == "real text\n"


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------


def test_every_recipe_materialises_its_licence(tmp_path, monkeypatch):
    """write_split_model is the choke point all ten recipes go through.

    Hooking it there is what stops a recipe from silently shipping a derivative
    without the agreement, which is exactly how the three published LTX repos
    ended up with neither license_name, license_link, nor a LICENSE file.
    """
    src = tmp_path / "src"
    src.write_text("agreement\n")
    monkeypatch.setattr(convert, "hf_hub_download", lambda **kw: str(src))

    out = tmp_path / "out"
    out.mkdir()
    convert.write_split_model(out, _ltx_split_info())

    assert (out / "LICENSE").read_text() == "agreement\n"
    manifest = json.loads((out / "split_model.json").read_text())
    assert manifest["license_name"] == "ltx-2-community-license-agreement"


def test_card_only_refresh_pushes_the_licence(tmp_path, monkeypatch):
    """--card-only is how a published repo is brought up to date.

    It pushes a card linking to `./LICENSE`; pushing that to a repo without one
    would leave a dead link where the agreement is supposed to be.
    """
    from mlx_forge import upload as upload_mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "README.md").write_text("card\n")
    (model_dir / "LICENSE").write_text("agreement\n")
    (model_dir / "split_model.json").write_text(json.dumps(_ltx_split_info()))

    pushed: list[str] = []

    class FakeApi:
        def create_repo(self, **kwargs):
            return "https://huggingface.co/dgrauet/ltx-2.3-mlx"

        def upload_file(self, *, path_in_repo, **kwargs):
            pushed.append(path_in_repo)

    upload_mod.upload_model(
        model_dir,
        # A stub standing in for HfApi: upload_model only calls the two methods
        # above, and hitting the Hub to check which files a refresh pushes would
        # defeat the point of the test.
        api=cast("HfApi", FakeApi()),
        repo_id="dgrauet/ltx-2.3-mlx",
        card_only=True,
    )

    assert pushed == ["README.md", "split_model.json", "LICENSE"]
    # What --dry-run promises must be exactly what the run performs. The two
    # stated the list independently, and the dry run went on claiming two files
    # once the licence copy made it three.
    assert upload_mod.card_only_files(model_dir) == pushed


def test_dry_run_lists_the_card_before_it_exists(tmp_path):
    """--dry-run asks before the README is written; it is still going up."""
    from mlx_forge.upload import card_only_files

    (tmp_path / "split_model.json").write_text(json.dumps(_ltx_split_info()))
    (tmp_path / "LICENSE").write_text("agreement\n")

    assert card_only_files(tmp_path) == ["README.md", "split_model.json", "LICENSE"]


def test_full_upload_includes_the_licence(tmp_path):
    """The deny-list upload must not filter it out."""
    from mlx_forge.upload import iter_model_files

    (tmp_path / "LICENSE").write_text("agreement\n")
    (tmp_path / "model.safetensors").write_bytes(b"\0")

    names = {p.name for p in iter_model_files(tmp_path)}
    assert "LICENSE" in names


def test_license_path_inside_upstream_repo_lands_at_the_root(tmp_path, monkeypatch):
    """`license_file` names a path upstream; locally it is always the basename.

    The card links to `./LICENSE`, so a nested upstream path must not produce a
    file the card cannot reach.
    """
    src = tmp_path / "src"
    src.write_text("agreement\n")
    monkeypatch.setattr(convert, "hf_hub_download", lambda **kw: str(src))

    info = {"source": "acme/Demo", "license_file": "legal/LICENSE.txt"}
    written = convert.ensure_license_file(tmp_path, info)

    assert written == [tmp_path / "LICENSE.txt"]

    card = generate_model_card(
        tmp_path,
        split_info=info,
        config={},
        repo_id="acme/demo-mlx",
        file_listing={"LICENSE.txt": 10},
    )
    assert "(./LICENSE.txt)" in card


def test_upstream_that_is_not_a_hub_repo_is_reported(tmp_path):
    """vjepa's source is a source tree, not a repo: fail loudly, not silently."""
    info = {"source": "facebookresearch/vjepa2 (app/vjepa_2_1)", "license_file": "LICENSE"}

    with pytest.raises(SystemExit):
        convert.ensure_license_file(tmp_path, info)


def test_base_model_wins_over_source_as_the_licence_origin(tmp_path, monkeypatch):
    """base_model exists precisely for when `source` is not the Hub repo."""
    src = tmp_path / "src"
    src.write_text("agreement\n")
    seen = {}

    def fake(*, repo_id, filename, **kwargs):
        seen["repo_id"] = repo_id
        return str(src)

    monkeypatch.setattr(convert, "hf_hub_download", fake)

    convert.ensure_license_file(
        tmp_path,
        {
            "source": "facebookresearch/vjepa2 (app/vjepa_2_1)",
            "base_model": "facebook/vjepa2-vitl-fpc64-256",
            "license_file": "LICENSE",
        },
    )

    assert seen["repo_id"] == "facebook/vjepa2-vitl-fpc64-256"


def test_hunyuan3d_ships_the_notice_with_the_licence(tmp_path, monkeypatch):
    """Upstream attaches a Notice.txt; passing on half of it is not passing it on."""
    metadata = load_recipe("hunyuan3d-2.1").METADATA
    assert metadata.license_name == "tencent-hunyuan-community"
    assert metadata.license_file == ("LICENSE", "Notice.txt")

    fetched = []

    def fake(*, repo_id, filename, **kwargs):
        fetched.append(filename)
        src = tmp_path / f"src-{filename}"
        src.write_text(f"{filename} text\n")
        return str(src)

    monkeypatch.setattr(convert, "hf_hub_download", fake)

    out = tmp_path / "out"
    out.mkdir()
    written = convert.ensure_license_file(out, dict(metadata.as_split_fields()))

    assert fetched == ["LICENSE", "Notice.txt"]
    assert [p.name for p in written] == ["LICENSE", "Notice.txt"]
    assert (out / "Notice.txt").read_text() == "Notice.txt text\n"

    card = generate_model_card(
        out,
        split_info=dict(metadata.as_split_fields()),
        config={},
        repo_id="dgrauet/hunyuan3d-2.1-mlx",
        file_listing={"LICENSE": 10, "Notice.txt": 10},
    )
    assert "(./LICENSE)" in card and "(./Notice.txt)" in card


def test_matrix_game_mirrors_its_permissive_upstream():
    """Upstream declares apache-2.0; "other" understated it and named nothing."""
    metadata = load_recipe("matrix-game-3.0").METADATA

    assert metadata.license == "apache-2.0"
    assert metadata.license_file is None


def test_a_partial_copy_is_completed_not_assumed_done(tmp_path, monkeypatch):
    """One of two files present must not short-circuit the other's fetch."""
    metadata = load_recipe("hunyuan3d-2.1").METADATA
    (tmp_path / "LICENSE").write_text("already here\n")

    fetched = []

    def fake(*, repo_id, filename, **kwargs):
        fetched.append(filename)
        src = tmp_path / "src"
        src.write_text("notice\n")
        return str(src)

    monkeypatch.setattr(convert, "hf_hub_download", fake)
    convert.ensure_license_file(tmp_path, dict(metadata.as_split_fields()))

    assert fetched == ["Notice.txt"]
    assert (tmp_path / "LICENSE").read_text() == "already here\n"


def test_a_corrected_licence_reaches_an_already_converted_pack(tmp_path, monkeypatch):
    """Correcting the recipe must reach packs converted before the correction.

    matrix-game-3.0's published manifest records `license: other`, written when
    the recipe said so; Skywork publishes apache-2.0. Under fill-only-what-is-
    absent that stale value was unreachable — the source fix would never have
    changed the repo.
    """
    from mlx_forge.upload import backfill_from_recipe

    stale = {
        "recipe": "matrix-game-3.0",
        "source": "Skywork/Matrix-Game-3.0",
        "license": "other",
        "components": ["dit"],
    }
    (tmp_path / "split_model.json").write_text(json.dumps(stale))

    merged = backfill_from_recipe(tmp_path, stale)

    assert merged["license"] == "apache-2.0"
    assert merged["components"] == ["dit"], "non-licence manifest content must survive"
    assert json.loads((tmp_path / "split_model.json").read_text())["license"] == "apache-2.0"


def test_the_manifest_still_wins_outside_the_licence(tmp_path):
    """Recipe-wins is narrow: it must not overwrite what describes the build."""
    from mlx_forge.upload import backfill_from_recipe

    info = {
        "recipe": "ltx-2.3",
        "source": "Lightricks/LTX-2.3",
        "usage_url": "https://example.invalid/operator-choice",
    }
    (tmp_path / "split_model.json").write_text(json.dumps(info))

    merged = backfill_from_recipe(tmp_path, info)

    assert merged["usage_url"] == "https://example.invalid/operator-choice"
    assert merged["license_name"] == "ltx-2-community-license-agreement"


def test_a_licence_field_no_longer_declared_is_dropped(tmp_path):
    """A leftover license_name would keep naming an agreement that no longer applies."""
    from mlx_forge.upload import backfill_from_recipe

    info = {
        "recipe": "matrix-game-3.0",
        "source": "Skywork/Matrix-Game-3.0",
        "license": "other",
        "license_name": "some-community-license",
    }
    (tmp_path / "split_model.json").write_text(json.dumps(info))

    merged = backfill_from_recipe(tmp_path, info)

    assert merged["license"] == "apache-2.0"
    assert "license_name" not in merged
    assert "license_name" not in json.loads((tmp_path / "split_model.json").read_text())


def test_backfill_writes_nothing_when_already_correct(tmp_path):
    """Idempotent: a refresh of an up-to-date pack must not churn the manifest."""
    from mlx_forge.upload import backfill_from_recipe

    info = dict(load_recipe("ltx-2.3").METADATA.as_split_fields())
    manifest = tmp_path / "split_model.json"

    assert backfill_from_recipe(tmp_path, info) == info
    assert not manifest.exists(), "nothing to change must mean nothing written"


def test_dry_run_reads_a_corrected_front_matter_value_as_a_change(tmp_path, monkeypatch, capsys):
    """A corrected value is not lost content, and must not warn as if it were.

    `license: other` -> `license: apache-2.0` was counted as a line disappearing.
    A warning that cries wolf on every correction is one nobody reads when it
    means it — and it is the safety net standing between a refresh and a
    degraded published card.
    """
    from mlx_forge import cli

    published = tmp_path / "published.md"
    published.write_text("---\nlibrary_name: mlx\nlicense: other\n---\n\n# demo\n")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **kw: str(published), raising=True
    )

    regenerated = "---\nlibrary_name: mlx\nlicense: apache-2.0\n---\n\n# demo\n"
    cli._show_card_diff(None, "acme/demo-mlx", regenerated, tmp_path, card_only=True)

    out = capsys.readouterr().out
    assert "No content would be lost." in out
    assert "WARNING" not in out


def test_dry_run_pairs_a_file_entry_that_gained_a_size(tmp_path, monkeypatch, capsys):
    """A hand-written card lists files bare; the template always adds a size.

    Pairing on "` (" left the two keys one backtick apart, so a file still in
    the repo was reported as content disappearing.
    """
    from mlx_forge import cli

    published = tmp_path / "published.md"
    published.write_text("---\nlicense: mit\n---\n\n## Files\n\n- `config.json`\n")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **kw: str(published), raising=True
    )

    regenerated = "---\nlicense: mit\n---\n\n## Files\n\n- `config.json` (365.00 B)\n"
    cli._show_card_diff(None, "acme/demo-mlx", regenerated, tmp_path, card_only=True)

    out = capsys.readouterr().out
    assert "No content would be lost." in out
    assert "WARNING" not in out


def test_dry_run_still_warns_when_a_line_really_disappears(tmp_path, monkeypatch, capsys):
    """The pairing must not become a blanket excuse for every removal."""
    from mlx_forge import cli

    published = tmp_path / "published.md"
    published.write_text("---\nlicense: mit\n---\n\n## Related Projects\n\n- **Code:** https://x\n")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **kw: str(published), raising=True
    )

    cli._show_card_diff(None, "acme/demo-mlx", "---\nlicense: mit\n---\n", tmp_path, card_only=True)

    assert "WARNING" in capsys.readouterr().out


def test_no_stray_license_file_left_in_the_repo():
    """mlx-forge's own tree must not accumulate fetched licence copies."""
    root = Path(__file__).resolve().parents[1]
    strays = [p for p in root.glob("src/mlx_forge/**/LICENSE") if p.is_file()]
    assert strays == []
