"""Parity between what a recipe's sanitizers/policies do to the UPSTREAM keys and
what the PUBLISHED pack actually holds — checked from fixtures, no conversion.

This is the executable definition of "refactor pur" for the process_component
factorisation: green before a recipe migrates, green after, or the migration
changed the emitted files.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

DtypeMap = Callable[[str], str]


def keep_dtype(name: str) -> str:
    return name


def cast_all_to(target: str) -> DtypeMap:
    return lambda name: target


def cast_f32_to_bf16(name: str) -> str:
    return "BF16" if name == "F32" else name


@dataclass(frozen=True)
class ComponentParity:
    recipe: str
    component: str
    upstream_fixture: str  # e.g. "upstream/void-model.json"
    upstream_file: str  # record key inside that fixture
    published_fixture: str  # e.g. "published/void-model-mlx.json"
    published_file: str
    sanitizer: Callable[[str], str | None]
    prefix: str | None  # None: the pack holds bare keys
    dtype_map: DtypeMap = keep_dtype
    finalize_keys: Callable[[set[str]], set[str]] | None = (
        None  # name-level rewrite, e.g. QKV fusion
    )
    complete_upstream_keys: Callable[[set[str]], set[str]] | None = (
        None  # applied to the raw upstream keys BEFORE the sanitizer runs, so any
        #  reconstructed sibling still goes through the real sanitizer instead of
        #  hardcoding its output name — see finalize_keys vs. complete_upstream_keys
        #  in task-4-report.md
    )
    sanitizer_drops_keys: bool = False


def _record(fixture: str, file: str) -> dict:
    data = json.loads((FIXTURES / fixture).read_text())
    assert file in data, f"{fixture} has no record for {file}: {sorted(data)}"
    return data[file]


def check_parity(spec: ComponentParity) -> None:
    upstream = _record(spec.upstream_fixture, spec.upstream_file)
    published = _record(spec.published_fixture, spec.published_file)
    derived = upstream.get("source") == "derived-from-published"

    raw_keys = set(upstream["keys"])
    if spec.complete_upstream_keys is not None:
        raw_keys = spec.complete_upstream_keys(raw_keys)

    emitted = set()
    for key in raw_keys:
        new_key = spec.sanitizer(key)
        if new_key is None:
            continue
        emitted.add(new_key if spec.prefix is None else f"{spec.prefix}.{new_key}")
    if spec.finalize_keys is not None:
        emitted = spec.finalize_keys(emitted)

    expected = set(published["keys"])
    missing = sorted(expected - emitted)
    extra = sorted(emitted - expected)
    assert not missing and not extra, (
        f"{spec.recipe}/{spec.component}: key set differs from the published pack\n"
        f"  missing from emitted: {missing[:10]}\n  extra in emitted: {extra[:10]}"
    )

    if derived:
        return  # no real upstream record: the key check above is all this fixture can support

    mapped: dict[str, int] = {}
    for name, count in upstream["dtypes"].items():
        target = spec.dtype_map(name)
        mapped[target] = mapped.get(target, 0) + count
    exact_counts = not spec.sanitizer_drops_keys and spec.finalize_keys is None
    if exact_counts:
        assert mapped == published["dtypes"], (
            f"{spec.recipe}/{spec.component}: dtype histogram {mapped} "
            f"!= published {published['dtypes']}"
        )
        assert upstream["tensor_count"] == published["tensor_count"]
    else:
        assert set(published["dtypes"]) <= set(mapped), (
            f"{spec.recipe}/{spec.component}: published dtypes {set(published['dtypes'])} "
            f"not all producible by the policy ({set(mapped)})"
        )
        assert published["tensor_count"] <= upstream["tensor_count"]
