"""process_component's four factorisation parameters, each at the mutation-probe standard."""

from __future__ import annotations

import json
import struct

import mlx.core as mx
import pytest

from mlx_forge.convert import process_component


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _run(tmp_path, weights, **kwargs):
    kwargs.setdefault("sanitizer", lambda k: k)
    process_component(
        weights, "comp", list(weights), tmp_path, kwargs.pop("prefix", "comp"), **kwargs
    )
    return mx.load(str(tmp_path / "comp.safetensors"))


class TestPrefix:
    def test_none_emits_bare_keys(self, tmp_path):
        out = _run(tmp_path, {"a.w": mx.zeros((2, 2))}, prefix=None)
        assert set(out) == {"a.w"}

    def test_default_still_prefixes(self, tmp_path):
        out = _run(tmp_path, {"a.w": mx.zeros((2, 2))})
        assert set(out) == {"comp.a.w"}


class TestDtype:
    def test_fixed_dtype_casts_everything(self, tmp_path):
        out = _run(tmp_path, {"a": mx.zeros((2, 2), dtype=mx.float32)}, dtype=mx.float16)
        assert out["comp.a"].dtype == mx.float16

    def test_predicate_casts_selectively(self, tmp_path):
        weights = {
            "a": mx.zeros((2, 2), dtype=mx.float32),
            "b": mx.zeros((2, 2), dtype=mx.bfloat16),
        }
        out = _run(
            tmp_path, weights, dtype=lambda k, w: mx.bfloat16 if w.dtype == mx.float32 else None
        )
        assert out["comp.a"].dtype == mx.bfloat16 and out["comp.b"].dtype == mx.bfloat16

    def test_none_keeps_dtype(self, tmp_path):
        out = _run(tmp_path, {"a": mx.zeros((2,), dtype=mx.float32)})
        assert out["comp.a"].dtype == mx.float32

    def test_cast_happens_after_transform(self, tmp_path):
        # A transform that up-casts must be overridden by the dtype policy, not the reverse.
        out = _run(
            tmp_path,
            {"a": mx.zeros((2, 2), dtype=mx.float16)},
            transform=lambda k, w, c: w.astype(mx.float32),
            dtype=mx.float16,
        )
        assert out["comp.a"].dtype == mx.float16

    def test_bad_predicate_result_is_a_typeerror_naming_the_key(self, tmp_path):
        with pytest.raises(TypeError, match="comp.*'a'"):
            _run(tmp_path, {"a": mx.zeros((2,))}, dtype=lambda k, w: "float16")

    def test_static_nonsense_dtype_is_a_typeerror_naming_the_component(self, tmp_path):
        with pytest.raises(TypeError, match="comp"):
            _run(tmp_path, {"a": mx.zeros((2,))}, dtype="float16")


class TestLoadWeight:
    def test_raw_values_go_through_the_adapter(self, tmp_path):
        import numpy as np

        out = _run(
            tmp_path, {"a": np.ones((2, 2), dtype=np.float32)}, load_weight=lambda v: mx.array(v)
        )
        assert out["comp.a"].dtype == mx.float32 and float(out["comp.a"].sum()) == 4.0

    def test_non_mx_value_without_adapter_fails_early_and_named(self, tmp_path):
        import numpy as np

        with pytest.raises(TypeError, match="comp"):
            _run(tmp_path, {"a": np.ones((2, 2))})


class TestFinalize:
    def test_sees_the_whole_dict_and_may_replace_it(self, tmp_path):
        def fuse(d):
            return {"comp.fused": mx.concatenate([d["comp.q"], d["comp.k"]], axis=0)}

        out = _run(tmp_path, {"q": mx.zeros((1, 2)), "k": mx.ones((1, 2))}, finalize=fuse)
        assert set(out) == {"comp.fused"} and out["comp.fused"].shape == (2, 2)

    def test_non_dict_result_is_a_typeerror(self, tmp_path):
        with pytest.raises(TypeError, match="finalize"):
            _run(tmp_path, {"a": mx.zeros((2,))}, finalize=lambda d: list(d))


def test_metadata_is_still_carried(tmp_path):
    _run(tmp_path, {"a": mx.zeros((2,))}, metadata={"model_version": "9"})
    assert _header(tmp_path / "comp.safetensors")["__metadata__"] == {"model_version": "9"}
