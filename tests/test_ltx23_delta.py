"""Tests for LTX-2.3 delta-mode convert/validate."""

import argparse

from mlx_forge.recipes.ltx_23 import add_convert_args


class TestSkipSharedFlag:
    def test_default_is_false(self):
        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        args = parser.parse_args([])
        assert args.skip_shared is False

    def test_flag_sets_true(self):
        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        args = parser.parse_args(["--skip-shared"])
        assert args.skip_shared is True
