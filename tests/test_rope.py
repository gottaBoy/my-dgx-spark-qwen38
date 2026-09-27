"""YaRN, checked against MiaAI-Lab's start.sh rather than against itself.

This module is borrowed behaviour: the rope_parameters payload and the companion
container env var both come from a recipe that has run on this hardware. The
tests pin the parts a future edit could break without noticing -- the factor
derivation, and the pair-when-empty rule that keeps a native boot clean.
"""

from __future__ import annotations

import json
import unittest

from qwen38 import rope


class TestFactor(unittest.TestCase):
    def test_the_two_card_validated_points(self):
        # The model card validates 2.0 at 524288 and 4.0 at 1M. Anything else is
        # extrapolation and the comment in rope.py says so rather than hiding it.
        self.assertEqual(rope.derive_factor(524288), 2)
        self.assertEqual(rope.derive_factor(1000000), 4)

    def test_native_and_below_need_no_override(self):
        self.assertFalse(rope.needs_yarn(rope.NATIVE_CONTEXT))
        self.assertFalse(rope.needs_yarn(8192))
        self.assertTrue(rope.needs_yarn(rope.NATIVE_CONTEXT + 1))

    def test_factor_never_drops_below_one(self):
        self.assertEqual(rope.derive_factor(1000), 1)


class TestPayload(unittest.TestCase):
    def test_shape_matches_the_model_card_recipe(self):
        params = json.loads(rope.override_json(4))["text_config"]["rope_parameters"]
        self.assertEqual(params["rope_type"], "yarn")
        self.assertEqual(params["factor"], 4)
        self.assertEqual(params["original_max_position_embeddings"], rope.NATIVE_CONTEXT)
        self.assertEqual(params["mrope_section"], [11, 11, 10])
        self.assertTrue(params["mrope_interleaved"])
        self.assertEqual(params["rope_theta"], 10000000)
        self.assertEqual(params["partial_rotary_factor"], 0.25)

    def test_is_valid_json_as_a_single_argument(self):
        # It travels as one argv element. A stray quote here becomes a shell or
        # argparse problem at boot rather than here, which is the wrong place to
        # find out.
        json.loads(rope.override_json(2))


class TestArgs(unittest.TestCase):
    def test_native_carries_no_override_and_no_env(self):
        args, env = rope.context_args(rope.NATIVE_CONTEXT)
        self.assertEqual(args, ["--context-length", "262144"])
        self.assertEqual(env, [])

    def test_long_window_carries_both_halves(self):
        """The env var is not optional. Without it SGLang logs a warning, keeps
        262K, and the server advertises a window it cannot accept."""
        args, env = rope.context_args(1000000)
        self.assertIn("--json-model-override-args", args)
        self.assertEqual(env, [("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")])


class TestWiredIntoPlan(unittest.TestCase):
    def test_plan_emits_both_halves_only_above_native(self):
        from qwen38 import plan as plan_mod
        def argv(**over):
            base = dict(model="M", served_name="q", image_ref="I", container_name="c",
                        bind="127.0.0.1", port=1, context_length=262144, max_concurrent=8,
                        chunked_prefill=8192, mem_fraction=0.6, profile="mtp",
                        cuda_total_gib=119.7)
            base.update(over)
            return plan_mod.build(plan_mod.Inputs(**base))
        native = argv()
        self.assertNotIn("--json-model-override-args", list(native.server_args))
        self.assertFalse([a for a in native.docker_argv if "ALLOW_OVERWRITE" in a])
        long_ = argv(context_length=1000000)
        self.assertIn("--json-model-override-args", list(long_.server_args))
        self.assertTrue([a for a in long_.docker_argv if "ALLOW_OVERWRITE" in a])


if __name__ == "__main__":
    unittest.main()
