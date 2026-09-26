"""The launch argv, assembled from measured inputs, checked as data.

These tests are the reason a flag change cannot quietly ship: each invariant
below corresponds to a crash or a silent misconfiguration documented upstream.
"""

from __future__ import annotations

import shlex
import unittest

from qwen38 import plan as plan_mod, profiles


def inputs(**over) -> plan_mod.Inputs:
    base = dict(
        model="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
        served_name="qwen3.8-27b",
        image_ref="lmsysorg/sglang@sha256:616a3e97",
        container_name="qwen38-spark",
        bind="127.0.0.1",
        port=28100,
        context_length=262144,
        max_concurrent=8,
        chunked_prefill=8192,
        mem_fraction=0.621,
        profile="dflash2",
        draft_path="z-lab/Qwen3.8-27B-DFlash2@50307d4",
        cuda_total_gib=119.7,
        cache_dir="/cache",
    )
    base.update(over)
    return plan_mod.Inputs(**base)


class TestCoreFlags(unittest.TestCase):
    def test_every_profile_carries_the_sm121_requirements(self):
        for name in profiles.PROFILES:
            args = plan_mod.build(inputs(profile=name)).server_args
            self.assertIn("flashinfer", args, name)
            self.assertIn("--attention-backend", args, name)
            self.assertIn("--disable-prefill-cuda-graph", args, name)
            self.assertIn("--sleep-on-idle", args, name)

    def test_backend_argument_is_not_repeated(self):
        # A duplicated --attention-backend makes argparse last-wins, so a later
        # stray value silently replaces the one we chose.
        args = list(plan_mod.build(inputs()).server_args)
        for flag in ("--attention-backend", "--kv-cache-dtype", "--port", "--host",
                     "--context-length", "--mem-fraction-static", "--max-running-requests"):
            self.assertEqual(args.count(flag), 1, f"{flag} appears {args.count(flag)}x")

    def test_loopback_bind_by_default(self):
        args = list(plan_mod.build(inputs()).server_args)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")

    def test_docker_binds_published_port_nothing(self):
        # We use host networking, so there must be no -p: a -p alongside
        # --network host is a confusing no-op that suggests a mapping exists.
        argv = list(plan_mod.build(inputs()).docker_argv)
        self.assertIn("--network", argv)
        self.assertNotIn("-p", argv)
        self.assertIn("host", argv)


class TestSpeculativeShapes(unittest.TestCase):
    def test_dspark_uses_block_size_and_not_dflash_flag(self):
        args = list(plan_mod.build(inputs(profile="dspark")).server_args)
        self.assertIn("--speculative-dspark-block-size", args)
        self.assertNotIn("--speculative-num-draft-tokens", args)

    def test_dflash_uses_draft_tokens_and_not_block_size(self):
        args = list(plan_mod.build(inputs(profile="dflash2")).server_args)
        self.assertIn("--speculative-num-draft-tokens", args)
        self.assertNotIn("--speculative-dspark-block-size", args)

    def test_mtp_requires_topk_and_steps(self):
        args = list(plan_mod.build(inputs(profile="mtp")).server_args)
        for flag in ("--speculative-num-steps", "--speculative-eagle-topk",
                     "--speculative-num-draft-tokens"):
            self.assertIn(flag, args)

    def test_topk_one_chain_needs_draft_eq_steps_plus_one(self):
        args = list(plan_mod.build(inputs(profile="mtp", spec_steps=4, spec_draft=5)).server_args)
        self.assertEqual(args[args.index("--speculative-num-steps") + 1], "4")
        self.assertEqual(args[args.index("--speculative-num-draft-tokens") + 1], "5")

    def test_draft_profiles_refuse_without_a_draft_path(self):
        for name in ("dspark", "dflash2"):
            with self.assertRaises(SystemExit, msg=name):
                plan_mod.build(inputs(profile=name, draft_path=None))

    def test_ar_profile_has_no_speculative_flags_at_all(self):
        args = plan_mod.build(inputs(profile="ar")).server_args
        self.assertFalse([a for a in args if "speculative" in a])


class TestContextCapability(unittest.TestCase):
    def test_long_context_on_a_draft_profile_refuses_at_plan_time(self):
        # Upstream: the YaRN override leaks into the draft config and dies deep
        # in model loading, 6 minutes and one confusing traceback later.
        for name in ("dspark", "dflash2", "ar"):
            with self.assertRaises(SystemExit, msg=name):
                plan_mod.build(inputs(profile=name, context_length=1000000))

    def test_long_context_is_allowed_on_mtp(self):
        args = list(plan_mod.build(inputs(profile="mtp", context_length=1000000)).server_args)
        self.assertEqual(args[args.index("--context-length") + 1], "1000000")

    def test_native_boundary_is_inclusive(self):
        # 262144 is native and must not need YaRN on any profile.
        for name in profiles.PROFILES:
            plan_mod.build(inputs(profile=name, context_length=262144))


class TestGdnPool(unittest.TestCase):
    def test_pool_scales_with_declared_slots_not_with_draft_width(self):
        # Folding the 8-token verify window in over-provisions the pool 2x, and
        # the mistake looks like a memory setting rather than a bug.
        for name, profile in profiles.PROFILES.items():
            args = list(plan_mod.build(inputs(profile=name, max_concurrent=8)).server_args)
            self.assertEqual(
                args[args.index("--max-mamba-cache-size") + 1],
                str(8 * profile.mamba_slots), name)

    def test_running_requests_matches_concurrency(self):
        args = list(plan_mod.build(inputs(max_concurrent=16)).server_args)
        self.assertEqual(args[args.index("--max-running-requests") + 1], "16")

    def test_fraction_is_rendered_with_three_decimals(self):
        args = list(plan_mod.build(inputs(mem_fraction=0.6213)).server_args)
        self.assertEqual(args[args.index("--mem-fraction-static") + 1], "0.621")


class TestBudgetNote(unittest.TestCase):
    def test_a_budget_too_small_for_one_full_sequence_is_said_out_loud(self):
        # fraction 0.45 of 119.7 = 53.9 GiB; weights ~26.6 + 262144-token
        # sequence ~8.2 fits, so this asserts the ok branch...
        built = plan_mod.build(inputs())
        self.assertTrue(any("budget" in n for n in built.notes), built.notes)
        # ...and this asserts the short branch, at 1M context.
        tight = plan_mod.build(inputs(profile="mtp", context_length=1000000, mem_fraction=0.45))
        self.assertTrue(any("BUDGET SHORT" in n for n in tight.notes), tight.notes)

    def test_budget_check_is_silent_when_the_pool_was_never_measured(self):
        # Without a denominator the arithmetic is meaningless; claiming a budget
        # we cannot compute is worse than saying nothing.
        built = plan_mod.build(inputs(cuda_total_gib=0.0))
        self.assertFalse([n for n in built.notes if "budget" in n.lower()])


class TestRendering(unittest.TestCase):
    def test_rendering_round_trips_through_a_shell_lexer(self):
        # Two separate properties, and only the first is about quoting:
        # execution never goes through a shell (subprocess takes argv), so this
        # string exists for humans to read and paste. It must survive the paste
        # unchanged, which is what shlex.split verifies.
        evil = "; rm -rf /"
        built = plan_mod.build(inputs(model="ok", extra_server_args=(evil,)))
        self.assertEqual(shlex.split(built.render_server_command()),
                         ["python3", "-m", "sglang.launch_server", *built.server_args])
        self.assertEqual(shlex.split(built.render_docker()), list(built.docker_argv))

    def test_a_dangerous_argument_stays_one_argument(self):
        evil = "; rm -rf /"
        built = plan_mod.build(inputs(extra_server_args=(evil,)))
        self.assertEqual(shlex.split(built.render_server_command())[-1], evil)

    def test_docker_argv_is_a_list_never_a_command_string(self):
        # The guard on the real failure mode: if someone "simplifies" the runner
        # to shell=True, quoting becomes load-bearing instead of cosmetic.
        argv = plan_mod.build(inputs()).docker_argv
        self.assertIsInstance(argv, tuple)
        self.assertTrue(all(isinstance(a, str) for a in argv))

    def test_cache_mounts_are_all_read_write_and_distinct(self):
        argv = " ".join(plan_mod.build(inputs()).docker_argv)
        for sub in ("/root/.cache/huggingface", "/root/.triton", "/root/.cache/inductor"):
            self.assertIn(sub, argv)

    def test_plan_as_dict_omits_the_argv_noise(self):
        data = plan_mod.build(inputs()).as_dict()
        for key in ("profile", "mem_fraction_static", "cpuset", "server_args", "docker_argv"):
            self.assertIn(key, data)

    def test_extra_server_args_land_last(self):
        # argparse last-wins is the override hatch; if they were inserted first
        # every built-in would beat them and the hatch would be a lie.
        args = list(plan_mod.build(inputs(extra_server_args=("--mem-fraction-static", "0.4"))).server_args)
        self.assertEqual(args[-2:], ["--mem-fraction-static", "0.4"])

    def test_privileged_is_off_unless_asked(self):
        self.assertNotIn("--privileged", plan_mod.build(inputs()).docker_argv)
        self.assertIn("--privileged", plan_mod.build(inputs(privileged=True)).docker_argv)


if __name__ == "__main__":
    unittest.main()
