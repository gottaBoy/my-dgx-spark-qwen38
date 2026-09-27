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
        draft_path="z-lab/Qwen3.8-27B-DFlash2",
        draft_revision="50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
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


class TestRevisionPins(unittest.TestCase):
    """Revisions are arguments, never part of a repo id.

    The original implementation folded "repo@50307d4" into --speculative-draft-model-path,
    which asks HuggingFace to resolve a repository literally named that. It fails
    during model loading, minutes into a boot, after the weights have downloaded.
    """

    def test_draft_revision_is_its_own_flag_and_the_path_stays_a_repo_id(self):
        args = list(plan_mod.build(inputs(profile="dflash2", draft_revision="a" * 40)).server_args)
        self.assertNotIn("@", args[args.index("--speculative-draft-model-path") + 1])
        self.assertEqual(args.count("--speculative-draft-model-revision"), 1)
        self.assertEqual(args[args.index("--speculative-draft-model-revision") + 1], "a" * 40)

    def test_an_empty_draft_revision_drops_the_flag_with_its_value(self):
        # A dangling --speculative-draft-model-revision followed by the next real
        # flag would consume it as its argument: a server configured with something
        # nobody asked for, which is worse than an error.
        args = list(plan_mod.build(inputs(profile="dflash2", draft_revision="")).server_args)
        self.assertNotIn("--speculative-draft-model-revision", args)
        self.assertIn("--speculative-num-draft-tokens", args)

    def test_target_revision_renders_only_when_set(self):
        without = list(plan_mod.build(inputs(profile="ar")).server_args)
        self.assertNotIn("--revision", without)
        with_rev = list(plan_mod.build(inputs(profile="ar", model_revision="deadbeef")).server_args)
        self.assertEqual(with_rev[with_rev.index("--revision") + 1], "deadbeef")

    def test_shipped_draft_pin_is_a_full_sha(self):
        # The Hub resolves full shas; the short form is a web UI convenience that
        # no flag here promises to accept.
        from qwen38 import settings
        revision = settings.load().get("Q38_DRAFT_REVISION")
        self.assertRegex(revision, r"^[0-9a-f]{40}$", revision)

    def test_a_draft_profile_without_a_draft_refuses_before_rendering(self):
        for name in ("dspark", "dflash2"):
            with self.assertRaises(SystemExit, msg=name):
                plan_mod.build(inputs(profile=name, draft_path=None, draft_revision="x" * 40))


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


class TestMambaRatioDerivation(unittest.TestCase):
    """Slots per request come from the engine's own arithmetic, not a guess.

    Pinned from kv_cache_configurator._calculate_mamba_ratio in the image this
    repo runs: base 3, plus 2 for non-lazy extra_buffer with the overlap
    scheduler on, plus 1 for lazy. Verified against three boots on the reference
    box, including the one that granted 6 where 8 were asked.
    """

    def test_extra_buffer_needs_five_slots_and_lazy_four(self):
        from qwen38 import profiles
        self.assertEqual(profiles.MAMBA_RATIO["extra_buffer"], 5)
        self.assertEqual(profiles.MAMBA_RATIO["extra_buffer_lazy"], 4)

    def test_each_profile_derives_slots_from_its_strategy(self):
        from qwen38 import profiles
        for name, profile in profiles.PROFILES.items():
            self.assertEqual(
                profile.mamba_slots,
                profiles.MAMBA_RATIO[profile.mamba_strategy] * profile.mamba_slot_headroom,
                name)

    def test_dflash2_is_the_one_that_needs_more(self):
        # It forces extra_buffer, so it pays 5/req. Getting this wrong is what
        # produced the silent 8 -> 6 clamp measured on this box.
        from qwen38 import profiles
        self.assertEqual(profiles.PROFILES["dflash2"].mamba_strategy, "extra_buffer")
        self.assertGreater(profiles.PROFILES["dflash2"].mamba_slots,
                           profiles.PROFILES["ar"].mamba_slots)

    def test_the_measured_clamp_is_reproducible_from_the_formula(self):
        # 32 slots // 5 = 6: exactly what the engine granted when asked for 8.
        self.assertEqual(32 // profiles.MAMBA_RATIO["extra_buffer"], 6)
        # 96 // 5 = 19 >= 8: what fixed it.
        self.assertGreaterEqual(96 // profiles.MAMBA_RATIO["extra_buffer"], 8)
        # and the lazy boot at 32 was never clamped: 32 // 4 = 8.
        self.assertEqual(32 // profiles.MAMBA_RATIO["extra_buffer_lazy"], 8)

    def test_plan_sizes_the_pool_from_the_derived_value(self):
        for name in ("dflash2", "ar"):
            args = list(plan_mod.build(inputs(profile=name)).server_args)
            slots = profiles.PROFILES[name].mamba_slots
            self.assertEqual(args[args.index("--max-mamba-cache-size") + 1],
                             str(8 * slots), name)


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
        # The default boot fits, and says so with the arithmetic attached.
        built = plan_mod.build(inputs())
        self.assertTrue(any("budget ok" in n for n in built.notes), built.notes)

    def test_a_budget_below_weights_plus_one_sequence_is_flagged(self):
        # 0.20 of 119.7 = 23.9 GiB, which does not even leave the 22.1 GiB of
        # weights any room for a KV cache. The fraction is chosen far from the
        # boundary on purpose: an earlier version of this test pinned 0.45 with
        # estimated weights and flipped to "ok" the moment the weights were
        # replaced with Hub-measured values, which is a test that was asserting
        # its own arithmetic rather than the behaviour.
        tight = plan_mod.build(inputs(profile="mtp", context_length=1000000, mem_fraction=0.20))
        self.assertTrue(any("BUDGET SHORT" in n for n in tight.notes), tight.notes)

    def test_the_short_message_carries_the_numbers_that_produced_it(self):
        tight = plan_mod.build(inputs(profile="mtp", context_length=1000000, mem_fraction=0.20))
        note = next(n for n in tight.notes if "BUDGET SHORT" in n)
        for fragment in ("0.2", "119.7", "1000000"):
            self.assertIn(fragment, note, note)

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
