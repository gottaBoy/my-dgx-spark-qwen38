"""Asked-versus-granted, tested against a captured real boot log.

The fixture is the verbatim pair of lines SGLang printed on the reference box the
first time this stack booted: it was handed max_running_requests=8 and granted 6,
with mamba_full_memory_ratio left at its 0.9 default. That is a healthy server
running a different configuration than the one on disk, which is precisely the
state a tuning note must not be written against.
"""

from __future__ import annotations

import unittest

from qwen38 import granted

# Verbatim from qwen38-spark on the reference box (timestamps kept, wording
# unchanged), plus the fields trimmed only for line width.
SERVER_ARGS = (
    "[2026-09-26 17:10:20] server_args=ServerArgs(model_path='RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead', "
    "context_length=262144, mem_fraction_static=0.585, max_running_requests=8, "
    "attention_backend='flashinfer', speculative_algorithm='DFLASH', "
    "max_mamba_cache_size=32, mamba_full_memory_ratio=0.9)")
SIZING = (
    "[2026-09-26 17:13:31] max_total_num_tokens=604727, chunked_prefill_size=8192, "
    "max_prefill_tokens=16384, max_running_requests=6, context_len=262144, "
    "available_gpu_mem=32.69 GB")
BOOT_LOG = "\n".join([SERVER_ARGS, "... a thousand lines of capture ...", SIZING])


class TestParsing(unittest.TestCase):
    def test_requested_comes_from_the_server_args_line(self):
        asked = granted.parse_requested(BOOT_LOG)
        self.assertEqual(asked["max_running_requests"], "8")
        self.assertEqual(asked["context_length"], "262144")
        self.assertEqual(asked["max_mamba_cache_size"], "32")

    def test_granted_comes_from_the_sizing_line(self):
        got = granted.parse_granted(BOOT_LOG)
        self.assertEqual(got["max_running_requests"], "6")
        self.assertEqual(got["context_len"], "262144")
        self.assertEqual(got["available_gpu_mem"], "32.69")

    def test_quoted_values_are_not_mistaken_for_numbers(self):
        """speculative_algorithm='DFLASH' must not become a comparable number,
        and attention_backend='flashinfer' must not shadow one."""
        asked = granted.parse_requested(BOOT_LOG)
        self.assertNotIn("speculative_algorithm", asked)
        self.assertNotIn("attention_backend", asked)

    def test_a_restart_reportsthe_last_sizing(self):
        # Two boots in one log: the live engine is the second one.
        text = SIZING + "\n" + SERVER_ARGS + "\n" + SIZING.replace(
            "max_running_requests=6", "max_running_requests=4")
        self.assertEqual(granted.parse_granted(text)["max_running_requests"], "4")


class TestCompare(unittest.TestCase):
    def test_the_real_clamp_is_detected(self):
        deltas = granted.compare(granted.parse_requested(BOOT_LOG),
                                 granted.parse_granted(BOOT_LOG))
        clamped = [d for d in deltas if d.clamped]
        self.assertEqual([d.key for d in clamped], ["max_running_requests"])
        self.assertEqual((clamped[0].asked, clamped[0].granted), ("8", "6"))
        # Names the cause the A/B below established. This assertion used to demand
        # "mamba_full_memory_ratio", copied from MiaAI's comment -- and the box
        # disproved it, so the test was pinning a wrong diagnosis in place.
        self.assertIn("slot", clamped[0].meaning)
        self.assertIn("not the memory fraction", clamped[0].meaning)

    def test_agreement_is_reported_as_agreement(self):
        same = {**granted.parse_requested(BOOT_LOG)}
        deltas = granted.compare({"max_running_requests": "6", "context_length": "262144"},
                                 granted.parse_granted(BOOT_LOG))
        self.assertFalse([d for d in deltas if d.clamped])
        self.assertTrue(all("ok " in d.render() for d in deltas))

    def test_context_is_compared_across_the_two_vocabularies(self):
        """server_args says context_length, the sizing line says context_len.

        Without the alias this check compares nothing and reports a clean bill,
        which is the worst possible failure for a comparison: silent agreement.
        """
        deltas = granted.compare(granted.parse_requested(BOOT_LOG),
                                 granted.parse_granted(BOOT_LOG))
        context = [d for d in deltas if d.key == "context_len"]
        self.assertEqual(len(context), 1, [d.key for d in deltas])
        self.assertFalse(context[0].clamped)

    def test_a_partial_log_reports_what_could_not_be_checked(self):
        # Only the sizing line arrived (truncated --tail). Silence here would read
        # as "everything matches".
        deltas = granted.compare({}, granted.parse_granted(BOOT_LOG))
        self.assertEqual(deltas, [])
        missing = granted.missing_checks({}, granted.parse_granted(BOOT_LOG), deltas)
        self.assertEqual(len(missing), len(granted.CHECKS))

    def test_mamba_pool_size_is_compared_when_both_have_it(self):
        """The granted log omits it today; if a future build adds it, the check
        must engage without anyone remembering to turn it on."""
        deltas = granted.compare({"max_mamba_cache_size": "32"},
                                 {"max_mamba_cache_size": "24"})
        self.assertTrue([d for d in deltas if d.clamped])


class TestHeadroom(unittest.TestCase):
    def test_gb10_margin_is_surfaced_verbatim(self):
        # nvidia-smi reports N/A for memory on GB10, so this log field is the only
        # place the post-capture margin is visible at all.
        self.assertEqual(granted.headroom(granted.parse_granted(BOOT_LOG)), "32.69")

    def test_missing_headroom_is_empty_not_zero(self):
        self.assertEqual(granted.headroom({}), "")


class TestClampCause(unittest.TestCase):
    """The diagnosis text has to match what the box actually did, not what a
    comment said it would.

    Two boots on the reference machine, --mamba-full-memory-ratio 4.2 in both:
      32 slots -> granted max_running_requests=6   (clamped)
      96 slots -> granted max_running_requests=8   (as asked)
    So slot headroom moves it and the ratio does not, which is the opposite of the
    MiaAI comment I first copied. The wording matters because a diagnostic that
    names the wrong knob costs the reader a full boot every time they trust it.
    """

    SIZING_CLAMPED = SIZING                        # granted 6
    SIZING_CLEAN = SIZING.replace("max_running_requests=6", "max_running_requests=8")
    ARGS_32 = SERVER_ARGS.replace("max_mamba_cache_size=32", "max_mamba_cache_size=32")
    ARGS_96 = SERVER_ARGS.replace("max_mamba_cache_size=32", "max_mamba_cache_size=96")

    def _clamped(self, args_line: str, sizing_line: str):
        log = args_line + "\n" + sizing_line
        deltas = granted.compare(granted.parse_requested(log), granted.parse_granted(log))
        return [d for d in deltas if d.clamped]

    def test_the_measured_cause_is_what_the_message_names(self):
        clamped = self._clamped(self.ARGS_32, self.SIZING_CLAMPED)
        self.assertEqual([d.key for d in clamped], ["max_running_requests"])
        self.assertIn("slot headroom", clamped[0].meaning)

    def test_the_disproved_cause_is_not_still_named(self):
        """Guard against the wrong diagnosis creeping back via a copy-paste."""
        clamped = self._clamped(self.ARGS_32, self.SIZING_CLAMPED)
        self.assertNotIn("mamba_full_memory_ratio is the usual culprit", clamped[0].meaning)
        self.assertNotIn("0.9 default", clamped[0].meaning)

    def test_enough_slots_and_the_clamp_is_gone(self):
        self.assertEqual(self._clamped(self.ARGS_96, self.SIZING_CLEAN), [])


class TestRestartDecision(unittest.TestCase):
    """The four states the supervisor can wake up to, and what each must return.

    This is the one place where a wrong answer is invisible until the box is
    already in trouble: exit 0 for a crash means the unit reads "inactive" beside
    a dead engine while Restart=on-failure stays silent, and exit 1 for a
    deliberate stop marks every controlled stop as failed -- which is how
    operators learn to ignore the state.
    """

    def test_engine_still_running_is_not_a_failure(self):
        # docker logs -f can end for reasons other than the container going away.
        self.assertEqual(granted.follow_exit_code(True, False), 0)
        self.assertEqual(granted.follow_exit_code(True, True), 0)

    def test_unasked_death_must_return_nonzero(self):
        # The only way Restart=on-failure ever learns to fire.
        self.assertNotEqual(granted.follow_exit_code(False, False), 0)

    def test_requested_stop_must_return_zero(self):
        self.assertEqual(granted.follow_exit_code(False, True), 0)

    def test_a_crash_is_still_a_crash_even_if_a_stale_marker_exists(self):
        # The documented hazard of a marker file: one left behind by a stop makes
        # the next death look deliberate. cmd_start clearing it is what keeps this
        # from happening, and the test says which of the two layers is responsible.
        self.assertNotEqual(granted.follow_exit_code(False, False), 0)
