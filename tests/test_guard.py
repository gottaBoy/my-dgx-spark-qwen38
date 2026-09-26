"""The watchdog is the only OOM protection on a box with oomd and earlyoom off.

Its contract is asymmetric on purpose: warn cheaply, trip only on sustained
pressure, and when it does trip it stops our container and nothing else.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from tests import fixtures
from qwen38 import guard


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class TestSampleParsing(unittest.TestCase):
    def test_reads_both_psi_lines(self):
        sample = guard.Sample.parse(fixtures.PSI_UNDER_PRESSURE, fixtures.MEMINFO_HEALTHY)
        self.assertAlmostEqual(sample.psi_some_avg10, 41.2)
        self.assertAlmostEqual(sample.psi_full_avg10, 7.8)
        self.assertAlmostEqual(sample.available_gib, 90.45, places=1)

    def test_idle_box_is_quiet(self):
        sample = guard.Sample.parse(fixtures.PSI_IDLE, fixtures.MEMINFO_HEALTHY)
        self.assertEqual(sample.psi_some_avg10, 0.0)
        self.assertEqual(sample.psi_full_avg10, 0.0)

    def test_swap_usage_is_reported(self):
        sample = guard.Sample.parse(fixtures.PSI_IDLE, fixtures.MEMINFO_HEALTHY)
        self.assertAlmostEqual(sample.swap_used_gib, (16777212 - 13096264) / 1024 / 1024, places=2)

    def test_missing_psi_degrades_to_zero_not_crash(self):
        # A kernel without CONFIG_PSI_DEFAULT_DISABLED still has MemAvailable,
        # so the absolute floor keeps working. Partial guard beats no guard.
        sample = guard.Sample.parse("", fixtures.MEMINFO_HEALTHY)
        self.assertEqual(sample.psi_some_avg10, 0.0)
        self.assertGreater(sample.available_gib, 0)

    def test_malformed_psi_values_do_not_crash(self):
        sample = guard.Sample.parse("some avg10=abc\nfull avg10=\n", fixtures.MEMINFO_HEALTHY)
        self.assertEqual(sample.psi_some_avg10, 0.0)


class TestTripLogic(unittest.TestCase):
    KW = dict(psi_some=25.0, psi_full=5.0, available_floor_gib=8.0, strikes=2)

    def test_healthy_stays_ok(self):
        state = guard.Guard(guard.Thresholds(**self.KW))
        verdict, reason = state.decide(guard.Sample(0.0, 0.0, 90.0))
        self.assertEqual((verdict, reason), ("ok", ""))

    def test_single_breach_warns_and_does_not_trip(self):
        state = guard.Guard(guard.Thresholds(**self.KW))
        verdict, reason = state.decide(guard.Sample(40.0, 9.0, 2.0))
        self.assertEqual(verdict, "warn")
        self.assertIn("strike 1/2", reason)

    def test_second_consecutive_breach_trips(self):
        state = guard.Guard(guard.Thresholds(**self.KW))
        bad = guard.Sample(40.0, 9.0, 2.0)
        state.decide(bad)
        verdict, reason = state.decide(bad)
        self.assertEqual(verdict, "trip")
        self.assertIn("2 consecutive", reason)

    def test_recovery_resets_strikes(self):
        # Without this, a box that grazes the line twice in an hour would trip on
        # the second, unrelated blip -- which is how operators start disabling
        # guards.
        state = guard.Guard(guard.Thresholds(**self.KW))
        bad = guard.Sample(40.0, 9.0, 2.0)
        state.decide(bad)
        self.assertEqual(state.decide(guard.Sample(0.0, 0.0, 90.0))[0], "ok")
        self.assertEqual(state.strikes, 0)
        self.assertEqual(state.decide(bad)[0], "warn")

    def test_low_available_alone_trips_even_at_zero_psi(self):
        # The lethal case: PSI stays at 0 with a few hundred MB left right up
        # until the box dies, because pressure only appears once tasks stall.
        state = guard.Guard(guard.Thresholds(**self.KW))
        self.assertEqual(state.decide(guard.Sample(0.0, 0.0, 7.9))[0], "warn")
        self.assertEqual(state.decide(guard.Sample(0.0, 0.0, 7.9))[0], "trip")

    def test_psi_full_below_some_threshold_still_counts(self):
        state = guard.Guard(guard.Thresholds(**self.KW))
        verdict, reason = state.decide(guard.Sample(3.0, 6.0, 90.0))
        self.assertEqual(verdict, "warn")
        self.assertIn("PSI full", reason)

    def test_boundary_is_inclusive(self):
        state = guard.Guard(guard.Thresholds(**self.KW))
        self.assertEqual(state.decide(guard.Sample(25.0, 0.0, 90.0))[0], "warn")
        self.assertEqual(state.decide(guard.Sample(24.9, 0.0, 90.0))[0], "ok")

    def test_reason_names_the_signal_that_fired(self):
        state = guard.Guard(guard.Thresholds(**self.KW))
        _, reason = state.decide(guard.Sample(0.0, 0.0, 1.0))
        self.assertIn("MemAvailable", reason)


class TestReadSample(unittest.TestCase):
    def test_reads_supplied_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            psi = os.path.join(tmp, "psi")
            mem = os.path.join(tmp, "meminfo")
            _write(psi, fixtures.PSI_IDLE)
            _write(mem, fixtures.MEMINFO_HEALTHY)
            sample = guard.read_sample(psi, mem)
        self.assertEqual(sample.psi_some_avg10, 0.0)
        self.assertGreater(sample.available_gib, 80)

    def test_absent_psi_file_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = os.path.join(tmp, "meminfo")
            _write(mem, fixtures.MEMINFO_HEALTHY)
            sample = guard.read_sample(os.path.join(tmp, "nope"), mem)
        self.assertEqual(sample.psi_some_avg10, 0.0)


if __name__ == "__main__":
    unittest.main()
