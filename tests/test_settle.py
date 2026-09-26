"""The boot-settle race, tested without anyone sleeping.

MemAvailable at boot is at its temporary peak while neighbours are still
starting. A fraction fitted then over-asks and the box dies later at CUDA graph
capture. These tests hold the clock and the sleeper still so the deadline logic
is checked in microseconds.
"""

from __future__ import annotations

import unittest

from qwen38 import settle


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeReads:
    """Serve a scripted series.

    cycle=False repeats the last value, which is how you test a box that settles.
    cycle=True loops the series forever, which is how you test one that does not:
    a scripted series that ends at a steady value settles by construction, and a
    "never settles" test written that way passes for the wrong reason.
    """

    def __init__(self, values, cycle: bool = False):
        self.values = list(values)
        self.calls = 0
        self.cycle = cycle

    def __call__(self):
        if self.cycle:
            index = self.calls % len(self.values)
        else:
            index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return self.values[index]


class TestEvaluate(unittest.TestCase):
    KW = dict(max_drift_gib=2.0, stable_samples=3)

    def test_needs_the_whole_window_before_asking_for_drift(self):
        verdict = settle.evaluate([90.0, 89.0], **self.KW)
        self.assertFalse(verdict.settled)
        # The partial window comes back rather than nothing: a caller reporting
        # "not settled yet" should be able to say what it has seen.
        self.assertEqual(verdict.samples, (90.0, 89.0))
        self.assertEqual(verdict.drift_gib, float("inf"))

    def test_drift_measured_over_the_trailing_window_only(self):
        # An hour-old excursion must not hold the verdict; what matters is whether
        # the number we are about to use is still moving.
        verdict = settle.evaluate([10.0, 11.0, 90.0, 90.4, 90.2], **self.KW)
        self.assertTrue(verdict.settled)
        self.assertAlmostEqual(verdict.drift_gib, 0.4, places=1)   # 90.4 - 90.0
        self.assertEqual(verdict.samples, (90.0, 90.4, 90.2))

    def test_still_moving_is_not_settled(self):
        verdict = settle.evaluate([95.0, 90.0, 85.0], **self.KW)
        self.assertFalse(verdict.settled)
        self.assertAlmostEqual(verdict.drift_gib, 10.0, places=1)

    def test_boundary_drift_is_settled(self):
        self.assertTrue(settle.evaluate([90.0, 88.0, 90.0], **self.KW).settled)
        self.assertFalse(settle.evaluate([90.0, 87.9, 90.0], **self.KW).settled)


class TestWaitUntilStable(unittest.TestCase):
    KW = dict(max_drift_gib=2.0, stable_samples=3, interval_s=10, deadline_s=300)

    def _run(self, values, **over):
        clock = FakeClock()
        cycle = over.pop("cycle", False)
        reads = FakeReads(values, cycle=cycle)
        kwargs = {**self.KW, **over}
        verdict, spent = settle.wait_until_stable(
            reads, sleep=clock.sleep, clock=clock.monotonic, **kwargs)
        return verdict, spent, reads, clock

    def test_a_steady_box_settles_after_exactly_the_window(self):
        verdict, spent, reads, _ = self._run([90.0])
        self.assertTrue(verdict.settled)
        self.assertEqual(reads.calls, 3)
        # Two intervals, not three: the third sample ends the wait.
        self.assertAlmostEqual(spent, 20.0, places=6)

    def test_a_settling_box_waits_for_the_early_move(self):
        verdict, spent, reads, _ = self._run([110.0, 105.0, 95.0, 94.8, 95.1, 95.0])
        self.assertTrue(verdict.settled)
        # Settles the moment a window qualifies, not when the script runs out:
        # (95.0, 94.8, 95.1) is steady at the fifth read.
        self.assertEqual(reads.calls, 5)
        self.assertAlmostEqual(spent, 40.0, places=6)

    def test_a_box_that_never_settles_hits_the_deadline_and_says_so(self):
        verdict, spent, _, _ = self._run([90.0, 60.0, 30.0], deadline_s=100, cycle=True)
        self.assertFalse(verdict.settled)
        self.assertTrue(verdict.timed_out)
        self.assertGreaterEqual(spent, 100.0)

    def test_timeout_is_reported_not_silently_treated_as_stable(self):
        # The failure that would boot the engine on a lying number.
        verdict, _, _, _ = self._run([90.0, 10.0, 70.0], deadline_s=50, cycle=True)
        self.assertFalse(verdict.settled)
        self.assertTrue(verdict.timed_out)

    def test_a_sawtooth_never_settles_however_long_it_runs(self):
        # Drift is measured inside the window, so a swing bigger than the
        # tolerance must keep refusing at every window position.
        verdict, spent, _, _ = self._run([90.0, 80.0], deadline_s=1000, cycle=True)
        self.assertFalse(verdict.settled)
        self.assertAlmostEqual(spent, 1000.0, places=1)

    def test_first_read_failing_is_none_not_a_zero_sample(self):
        verdict, _, reads, _clock = self._run([None])
        self.assertIsNone(verdict)
        self.assertEqual(reads.calls, 1)

    def test_read_failing_mid_series_keeps_what_was_seen(self):
        verdict, _, _, _ = self._run([90.0, 89.5, None])
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict.settled)
        self.assertEqual(verdict.samples, (90.0, 89.5))

    def test_a_raising_reader_is_tolerated_as_a_failed_read(self):
        clock = FakeClock()

        def reader():
            raise OSError("/proc unreadable")

        verdict, _ = settle.wait_until_stable(reader, sleep=clock.sleep,
                                             clock=clock.monotonic, **self.KW)
        self.assertIsNone(verdict)

    def test_deadline_of_zero_still_returns_a_verdict(self):
        # No infinite loop when someone configures an impatient deadline.
        verdict, spent, _, _ = self._run([90.0, 80.0, 70.0], deadline_s=0)
        self.assertFalse(verdict.settled)
        self.assertTrue(verdict.timed_out)


if __name__ == "__main__":
    unittest.main()
