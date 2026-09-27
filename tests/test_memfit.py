"""Memory budget: units, clamps, and refusal. The units case exists because I
wrote the bug -- dividing kB by GiB made MemAvailable read 0.09 on a healthy box,
which would have refused every launch. That is what a test is for.
"""

from __future__ import annotations

import unittest

from tests import fixtures
from qwen38 import memfit


class TestParsing(unittest.TestCase):
    def test_kb_are_not_bytes(self):
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_HEALTHY)
        self.assertAlmostEqual(host.total_gib, 119.67, places=2)
        self.assertAlmostEqual(host.available_gib, 90.45, places=1)

    def test_swap_is_parsed_in_gib(self):
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_HEALTHY)
        self.assertAlmostEqual(host.swap_used_gib, (16777212 - 13096264) / 1024**2, places=2)

    def test_starved_box_reads_small_not_zero(self):
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_STARVED)
        self.assertAlmostEqual(host.available_gib, 2.0, places=1)

    def test_missing_field_is_an_error_not_a_default(self):
        with self.assertRaises(ValueError):
            memfit.HostMemory.from_meminfo("MemTotal: 1024 kB\n")

    def test_lines_without_kib_unit_are_ignored(self):
        # /proc/meminfo has hugepage lines with no unit on some kernels.
        text = fixtures.MEMINFO_HEALTHY + "HugePages_Total:   0\nHugepagesize:    2048 kB\n"
        host = memfit.HostMemory.from_meminfo(text)
        self.assertAlmostEqual(host.available_gib, 90.45, places=1)


class TestSolve(unittest.TestCase):
    KW = dict(min_fraction=0.45, max_fraction=0.72)

    def test_healthy_shared_box_lands_in_the_band(self):
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_HEALTHY)
        fit = memfit.solve(host, 119.7, reserved_gib=16, **self.KW)
        self.assertTrue(0.45 <= fit.fraction <= 0.72, fit.fraction)
        self.assertEqual(fit.bound_by, "askable")
        self.assertFalse(fit.clamped)

    def test_fraction_never_exceeds_the_ceiling(self):
        # A dedicated box with nothing else running would happily take 0.95.
        fit = memfit.solve(memfit.HostMemory(119.67, 118.0), 119.7, reserved_gib=4, **self.KW)
        self.assertEqual(fit.fraction, 0.72)
        self.assertEqual(fit.bound_by, "max_fraction")
        self.assertTrue(fit.clamped)

    def test_reservation_is_subtracted_before_dividing(self):
        host = memfit.HostMemory(100.0, 60.0)
        tight = memfit.solve(host, 100.0, reserved_gib=50, **self.KW)
        loose = memfit.solve(host, 100.0, reserved_gib=10, **self.KW)
        self.assertLess(tight.fraction, loose.fraction)
        self.assertEqual(tight.askable_gib, 10.0)

    def test_no_headroom_refuses_instead_of_clamping_to_floor(self):
        # Clamping up to min_fraction here would launch a container that thrashes
        # the neighbours, which is exactly the failure this repo exists to avoid.
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_STARVED)
        fit = memfit.solve(host, 119.7, reserved_gib=16, **self.KW)
        self.assertEqual(fit.fraction, 0.0)
        self.assertEqual(fit.bound_by, "no-headroom")
        self.assertTrue(fit.warnings)

    def test_unmeasured_pool_falls_back_loudly(self):
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_HEALTHY)
        fit = memfit.solve(host, None, reserved_gib=16, **self.KW)
        self.assertEqual(fit.cuda_total_source, "unmeasured-fallback")
        self.assertTrue(any("not measured" in w for w in fit.warnings))

    def test_zero_pool_is_treated_as_unmeasured(self):
        fit = memfit.solve(memfit.HostMemory(119.67, 90.0), 0, reserved_gib=16, **self.KW)
        self.assertEqual(fit.bound_by, "fallback")

    def test_below_floor_refuses_instead_of_spending_reserved_headroom(self):
        # 30 available, 16 reserved, 119.7 pool -> 0.117, below the floor.
        fit = memfit.solve(memfit.HostMemory(119.67, 30.0), 119.7, reserved_gib=16, **self.KW)
        self.assertEqual(fit.fraction, 0.0)
        self.assertEqual(fit.bound_by, "no-headroom")
        self.assertTrue(any("floor" in w for w in fit.warnings))
        self.assertTrue(any("Refusing" in w for w in fit.warnings))

    def test_rounding_cannot_silently_exceed_the_budget(self):
        host = memfit.HostMemory(119.67, 90.45)
        fit = memfit.solve(host, 119.7, reserved_gib=16, **self.KW)
        self.assertLessEqual(fit.fraction * fit.cuda_total_gib, fit.askable_gib + 1e-9)

    def test_fit_carries_every_input_that_produced_it(self):
        # The provenance is the point: a bare number is not reproducible.
        host = memfit.HostMemory.from_meminfo(fixtures.MEMINFO_HEALTHY)
        fit = memfit.solve(host, 119.7, reserved_gib=16, **self.KW).as_dict()
        for key in ("fraction", "cuda_total_gib", "available_gib", "reserved_gib",
                    "askable_gib", "bound_by"):
            self.assertIn(key, fit)

    def test_old_swap_is_reported_without_reducing_the_budget(self):
        plain = memfit.solve(memfit.HostMemory(119.67, 90.0), 119.7, reserved_gib=32, **self.KW)
        swapped = memfit.solve(memfit.HostMemory(119.67, 90.0, 12.0), 119.7, reserved_gib=32, **self.KW)
        self.assertEqual(plain.fraction, swapped.fraction)
        self.assertEqual(swapped.swap_used_gib, 12.0)
        self.assertTrue(any("si/so" in warning for warning in swapped.warnings))


class TestCudaProbeParsing(unittest.TestCase):
    def test_parses_the_last_json_line(self):
        stdout = "some torch noise\nwarnings here\n" + '{"free_gib": 74.0, "total_gib": 119.7}\n'
        self.assertEqual(memfit.parse_cuda_total(stdout), 119.7)

    def test_unparseable_is_none_not_a_guess(self):
        self.assertIsNone(memfit.parse_cuda_total("Traceback (most recent call last):"))
        self.assertIsNone(memfit.parse_cuda_total(""))
        self.assertIsNone(memfit.parse_cuda_total('{"total_gib": 0}'))
        self.assertIsNone(memfit.parse_cuda_total('{"total_gib": "big"}'))

    def test_probe_statement_reads_mem_get_info_not_mem_prop(self):
        # mem_get_info returns (free, total) for the device as CUDA sees it,
        # which is the same quantity the engine divides by when sizing its pool.
        stmt = memfit.MEASURE_CUDA_TOTAL
        self.assertIn("mem_get_info", stmt)
        self.assertNotIn("mem_prop", stmt)
        compile(stmt, "<probe>", "exec")


class TestRealBoxNumbers(unittest.TestCase):
    def test_reference_box_gets_a_fraction_below_every_published_recipe(self):
        fit = memfit.solve(
            memfit.HostMemory.from_meminfo(fixtures.MEMINFO_HEALTHY), 119.7,
            reserved_gib=16, min_fraction=0.45, max_fraction=0.72,
        )
        # Upstream pins for contrast: 0.95 and 0.90 on a dedicated box, 0.76 for
        # the 1M preset, 0.70 and 0.50 elsewhere. With a dozen business
        # containers resident here, the high pins are the documented reboot.
        self.assertLess(fit.fraction, 0.72)
        self.assertGreaterEqual(fit.fraction, 0.45)
        self.assertAlmostEqual(fit.askable_gib, 74.45, places=1)


if __name__ == "__main__":
    unittest.main()
    def test_fit_for_box_reads_a_supplied_meminfo_path(self):
        # The CLI entry point must be drivable from a fixture file, or the tests
        # would be asserting about the machine running them.
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meminfo")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(fixtures.MEMINFO_HEALTHY)
            fit = memfit.fit_for_box(reserved_gib=16, min_fraction=0.45,
                                     max_fraction=0.72, meminfo_path=path,
                                     cuda_total_gib=119.7)
        self.assertEqual(fit.bound_by, "askable")
        self.assertEqual(fit.available_gib, 90.45)
