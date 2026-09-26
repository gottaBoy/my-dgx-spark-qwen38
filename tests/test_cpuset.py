"""The cpuset must be derived, not remembered, and an explicit pin must win."""

from __future__ import annotations

import os
import tempfile
import unittest

from tests import fixtures
from qwen38 import cpuset


def _fake_sysfs(freqs: dict[int, int]) -> str:
    tmp = tempfile.mkdtemp()
    for cpu, freq in freqs.items():
        path = os.path.join(tmp, f"cpu{cpu}", "cpufreq")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "scaling_max_freq"), "w", encoding="utf-8") as handle:
            handle.write(f"{freq}\n")
    return tmp


class TestParsing(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(cpuset.format_cpu_list(cpuset.parse_cpu_list("5-9,15-19")), "5-9,15-19")

    def test_shuffled_input_comes_out_sorted_and_ranged(self):
        self.assertEqual(cpuset.format_cpu_list(cpuset.parse_cpu_list("19,15,5,6,7,8,9")), "5-9,15,19")
        self.assertEqual(cpuset.format_cpu_list(cpuset.parse_cpu_list("19,15,16,5,6,7,8,9")), "5-9,15-16,19")

    def test_singletons(self):
        self.assertEqual(cpuset.format_cpu_list([0, 2, 3, 9]), "0,2-3,9")

    def test_empty(self):
        self.assertEqual(cpuset.parse_cpu_list(""), [])
        self.assertEqual(cpuset.format_cpu_list([]), "")

    def test_rejects_garbage(self):
        for bad in ("abc", "5-", "-5", "1,", "3-1", "0,,1", "1-2-3"):
            with self.assertRaises(ValueError, msg=bad):
                cpuset.parse_cpu_list(bad)

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            cpuset.parse_cpu_list("1-3,2-4")


class TestDerivation(unittest.TestCase):
    def test_reference_box_matches_the_upstream_constant(self):
        root = _fake_sysfs(fixtures.CPU_FREQS_GB10)
        got, note = cpuset.resolve("auto", root)
        self.assertEqual(got, "5-9,15-19")
        self.assertIn("3900 MHz", note)
        self.assertIn("0-4,10-14", note)

    def test_a_different_sku_is_not_forced_into_the_old_constant(self):
        root = _fake_sysfs(fixtures.CPU_FREQS_SHUFFLED)
        got, _ = cpuset.resolve("auto", root)
        self.assertEqual(got, "0-3,12-15")

    def test_all_cores_equal_yields_everything(self):
        root = _fake_sysfs({i: 2400000 for i in range(8)})
        got, note = cpuset.resolve("auto", root)
        self.assertEqual(got, "0-7")
        self.assertIn("8 performance cores", note)

    def test_missing_cpufreq_leaves_it_unpinned(self):
        # No pinning is slower but honest; a wrong pin can be far slower and
        # shows up as an unexplained benchmark regression.
        with tempfile.TemporaryDirectory() as tmp:
            got, note = cpuset.resolve("auto", tmp)
        self.assertEqual(got, "")
        self.assertIn("leaving unpinned", note)

    def test_partial_topology_still_works(self):
        root = _fake_sysfs({0: 2000000, 1: 3000000})
        # cpu2 exists in the dir name but has no readable freq; must not crash.
        os.makedirs(os.path.join(root, "cpu2", "cpufreq"), exist_ok=True)
        got, _ = cpuset.resolve("auto", root)
        self.assertEqual(got, "1")

    def test_explicit_request_passes_through_verbatim(self):
        got, note = cpuset.resolve("5-9,15-19", _fake_sysfs(fixtures.CPU_FREQS_SHUFFLED))
        self.assertEqual(got, "5-9,15-19")
        self.assertIn("explicit", note)

    def test_a_lone_big_core_is_reported_by_index_not_by_default(self):
        # Guards the "always pin 0-1 on a 2-core skew" style of regression.
        root = _fake_sysfs({0: 2808000, 1: 3900000})
        got, _ = cpuset.resolve("auto", root)
        self.assertEqual(got, "1")


if __name__ == "__main__":
    unittest.main()
