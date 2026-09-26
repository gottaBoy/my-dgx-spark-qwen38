"""Port selection and the run ledger: the two mechanisms that keep us from
stepping on a neighbour and from losing an experiment.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

from tests import fixtures
from qwen38 import evidence, ports


class TestEphemeralFloor(unittest.TestCase):
    def test_reads_the_kernel_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "range")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(fixtures.IP_LOCAL_PORT_RANGE)
            self.assertEqual(ports.ephemeral_floor(path), 32768)

    def test_unreadable_is_no_opinion(self):
        self.assertEqual(ports.ephemeral_floor("/nonexistent/range"), 0)

    def test_garbage_is_no_opinion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "range")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("nonsense\n")
            self.assertEqual(ports.ephemeral_floor(path), 0)


class TestIsFree(unittest.TestCase):
    def test_a_listening_socket_is_not_free(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            self.assertFalse(ports.is_free(port))

    def test_an_unanswered_port_is_free(self):
        # Bind then close: the port is released and a connect is refused.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.assertTrue(ports.is_free(port))


class TestChoose(unittest.TestCase):
    def test_skips_a_taken_port(self):
        # ephemeral_floor is pinned to 0 here because a socket bound by the OS
        # lands inside the real ephemeral range, which the next test asserts we
        # reject. Without this the two properties cannot be told apart.
        import unittest.mock as mock
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            taken = listener.getsockname()[1]
            with mock.patch.object(ports, "ephemeral_floor", return_value=0):
                chosen, probes = ports.choose([taken, taken + 1])
        self.assertIsNotNone(chosen)
        self.assertNotEqual(chosen, taken)
        self.assertEqual(len(probes), 2)

    def test_refuses_to_pick_inside_the_ephemeral_range(self):
        # A port the kernel may hand to any outgoing connection can be squatted
        # mid-run; on a shared box that means our engine loses its own listener.
        floor = ports.ephemeral_floor()
        if not floor:
            self.skipTest("no readable ip_local_port_range")
        chosen, _ = ports.choose([floor + 1, floor + 2])
        self.assertIsNone(chosen)

    def test_no_candidates_yields_no_choice(self):
        chosen, probes = ports.choose([])
        self.assertIsNone(chosen)
        self.assertEqual(probes, [])

    def test_shipped_ports_clear_this_box(self):
        """The shipped defaults must be sound as *configuration*, not as a snapshot.

        This test used to assert the ports were free at this instant. That is a
        property of the moment, not of the file: it passed on an idle box and
        failed the first time the engine was actually deployed on it -- a test
        that breaks when the thing it describes succeeds. Whether anything is
        listening is checked at launch by `ports.choose`, which is where that fact
        belongs. What stays here are the invariants a config file can be wrong about
        on its own.
        """
        from qwen38 import settings
        loaded = settings.load()
        floor = ports.ephemeral_floor()
        shipped = {key: loaded.int(key) for key in ("Q38_PORT", "Q38_STATUS_PORT")}
        self.assertEqual(len(set(shipped.values())), 2, shipped)
        for key, port in shipped.items():
            # Below 1024 needs root to bind, which this stack never assumes.
            self.assertGreaterEqual(port, 1024, key)
            if floor:
                # Inside the ephemeral range, the kernel can hand this port to any
                # outgoing connection on the box and our listener loses it.
                self.assertLess(port, floor, f"{key}={port} is inside the ephemeral range")
            # The neighbours that were here before this repo, taken from the box's
            # own published ports. Date-stamped on purpose: this list needs a
            # conscious edit when the machine changes, which is the point.
            self.assertNotIn(port, {30000, 30086, 30088, 30250, 30251,
                                    32002, 32003, 32666}, f"{key} collides")

    def test_launch_still_probes_before_claiming_a_port(self):
        # The other half of the previous test, in the place it belongs: choose()
        # must not hand back a port something is already holding.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            taken = listener.getsockname()[1]
            chosen, _probes = ports.choose([taken])
        self.assertNotEqual(chosen, taken)


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = evidence.Ledger(Path(tempfile.mkdtemp()))
        self.run_id = evidence.new_run_id("dflash2")
        self.ledger.create(self.run_id, git_sha="abc1234",
                           plan={"profile": "dflash2", "mem_fraction_static": 0.621},
                           fit={"fraction": 0.621, "bound_by": "askable"})

    def test_round_trip(self):
        record = self.ledger.load(self.run_id)
        self.assertEqual(record.git_sha, "abc1234")
        self.assertEqual(record.status, "open")
        self.assertEqual(record.plan["profile"], "dflash2")

    def test_events_append_in_order_and_are_never_edited(self):
        self.ledger.append(self.run_id, event={"kind": "launch"})
        self.ledger.append(self.run_id, status="ready", event={"kind": "ready", "seconds": 412})
        record = self.ledger.load(self.run_id)
        self.assertEqual([e["kind"] for e in record.events], ["launch", "ready"])
        self.assertEqual(record.status, "ready")

    def test_metrics_merge_across_appends(self):
        self.ledger.append(self.run_id, metrics={"bench_code": 54.6})
        self.ledger.append(self.run_id, metrics={"bench_essay": 25.4})
        self.assertEqual(self.ledger.load(self.run_id).metrics,
                         {"bench_code": 54.6, "bench_essay": 25.4})

    def test_recreating_a_run_is_refused(self):
        # Immutable history: a run id carries a timestamp, so colliding means
        # someone is overwriting a measurement rather than making a new one.
        with self.assertRaises(FileExistsError):
            self.ledger.create(self.run_id, git_sha="x", plan={})

    def test_path_traversal_in_a_run_id_is_refused(self):
        # The id arrives from --run on the command line and then becomes a path,
        # so it is untrusted input. Anything outside the timestamped shape fails.
        for bad in ("../../etc/passwd", "no-timestamp-dflash2", "UPPER", "",
                    "20260926T000000Z-a/../../../x", "20260926T000000Z-b;rm -rf",
                    "20260926T000000Z-" + "x" * 60):
            with self.assertRaises(ValueError, msg=bad):
                self.ledger.path_for(bad)

    def test_written_file_is_valid_json_with_sorted_keys(self):
        path = self.ledger.path_for(self.run_id)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(list(data), sorted(data))

    def test_no_temporary_files_are_left_behind(self):
        # Atomic replace means a crash mid-write cannot truncate a record.
        self.ledger.append(self.run_id, event={"kind": "check"})
        leftovers = list(self.ledger.path_for(self.run_id).parent.glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_list_runs_is_sorted_and_ignores_junk(self):
        (self.ledger.root / "not-a-run").mkdir()
        (self.ledger.root / "scratch.txt").write_text("x")
        self.assertEqual(self.ledger.list_runs(), [self.run_id])

    def test_missing_root_lists_nothing(self):
        self.assertEqual(evidence.Ledger(Path(tempfile.mkdtemp()) / "absent").list_runs(), [])


if __name__ == "__main__":
    unittest.main()
