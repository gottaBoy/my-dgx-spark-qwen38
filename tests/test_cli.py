"""CLI behaviour that does not need a GPU, a container, or a real engine.

These pin the parts where the CLI is the product: what it prints, what it
refuses, and whether the commands it advises are the commands that work. A tune
protocol that prints a variable name the loader rejects is worse than no protocol.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import socket
import tempfile
import unittest
import urllib.request
from unittest.mock import Mock, patch
from pathlib import Path

from qwen38 import guard, net, settings

_ROOT = Path(__file__).resolve().parents[1]


def load_cli():
    """Import bin/qwen38 (no .py suffix, so it needs an explicit loader)."""
    spec = importlib.util.spec_from_loader(
        "q38cli", importlib.machinery.SourceFileLoader("q38cli", str(_ROOT / "bin" / "qwen38")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load_cli()


def run_cli(*argv) -> tuple[int, str, str]:
    """Invoke the CLI and capture its outcome, including a deliberate refusal.

    SystemExit is a normal return path here: a refusal to launch is the command
    succeeding at its job. Letting it escape the helper would turn every
    refuse-and-explain test into an error.
    """
    out, err = io.StringIO(), io.StringIO()
    # Failure probes must not append simulated failures to a deployed run.
    with tempfile.TemporaryDirectory(prefix="qwen38-cli-test-") as state, \
         patch.dict(os.environ, {"Q38_STATE_DIR": state}):
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if exc.code and isinstance(exc.code, str):
                err.write(exc.code)
    return code, out.getvalue(), err.getvalue()


@contextlib.contextmanager
def closed_port_in_env():
    """Point Q38_PORT at a port nothing is listening on, for the duration.

    These tests assert what the CLI says when the engine is *not* there. They used
    to rely on the shipped default being free, which held right up until the engine
    was started on this box and then failed three tests for a reason that had
    nothing to do with the code under test. Bind a port, read the number, release
    it: the kernel will not hand it to anyone else in the microseconds we use it.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    saved = os.environ.get("Q38_PORT")
    os.environ["Q38_PORT"] = str(port)
    try:
        yield port
    finally:
        if saved is None:
            os.environ.pop("Q38_PORT", None)
        else:
            os.environ["Q38_PORT"] = saved


class TestTuneProtocolIsExecutable(unittest.TestCase):
    def test_printed_variable_names_are_real_settings(self):
        # The whole value of `tune` is that you can paste what it prints. A name
        # that settings.load() would reject makes the tool a lie.
        from qwen38 import settings
        known = set(settings.load().values)
        for knob, (_values, _why, env_key) in cli.KNOBS.items():
            self.assertIn(env_key, known, f"knob {knob} prints {env_key}, which is not a setting")

    def test_header_and_command_agree_on_the_variable(self):
        for knob in cli.KNOBS:
            _code, out, _err = run_cli("tune", knob)
            header = out.splitlines()[0].split(":")[1].strip().split()[0]
            self.assertIn(header + "=", out, f"knob {knob}: header says {header}")

    def test_unknown_knob_is_a_clean_error(self):
        code, _out, err = run_cli("tune", "mem_fraction_static")
        self.assertEqual(code, 1)
        self.assertIn("Available:", err)

    def test_spec_steps_sweep_moves_its_companion_knob(self):
        # topk=1 requires draft == steps+1; sweeping steps alone boots a config
        # the engine refuses, so the printed command must carry both.
        _code, out, _err = run_cli("tune", "spec_steps")
        for line in out.splitlines():
            if "Q38_SPEC_STEPS=" in line and "qwen38 start" in line:
                steps = int(line.split("Q38_SPEC_STEPS=")[1].split()[0])
                self.assertIn(f"Q38_SPEC_DRAFT={steps + 1}", line, line)

    def test_context_sweep_says_which_profiles_it_forces(self):
        _code, out, _err = run_cli("tune", "context_length")
        self.assertIn("cannot serve", out)
        self.assertIn("Q38_PROFILE=mtp", out)

    def test_the_control_step_and_stopping_rule_are_printed(self):
        # These are the two lines people skip, so they are the two worth pinning.
        _code, out, _err = run_cli("tune", "profile")
        self.assertIn("re-run the control", out)
        self.assertIn("stopping rule", out)


class TestRefusals(unittest.TestCase):
    def test_1m_on_a_draft_profile_refuses_without_touching_the_box(self):
        code, out, err = run_cli("plan", "--profile", "dflash2",
                                 "--context-length", "1000000", "--no-probe",
                                 "--mem-fraction", "0.6")
        self.assertNotEqual(code, 0)
        self.assertIn("cannot serve", out + err)
        self.assertNotIn("docker run", out)     # refused, so nothing was planned out

    def test_the_refusal_names_the_way_out(self):
        # A refusal that does not say what to do instead costs the user a read of
        # the source. This one names the profile that works.
        _code, out, err = run_cli("plan", "--profile", "dspark", "--context-length", "524288",
                                  "--no-probe", "--mem-fraction", "0.6")
        self.assertIn("mtp", out + err)

    def test_unknown_profile_is_rejected_by_argparse(self):
        code, _out, err = run_cli("plan", "--profile", "quantum")
        self.assertNotEqual(code, 0)
        self.assertIn("invalid choice", err)
        # argparse lists the valid names, which is the part the user needs.
        for name in ("dflash2", "mtp", "dspark", "ar"):
            self.assertIn(name, err)

    def test_bench_against_no_engine_fails_clearly(self):
        with closed_port_in_env():
            code, _out, err = run_cli("bench", "--probe", "code")
        self.assertEqual(code, 1)
        self.assertIn("unreachable", err + _out)

    def test_metrics_against_no_engine_says_so(self):
        with closed_port_in_env():
            code, _out, err = run_cli("metrics")
        self.assertEqual(code, 1)
        self.assertIn("unreachable", err + _out)


class TestMetricParsing(unittest.TestCase):
    SAMPLE = "\n".join([
        "# HELP sglang:spec_accept_length mean accepted tokens",
        "# TYPE sglang:spec_accept_length gauge",
        'sglang:spec_accept_length{model="qwen3.8-27b"} 2.89',
        "sglang:gen_throughput 54.6",
        "sglang:token_usage 0.31",
        "sglang:num_not_interesting 999",
        "",
    ])

    def test_reads_only_the_known_counters(self):
        found = cli._parse_metrics(self.SAMPLE)
        self.assertEqual(found, {"sglang:spec_accept_length": 2.89,
                                 "sglang:gen_throughput": 54.6,
                                 "sglang:token_usage": 0.31})

    def test_labels_are_stripped_not_kept_as_distinct_keys(self):
        found = cli._parse_metrics('sglang:token_usage{a="1"} 0.5\nsglang:token_usage{a="2"} 0.6\n')
        self.assertEqual(len(found), 1)

    def test_unparseable_value_is_dropped_not_zero(self):
        # NaN or a truncated line must not become a reading of 0.0, which would
        # look like "accept length collapsed" rather than "we could not read it".
        found = cli._parse_metrics("sglang:gen_throughput not-a-number\n")
        self.assertNotIn("sglang:gen_throughput", found)

    def test_absent_counter_is_reported_as_absent_in_the_output(self):
        # The distinction between "zero" and "this build does not expose it" is a
        # version fact about the image, and it changes what you conclude.
        self.assertIn("absent", " ".join(cli.METRIC_MEANINGS) + "(absent)")
        for name in cli.METRIC_MEANINGS:
            self.assertTrue(name.startswith("sglang:"), name)


class TestStopIntentMarker(unittest.TestCase):
    """The marker's lifecycle, which is the only thing separating a stop from a crash.

    Written by stop and by the guard, cleared by start -- and *not* by the writer.
    A marker deleted on the way out would leave the supervisor reading an absent
    file and grading a deliberate stop as a crash, restarting the engine the
    operator just stopped.
    """

    def _fake(self, tmp):
        import types
        return types.SimpleNamespace(state_dir=Path(tmp))

    def test_mark_survives_the_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._fake(tmp)
            cli._mark_stop_intended(s)
            self.assertTrue(cli._stop_is_intended(s), "marker must outlive the writer")

    def test_unmarked_box_reads_as_not_intended(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(cli._stop_is_intended(self._fake(tmp)))

    def test_start_clears_a_marker_left_by_a_previous_run(self):
        # The stale-marker hazard: one stop makes every later crash look deliberate.
        with tempfile.TemporaryDirectory() as tmp:
            s = self._fake(tmp)
            cli._mark_stop_intended(s)
            cli._intent_path(s).unlink(missing_ok=True)      # what cmd_start does
            self.assertFalse(cli._stop_is_intended(s))

    def test_marker_is_written_before_the_container_is_touched(self):
        """Order matters: writing it after the stop leaves a window in which a
        killed supervisor sees a dead container and no marker, i.e. a crash."""
        src = (_ROOT / "bin" / "qwen38").read_text()
        body = src[src.index("def cmd_stop"):src.index("def cmd_metrics")]
        self.assertLess(body.index("_mark_stop_intended"),
                        body.index('docker", "stop"'),
                        "intent must be recorded before docker stop runs")

    def test_guard_marks_intent_before_it_stops_the_engine(self):
        # Same ordering rule as cmd_stop, for the same reason: if the trip stopped
        # first and marked second, a supervisor already dead would have graded the
        # trip as a crash and systemd would relaunch the engine straight back into
        # the memory pressure that just triggered it -- the watchdog undoing itself.
        src = (_ROOT / "bin" / "qwen38").read_text()
        body = src[src.index("def cmd_guard"):src.index("def cmd_compare")]
        self.assertIn("_mark_stop_intended(s)", body)
        self.assertLess(body.index("_mark_stop_intended(s)"),
                        body.index('docker", "stop"'),
                        "guard must record intent before it stops the container")

class TestServiceWatchdog(unittest.TestCase):
    def test_dry_run_does_not_start_a_watchdog_or_wait_for_settle(self):
        args = cli.build_parser().parse_args(["service-start", "--dry-run"])
        with patch.object(cli, "cmd_start", return_value=0) as start, \
             patch.object(cli.subprocess, "Popen") as popen, \
             patch.object(cli.settle, "wait_until_stable") as settle_wait:
            self.assertEqual(cli.cmd_service_start(settings.load(), args), 0)
        start.assert_called_once()
        popen.assert_not_called()
        settle_wait.assert_not_called()

    def test_guard_is_started_before_launch_and_cleaned_up_on_failure(self):
        args = cli.build_parser().parse_args(["service-start"])
        watchdog = Mock()
        verdict = Mock(settled=True, drift_gib=0.0, samples=[90.0] * 3)
        calls = []
        with patch.object(cli.settle, "wait_until_stable", return_value=(verdict, 20)), \
             patch.object(cli.subprocess, "Popen", side_effect=lambda *a, **k: calls.append("guard") or watchdog), \
             patch.object(cli, "cmd_start", side_effect=lambda *a: calls.append("start") or 1), \
             patch.object(cli, "_stop_is_intended", return_value=False):
            self.assertEqual(cli.cmd_service_start(settings.load(), args), 1)
        self.assertEqual(calls, ["guard", "start"])
        watchdog.terminate.assert_called_once()
        watchdog.wait.assert_called_once_with(timeout=10)

    def test_guard_trip_during_boot_does_not_trigger_restart(self):
        args = cli.build_parser().parse_args(["service-start"])
        verdict = Mock(settled=True, drift_gib=0.0, samples=[90.0] * 3)
        with patch.object(cli.settle, "wait_until_stable", return_value=(verdict, 20)), \
             patch.object(cli.subprocess, "Popen"), \
             patch.object(cli, "cmd_start", return_value=1), \
             patch.object(cli, "_stop_is_intended", return_value=True):
            self.assertEqual(cli.cmd_service_start(settings.load(), args), 0)

    def test_guard_latches_readiness_and_uses_configured_full_threshold(self):
        s = settings.load()
        s.values["Q38_GUARD_PSI_FULL"] = "12.0"
        s.values["Q38_GUARD_PSI_SOME"] = "25.0"
        sample = guard.Sample(0.0, 7.0, 99.0)
        with patch.object(cli, "_http_ok", side_effect=[False, True]) as health, \
             patch.object(cli.guard, "read_sample", return_value=sample), \
             patch.object(cli.time, "sleep", side_effect=[None, None, KeyboardInterrupt]), \
             patch.object(cli, "_run") as run:
            self.assertEqual(cli.cmd_guard(s, Mock()), 0)
        self.assertEqual(health.call_count, 2)
        run.assert_not_called()


class TestHelpSurface(unittest.TestCase):
    @classmethod
    def _command_names(cls) -> list[str]:
        """Read the command list out of --help rather than keeping a second copy.

        A hand-maintained list is a second source of truth, and it goes stale in
        the direction that hides the problem: a subcommand gets added, the list
        does not, and then any message pointing at it passes un-checked. This is
        the same class of bug the upstream repos each needed a CI invariant for.
        """
        _code, out, _err = run_cli("--help")
        match = re.search(r"\{([a-z][a-z0-9,-]*)\}", out)
        if not match:
            raise AssertionError(f"no subcommand list found in --help:\n{out[:400]}")
        return sorted(match.group(1).split(","))

    def test_every_subcommand_has_a_help_page(self):
        names = self._command_names()
        # A floor, not an equality: the list must be real, and an empty or
        # truncated parse must fail loudly rather than make the loop below vacuous.
        self.assertGreaterEqual(len(names), 20, names)
        _code, out, _err = run_cli("--help")
        for name in names:
            self.assertIn(name, out, name)

    def test_service_start_and_prefetch_are_discoverable(self):
        # Added by name on purpose. These two are load-bearing for the unit and
        # for a first install, and a derived list would not notice either vanishing.
        names = self._command_names()
        for name in ("service-start", "prefetch", "verify-pins", "start", "doctor"):
            self.assertIn(name, names, name)

    def test_service_start_is_a_real_command_the_unit_can_call(self):
        # The unit template references it by name; a rename that misses the
        # template produces a service that fails at boot, not at install.
        code, _out, _err = run_cli("service-start", "--help")
        self.assertEqual(code, 0)

    def test_the_unit_template_only_calls_commands_that_exist(self):
        # Catches the rename-on-one-side-only bug class the upstream projects each
        # needed a CI invariant for.
        text = (_ROOT / "unit" / "qwen38-spark.service.in").read_text()
        referenced = {line.split("bin/qwen38")[1].split()[0]
                      for line in text.splitlines() if "bin/qwen38" in line}
        self.assertTrue(referenced)
        _code, out, _err = run_cli("--help")
        for command in referenced:
            self.assertIn(command, out, f"unit calls {command}, which the CLI does not have")

    def test_advice_printed_to_the_user_names_real_commands(self):
        """Every `qwen38 <cmd>` in a message, docstring or comment must exist.

        This caught a real one: a warning told the operator to run
        `qwen38 preflight`, a command that was never written. Advice that points
        at a nonexistent command is worse than no advice, because the reader
        concludes the tool is unreliable rather than that one string is stale.
        """
        sources = [*_ROOT.glob("bin/*"), *_ROOT.glob("lib/qwen38/*.py"),
                   *_ROOT.glob("docs/*.md"), _ROOT / "README.md",
                   *_ROOT.glob("*.sh"), *_ROOT.glob("unit/*")]
        pattern = re.compile(r"\bqwen38 ([a-z][a-z-]+)")
        known = set(self._command_names()) | {"--help"}
        offenders: list[str] = []
        for path in sources:
            if not path.is_file():
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                # Python imports read as "from qwen38 import x", which is a module
                # path, not advice to a human. Skipping them keeps the rule about
                # what the tool tells people to type.
                if line.lstrip().startswith(("from ", "import ")):
                    continue
                # A deliberate, greppable exemption for prose that *quotes* a dead
                # command while explaining why it was wrong. Suppressions must be
                # visible in the file they excuse, not in a list nobody reads.
                if "name-check:ignore" in line:
                    continue
                for match in pattern.finditer(line):
                    word = match.group(1)
                    if word not in known:
                        offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: qwen38 {word}")
        self.assertEqual(offenders, [], "\n" + "\n".join(offenders))


class TestLocalRequestsBypassProxies(unittest.TestCase):
    """A real failure on a real box, tested behaviourally.

    The reference machine exports HTTP_PROXY and lists `127.*` in no_proxy.
    Python matches no_proxy entries by suffix rather than wildcard, so `127.*`
    does not bypass 127.0.0.1, and a request to a healthy local engine returns a
    502 from the proxy. That reads as "engine unreachable" and sends you off to
    debug the engine. So: serve on loopback, point the environment at a dead
    proxy, and check which of the two paths still works.
    """

    def setUp(self):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"DIRECT-OK"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):      # keep the test output clean
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        # A proxy that will refuse: stands in for the box's real one, which
        # answers but 502s on anything it cannot reach.
        self._saved = {k: os.environ.get(k) for k in
                       ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy")}
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"
        os.environ["http_proxy"] = "http://127.0.0.1:1"
        os.environ.pop("NO_PROXY", None)
        os.environ.pop("no_proxy", None)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_local_opener_reaches_loopback_with_a_proxy_configured(self):
        body = net.local_opener().open(f"http://127.0.0.1:{self.port}/", timeout=5).read()
        self.assertEqual(body, b"DIRECT-OK")

    def test_the_naive_path_is_the_one_that_breaks(self):
        # Guard the premise. If the standard library ever fixes loopback bypass,
        # this fails, and the workaround in net.local_opener can be deleted rather
        # than left as folklore.
        with self.assertRaises(OSError):
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5)

    def test_remote_opener_keeps_the_system_proxy(self):
        # The opposite rule, and the one that cost a 30 s hang to learn: the Hub
        # is reachable HERE only through the proxy, so the shared helper must not
        # strip it for external calls the way it does for loopback.
        # Asserted by behaviour, not by poking at handler internals: ProxyHandler
        # is not in OpenerDirector.handlers, so an internals test here was wrong
        # twice before it was right. setUp points the environment proxy at a dead
        # port, so "honours the proxy" means "fails", and "bypasses it" means
        # "connects". Same fixture, opposite expectations.
        with self.assertRaises(OSError):
            net.remote_opener().open(f"http://127.0.0.1:{self.port}/", timeout=5)
        body = net.local_opener().open(f"http://127.0.0.1:{self.port}/", timeout=5).read()
        self.assertEqual(body, b"DIRECT-OK")


class TestHealthProbesDoNotLie(unittest.TestCase):
    def test_cli_state_is_isolated_cleaned_up_and_environment_restored(self):
        seen = []

        def fake_canary(s, args):
            seen.append(s.state_dir)
            cli._mark_stop_intended(s)
            self.assertTrue(cli._stop_is_intended(s))
            return 0

        with tempfile.TemporaryDirectory() as live_state, \
             patch.dict(os.environ, {"Q38_STATE_DIR": live_state}), \
             patch.object(cli, "cmd_canary", side_effect=fake_canary):
            code, _out, _err = run_cli("canary")
            self.assertEqual(code, 0)
            self.assertNotEqual(seen[0], Path(live_state))
            self.assertFalse(seen[0].exists())
            self.assertFalse(cli._stop_is_intended(settings.load()))
            self.assertEqual(os.environ["Q38_STATE_DIR"], live_state)

    def test_refused_connection_is_reported_as_unreachable_not_as_a_crash(self):
        # An OSConnectionError must land in the "unreachable" branch; the 502
        # above arrived as HTTPError, which is a different exception class and
        # used to escape cmd_bench's handler entirely.
        with closed_port_in_env():
            code, _out, err = run_cli("canary")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", _out + err)


class TestBenchmarkReporting(unittest.TestCase):
    def test_different_workloads_are_not_reported_as_one_noise_distribution(self):
        counts = [16, 60, 600, 60, 600]
        times = [0, 1, 1, 2, 2, 12, 12, 13, 13, 43]
        with patch.object(cli, "_api", side_effect=[
                {"usage": {"completion_tokens": count}} for count in counts]), \
             patch.object(cli.time, "time", side_effect=times):
            code, out, err = run_cli("bench", "--repeats", "1")
        self.assertEqual(code, 0, err)
        self.assertIn("60.00 tok/s", out)
        self.assertIn("18.62 tok/s", out)
        self.assertIn("per-probe ranges", out)
        self.assertNotIn("Run-to-run spread", out)


if __name__ == "__main__":
    unittest.main()
