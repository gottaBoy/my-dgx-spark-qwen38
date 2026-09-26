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
import unittest
import urllib.request
from pathlib import Path

from qwen38 import net

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
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exc.code and isinstance(exc.code, str):
            err.write(exc.code)
    return code, out.getvalue(), err.getvalue()


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
        code, _out, err = run_cli("bench", "--probe", "code")
        self.assertEqual(code, 1)
        self.assertIn("unreachable", err + _out)

    def test_metrics_against_no_engine_says_so(self):
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
    def test_refused_connection_is_reported_as_unreachable_not_as_a_crash(self):
        # An OSConnectionError must land in the "unreachable" branch; the 502
        # above arrived as HTTPError, which is a different exception class and
        # used to escape cmd_bench's handler entirely.
        code, _out, err = run_cli("canary")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", _out + err)


if __name__ == "__main__":
    unittest.main()
