"""Config loading: one file format, read identically by shell and Python.

The rules being protected: an unknown key is an error in every channel, and a
real environment variable beats a file. Both exist because the failure they
prevent is a silently-ignored knob, which in a tuning harness means an
experiment that reports the wrong thing while looking entirely healthy.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from qwen38 import settings


def _repo(defaults: str, local: str = "") -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "conf").mkdir()
    (root / "conf" / "config.defaults").write_text(defaults, encoding="utf-8")
    (root / "conf" / "config.local").write_text(local, encoding="utf-8")
    return root


class TestParsing(unittest.TestCase):
    def test_comments_and_blanks_are_skipped(self):
        parsed = settings.parse_text("# hi\n\nA=\"1\"\n")
        self.assertEqual(parsed, {"A": "1"})

    def test_empty_value_is_allowed(self):
        self.assertEqual(settings.parse_text('A=""\n'), {"A": ""})

    def test_malformed_line_is_an_error(self):
        with self.assertRaises(SystemExit):
            settings.parse_text("A=1\n")            # unquoted
        with self.assertRaises(SystemExit):
            settings.parse_text("bad line here\n")

    def test_shell_expansion_is_rejected_not_evaluated(self):
        # The format is deliberately not shell: ${HOME} would need a bash
        # emulator in Python, and the two parsers drift.
        with self.assertRaises(SystemExit):
            settings.parse_text('A="${HOME}/x"\n')


class TestPrecedence(unittest.TestCase):
    def setUp(self):
        self.root = _repo('Q38_PORT="28100"\nQ38_BIND="127.0.0.1"\n', 'Q38_PORT="28111"\n')
        self._saved = {k: os.environ.pop(k, None) for k in ("Q38_PORT", "Q38_BIND")}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is not None:
                os.environ[key] = value

    def test_local_beats_defaults(self):
        self.assertEqual(settings.load(self.root).get("Q38_PORT"), "28111")

    def test_environment_beats_local(self):
        os.environ["Q38_PORT"] = "28122"
        loaded = settings.load(self.root)
        self.assertEqual(loaded.get("Q38_PORT"), "28122")
        self.assertEqual(loaded.origins["Q38_PORT"], "env")

    def test_typed_getters(self):
        loaded = settings.load(self.root)
        self.assertEqual(loaded.int("Q38_PORT"), 28111)
        self.assertEqual(loaded.float("Q38_PORT"), 28111.0)

    def test_flag_reads_the_documented_falsy_set(self):
        values = ("", "0", "false", "no", "1", "auto")
        root = _repo("\n".join(f'FLAG{i}="{v}"' for i, v in enumerate(values)) + "\n")
        loaded = settings.load(root)
        for i, expected in enumerate((False, False, False, False, True, True)):
            self.assertEqual(loaded.flag(f"FLAG{i}"), expected, values[i])

    def test_lowercase_key_is_rejected(self):
        # Keys are Q38_* shell identifiers; accepting lowercase here would let a
        # config file and the environment disagree about the same knob.
        with self.assertRaises(SystemExit):
            settings.parse_text('q38_port="28100"\n')

    def test_unknown_key_raises_not_returns_none(self):
        # Silent None becomes a default two layers away, and the boot looks fine.
        with self.assertRaises(KeyError):
            settings.load(self.root).get("Q38_NOPE")


class TestGuardrails(unittest.TestCase):
    def test_unknown_key_in_local_is_refused(self):
        root = _repo('Q38_PORT="28100"\n', 'Q38_PROT="28111"\n')   # typo
        with self.assertRaises(SystemExit) as caught:
            settings.load(root)
        self.assertIn("Q38_PROT", str(caught.exception))

    def test_unknown_env_var_is_refused(self):
        root = _repo('Q38_PORT="28100"\n')
        os.environ["Q38_SPEC_STEP"] = "3"          # the singular-vs-plural trap
        try:
            with self.assertRaises(SystemExit) as caught:
                settings.load(root)
            self.assertIn("Q38_SPEC_STEP", str(caught.exception))
        finally:
            del os.environ["Q38_SPEC_STEP"]

    def test_non_q38_env_vars_are_ignored(self):
        root = _repo('Q38_PORT="28100"\n')
        os.environ["PATH"] = os.environ.get("PATH", "") + ":/nowhere"
        os.environ["HOME_SOMETHING"] = "x"
        try:
            settings.load(root)
        finally:
            del os.environ["HOME_SOMETHING"]


class TestRealDefaultsFile(unittest.TestCase):
    """The shipped config must be loadable and self-consistent, not just parseable."""

    def setUp(self):
        self.settings = settings.load()

    def test_every_value_is_typed_as_its_consumer_expects(self):
        for key in ("Q38_PORT", "Q38_STATUS_PORT", "Q38_CONTEXT_LENGTH", "Q38_MAX_CONCURRENT",
                    "Q38_CHUNKED_PREFILL", "Q38_SPEC_STEPS", "Q38_SPEC_TOPK", "Q38_SPEC_DRAFT",
                    "Q38_DSPARK_BLOCK", "Q38_DFLASH_TOKENS", "Q38_GUARD_INTERVAL_S",
                    "Q38_GUARD_TRIP_STRIKES"):
            self.settings.int(key)
        for key in ("Q38_MIN_FRACTION", "Q38_MAX_FRACTION", "Q38_RESERVED_GIB",
                    "Q38_GUARD_PSI_SOME", "Q38_GUARD_AVAIL_FLOOR_GIB"):
            self.settings.float(key)

    def test_fraction_band_is_sane(self):
        self.assertLess(self.settings.float("Q38_MIN_FRACTION"),
                        self.settings.float("Q38_MAX_FRACTION"))
        self.assertGreater(self.settings.float("Q38_MIN_FRACTION"), 0)
        self.assertLess(self.settings.float("Q38_MAX_FRACTION"), 1)

    def test_ports_do_not_collide_with_the_neighbours_on_this_box(self):
        # 30000/30086/30250/32666 are taken by services that were running before
        # this repo existed. The assertion is a date-stamped fact, so it needs a
        # conscious edit when the box changes -- which is the point.
        taken = {30000, 30086, 30088, 30250, 30251, 32002, 32003, 32666}
        for key in ("Q38_PORT", "Q38_STATUS_PORT"):
            self.assertNotIn(self.settings.int(key), taken, key)

    def test_image_is_pinned_by_digest_not_tag(self):
        digest = self.settings.get("Q38_IMAGE_DIGEST")
        self.assertIn("@sha256:", digest)
        self.assertEqual(len(digest.split("@sha256:")[1]), 64)

    def test_topk_one_requires_draft_eq_steps_plus_one(self):
        # Upstream validates this at launch; catching it in config means a bad
        # edit never costs an eight-minute boot to explain itself.
        if self.settings.int("Q38_SPEC_TOPK") == 1:
            self.assertEqual(self.settings.int("Q38_SPEC_DRAFT"),
                             self.settings.int("Q38_SPEC_STEPS") + 1)

    def test_profile_names_exist(self):
        from qwen38 import profiles
        self.assertIn(self.settings.get("Q38_PROFILE"), profiles.PROFILES)

    def test_no_secrets_in_the_tracked_file(self):
        text = (self.settings.root / settings.DEFAULTS_RELPATH).read_text(encoding="utf-8")
        for pattern in ("hf_", "sk-", "token=", "password"):
            self.assertNotIn(pattern, text.lower(), pattern)


if __name__ == "__main__":
    unittest.main()
