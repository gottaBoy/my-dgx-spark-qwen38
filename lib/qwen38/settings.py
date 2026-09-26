"""Config loading: one file format, read the same way by shell and Python.

conf/config.defaults is plain KEY="value" lines, no shell expansion. That is a
deliberate restriction: the alternative is a Python parser that reimplements
bash quoting, and a parser nobody tests against the shell it imitates is where
"works when I source it, breaks when the CLI reads it" bugs are born.

Precedence, highest first:
  1. real environment (Q38_*)
  2. conf/config.local    -- gitignored, per-box
  3. conf/config.defaults -- tracked, the design record
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Literal values only: no $, no backquote, no nested quote, no brace. This makes
# "the format excludes shell syntax" a rule the parser enforces rather than a
# convention the README asks you to respect.
_LINE = re.compile(r'^([A-Z0-9_]+)="([A-Za-z0-9 ._/:@~=-]*)"$')

DEFAULTS_RELPATH = "conf/config.defaults"
LOCAL_RELPATH = "conf/config.local"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_text(text: str, source: str = "<text>") -> dict[str, str]:
    """Parse KEY="value" lines. Unparseable non-comment lines are an error.

    Strict on purpose. A silently skipped line is a knob that stops existing,
    and the resulting behaviour looks like a mysterious performance change
    rather than a typo.

    The format deliberately excludes shell syntax. Values are never expanded,
    because a Python parser that reimplements bash quoting drifts from the bash
    that sources the same file, and "works when sourced, breaks when read" is
    where config bugs hide. `~` is the only substitution, done by expanduser().
    Anything else that looks like an expansion is rejected at parse time.
    """
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            raise SystemExit(
                f'{source}:{lineno}: not a KEY="value" line: {raw!r}'
            )
        out[match.group(1)] = match.group(2)
    return out


def parse_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return parse_text(path.read_text(encoding="utf-8"), str(path))


@dataclass(frozen=True)
class Settings:
    values: dict[str, str]
    root: Path = field(default_factory=repo_root)
    origins: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str:
        try:
            return self.values[key]
        except KeyError:
            raise KeyError(f"{key} is not a known setting") from None

    def int(self, key: str) -> int:
        return int(self.get(key))

    def float(self, key: str) -> float:
        return float(self.get(key))

    def flag(self, key: str) -> bool:
        return self.get(key) not in ("", "0", "false", "no")

    def wordlist(self, key: str) -> tuple[str, ...]:
        value = self.get(key).strip()
        return tuple(value.split()) if value else ()

    @property
    def state_dir(self) -> Path:
        return self.root / self.get("Q38_STATE_DIR")

    @property
    def cache_dir(self) -> Path:
        return Path(self.get("Q38_CACHE_DIR")).expanduser()

    def evidence_dir(self) -> Path:
        return self.state_dir / "evidence"


def load(root: Path | None = None) -> Settings:
    root = root or repo_root()
    values: dict[str, str] = {}
    origins: dict[str, str] = {}
    known = set(parse_file(root / DEFAULTS_RELPATH))
    for relpath, origin in ((DEFAULTS_RELPATH, "defaults"), (LOCAL_RELPATH, "local")):
        for key, value in parse_file(root / relpath).items():
            # A typo in config.local otherwise reads fine and does nothing, which
            # is the worst possible failure for a tuning harness: the experiment
            # you think you ran is not the one that booted.
            if key not in known:
                raise SystemExit(
                    f"{root / relpath}: {key} is not a known setting. Check the "
                    f"spelling against {DEFAULTS_RELPATH}."
                )
            values[key] = value
            origins[key] = origin
    for key in list(values):
        env = os.environ.get(key)
        if env is not None:
            values[key] = env
            origins[key] = "env"
    # An unknown Q38_* in the environment is the same trap as an unknown key in
    # config.local, and it is easy to hit because `qwen38 tune` prints commands of
    # exactly this shape: a typo boots the default config while you believe you
    # changed one knob, and the ledger records the config that really ran -- so
    # the sweep looks plausible and is wrong.
    unknown_env = sorted(
        key for key in os.environ
        if key.startswith("Q38_") and key not in known
    )
    if unknown_env:
        raise SystemExit(
            "unknown setting(s) in the environment: " + ", ".join(unknown_env)
            + f". Check the spelling against {DEFAULTS_RELPATH}."
        )
    return Settings(values=values, root=root, origins=origins)
