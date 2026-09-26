"""Append-only run ledger: the thing that turns a boot into a data point.

The reason a tuning exercise usually teaches nothing: the config that produced
a number is not recorded with the number. You end up with a table of tok/s and
no way to answer "was the slow run the one with extra_buffer, or the one before
I changed concurrency?" So a run gets one id, and everything -- the plan, the
measured fit, the boot timeline, the canary, the benchmarks -- is written under
that id and cross-linked by git sha.

Two rules that make the ledger worth reading a month later:
  * a record always stores the whole plan, not a diff against "usual". The usual
    changes, and a diff against a moving baseline is how both upstream repos
    ended up documenting flags in prose comments.
  * a run is immutable once finished. To try something, make a new run. The
    comparison function then works on two files instead of on your memory.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

_ID = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9][a-z0-9._-]{0,40}$")


def new_run_id(profile: str, when: float | None = None) -> str:
    stamp = time.gmtime(when if when is not None else time.time())
    safe = re.sub(r"[^a-z0-9._-]", "-", profile.lower())
    return f"{time.strftime('%Y%m%dT%H%M%SZ', stamp)}-{safe}"


@dataclass
class RunRecord:
    run_id: str
    created_utc: str
    git_sha: str
    plan: dict
    fit: dict | None = None
    # open -> ready -> finished, or open -> failed. A run that is still "open"
    # when you look at it next is the paper trail of a crash, which is the most
    # informative record the ledger ever holds.
    status: str = "open"
    events: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class Ledger:
    """One directory per run, one JSON file per run inside it."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, run_id: str) -> Path:
        if not _ID.match(run_id):
            raise ValueError(f"malformed run id: {run_id!r}")
        return self.root / run_id / "run.json"

    def create(self, run_id: str, *, git_sha: str, plan: dict, fit: dict | None = None) -> RunRecord:
        """Open a run. Refuses to overwrite: ids carry a timestamp, so a
        collision means someone is rewriting history rather than measuring."""
        path = self.path_for(run_id)
        if path.exists():
            raise FileExistsError(f"run {run_id} already recorded at {path}")
        record = RunRecord(
            run_id=run_id,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            git_sha=git_sha,
            plan=plan,
            fit=fit,
        )
        self._write(path, record)
        return record

    def load(self, run_id: str) -> RunRecord:
        path = self.path_for(run_id)
        with path.open("r", encoding="utf-8") as handle:
            return RunRecord(**json.load(handle))

    def append(self, run_id: str, *, status: str | None = None,
               event: dict | None = None, metrics: dict | None = None) -> RunRecord:
        """Mutate one record in place. Events are appended, never edited."""
        record = self.load(run_id)
        if status:
            record.status = status
        if event is not None:
            record.events.append({"at": time.strftime("%H:%M:%S", time.gmtime()), **event})
        if metrics:
            record.metrics.update(metrics)
        self._write(self.path_for(run_id), record)
        return record

    def list_runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            entry.name for entry in self.root.iterdir()
            if (entry / "run.json").is_file() and _ID.match(entry.name)
        )

    def _write(self, path: Path, record: RunRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)      # atomic: a crash cannot truncate a record
