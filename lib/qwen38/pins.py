"""Resolve what the pins actually point at, before a boot spends eight minutes.

The claim this checks is cheap and load-bearing: "revision X exists, and the
repo at X is the architecture this profile needs". Getting it wrong costs a
full download plus a load-time failure, or worse -- a draft with the wrong
packaging that loads and serves while being silently wrong.

Parsing is separate from fetching so the interesting logic (what counts as a
match, what to say when a field is missing) is testable without a network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import profiles

HF_MODEL_API = "https://huggingface.co/api/models/{repo}"


@dataclass(frozen=True)
class RepoFacts:
    """What the Hub says about one repository, with the fields we act on."""

    repo: str
    sha: str = ""
    architectures: tuple[str, ...] = ()
    storage_gib: float = 0.0
    last_modified: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.sha)

    @classmethod
    def from_json(cls, repo: str, payload: str) -> "RepoFacts":
        """Parse a Hub /api/models response. Never raises: a bad response is data.

        The failure modes are all real -- a 401 HTML page for a gated repo, a
        proxy's 502 body, a 404 with a JSON error -- and each must produce a
        specific message rather than a traceback that looks like our bug.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            snippet = payload.strip().splitlines()
            return cls(repo=repo, error=f"not JSON ({(snippet[0][:60] if snippet else 'empty')!r})")
        if not isinstance(data, dict):
            return cls(repo=repo, error=f"unexpected {type(data).__name__} response")
        if "error" in data or "errors" in data:
            detail = data.get("error") or (data.get("errors") or [{}])[0].get("message", "")
            return cls(repo=repo, error=str(detail)[:120] or "rejected by the Hub")
        config = data.get("config") or {}
        return cls(
            repo=repo,
            sha=str(data.get("sha") or ""),
            architectures=tuple(config.get("architectures") or ()),
            storage_gib=round((data.get("usedStorage") or 0) / 1073741824, 2),
            last_modified=str(data.get("lastModified") or "")[:10],
        )


@dataclass
class PinCheck:
    """One verdict about one (repo, revision, expected architecture) triple."""

    role: str          # target | draft
    repo: str
    requested: str     # the pinned revision, or "" for untracked
    status: str        # PASS | WARN | FAIL
    detail: str
    head_sha: str = ""
    storage_gib: float = 0.0
    drifted: bool = False


def _matches(commit_sha: str, pinned: str) -> bool:
    """Accept a full-sha equality or a prefix that is unambiguous.

    A short prefix is deliberately not a pass: the Hub resolves full shas, and
    `50307d4` naming a commit while `50307d4c...` is what the flag needs is the
    exact confusion this module was written to prevent.
    """
    return bool(commit_sha) and commit_sha.lower() == pinned.lower()


def check(role: str, repo: str, pinned: str, expected_arch: str,
          head: RepoFacts, pinned_view: RepoFacts | None = None) -> PinCheck:
    """Judge one pin from two Hub reads: the default branch, and the pinned rev.

    Two reads because one cannot answer both questions. The default branch tells
    you whether the pin has drifted; the pinned revision tells you whether it
    still resolves at all -- and the two can disagree in either direction.
    """
    if not head.ok:
        return PinCheck(role, repo, pinned, "FAIL", f"unreachable: {head.error}")

    if expected_arch:
        if expected_arch not in head.architectures:
            return PinCheck(
                role, repo, pinned, "FAIL",
                f"config declares {list(head.architectures) or '(none)'}, expected "
                f"{expected_arch!r}. The repo was repackaged or replaced under this name.",
                head_sha=head.sha, storage_gib=head.storage_gib)

    if not pinned:
        # Unpinned is a legitimate choice, but it must be a conscious one, so it
        # reports the sha that would have been the pin.
        return PinCheck(role, repo, pinned, "WARN",
                        f"no revision pinned; this boot serves whatever {head.sha[:12]} "
                        f"or newer. Last modified {head.last_modified or 'unknown'}.",
                        head_sha=head.sha, storage_gib=head.storage_gib)

    if pinned_view is None or not pinned_view.ok:
        detail = (pinned_view.error if pinned_view else "unreadable") or "unreadable"
        if detail == "not found":
            detail = "revision not found (deleted, or never existed under this repo)"
        return PinCheck(role, repo, pinned, "FAIL",
                        f"pinned revision does not resolve: {detail}. Head is "
                        f"{head.sha[:12]}; use it, or pick a revision that exists.",
                        head_sha=head.sha)

    drifted = not _matches(head.sha, pinned)
    if drifted:
        # Drift is a WARN, not a FAIL: a pin that resolves is still serving what
        # you chose. It is the upstream having moved on, which is information.
        return PinCheck(role, repo, pinned, "WARN",
                        f"pin resolves; upstream head moved to {head.sha[:12]} "
                        f"({head.last_modified or 'date unknown'}). Re-pin only after "
                        "re-running the acceptance benchmarks.",
                        head_sha=head.sha, storage_gib=pinned_view.storage_gib,
                        drifted=True)

    return PinCheck(role, repo, pinned, "PASS",
                    f"pin is current head ({pinned[:12]}); "
                    f"{pinned_view.architectures[0] if pinned_view.architectures else '?'}",
                    head_sha=head.sha, storage_gib=pinned_view.storage_gib)


def fetch(repo: str, revision: str = "", *, timeout: int = 30) -> RepoFacts:
    """Read one repo's facts from the Hub. Returns an error RepoFacts, never raises.

    Uses the system proxy, deliberately the opposite of the local engine calls.
    On the reference box the Hub is only reachable through the proxy -- direct
    connection hung past a 30 s timeout while the proxied request answered in 4.6 s.
    net.remote_opener owns that rule so the two cannot drift apart.
    """
    import urllib.error
    from . import net

    url = HF_MODEL_API.format(repo=repo)
    if revision:
        url += f"/revision/{revision}"
    opener = net.remote_opener()
    try:
        with opener.open(url, timeout=timeout) as response:
            return RepoFacts.from_json(repo, response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:       # noqa: BLE001 - an unreadable body must not mask the status
            body = ""
        if exc.code == 404:
            return RepoFacts(repo=repo, error="not found")
        facts = RepoFacts.from_json(repo, body) if body else RepoFacts(repo=repo, error=f"HTTP {exc.code}")
        return facts if facts.error else RepoFacts(repo=repo, error=f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError) as exc:
        return RepoFacts(repo=repo, error=f"{type(exc).__name__}: {exc}")


def verify_all(settings, profile, *, fetcher=fetch) -> list[PinCheck]:
    """Check the two repos this configuration would boot: the target, and the
    draft the active profile needs.

    Kept to exactly what a boot touches. Sweeping every known draft repo would
    cost more requests than it saves, and the ones you are not starting cannot
    fail your start.
    """
    model = settings.get("Q38_MODEL")
    model_rev = settings.get("Q38_MODEL_REVISION")
    checks = [check("target", model, model_rev, profiles.TARGET_ARCH,
                    fetcher(model),
                    fetcher(model, model_rev) if model_rev else None)]
    if profile.needs_draft:
        draft = settings.get("Q38_DRAFT_MODEL")
        draft_rev = settings.get("Q38_DRAFT_REVISION")
        checks.append(check(f"draft ({profile.name})", draft, draft_rev, profile.draft_arch,
                            fetcher(draft),
                            fetcher(draft, draft_rev) if draft_rev else None))
    return checks


def summarise(checks: list[PinCheck]) -> tuple[int, int, int]:
    return (sum(c.status == "PASS" for c in checks),
            sum(c.status == "WARN" for c in checks),
            sum(c.status == "FAIL" for c in checks))
