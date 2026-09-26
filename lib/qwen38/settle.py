"""Wait until the host's free memory has stopped moving, then fit against it.

The bug this closes is specific and expensive: at boot, neighbours are still
starting, so MemAvailable is temporarily the highest it will ever be. A fraction
fitted at that instant is too large, the engine allocates against a promise the
box cannot keep, and the failure surfaces minutes later at CUDA graph capture --
which upstream repos each document as "the box hard-rebooted, root cause unclear"
and each fixed by lowering the constant. Lowering the constant hides the race
instead of closing it.

So: sample, and act only when the samples agree. The predicate is a real
convergence check (drift under `max_drift_gib` across `stable_samples`
consecutive observations) with a hard deadline, so a box that never settles
refuses to launch rather than guessing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    settled: bool
    drift_gib: float
    samples: tuple[float, ...]
    timed_out: bool = False


def evaluate(observations: list[float], *, max_drift_gib: float, stable_samples: int) -> Verdict:
    """Decide whether a series of MemAvailable readings is steady enough to fit.

    Drift is measured across the trailing window only: what matters is whether
    the number we are about to use is still changing, not what it did an hour
    ago. Fewer samples than the window asks for is always "not yet".
    """
    if len(observations) < stable_samples:
        return Verdict(settled=False, drift_gib=float("inf"), samples=tuple(observations))
    window = observations[-stable_samples:]
    drift = max(window) - min(window)
    return Verdict(settled=drift <= max_drift_gib, drift_gib=round(drift, 2), samples=tuple(window))


def wait_until_stable(
    read_available,
    *,
    max_drift_gib: float,
    stable_samples: int,
    interval_s: float,
    deadline_s: float,
    sleep=None,
    clock=None,
) -> tuple[Verdict | None, float]:
    """Poll `read_available()` until stable or the deadline passes.

    Returns (verdict_or_None, seconds_spent). None means the very first read
    failed, which the caller treats as "cannot tell" and refuses.

    sleep/clock are injected so the deadline logic is testable without anyone
    actually waiting. A test that sleeps is a test that gets deleted.
    """
    # Bound here rather than in the signature: a default of `time.sleep` in the
    # parameter list is captured at import and cannot be patched by a test.
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    started = clock()
    observations: list[float] = []
    while True:
        try:
            value = read_available()
        except OSError:
            value = None
        if value is None:
            # A read that fails mid-series does not invalidate what we already
            # saw; we report the window and let the caller decide.
            seen = Verdict(False, float("inf"), tuple(observations))
            return (seen if observations else None), clock() - started
        observations.append(float(value))
        verdict = evaluate(observations, max_drift_gib=max_drift_gib, stable_samples=stable_samples)
        if verdict.settled:
            return verdict, clock() - started
        if clock() - started >= deadline_s:
            return Verdict(False, verdict.drift_gib, verdict.samples, timed_out=True), clock() - started
        sleep(interval_s)
