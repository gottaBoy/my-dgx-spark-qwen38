"""Compare what the launch asked for against what the engine actually granted.

SGLang recomputes several of its own limits after it sizes the memory pool, and
it does so by printing a line and moving on. Ask for 8 concurrent requests on a
box where the GDN pool came out differently and you get 6, silently: the server
is healthy, the API answers, and throughput is just lower than the config says it
should be. That gap is where tuning notes go wrong -- you measure the engine that
actually ran, attribute the result to the config you meant to run, and the
conclusion is about nothing.

Found here the first time the stack booted for real: requested
max_running_requests=8, granted 6, with mamba_full_memory_ratio left at its 0.9
default, which is the value upstream says over-provisions KV and clamps
concurrency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The line SGLang prints once the pool is sized. Everything on it is *granted*,
# not requested, which is the entire point.
_GRANT_PREFIXES = ("max_total_num_tokens=", )


@dataclass(frozen=True)
class Delta:
    key: str
    asked: object
    granted: object
    meaning: str

    @property
    def clamped(self) -> bool:
        return self.granted != self.asked

    def render(self) -> str:
        if not self.clamped:
            return f"ok      {self.key} = {self.granted}"
        return (f"CLAMPED {self.key}: asked {self.asked}, granted {self.granted}"
                f" -- {self.meaning}")


def _numbers(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in re.finditer(r"\b([a-z_0-9]+)=([0-9]+(?:\.[0-9]+)?)", line):
        out.setdefault(match.group(1), match.group(2))
    return out


def parse_granted(log_text: str) -> dict[str, str]:
    """Every numeric field on the engine's post-sizing line, last one wins.

    A restart within one log file appends a fresh line; reading the last is what
    describes the currently-running engine.
    """
    granted: dict[str, str] = {}
    for line in log_text.splitlines():
        if not any(prefix in line for prefix in _GRANT_PREFIXES):
            continue
        granted.update(_numbers(line))
    return granted


def parse_requested(log_text: str) -> dict[str, str]:
    """The server_args echo: what the process was handed, before any resize."""
    requested: dict[str, str] = {}
    for line in log_text.splitlines():
        if "server_args=ServerArgs" not in line:
            continue
        requested.update(_numbers(line))
    return requested


# The comparisons worth failing on, and one line each on why they matter. Keys are
# the engine's own names on both lines, so a rename shows up as a missing field
# rather than a false agreement.
#
# The max_running_requests text used to name mamba_full_memory_ratio as the usual
# culprit, copied from MiaAI's comment. Measured here and it is wrong: with
# --mamba-full-memory-ratio 4.2 passed and 32 slots the clamp to 6 held, and with
# the same 4.2 and 96 slots the clamp went away. Slot headroom is the lever;
# carrying a confident wrong cause in a diagnostic is worse than carrying none,
# because it sends the reader to change the one knob that will not help.
CHECKS: dict[str, str] = {
    "max_running_requests":
        "fewer concurrent requests than configured, so aggregate throughput is "
        "bounded here. Usually GDN slot headroom: raise mamba_slots, not the "
        "memory fraction",
    "context_len":
        "the window the server will actually accept; a mismatch with the config means "
        "a rope/YaRN override did not take",
    "max_mamba_cache_size":
        "GDN state pool slots; the sizing rule's exact minimum is a throughput "
        "cost, not just a memory saving (hasso measured ~14% at the minimum)",
}

# The two lines do not share a vocabulary: server_args echoes context_length and the
# sizing line prints context_len. Without this the interesting check silently
# compares nothing, which is the worst outcome a comparison can have -- it looks
# like agreement.
REQUESTED_ALIASES = {"context_len": "context_length"}


def compare(requested: dict[str, str], granted: dict[str, str]) -> list[Delta]:
    """Deltas for keys present on both sides, after aliasing.

    A key missing on either side is reported separately by missing_checks rather
    than skipped in silence: an engine rename should surface as a gap, not as a
    clean report.
    """
    out: list[Delta] = []
    for key, meaning in CHECKS.items():
        asked_key = REQUESTED_ALIASES.get(key, key)
        if asked_key in requested and key in granted:
            out.append(Delta(key, requested[asked_key], granted[key], meaning))
    return out


def missing_checks(requested: dict[str, str], granted: dict[str, str],
                   compared: list[Delta]) -> list[str]:
    """Keys we meant to compare and could not, named."""
    done = {delta.key for delta in compared}
    return [key for key in CHECKS if key not in done]


def headroom(granted: dict[str, str]) -> str:
    """The number to watch on GB10 after graph capture, straight off the log."""
    return granted.get("available_gpu_mem", "")


# --- engine lifetime decision ---------------------------------------------------
# Lives here rather than in the unit file because the unit cannot know what the
# wrapper knows: whether a vanished container was asked for.
#
# hasso needs Restart=always because its ExecStart is `docker run` directly, so an
# engine that exits 0 on a Triton compile crash looks like success to systemd and
# on-failure never relaunches it. Here a supervisor sits in between and can ask a
# question the unit cannot: did anyone request this stop? That is worth knowing,
# because a deliberate stop that returns non-zero marks the unit "failed" and
# trains the operator to ignore the state, which is the failure mode I wrote
# SuccessExitStatus to avoid.


def follow_exit_code(container_running: bool, stop_requested: bool) -> int:
    """Exit code for the log-following supervisor when it loses the container.

    0 for a stop someone asked for, 1 for an unrequested death. Returning 0 for
    both would leave the unit "inactive" beside a crashed engine with
    Restart=on-failure never firing -- the one failure shape in this design that
    would go completely unnoticed.
    """
    if container_running:
        return 0                      # follow ended another way; engine is fine
    return 0 if stop_requested else 1
