"""The coexistence watchdog's decision logic.

Why this exists at all: on this box `systemd-oomd` and `earlyoom` are both
inactive, and `nvidia-smi --query-gpu=memory.used` returns N/A on GB10. So the
two mechanisms an operator would normally reach for are unavailable, and the
only honest signals left are /proc/pressure (PSI) and MemAvailable. That is what
we watch.

The failure this guards against is documented in every upstream repo in the
same words -- "the box hard-rebooted, had to pull the power" -- and each of them
fixed it by lowering a constant. Lowering a constant is a bet about average
behaviour; this is a check against observed behaviour, which is the difference
between "usually safe" and "safe, and it will tell you when it is not".

Two strikes before acting. A single noisy sample must never stop an engine that
is mid-answer for somebody else; but two consecutive breaches at a 5 s interval
means 10 s of real pressure, which is not noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sample:
    """One observation of host memory pressure."""

    psi_some_avg10: float          # % of the last 10s in which some task stalled
    psi_full_avg10: float          # % where ALL tasks stalled (the lethal one)
    available_gib: float
    swap_used_gib: float = 0.0

    @classmethod
    def parse(cls, psi_text: str, meminfo_text: str) -> "Sample":
        psi_some = psi_full = 0.0
        for line in psi_text.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "some":
                psi_some = _avg10(parts)
            elif parts[0] == "full":
                psi_full = _avg10(parts)
        avail = 0.0
        swap_total = swap_free = 0
        for line in meminfo_text.splitlines():
            key, _, rest = line.partition(":")
            bits = rest.split()
            if key.strip() == "MemAvailable" and bits:
                avail = int(bits[0]) / (1024 * 1024)      # kB -> GiB
            elif key.strip() == "SwapTotal" and bits:
                swap_total = int(bits[0])
            elif key.strip() == "SwapFree" and bits:
                swap_free = int(bits[0])
        return cls(
            psi_some_avg10=psi_some,
            psi_full_avg10=psi_full,
            available_gib=round(avail, 2),
            swap_used_gib=round((swap_total - swap_free) / (1024 * 1024), 2),
        )


def _avg10(parts: list[str]) -> float:
    for part in parts:
        if part.startswith("avg10="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return 0.0
    return 0.0


@dataclass(frozen=True)
class Thresholds:
    psi_some: float = 25.0
    psi_full: float = 5.0
    available_floor_gib: float = 8.0
    strikes: int = 2


@dataclass
class Guard:
    """Stateful trip logic. `decide` is pure; the strikes field is the state."""

    thresholds: Thresholds
    strikes: int = 0
    tripped_reasons: list[str] = field(default_factory=list)

    def decide(self, sample: Sample) -> tuple[str, str]:
        """Return (verdict, reason) where verdict is ok | warn | trip.

        PSI alone is not enough: a box can sit at 0 pressure with 200 MB left
        right up until it dies, because pressure only appears once tasks are
        already stalled. The absolute floor catches that case, which is why both
        signals run on every sample.
        """
        reasons: list[str] = []
        if sample.psi_full_avg10 >= self.thresholds.psi_full:
            reasons.append(f"PSI full avg10 {sample.psi_full_avg10:.1f}%")
        if sample.psi_some_avg10 >= self.thresholds.psi_some:
            reasons.append(f"PSI some avg10 {sample.psi_some_avg10:.1f}%")
        if sample.available_gib < self.thresholds.available_floor_gib:
            reasons.append(f"MemAvailable {sample.available_gib:.1f} GiB below floor")

        if not reasons:
            self.strikes = 0
            return "ok", ""

        self.strikes += 1
        joined = "; ".join(reasons)
        if self.strikes >= self.thresholds.strikes:
            self.tripped_reasons.append(joined)
            return "trip", f"{joined} (after {self.strikes} consecutive breaches)"
        return "warn", f"{joined} (strike {self.strikes}/{self.thresholds.strikes})"


# --- where each sample comes from ------------------------------------------------

PSI_MEMORY_PATH = "/proc/pressure/memory"
MEMINFO_PATH = "/proc/meminfo"


def read_sample(psi_path: str = PSI_MEMORY_PATH, meminfo_path: str = MEMINFO_PATH) -> Sample:
    """Read one sample. PSI may be absent (kernel without configurable PSI);
    we degrade to the MemAvailable floor rather than crashing, and say so via
    psi values of 0.0, which can never trip the PSI rules but leave the floor
    active. Silent death would be worse than a partial guard."""
    try:
        with open(psi_path, "r", encoding="utf-8") as handle:
            psi_text = handle.read()
    except OSError:
        psi_text = ""
    with open(meminfo_path, "r", encoding="utf-8") as handle:
        meminfo_text = handle.read()
    return Sample.parse(psi_text, meminfo_text)
