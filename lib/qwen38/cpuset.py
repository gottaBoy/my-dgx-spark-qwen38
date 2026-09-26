"""Derive the performance-core cpuset from sysfs instead of trusting a string.

Every published recipe ships CPUSET="5-9,15-19" for GB10. That is correct for
the Spark as shipped, and it is a hardcoded fact about the silicon -- the kind
that quietly misleads you the first time the core enumeration differs (a
different SKU, a renamed topology, a VM with a shuffled CPU order). Pinning the
scheduler to efficiency cores by mistake costs real throughput and shows up as
a confusing benchmark regression.

So: read scaling_max_freq per core, take the top frequency band, and report the
set. "auto" means derive at launch; an explicit value is honoured verbatim so a
known-good pin can still be pinned for a controlled experiment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class CoreTopology:
    """cpu id -> max frequency in kHz."""

    freqs: dict[int, int]

    @property
    def big_freq(self) -> int:
        return max(self.freqs.values())

    def big_cores(self) -> list[int]:
        return sorted(cpu for cpu, freq in self.freqs.items() if freq == self.big_freq)

    def little_cores(self) -> list[int]:
        return sorted(cpu for cpu, freq in self.freqs.items() if freq != self.big_freq)


def parse_cpu_list(text: str) -> list[int]:
    """'0,2-4' -> [0,2,3,4]. Rejects anything that is not a CPU range list."""
    out: list[int] = []
    text = text.strip()
    if not text:
        return out
    for part in text.split(","):
        match = _RANGE.match(part.strip())
        if not match:
            raise ValueError(f"bad cpuset fragment: {part!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            raise ValueError(f"reversed cpuset range: {part!r}")
        out.extend(range(start, end + 1))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate cpu in cpuset: {text!r}")
    return out


def format_cpu_list(cpus: list[int]) -> str:
    """[5,6,7,8,9,15,16] -> '5-9,15-16'. Docker wants ranges, humans read them."""
    if not cpus:
        return ""
    cpus = sorted(cpus)
    chunks: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        chunks.append(_chunk(start, prev))
        start = prev = cpu
    chunks.append(_chunk(start, prev))
    return ",".join(chunks)


def _chunk(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def read_topology(sysfs_root: str = "/sys/devices/system/cpu") -> CoreTopology:
    """Best-effort read of the host topology. Empty result means 'cannot tell'."""
    freqs: dict[int, int] = {}
    for name in os.listdir(sysfs_root) if os.path.isdir(sysfs_root) else []:
        match = re.match(r"^cpu(\d+)$", name)
        if not match:
            continue
        path = os.path.join(sysfs_root, name, "cpufreq", "scaling_max_freq")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                freqs[int(match.group(1))] = int(handle.read().strip())
        except (OSError, ValueError):
            continue
    return CoreTopology(freqs)


def resolve(requested: str, sysfs_root: str = "/sys/devices/system/cpu") -> tuple[str, str]:
    """Return (cpuset_for_docker, how_we_know). 'auto' derives; a value passes through."""
    if requested and requested != "auto":
        cpus = parse_cpu_list(requested)
        return format_cpu_list(cpus), f"explicit ({len(cpus)} cores)"

    topo = read_topology(sysfs_root)
    if not topo.freqs:
        # Cannot see frequencies (no cpufreq driver, restricted container).
        # Falling back to "no pinning" is the honest answer: unpinned schedulers
        # are slower, but pinning to a guess can be much slower.
        return "", "unavailable - no cpufreq, leaving unpinned"

    big = topo.big_cores()
    note = (
        f"derived {len(big)} performance cores at {topo.big_freq // 1000} MHz "
        f"(efficiency set: {format_cpu_list(topo.little_cores()) or 'none'})"
    )
    return format_cpu_list(big), note
