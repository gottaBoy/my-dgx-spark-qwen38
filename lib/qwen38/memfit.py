"""Turn --mem-fraction-static from a pinned constant into a measured budget.

Every published Spark recipe hardcodes this number (0.95, 0.90, 0.76, 0.70,
0.50) and none of them says what the fraction is taken *of*. That is the whole
reason a recipe that works on a dedicated box hard-reboots a shared one: the
denominator is the CUDA-visible pool, but the constraint that actually kills
you is what the *host* has left after everyone else's containers ate.

On GB10 the two are the same physical RAM, and nvidia-smi reports memory.total
as N/A, so the denominator cannot be read from the usual place. It is measured
instead (see cuda_total_gib below), and the fit is pure arithmetic:

    askable_gib = MemAvailable - reserved_for_everyone_else
    fraction    = askable_gib / cuda_total_gib        (clamped)

The clamp matters in both directions. At the top it is what stops a boot from
reaching zero host memory. At the bottom it is a refusal: below ~0.45 the
engine cannot hold a 27B NVFP4 target plus a draft plus a usable KV pool, and
launching anyway produces a container that thrashes the box for an hour before
dying somewhere that looks unrelated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB


@dataclass(frozen=True)
class HostMemory:
    """What the host reports, in GiB, at one instant."""

    total_gib: float
    available_gib: float
    swap_used_gib: float = 0.0

    @classmethod
    def from_meminfo(cls, text: str) -> "HostMemory":
        """Parse /proc/meminfo. Its values are in kB, not bytes; the unit is
        asserted, because a silent 1024x error here reads as "the box is out of
        memory" and would refuse every launch on a healthy machine."""
        vals = {}
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if len(parts) >= 2 and parts[1] == "kB":
                vals[key.strip()] = int(parts[0])
        try:
            total = vals["MemTotal"]
            avail = vals["MemAvailable"]
            swap_total = vals.get("SwapTotal", 0)
            swap_free = vals.get("SwapFree", swap_total)
        except KeyError as exc:  # pragma: no cover - malformed /proc
            raise ValueError(f"/proc/meminfo missing {exc}") from None
        return cls(round(total * KIB / GIB, 2), round(avail * KIB / GIB, 2),
                   round(max(0, swap_total - swap_free) * KIB / GIB, 2))


@dataclass(frozen=True)
class Fit:
    """A solved budget, plus every input that produced it.

    The inputs travel with the answer because a fraction with no provenance is
    just a superstition with more steps. This dict is what lands in the run
    evidence, so a month later you can ask "why 0.62 on that boot" and get a
    number instead of a memory.
    """

    fraction: float
    cuda_total_gib: float
    cuda_total_source: str
    available_gib: float
    reserved_gib: float
    askable_gib: float
    bound_by: str
    clamped: bool = False
    warnings: tuple[str, ...] = ()
    swap_used_gib: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def solve(
    host: HostMemory,
    cuda_total_gib: float | None,
    *,
    reserved_gib: float,
    min_fraction: float,
    max_fraction: float,
    cuda_total_source: str = "measured",
    swap_ceiling_gib: float = 2.0,
) -> Fit:
    """Fit a static-memory fraction to what this box can actually give.

    cuda_total_gib is None when the denominator could not be measured. We do
    not silently substitute a guess: the caller gets max_fraction with a loud
    warning, which is the behaviour of every other recipe, but now it is
    labelled as the fallback rather than presented as knowledge.

    swap_ceiling_gib is diagnostic, not a second allocation charge. Swap can
    retain inactive pages long after pressure has subsided; usage alone does
    not prove current paging. Correlate it with vmstat si/so and PSI when
    interpreting a benchmark.
    """
    warnings: list[str] = []
    askable = round(host.available_gib - reserved_gib, 2)

    # Swap usage can persist after pressure subsides. Treat it as a diagnostic,
    # not a second charge on the operator's reservation or proof of active I/O.
    if host.swap_used_gib > swap_ceiling_gib:
        warnings.append(
            f"{host.swap_used_gib:.1f} GiB in swap (ceiling {swap_ceiling_gib} GiB). "
            "Check vmstat si/so and PSI for active paging before comparing tok/s. "
            "Swap usage alone does not prove current pressure; not charged again."
        )

    if cuda_total_gib is None or cuda_total_gib <= 0:
        # Unknown denominator: fall back to the ceiling and say so.
        return Fit(
            fraction=max_fraction,
            cuda_total_gib=cuda_total_gib or 0.0,
            cuda_total_source="unmeasured-fallback",
            available_gib=host.available_gib,
            reserved_gib=reserved_gib,
            askable_gib=askable,
            bound_by="fallback",
            clamped=True,
            swap_used_gib=host.swap_used_gib,
            warnings=tuple(warnings) + (
                "CUDA pool size not measured; using the configured ceiling "
                f"{max_fraction}. Run `qwen38 fit` on the box (without --no-probe) "
                "to measure the pool through the pinned image.",
            ),
        )

    if askable <= 0:
        # Nothing left after the reservation: refuse, do not launch.
        return Fit(
            fraction=0.0,
            cuda_total_gib=cuda_total_gib,
            cuda_total_source=cuda_total_source,
            available_gib=host.available_gib,
            reserved_gib=reserved_gib,
            askable_gib=askable,
            bound_by="no-headroom",
            clamped=True,
            swap_used_gib=host.swap_used_gib,
            warnings=(
                f"MemAvailable {host.available_gib} GiB is at or under the "
                f"{reserved_gib} GiB reservation. Refusing to launch: free some "
                "host memory or lower Q38_RESERVED_GIB if you own this box.",
            ),
        )

    raw = askable / cuda_total_gib
    fraction = raw
    bound_by = "askable"
    clamped = False

    if fraction > max_fraction:
        fraction, bound_by, clamped = max_fraction, "max_fraction", True
    elif fraction < min_fraction:
        fraction, bound_by, clamped = 0.0, "no-headroom", True
        warnings.append(
            f"Budget wants {raw:.3f}, below the {min_fraction} floor. Refusing "
            "to launch: clamping upward would spend reserved neighbour memory. "
            "Wait for sufficient host headroom."
        )
    else:
        # Floor, never round. The fraction multiplies back into an allocation,
        # so the direction of the error is the whole question: rounding up can
        # spend memory we explicitly reserved for the neighbours. Found by the
        # test that asserts fraction*pool <= askable -- a round(,3) here passed
        # the band and still over-asked by 0.003 GiB.
        fraction = math.floor(raw * 1000) / 1000

    return Fit(
        fraction=fraction,
        cuda_total_gib=cuda_total_gib,
        cuda_total_source=cuda_total_source,
        available_gib=host.available_gib,
        reserved_gib=reserved_gib,
        askable_gib=askable,
        bound_by=bound_by,
        clamped=clamped,
        warnings=tuple(warnings),
        swap_used_gib=host.swap_used_gib,
    )


# --- measurement ---------------------------------------------------------------


# What to run inside the container to learn the denominator. mem_get_info is
# the right API, not mem_prop: it returns (free, total) for the device as CUDA
# sees it, and SGLang's own sizing starts from the same call, so measuring here
# measures the same quantity the engine will divide by.
MEASURE_CUDA_TOTAL = (
    "import json,"
    "torch;"
    "f,t=torch.cuda.mem_get_info();"
    "print(json.dumps({'free_gib':round(f/%d,2),'total_gib':round(t/%d,2)}))" % (GIB, GIB)
)


def parse_cuda_total(stdout: str) -> float | None:
    """Pull total_gib out of MEASURE_CUDA_TOTAL's output; None if unusable."""
    import json

    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = data.get("total_gib")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def read_host_memory(path: str = "/proc/meminfo") -> HostMemory:
    with open(path, "r", encoding="utf-8") as handle:
        return HostMemory.from_meminfo(handle.read())


def fit_for_box(
    *,
    reserved_gib: float,
    min_fraction: float,
    max_fraction: float,
    meminfo_path: str = "/proc/meminfo",
    cuda_total_gib: float | None = None,
    cuda_total_source: str = "measured",
) -> Fit:
    """One-call entry point used by the CLI."""
    host = read_host_memory(meminfo_path)
    return solve(
        host,
        cuda_total_gib,
        reserved_gib=reserved_gib,
        min_fraction=min_fraction,
        max_fraction=max_fraction,
        cuda_total_source=cuda_total_source,
    )
