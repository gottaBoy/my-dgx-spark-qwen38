"""Port selection as a check, not a constant.

Every upstream recipe ships a fixed port (8888, 30000, 30001, 30090) and says
nothing about what happens if the box already owns it. On this box 30000 is
taken by a business container that has been up for 47 hours, and several more
sit in the same band. The engine must never be the reason someone's staging
service falls over, so: we probe, we report, and we refuse rather than fight.

Detection reads what the OS says is listening. We do not parse `docker ps`
output to name the owner (that would be a second, weaker source of truth), but
we do ask Docker for a hint when it is available, purely to make the refusal
message actionable for a human.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    port: int
    free: bool
    note: str = ""


def is_free(port: int, host: str = "127.0.0.1", timeout: float = 0.35) -> bool:
    """True when nothing answered on host:port.

    A connect() that is refused means the port is open for use. Anything else
    (a completed handshake, a timeout) we treat as taken: on a shared box a
    wrong "free" costs someone else's outage, while a wrong "taken" costs us one
    retry. So the bias goes towards calling it busy.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
        except ConnectionRefusedError:
            return True
        except OSError:
            return False
        return False


def ephemeral_floor(proc_path: str = "/proc/sys/net/ipv4/ip_local_port_range") -> int:
    """Lower bound of the kernel's ephemeral range. Ports at/above it can be
    grabbed by any outgoing connection on the box, so a long-lived service
    should live below it. Returns 0 if unreadable, meaning "no opinion"."""
    try:
        with open(proc_path, "r", encoding="utf-8") as handle:
            low, _ = handle.read().split()[:2]
        return int(low)
    except (OSError, ValueError):
        return 0


def scan(ports: list[int], host: str = "127.0.0.1") -> list[Probe]:
    floor = ephemeral_floor()
    out: list[Probe] = []
    for port in ports:
        free = is_free(port, host)
        note = ""
        if not free:
            note = "already listening; another workload owns it"
        elif floor and port >= floor:
            note = f"free but inside the ephemeral range (starts {floor}); may be squatted"
        out.append(Probe(port=port, free=free, note=note))
    return out


def choose(candidates: list[int], host: str = "127.0.0.1") -> tuple[int | None, list[Probe]]:
    """First candidate that is free and outside the ephemeral range.

    Returns None when nothing qualifies; the caller must then refuse with the
    probe list rather than fall back to the first candidate.
    """
    probes = scan(candidates, host)
    floor = ephemeral_floor()
    for probe in probes:
        if not probe.free:
            continue
        if floor and probe.port >= floor:
            continue
        return probe.port, probes
    return None, probes
