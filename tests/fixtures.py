"""Captured text from the reference box, so the tests assert real behaviour.

These are verbatim /proc and sysfs shapes from a DGX Spark (GB10, aarch64,
119.67 GiB unified) with a dozen business containers running. They are pinned
as text rather than read live so the suite runs on any machine and so a change
in the reference box cannot silently redefine what "correct" means.
"""

# /proc/meminfo, trimmed to the fields the code reads. Values are kB.
MEMINFO_HEALTHY = """MemTotal:       125483396 kB
MemFree:        13355956 kB
MemAvailable:   94849180 kB
Buffers:         5256824 kB
Cached:         69323224 kB
SwapTotal:      16777212 kB
SwapFree:       13096264 kB
"""

# The same box after something has eaten the pool: enough to make the point that
# MemAvailable, not MemFree, is the number to watch.
MEMINFO_STARVED = """MemTotal:       125483396 kB
MemFree:          312000 kB
MemAvailable:    2097152 kB
Buffers:            8192 kB
Cached:          1200000 kB
SwapTotal:      16777212 kB
SwapFree:        1048576 kB
"""

PSI_IDLE = """some avg10=0.00 avg60=0.00 avg300=0.00 total=8071105
full avg10=0.00 avg60=0.00 avg300=0.00 total=7939551
"""

PSI_UNDER_PRESSURE = """some avg10=41.20 avg60=12.30 avg300=4.10 total=8120000
full avg10=7.80 avg60=2.10 avg300=0.40 total=7940000
"""

# GB10 as enumerated on the reference box: 20 cores, two frequency bands,
# interleaved rather than contiguous. This is the shape a hardcoded
# "5-9,15-19" happens to match and a naive "first half is big cores" breaks on.
CPU_FREQS_GB10 = {
    **{i: 2808000 for i in range(0, 5)},
    **{i: 3900000 for i in range(5, 10)},
    **{i: 2808000 for i in range(10, 15)},
    **{i: 3900000 for i in range(15, 20)},
}

# A hypothetical SKU where the performance cores are not in the same slots.
# Used to prove the derivation, not the constant, is what we act on.
CPU_FREQS_SHUFFLED = {
    **{i: 3900000 for i in range(0, 4)},
    **{i: 2808000 for i in range(4, 12)},
    **{i: 3900000 for i in range(12, 16)},
}

IP_LOCAL_PORT_RANGE = "32768\t60999\n"

# torch.cuda.mem_get_info() on GB10: the CUDA-visible pool is essentially the
# whole unified memory, and nvidia-smi reports memory.total as N/A, which is
# exactly why this has to be measured from inside the container.
CUDA_PROBE_OUTPUT = '{"free_gib": 91.2, "total_gib": 119.7}\n'
