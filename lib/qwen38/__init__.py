"""qwen38-spark: a measured, coexistence-first serving harness for Qwen3.8 on GB10.

Stdlib only. Every module here is pure logic over injected inputs so it can be
tested on a machine with no GPU at all; the parts that touch the box live in
adapters at the bottom of each module.
"""

__version__ = "0.1.0"
