"""YaRN: the rope override that makes >262144 possible, and its two traps.

This module exists because the first real boot exposed a gap between what
docs/TUNING-CURRICULUM.md promised and what plan.py did. Both upstream recipes
that serve a 1M window pass a JSON rope override plus one container env var:

  * MiaAI-Lab start.sh -- --json-model-override-args '{"text_config":
    {"rope_parameters": {...,"factor": F}}}' plus
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  * the model card validates factor 2.0 at 524288 and 4.0 at 1M; the factor is
    derived as round(context / 262144), so in-between windows get unvalidated
    factors (MiaAI says this explicitly rather than pretending otherwise)

The env var is not decoration. Without it SGLang logs "User-specified
context_length is greater than the derived context_length", *ignores* the
longer --context-length, and serves 262K while the config claims 1M -- a server
that advertises a window it cannot accept.

The second trap is why this is gated per profile: the override leaks into the
draft's ModelConfig on this build, injecting a text_config dict into a flat
config, and transformers' rope validator then dies on max_position_embeddings.
Six minutes into a boot, with a traceback that never mentions context length.
"""

from __future__ import annotations

import json

NATIVE_CONTEXT = 262144

# From the Qwen3.8 model card's SGLang recipe, copied through MiaAI-Lab's
# start.sh. Field order is irrelevant; the values are not.
ROPE_THETA = 10000000
MROPE_SECTION = [11, 11, 10]
PARTIAL_ROTARY_FACTOR = 0.25

# The env var SGLang needs or it silently keeps the native window.
ALLOW_LONGER_ENV = ("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")


def derive_factor(context_length: int) -> int:
    """round(context / native), floored at 1. The card validates 2 and 4."""
    return max(1, round(context_length / NATIVE_CONTEXT))


def needs_yarn(context_length: int) -> bool:
    return context_length > NATIVE_CONTEXT


def override_json(factor: int) -> str:
    """The --json-model-override-args payload for a given factor."""
    return json.dumps({"text_config": {"rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": list(MROPE_SECTION),
        "rope_type": "yarn",
        "rope_theta": ROPE_THETA,
        "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
        "factor": factor,
        "original_max_position_embeddings": NATIVE_CONTEXT,
    }}}, separators=(", ", ": "))


def context_args(context_length: int) -> tuple[list[str], list[tuple[str, str]]]:
    """(server args, container env) for a requested window.

    Returns the plain --context-length alone at or below native, so a native
    boot carries no rope override and cannot be blamed for one.
    """
    args = ["--context-length", str(context_length)]
    if not needs_yarn(context_length):
        return args, []
    return [*args, "--json-model-override-args", override_json(derive_factor(context_length))], \
        [ALLOW_LONGER_ENV]
