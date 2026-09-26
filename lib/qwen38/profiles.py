"""Serving profiles as data.

The three upstream recipes each own a pile of shell that spells flags, and the
flags disagree between them for reasons that live in prose comments. Encoding a
profile as a value means: the differences are diffable, the common core exists
once, and a sweep can generate profiles rather than copy-paste a script.

Two kinds of knob show up below and they are not the same thing:

  server flags    -- SGLang arguments, mostly steady-state throughput levers
  container bounds -- what we promise the host (memory, cpuset); these are the
                     coexistence surface and they are computed, never pinned
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Profile:
    name: str
    summary: str
    spec_flags: tuple[str, ...] = ()
    # Extra flags this profile needs beyond the shared core.
    extra_flags: tuple[str, ...] = ()
    # GDN radix cache strategy, rendered into the core rather than appended as a
    # duplicate. Two spellings of one flag rely on argparse last-wins, which works
    # and reads like an accident; the log then shows one "effective" value with no
    # hint that the profile fought the default.
    mamba_strategy: str = "extra_buffer_lazy"
    # Whether the profile can carry a YaRN-extended context. Upstream found
    # DSpark/DFlash2 draft configs inherit the rope override and crash the
    # validator; we encode that as a capability so the CLI can refuse cleanly
    # instead of booting a container that dies 6 minutes in.
    yarn_compatible: bool = False
    # GDN state slots per concurrent request. Upstream verified S=4 for
    # extra_buffer_lazy + the overlap scheduler by reading this build's
    # kv_cache_configurator, and S=3 only under MAMBA_SKIP_DECODE_LOCK=1. There
    # is no measurement for plain extra_buffer, so every profile keeps 4:
    # over-provisioning costs a little memory, while under-provisioning silently
    # clamps --max-running-requests, which is the worse failure. This is a sweep
    # knob -- record what the boot actually granted, do not trust this number.
    mamba_slots: int = 4
    # Static weight footprint in GiB, used to sanity-check the fitted fraction
    # before an 8-minute boot. These are Hub-measured (usedStorage), not
    # estimates: target 22.13, DFlash2 draft 3.58, DSpark draft 5.99. The
    # previous values were guesses and were wrong in both directions -- DSpark
    # was under-stated by more than 3 GiB, which is the one direction that makes
    # a budget check pass when it should warn. Re-check with `verify-pins`.
    approx_weight_gib: float = 22.1
    needs_draft: bool = False
    # The architecture a correct draft declares in its config.json. Checked
    # against the Hub by `verify-pins`, because a draft of the wrong packaging
    # loads and serves while being silently wrong -- r0b0tlab warn specifically
    # about the vLLM Qwen3DSparkModel build for this reason.
    draft_arch: str = ""

    def launch_flags(self) -> list[str]:
        return list(self.spec_flags) + list(self.extra_flags)


# Shared core: present in every profile, spelled once.
CORE_FLAGS: tuple[str, ...] = (
    "--trust-remote-code",
    "--attention-backend", "flashinfer",   # trtllm_mha is SM100-only; SM121 needs this
    "--kv-cache-dtype", "fp8_e4m3",
    "--mamba-ssm-dtype", "bfloat16",        # fp32 default doubles the state pool
    "--mamba-radix-cache-strategy", "{mamba_strategy}",
    "--chunked-prefill-size", "{chunked_prefill}",
    "--context-length", "{context_length}",
    "--max-running-requests", "{max_concurrent}",
    "--max-mamba-cache-size", "{mamba_cache_size}",
    "--mem-fraction-static", "{mem_fraction}",
    "--disable-prefill-cuda-graph",
    "--sleep-on-idle",                      # stops the idle scheduler busy-spin
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
    "--sampling-defaults", "model",
    "--enable-metrics",
    "--enable-cache-report",
    "--host", "{bind}",
    "--port", "{port}",
    "--served-model-name", "{served_name}",
)

# The architecture every Qwen3.8-27B export declares, checked against the Hub by
# `verify-pins`. Verified identical across four exports on 2026-09-26: the bf16
# base, Qwen's own FP8, and both RadixArk NVFP4 exports. That is what makes one
# constant correct for the family; a repackaged checkpoint under the same name
# fails here instead of booting, serving, and being quietly wrong.
TARGET_ARCH = "Qwen3_5ForConditionalGeneration"

PROFILES: dict[str, Profile] = {
    # No drafter at all. The reference every other number is compared against,
    # and the profile to run when a drafter is misbehaving and you need to know
    # whether it is the drafter. Slowest on decode, cheapest in memory, no
    # second checkpoint.
    "ar": Profile(
        name="ar",
        summary="plain autoregressive decode; the correctness and ceiling baseline",
        approx_weight_gib=24.0,
    ),
    # EAGLE over the checkpoint's own MTP head: no extra download, and the only
    # profile that can take YaRN, so it is the long-context option.
    "mtp": Profile(
        name="mtp",
        summary="EAGLE drafting from the in-checkpoint MTP head (3/1/4); YaRN-capable",
        spec_flags=(
            "--speculative-algorithm", "EAGLE",
            "--speculative-num-steps", "{spec_steps}",
            "--speculative-eagle-topk", "{spec_topk}",
            "--speculative-num-draft-tokens", "{spec_draft}",
        ),
        yarn_compatible=True,
    ),
    "dspark": Profile(
        name="dspark",
        summary="DSpark block drafter (RadixArk draft); block 7 was the code peak",
        spec_flags=(
            "--speculative-algorithm", "DSPARK",
            "--speculative-draft-model-path", "{draft_path}",
            "--speculative-draft-model-revision", "{draft_revision}",
            "--speculative-dspark-block-size", "{dspark_block}",
            "--speculative-draft-model-quantization", "unquant",
        ),
        needs_draft=True,
        approx_weight_gib=28.12,       # 22.13 target + 5.99 DSpark draft
        draft_arch="DSparkDraftModel",
    ),
    "dflash2": Profile(
        name="dflash2",
        summary="DFlash2 block-diffusion drafter (default): fastest code+essay, dense-head path",
        spec_flags=(
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", "{draft_path}",
            "--speculative-draft-model-revision", "{draft_revision}",
            "--speculative-num-draft-tokens", "{dflash_tokens}",
        ),
        # DFLASH rejects extra_buffer_lazy on the image we pin (upstream #34763
        # adds it; untested here), so it asks for the plain buffer strategy.
        mamba_strategy="extra_buffer",
        needs_draft=True,
        approx_weight_gib=25.71,       # 22.13 target + 3.58 DFlash2 draft
        draft_arch="DFlash2DraftModel",
    ),
}

DEFAULT_PROFILE = "dflash2"


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"unknown profile {name!r}; choose from "
            + ", ".join(sorted(PROFILES))
        ) from None


def with_context(profile: Profile, context_length: int) -> Profile:
    """Return the profile as it should actually run for a given window.

    Raises rather than launching a configuration that upstream documented as a
    crash: a draft config that inherits a rope override dies deep inside model
    loading, and the traceback does not mention context length anywhere.
    """
    if context_length <= 262144:
        return profile
    if not profile.yarn_compatible:
        raise SystemExit(
            f"profile {profile.name!r} cannot serve {context_length} tokens: the "
            "YaRN override leaks into the draft config and crashes the rope "
            "validator. Use profile 'mtp' for >262144, or lower the context."
        )
    return profile
