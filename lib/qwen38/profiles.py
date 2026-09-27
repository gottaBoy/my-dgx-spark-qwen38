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
    # Headroom multiplier on the engine's own slots-per-request ratio. See
    # MAMBA_RATIO and the mamba_slots property: the ratio is what the engine
    # divides by to cap concurrency, and this multiplier is how much of the state
    # pool we keep free.
    mamba_slot_headroom: int = 2
    # KV-versus-state split of the static budget. MiaAI pins 4.21 and r0b0 4.59
    # against the engine's 0.9 default, and MiaAI's comment blames the default for
    # clamping concurrency. Tested on this build and that causality is wrong: with
    # 4.2 passed and 32 slots, the clamp to 6 persisted; the slot count is what
    # moved it. hasso does not set this flag at all and reports no clamp.
    # Kept because 0.9 is clearly the wrong shape for a 262K window, but no longer
    # claimed as the fix for concurrency -- see granted.py.
    full_memory_ratio: float = 4.2
    # hasso's determinism lever: flashinfer's autotuner chooses kernels per boot,
    # which is the "boot lottery" that makes morning-to-morning numbers
    # incomparable. A measurement you cannot repeat is not a measurement.
    disable_autotune: bool = True
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

    @property
    def mamba_slots(self) -> int:
        """State slots per request: the engine's ratio, times headroom.

        Derived, never configured. Two measurements on this box settle it:
        dflash2 (extra_buffer) with 32 slots granted max_running_requests=6 when
        8 were asked, and the same boot with 96 granted 8. Both follow from
        max_mamba_cache_size // ratio, with ratio 5 for extra_buffer and 4 for
        extra_buffer_lazy -- read out of the pinned image's own
        kv_cache_configurator._calculate_mamba_ratio, and confirmed against a
        third boot (ar, extra_buffer_lazy, 32 slots, granted 8).

        Headroom is not free -- ~78.4 MB per slot in BF16 -- but the sizing
        rule's exact minimum costs throughput: hasso measured ~14% of the
        concurrency-8 aggregate at the minimum, recovered by 64. So 2x.
        """
        return MAMBA_RATIO[self.mamba_strategy] * self.mamba_slot_headroom


# GDN state slots consumed per request, by radix cache strategy, with the overlap
# scheduler left on (our default). These are the engine's own constants:
# MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO (3) plus the V2_ADDITIONAL_* term
# (lazy overlap +1, non-lazy overlap +2). A clamp is silent in the metrics and
# loud in the log, which is why granted.py exists.
MAMBA_RATIO = {"extra_buffer_lazy": 4, "extra_buffer": 5}


# Shared core: present in every profile, spelled once.
CORE_FLAGS: tuple[str, ...] = (
    "--trust-remote-code",
    "--attention-backend", "flashinfer",   # trtllm_mha is SM100-only; SM121 needs this
    "--kv-cache-dtype", "fp8_e4m3",
    "--mamba-ssm-dtype", "bfloat16",        # fp32 default doubles the state pool
    "--mamba-full-memory-ratio", "{full_memory_ratio}",
    "--mamba-radix-cache-strategy", "{mamba_strategy}",
    "--disable-flashinfer-autotune",
    "--chunked-prefill-size", "{chunked_prefill}",
    "--context-length", "{context_length}",
    # Empty at or below the native window, which drops this flag with it -- see
    # plan._render. A native boot must not carry a rope override, because the
    # override is the thing that breaks the draft profiles.
    "--json-model-override-args", "{rope_override}",
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
        # MiaAI's measured DSpark stack (start-dspark.sh), copied rather than
        # reinvented. --enable-torch-compile pays minutes of boot time for a
        # steady-state win worth about +1 tok/s there; --num-continuous-decode-steps
        # 2 keeps the scheduler in decode for two steps before it reconsiders the
        # batch. The graph cap is MiaAI's own value for this drafter, not a
        # universal one -- see the dflash2 profile for why the two differ.
        extra_flags=("--enable-torch-compile", "--torch-compile-max-bs", "4",
                     "--cuda-graph-max-bs-decode", "4",
                     "--num-continuous-decode-steps", "2"),
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
        # hasso's measured DFlash2 stack (qwen38-sglang.service.template). Note
        # the two graph caps differ per drafter and neither is a general value:
        # hasso measured --cuda-graph-max-bs 8 at +6.5% aggregate at concurrency 8,
        # reproduced across boots, because with max-running-requests 8 the decode
        # batches of 5 to 8 were falling outside the captured graphs and running
        # eager -- about 0.4 GB of extra capture memory for it. torch-compile-max-bs
        # stays 4 because compiled graphs are per-shape and a cap above the batch
        # sizes actually seen buys compile time that never gets used.
        #
        # Measured cost of this stack on the box, from the boot that produced it:
        # 449 s to ready against 117 s without --enable-torch-compile. That trade is
        # hasso's to have made and mine to record, not to quietly re-litigate.
        extra_flags=("--enable-torch-compile", "--torch-compile-max-bs", "4",
                     "--cuda-graph-max-bs", "8",
                     "--num-continuous-decode-steps", "2"),
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
