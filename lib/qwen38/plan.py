"""Assemble the exact launch plan, from measured inputs, without touching the box.

Kept separate from the shell that runs it for one reason: the plan is the
artifact worth recording and comparing. `qwen38 plan` prints it and changes
nothing, so you can see what a config change will do before spending eight
minutes on a boot, and evidence can store the plan verbatim next to the numbers
it produced.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
import re

from . import profiles
from . import rope


@dataclass(frozen=True)
class Inputs:
    """Everything the plan is a function of. Strings come from config + measurement."""

    model: str
    served_name: str
    image_ref: str
    container_name: str
    bind: str
    port: int
    context_length: int
    max_concurrent: int
    chunked_prefill: int
    mem_fraction: float
    profile: str
    draft_path: str | None = None
    draft_model: str | None = None
    # Revisions are their own arguments, never folded into the repo id. Writing
    # "org/repo@50307d4" asks HuggingFace to resolve a repo literally named that,
    # which fails at load time -- several minutes and one download into the boot.
    # Both upstream recipes pin with --speculative-draft-model-revision, and with a
    # full 40-char sha; a short sha is not something that flag promises to accept.
    draft_revision: str = ""
    model_revision: str = ""
    # Speculative knob defaults, chosen to match the measured upstream peaks.
    spec_steps: int = 3
    spec_topk: int = 1
    spec_draft: int = 4
    dspark_block: int = 7
    dflash_tokens: int = 8
    cpuset: str = ""
    container_mem: str = "96g"
    shm_size: str = "16g"
    privileged: bool = False
    cache_dir: str = ""
    # Refuse Hub lookups at boot. On a box where DNS poisons huggingface.co, an
    # online boot hangs in connect() forever (measured: 8 CPU-seconds in 10
    # minutes) while offline fails in seconds and names the missing file.
    offline: bool = False
    extra_server_args: tuple[str, ...] = ()
    extra_docker_args: tuple[str, ...] = ()
    # Measured CUDA pool, in GiB. Zero when it could not be measured, which
    # turns the budget check below off rather than making it guess.
    cuda_total_gib: float = 0.0


@dataclass(frozen=True)
class Plan:
    inputs: Inputs
    server_args: tuple[str, ...]
    docker_argv: tuple[str, ...] = field(default=())
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "image": self.inputs.image_ref,
            "profile": self.inputs.profile,
            "container": self.inputs.container_name,
            "listen": f"{self.inputs.bind}:{self.inputs.port}",
            "mem_fraction_static": self.inputs.mem_fraction,
            "cpuset": self.inputs.cpuset or None,
            "container_mem": self.inputs.container_mem,
            "context_length": self.inputs.context_length,
            "max_running_requests": self.inputs.max_concurrent,
            "server_args": list(self.server_args),
            "docker_argv": list(self.docker_argv),
            "notes": list(self.notes),
        }

    def render_server_command(self) -> str:
        return " ".join(shlex.quote(a) for a in ("python3", "-m", "sglang.launch_server", *self.server_args))

    def render_docker(self) -> str:
        return " ".join(shlex.quote(a) for a in self.docker_argv)


def _mamba_cache_size(max_concurrent: int, slots_per_request: int) -> int:
    """Pool size = concurrency x S, with S declared by the profile.

    The engine divides this pool by S alone and keeps the speculative verify
    window in a separate buffer, so folding draft tokens in (x8)
    over-provisions 2x -- and getting S too low silently clamps concurrency,
    which looks like a throughput bug rather than a config bug. The value is
    declared per profile, not inferred from the cache strategy, because the
    relationship between the two was only ever measured for extra_buffer_lazy.
    """
    return max_concurrent * slots_per_request


# A template that is nothing but one placeholder is a *value slot*. Flag names in
# these tuples are literals, so flag/value adjacency lets us drop a pair cleanly.
_VALUE_SLOT = re.compile(r"^\{(\w+)\}$")


def _render(flags: tuple[str, ...] | list[str], values: dict[str, str]) -> list[str]:
    """Format a flag template list, dropping pairs whose value is an empty slot.

    An optional pin that resolves to nothing must take its flag with it. Passing
    `--speculative-draft-model-revision` with an empty argument is an argparse
    error at best, and at worst it swallows the next flag as its value and boots a
    server configured with something nobody asked for.
    """
    out: list[str] = []
    for template in flags:
        rendered = template.format(**values) if "{" in template else template
        slot = _VALUE_SLOT.match(template)
        if slot and rendered == "":
            if out:
                out.pop()          # the flag this value belonged to
            continue
        out.append(rendered)
    return out


def build(inputs: Inputs) -> Plan:
    profile = profiles.get(inputs.profile)
    profile = profiles.with_context(profile, inputs.context_length)

    # Validated before rendering: a draft profile with no draft model is a
    # configuration error, not an empty string to be silently dropped.
    if profile.needs_draft and not inputs.draft_path:
        raise SystemExit(
            f"profile {profile.name} needs a draft model. Set Q38_DRAFT_MODEL, or "
            "use --profile mtp, which drafts from the target's own MTP head."
        )

    values = {
        "chunked_prefill": str(inputs.chunked_prefill),
        "context_length": str(inputs.context_length),
        "max_concurrent": str(inputs.max_concurrent),
        "mamba_cache_size": str(_mamba_cache_size(inputs.max_concurrent, profile.mamba_slots)),
        "mamba_strategy": profile.mamba_strategy,
        "full_memory_ratio": f"{profile.full_memory_ratio:g}",
        # An empty slot drops the flag with it (see _render), which is how a
        # native boot carries no rope override at all.
        "rope_override": (rope.override_json(rope.derive_factor(inputs.context_length))
                          if rope.needs_yarn(inputs.context_length) else ""),
        "mem_fraction": f"{inputs.mem_fraction:.3f}",
        "bind": inputs.bind,
        "port": str(inputs.port),
        "served_name": inputs.served_name,
        "spec_steps": str(inputs.spec_steps),
        "spec_topk": str(inputs.spec_topk),
        "spec_draft": str(inputs.spec_draft),
        "dspark_block": str(inputs.dspark_block),
        "dflash_tokens": str(inputs.dflash_tokens),
        "draft_path": inputs.draft_path or "",
        "draft_revision": inputs.draft_revision or "",
        "model_revision": inputs.model_revision or "",
    }

    core = _render(profiles.CORE_FLAGS, values)
    spec = _render(profile.launch_flags(), values)

    missing = [a for a in spec if "{" in a]
    if missing:
        raise SystemExit(f"profile {profile.name} references unset knobs: {missing}")

    server_args = [
        "--model-path", inputs.model,
        *_render(["--revision", "{model_revision}"], values),
        *core,
        *spec,
        *inputs.extra_server_args,
    ]

    # A cheap pre-flight arithmetic check, before an eight-minute boot tells us
    # the same thing with a worse traceback.
    notes: list[str] = []
    # KV is ~32.8 KB/token on this hybrid with fp8_e4m3, so a full advertised
    # window has a real price; the fitted fraction has to cover weights AND that.
    budget_gib = round(inputs.mem_fraction * inputs.cuda_total_gib, 1) if inputs.cuda_total_gib else 0.0
    if budget_gib and profile.approx_weight_gib:
        kv_gib = inputs.context_length * 32.8 * 1024 / (1024**3)
        need = round(profile.approx_weight_gib + kv_gib, 1)
        if need > budget_gib:
            notes.append(
                f"BUDGET SHORT: fraction {inputs.mem_fraction} of "
                f"{inputs.cuda_total_gib} GiB = {budget_gib} GiB, but weights "
                f"{profile.approx_weight_gib} + one {inputs.context_length}-token "
                f"sequence {kv_gib:.1f} = {need} GiB. The engine will boot with a KV "
                "pool smaller than the window it advertises."
            )
        else:
            notes.append(
                f"budget ok: {budget_gib} GiB vs {need} GiB needed "
                f"(weights {profile.approx_weight_gib} + one full sequence {kv_gib:.1f}), "
                f"{round(budget_gib - need, 1)} GiB spare."
            )

    docker = ["docker", "run", "-d", "--name", inputs.container_name, "--gpus", "all"]
    if inputs.privileged:
        docker += ["--privileged"]
    docker += [
        "--memory", inputs.container_mem,
        "--memory-swap", inputs.container_mem,
        "--shm-size", inputs.shm_size,
        "--network", "host",
        "--ipc", "host",
        "--restart", "no",
        "--label", "qwen38-spark.managed=1",
    ]
    if inputs.cpuset:
        docker += ["--cpuset-cpus", inputs.cpuset]
    docker += list(inputs.extra_docker_args)
    docker += [
        "-e", "HF_HOME=/root/.cache/huggingface",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor",
        "-e", "TRITON_CACHE_DIR=/root/.triton",
        # HF_HUB_OFFLINE is the one that matters here: it makes a cache miss an
        # immediate LocalEntryNotFoundError instead of a network attempt. It is
        # deliberately not TRANSFORMERS_OFFLINE too -- that flag is a broader
        # hammer other libraries read differently, and one lever that is well
        # understood beats two that are not.
        *(["-e", "HF_HUB_OFFLINE=1"] if inputs.offline else []),
        # Without this SGLang logs "User-specified context_length is greater than
        # the derived context_length", ignores --context-length, and serves the
        # native 262K while the config claims 1M. That is the silent half of the
        # YaRN recipe; MiaAI passes the same variable, and the granted context_len
        # in `logs` is what proves it took. Only set when a rope override is there
        # to authorise it, so a native boot carries neither.
        *(["-e", "%s=%s" % rope.ALLOW_LONGER_ENV]
          if rope.needs_yarn(inputs.context_length) else []),
        "-v", f"{inputs.cache_dir}/huggingface:/root/.cache/huggingface",
        "-v", f"{inputs.cache_dir}/triton:/root/.triton",
        "-v", f"{inputs.cache_dir}/inductor:/root/.cache/inductor",
        inputs.image_ref,
        "python3", "-m", "sglang.launch_server",
        *server_args,
    ]

    return Plan(inputs=inputs, server_args=tuple(server_args), docker_argv=tuple(docker), notes=tuple(notes))
