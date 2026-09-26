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

from . import profiles


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


def build(inputs: Inputs) -> Plan:
    profile = profiles.get(inputs.profile)
    profile = profiles.with_context(profile, inputs.context_length)

    values = {
        "chunked_prefill": str(inputs.chunked_prefill),
        "context_length": str(inputs.context_length),
        "max_concurrent": str(inputs.max_concurrent),
        "mamba_cache_size": str(_mamba_cache_size(inputs.max_concurrent, profile.mamba_slots)),
        "mamba_strategy": profile.mamba_strategy,
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
    }

    core = [v.format(**values) if "{" in v else v for v in profiles.CORE_FLAGS]
    spec = [v.format(**values) if "{" in v else v for v in profile.launch_flags()]

    missing = [a for a in spec if "{" in a]
    if missing:
        raise SystemExit(f"profile {profile.name} references unset knobs: {missing}")
    if profile.needs_draft and not inputs.draft_path:
        raise SystemExit(
            f"profile {profile.name} needs a draft model. Set Q38_DRAFT_MODEL or run "
            "with --no-draft to fall back to profile 'mtp'."
        )

    server_args = [
        "--model-path", inputs.model,
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
        "-v", f"{inputs.cache_dir}/huggingface:/root/.cache/huggingface",
        "-v", f"{inputs.cache_dir}/triton:/root/.triton",
        "-v", f"{inputs.cache_dir}/inductor:/root/.cache/inductor",
        inputs.image_ref,
        "python3", "-m", "sglang.launch_server",
        *server_args,
    ]

    return Plan(inputs=inputs, server_args=tuple(server_args), docker_argv=tuple(docker), notes=tuple(notes))
