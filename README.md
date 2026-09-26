# qwen38-spark

Serving Qwen3.8-27B NVFP4 with SGLang on an NVIDIA GB10 (DGX Spark), built for a
box that is already running other people's workloads.

The three published recipes for this hardware (MiaAI-Lab, hasso5703, r0b0tlab)
are excellent and all assume the machine is dedicated. This one does not. Every
number that the upstream recipes hardcode is computed from the box at launch
here, and every claim about performance is stored next to the configuration that
produced it.

```bash
./bin/qwen38 doctor        # what this box can actually give you, 12 checks
./bin/qwen38 fit           # solve --mem-fraction-static instead of guessing it
./bin/qwen38 start --dry-run   # the exact argv, without touching anything
```

## What is different, and why

| Upstream | Here | The reason |
|---|---|---|
| `--mem-fraction-static 0.95` / 0.90 / 0.76 / 0.70 / 0.50, pinned in a script or unit | computed at launch from `MemAvailable`, clamped to a band | A constant is a bet about average behaviour. On a box with 24 resident containers the fraction that fits depends on who else is awake, and the failure lands minutes later at CUDA graph capture |
| Boot the engine, fit the limits afterwards | `service-start` waits until host memory stops moving, then fits, and **refuses** if it never settles | At boot `MemAvailable` is at its temporary peak because the neighbours are still starting. Fitting then over-asks. This is the actual mechanism behind "hard reboot after graph capture, root cause unclear" |
| `CPUSET=5-9,15-19` hardcoded | derived from `scaling_max_freq` per core | The big/LITTLE split is a property of the SoC. A hardcoded pin is correct until the day it silently pins the scheduler to 2.8 GHz cores |
| Port 30000 / 8888 assumed free | probed at every launch, below the ephemeral range | On this box 30000 belongs to a business container that had been up 47 hours |
| `--privileged`, bind `0.0.0.0` | no privileged mode, bind `127.0.0.1` | An open unauthenticated inference port on a shared host is a liability, not a convenience |
| Tuning findings live in README prose and comment blocks | every launch writes a run record with its full plan; `compare` diffs two runs | A number without the configuration that made it cannot be reused or falsified |

## Layout

```
bin/qwen38          the CLI: 19 subcommands, stdlib only
lib/qwen38/         pure logic, tested offline on any machine
conf/config.defaults  every knob, one KEY="value" file, read by shell and Python alike
unit/               systemd template, rendered by install.sh
tests/              143 offline tests with captured /proc and sysfs fixtures
docs/               the operational tutorial and the tuning curriculum
state/              logs, the run ledger, watch records (gitignored)
```

## Install

```bash
git clone <this repo> && cd my-dgx-spark-qwen38
./install.sh --print-unit     # inspect what would be written, change nothing
./install.sh                  # render + enable the unit (starts nothing)
./bin/qwen38 fetch-image      # the engine image, 13.41 GiB compressed
./bin/qwen38 prefetch         # 25.72 GiB of weights, on the host
sudo systemctl start qwen38-spark     # 7-9 min boot
./bin/qwen38 canary && ./bin/qwen38 bench --save
```

`install.sh` never pulls an image and never downloads weights. Those are big,
visible, disk-consuming decisions and each has its own command.

`fetch-image` exists because `docker pull` does not work on the reference box:
dockerd carries no proxy, this network reaches the registry only through one, and
the pull stalled at 59 KiB/s. Reconfiguring dockerd would restart the 24
containers already running here, so the image arrives by registry client plus
`docker load` instead. `prefetch` exists for the same reason one level down: a
container here cannot reach huggingface.co at all (measured: connection timeout),
so weights that "download on first boot" never finish downloading.

## Reading

| If you want | Read |
|---|---|
| Deploy it, operate it, upgrade it, recover it | [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| Understand *why* a knob matters and how to measure it | [docs/TUNING-CURRICULUM.md](docs/TUNING-CURRICULUM.md) |
| The reasoning behind the design, in one pass | [docs/DESIGN.md](docs/DESIGN.md) |
| Every command | `./bin/qwen38 --help`, then `--help` on the subcommand |

## The four facts about this hardware that shaped the design

Measured on the reference box (DGX Spark, GB10, 119.67 GiB unified, aarch64),
not copied from a manual:

1. `nvidia-smi --query-gpu=memory.used,memory.total` returns **N/A**. The usual
   GPU-memory watchdog cannot work here at all. Pressure has to come from
   `/proc/pressure/memory` and `MemAvailable`.
2. `docker stats --no-stream` on all 24 running containers takes **over 60
   seconds**. Any monitoring loop that enumerates the whole host is useless; ask
   for one container by name and it returns in about a second.
3. `systemd-oomd` and `earlyoom` are **both inactive**. Nothing on the box
   protects you from an out-of-memory freeze, which is why `qwen38 guard` exists
   rather than being delegated to the distro.
4. A `docker run` on a missing image starts an **implicit 14 GB pull**, and
   killing the docker client does not stop the daemon-side download. This is why
   `start` refuses rather than pulling, and why the probe that measures the CUDA
   pool requires the image to be local first.
