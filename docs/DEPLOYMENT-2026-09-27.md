# Local deployment acceptance: 2026-09-27

This is a working shared-host baseline, not a dedicated-host benchmark or a
production stability certification. Only `my-dgx-spark-qwen38` was changed.
Docker and neighbouring services were not restarted; swap was not cleared.

## Deployment

| Item | Accepted value |
|---|---|
| Host | DGX Spark, GB10, aarch64, 119.67 GiB unified memory |
| Manager | `systemctl --user`, unit `qwen38-spark.service`, enabled |
| User lingering | `Linger=yes`; boot without login is configured, not reboot-tested |
| Container | `qwen38-spark`, not privileged |
| API base | `http://127.0.0.1:28100/v1` |
| Served model | `qwen3.8-27b` |
| Profile | DFlash2, 8 draft tokens |
| Context limit | 131072 tokens (128K), not 262K |
| Request limit | 8; not a claim that 8 full 128K windows fit |
| Effective KV pool | `max_total_num_tokens=250761` |
| GDN state slots | `max_mamba_cache_size=80`, `extra_buffer` strategy |
| Memory fraction | 0.500, with 32 GiB launch reservation |
| KV / state types | FP8 E4M3 KV / BF16 Mamba state |
| CPU affinity | `5-9,15-19`, 10 performance cores |
| Readiness | 10:53:58 Asia/Shanghai, 505 seconds after container launch |
| Run | `20260927T024533Z-dflash2` |

The endpoint binds loopback only and has no separately configured API key.
Remote access, an authenticated gateway, and Anthropic-client integration have
not been accepted in this deployment.

### Pinned components and borrowed implementation

- Image: `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`.
- Target: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`,
  revision `009632fef96dd349150baa780c984e62e70e91fe`.
- Draft: `z-lab/Qwen3.8-27B-DFlash2`,
  revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- MiaAI local fork: `9fb18edf8cfb3364e8aa89258e6d5ab1fe1fd11a`;
  DFlash2 launch parameters, cache handling and two-call benchmark method.
- hasso local fork: `6f21e1b3593418ed8ec4287268b1907728f180c2`;
  container-only deployment, offline weights, systemd lifecycle and compile
  limits. The local wrapper adds shared-host checks; it is not a new engine.

The run ledger records base commit `89ac9d3`; deployment changes were still
uncommitted at launch. Do not treat that SHA as a pristine reproduction of the
running code. A future clean-revision restart should establish clean provenance.
The installed unit also predates the template's logging-only
`PYTHONUNBUFFERED=1` addition; it can pick that up at the next planned reinstall.
No restart was done merely to align this logging setting.

## Acceptance and baseline

`./bin/qwen38 canary` passed 3/3: arithmetic without thinking (`437`),
arithmetic with thinking (`437`), and the `get_weather` tool-call path.
`/health` and `/v1/models` returned HTTP 200; the model list reports the 128K
limit.

`./bin/qwen38 bench --repeats 3 --save` used the upstream two-call net-decode
method and discarded a warmup call:

| Probe | Median tok/s | Range | Valid samples |
|---|---:|---:|---:|
| Code | 40.17 | 36.70-40.20 | 3 |
| Essay | 13.32 | 10.36-15.75 | 3 |

These are single-request decode estimates, not aggregate 8-request throughput,
end-to-end latency, or a quality comparison with another quantization.
The essay range is wide; do not advertise its median as a stable guarantee.
Different probe types must not be pooled into a "noise" percentage.

Repository checks passed: 221 offline tests, shell syntax checks for the
installer/uninstaller, `git diff --check`, and verification of the installed
user systemd unit. Offline tests do not replace the live-engine canaries.

### Evidence correction

The failed canary events at 02:51:24Z and 02:55:27Z were offline
connection-refusal tests, not engine requests. The tests inherited the live
state directory and appended simulated failures. The originals are preserved
and annotated in the ledger; CLI tests now use temporary state directories,
with a regression test for isolation and environment restoration.

## Memory: weights are not the serving footprint

- Cached weights: 25.72 GiB (22.14 target + 3.58 draft), measured on disk.
- Static serving budget: `0.50 * 119.67 = 59.84 GiB`; this is a sizing parameter,
  not an exact process-memory measurement or a hard unified-memory cap.
- Launch-time `MemAvailable` in the ledger: 92.93 GiB.
- In 36 post-readiness samples, 02:53:59Z through 03:00:25Z, host available
  memory ranged from 30.34 to 45.05 GiB; the last sample was 44.36 GiB.
- Maximum sampled PSI some/full in that window: 0.22% / 0.22%.
- Swap occupancy ranged from 12.03 to 12.63 GiB. A separate `vmstat 1 5` sample
  showed small swap-ins (4-228 KiB/s after the first row) and no swap-outs.
  This short observation does not establish longer-term paging behaviour.

The host has other active workloads. Available-memory changes cannot be
attributed exclusively to this container. Neither container cgroup memory nor
`systemctl status` supervisor memory measures all CUDA unified allocations.
This deployment is therefore not a verified "under 30 GiB total RAM" setup.

## Operate the accepted baseline

```bash
cd /home/my/workspace/llm/my-dgx-spark-qwen38
systemctl --user status qwen38-spark --no-pager
./bin/qwen38 status
./bin/qwen38 metrics
./bin/qwen38 logs --tail 100
journalctl --user -u qwen38-spark -n 100 --no-pager
```

The service supervises its memory guard during startup and serving. Before
readiness, PSI breaches warn; the 8 GiB available-memory floor remains active.
After readiness, PSI some >=25% or full >=5%, or available memory below the
floor, trips after two consecutive 5-second samples. Readiness is latched.
A trip stops only this container and intentionally avoids automatic restart.
This is best-effort protection, not a guarantee against a host freeze.

`observe --interval 10` is an optional foreground recorder; the acceptance
recorder is not a permanently installed monitoring dashboard.

## Upgrade and tuning order

1. Use [OPERATIONS.md](OPERATIONS.md) for installation, pinned image/model
   upgrades and rollback. Keep `conf/config.local` and caches; record the
   known-good revision before replacing anything.
2. Stop only our user unit before changing deployment code or pins. Run the
   offline tests, inspect the rendered unit, reinstall into the same manager,
   start, wait for health, then repeat canary and baseline.
3. Do not clear swap, restart Docker, or stop neighbouring services to improve
   a benchmark. Do not change the image, fraction, concurrency and context in
   one experiment.
4. Follow [TUNING-CURRICULUM.md](TUNING-CURRICULUM.md): record a control, change
   one variable, repeat the control, compare configurations before throughput.
5. Keep this baseline until sustained-load and longer-context tests are done.
   Upstream nominal context support is not a local acceptance result.

## Remaining limits

- No sustained 8-request load test, full 128K prompt test, soak test, or host
  reboot test has been completed.
- The current image predates the upstream zombie-request cancellation fix;
  an abandoned request can keep decoding. Upgrade the pinned image in a
  separate maintenance window and test cancellation before exposing the API.
  See [UPSTREAM-ANALYSIS.md](UPSTREAM-ANALYSIS.md).
- Three short canaries do not prove quantization quality or all tool schemas.
- Basic deployment is accepted. Optimization and production readiness are
  separate stages, not implied by a green health endpoint.
