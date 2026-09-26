# Deployment and operations

The working procedure for this stack: first deploy, day two operate, and the
upgrade path that keeps both. Every command here is real; the outputs are from
the reference box (DGX Spark / GB10, 119.67 GiB unified, ~24 business containers
resident).

## 0. Before anything else: learn what the box is

Do not skip this. Three of the four things that will surprise you about serving
on GB10 are properties of *this specific machine*, not of the model.

```bash
./bin/qwen38 doctor
```

```
  PASS docker daemon         29.2.1
  PASS our container free    not running
  WARN pinned image present  lmsysorg/sglang@sha256:616a3e97...  -- run: qwen38 pull --pull
  PASS GPU visible to host   GPU 0: NVIDIA GB10 (UUID: GPU-982473cf-...)
  PASS port 28100            free
  PASS port 28101            free
  PASS usable engine port    28100
  PASS cpuset                5-9,15-19 -- derived 10 performance cores at 3900 MHz (efficiency set: 0-4,10-14)
  PASS host memory readable  91.19 GiB available of 119.67
  PASS memory pressure       PSI some 0.0% / full 0.0% / avail 91.19 GiB
  PASS cache disk >= 45 GiB  2275 GiB free at /home/my/.cache/qwen38-spark
  WARN host OOM daemon       inactive inactive -- qwen38 guard supplies this role
```

Read the two WARNs as instructions, not noise:

* **image not present** is expected on a fresh box. Installing does not need it.
* **no OOM daemon** is the important one. `systemd-oomd` and `earlyoom` are both
  inactive here, so nothing on this machine will stop a process that eats all the
  RAM. That gap is the reason `guard` exists; if your box has oomd active, you
  have a second line of defence and can be less conservative.

Two facts `doctor` cannot tell you but you should know before tuning:
`nvidia-smi` reports `memory.used` and `memory.total` as `N/A` on GB10, so the
GPU-memory dashboards you may reach for are blank by construction; and
`docker stats --no-stream` without a container argument takes over a minute on a
busy box, so never put it in a loop.

## 1. Deploy

```bash
./install.sh --print-unit      # inspect the rendered systemd unit; writes nothing
./install.sh                   # render, install, enable the unit; starts nothing
```

`install.sh` deliberately does not download anything. The two large fetches are
separate, visible commands, because on a shared machine they are disk and
bandwidth decisions someone should make on purpose:

```bash
./bin/qwen38 fetch-image      # the engine image: 13.41 GiB compressed, arm64
./bin/qwen38 prefetch         # the weights: 25.72 GiB (22.14 target + 3.58 draft)
```

Both run on the host, and on the reference box they have to. Two measured facts:

* `docker pull` of the engine image stalled at **59 KiB/s**, because dockerd
  carries no proxy and this network reaches the registry only through one.
  Reconfiguring dockerd would restart the 24 containers already running here, so
  `fetch-image` uses a proxy-honouring registry client and hands the result to
  `docker load`. dockerd never needs the network.
* A container here cannot reach huggingface.co at all (verified from inside one:
  connection timed out). So the "weights download themselves on first boot"
  behaviour every upstream recipe relies on is not a slow start here, it is a
  hang. `prefetch` lands them in the cache the container mounts.

Use `pull --pull` instead of `fetch-image` only where dockerd itself has egress.

`prefetch` needs `huggingface_hub`, which is deliberately not a dependency of the
CLI (the CLI is stdlib-only, so the suite runs anywhere):

```bash
python3 -m venv .venv && .venv/bin/pip install huggingface_hub
```

Both are resumable. Interrupted downloads leave `.incomplete` blobs that the
client continues from, which is why `qwen38 doctor` reports in-flight bytes
rather than treating a partial cache as either empty or ready.

Then start:

```bash
sudo systemctl start qwen38-spark
watch -n10 ./bin/qwen38 status
```

Expect 7-9 minutes on a cold boot (CUDA graph capture plus kernel compilation),
5-7 afterwards. The unit runs `service-start`, which is `start` with one
addition: it waits for host memory to stop moving before it fits a fraction.

### No systemd? Run it in the foreground

```bash
./install.sh --no-service
./bin/qwen38 start
```

Same fitted fraction, same port probe, same ledger entry. The difference is that
nobody restarts it when the box does.

## 2. Accept it (do not skip this step)

A server that answers is not a server that is correct. Run the gate before you
believe any number that comes after it:

```bash
./bin/qwen38 canary
```

Shape of the output, from a run against a live engine (this repo has not booted
one yet, so the strings are illustrative and the pass criteria are real):

```
  PASS arithmetic greedy     '437'
  PASS arithmetic thinking   '437'
  PASS tool calling          [{"function": {"name": "get_weather"

3/3 canaries passed
```

Three decode paths, three checks: greedy short, greedy with reasoning on (the
thinking path changes the token stream and is where a speculative drafter shows
defects first), and tool calling (which exercises the parser, not the model).
The arithmetic probe is 19x23 -> 437; `417` is a known FP8-KV defect in other
builds, so this specific number is a regression tripwire, not a maths test.

Then take a baseline:

```bash
./bin/qwen38 bench --save
./bin/qwen38 runs
```

## 3. Day to day

| Command | What it is for |
|---|---|
| `status` | container state, host PSI/available, ledger tail. Safe to run constantly |
| `logs --tail 200` | engine log **plus** a grep for the flags that actually took effect |
| `metrics` | the engine's own counters, each annotated with what it is evidence for |
| `observe --interval 10` | append host + engine + container samples to `state/observe.jsonl` |
| `guard` | foreground watchdog; stops our container if the host gets into trouble |
| `bench --save` | net-decode tok/s, recorded against the current run |
| `compare <a> <b>` | config diff first, number diff second |
| `stop` / `start` | the container only, never a neighbour's |

`logs` exists in that list because of a specific trap: a flag that silently loses
to a default produces a healthy-looking server with the wrong behaviour. The
extra lines it prints are the engine confirming what it actually parsed:

```
  effect: speculative_algorithm='DFLASH'
  effect: context_len=262144
  effect: max_running_requests=8
  effect: max_mamba_cache_size=32
```

If those do not match what you configured, nothing else you measure is about the
thing you think you changed.

### Clients

```
OpenAI      http://127.0.0.1:28100/v1     model "qwen3.8-27b"
Anthropic   http://127.0.0.1:28100/v1/messages   (ANTHROPIC_BASE_URL without /v1)
```

Loopback by default. To serve a LAN client, set `Q38_BIND=0.0.0.0` in
`conf/config.local` and restart -- and then treat that port as a published
service, because it is unauthenticated.

## 4. Monitoring that is worth keeping

Start `observe` in a tmux pane during any experiment. It writes one JSON object
per line, under the same key names `metrics` prints:

```json
{"avail_gib": 91.78, "container": "not running", "engine": "unreachable",
 "psi_full": 0.0, "psi_some": 0.0, "swap_used_gib": 3.45, "t": "2026-09-26T15:17:08Z"}
```

That line is verbatim from this box with no engine running, which is the point:
the host half of the sample is always available, and a missing engine records
`"engine": "unreachable"` rather than zeros. With an engine up, the same line
carries the counters under the identical keys `metrics` prints:

```json
{"avail_gib": 41.2, "container_mem": "74.1GiB / 96GiB", "psi_some": 0.0,
 "sglang:token_usage": 0.31, "sglang:spec_accept_length": 2.89,
 "sglang:num_running_reqs": 3, "t": "..."}
```

The second block is illustrative of the keys and not yet a measurement: this repo
has not booted an engine on the reference box, so its numbers are placeholders.
A gap in the file is an honest record of a gap; a zero would be a fabricated
reading.

Three things to look at, in this order:

1. **`accept_length`** (drafted tokens accepted per verify step). The single most
   informative speculative-decoding number. It falls when the drafter and target
   disagree, which means it tracks the *workload*, not just the configuration.
2. **`kv_usage`** pinned near 1.0 means you are KV-bound: the pool, not the GPU,
   is the limit. The fix is more memory or fewer/shorter concurrent requests, not
   a faster drafter.
3. **`avail_gib`** trending down across hours while nothing else changed is a
   leak. This is the number to alert on, because on GB10 it is the only one that
   reflects GPU memory too.

`guard` is the active half of the same information: it samples host pressure, and
after two consecutive breaches it stops *our* container and records a `guard_trip`
event in the ledger. It never touches another container. Run it in the foreground
while you are learning; once the numbers stop moving, put it in a unit.

## 5. Upgrading

The stack has four independent moving parts. Upgrading means deciding, per part,
whether you want the new thing -- not running one command that changes all four.

### 5.1 The engine image

```bash
./bin/qwen38 pull          # compare the pin against what the tag now resolves to
```

```
DRIFT: pinned sha256:0000...
       upstream sha256:616a...

Upstream moved the tag. Read the changelog for the new build before
re-pinning, then re-run the acceptance benchmarks -- a newer image is
not automatically a faster one on this hardware.
```

`pull` with no `--pull` is a *check*, not a fetch. That ordering is the point:
a tag is a moving pointer, so the question "has upstream moved?" must be cheap to
ask. When it has:

1. Read what moved. On this model family the difference between two images has
   measured anywhere from 0% to +14% depending on the checkpoint export, so "newer
   is faster" is not a law.
2. Re-pin by editing `Q38_IMAGE_DIGEST` in `conf/config.defaults` (get the digest
   from `docker buildx imagetools inspect <alias>`).
3. `./bin/qwen38 pull --pull`, then `systemctl restart`, then `canary`, then
   `bench --save`.
4. `compare` the new run against the old one. If the config diff shows only the
   image, the number difference is about the image.

### 5.2 The code

```bash
git fetch && git log --oneline HEAD..origin/main
git pull
python3 -m unittest discover -s tests -t .    # must be green before you restart anything
./install.sh --print-unit                     # see whether the unit changed
./install.sh
sudo systemctl restart qwen38-spark
./bin/qwen38 canary
```

`install.sh` re-reads what is already installed and keeps your `conf/config.local`
choices, so an upgrade does not reset a tuned box to its defaults. It backs up a
unit it is about to replace instead of overwriting a hand edit.

### 5.3 The checkpoints

Pins are the difference between "reproducible" and "it worked in August".
`Q38_DRAFT_REVISION` exists because a draft repo moves; `./bin/qwen38 doctor`
will tell you if a revision cannot be resolved. Before you bump a revision, know
that a draft trained against a different target revision can silently lose
acceptance, which shows up as a slow server with no error anywhere.

### 5.4 Rollback

`git checkout <sha> && ./install.sh` puts the previous unit back. The ledger keeps
the old run records, so `compare <old-run> <new-run>` tells you whether the
rollback achieved anything. This is the reason the ledger stores the whole plan
per run rather than a diff against "current": the current changes.

## 6. When it goes wrong

| Symptom | First thing to check | Usually means |
|---|---|---|
| `start` says REFUSED, memory pressure | `qwen38 status` | A neighbour is legitimately eating the box. This is the guard working; wait or free memory |
| `service-start` refuses after the settle deadline | `journalctl -u qwen38-spark` | Something is still starting at boot. Raise `Q38_SETTLE_DEADLINE_S`, or add `After=` on the unit |
| Container exits before ready | `qwen38 logs --tail 200` | Read the traceback; the two common ones are a fraction too high for the KV pool, and a draft revision that no longer matches |
| Unit says `inactive`, engine answers anyway | did you run `start` manually instead of the service? | Two managers for one container. `qwen38 stop` then `systemctl start` |
| Slow, no errors | `qwen38 metrics` | Look at `accept_length` then `kv_usage`. See the curriculum, step 3 |
| Box froze and needed a power cycle | `journalctl -k -b -1` | Not this stack's guard being insufficient -- check whether something was started outside it with a hand-set fraction |

The ledger is the debugging tool. `qwen38 runs` shows each run's status; a run
stuck at `open` is a boot that never became ready, and its `plan` is the exact
configuration that failed. That is worth more than reproducing from memory.

## 7. Removing it

```bash
./uninstall.sh --list     # everything this repo put on the box, changes nothing
./uninstall.sh            # unit + container; caches and ledger stay
./uninstall.sh --purge    # also delete ~24 GB of weights
```

The run ledger under `state/evidence` is never deleted by either mode. It is the
measurement record, and it is the one artifact here you cannot regenerate.
