# Design

Why this is shaped differently from the three published recipes it is based on.

## The observation

MiaAI-Lab, hasso5703 and r0b0tlab all solve the same problem well on a dedicated
box, and each of them documents the same class of failure: *the machine hard-
rebooted and had to be power-cycled*, root cause reported as unclear or
attributed to a memory fraction. Each project's fix was to lower a constant.

That is the right emergency response and the wrong permanent one. A constant is a
bet about average behaviour, and the failure is not about the average -- it is
about the specific moment when something else on the box allocates.

The reference machine here is not dedicated. It runs 24 containers: three
databases, a message broker, a gateway, a service mesh, CARLA simulations. It has
no OOM protection of any kind (`systemd-oomd` and `earlyoom` are both inactive),
and `nvidia-smi` cannot report GPU memory on GB10 at all. So the coexistence
question is not a hypothetical to be documented as a warning; it is the primary
design constraint.

## The principle

**Anything that is a fact about the box is measured at launch. Anything that is a
fact about the model is pinned. Anything that is a judgement is recorded with the
evidence that formed it.**

Applied, that gives four substitutions:

| Constant | Measurement | Module |
|---|---|---|
| `--mem-fraction-static 0.95` | `(MemAvailable - reserved) / measured CUDA pool`, clamped | `memfit.py` |
| `CPUSET=5-9,15-19` | `scaling_max_freq` per core, take the top band | `cpuset.py` |
| port 30000 | probe each candidate, reject the ephemeral range | `ports.py` |
| "boot and hope it survives" | PSI + MemAvailable sampled, refuse before launching | `guard.py` |

Each is small. The reason they are not written inline in a shell script is that
the derivation is the interesting part, and a derivation you cannot test is a
derivation you will eventually get wrong. `memfit.py` shipped a 1024x unit error
for about four minutes; the test that catches it was written immediately after,
and it is annotated with the mistake because that is the information a future
reader needs.

## The boot race

The one genuinely new piece is `settle.py`, and it is worth explaining because it
is where "lower the constant" hides the most.

At boot, the neighbours are still starting. So `MemAvailable` is temporarily the
highest it will ever be. A fraction fitted at that instant is too large by exactly
the amount the other services subsequently consume. The engine loads, allocates
against a promise the box cannot keep, and fails at CUDA graph capture -- which is
minutes later, in a different subsystem, in a way that looks nothing like a memory
accounting error. That is the shape of "hard reboot, root cause unclear".

The fix is not a longer sleep, because sleep duration is a guess about how long
other people's services take to start. It is a convergence check: sample until
`MemAvailable` stops moving, then fit. If it never settles within a deadline,
**refuse to launch**. That last part matters. A guard that eventually launches
anyway is a guard that will be blamed and disabled; a guard that refuses and says
why is one you can leave running. `Restart=on-failure` means a refusal is a retry,
not an outage.

## Profiles as data

`profiles.py` exists because the three upstream recipes each spread one
configuration across a launcher, a wrapper script, a systemd unit and a README
table -- and then each needed a CI invariant whose only job was keeping those four
spellings of the same list in agreement. Encoding a profile as a value makes the
differences diffable and deletes the class of bug entirely.

The capability flags on a profile are the part that earns its keep:
`yarn_compatible=False` turns a six-minute boot ending in an unrelated traceback
into a two-line refusal at plan time. Those flags encode someone else's crash, and
the comment says so, with the upstream PR number that may eventually make it
obsolete. Knowledge with an expiry date attached is more useful than a rule.

## The ledger

Every launch writes a run record containing the *entire* plan, not a diff against
whatever was usual at the time. Immutable once finished.

The reason for storing the whole thing: "usual" changes, and a diff against a
moving baseline is how the upstream projects ended up with flag meanings written
as prose comments in READMEs. The reason for immutability: to compare two things
they both have to still exist.

This is the piece that turns deployment into learning. `compare` prints the config
diff before the number diff, deliberately. Read a delta before you know what
changed and you will invent a story for it; that is the single most common way a
tuning exercise produces a confident wrong answer.

## What is deliberately not here

* **No dashboard.** A JSONL file you can grep, join against the ledger and diff
  between boots is worth more at three in the morning than a graph, and it costs
  nothing to keep. Rendering is a separate concern with different failure modes.
* **No vendored engine patches.** Two upstream projects maintain or maintained a
  patched SGLang image. That is a real capability and a permanent maintenance
  liability; pinning an official digest and re-measuring on upgrade is the cheaper
  trade until you actually need the patch.
* **No automated sweep.** `tune` prints a protocol and refuses to run it. See step
  4 of the curriculum.
* **No multi-tenancy, no auth.** The engine binds loopback. If you publish it,
  that is your decision and your problem, and the config comment says so rather
  than pretending the exposure is safe.

## Known open edges

* The GDN slot factor (`mamba_slots`) is 4 everywhere, which is the verified value
  for `extra_buffer_lazy` and an unverified guess for plain `extra_buffer`. 4 is
  the conservative direction: too few silently clamps concurrency, which looks like
  a throughput bug rather than a config error. It is flagged in `profiles.py` as a
  sweep target, and the boot's granted value should be read from `logs`, not
  assumed from this file.
* `observe` records whatever counters the image exposes. Upstream renames these;
  an absent one prints `(absent)` rather than `0`, because a missing metric read as
  zero is a false conclusion.
* Nothing here has been benchmarked against the real model yet. Every number in
  the docs that came from anywhere is attributed to the project that measured it,
  and the ones that are ours are still to be made. That is the honest state of this
  repo, and the ledger is how it stops being true.
