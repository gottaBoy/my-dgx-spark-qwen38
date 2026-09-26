# Learning to tune an inference engine on GB10

This is a course, not a tips list. It assumes you have the stack deployed and a
baseline recorded. Each step has one question to answer, one measurement that
answers it, and a criterion for when to stop. Do them in order: every later step
is interpreted against a baseline from an earlier one.

The reason a tuning attempt usually teaches nothing is that the person changing
knobs does not know which quantity each knob is supposed to move. So each step
below names the **observable** before it names the **knob**.

---

## Step 0. Learn what the hardware can do with no cleverness in it

```bash
Q38_PROFILE=ar ./bin/qwen38 start
./bin/qwen38 bench --save
```

**Question:** what is the raw decode ceiling of this box for this model?

This is the number every other number is a percentage of, and it is the one
measurement none of the published recipes bother to take. Without it you cannot
say whether a drafter is helping; you can only say it feels fast.

What you should learn from it: a 27B NVFP4 model at ~120 GB/s of effective
bandwidth per memory channel set decodes at a rate bounded by *reading weights*,
not by arithmetic. On unified memory the weights share the bus with everything
else on the box, which is why your number will be lower than a dedicated box's
and why a neighbour starting a backup degrades your tok/s.

**Stop when** you have `ar` recorded with two separate boots and the spread
between them. That spread is your error bar for everything after this point.

---

## Step 1. Understand what speculative decoding actually buys

```bash
./bin/qwen38 start                     # dflash2, the default
./bin/qwen38 bench --save
./bin/qwen38 metrics                   # look at accept_length
```

**Question:** where does the speedup come from, and why is it not a constant?

A decode step is memory-bound: the GPU reads the whole model to produce one
token. Drafting K tokens and verifying them in one step costs roughly the same
memory traffic as producing one, so each *accepted* token after the first is
nearly free. Speedup is therefore proportional to mean accepted length, not to K.

That single fact explains everything counterintuitive about this technique:

* Code and structured text are predictable, so acceptance is high and the drafter
  wins big. Free prose is unpredictable, so acceptance collapses and a deep draft
  can be a *net loss* -- you pay the draft and verify cost for tokens that get
  rejected.
* Bigger K is not better. Beyond the point where the extra tokens are usually
  rejected, you are paying to verify garbage.
* The right question is never "how fast is it" but "what is the user generating".

**Measure:** `accept_length` from `metrics` during each probe type. Record the
number for code and for essay separately; a single average hides the whole effect.

---

## Step 2. Learn the difference between a benchmark and a measurement

```bash
./bin/qwen38 bench --repeats 5 --save
./bin/qwen38 compare <run-a> <run-b>
```

**Question:** how would I know if this number were wrong?

Three things this harness does that a hand-rolled curl loop does not, each bought
with somebody's wasted afternoon:

1. **Two-call delta.** One timed generation charges prefill and template cost to
   "decode speed", so the number depends on prompt length. Same prompt at 60 and
   600 tokens, subtracted, cancels the fixed cost. That is what `bench` prints.
2. **Discard the first pass.** Upstream measured 89, 183, and *negative 371* tok/s
   on the first call after a boot. A negative throughput is a statement about your
   method, not about the model.
3. **Interleave, and stay inside one boot.** Two boots of the *same* image differed
   by 6.5% on the same probe. Any cross-session comparison is measuring the box as
   much as the configuration.

`compare` prints the config diff **before** the number diff on purpose. If you
read the delta first you will invent a story for it.

**Stop when** you can state, for your own baseline, the run-to-run spread on each
probe. Every future claim must clear that bar. On this hardware, deltas under
~15% within a boot are noise.

---

## Step 3. Learn where the bottleneck actually is

```bash
./bin/qwen38 metrics
./bin/qwen38 observe --interval 5     # in a second pane, under load
```

**Question:** what is the limiting factor right now, and how would I know if it
changed?

Read them in this order. Each answer tells you which knob is worth touching and
which is a waste of a boot:

| Counter | Reading | Conclusion | Knob |
|---|---|---|---|
| `accept_length` | high, near the drafted width | drafter is doing its job | none; you are done |
| `accept_length` | low on your real workload | draft and target disagree | shallower draft, different profile, or accept that prose is slow |
| `token_usage` | pinned near 1.0 | **KV-bound**: the pool is the limit | fewer concurrent requests, shorter context, or more memory fraction |
| `num_running_reqs` | below `max_running_requests` with a queue behind it | admission limited by GDN state pool | `MAX_CONCURRENT_REQUESTS` (which sizes `max_mamba_cache_size`) |
| `num_queue_reqs` | persistently nonzero under load | you are at the box's throughput ceiling | stop here; more concurrency will not help |
| `cache_hit_rate` | near zero with a long shared prompt | prefix caching is not engaging | check `logs` for the effective cache config |

The mistake to avoid is optimising a knob whose quantity is not the constraint.
Increasing `--mem-fraction-static` when you are accept-length-bound buys nothing
and risks the box; a faster drafter when you are KV-bound is a pure waste of an
afternoon.

---

## Step 4. Sweep one knob at a time, with a control

```bash
./bin/qwen38 tune concurrency
./bin/qwen38 tune dspark_block
./bin/qwen38 tune reserved_gib
```

**Question:** what does this specific knob cost and gain, on *my* workload?

`tune` prints the protocol and the commands, and refuses to run them. It should
refuse: an automated sweep that can reboot a shared box is a hazard, not a tool.
The protocol it prints is the part worth memorising:

1. One boot per value. Never compare across days.
2. Discard the first measurement after each boot. Always.
3. Interleave `A B B A` so thermal and power drift cancels instead of biasing.
4. **Re-run the control at the end.** If the control moved more than the effect
   you are claiming, the sweep measured the box, not the knob. This step is the
   one everyone skips and the one that separates a result from a story.
5. Stop when the median across repeats varies less than the within-value spread.
   Past that point you are fitting noise.

What each knob is really for, in one line:

* `profile` -- which drafter, and therefore which workload it suits. The largest
  single effect available, and the only one that can change the essay/code balance.
* `MAX_CONCURRENT_REQUESTS` -- throughput versus per-stream latency, and it sizes
  the GDN state pool. Not "more is better": over-provisioning the pool clamps
  concurrency silently.
* `CHUNKED_PREFILL` -- prefill and TTFT, barely decode. Tune it when time-to-first-
  token is the complaint, never when tok/s is.
* `RESERVED_GIB` -- the coexistence dial. On a shared box this trades your KV pool
  against the neighbours' safety, and it is the knob the other recipes do not have.
* `dspark_block` / `spec_steps` -- draft depth. Sharply peaked; the published peaks
  are for *their* workloads, which is why you measure rather than copy.

---

## Step 5. Long context: understand the price before paying it

```bash
Q38_PROFILE=mtp Q38_CONTEXT_LENGTH=1000000 ./bin/qwen38 plan   # look, do not launch
```

**Question:** what does 1M context cost, and what breaks?

Three things, and the third is the one that will bite you:

1. KV is roughly 32.8 KB/token on this hybrid with fp8_e4m3, so one full
   1M-token sequence is ~33 GB. On a dedicated box that is most of the machine.
2. YaRN scaling is required above 262144. `plan` derives the factor for you
   (524288 -> 2.0, 1000000 -> 4.0; the model card validates exactly those two).
3. **YaRN is incompatible with the draft-based profiles on this build.** The rope
   override leaks into the draft config and crashes the validator deep inside
   model loading, six minutes into a boot, with a traceback that does not mention
   context length. This harness refuses at plan time instead:

```
profile 'dflash2' cannot serve 1000000 tokens: the YaRN override leaks into the
draft config and crashes the rope validator. Use profile 'mtp' for >262144, or
lower the context.
```

Encoding a known-incompatible combination as a capability flag on the profile is
the difference between a two-line error and an afternoon. Note the honest shape of
that knowledge: it came from someone else's crash, it is recorded as a constraint,
and the comment in `lib/qwen38/profiles.py` says it is *their* finding on *their*
build -- so the first thing to check when a future image changes it is upstream
#34763 and the release notes, not the constant.

---

## Step 6. Quality: the measurement people forget

```bash
./bin/qwen38 canary
```

**Question:** is it still saying the right things?

Speculative decoding is lossless *by construction* -- the target verifies every
drafted token -- which is exactly why a defect in that path is so damaging: it
looks like the model getting worse, not like a bug. Quantisation is not lossless
by construction, and the FP4 `lm_head` variant of this checkpoint has a documented
history of subtle output differences.

So every speed claim in your notebook must have a correctness result beside it.
A drafter that accepts more because it skips verification would be extremely fast.

The discipline, in order: `canary` after every config change; a fixed prompt set
you read the output of occasionally (metrics do not catch a model that became
subtly wrong); and never accept a speedup that arrived with a canary failure you
did not investigate.

---

## The mental model, when you are done

Four quantities explain almost everything you will see on this hardware:

1. **Memory bandwidth** sets the decode ceiling. Unified memory means your
   neighbours share it, so your tok/s has a social component.
2. **Accept length** sets how much of that ceiling a drafter recovers. It is a
   property of the *output distribution*, so it is workload-dependent and no
   published number transfers to your traffic.
3. **Pool sizing** (KV plus GDN state) sets how many requests and how long a
   context you can serve at once. It is a budget, and the engine's own accounting
   on this chip is not the same as the host's.
4. **Prefill cost** is what a user experiences as latency and what a decode
   benchmark hides. Two different numbers; measure the one you care about.

Every knob moves one of those four. If you cannot say which, you are not tuning
-- you are shopping.

## Habits worth keeping

* Record before you conclude. `bench --save` costs nothing; reconstructing a
  configuration from memory costs an afternoon and produces a wrong answer.
* Keep the failure records. A run stuck at `open` in the ledger is the paper trail
  of a crash, and it is more informative than the successful runs.
* Distrust a good number from an unrecorded configuration. That instinct is the
  entire skill.
* When two runs agree exactly, you have probably measured your own caching, not
  the model.
* Re-measure after any image upgrade. Upstream fixes the thing you worked around
  and changes the thing you did not know was load-bearing.
