# Offloading the Qwen3.8 output head to the GPU

Not implemented. This is the one direction left that has a quantified payoff and
has not been ruled out by measurement.

## Why it is worth doing

Decode is the one path that did not improve. It runs at 2.77 t/s, which is
362 ms per token, and it is bound by bytes read: 15.66 GB per token at 43 GB/s
against a practical ceiling of 55 to 63, and every knob that could raise that -
thread count, spin, granularity, batch size, alignment, layout - has been
measured and is already at its optimum.

Of those 15.66 GB, the output head is 3.4 GB, 22 percent:

    151936 rows x 5120 = 778M parameters, Q6_K, 3.4 GB
    3.4 GB / 43 GB/s = 79 ms of the 362

On a GPU that matvec is nothing. An RTX 5070 Laptop has 8151 MiB free and the
head needs 3.4 GB, so it fits with room for the logits row (151936 floats,
600 KB).

    keep the head on the GPU, upload once
    CPU forward produces the hidden vector, 20 KB, one transfer
    GPU computes logits, returns 600 KB
    expected 362 ms -> 285 ms, so 2.77 -> 3.5 t/s, plus 27 percent

## Why the existing switch does not do it

DS4_HYBRID_GPU_HEAD exists and offloads exactly this for the DS4/V4 model, whose
vocab projection is 129280x4096 Q8_0, about 0.53 GB. It does nothing here:

  the shape is hard-coded for the other model
  there is no DS4_BACKEND_HYBRID and no --hybrid flag in this tree, so the
  "hybrid GPU-prefill engine" that is supposed to keep CUDA alive is not running
  under --cpu, and the offload has nowhere to go

Measured with DS4_HYBRID_GPU_HEAD=1 and =2 under --cpu: 2.76 and 2.76 against a
2.77 baseline. Nothing.

## What it needs

  1. CUDA initialisation under --cpu, or a backend mode that keeps the device
     open without putting the 15.66 GB trunk on it
  2. A Qwen35 Q6_K matvec kernel on the device - the head is Q6_K, not the Q8_0
     the DS4 path uploads
  3. A 3.4 GB upload at session start, once
  4. Per token: hidden vector up, 20 KB; logits down, 600 KB; one synchronise

The transfers are small enough not to matter. The work is (1) and (2).

## Risks

  the logits must be bit-comparable with the CPU path or the sampled token can
  differ; DS4_HYBRID_GPU_HEAD=1 was written to be bit-exact and its approach is
  the one to follow, =2 is an approximation and was rejected there for that
  reason

  3.4 GB of VRAM is held for the life of the process, which on a 8 GB laptop
  device leaves little for anything else the user runs

## The cheaper alternative, not tried

Requantise just the output head from Q6_K to Q4_K. That halves its traffic,
3.4 GB to 1.7 GB, for the same 79 ms -> 40 ms, so about 11 percent on decode,
with no GPU involved at all. It changes the head's numerics, so it needs the
same bit-comparison discipline, but it is a repack rather than a new backend and
is far less work than the offload.


## The part that makes it worth doing: the MTP draft rides along

Offloading the head alone is 27 percent. Offloading the head and the MTP draft
together is a different order of thing, because CPU verification is almost free.

Measured, in this session:

    CPU decode, one token                2.77 t/s    362 ms
    CPU decode with MTP speculation     1.25 t/s    800 ms

MTP loses on the CPU because the draft costs about as much as the forward it is
meant to save - a nextn layer plus a 3.4 GB walk of the head, roughly 360 ms
against 362. With the head already resident on the GPU the draft reuses it and
costs milliseconds instead.

The reason that compounds is this: verifying k tokens on the CPU costs the same
as verifying one, because the batched path reads the 12.3 GB trunk once for up
to eight tokens per pass. That path already exists - it is the same
qwen35_cpu_forward_batch that the output-head change made reachable.

So the arithmetic becomes:

    head on GPU, one token per pass       283 ms   ->  3.5 t/s
    head and draft on GPU, 2 accepted     283 ms   -> 10.6 t/s
    head and draft on GPU, 3 accepted     283 ms   -> 14.1 t/s

The upper rows depend on the acceptance rate, which is a property of the draft
model and not something this document can assume. But the shape is right: the
CPU does one pass and emits several tokens, and the piece that made speculation
unprofitable - the draft reading the head - is exactly the piece that moving the
head removes.

Nothing about this changes the numerics. It is not the requantisation trade
described below, which buys 11 percent for a quality risk.

## Scope

Around four hundred lines, most of it new CUDA:

    initialise CUDA under --cpu without putting the trunk on the device   100
    a Qwen35 Q6_K matvec kernel for the device                            150
    upload the 3.4 GB head and the nextn layer, once                       50
    per round: hidden up 20 KB, logits and drafts down, one sync           80
    CPU batch verification                                                 exists

The last line is already there. The rest is a feature, not a patch, and it wants
a session with budget to build and verify it rather than the tail of one that has
already spent its own.
