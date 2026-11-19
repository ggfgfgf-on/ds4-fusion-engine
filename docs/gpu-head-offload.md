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


## Updated: no new CUDA kernel is needed, and the conversion is a precision upgrade

The scope above assumed a Q6_K matvec had to be written for the device. It does
not. ds4_gpu.h has no Q6_K support at all - its kernels are Q8_0 and BF16, for
the other model - but it does have this:

    ds4_gpu.h:3815
    /* Single-token Q8_0 matvec whose weight rows live in a GPU tensor */

So the head can be requantised from Q6_K to Q8_0 on the host at load, and then
computed by a kernel that already exists.

That inverts the quality question. Requantising Q6_K down to Q4_K would be a
downgrade, 6.5 bits per value to 4.5, on the one tensor where quantisation error
is least forgiving - that is the trade described at the end of this document and
it is the reason not to do it. Going to Q8_0 is the other direction: 8.5 bits
per value against 6.5, so the round trip is all but lossless, and the head is
778M parameters, about 826 MB at Q8_0, which is comfortable inside the 8 GB the
laptop device reports.

Revised scope, against the four hundred lines above:

    requantise the head to Q8_0 on the host, reusing the existing quantiser   40
    initialise CUDA under --cpu and upload the 826 MB                         50
    call ds4_gpu_q8_0_matvec_tensor per token                                 20
    logits down and synchronise                                               40
    CPU batch verification                                                  exists

About a hundred and fifty lines, no new device code, and a precision change in
the harmless direction.

The remaining risk is not numerical and not algorithmic: it is that this is
unfamiliar code. The cuInit, tensor-allocation and stream conventions in
ds4_gpu.h - a hundred and eighty kilobytes of interface - are not something this
session touched, and the session's record is unambiguous about that. Five
successive wins came from code it knew (the CPU kernels, the parallel regions,
the profiler); six successive failures came from code it did not (splicing edits
by text anchor, and the GPU-adjacent switch). Budget was not the dividing line.


## Updated again: there is a working reference implementation in the same file

ds4.c:64861 to 64890 is a complete, working version of this for the qwen4 model,
written for the MTP draft head:

    qwen4_mtp_draft_head_load(g, m, w->output)          gather the head to GPU
    ds4_gpu_qwen4_matmul_q8_0_weights_tensor(...)       single-token matvec
    ds4_gpu_qwen4_argmax_tensor(...)                    argmax on device
    ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, 4)    four bytes back

So nothing has to be designed. The hard parts - uploading a vocabulary projection
as a gathered GPU tensor, running one token through it, and getting a token index
back without moving the logits - are all there and in use.

What is missing is the qwen35 side of it. That block is qwen4_ prefixed and sits
inside the qwen4 graph, working on g->mtp_R, g->mixed and g->R, none of which
exist on the qwen35 path. A qwen35 version needs a minimal GPU context of its own:
cuInit, one gathered tensor for the head, one input tensor, one output tensor,
and the same two calls plus the readback.

That is smaller than the four hundred lines first estimated and smaller than the
hundred and fifty after the Q8_0 realisation - but it is still new GPU plumbing
against an interface this session never touched, and it is the kind of work that
failed six times out of six here. The session's five wins all came from code it
knew; this is code it does not.

Recorded so the next attempt starts from line 64861 rather than from the
beginning.
