# The wide Q6_K path: measured, and it is a dead end

The tree carried a note warning against hoisting the wide Q6_K dispatch outward,
on the grounds that the wide record is forty-seven percent more bytes and running
it gave 6.5 tokens a second against 36.

I thought that measurement was stale. It predates the loop order fix, in which
every token group re-walked the whole weight tensor, so the path looked like it
had been judged under a bug that no longer existed. Two changes followed, both
correct:

1. `qwen35_q6_worker` now takes its token groups inside the row loop, the order
   `qwen35_x8_worker` uses for Q4_K. Before, the dispatcher ran a separate
   parallel region per token group, so each group re-read the entire tensor.

2. The branch was hoisted out of the Q4_K condition it had been nested in. It had
   been unreachable there: the enclosing condition required `DS4_TENSOR_Q4_K` and
   the inner one required `DS4_TENSOR_Q6_K`, so the code had never executed and
   every measurement of it had been a measurement of nothing.

Both are in the tree's history and both were reverted, because neither helps.

## What the numbers say

    117 token prefill, 28 threads

    tokens   wide path   disabled
       118       11.37      45.73
       292       21.17      44.66
       640       29.74      41.41

Throughput rises with token count, which is the signature of a fixed cost. Taking
the differences:

    (292 - 118) tokens in (13.79 - 10.38) s  =  51 tokens a second
    (640 - 292) tokens in (21.52 - 13.79) s  =  45 tokens a second

    fixed cost  about 7.9 s

The marginal rate is the same as the path it replaces. So the wide kernel is not
faster, despite being 512-bit `vpdpbusd` against the 2x4 kernel's AVX2
`madd_epi16`. Forty-seven percent more bytes appears to eat the gain exactly. And
it costs eight seconds to build, inside the prefill timer, because the Q6_K
builder is lazy where the Q4_K one is eager.

## What this rules out

The Q6_K share of CPU time is real - thirty-nine percent on twenty-two percent of
the weight bytes - but neither the loop order nor the wide layout moves it. Two of
the three plausible fixes are now measured and dead.

## What is left

A kernel that keeps the packed six-bit layout and uses `vpdpbusd` anyway. The 2x4
kernel already unpacks six-bit values into int8 on the fly; it then feeds them to
`madd_epi16` as int16. Doing the same unpacking but stopping at int8, and feeding
`vpdpbusd` with 512-bit registers, would keep the packed record - no forty-seven
percent, no eight-second build - and get the wider instruction.

The obstacle is that Q6_K carries one int8 scale per sixteen values, while
`vpdpbusd` accumulates sixty-four values per instruction with no per-group scaling.
The scale would have to be applied every sixteen values, four times per dpbusd
group, in float. That is more float work than Q4_K does, which applies one scale
per thirty-two values.

Whether it wins is not obvious and would need building to find out.
