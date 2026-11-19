# Q6_K in the wide layout - design, not yet implemented

Measured state this is based on, from the sampling profiler after the four
optimisations (SiLU parallel, attention two-phase, batched matvec region merge,
decode matvec region merge):

    34.7%  idle (ntdll waits plus the pool spin)
     8.7%  f16_to_f32
     7.5%  qwen35_dot_q6_K_q8_K_avx2
     4.2%  qwen35_gemm_q4_K_q8_K_2x4      decode, single token
     3.7%  qwen35_x8_build                eager, once
     1.2%  qwen35_gemm_q4_K_16x8          the wide kernel, 12.3 GB of weights

Q6_K is 3.4 GB of the 15.66 GB model, 22 percent, and takes 16.2 percent of the
samples against the wide kernel's 1.2 percent for 79 percent of the weights. It
is the last large piece still on the old path.

## Why the f16 conversion cannot be fixed in place

`f16_to_f32` is 8.7 percent of samples, but replacing it with the hardware
instruction `_cvtsh_ss` changed prefill by nothing at all - measured, 17.82
against 17.84. The samples are parked on a cache miss, not on the conversion:
`d` is two bytes at the head of a 210-byte block, so each one is a separate line
to fetch. Making the fetch cheaper is impossible; making it a sequential f32
array is the fix, which is what the wide layout already does for Q4_K.

## Block format

    block_q6_K   210 bytes
        ggml_half d            2     super-block scale
        uint8_t   scales[16]  16     one signed 6-bit scale per 16 values
        uint8_t   qh[64]      64     the high two bits of every value
        uint8_t   ql[128]    128     the low four bits

    value = (ql[p] & 0x0F) | ((qh[p / 4] >> (2 * (p % 4)) & 3) << 4), 0..63
    signed value used by the reference is that minus 32

## Wide layout

Sixteen rows interleaved, four bytes per row so `dpbusd` lane k is row k, the
same rule as Q4_K:

    struct {
        float   d[16];        64
        int8_t  sc[16][16];  256     per row, per sixteenth of the super-block
        uint8_t val[16][256]; 4096   every value widened to its own byte
    }                        4416 per 16 rows per super-block

The `val` table is the expensive part: 6.7 percent more than the 4101 bytes the
sixteen source rows actually occupy, because six-bit values cannot stay packed
once they have to land one per lane.

## Kernel

`dpbusd` needs unsigned by signed, and Q6_K values 0..63 are already unsigned
while the q8 activation is signed - the pairing works directly, exactly as it
does for Q4_K. The only difference from the Q4_K kernel is how the byte is
produced:

    q4:  q = raw & 0x0F, or (raw >> 4) & 0x0F, waste half of each load
    q6:  q = val[chunk], already one value per byte, no shifting at all

So the Q6_K inner loop is simpler than the Q4_K one: a 64-byte aligned load per
chunk, no masking, one `dpbusd` per token per chunk. The scale is a plain int8
per row per sixteenth, applied in float at flush time as `base = d * sc * yd`.

The reference subtracts 32 from every value; that is a constant offset over 256
values per block per row, so it contributes `-32 * d * sc * sum(activation)`,
which folds into the existing min-term machinery rather than costing anything in
the inner loop.

## Expected

Q6_K is 22 percent of the weights and should end up where Q4_K is - about a fifth
of the weight share in samples, so 4 to 5 percent instead of 16.2. Roughly 10 to
12 percent off prefill, taking 20.8 to about 23.

## Traps already paid for

  `_mm512_load_si512` faults when the buffer is not sixty-four byte aligned;
  `xmalloc` gives sixteen, so either over-allocate and align or use `loadu`.

  a scale stored as bytes must be widened with `cvtepu8_epi32` before it is
  used as float - reading `(float*)bytes` gives a relative error of one.

  `ds4_parallel_for` hands each chunk a contiguous span, so a worker must
  resolve its chunk to (tensor, start, run) and pass the whole run down. A
  per-row dispatch took decode from 2.71 to 0.09 t/s.

  eager construction must walk `model->tensors`, not a hand-written list of
  layer fields; the list missed a 10240-row projection the forward pass does
  reach, and whatever was missed got built inside the first prefill.
