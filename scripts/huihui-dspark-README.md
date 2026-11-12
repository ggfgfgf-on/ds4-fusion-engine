# DSpark speculative decoding drafters for DeepSeek-V4-Flash-0731

Preliminary DSpark drafter modules extracted from the official
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
checkpoint, for use with llama.cpp speculative decoding.

These are **optional add-ons**. The quants in this repo are unchanged and work exactly as
before without them. Nothing in this folder is loaded unless you explicitly ask for it.

## Files

| File | Size | FP8 source weights | Markov / confidence heads | Routed experts |
|---|---|---|---|---|
| `dspark-DeepSeek-V4-Flash-0731-BF16.gguf` | 11.31 GB | BF16, bit exact | BF16 (source fidelity) | MXFP4 passthrough |
| `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` | 10.90 GB | Q8_0 | BF16 (source fidelity) | MXFP4 passthrough |

Both files contain 81 tensors and are `general.architecture = dflash`.

**Which one:** they measured identically in our testing (same acceptance rate, byte identical
output text). Take `BF16` if you want a provably exact reproduction of DeepSeek's weights,
or `Q8_0` to save 0.41 GB. There is no measured quality difference between them.

### About the names

The 0731 checkpoint stores its drafter as 25 FP8 (`E4M3`) projections, natively FP4 routed
experts, and BF16/F32 for everything small. The two files differ **only** in how those 25 FP8
tensors are stored:

- `BF16` upcasts them exactly. FP8 `E4M3` carries 4 significant bits and its `E8M0` scale is a
  pure power of two, so BF16 (8 significant bits, full F32 exponent range) reproduces every
  value with zero error. Verified `max|diff| = 0.0` against an independent dequantisation of
  the source safetensors. This file contains **no Q8_0 tensors at all**.
- `Q8_0` stores them as Q8_0, which is what upstream `convert_hf_to_gguf.py` does by default.

The routed experts are byte identical in both files. They are already MXFP4 in the source
checkpoint and are never requantised.

Note that `ggml-org/DeepSeek-V4-Flash-0731-GGUF` publishes a file also called `BF16` which is
equivalent to our `Q8_0` (its FP8 weights are Q8_0, and only 1.3% of it is actually BF16).
Our `BF16` is the fully lossless build, which is why it is larger.

## Usage

Speculative decoding is opt in. Pass `--spec-type draft-dspark` or nothing happens.

```bash
llama-server \
  -m  huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/DeepSeek-V4-Flash-Q2-0731.gguf \
  -md huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 5 \
  --fit off \
  -ngl 99 -ngld 99 -fa on -c 8192
```

Requirements and gotchas:

- `--fit off` is required when using a DSpark drafter.
- Do **not** pass `-devd` / `--spec-draft-device`. The drafter ships no token embeddings or
  output head by design and borrows the target's, so it must span the same devices as the
  target. Pinning it to one GPU fails with
  `pre-allocated tensor (output.weight) in a buffer (CUDA0) that cannot run the operation`.
- `--spec-draft-n-max` is clamped to the trained block size, which is 5 for this model.

### Multi GPU requires a rebuild for now

On a device split target, speculative decoding currently aborts on stock builds:

```
ggml/src/ggml-backend.cpp:1356: GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
```

The DeepSeek-V4 graph needs 39 graph inputs once the speculative layer taps are enabled, and
the compile time default is 30. Until this is fixed upstream, rebuild llama.cpp with a higher
cap:

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CXX_FLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=48" \
  -DCMAKE_CUDA_FLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=48"
```

Both flags are needed, otherwise translation units disagree on struct layout. There is no
runtime flag or environment variable for this. Single GPU setups are unaffected and work on
stock binaries. Raising the cap has no measured throughput or resident memory cost.

This is tracked upstream in the discussion on
[ggml-org/llama.cpp#25784](https://github.com/ggml-org/llama.cpp/pull/25784).

## Measured performance

On 4x B200 against `UD-Q4_K_XL`, greedy, short completions:

| | tokens/s | acceptance |
|---|---|---|
| target only | 62.9 | n/a |
| `--spec-type draft-dspark --spec-draft-n-max 5` | 75.6 to 81.9 | 0.49 to 0.59 |

That is roughly **1.2x to 1.3x**. The gain is workload and hardware dependent; community
reports on larger GPU counts and longer generations have seen up to 2x. Acceptance rate is
stable across drafter variants but sensitive to prompt and generation length.

### Output is not bit identical to non speculative decoding

Speculative decoding should be a pure speed optimisation, but on this model greedy output
diverges from a non speculative run of the same prompt. This is a known llama.cpp issue,
tracked at [ggml-org/llama.cpp#25618](https://github.com/ggml-org/llama.cpp/issues/25618),
and is not specific to these files.
