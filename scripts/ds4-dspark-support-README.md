---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-DSpark
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - deepseek
  - deepseek-v4
  - speculative-decoding
  - dspark
  - draft-model
  - ds4
library_name: gguf
---

# DeepSeek-V4-Flash DSpark support GGUF (for the ds4 engine)

A standalone **DSpark speculative-decoding draft ("support") model**, packaged as a
single 5.6 GiB GGUF for the [ds4 engine](https://github.com/antirez/ds4). It is the
drafter only — it does **not** generate on its own. Pair it with a DeepSeek-V4-Flash
main-model GGUF and ds4 proposes-and-verifies, committing only accepted prefixes.

This file is what `./download_model.sh dspark-support` produces, but pre-built: you can
skip the ~167 GB DSpark source download and the local conversion.

## What it is

DSpark is DeepSeek's official auxiliary draft module for DeepSeek-V4-Flash: three MTP
(multi-token-prediction) stages that read the main model's hidden states at layers 40–42
and propose a block of future tokens, with a Markov head for the block continuation. ds4
verifies every proposal against the main model, so **accepted tokens are exactly what the
main model would have produced** — speculative decoding is a speed optimization, not a
quality change.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `DeepSeek-V4-Flash-DSpark-support.gguf` | 5,989,114,272 B (5.6 GiB) | `8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e` |

- GGUF v3, 81 tensors, `general.architecture = deepseek4-dspark`.
- DSpark metadata: `block_size = 5`, `markov_rank = 256`, `n_layers = 3`,
  `target_layer_ids = [40, 41, 42]`, `noise_token_id = 128799`.
- Quantization (mixed, from the ds4 `deepseek4-quantize` DSpark builder): routed experts
  `ffn_down` Q2_K / `ffn_gate`,`ffn_up` IQ2_XXS; attention, shared experts and `main_proj`
  Q8_0; norms F32; hyper-connection tensors F16.

## Usage

Requires a DeepSeek-V4-Flash main-model GGUF (e.g. `ds4flash.gguf`) and a recent ds4 build.

```bash
./ds4 -m ds4flash.gguf \
  --mtp DeepSeek-V4-Flash-DSpark-support.gguf \
  --dspark --temp 0
```

Notes:
- `--dspark` activates the runtime; `--dspark-confidence 0.9` is the default acceptance
  threshold (`--dspark-confidence 0` forces fixed five-token blocks for diagnostics).
- DSpark requires **greedy decoding** (`--temp 0`); sampled and reasoning/think modes
  disable it.
- It helps most on predictable continuations (e.g. code); low-yield prompts may see no
  gain. Upstream reports draft acceptance ≈ 81–91% on structured/tool output and ≈ 60–68%
  on general prose — those are DeepSeek/ds4 figures, not measured in this repository.

## Provenance & method

1. **Source**: [`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
   (MIT) — the DeepSeek-V4-Flash checkpoint plus the DSpark drafter module. Only the
   drafter tensors (the `mtp.*` stages + Markov head, checkpoint shards 46–48) were used.
2. **Conversion**: ds4's `gguf-tools/deepseek4-quantize` DSpark support builder, from
   [antirez/ds4](https://github.com/antirez/ds4) commit `fc9efd1` ("Add DSpark speculative
   decoding"). No engine code was modified to produce this file.
3. **Verification**: GGUF metadata read back and checked against the source config
   (`block_size`, `markov_rank`, `target_layer_ids`, `noise_token_id`).

This is a straightforward format conversion of MIT-licensed weights. It has been built and
metadata-verified; end-to-end acceptance/throughput on your specific hardware should be
confirmed with ds4's `make dspark-acceptance` fixture.

## License & attribution

MIT, inherited from the base model. This is a derivative of DeepSeek's checkpoint and adds
no additional restrictions. Not affiliated with or endorsed by DeepSeek or the ds4 author;
all credit for the DSpark method and weights goes to DeepSeek, and to antirez for the ds4
engine and its DSpark GGUF builder.
