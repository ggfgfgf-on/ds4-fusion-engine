# ds4-fusion-engine

**A Windows CUDA/CPU port and fusion-enhanced fork of [antirez/ds4](https://github.com/antirez/ds4) (MIT),
purpose-built for running DeepSeek-V4-Flash locally on memory-constrained hardware.**

> 中文摘要：本仓库是 antirez/ds4 的 Windows 移植 + 融合增强版，面向显存受限机器的
> DeepSeek-V4-Flash 本地推理。核心增强：SSD 流式专家缓存（WDDM 带宽调度器 auto 档）、
> GPU 预填充 + CPU 解码混合服务器（热态 ~4.5 t/s）、位级确定化与字节门认证体系、
> 训练/量化/下载工具链。发布为源码（src/）与编译产物（bin/）分离。

---

## What this is

Upstream ds4 is a portable DwarfStar inference engine for DeepSeek-V4 series models
(Metal / CUDA / ROCm). This fork adds:

- **Native Windows build** (`win_compat` POSIX layer + `build_win.ps1`, MSVC ABI via clang,
  CUDA sm_120a via nvcc) — upstream does not ship a Windows port.
- **WDDM band scheduler** for SSD-streamed expert caches
  (`--ssd-streaming-cache-experts auto`): dedicated VRAM + safe shared-pool spill − fixed
  consumers, clamped to the measured thrash cliff (5.00 GiB on 8 GB cards — lands exactly
  on the hand-tuned optimum: 3.38 GiB prefill reserve + 1.62 GiB dynamic / 246 experts).
  Parameterized: `DS4_SSD_AUTO_CACHE_SPILL_MB / _FIXED_MB / _CLIFF_MB / _PCT`.
- **Prefill-safe weight-arena LRU** (`DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU`, default on)
  with a 3 GiB output-family tail guard.
- **Hybrid server** (`--hybrid`): GPU SSD-streamed prefill + CPU decode with KV disk
  handoff — the production path on 8 GB VRAM machines.
- **CPU batched decode** (`DS4_SERVER_CPU_BATCH_DECODE`, default on; `--batched-session`).
- **Speculative decode support**: MTP draft 2 (legacy), official DSpark support model
  (`--dspark` + `--mtp-model <DeepSeek-V4-Flash-DSpark-support.gguf>`), SplitKV family
  (audited, sealed — see [docs/FUSION_REPORT.md] in the source archive).
- **Byte-gate certification harness**: bit-exact 48-token temp-0 output gates
  (`rel_gate_0926.ps1`), kept green across every change.
- **Toolchain**: model download/resume repair scripts, `gguf-tools`
  (`deepseek4-quantize` with a reproducible Windows build), imatrix collection,
  quality scoring fixtures, router fine-tuning notes.

## Repository layout

```
ds4-fusion-engine/
├── src/
│   ├── engine/          # inference engine source (all build-essential files)
│   ├── gguf-tools/      # quantization / imatrix / quality-scoring tools
│   └── model-tools/     # GGUF download-hole repair tools (C# sources)
├── bin/
│   ├── cuda/            # Windows CUDA binaries (sm_120a) + .pdb
│   ├── cpu/             # Windows CPU binaries
│   └── tools/           # deepseek4-quantize.exe, fill_ranges.exe, ...
├── scripts/             # download/resume/verify scripts + DSpark model-prep assets
├── docs/                # README / MODEL_CARD / WINDOWS_PORT / QA / help_*.txt exports
├── rel_gate_0926.ps1    # release byte-gate (16-expert and auto modes)
└── RELEASE.md           # release notes
```

## Building

Prerequisites: VS2022 Build Tools, LLVM clang, CUDA 13.x.

```powershell
src\engine\build_win.ps1            # CPU smoke build (5 exes)
src\engine\build_win.ps1 -Cuda      # full CUDA build (ds4, ds4-server, ds4-bench,
                                    #   ds4-eval, ds4-agent)
src\gguf-tools\build_quantize_win.ps1   # quantize tool
```

## Models

The engine expects a DeepSeek-V4-Flash GGUF, e.g.:

- `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (~81 GB)
- optional MTP drafter: `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (~3.5 GB)
- optional DSpark drafter: `DeepSeek-V4-Flash-DSpark-support.gguf` (~5.6 GB, SHA-256
  `8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e`)

> DeepSeek-V4.1-Flash (552B params) is supported by *upstream* ds4; its GGUF quant is
> ~340 GB, which exceeds the warm page-cache capacity of 64 GB-RAM machines — this fork's
> Windows port of upstream b3f5018 compiles (see the upstream-port branch), but V4.1 is
> not the recommended model for this hardware class.

## Quick start

```powershell
# Production: hybrid (GPU prefill + CPU decode, warm decode ~4.5 t/s)
bin\cuda\ds4-server.exe --hybrid --cuda -m <model.gguf> --ctx 8192

# Pure GPU with the WDDM band scheduler
bin\cuda\ds4-server.exe --cuda --ssd-streaming -m <model.gguf> `
    --ssd-streaming-cache-experts auto --ctx 8192

# CLI one-shot
bin\cuda\ds4.exe --cuda --ssd-streaming -m <model.gguf> `
    --ssd-streaming-cache-experts auto -p "hello"

# Agent
bin\cuda\ds4-agent.exe --cuda --ssd-streaming -m <model.gguf> `
    --ssd-streaming-cache-experts auto
```

CUDA runtime DLLs must be on PATH (e.g. the CUDA toolkit's `bin\x86_64`).

## Measured performance (RTX 5070 Laptop 8 GB / 64 GB RAM / Win11)

| Mode | Decode |
|---|---|
| Hybrid (GPU prefill + CPU decode), warm | **4.45–4.62 t/s** |
| Pure GPU, plain (auto cache = 5 GiB) | 0.42–0.47 t/s |
| Pure GPU + MTP draft 2 + KV disk | 0.50–0.52 t/s |
| Pure GPU + DSpark | 0.44–0.50 t/s (no gain — a·B bound) |

Byte gate (48-token temp-0 vs canonical reference, 286 B): green in all shipped configs.

## Certification discipline

This fork's development practice: every change is certified by a **byte gate** —
48-token greedy output must match the canonical reference byte-for-byte
(`--ssd-streaming-cache-experts 16` and `auto` modes in `rel_gate_0926.ps1`).
Env-gated features default OFF (certified FP32 baseline); enable them explicitly
after their own gate run.

## Upstream & license

- Upstream: [antirez/ds4](https://github.com/antirez/ds4) (MIT). This fork is a
  Windows port + fusion enhancement; all engine credit goes to the ds4 author.
- Model weights: DeepSeek-V4-Flash (MIT) — download separately from Hugging Face.
- This repository: MIT (inherited); see `LICENSE`.

## Status / known limitations

- Pure-GPU decode is SSD-latency-bound on 8 GB cards (~0.5 t/s); hybrid is the
  production path. The speculative-decode family (MTP/DSpark/SplitKV) has been
  measured and sealed on this hardware (no realized gain — see the fusion report).
- V4.1 support: upstream code compiles on Windows CUDA (ported branch), but the
  340 GB GGUF is impractical for 64 GB-RAM machines; postponed.
