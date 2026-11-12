# ds4 正式发布版 2026-09-26（源码 / 编译产物分离）

本目录是 2026-09-26 从开发工作区 `D:\DSHwork\src\ds4` 抽取的**当前最新正式发布版**。
相比上一正式版（`ds4-release`，2026-09-21），新增下载/训练类资产，并将**源码与编译产物分开存放**。

## 目录结构

```
ds4-release-0926/
├── src/
│   ├── engine/                # 推理引擎源码（全部构建必需 .c/.h/.cuh + CUDA 内核）
│   │                         #   + win_compat + third_party/zstd + build_win.ps1 + run_*.cmd
│   ├── gguf-tools/            # 训练/量化工具源码（deepseek4-quantize、imatrix、质量评分器）
│   │                         #   + build_quantize_win.ps1（Windows 构建脚本）
│   └── model-tools/           # 模型下载修复工具源码（fill_ranges / scan_allocated / scan_zero，C#）
├── bin/
│   ├── cuda/                  # CUDA 编译产物（sm_120a）：ds4 / ds4-server / ds4-bench /
│   │                         #   ds4-eval / ds4-agent / ds4-router-train
│   ├── cpu/                   # CPU 编译产物：ds4-cpu / ds4-server-cpu / ds4-bench-cpu /
│   │                         #   ds4-eval-cpu / ds4-agent-cpu
│   └── tools/                 # 训练/下载辅助产物：deepseek4-quantize.exe、
│                             #   fill_ranges.exe、scan_allocated.exe、scan_zero.exe
├── scripts/                   # 模型下载/修复/验证脚本 + DSpark 模型制备资产 + watchdog
├── docs/                      # README/MODEL_CARD/WINDOWS_PORT/QA 清单 + help_*.txt 导出
├── rel_gate_0926.ps1          # 本版发布门（字节门，16 专家 + auto 两种配置）
└── RELEASE.md                 # 本文件
```

## 源码（src/）

- **engine/**：与全部认证（字节门 / 哈希链 / ppl 阶梯 / 位级 A/B）所跑的是同一份代码，逐字节未动。
  剔除：实验/压测脚本、调试 dump、构建产物、备份快照、Metal/ROCm 后端源码（Apple/Linux 平台）
  与临时缓存。新特性（相对 9/21 版）：
  - `--ssd-streaming-cache-experts auto`：WDDM 带宽调度器（专用显存 + 安全共享池溢出 − 固定占用，
    × PCT，clamp 到 [悬崖, 天花板]）。参数化 env：`DS4_SSD_AUTO_CACHE_SPILL_MB`（默认 1536）、
    `DS4_SSD_AUTO_CACHE_FIXED_MB`（默认 3584）、`DS4_SSD_AUTO_CACHE_CLIFF_MB`（默认 5120）、
    `DS4_SSD_AUTO_CACHE_PCT`（50-95，默认 80）。实测落在 5.00 GiB（= 3.38 预填预留 + 1.62 动态，
    246 专家），与历史手工 5GB 好点逐位一致。
  - 权重竞技场 prefill 安全 LRU（`DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU` 默认开、
    `DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU_MAX_MB`、`DS4_CUDA_WEIGHT_ARENA_TAIL_MB` 默认 3072）。
  - CPU 批量解码（`DS4_SERVER_CPU_BATCH_DECODE` 默认开，hybrid 模式）、
    `--batched-session N`（`DS4_SERVER_DECODE_COALESCE_US` 默认 2000）。
- **gguf-tools/**：量化（deepseek4-quantize，Windows 构建见 `build_quantize_win.ps1`）、
  imatrix 采集、质量评分器（quality-testing/score_official.c + 100 例 fixture）、
  GLM 系列 Python 工具。上游 Makefile 目标 `quality-score`/`quality-llama-score` 保留。
- **model-tools/**：GGUF 下载中断修复工具（C# 源码，对应 bin\tools\ 的 exe）。

## 编译产物（bin/）

全部从本目录 `src\engine` **干净树重建**（2026-09-26）：

| 类别 | 产物 | 构建命令 |
|---|---|---|
| CUDA（sm_120a） | ds4.exe, ds4-server.exe, ds4-bench.exe, ds4-eval.exe, ds4-agent.exe, ds4-router-train.exe | `src\engine\build_win.ps1 -Cuda` |
| CPU | ds4-cpu.exe, ds4-server-cpu.exe, ds4-bench-cpu.exe, ds4-eval-cpu.exe, ds4-agent-cpu.exe | `src\engine\build_win.ps1` |
| 工具 | deepseek4-quantize.exe | `src\gguf-tools\build_quantize_win.ps1` |
| 工具 | fill_ranges.exe, scan_allocated.exe, scan_zero.exe | （C# 源码在 src\model-tools\，`csc` 编译） |

前置：VS2022 Build Tools、LLVM clang、CUDA 13.x（nvcc）。运行 CUDA 产物需 CUDA 运行时 DLL
在 PATH（如 `D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64`）。

## 推理运行

模型为外部文件（不在本目录内）：

```powershell
# 纯 GPU（推荐 auto 缓存）
bin\cuda\ds4-server.exe --cuda --ssd-streaming -m <model.gguf> --ssd-streaming-cache-experts auto --ctx 8192

# 混合（GPU 预填充 + CPU 解码，生产推荐，热态解码 ~4.5 t/s）
bin\cuda\ds4-server.exe --hybrid --cuda -m <model.gguf> --ctx 8192

# 纯 CPU
bin\cpu\ds4-server-cpu.exe -m <model.gguf> --ctx 8192

# CLI 单轮
bin\cuda\ds4.exe --cuda --ssd-streaming -m <model.gguf> --ssd-streaming-cache-experts auto -p "..."

# Agent
bin\cuda\ds4-agent.exe --cuda --ssd-streaming -m <model.gguf> --ssd-streaming-cache-experts auto
```

## 下载（scripts/）

- `watch_model.ps1`：带校验的 GGUF 下载/续传（HF）。
- `fill_missing_ranges.ps1` / `fill_loop.ps1` / `dl_resume_loop.ps1`：配合 bin\tools\ 的
  fill_ranges/scan_zero/scan_allocated，扫描并修复 GGUF 下载空洞（本机 0731 模型即此流程产出）。
- `verify_and_test.ps1`：SHA-256 校验 + 首轮推理冒烟。
- `watch_dspark.ps1` + `dspark-official-config.json` + `llama-*.cpp` 补丁：DSpark 投机草稿
  支持模型的下载/制备流程（对应 dspark-*-README.md）。

## 训练（src\gguf-tools\ + bin\tools\）

- 量化：`bin\tools\deepseek4-quantize.exe --hf <repo>`（源码可重建，见上文）。
- imatrix 采集：`bin\cuda\ds4.exe --imatrix-dataset <file> --imatrix-output <out>`。
- 路由微调：`bin\cuda\ds4-router-train.exe`（τ = 速度/质量旋钮，未训练模型净增益 +3.3%，
  质量墙见 FUSION_REPORT 附录）。
- 质量验收：`src\gguf-tools\quality-testing\`（score_official + 100 例 fixture + compare_scores.py，
  QA 门见 docs\QA_BEFORE_RELEASES.md §3）。

## 发布门（rel_gate_0926.ps1）

本版发布门 = 发布树 ds4-server.exe 双配置 48-token temp-0 字节门
（vs `src\ds4\.cache\cert_out_whole.txt`，286B）：

```powershell
.\rel_gate_0926.ps1 -Mode 16     # --ssd-streaming-cache-experts 16（与 9/21 版同门）
.\rel_gate_0926.ps1 -Mode auto   # --ssd-streaming-cache-experts auto（WDDM 调度器）
```

## 相对 9/21 版的变更

1. WDDM 带宽调度器（`auto`）实装并默认落在 5GB 好点；三次字节门全过（plain/MTP+DSpark 门已验证）。
2. 收录下载（scripts/ + bin\tools\）与训练（gguf-tools + quantize 构建脚本 + router-train）资产。
3. 源码与编译产物分离（src\ 与 bin\）。
4. 5GB 复测与 1.44 t/s 溯源、SplitKV 家族核查结论（见 FUSION_REPORT §7，不随本目录）。

## 0926 同日更新（resident auto + help 文档修复，2026-09-26 晚）

5. **`DS4_CUDA_INDEXER_PAGING_RESIDENT=auto`**：启动时按 ctx 算全常驻
   `resident = ceil(comp_cap/512)`——span 闸构造性不触发、永不换出；8k→5 页
   （与旧行为一致）、16k→9、32k→17、128k→65。价值 = 防闸死（省幅仍为 0，
   见 FUSION_REPORT §㉕/㉗）。
6. **help 文档修复（help_env 导出已同步）**：
   - `DS4_CUDA_KV_BITQ` 补 LOSSY 警告（输出族改变，需 whole==split 重认证）；
   - `DS4_CUDA_KV_BITQ_IDX` 补 GPU 独立 decode 不支持（hybrid 不受影响）；
   - `DS4_CUDA_INDEXER_PAGING` 补 requires BITQ_COMP（否则 ignored）+ CUDA TP 不支持；
   - `DS4_CUDA_INDEXER_PAGING_RESIDENT` 默认值 16→8（代码一致）并补 auto 语义。
7. 四族 env 开启裁决（hybrid 全绿：bitq B + indexer 2-stage/fused + 预读；纯 GPU 只开后三族）见 FUSION_REPORT §7.8。
