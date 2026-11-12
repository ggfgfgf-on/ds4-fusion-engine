#include "ds4_help.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    const char *off;
    const char *cyan;
    const char *title;
    const char *yellow;
    const char *grey;
    const char *red;
    const char *white;
    const char *bright;
} help_colors;

static help_colors help_make_colors(FILE *fp) {
    bool color = isatty(fileno(fp));
    help_colors c = {0};
    if (!color) return c;
    c.off = "\x1b[0m";
    c.cyan = "\x1b[38;5;81m";
    c.title = "\x1b[1;38;5;250m";
    c.yellow = "\x1b[38;5;179m";
    c.grey = "\x1b[38;5;240m";
    c.red = "\x1b[38;5;203m";
    c.white = "\x1b[38;5;252m";
    c.bright = "\x1b[1;38;5;231m";
    return c;
}

static void title(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->title ? c->title : "", s, c->off ? c->off : "");
}

static void title_red(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n", c->red ? c->red : "", s, c->off ? c->off : "");
}

static bool option_name_has_switch(const char *name) {
    bool word_start = true;
    while (*name) {
        if (word_start && (*name == '-' || *name == '/')) return true;
        word_start = (*name == ' ');
        name++;
    }
    return false;
}

static void print_colored_option_name(FILE *fp, const help_colors *c, const char *name) {
    bool has_switch = option_name_has_switch(name);
    bool word_start = true;
    while (*name) {
        const char *start = name;
        while (*name && *name != ' ') name++;
        bool is_option = !has_switch || *start == '-' || *start == '/' ||
                         (word_start && has_switch && *start != '[');
        const char *color = is_option ? c->cyan : c->bright;
        if (color) fputs(color, fp);
        fwrite(start, 1, (size_t)(name - start), fp);
        if (color && c->off) fputs(c->off, fp);
        if (*name == ' ') {
            fputc(*name++, fp);
            word_start = false;
        }
    }
}

static void opt(FILE *fp, const help_colors *c, const char *name, const char *desc) {
    if (c->cyan) {
        fputs("  ", fp);
        print_colored_option_name(fp, c, name);
        fprintf(fp, " %s|%s ", c->grey ? c->grey : "", c->grey ? c->off : "");
        fprintf(fp, "%s%s%s\n", c->white ? c->white : "", desc,
                c->white ? c->off : "");
        return;
    }

    const int col = 30;
    int n = (int)strlen(name);
    if (n > col) {
        fprintf(fp, "  %s\n      %s\n", name, desc);
    } else {
        fprintf(fp, "  %-30s %s\n", name, desc);
    }
}

static void para(FILE *fp, const help_colors *c, const char *s) {
    fprintf(fp, "%s%s%s\n",
            c->yellow ? c->yellow : "", s, c->yellow ? c->off : "");
}

static bool streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static bool topic_is(const char *topic, const char *name) {
    return topic && strcmp(topic, name) == 0;
}

static const char *tool_name(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4: return "ds4";
    case DS4_HELP_SERVER: return "ds4-server";
    case DS4_HELP_AGENT: return "ds4-agent";
    case DS4_HELP_BENCH: return "ds4-bench";
    case DS4_HELP_EVAL: return "ds4-eval";
    }
    return "ds4";
}

static const char *tool_usage(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4:
        return "Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]";
    case DS4_HELP_SERVER:
        return "Usage: ds4-server [options]";
    case DS4_HELP_AGENT:
        return "Usage: ds4-agent [options]";
    case DS4_HELP_BENCH:
        return "Usage: ds4-bench (--prompt-file FILE | --chat-prompt-file FILE) [options]";
    case DS4_HELP_EVAL:
        return "Usage: ds4-eval [options]";
    }
    return "Usage: ds4 [options]";
}

static const char *tool_summary(ds4_help_tool tool) {
    switch (tool) {
    case DS4_HELP_DS4:
        return "Chat with a local DwarfStar model, run one-shot prompts, inspect models, or coordinate distributed inference.";
    case DS4_HELP_SERVER:
        return "Serve one loaded DwarfStar model through OpenAI, Responses, Anthropic, and completion-compatible HTTP APIs.";
    case DS4_HELP_AGENT:
        return "Run the native terminal coding agent with live tools, session save/restore, and a responsive prompt while the model works.";
    case DS4_HELP_BENCH:
        return "Measure prefill, decode, context growth, and KV-cache size across repeatable context frontiers.";
    case DS4_HELP_EVAL:
        return "Run the built-in reasoning, math, science, and security evaluation harness with a live terminal UI.";
    }
    return "";
}

static void print_model_runtime(FILE *fp, const help_colors *c,
                                ds4_help_tool tool, bool full) {
    title(fp, c, "Model And Runtime");
    opt(fp, c, "-m, --model FILE", "GGUF model path. Default: ds4flash.gguf");
    if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_SERVER) {
        opt(fp, c, "--vision FILE", "Vision encoder GGUF for the selected model.");
    }
#ifdef DS4_ROCM_BUILD
    opt(fp, c, "--metal | --rocm | --cpu", "Select the backend explicitly.");
    opt(fp, c, "--backend NAME", "Backend name: metal, rocm, or cpu.");
#else
    opt(fp, c, "--metal | --cuda | --cpu", "Select the backend explicitly.");
    opt(fp, c, "--backend NAME", "Backend name: metal, cuda, or cpu.");
    if (tool == DS4_HELP_SERVER) {
        opt(fp, c, "--hybrid", "GPU (SSD-streamed, chunk 16) prefill + CPU decode, KV disk handoff. Warm decode 4.1-5.6 t/s measured.");
        opt(fp, c, "--compress-model-sidecar", "Offline tool mode: write \"<model>.dfc\" compressed sidecar and exit.");
    }
    opt(fp, c, "--gpu-vram N[,N,...]|auto", "CUDA VRAM budgets per device, in GiB, or auto-detect free VRAM.");
    opt(fp, c, "--gpu-devices N[,N,...]", "CUDA device indices used by multi-GPU placement.");
    if (tool != DS4_HELP_EVAL) {
        opt(fp, c, "--cuda-tensor-parallel", "Enable the paired DeepSeek tensor/expert path on an even multi-GPU CUDA placement.");
    }
#endif
    if (tool != DS4_HELP_BENCH) {
        opt(fp, c, "-c, --ctx N", "Allocated context tokens.");
    }
    if (tool == DS4_HELP_SERVER) {
        opt(fp, c, "-n, --tokens N", "Default max output tokens when clients omit a limit.");
    }
    opt(fp, c, "-t, --threads N", "CPU helper threads for host-side/reference work.");
    opt(fp, c, "--power N", "GPU duty-cycle target, 1..100. Default: 100");
    opt(fp, c, "--ssd-streaming", "Metal/CUDA/ROCm: opt in to SSD-backed model streaming instead of full residency.");
    opt(fp, c, "--ssd-streaming-cold", "SSD streaming: skip default popularity-based expert-cache preload.");
    opt(fp, c, "--ssd-streaming-cache-experts N|NGB|auto", "SSD streaming cache target. N requests dynamic expert slots; NGB also reserves two full prefill layers; auto runs the WDDM band scheduler on native CUDA (VRAM+spill-fixed, clamped to the thrash cliff) and the auto-plan on Metal/ROCm. Unset: auto-plan on Metal/ROCm; native discrete CUDA has no plan (measured: 80% of 8GiB VRAM < non-routed weights -> empty cache), set N/NGB/auto explicitly.");
    opt(fp, c, "--ssd-streaming-full-layers N", "GLM Metal streaming: keep the first N routed layers fully resident. Default: auto from NGB expert budget; use 0 to disable.");
    opt(fp, c, "--ssd-streaming-preload-experts N", "SSD streaming: upfront popularity preload count. DeepSeek auto-seeds by default; GLM demand-fills unless N is explicit.");
    opt(fp, c, "--simulate-used-memory NGB", "Diagnostic: lock N GiB before model load to simulate a smaller-memory machine.");
    opt(fp, c, "--prefill-chunk N", "Graph prefill chunk size. Default: CUDA TP 2048; PRO long prompts 8192; others 4096.");
    if (full) {
        if (tool == DS4_HELP_EVAL || tool == DS4_HELP_BENCH) {
            opt(fp, c, "--mtp-model FILE", "External MTP or DSpark support GGUF.");
        }
        if (tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_SERVER) {
            opt(fp, c, "--mtp", "Enable model-embedded MTP speculation.");
            opt(fp, c, "--mtp-model FILE", "External MTP or DSpark support GGUF.");
            opt(fp, c, "--mtp-draft N", "Maximum autoregressive MTP draft tokens. Default: 1");
            opt(fp, c, "--mtp-margin F", "Verifier confidence margin for fast MTP acceptance. Default: 3");
            opt(fp, c, "--mtp-timing", "Enable embedded MTP and print acceptance/timing counters.");
            opt(fp, c, "--dspark", "Enable DSpark using the support GGUF passed with --mtp-model.");
            opt(fp, c, "--dspark-confidence F", "Enable DSpark with confidence pruning threshold 0..1. Greedy/opportunistic default: Metal 0.6, CUDA/ROCm 0.7; exact sampling: 0.8");
            opt(fp, c, "--mtp-exact-sampling", "Preserve the ordinary temperature distribution instead of accepting target-matching greedy drafts directly.");
            opt(fp, c, "--dspark-strict", "Load DSpark support but keep target-only decode.");
        } else if (tool == DS4_HELP_BENCH) {
            opt(fp, c, "--dspark", "Benchmark greedy DSpark using the support GGUF passed with --mtp-model.");
            opt(fp, c, "--dspark-confidence F", "DSpark confidence pruning threshold 0..1.");
        }
        opt(fp, c, "--quality", "Prefer exact kernels where faster approximate paths exist.");
        opt(fp, c, "--warm-weights", "Touch mapped tensor pages at startup to reduce first-use stalls.");
        if (tool == DS4_HELP_DS4 || tool == DS4_HELP_BENCH) {
            opt(fp, c, "--expert-profile FILE", "Metal-only: write routed expert locality/cache simulation JSON.");
        }
    }
    fputc('\n', fp);
}

static void print_sampling(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "Prompt And Sampling");
    opt(fp, c, "-n, --tokens N", "Maximum generated tokens.");
    opt(fp, c, "--temp F", "Sampling temperature. 0 is greedy/deterministic.");
    opt(fp, c, "--top-p F", "Nucleus sampling probability.");
    opt(fp, c, "--min-p F", "Keep tokens scoring at least F times the top token.");
    opt(fp, c, "--seed N", "Sampling seed for reproducible non-greedy runs.");
    para(fp, c, "GLM CLI and agent runs default to temperature 1.0, top-p 0.95, and min-p 0 unless those options are set explicitly.");
    opt(fp, c, "--think", "Use normal thinking mode.");
    opt(fp, c, "--think-max", "Use Think Max when context is large enough.");
    opt(fp, c, "--nothink", "Disable thinking and ask for direct replies.");
    if (full) {
        opt(fp, c, "-sys, --system TEXT", "System prompt. Empty string disables the default where supported.");
        opt(fp, c, "-p, --prompt TEXT", "One-shot prompt text.");
        opt(fp, c, "--prompt-file FILE", "Read one-shot prompt text from FILE.");
        opt(fp, c, "--raw-prompt", "Tokenize the one-shot prompt without chat markers.");
    }
    fputc('\n', fp);
}

static void print_steering(FILE *fp, const help_colors *c) {
    title(fp, c, "Directional Steering");
    opt(fp, c, "--dir-steering-file FILE", "Load one f32 direction vector per layer.");
    opt(fp, c, "--dir-steering-ffn F", "Apply steering after FFN outputs. Default with file: 1");
    opt(fp, c, "--dir-steering-attn F", "Apply steering after attention outputs. Default: 0");
    fputc('\n', fp);
}

static void print_distributed(FILE *fp, const help_colors *c) {
    title(fp, c, "Distributed Inference");
    fputc('\n', fp);
    para(fp, c, "Distributed mode runs one logical session across several machines by assigning contiguous model layer ranges to workers. Workers own their layer slice and KV-cache shard; the coordinator owns the prompt, sampling loop, and client/API flow. Start workers first, then start the coordinator. The coordinator waits for a complete route and streams hidden states through the workers.");
    fputc('\n', fp);
    opt(fp, c, "--role ROLE", "Distributed role: coordinator or worker.");
    opt(fp, c, "--layers A:B", "Inclusive layer slice, e.g. 0:20 or 21:output.");
    opt(fp, c, "--listen HOST PORT", "Coordinator listen address; workers may use it for their data listener.");
    opt(fp, c, "--coordinator HOST PORT", "Coordinator address for --role worker.");
    opt(fp, c, "--dist-prefill-chunk N", "Coordinator prefill pipeline chunk size. Default: session cap.");
    opt(fp, c, "--dist-prefill-window N", "Max prefill chunks in flight. Default: workers+2, capped at 8.");
    opt(fp, c, "--dist-activation-bits N", "Hidden-state transport width: 32, 16, or 8. Default: 32");
    opt(fp, c, "--dist-replay-check", "Diagnostic: reset and replay prompt, then compare logits.");
    opt(fp, c, "--debug", "Print coordinator route/debug logs.");
    fputc('\n', fp);
    title(fp, c, "Tensor Parallelism");
    fputc('\n', fp);
    para(fp, c, "Tensor parallelism uses the same coordinator/worker addresses as distributed mode, but always runs one 50/50 worker. Add --tensor-parallel, omit --layers, start the worker, then start the coordinator.");
    fputc('\n', fp);
    opt(fp, c, "--tensor-parallel", "Switch --role/--listen/--coordinator to two-machine tensor parallelism.");
    opt(fp, c, "--transport auto|rdma|tcp", "Tensor gate transport. Default: auto");
    opt(fp, c, "--rdma-device NAME", "Select a verbs device when auto-detection is ambiguous.");
    opt(fp, c, "--rdma-gid-index N", "Select the local verbs GID index.");
    opt(fp, c, "--tensor-parallel-token-prefill", "GLM diagnostic: prefill one token at a time for exact arithmetic.");
    opt(fp, c, "--debug-hash N", "Cross-check hidden state every N tokens.");
    fputc('\n', fp);
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c);

static void print_cli_specific(FILE *fp, const help_colors *c, bool full) {
    title(fp, c, "CLI Modes");
    opt(fp, c, "ds4", "Start the interactive prompt.");
    opt(fp, c, "ds4 -p TEXT", "Run one prompt and exit.");
    opt(fp, c, "ds4 --prompt-file FILE", "Run a long prompt from a file and exit.");
    opt(fp, c, "--prefix-file FILE", "Preload complete alternating USER:/ASSISTANT: turns before the live conversation.");
    fputc('\n', fp);
    if (full) {
        print_cli_diagnostics(fp, c);
    }
}

static void print_cli_diagnostics(FILE *fp, const help_colors *c) {
    title(fp, c, "Diagnostics And Data Collection");
    opt(fp, c, "--inspect", "Load the model and print a summary only.");
    opt(fp, c, "--dump-tokens", "Print the exact CLI prompt token stream, then exit. Use --raw for literal text.");
    opt(fp, c, "--dump-logits FILE", "Write full next-token logits as JSON.");
    opt(fp, c, "--dump-logprobs FILE", "Write greedy continuation top-logprobs as JSON.");
    opt(fp, c, "--logprobs-top-k N", "Alternatives stored by --dump-logprobs. Default: 20");
    opt(fp, c, "--decode-consistency N", "Compare N-token decode logits with a fresh full prefill.");
    opt(fp, c, "--expert-profile FILE", "Metal-only: write routed expert locality/cache simulation JSON.");
    opt(fp, c, "--perplexity-file FILE", "Score raw text with teacher-forced NLL.");
    opt(fp, c, "--imatrix-dataset FILE", "Rendered prompt dataset for imatrix collection.");
    opt(fp, c, "--imatrix-out FILE", "Write llama-compatible routed-MoE imatrix .dat.");
    opt(fp, c, "--imatrix-max-prompts N", "Stop imatrix collection after N prompts.");
    opt(fp, c, "--imatrix-max-tokens N", "Stop imatrix collection after N prompt tokens.");
    opt(fp, c, "--imatrix-min-expert-samples N", "Continue until every routed expert has N samples.");
    opt(fp, c, "--head-test", "Run the output HC/logits head after the native slice.");
    opt(fp, c, "--first-token-test", "Run exact CPU whole-model pass for the first prompt token.");
    opt(fp, c, "--metal-graph-test", "Compare first GPU-resident graph stages with CPU.");
    opt(fp, c, "--metal-graph-full-test", "Run the GPU-resident self-token graph across all layers.");
    opt(fp, c, "--metal-graph-prompt-test", "Compare CPU and GPU graph logits for the full prompt.");
    fputc('\n', fp);
}

static void print_cli_commands(FILE *fp, const help_colors *c) {
    title_red(fp, c, "Interactive Commands");
    opt(fp, c, "/help", "Show interactive commands.");
    opt(fp, c, "/think, /think-max, /nothink", "Switch thinking mode.");
    opt(fp, c, "/ctx N", "Restart the interactive session with a new context size.");
    opt(fp, c, "/power N", "Set GPU duty cycle percentage, 1..100.");
    opt(fp, c, "/read FILE", "Submit a text file, PNG, or JPEG as the next user message.");
    opt(fp, c, "/quit, /exit", "Leave the prompt.");
    opt(fp, c, "Ctrl+C", "Stop current generation and return to ds4>.");
    fputc('\n', fp);
}

static void print_agent_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Agent Options");
    opt(fp, c, "-p, --prompt TEXT", "Submit an initial prompt after startup.");
    opt(fp, c, "--prompt-file FILE", "Read the initial prompt from FILE.");
    opt(fp, c, "--prefix-file FILE", "Preload complete alternating USER:/ASSISTANT: turns before the live task.");
    opt(fp, c, "--non-interactive", "Run without TUI. With an initial prompt: one turn; otherwise: repeated stdin prompts.");
    opt(fp, c, "--raw-prompt", "Non-interactive initial prompt only: omit agent chat/tool text.");
    opt(fp, c, "--edit-upto", "Enable anchored [upto] edits and automatic marker insertion.");
    opt(fp, c, "-sys, --system TEXT", "Extra system prompt. Empty disables extra text.");
    opt(fp, c, "--trace FILE", "Write prompt, token, and DSML debug trace.");
    opt(fp, c, "--chdir DIR", "Change working directory before loading runtime assets.");
    fputc('\n', fp);
}

static void print_agent_sessions(FILE *fp, const help_colors *c) {
    title(fp, c, "Agent Runtime Commands");
    opt(fp, c, "/save", "Save the current session in ~/.ds4/kvcache.");
    opt(fp, c, "/compact", "Compact the current session context now.");
    opt(fp, c, "/list", "List saved sessions, sorted by recent update time.");
    opt(fp, c, "/switch ID", "Load a saved session and show recent history.");
    opt(fp, c, "/del ID", "Delete a saved session.");
    opt(fp, c, "/strip ID", "Remove KV payload; the text history can be rebuilt later.");
    opt(fp, c, "/history [N]", "Show N recent user turns from the current session.");
    opt(fp, c, "/power N", "Set GPU duty cycle percentage, 1..100.");
    opt(fp, c, "/new", "Start a fresh session from the system prompt.");
    opt(fp, c, "/quit, /exit", "Exit.");
    fputc('\n', fp);
}

static void print_server_api(FILE *fp, const help_colors *c) {
    title(fp, c, "HTTP API");
    opt(fp, c, "--host HOST", "Bind address. Default: 127.0.0.1");
    opt(fp, c, "--port N", "Bind port. Default: 8000");
    opt(fp, c, "--cors", "Add Access-Control-Allow-* headers for browser JS clients.");
    opt(fp, c, "--trace FILE", "Write prompts, cache decisions, output, and tool calls.");
    opt(fp, c, "--batched-session N", "Keep N resident sessions and batch decode-ready requests.");
    opt(fp, c, "--mixed-prefill-quantum N", "Prefill chunk while generations are active. Default: 128; GLM-5.3 minimum: 1024");
    para(fp, c, "Endpoints: /v1/chat/completions, /v1/responses, /v1/completions, and /v1/messages.");
    para(fp, c, "Model endpoint aliases include deepseek-v4-flash and deepseek-v4-pro; both serve the loaded GGUF.");
    fputc('\n', fp);
}

static void print_server_thinking(FILE *fp, const help_colors *c) {
    title(fp, c, "Server Thinking Defaults");
    para(fp, c, "DeepSeek-compatible chat requests default to high-effort thinking.");
    para(fp, c, "reasoning_effort=max or output_config.effort=max requests Think Max.");
    para(fp, c, "Think Max requires --ctx >= 393216; smaller contexts use high.");
    para(fp, c, "thinking={type:disabled}, think=false, or model=deepseek-chat selects non-thinking mode.");
    para(fp, c, "In thinking mode, client sampling knobs are ignored like the official API.");
    fputc('\n', fp);
}

static void print_kv_cache(FILE *fp, const help_colors *c) {
    title(fp, c, "Disk KV Cache");
    opt(fp, c, "--kv-disk-dir DIR", "Enable disk KV checkpoints in DIR.");
    opt(fp, c, "--kv-disk-space-mb N", "Disk budget. Default when enabled: 4096");
    opt(fp, c, "--kv-cache-min-tokens N", "Do not save/load checkpoints shorter than N. Default: 512");
    opt(fp, c, "--kv-cache-cold-max-tokens N", "Save cold first prompts up to N tokens. 0 disables. Default: 30000");
    opt(fp, c, "--kv-cache-continued-interval-tokens N", "Save aligned continued frontiers. 0 disables. Default: 10000");
    opt(fp, c, "--kv-cache-boundary-trim-tokens N", "Trim tail tokens for cold boundary saves. Default: 32");
    opt(fp, c, "--kv-cache-boundary-align-tokens N", "Align cold boundary saves to this multiple. Default: 2048");
    opt(fp, c, "--kv-cache-reject-different-quant", "Reject checkpoints written with different routed-expert quantization.");
    opt(fp, c, "--disable-exact-dsml-tool-replay", "Disable exact sampled DSML tool replay map.");
    opt(fp, c, "--tool-memory-max-ids N", "Exact tool-call IDs kept in RAM. Default: 100000");
    fputc('\n', fp);
}

static void print_bench_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Benchmark Input");
    opt(fp, c, "--prompt-file FILE", "Raw benchmark text; token sequence is sliced at each frontier.");
    opt(fp, c, "--chat-prompt-file FILE", "Render FILE as one no-thinking chat user message.");
    opt(fp, c, "-sys, --system TEXT", "System prompt used only with --chat-prompt-file.");
    fputc('\n', fp);
    title(fp, c, "Benchmark Sweep");
    opt(fp, c, "--ctx-start N", "First measured frontier. Default: 2048");
    opt(fp, c, "--ctx-max N", "Last measured frontier. Default: 32768");
    opt(fp, c, "--ctx-alloc N", "Allocated context. Default: ctx-max + gen-tokens + 1");
    opt(fp, c, "--step-mul F", "Multiplicative step. Default: 1");
    opt(fp, c, "--step-incr N", "Linear step when --step-mul is 1. Default: 2048");
    opt(fp, c, "--gen-tokens N", "Greedy decode tokens per frontier. 0 for pure prefill. Default: 128");
    opt(fp, c, "--teacher-forced-decode", "Decode the following prompt tokens instead of each predicted argmax.");
    opt(fp, c, "--ppl", "With --teacher-forced-decode, also score the true next tokens and report per-frontier perplexity.");
    opt(fp, c, "--decode-replay N", "At the final frontier, decode the same window N times from one prefill snapshot (per-replay ppl + REPLAY_START markers).");
    opt(fp, c, "--csv FILE", "Write CSV there instead of stdout.");
    opt(fp, c, "--dump-frontier-logits-dir DIR", "Write one full-logit JSON file per frontier.");
    fputc('\n', fp);
}

static void print_eval_specific(FILE *fp, const help_colors *c) {
    title(fp, c, "Evaluation");
    opt(fp, c, "-n, --tokens N", "Max generated tokens per question. Default: 16000");
    opt(fp, c, "--questions N", "Run only the first N embedded questions.");
    opt(fp, c, "--case-sequence LIST", "Run 1-based case numbers in this comma-separated order.");
    opt(fp, c, "--trace FILE", "Write questions, outputs, and grading decisions.");
    opt(fp, c, "--regrade-trace FILE", "Regrade a prior trace without loading the model.");
    opt(fp, c, "--soft-limit-reply-budget N", "Soft close thinking near the end of reply budget. Default: 1024");
    opt(fp, c, "--hard-limit-reply-budget N", "Force </think> with N tokens left. Default: 512");
    opt(fp, c, "--soft-limit-think-close-rank N", "Soft-close when </think> is in top N tokens. Default: 3");
    opt(fp, c, "--pause-ms N", "Pause after each result in the TTY UI. Default: 350");
    opt(fp, c, "--plain", "Disable split-screen ANSI UI.");
    opt(fp, c, "--self-test-extractors", "Run answer-extractor self-tests and exit.");
    fputc('\n', fp);
}

static bool tool_has_topic(ds4_help_tool tool, const char *topic) {
    if (!topic) return true;
    if (streq(topic, "all")) return true;
    if (streq(topic, "runtime") || streq(topic, "distributed")) return true;
    if (streq(topic, "env")) return true;
    if (streq(topic, "sampling"))
        return tool == DS4_HELP_DS4 || tool == DS4_HELP_AGENT || tool == DS4_HELP_EVAL;
    if (streq(topic, "steering"))
        return tool == DS4_HELP_DS4 || tool == DS4_HELP_SERVER || tool == DS4_HELP_AGENT;
    switch (tool) {
    case DS4_HELP_DS4:
        return streq(topic, "diagnostics") || streq(topic, "commands");
    case DS4_HELP_SERVER:
        return streq(topic, "api") || streq(topic, "kv-cache") || streq(topic, "thinking");
    case DS4_HELP_AGENT:
        return streq(topic, "sessions") || streq(topic, "commands") || streq(topic, "tools");
    case DS4_HELP_BENCH:
        return streq(topic, "benchmark");
    case DS4_HELP_EVAL:
        return streq(topic, "evaluation");
    }
    return false;
}

static void print_env_knobs(FILE *fp, const help_colors *c) {
    title(fp, c, "Environment Knobs (DS4_*)");
    para(fp, c, "Fusion-engine features added in this port are env-gated and default OFF (certified FP32 baseline). Manual envs override the auto scheduler.");
    opt(fp, c, "DS4_CUDA_KV_BITQ", "Low-bit KV quantization master switch (raw ring pack). LOSSY: the dequantized KV differs from the FP32 ring, so the attention numeric family changes and greedy output can flip near-threshold tokens - re-certify (whole==split + new canonical reference) before enabling.");
    opt(fp, c, "DS4_CUDA_KV_BITQ_BITS", "Global code width 2-8. Default: 4.");
    opt(fp, c, "DS4_CUDA_KV_BITQ_RAW_BITS", "Raw ring width override (B profile: 6, A: 4).");
    opt(fp, c, "DS4_CUDA_KV_BITQ_COMP_BITS", "Attention-compressed cache width (profiles: 4).");
    opt(fp, c, "DS4_CUDA_KV_BITQ_IDX_BITS", "Indexer cache width (profiles: 4).");
    opt(fp, c, "DS4_CUDA_KV_BITQ_COMP", "Pack the attention-compressed cache (standalone or with raw).");
    opt(fp, c, "DS4_CUDA_KV_BITQ_IDX", "Pack the indexer cache (prefill side). GPU standalone decode is unsupported under it; the hybrid path (CPU decode) is unaffected.");
    opt(fp, c, "DS4_CUDA_KV_BITQ_AUTO", "Startup bit-width scheduler: quality (OFF->B->A, first that fits) or capacity (A).");
    opt(fp, c, "DS4_CUDA_INDEXER_2STAGE", "Two-stage indexer retrieval: Cauchy-Schwarz row pruning + incremental clustering (provably exact).");
    opt(fp, c, "DS4_CUDA_INDEXER_PAGING", "Indexer KV paging: 512-row pages, cold pages GPU->pinned->SSD, pull-back bit-equal. Requires DS4_CUDA_KV_BITQ_COMP (ignored without it); unsupported with CUDA TP attention.");
    opt(fp, c, "DS4_CUDA_INDEXER_PAGING_RESIDENT", "Resident page budget for paging (1..64, default 8). auto = full residency for the current ctx (never evicts; the span gate can never abort).");
    opt(fp, c, "DS4_CUDA_INDEXER_PAGING_LOG", "Paging verbose log.");
    opt(fp, c, "DS4_CUDA_INDEXER_FUSED_DECODE", "Fused indexer decode kernel (bit-exact; +8.1% decode at 32k).");
    opt(fp, c, "DS4_SLIMINFER_RATIO", "Attention Q-row pruning ratio 0..1 (APIA semantics, KV stays fresh).");
    opt(fp, c, "DS4_SLIMINFER_PRUNE_LAYER", "Pruning decision layer. Default: 1.");
    opt(fp, c, "DS4_SLIMINFER_START_POS", "Pruning start position.");
    opt(fp, c, "DS4_CUDA_PREFILL_GRAPH", "Prefill graph capture/replay (-4.9% vs layer-major).");
    opt(fp, c, "DS4_PG_BUDGET_MB", "pg adaptive-split budget (MiB).");
    opt(fp, c, "DS4_PG_MIN_SUB", "pg split floor (whole==split certification uses 16).");
    opt(fp, c, "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB", "GPU weight-arena cap. 0=unlimited (8GB cards may OOM at boot; 2-4 as emergency, slower).");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_HEADROOM_GB", "WDDM shared-pool headroom kept free for DWM/OS. Default: 8. 0 disables the cap. The cap freezes arena growth when hit; with the prefill LRU scheduler (default on) long-prompt prefill recycles instead of starving. Measured without recycling: ~2.7x throttle (2.94 -> 8.07 t/s).");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU", "Prefill-safe weight-arena LRU (default on): at each SSD-streaming prefill chunk boundary, drain the device and recycle ranges from completed chunks inside the headroom budget. 0 restores the static-cap behavior.");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU_MAX_MB", "Largest range the prefill LRU may evict (default 0 = unlimited); model tail always protected. Small caps keep big expert tables resident at the cost of arena pressure.");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_LRU", "Weight-arena LRU eviction (default off): evict least-recently-used spans instead of stopping at the cap.");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_LRU_MAX_MB", "LRU evictable range size cap (default 1 MiB); model tail always protected.");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_TAIL_MB", "Model-tail guard for arena eviction (default 3072 MiB): output-family tensors hold direct device pointers until the final read and are never evicted.");
    opt(fp, c, "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB", "Weight arena chunk size. Default: 1792.");
    opt(fp, c, "DS4_CUDA_WEIGHT_CACHE", "Optional device weight cache (fd-cache path) enable switch.");
    opt(fp, c, "DS4_CUDA_WEIGHT_PRELOAD", "Optional full weight preload into the device cache.");
    opt(fp, c, "DS4_CUDA_DIRECT_MODEL", "Serve weights straight from mapped host memory (no device cache).");
    opt(fp, c, "DS4_CUDA_PINNED_LIMIT_GB", "Rolling pinned host-RAM budget for zero-copy ranges. Default: total/4 clamped [2,12].");
    opt(fp, c, "DS4_CUDA_PINNED_ZEROCOPY", "Route >=64MiB expert ranges through the pinned zero-copy registry.");
    opt(fp, c, "DS4_CUDA_SLOT_BUDGET_MB", "Streaming slot-table budget (MiB). FUSION certified config uses 2048.");
    opt(fp, c, "DS4_CUDA_STAGE_LRU", "Windows stage-read LRU: cold pages bypass the OS page cache.");
    opt(fp, c, "DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "Disable streaming cold-decode prefill path (set by every certified launch script).");
    opt(fp, c, "DS4_CUDA_Q8_F16_CACHE_MB", "Q8->F16 dequant cache budget (MiB; 0 disables).");
    opt(fp, c, "DS4_SSD_AUTO_CACHE_PCT", "SSD expert auto-cache percent 50-95 (Metal/ROCm auto-plan and native-CUDA WDDM band scheduler). Default: 80.");
    opt(fp, c, "DS4_SSD_AUTO_CACHE_SPILL_MB", "WDDM band scheduler (--ssd-streaming-cache-experts auto, native CUDA): shared-pool spill allowed past dedicated VRAM, in MiB. Default: 1536.");
    opt(fp, c, "DS4_SSD_AUTO_CACHE_FIXED_MB", "WDDM band scheduler: fixed consumers (KV, buffers, resident model) subtracted from VRAM+spill, in MiB. Default: 3584.");
    opt(fp, c, "DS4_SSD_AUTO_CACHE_CLIFF_MB", "WDDM band scheduler: thrash-cliff floor the total cache budget is clamped to, in MiB. Default: 5120.");
    opt(fp, c, "DS4_CUDA_NO_DEFER_SYNC", "Disable deferred upload sync (rollback switch).");
    opt(fp, c, "DS4_MTP_TIMING", "Print MTP speculative timing counters.");
    opt(fp, c, "DS4_MTP_SPEC_LOG", "Per-token MTP speculative log.");
    opt(fp, c, "DS4_HYBRID_PREFILL_CHUNK", "Hybrid-server prefill chunk. Certified config: 0.");
    opt(fp, c, "DS4_METAL_PREFILL_CHUNK", "Graph prefill chunk override.");
    opt(fp, c, "DS4_THREADS", "CPU helper thread count.");
    opt(fp, c, "DS4_POOL_SPIN_US", "CPU pool spin microseconds. Certified config: 500.");
    opt(fp, c, "DS4_POOL_PIN", "Pin CPU pool workers.");
    opt(fp, c, "DS4_POOL_MIN_ROWS", "Minimum rows to engage the CPU pool.");

    para(fp, c, "DSpark (speculative drafter):");
    opt(fp, c, "DS4_DSPARK_EXACT_BOUNDARY", "Replay the last committed row with the exact decode kernels on full-block commit so boundary logits/KV match ordinary decode (default ON; 0 = batched-logits commit).");
    opt(fp, c, "DS4_DSPARK_VERIFY_CAP", "Verification row cap; 0 = drafter block size (CUDA default), 5 on the gfx1151 fast path.");
    opt(fp, c, "DS4_DSPARK_SEED_BATCH", "Seed-batch spec layout (default off on CUDA; gfx1151 fast path enables it).");
    opt(fp, c, "DS4_DSPARK_SCHEDULER", "Adaptive propose scheduler (default ON; 0 disables).");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_WINDOW", "Scheduler acceptance window in tokens. Default: 4 (16 gfx1151).");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_SKIP", "Skip cycles after a draft miss. Default: 2 (4 gfx1151).");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_SLOW_SKIP", "Skip cycles after a slow proposal. Default: 4.");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", "Minimum remaining tokens before proposing is skipped.");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW", "Scheduler break-even window (tokens) for accept accounting.");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP", "Skip cycles after a cold low-confidence accept.");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI", "Cold low-confidence proposal time threshold (ms). Default: 500.");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", "Skip cycles when the drafter produces no proposal.");
    opt(fp, c, "DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", "Skip cycles after a short accept that produced no draft.");
    opt(fp, c, "DS4_DSPARK_VERIFY_SPLIT_HEAD", "Split verifier head kernels (default: fused head).");
    opt(fp, c, "DS4_DSPARK_VERIFY_HEAD_NO_TP", "Disable tensor parallelism for the verifier head.");
    opt(fp, c, "DS4_DSPARK_VERIFY_SELECTED_PROFILE", "Verify through the streaming selected-profile expert path (opt-in, default off).");
    opt(fp, c, "DS4_DSPARK_DISABLE_VERIFY_SELECTED_PROFILE", "Force-disable the selected-profile verifier.");
    opt(fp, c, "DS4_DSPARK_VERIFY_NONCAUSAL", "Diagnostic: noncausal-attention verifier probe (3 runs).");
    opt(fp, c, "DS4_DSPARK_VERIFY_CACHE", "Enable DSpark verifier cache bookkeeping.");
    opt(fp, c, "DS4_DSPARK_CACHE_RESERVE_GB", "Reserve extra GiB of the DSpark cache window.");
    opt(fp, c, "DS4_DSPARK_DISABLE_FINAL_OUTPUT_ALIAS", "Disable aliasing the drafter's final output head over the graph's batch buffers.");
    opt(fp, c, "DS4_DSPARK_EXPERT_OVERLAP", "Diagnostic: report distinct routed experts across verifier rows.");
    opt(fp, c, "DS4_DSPARK_EXEC_TIER", "Multi-GPU: logical tier that executes the drafter.");
    opt(fp, c, "DS4_DSPARK_NO_MARKOV", "Disable Markov-bias correction (propose straight from base logits; needs exact verify).");
    opt(fp, c, "DS4_DSPARK_NO_GPU_MARKOV", "Force the CPU fused Markov argmax (GPU propose reads logits back to host).");
    opt(fp, c, "DS4_DSPARK_DISABLE_FUSED_CPU_MARKOV_ARGMAX", "Disable the fused CPU Markov argmax.");
    opt(fp, c, "DS4_DSPARK_DISABLE_REUSE_CONFIDENCE0_MARKOV", "Disable reuse of the confidence-0 Markov state.");
    opt(fp, c, "DS4_DSPARK_FAKE_ARGMAX_PROPOSAL", "Test mode: propose the base argmax instead of the drafter chain.");
    opt(fp, c, "DS4_DSPARK_PROBE", "Diagnostic: full DSpark propose/seed probe logging.");
    opt(fp, c, "DS4_DSPARK_SPEC_LOG", "Diagnostic: per-cycle speculative accept/reject log.");
    opt(fp, c, "DS4_DSPARK_STATS", "Diagnostic: acceptance/verify timing counters.");
    opt(fp, c, "DS4_DSPARK_CYCLE_TRACE", "Diagnostic: per-cycle timing trace (needs DS4_DSPARK_STATS).");
    opt(fp, c, "DS4_DSPARK_STAGE_PROFILE", "Diagnostic: per-stage timing profile.");
    opt(fp, c, "DS4_DSPARK_PROP_PROFILE", "Diagnostic: proposal-stage timing profile.");
    opt(fp, c, "DS4_DSPARK_VERIFY_PROFILE", "Diagnostic: per-layer verifier timing profile.");
    opt(fp, c, "DS4_DSPARK_VERIFY_TOPS_CHECK", "Diagnostic: verify GPU top-1 against CPU argmax per row.");
    opt(fp, c, "DS4_ROCM_DSPARK_FAST", "ROCm: gfx1151 DSpark fast path.");
    opt(fp, c, "DS4_ROCM_DSPARK_VERIFY_FAST", "ROCm: fast verifier mode.");
    opt(fp, c, "DS4_ROCM_DSPARK_GREEDY_GPU_MARKOV", "ROCm: GPU greedy Markov argmax.");

    para(fp, c, "MTP speculation:");
    opt(fp, c, "DS4_MTP_STRICT", "GPU MTP: strict (exact) verification mode.");
    opt(fp, c, "DS4_MTP_VERIFY_SERIAL", "CPU MTP: force serial per-token verify (greedy already forces it).");
    opt(fp, c, "DS4_MTP_BATCH_VERIFY", "GPU/ROCm MTP: roll back to the generic batch verifier (diagnostic).");
    opt(fp, c, "DS4_MTP_MIN_MARGIN", "GPU MTP: minimum draft margin for fast acceptance.");
    opt(fp, c, "DS4_MTP_SPEC_DISABLE", "Disable MTP speculation entirely.");
    opt(fp, c, "DS4_MTP_NO_SNAP", "CPU MTP: skip the MTP-window snapshot (faster; partial-accept rollback then unavailable).");
    opt(fp, c, "DS4_MTP_FORCE_SNAPSHOT", "GPU MTP: force the frontier snapshot.");
    opt(fp, c, "DS4_MTP_CAPTURE_PREFIX1", "GPU MTP: prefix-1 capture instead of partial-accept replay.");
    opt(fp, c, "DS4_MTP_EXACT_REPLAY", "GPU MTP: diagnostic exact-replay on partial accept.");
    opt(fp, c, "DS4_MTP_FULL_LOGITS", "Diagnostic: full MTP logits readback.");
    opt(fp, c, "DS4_MTP_CONF_LOG", "Diagnostic: MTP confidence log.");
    opt(fp, c, "DS4_MTP_PROBE", "Diagnostic: MTP acceptance probe.");

    para(fp, c, "CPU decode parallelism and batching:");
    opt(fp, c, "DS4_WARM_NEXT", "Prefetch the next layer's weights during the current layer.");
    opt(fp, c, "DS4_ROUTED_TOKEN_PARALLEL", "Parallelize routed-MoE expert work across CPU tokens.");
    opt(fp, c, "DS4_NO_ROUTED_TOKEN_PARALLEL", "Disable routed-MoE token parallelism.");
    opt(fp, c, "DS4_PARALLEL_ATTN_ROWS", "Parallelize attention rows on CPU.");
    opt(fp, c, "DS4_NO_PARALLEL_ATTN_ROWS", "Disable parallel attention rows.");
    opt(fp, c, "DS4_NO_BATCHED_ATTN", "Disable batched attention (CPU batch verify).");
    opt(fp, c, "DS4_NO_BATCHED_ROPE", "Disable batched RoPE.");
    opt(fp, c, "DS4_BATCHED_ROPE_MAX", "Batched RoPE size cap.");
    opt(fp, c, "DS4_ATTN_TOKENTILE", "Attention token tile size.");
    opt(fp, c, "DS4_PARALLEL_FFN", "Parallelize FFN work on CPU.");
    opt(fp, c, "DS4_BATCHED_FFN", "Batch FFN across tokens (CPU).");
    opt(fp, c, "DS4_NO_SHARED_BATCH_FFN", "Disable the shared-FFN batch path.");
    opt(fp, c, "DS4_PREFILL_BATCH", "CPU prefill batch mode.");
    opt(fp, c, "DS4_LARGE_PAGES", "Map per-layer compressor tensors with 2MB large pages.");
    opt(fp, c, "DS4_CPU_DISABLE_UNROLLED_ARGMAX", "CPU: fall back from the unrolled argmax kernel.");
    opt(fp, c, "DS4_SERVER_CPU_BATCH_DECODE", "CPU batched decode for --batched-session N (default on; =0 forces serial per-session): K sessions' tokens advance in one lockstep forward. Measured on this machine: ~no speed change vs serial (CPU forward is compute-bound per row), byte gate stays True; the concurrency win comes from --batched-session N itself.");

    para(fp, c, "CUDA memory, cache, and streaming:");
    opt(fp, c, "DS4_CUDA_SLOT_RESIDENT_MB", "Streaming slot-table resident budget (MiB).");
    opt(fp, c, "DS4_CUDA_SLOT_PINNED", "Use pinned staging for the slot cache.");
    opt(fp, c, "DS4_CUDA_SLOT_NO_GUARD", "Disable the slot budget guard.");
    opt(fp, c, "DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB", "Optional weight-preload span cap (MiB).");
    opt(fp, c, "DS4_CUDA_MODEL_COPY_CHUNK_MB", "Chunked model-copy chunk size (MiB). Default: 64.");
    opt(fp, c, "DS4_CUDA_COPY_MODEL", "Copy the whole model image to the device at startup.");
    opt(fp, c, "DS4_CUDA_COPY_MODEL_CHUNKED", "Chunked whole-model copy with staged prefetch.");
    opt(fp, c, "DS4_CUDA_NO_MODEL_COPY", "Disable whole-model copy paths.");
    opt(fp, c, "DS4_CUDA_KEEP_MODEL_PAGES", "Keep model pages resident (skip page-drop hints).");
    opt(fp, c, "DS4_CUDA_MODEL_DECOMPRESS", "Read weights through the DFC sidecar (default OFF).");
    opt(fp, c, "DS4_DFC_VERIFY", "Offline: verify the DFC sidecar after compression.");
    opt(fp, c, "DS4_CUDA_MODEL_SEQ_SCAN", "Open the model file with sequential-scan hint.");
    opt(fp, c, "DS4_CUDA_DIRECT_MMAP_COPY", "Copy via direct mmap staging.");
    opt(fp, c, "DS4_CUDA_NO_DIRECT_IO", "Disable direct-I/O staging.");
    opt(fp, c, "DS4_CUDA_NO_FD_CACHE", "Disable the fd-based range cache.");
    opt(fp, c, "DS4_CUDA_STRICT_WEIGHT_CACHE", "Never fall back to host pointers on cache miss.");
    opt(fp, c, "DS4_CUDA_Q8_F32_PRELOAD", "Preload Q8 weights into the F32 cache.");
    opt(fp, c, "DS4_CUDA_Q8_F32_ALL", "F32-cache all Q8 weights.");
    opt(fp, c, "DS4_CUDA_Q8_F32_LARGE", "F32-cache large Q8 weights only.");
    opt(fp, c, "DS4_CUDA_Q8_F16_ALL", "F16-convert all Q8 weights.");
    opt(fp, c, "DS4_CUDA_MMQ", "Enable MMQ dense/prefill GEMM tiers.");
    opt(fp, c, "DS4_CUDA_Q8_NO_ALIGNED", "Disable aligned Q8 kernels.");
    opt(fp, c, "DS4_CUDA_NO_Q8_DP4A", "Disable DP4A Q8 kernels.");
    opt(fp, c, "DS4_CUDA_NO_Q8_MMA", "Disable Q8 MMA kernels.");
    opt(fp, c, "DS4_CUDA_Q8_MMA_MIN_TOK", "Minimum tokens for Q8 MMA dispatch.");
    opt(fp, c, "DS4_CUDA_Q8_PAIR_BATCH", "Q8 pair batch matmul enable.");
    opt(fp, c, "DS4_CUDA_NO_Q8_BATCH_TOK4", "Disable 4-token Q8 batch kernel.");
    opt(fp, c, "DS4_CUDA_NO_Q8_BATCH_TOK8", "Disable 8-token Q8 batch kernel.");
    opt(fp, c, "DS4_CUDA_NO_Q8_BATCH_WARP", "Disable warp Q8 batch kernel.");
    opt(fp, c, "DS4_CUDA_NO_Q8_BATCH_EXACT_TOK2", "Disable exact 2-token Q8 batch kernel.");
    opt(fp, c, "DS4_CUDA_NO_Q8_PAIR_BATCH_EXACT_TOK2", "Disable exact 2-token Q8 pair batch.");
    opt(fp, c, "DS4_CUDA_F16_CUBLAS_BATCH", "cuBLAS F16 batch matmul enable.");
    opt(fp, c, "DS4_CUDA_NO_F16_PAIR_MATMUL", "Disable paired F16 matmul.");
    opt(fp, c, "DS4_CUDA_ATTENTION_OUTPUT_PRELOAD", "Preload attention-output weights.");
    opt(fp, c, "DS4_CUDA_NO_CUBLAS_ATTENTION", "Disable cuBLAS attention.");
    opt(fp, c, "DS4_CUDA_WINDOW_ATTENTION", "Window attention enable.");
    opt(fp, c, "DS4_CUDA_NO_WINDOW_ATTENTION", "Disable window attention.");
    opt(fp, c, "DS4_CUDA_DECODE_GRAPHS", "Decode-island graph capture.");
    opt(fp, c, "DS4_CUDA_PREFILL_PIPELINE", "Prefill upload/compute pipeline enable.");
    opt(fp, c, "DS4_CUDA_PREFILL_PIPELINE_MB", "Prefill pipeline budget (MiB).");
    opt(fp, c, "DS4_CUDA_PREFETCH_PREDICT", "Predictive expert prefetch.");
    opt(fp, c, "DS4_CUDA_PREFETCH_PREDICT_EXPERTS", "Experts prefetched per prediction.");
    opt(fp, c, "DS4_CUDA_STAGE_THREADS", "Staging reader thread count.");
    opt(fp, c, "DS4_CUDA_STAGE_NORMAL", "Stage reads without direct I/O.");
    opt(fp, c, "DS4_CUDA_UPLOAD_SYNC", "Synchronous weight upload.");
    opt(fp, c, "DS4_CUDA_NO_DERIVED_WEIGHTS", "Disable derived (aligned artifact) weights.");
    opt(fp, c, "DS4_CUDA_NO_SETDEVICE_CACHE", "Disable cached cudaSetDevice.");

    para(fp, c, "Router / experts:");
    opt(fp, c, "DS4_ROUTER_DELTA", "Router-delta sidecar path (ReMoE Part 1).");
    opt(fp, c, "DS4_REMOE_REUSE", "ReMoE router-reuse enable.");
    opt(fp, c, "DS4_REMOE_REUSE_TAU", "ReMoE reuse temperature tau.");
    opt(fp, c, "DS4_EXPERT_HOTLIST", "Expert hotlist file.");
    opt(fp, c, "DS4_EXPERT_PROFILE", "Expert access profiling enable.");
    opt(fp, c, "DS4_CUDA_SERIAL_ROUTER", "Serialize router selection.");
    opt(fp, c, "DS4_CUDA_NO_PARALLEL_ROUTER_SELECT", "Disable parallel router select.");

    para(fp, c, "Server / hybrid:");
    opt(fp, c, "DS4_HYBRID_GPU_HEAD", "Hybrid: compute the output head on GPU.");
    opt(fp, c, "DS4_SERVER_DECODE_COALESCE_US", "Server decode coalescing window (us).");
    opt(fp, c, "DS4_SERVER_BATCH_LOG", "Diagnostic: server batch log.");
    opt(fp, c, "DS4_KVSAVE_DEBUG", "Diagnostic: KV save/load debug log.");
    opt(fp, c, "DS4_LOCK_FILE", "Instance lock file path.");

    para(fp, c, "Correctness verification (hash anchors):");
    opt(fp, c, "DS4_LM_LAYER_HASH", "Per-layer LM hash verification.");
    opt(fp, c, "DS4_LM_LAYER_HASH_LAST", "Hash only the last layer.");
    opt(fp, c, "DS4_LM_HASH_CACHE", "Hash the cache state.");
    opt(fp, c, "DS4_LM_HASH_FFN", "Hash the FFN output.");
    opt(fp, c, "DS4_LM_HASH_COMP", "Hash the compressor output.");
    opt(fp, c, "DS4_LM_HASH_LAYER", "Hash a specific layer.");
    opt(fp, c, "DS4_LM_HASH_POS", "Hash a specific position.");
    opt(fp, c, "DS4_LM_HASH_STAGE", "Hash a specific stage.");
    opt(fp, c, "DS4_LM_HASH_SHARED", "Hash the shared expert output.");
    opt(fp, c, "DS4_ORACLE_LOGITS", "Diagnostic: dump oracle logits.");
    para(fp, c, "Not listed: per-kernel dispatch/fusion switches (DS4_CUDA_MOE_*, DS4_METAL_*), *_PROFILE/_LOG/_DUMP/_TRACE diagnostics, and GLM-family knobs.");
    fputc('\n', fp);
}

static void more_line(FILE *fp, const help_colors *c, const char *label, const char *topic) {
    static const char *colors[] = {
        "\x1b[38;5;81m", "\x1b[38;5;114m", "\x1b[38;5;179m",
        "\x1b[38;5;141m", "\x1b[38;5;147m"
    };
    static size_t idx;
    const char *on = c->cyan ? colors[idx++ % (sizeof(colors) / sizeof(colors[0]))] : "";
    if (streq(label, "Interactive commands:") && c->red) on = c->red;
    const char *off = c->off ? c->off : "";
    fprintf(fp, "    %s%-26s%s --help %s\n", on, label, off, topic);
}

static void print_more_info(FILE *fp, const help_colors *c, ds4_help_tool tool) {
    title(fp, c, "More Info");
    more_line(fp, c, "Runtime full info:", "runtime");
    if (tool_has_topic(tool, "sampling"))
        more_line(fp, c, "Sampling full info:", "sampling");
    more_line(fp, c, "Distributed inference:", "distributed");
    if (tool_has_topic(tool, "steering"))
        more_line(fp, c, "Steering full info:", "steering");
    if (tool == DS4_HELP_DS4) {
        more_line(fp, c, "Interactive commands:", "commands");
        more_line(fp, c, "Diagnostics:", "diagnostics");
    } else if (tool == DS4_HELP_SERVER) {
        more_line(fp, c, "HTTP API:", "api");
        more_line(fp, c, "Disk KV cache:", "kv-cache");
        more_line(fp, c, "Thinking behavior:", "thinking");
        more_line(fp, c, "Env knobs (DS4_*):", "env");
    } else if (tool == DS4_HELP_AGENT) {
        more_line(fp, c, "Agent sessions:", "sessions");
        more_line(fp, c, "Agent commands:", "commands");
        more_line(fp, c, "Agent tool system:", "tools");
    } else if (tool == DS4_HELP_BENCH) {
        more_line(fp, c, "Benchmark sweep:", "benchmark");
    } else if (tool == DS4_HELP_EVAL) {
        more_line(fp, c, "Evaluation options:", "evaluation");
    }
    fputc('\n', fp);
}

static void print_examples(FILE *fp, const help_colors *c, ds4_help_tool tool, const char *topic) {
    title(fp, c, "Examples");
    if (topic_is(topic, "distributed")) {
        opt(fp, c, "worker", "./ds4 --role worker --layers 21:output --coordinator 192.168.0.181 9000 -m ds4flash.gguf");
        opt(fp, c, "coordinator", "./ds4 --role coordinator --layers 0:20 --listen 0.0.0.0 9000 -p \"Hello\" -m ds4flash.gguf");
    } else if (topic_is(topic, "runtime")) {
        if (tool == DS4_HELP_SERVER) {
            opt(fp, c, "Metal API", "./ds4-server -m ds4flash.gguf --metal --ctx 100000");
            opt(fp, c, "quiet API", "./ds4-server --power 60 --host 127.0.0.1 --port 8000");
        } else if (tool == DS4_HELP_AGENT) {
            opt(fp, c, "agent", "./ds4-agent -m ds4flash.gguf --ctx 100000");
            opt(fp, c, "quiet agent", "./ds4-agent --power 50");
        } else if (tool == DS4_HELP_BENCH) {
            opt(fp, c, "bench", "./ds4-bench --prompt-file long.txt --ctx-max 32768");
            opt(fp, c, "quiet bench", "./ds4-bench --prompt-file long.txt --power 70");
        } else if (tool == DS4_HELP_EVAL) {
            opt(fp, c, "eval", "./ds4-eval --questions 10 --ctx 100000");
            opt(fp, c, "CPU debug", "./ds4-eval --cpu --questions 1 --tokens 32");
        } else {
            opt(fp, c, "Metal", "./ds4 -m ds4flash.gguf --metal -c 100000");
            opt(fp, c, "quiet thermals", "./ds4 -p \"Summarize README\" --power 50");
        }
    } else if (topic_is(topic, "steering")) {
        opt(fp, c, "steer FFN", "./ds4 -p \"Write tersely\" --dir-steering-file dir.bin --dir-steering-ffn 0.8");
    } else if (tool == DS4_HELP_SERVER || topic_is(topic, "api") || topic_is(topic, "kv-cache")) {
        opt(fp, c, "local API", "./ds4-server --ctx 100000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192");
        opt(fp, c, "curl", "curl http://127.0.0.1:8000/v1/models");
    } else if (tool == DS4_HELP_AGENT || topic_is(topic, "sessions") || topic_is(topic, "tools")) {
        opt(fp, c, "interactive", "./ds4-agent");
        opt(fp, c, "one shot", "./ds4-agent --non-interactive -p \"Create /tmp/hello.c\"");
    } else if (tool == DS4_HELP_BENCH || topic_is(topic, "benchmark")) {
        opt(fp, c, "csv", "./ds4-bench --prompt-file long.txt --ctx-max 32768 --csv speed.csv");
        opt(fp, c, "prefill only", "./ds4-bench --prompt-file long.txt --gen-tokens 0");
    } else if (tool == DS4_HELP_EVAL || topic_is(topic, "evaluation")) {
        opt(fp, c, "first 10", "./ds4-eval --questions 10 --trace eval.trace");
        opt(fp, c, "plain", "./ds4-eval --plain --nothink --tokens 512");
    } else {
        opt(fp, c, "chat", "./ds4");
        opt(fp, c, "one shot", "./ds4 -p \"Explain mmap in C\"");
        opt(fp, c, "long prompt", "./ds4 --think-max --prompt-file prompt.txt --ctx 393216");
    }
    fputc('\n', fp);
}

static void print_topic(FILE *fp, const help_colors *c, ds4_help_tool tool, const char *topic) {
    if (streq(topic, "all")) {
        print_model_runtime(fp, c, tool, true);
        print_env_knobs(fp, c);
        if (tool_has_topic(tool, "sampling")) print_sampling(fp, c, true);
        if (tool_has_topic(tool, "steering")) print_steering(fp, c);
        print_distributed(fp, c);
        if (tool == DS4_HELP_DS4) {
            print_cli_specific(fp, c, true);
            print_cli_commands(fp, c);
        } else if (tool == DS4_HELP_SERVER) {
            print_server_api(fp, c);
            print_server_thinking(fp, c);
            print_kv_cache(fp, c);
        } else if (tool == DS4_HELP_AGENT) {
            print_agent_specific(fp, c);
            print_agent_sessions(fp, c);
        } else if (tool == DS4_HELP_BENCH) {
            print_bench_specific(fp, c);
        } else if (tool == DS4_HELP_EVAL) {
            print_eval_specific(fp, c);
        }
        return;
    }

    if (streq(topic, "runtime")) print_model_runtime(fp, c, tool, true);
    else if (streq(topic, "env")) print_env_knobs(fp, c);
    else if (streq(topic, "sampling")) print_sampling(fp, c, true);
    else if (streq(topic, "steering")) print_steering(fp, c);
    else if (streq(topic, "distributed")) print_distributed(fp, c);
    else if (tool == DS4_HELP_DS4 && streq(topic, "diagnostics")) print_cli_diagnostics(fp, c);
    else if (tool == DS4_HELP_DS4 && streq(topic, "commands")) print_cli_commands(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "api")) print_server_api(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "kv-cache")) print_kv_cache(fp, c);
    else if (tool == DS4_HELP_SERVER && streq(topic, "thinking")) print_server_thinking(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "sessions")) print_agent_sessions(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "commands")) print_agent_sessions(fp, c);
    else if (tool == DS4_HELP_AGENT && streq(topic, "tools")) {
        title(fp, c, "Agent Tool System");
        para(fp, c, "The agent can read, search, write, edit, run bash, and browse through Chrome-backed web tools.");
        para(fp, c, "DeepSeek-family models emit DSML tool calls; GLM models use native <tool_call> syntax. Both are rendered live in the terminal.");
        para(fp, c, "Edit uses exact old/new replacement. --edit-upto enables anchored replacements between a unique head and tail.");
        fputc('\n', fp);
    } else if (tool == DS4_HELP_BENCH && streq(topic, "benchmark")) print_bench_specific(fp, c);
    else if (tool == DS4_HELP_EVAL && streq(topic, "evaluation")) print_eval_specific(fp, c);
}

static void print_default(FILE *fp, const help_colors *c, ds4_help_tool tool) {
    print_model_runtime(fp, c, tool, false);

    if (tool == DS4_HELP_DS4) {
        print_cli_specific(fp, c, true);
        print_sampling(fp, c, false);
    } else if (tool == DS4_HELP_SERVER) {
        print_server_api(fp, c);
        print_kv_cache(fp, c);
        print_env_knobs(fp, c);
    } else if (tool == DS4_HELP_AGENT) {
        print_agent_specific(fp, c);
        print_agent_sessions(fp, c);
    } else if (tool == DS4_HELP_BENCH) {
        print_bench_specific(fp, c);
    } else if (tool == DS4_HELP_EVAL) {
        print_eval_specific(fp, c);
    }
}

void ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic) {
    help_colors c = help_make_colors(fp);
    if (topic && !tool_has_topic(tool, topic)) {
        fprintf(fp, "%s: unknown help topic '%s'\n\n", tool_name(tool), topic);
        topic = NULL;
    }

    fprintf(fp, "%s%s%s\n", c.bright ? c.bright : "", tool_name(tool), c.off ? c.off : "");
    fprintf(fp, "%s\n\n", tool_summary(tool));
    fprintf(fp, "%s\n\n", tool_usage(tool));

    if (topic) print_topic(fp, &c, tool, topic);
    else {
        print_default(fp, &c, tool);
        print_more_info(fp, &c, tool);
    }
    print_examples(fp, &c, tool, topic);
}
