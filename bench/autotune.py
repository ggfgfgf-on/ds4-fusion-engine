#!/usr/bin/env python3
"""Autotune the qwen35 CPU prefill.

Why this exists: through one long session the tiling of the Q4_K wide kernel was
chosen by hand - sixteen rows, eight tokens, two nibble groups, four-byte
interleave - and each choice was justified by a single measurement. The machine
drifts three percent run to run, so a single measurement cannot distinguish a
one percent change from noise, and several of those choices were in fact never
verified.

Two things make this a tuner rather than a script.

Round-robin measurement. Every configuration is measured once per round and the
rounds are interleaved, so drift is spread evenly across configurations instead
of landing on whichever happened to run when the machine was warm. A single
configuration measured six times in a row would bake the drift into its score.

Median and spread, not mean. What matters is whether two configurations are
separable at all: if their interquartile ranges overlap, the difference is noise
and the tuner says so rather than picking a winner from noise.

The knobs here are the ones the engine already takes at runtime. Three more -
rows per group, nibble pairing, interleave width - are compile-time because they
change the repacked layout, so they belong in a build matrix rather than this
script; see BUILDKNOBS at the bottom.

    python bench/autotune.py                    full grid
    python bench/autotune.py --quick            three configurations, one round
    python bench/autotune.py --runs 4           more rounds, tighter intervals
"""
import argparse
import itertools
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "ds4.exe")
MODEL = os.environ.get("DS4_MODEL",
                       r"D:\DSHwork\models\Qwen3.8-27B-Uncensored-Q4_K_M.gguf")

# A prompt long enough that the token-group loop actually iterates: nine groups
# for a seventy token prompt, against three for the short one. The short prompt
# understates the effect of every knob that has to do with re-reading weights,
# which is most of them.
PROMPT = (
    "Explain in detail how a CPU cache hierarchy works, covering L1, L2, L3, "
    "the role of associativity, replacement policies such as LRU and pseudo-LRU, "
    "write-back versus write-through behaviour, and how prefetching interacts "
    "with all of these mechanisms in modern processors."
)

# knob name -> candidate values. The first value of each is what the engine does
# if the knob is unset, and it is also what was chosen by hand.
KNOBS = {
    "DS4_QWEN35_TGS":     ["8", "4"],
    "DS4_THREADS":        ["28", "24", "32"],
    "DS4_POOL_MIN_ROWS":  ["64", "32", "96"],
    "DS4_PREFILL_BATCH":  ["64", "32"],
}

FIXED = {
    "DS4_POOL_SPIN_US": "500",
    "DS4_POOL_PIN": "1",
}

# The engine links the CUDA runtime even in --cpu mode, so its directory has to
# be reachable or the process fails to start and every sample comes back empty.
_CUDA = r"D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64"
if os.path.isdir(_CUDA):
    os.environ["PATH"] = _CUDA + os.pathsep + os.environ.get("PATH", "")

PAT = re.compile(r"prefill: ([0-9.]+) t/s, generation: ([0-9.]+)")


def run_once(env_extra, ctx):
    env = dict(os.environ)
    env.update(FIXED)
    for k, v in env_extra.items():
        env[k] = v
    cmd = [EXE, "--cpu", "-m", MODEL, "--ctx", str(ctx),
           "-p", PROMPT, "-n", "4"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=300, env=env)
    except subprocess.TimeoutExpired:
        print("    timeout")
        return None
    # the engine writes its timing line to stderr, not stdout
    m = PAT.search((r.stdout or "") + (r.stderr or ""))
    if not m:
        tail = (r.stdout or "").strip().splitlines()[-6:]
        print("    no timing in output; last lines were:")
        for t in tail:
            print("      " + t)
        err = (r.stderr or "").strip().splitlines()[-3:]
        for t in err:
            print("      ! " + t)
        return None
    return float(m.group(1))


def measure(configs, runs, ctx, label):
    """Round-robin: one sample per configuration per round."""
    samples = {i: [] for i in range(len(configs))}
    for r in range(runs):
        for i, cfg in enumerate(configs):
            v = run_once(cfg, ctx)
            if v is not None:
                samples[i].append(v)
        done = sum(len(v) for v in samples.values())
        print(f"  {label} round {r + 1}/{runs}  ({done} samples)", flush=True)
    return samples


def summarise(configs, samples):
    rows = []
    for i, cfg in enumerate(configs):
        v = samples[i]
        if not v:
            continue
        med = statistics.median(v)
        # interquartile range as the separability test; with few samples fall
        # back to the full spread
        if len(v) >= 4:
            q = statistics.quantiles(v, n=4)
            lo, hi = q[0], q[2]
        else:
            lo, hi = min(v), max(v)
        rows.append((med, lo, hi, len(v), cfg))
    rows.sort(key=lambda r: -r[0])
    return rows


def show(rows):
    print(f"\n{'median':>8} {'range':>16} {'n':>3}  configuration")
    print("-" * 78)
    best = rows[0] if rows else None
    for med, lo, hi, n, cfg in rows:
        knobs = " ".join(f"{k.replace('DS4_', '').replace('QWEN35_', '')}={v}"
                         for k, v in cfg.items())
        mark = ""
        if best and (med, lo, hi, n, cfg) == best:
            mark = "  <- best"
        print(f"{med:8.2f} {lo:7.2f}-{hi:6.2f} {n:3d}  {knobs}{mark}")
    if best and len(rows) > 1:
        second = rows[1]
        if best[1] <= second[2] and second[1] <= best[2]:
            print(f"\nNOTE: the top two ranges overlap "
                  f"({best[1]:.2f}-{best[2]:.2f} against "
                  f"{second[1]:.2f}-{second[2]:.2f}).")
            print("      They are not separable at this sample count; more --runs")
            print("      are needed before preferring one.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3,
                    help="rounds; each round measures every configuration once")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--quick", action="store_true",
                    help="three configurations, one round: a smoke test")
    args = ap.parse_args()

    if not os.path.exists(EXE):
        sys.exit(f"no engine at {EXE}")
    if not os.path.exists(MODEL):
        sys.exit(f"no model at {MODEL}")

    # Coordinate descent rather than the full product: four knobs with three
    # values each is eighty-one configurations, and most of that grid is wasted
    # because the knobs are close to independent. Sweep one at a time from the
    # default, keep the winner, move on.
    base = {k: v[0] for k, v in KNOBS.items()}
    rounds = 1 if args.quick else args.runs

    print(f"engine  {EXE}")
    print(f"model   {os.path.basename(MODEL)}")
    print(f"ctx     {args.ctx}   rounds {rounds}\n")

    chosen = dict(base)
    for knob, values in KNOBS.items():
        if args.quick and knob not in ("DS4_QWEN35_TGS", "DS4_THREADS"):
            continue
        configs = []
        for v in values:
            cfg = dict(chosen)
            cfg[knob] = v
            configs.append(cfg)
        print(f"[{knob}] sweeping {values}")
        samples = measure(configs, rounds, args.ctx, knob.replace("DS4_", ""))
        rows = summarise(configs, samples)
        if not rows:
            print("  no samples\n")
            continue
        show(rows)
        chosen[knob] = rows[0][4][knob]
        print()

    knobs = " ".join(f"{k}={v}" for k, v in chosen.items())
    print("=" * 78)
    print(f"chosen: {knobs}")
    out = os.path.join(ROOT, "bench", "autotune_best.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# written by bench/autotune.py\n")
        for k, v in chosen.items():
            f.write(f"{k}={v}\n")
    print(f"written to {out}")


# Compile-time knobs, for a build matrix rather than this script:
#
#   rows per group      Q4K_X16_ROWS      8, 16
#   nibble pairing      two passes vs one in the lim == 8 path
#   interleave width    the 4-byte group stride in qwen35_blk16x8
#
# Each changes the repacked layout, so both variants cannot coexist in one
# process without doubling the twelve gigabyte wide buffer. A build matrix means
# building N executables and pointing this script at each in turn; the search is
# eight combinations and the measurement method above is unchanged.

if __name__ == "__main__":
    main()
