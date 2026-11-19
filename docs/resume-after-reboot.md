# Resume after the reboot

Everything below is the state at commit `007e3b0`, verified: prefill 36.8 t/s on
the twenty-two token prompt and 46.2 on the hundred and seventeen token one,
decode 2.71, output "The capital of France is Paris."

## Why the reboot

uProf reports `Virtual Machine: Yes`, `Hypervisor Vendor: Microsoft Hyper-V`, and
rejects `ibs_op` as an invalid event. IBS is not passed through to a Hyper-V root
partition, so the instruction-level stall attribution that would explain the last
factor of 2.9 is unavailable.

That factor is the whole remaining question:

    same kernel, one core, weights from DRAM      0.698 dpbusd per cycle
    same kernel, sixteen cores, weights from DRAM 0.238 per core

Sixteen cores are slower per core than one. DRAM is at 6.2 of the 51 to 55 GB/s
it can deliver, a pure streaming read reaches 51 to 55 with sixteen partitioned
threads, core utilisation is 76 percent, and 70 percent of CPU time is inside the
two kernels. Eight targeted experiments - flush removal, broadcast removal, weight
prefetch, activation transpose, nibble split, token group size, register spilling,
block granularity prefetch - all came back neutral or negative.

## What the reboot changed

Not bcdedit. `hypervisorlaunchtype` stays Auto; what was disabled are the features
that make Auto mean something:

    Microsoft-Hyper-V-Hypervisor      Enabled -> Disabled
    VirtualMachinePlatform            Enabled -> Disabled     (WSL2)
    Containers-DisposableClientVM     Enabled -> Disabled     (Windows Sandbox)

Memory integrity was already off - HVCI registry Enabled = 0, and DeviceGuard
reported no security services running - so the hypervisor was there for the
virtualisation features, not for security.

To put it back:

    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-Hypervisor
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM
    (then reboot)

Disabling Containers-DisposableClientVM takes the Windows Sandbox with it, so the
sandbox_* tools stop working until it is re-enabled.

## First thing to check

    & 'C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe' collect `
        --event ibs_op --output-dir $env:TEMP\ibs --interval 20000 `
        wsweep.exe out.csv

If `ibs_op` is accepted now, the five-year-hunt ends here. Then read the report:

    AMDuProfCLI.exe report --input-dir <session dir> --report-output out.csv

The session directory is one level below the output directory, named
`AMDuProf-<program>-<config>_<date>`.

## If IBS works, profile these two in this order

1. `bench/wsweep.exe` single-threaded. It reaches 0.698 dpbusd per cycle from
   DRAM where the engine reaches 0.238 per core across sixteen. Same code, so
   the difference is purely concurrency, and IBS on the single-threaded run is
   the control.
2. `bench/launch_ds4.exe` - the launcher exists because uProfCLI parses the
   arguments of the program it starts and takes the engine's `--cpu` and `--ctx`
   for its own. It bakes the engine's command line in so uProf sees a program it
   can start with none.

## Where the time actually goes (uProf, 117 token prefill)

    wall 3.668 s, 44.85 s CPU over 16 physical cores = 76 percent

    qwen35_cpu_matvec_batch_worker   17.43 s  38.9 %   Q6_K 2x4, inlined
    qwen35_gemm_q4_K_16x8            13.82 s  30.8 %   the wide kernel
    ds4_worker_main                   7.09 s  15.8 %   spin
    ntoskrnl.exe                      3.17 s   7.1 %

The Q6_K share is larger than the Q4_K share even though Q6_K is 4.4 GB against
14.4 - because the Q6_K path is the 2x4 kernel walking its tensors once per four
tokens, while the Q4_K path is the wide kernel walking each tensor once for all
tokens. That asymmetry was never attacked. It is the obvious first target if the
stall data does not say otherwise: Q6_K at 38.9 percent for 22 percent of the
weights is 1.8 times over-represented.

## Tools left behind

    bench/autotune.py        coordinate descent over the runtime knobs, interleaved
                             rounds, reports interquartile ranges and refuses to
                             pick a winner from noise. Wrote autotune_best.txt.
    bench/wsweep.c           working-set sweep, single-threaded kernel
    bench/wsweep_mt.c        the same, threaded
    bench/bwceil.c           true bandwidth ceiling and the kernel against it,
                             persistent pool with a spin barrier
    bench/dpbusd_thr.c       instruction ceiling, 10.5 Gop/s per core
    bench/launch_ds4.c       launcher so uProf can profile the engine
    bench/kernel16x8_test.c  standalone correctness test for the wide kernel

    DS4_QWEN35_TGS        tokens per kernel pass
    DS4_QWEN35_NIBS       nibble pairing, 1 (default) or 2
    DS4_QWEN35_PFD        block prefetch distance, 0 disables (measured neutral)
    DS4_QWEN35_TRANS      transposed activations (measured neutral)
    DS4_QWEN35_BYTE_DBG   phase and byte breakdown
    DS4_PROF              the in-repo sampling profiler

## The number that matters

    bandwidth floor for 117 tokens   15.8 GB / 55 GB/s  = 0.29 s   400 t/s
    compute floor                    2.57 T-MAC / 13.3  = 0.19 s   616 t/s
    actual                                               2.54 s    46 t/s

Eight and a half times off the bandwidth floor, not twelve percent of it. The
engine is not near any ceiling.
