// scan_zero.cs — find maximal all-zero regions in a partially-downloaded file.
// Real GGUF tensor bytes never contain long zero runs; preallocated-but-unwritten
// regions are exact zeros. Memory-mapped parallel scan, 64KB blocks.
// Usage: scan_zero.exe <file> <outJson>
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using System.Threading.Tasks;

class ScanZero
{
    static long BLOCK = 65536;

    static int Main(string[] args)
    {
        if (args.Length < 2) { Console.Error.WriteLine("usage: scan_zero.exe <file> <outJson> [blockBytes]"); return 2; }
        string path = args[0];
        if (args.Length > 2) BLOCK = long.Parse(args[2]);
        long fileLen = new FileInfo(path).Length;
        long nBlocks = fileLen / BLOCK;
        int threads = Math.Max(1, Environment.ProcessorCount - 1);
        var zeroBits = new byte[(nBlocks + 7) / 8];

        Console.WriteLine("fileLen=" + fileLen + " blocks=" + nBlocks + " threads=" + threads);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using (var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read))
        {
            Parallel.For(0, threads, t =>
            {
                long per = (nBlocks + threads - 1) / threads;
                long start = t * per, end = Math.Min(nBlocks, start + per);
                if (start >= end) return;
                long spanLen = (end - start) * BLOCK;
                long alignStart = (start * BLOCK) & ~0xFFFL;
                long alignEnd = ((end * BLOCK) + 0xFFF) & ~0xFFFL;
                if (alignEnd > fileLen) alignEnd = fileLen;
                var acc = mmf.CreateViewAccessor(alignStart, alignEnd - alignStart, MemoryMappedFileAccess.Read);
                var buf = new byte[BLOCK];
                for (long b = start; b < end; b++)
                {
                    long offInView = b * BLOCK - alignStart;
                    acc.ReadArray(offInView, buf, 0, (int)BLOCK);
                    bool zero = true;
                    for (int i = 0; i < BLOCK; i++) { if (buf[i] != 0) { zero = false; break; } }
                    if (zero) lock (zeroBits) { zeroBits[b >> 3] |= (byte)(1 << (int)(b & 7)); }
                }
            });
        }
        sw.Stop();
        Console.WriteLine("scan_done_sec=" + sw.Elapsed.TotalSeconds.ToString("F1"));

        // Merge zero blocks into ranges
        var ranges = new List<long[]>();
        long runStart = -1;
        for (long b = 0; b <= nBlocks; b++)
        {
            bool isZero = b < nBlocks && ((zeroBits[b >> 3] >> (int)(b & 7)) & 1) == 1;
            if (isZero && runStart < 0) runStart = b;
            if (!isZero && runStart >= 0)
            {
                long off = runStart * BLOCK;
                long len = (b - runStart) * BLOCK;
                ranges.Add(new long[] { off, len });
                runStart = -1;
            }
        }
        // handle tail beyond last full block
        long tailOff = nBlocks * BLOCK;
        if (fileLen > tailOff)
        {
            bool tailZero;
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                fs.Seek(tailOff, SeekOrigin.Begin);
                var tb = new byte[fileLen - tailOff];
                int got = 0; while (got < tb.Length) { int r = fs.Read(tb, got, tb.Length - got); if (r <= 0) break; got += r; }
                tailZero = true;
                foreach (var x in tb) if (x != 0) { tailZero = false; break; }
            }
            if (tailZero) ranges.Add(new long[] { tailOff, fileLen - tailOff });
        }

        long zeroBytes = 0; foreach (var r in ranges) zeroBytes += r[1];
        var sb = new System.Text.StringBuilder();
        sb.Append("[");
        for (int i = 0; i < ranges.Count; i++)
        {
            if (i > 0) sb.Append(",");
            sb.Append("[").Append(ranges[i][0]).Append(",").Append(ranges[i][1]).Append("]");
        }
        sb.Append("]");
        File.WriteAllText(args[1], sb.ToString());
        Console.WriteLine("zeroRanges=" + ranges.Count);
        Console.WriteLine("zeroBytes=" + zeroBytes + " (" + (zeroBytes / 1024.0 / 1024 / 1024).ToString("F2") + " GiB missing)");
        Console.WriteLine("manifest=" + args[1]);
        return 0;
    }
}
