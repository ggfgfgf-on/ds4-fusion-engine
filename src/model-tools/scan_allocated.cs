// Scan NTFS allocated ranges of a sparse file via FSCTL_QUERY_ALLOCATED_RANGES.
// Usage: scan_allocated.exe <file> [windowMB]
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

class AllocScan
{
    const uint FSCTL_QUERY_ALLOCATED_RANGES = 0x000940CF;

    [StructLayout(LayoutKind.Sequential)]
    struct Range { public long Offset; public long Length; }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool DeviceIoControl(IntPtr h, uint code, IntPtr inBuf, uint inSize,
                                       IntPtr outBuf, uint outSize, out uint returned, IntPtr ov);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern IntPtr CreateFileW(string name, uint access, uint share, IntPtr sec,
                                     uint disp, uint flags, IntPtr tpl);

    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);

    static int Main(string[] args)
    {
        if (args.Length < 1) { Console.Error.WriteLine("usage: scan_allocated.exe <file> [windowMB]"); return 2; }
        string path = args[0];
        long window = args.Length > 1 ? long.Parse(args[1]) * 1024L * 1024L : 128L * 1024 * 1024;
        long fileLen = new FileInfo(path).Length;
        if (fileLen == 0) { Console.WriteLine("[]"); return 0; }

        IntPtr h = CreateFileW(path, 0x80000000 /*GENERIC_READ*/, 7 /*R|W|D*/, IntPtr.Zero,
                               3 /*OPEN_EXISTING*/, 0, IntPtr.Zero);
        if (h == new IntPtr(-1)) { Console.Error.WriteLine("CreateFile failed: " + Marshal.GetLastWin32Error()); return 1; }

        var ranges = new List<Range>();
        int outBufRanges = 4096;
        for (long start = 0; start < fileLen; start += window)
        {
            long len = Math.Min(window, fileLen - start);
            while (true)
            {
                var inBuf = new Range[1] { new Range { Offset = start, Length = len } };
                var outBuf = new Range[outBufRanges];
                int inSize = Marshal.SizeOf(typeof(Range));
                int outSize = outBufRanges * inSize;
                IntPtr inPtr = Marshal.AllocHGlobal(inSize);
                IntPtr outPtr = Marshal.AllocHGlobal(outSize);
                try
                {
                    Marshal.StructureToPtr(inBuf[0], inPtr, false);
                    uint returned;
                    if (!DeviceIoControl(h, FSCTL_QUERY_ALLOCATED_RANGES, inPtr, (uint)inSize,
                                         outPtr, (uint)outSize, out returned, IntPtr.Zero))
                    {
                        int err = Marshal.GetLastWin32Error();
                        if (err == 234 /*ERROR_MORE_DATA*/)
                        {
                            outBufRanges *= 2;
                            continue;
                        }
                        Console.Error.WriteLine("DeviceIoControl failed: " + err);
                        return 1;
                    }
                    int n = (int)(returned / (uint)Marshal.SizeOf(typeof(Range)));
                    for (int i = 0; i < n; i++)
                    {
                        var r = (Range)Marshal.PtrToStructure(IntPtr.Add(outPtr, i * Marshal.SizeOf(typeof(Range))), typeof(Range));
                        ranges.Add(r);
                    }
                    break;
                }
                finally { Marshal.FreeHGlobal(inPtr); Marshal.FreeHGlobal(outPtr); }
            }
        }
        CloseHandle(h);

        // Merge adjacent ranges
        ranges.Sort((a, b) => a.Offset.CompareTo(b.Offset));
        var merged = new List<Range>();
        foreach (var r in ranges)
        {
            if (merged.Count > 0 && r.Offset <= merged[merged.Count - 1].Offset + merged[merged.Count - 1].Length)
            {
                var last = merged[merged.Count - 1];
                long end = Math.Max(last.Offset + last.Length, r.Offset + r.Length);
                last.Length = end - last.Offset;
                merged[merged.Count - 1] = last;
            }
            else merged.Add(r);
        }

        long allocated = 0;
        foreach (var r in merged) allocated += r.Length;
        long holes = 0;
        for (int i = 1; i < merged.Count; i++)
            holes += merged[i].Offset - (merged[i - 1].Offset + merged[i - 1].Length);

        var sb = new System.Text.StringBuilder();
        sb.Append("[");
        for (int i = 0; i < merged.Count; i++)
        {
            if (i > 0) sb.Append(",");
            sb.Append("[").Append(merged[i].Offset).Append(",").Append(merged[i].Length).Append("]");
        }
        sb.Append("]");
        File.WriteAllText(path + ".allocated.json", sb.ToString());
        Console.WriteLine("fileLength=" + fileLen);
        Console.WriteLine("allocatedBytes=" + allocated + " (" + (allocated / 1024.0 / 1024 / 1024).ToString("F2") + " GiB)");
        Console.WriteLine("allocatedRanges(merged)=" + merged.Count);
        Console.WriteLine("holeBytes=" + holes + " (" + (holes / 1024.0 / 1024 / 1024).ToString("F2") + " GiB)");
        Console.WriteLine("manifest=" + path + ".allocated.json");
        return 0;
    }
}
