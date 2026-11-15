// fill_ranges.cs — parallel in-place range filler for partially downloaded files.
// Usage: fill_ranges.exe <manifest.json> <outFile> <baseUrl> [proxy] [workers] [chunkBytes]
//   manifest: [[offset,length],...]
//   baseUrl:  e.g. https://hf-mirror.com/antirez/deepseek-v4-gguf/resolve/main/foo.gguf
//   proxy:    optional, e.g. http://127.0.0.1:7890  ("" = direct)
//   workers:  default 16; chunkBytes default 2MB
// Resumable: completed chunks recorded in <manifest>.progress.log
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

class FillRanges
{
    class Chunk { public long Off; public long Len; public int Retry; }

    static long doneBytes = 0, totalBytes = 0;
    static object logLock = new object();
    static StreamWriter progressLog;
    static HashSet<string> doneSet = new HashSet<string>();

    static int Main(string[] args)
    {
        if (args.Length < 3) { Console.Error.WriteLine("usage: fill_ranges.exe <manifest.json> <outFile> <baseUrl> [proxy] [workers] [chunkBytes]"); return 2; }
        string manifestPath = args[0], outFile = args[1], baseUrl = args[2];
        string proxy = args.Length > 3 ? args[3] : "";
        int workers = args.Length > 4 ? int.Parse(args[4]) : 16;
        long chunkBytes = args.Length > 5 ? long.Parse(args[5]) : 2L * 1024 * 1024;

        var ranges = new List<long[]>();
        ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072; // TLS 1.2
        ServicePointManager.DefaultConnectionLimit = 128;
        if (proxy == "direct" || proxy == "-" || proxy == "none") proxy = "";
        using (var sr = new StreamReader(manifestPath))
        {
            string s = sr.ReadToEnd();
            foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(s, @"\[(\d+),(\d+)\]"))
                ranges.Add(new long[] { long.Parse(m.Groups[1].Value), long.Parse(m.Groups[2].Value) });
        }
        if (ranges.Count == 0) { Console.Error.WriteLine("no ranges in manifest"); return 1; }

        var chunks = new List<Chunk>();
        foreach (var r in ranges)
        {
            long off = r[0], len = r[1];
            while (len > 0)
            {
                long l = Math.Min(chunkBytes, len);
                chunks.Add(new Chunk { Off = off, Len = l });
                off += l; len -= l;
            }
        }
        foreach (var c in chunks) totalBytes += c.Len;

        // load progress log for resume
        string progPath = manifestPath + ".progress.log";
        if (File.Exists(progPath))
            foreach (var line in File.ReadAllLines(progPath))
                if (line.Length > 0) doneSet.Add(line);

        // filter out already-done chunks
        var todo = new List<Chunk>();
        long skipBytes = 0;
        foreach (var c in chunks)
        {
            string key = c.Off + ":" + c.Len;
            if (doneSet.Contains(key)) skipBytes += c.Len;
            else todo.Add(c);
        }
        Console.WriteLine("chunks=" + chunks.Count + " todo=" + todo.Count + " skipBytes=" + skipBytes + " total=" + totalBytes);
        if (todo.Count == 0) { Console.WriteLine("all chunks already done"); return 0; }

        progressLog = new StreamWriter(progPath, true) { AutoFlush = true };
        doneBytes = skipBytes;

        var handler = new HttpClientHandler();
        if (!string.IsNullOrEmpty(proxy)) handler.Proxy = new WebProxy(proxy);
        handler.UseProxy = !string.IsNullOrEmpty(proxy);
        handler.MaxConnectionsPerServer = workers + 4;
        var client = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(120) };
        client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (fill-ranges)");
        client.DefaultRequestHeaders.Add("Accept-Encoding", "identity");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var fs = new FileStream(outFile, FileMode.Open, FileAccess.Write, FileShare.ReadWrite);

        var queue = new Queue<Chunk>(todo);
        int active = 0, failed = 0, fatal = 0;
        object qLock = new object();
        var lastLog = DateTime.MinValue;

        Action<Chunk, byte[]> writeChunk = (c, data) =>
        {
            lock (fs) { fs.Position = c.Off; fs.Write(data, 0, data.Length); }
            lock (logLock)
            {
                progressLog.WriteLine(c.Off + ":" + c.Len);
                doneSet.Add(c.Off + ":" + c.Len);
                doneBytes += c.Len;
                if ((DateTime.Now - lastLog).TotalSeconds >= 5)
                {
                    lastLog = DateTime.Now;
                    double pct = doneBytes * 100.0 / totalBytes;
                    double mbps = doneBytes / 1048576.0 / sw.Elapsed.TotalSeconds;
                    Console.WriteLine("[{0:HH:mm:ss}] {1:F1}% {2:F2}MiB/s eta={3:F0}min", DateTime.Now, pct, mbps, (totalBytes - doneBytes) / 1048576.0 / Math.Max(0.1, mbps) / 60.0);
                }
            }
        };

        var tasks = new List<Task>();
        for (int w = 0; w < workers; w++)
        {
            tasks.Add(Task.Run(async () =>
            {
                while (true)
                {
                    Chunk c = null;
                    lock (qLock) { if (queue.Count > 0) { c = queue.Dequeue(); Interlocked.Increment(ref active); } }
                    if (c == null) return;
                    try
                    {
                        var req = new HttpRequestMessage(HttpMethod.Get, baseUrl);
                        req.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(c.Off, c.Off + c.Len - 1);
                        using (var resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead))
                        {
                            if (resp.StatusCode != HttpStatusCode.PartialContent)
                            {
                                Console.Error.WriteLine("HTTP " + (int)resp.StatusCode + " for chunk @" + c.Off + " (range refused?)");
                                Interlocked.Increment(ref fatal);
                                throw new Exception("non-206");
                            }
                            using (var st = await resp.Content.ReadAsStreamAsync())
                            {
                                var buf = new byte[c.Len];
                                int got = 0;
                                while (got < buf.Length)
                                {
                                    int r = await st.ReadAsync(buf, got, buf.Length - got);
                                    if (r <= 0) throw new Exception("short read @" + c.Off);
                                    got += r;
                                }
                                writeChunk(c, buf);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Interlocked.Increment(ref failed);
                        c.Retry++;
                        if (c.Retry <= 8 && Volatile.Read(ref fatal) == 0)
                        {
                            lock (qLock) queue.Enqueue(c);
                            await Task.Delay(500 * c.Retry);
                        }
                        else
                        {
                            Console.Error.WriteLine("GAVE UP chunk @" + c.Off + ": " + ex.Message);
                        }
                    }
                    finally { Interlocked.Decrement(ref active); }
                }
            }));
        }
        Task.WaitAll(tasks.ToArray());
        fs.Close(); progressLog.Close();
        sw.Stop();
        double pct = doneBytes * 100.0 / totalBytes;
        Console.WriteLine("DONE {0:F1}% bytes={1} failures={2} elapsed={3:F1}s", pct, doneBytes, failed, sw.Elapsed.TotalSeconds);
        return pct > 99.99 ? 0 : 3;
    }
}
