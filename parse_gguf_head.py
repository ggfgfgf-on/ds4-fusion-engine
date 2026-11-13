import struct, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

GGUF_TYPES = {
    0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32',
    7: 'bool', 8: 'str', 9: 'arr', 10: 'u64', 11: 'i64', 12: 'f64',
}
ARR_SUBTYPES = {
    0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32',
    7: 'bool', 8: 'str', 10: 'u64', 11: 'i64', 12: 'f64',
}

def read_str(f, off):
    n = struct.unpack_from('<Q', f, off)[0]
    off += 8
    s = f[off:off+n].decode('utf-8', 'replace')
    return s, off + n

def main(path):
    with open(path, 'rb') as fp:
        f = fp.read()
    magic, ver, n_tensors, n_kv = struct.unpack_from('<IIQQ', f, 0)
    print(f"magic={magic:#x} version={ver} n_tensors={n_tensors} n_kv={n_kv}")
    off = 24
    for i in range(n_kv):
        key, off = read_str(f, off)
        t = struct.unpack_from('<I', f, off)[0]; off += 4
        tname = GGUF_TYPES.get(t, '?')
        if t == 8:
            val, off = read_str(f, off)
        elif t in (0,1,2,3,4,5,6,7,10,11,12):
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
            val = struct.unpack_from('<'+('?' if t==7 else {'0':'B','1':'b','2':'H','3':'h','4':'I','5':'i','6':'f','10':'Q','11':'q','12':'d'}[str(t)]), f, off)[0]; off += sz
        elif t == 9:
            st, n = struct.unpack_from('<IQ', f, off); off += 12
            if st == 8:
                vals = []
                for _ in range(n):
                    v, off = read_str(f, off)
                    vals.append(v)
                val = vals
            elif st == 4:
                val = struct.unpack_from(f'<{n}I', f, off); off += 4*n
            elif st == 6:
                val = struct.unpack_from(f'<{n}f', f, off); off += 4*n
            elif st == 10:
                val = struct.unpack_from(f'<{n}Q', f, off); off += 8*n
            elif st == 5:
                val = struct.unpack_from(f'<{n}i', f, off); off += 4*n
            else:
                off += n * {0:1,1:1,2:2,3:2,7:1,11:8,12:8}.get(st, 4)
                val = f'<array[{st}] skipped>'
        else:
            val = '?'
        vs = str(val)
        if len(vs) > 110: vs = vs[:110] + '...'
        print(f"  {key} [{tname}] = {vs}")
    # tensor names
    print(f"--- tensors (offset {off}) ---")
    for i in range(n_tensors):
        name, off = read_str(f, off)
        nd, off = struct.unpack_from('<I', f, off)[0], off + 4
        dims = struct.unpack_from(f'<{nd}Q', f, off); off += 8*nd
        ttype = struct.unpack_from('<I', f, off)[0]; off += 4
        toff = struct.unpack_from('<Q', f, off)[0]; off += 8
        print(f"  {name}: dims={dims} type={ttype} off={toff}")

main(sys.argv[1])
