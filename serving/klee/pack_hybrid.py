"""Pack hybrid A into one checkpoint: klee100 3bpw with the Intel W4 experts of layers 12-35.

Hybrid A used to be assembled at serve time from the full klee parts (arc3-klee3bpw-p1..p4) plus the Intel
shards that hold the experts of layers 12-35 (hybrid.py), with a tensor-filter hook dropping the tensors the
index does not map to a shard. The loader and the precache therefore read ~65 GB that are never used. This
module writes the hybrid as its own checkpoint, split into Kaggle notebook outputs (<20 GB each):

  * every klee shard (model-000NN-of-00011, mtp-*), with klee's own experts of layers 12-35 removed and every
    other tensor kept byte for byte;
  * one shard per swapped layer, `model-kleeA-mid-LL.safetensors`, holding that layer's Intel expert tensors
    (qweight/qzeros/scales of every expert and projection) byte for byte;
  * model.safetensors.index.json (klee's, with the swapped keys pointed at the new shards; the PLE keys still
    point at klee shard 1, which assemble.py replaces with the Intel PLE files), config.humming.json (klee's
    rules with the hybrid rule `+:...layers.(12|..|35).mlp.experts$` -> W4 g128 first, as the notebook did),
    EXPECTED_SPECS.json (precision of every module from the written tensor shapes, read by the preflight),
    the small repo files, and PACK_REPORT.json.

The output directory is named `klee3bpw`, so assemble.py serves it unchanged (KLEE=True, KLEE_HYBRID=False:
no hybridize, no tensor filter). Only the standard library is used; tensors are copied in 64 MB chunks.

Usage (Kaggle CPU notebook with arc3-klee3bpw-p1..p4 and the Intel model attached):
    import pack_hybrid
    pack_hybrid.write_part(part, n_parts=4)
"""
import collections
import json
import math
import re
import shutil
import struct
from pathlib import Path

LM = 'model.language_model.layers.'
EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(qweight|qzeros|scales|g_idx)$')
MIDDLE = range(12, 36)
MID_SPEC = (4, 128)
PLE_SHARD = 'model-00001-of-00011.safetensors'
PART_CAP = 17.5e9  # bytes per notebook output (Kaggle caps one output at 20 GB)
CHUNK = 64 << 20
DSIZE = {'I32': 4, 'F16': 2, 'BF16': 2, 'F32': 4, 'I64': 8, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'I16': 2, 'U16': 2}
FUSE = [(re.compile(r'(.*\.linear_attn)\.in_proj_(qkv|z)$'), r'\1.in_proj_qkvz'),
        (re.compile(r'(.*\.linear_attn)\.in_proj_(b|a)$'), r'\1.in_proj_ba'),
        (re.compile(r'(.*\.self_attn)\.(q|k|v)_proj$'), r'\1.qkv_proj'),
        (re.compile(r'(.*\.mlp\.shared_expert)\.(gate|up)_proj$'), r'\1.gate_up_proj'),
        (re.compile(r'(.*\.mlp\.experts)\.\d+\.(gate|up|down)_proj$'), r'\1')]


def read_header(path):
    """Return a safetensors file's header (with __metadata__) and its data offset."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def is_middle_expert(name):
    m = EXPERT.match(name)
    return bool(m) and int(m.group(1)) in MIDDLE


def fused(name):
    for p, r in FUSE:
        if p.match(name):
            return p.sub(r, name)
    return name


def module_spec(t):
    """(bits, group) from the GPTQ tensor shapes of one module."""
    qw, qz, sc = t['qweight']['shape'], t['qzeros']['shape'], t['scales']['shape']
    bits = qz[1] * 32 / sc[1]
    assert bits == int(bits) and int(bits) in (2, 3, 4, 8), (qw, qz, sc)
    n_in = qw[0] * 32 / bits
    assert n_in == int(n_in) and qw[1] == sc[1] and n_in % sc[0] == 0, (qw, qz, sc)
    return int(bits), int(n_in) // sc[0]


def find_sources(root='/kaggle/input'):
    """Return ({file name: path} of the klee parts, Intel model directory)."""
    parts = sorted({p.parent for p in Path(root).rglob('klee3bpw/model.safetensors.index.json')}
                   | {p.parent for p in Path(root).rglob('klee3bpw/*.safetensors')})
    files = {}
    for part in parts:
        for p in part.iterdir():
            if p.is_file():
                files.setdefault(p.name, p)
    intel = [p.parent for p in Path(root).rglob('model.safetensors.index.json')
             if 'intel-qwen3.8-flash-next-w4a16' in str(p).lower()]
    assert len(parts) == 4, f'expected klee parts p1..p4, found {parts}'
    assert len(intel) == 1, f'expected one Intel model directory, found {intel}'
    return files, intel[0]


def plan(root='/kaggle/input', n_parts=4):
    """Everything to write: output files with their sources, part assignment, index, config, specs."""
    files, intel = find_sources(root)
    index = json.loads(files['model.safetensors.index.json'].read_text())
    wmap = index['weight_map']
    iwmap = json.loads((intel / 'model.safetensors.index.json').read_text())['weight_map']
    kk = {k for k in wmap if is_middle_expert(k)}
    ik = {k for k in iwmap if is_middle_expert(k)}
    if kk != ik:
        raise RuntimeError(f'expert keys differ: klee-only {sorted(kk - ik)[:3]}, intel-only {sorted(ik - kk)[:3]}')
    heads = {}
    out = []  # (name, kind, payload, nbytes): kind 'slim' (klee path, kept keys), 'mid' ([(path, key)]), 'copy'
    for f in sorted(set(wmap.values()) - {PLE_SHARD}):
        h, _ = read_header(files[f])
        h.pop('__metadata__', None)
        heads[f] = h
        keep = [k for k in h if not is_middle_expert(k)]
        size = sum(h[k]['data_offsets'][1] - h[k]['data_offsets'][0] for k in keep)
        out.append((f, 'slim' if len(keep) < len(h) else 'copy', keep, size))
    by_layer = collections.defaultdict(list)
    iheads = {}
    for k in sorted(ik):
        f = iwmap[k]
        if f not in iheads:
            iheads[f] = read_header(intel / f)[0]
        by_layer[int(EXPERT.match(k).group(1))].append((f, k))
    new_wmap = dict(wmap)
    for layer, items in sorted(by_layer.items()):
        name = f'model-kleeA-mid-{layer:02d}.safetensors'
        size = sum(iheads[f][k]['data_offsets'][1] - iheads[f][k]['data_offsets'][0] for f, k in items)
        out.append((name, 'mid', items, size))
        for _, k in items:
            new_wmap[k] = name
    # parts: shards in name order, each part filled up to `cap` (an even share plus slack, at most PART_CAP);
    # the small files go with the last part
    cap = sum(o[3] for o in out) / n_parts
    while True:
        cap = min(PART_CAP, cap * 1.05)
        parts, cur, used = [], [], 0.0
        for o in out:
            if cur and used + o[3] > cap:
                parts.append(cur); cur, used = [], 0.0
            cur.append(o); used += o[3]
        parts.append(cur)
        if len(parts) <= n_parts or cap >= PART_CAP:
            break
    if len(parts) > n_parts:
        raise RuntimeError(f'{len(parts)} parts needed at {cap / 1e9:.1f} GB each: '
                           + ', '.join(f'{sum(o[3] for o in p) / 1e9:.1f}' for p in parts))
    while len(parts) < n_parts:
        parts.append([])
    small = [n for n, p in files.items() if not n.endswith('.safetensors')
             and n not in ('model.safetensors.index.json', 'config.humming.json')
             and not n.startswith(('KLEE_HEADERS.', 'MARKER_')) and p.stat().st_size < 50e6]
    # precision of every language-model module from the tensor shapes as written
    tensor = {}
    for k, f in new_wmap.items():
        if not k.startswith(LM):
            continue
        if f == PLE_SHARD:  # PLE tables (BF16, unquantized) come from the Intel files; no header needed
            tensor[k] = {}
            continue
        if f.startswith('model-kleeA-mid-'):
            src = next(sf for sf, kk2 in by_layer[int(EXPERT.match(k).group(1))] if kk2 == k)
            tensor[k] = iheads[src][k]
        else:
            tensor[k] = heads[f][k]
    per_mod = collections.defaultdict(dict)
    for k, t in tensor.items():
        m, leaf = k.rsplit('.', 1)
        per_mod[m][leaf] = t
    expect = collections.defaultdict(set)
    for m, t in per_mod.items():
        if 'qweight' in t:
            expect[fused(m)].add(module_spec(t))
        elif 'weight' in t:
            expect[fused(m)].add(None)
    mixed = {k: v for k, v in expect.items() if len(v) > 1}
    if mixed:
        raise RuntimeError(f'fused modules with mixed precision: {list(mixed.items())[:4]}')
    expect = {k: next(iter(v)) for k, v in expect.items()}
    lay = lambda m: int(m[len(LM):].split('.', 1)[0])
    bad = [(m, s) for m, s in expect.items() if m.endswith('.mlp.experts')
           and s != (MID_SPEC if lay(m) in MIDDLE else (3, 128))]
    if bad:
        raise RuntimeError(f'unexpected expert precision: {bad[:6]}')
    cfg = json.loads(files['config.humming.json'].read_text())
    q = cfg['quantization_config']
    rule = '+:' + re.escape(LM) + '(' + '|'.join(str(i) for i in MIDDLE) + r')\.mlp\.experts$'
    q['dynamic'] = {rule: {'bits': MID_SPEC[0], 'group_size': MID_SPEC[1], 'sym': True},
                    **{k: v for k, v in q['dynamic'].items() if k != rule}}
    if isinstance(cfg.get('text_config'), dict) and 'quantization_config' in cfg['text_config']:
        cfg['text_config']['quantization_config'] = q
    return dict(files=files, intel=intel, parts=parts, small=small, index=dict(index, weight_map=new_wmap),
                config=cfg, specs={k: (list(v) if v else None) for k, v in expect.items()},
                swapped=len(kk), layers=sorted(by_layer))


def _copy_range(src, dst, start, length):
    src.seek(start)
    while length:
        buf = src.read(min(CHUNK, length))
        if not buf:
            raise IOError(f'short read in {src.name}')
        dst.write(buf)
        length -= len(buf)


def _write_tensors(dst_path, items):
    """items: [(src_path, key)] -> one safetensors file with those tensors, in the given order."""
    heads = {}
    for p, _ in items:
        if p not in heads:
            heads[p] = read_header(p)
    new, offset = {}, 0
    for p, k in items:
        a, b = heads[p][0][k]['data_offsets']
        new[k] = {**heads[p][0][k], 'data_offsets': [offset, offset + b - a]}
        offset += b - a
    blob = json.dumps(new, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    srcs = {}
    try:
        with open(dst_path, 'wb') as dst:
            dst.write(struct.pack('<Q', len(blob)) + blob)
            for p, k in items:
                if p not in srcs:
                    srcs[p] = open(p, 'rb')
                a, b = heads[p][0][k]['data_offsets']
                _copy_range(srcs[p], dst, heads[p][1] + a, b - a)
    finally:
        for f in srcs.values():
            f.close()
    _verify(dst_path, items, heads)


def _verify(dst_path, items, heads):
    """Same names, dtypes, shapes and sizes; first/middle/last 64 bytes of every tensor equal."""
    nh, nbase = read_header(dst_path)
    nh.pop('__metadata__', None)
    if sorted(nh) != sorted(k for _, k in items):
        raise RuntimeError(f'{dst_path}: tensor names differ')
    with open(dst_path, 'rb') as fn:
        for p, k in items:
            oh, obase = heads[p]
            oa, ob = oh[k]['data_offsets']
            na, nb = nh[k]['data_offsets']
            if (nh[k]['dtype'], nh[k]['shape']) != (oh[k]['dtype'], oh[k]['shape']) or nb - na != ob - oa:
                raise RuntimeError(f'{dst_path}: {k} changed')
            if DSIZE.get(nh[k]['dtype'], 0) * math.prod(nh[k]['shape']) != nb - na:
                raise RuntimeError(f'{dst_path}: {k} size does not match dtype x shape')
            with open(p, 'rb') as fo:
                for off in {0, max(0, (ob - oa) // 2 - 32), max(0, ob - oa - 64)}:
                    fo.seek(obase + oa + off)
                    fn.seek(nbase + na + off)
                    if fo.read(64) != fn.read(64):
                        raise RuntimeError(f'{dst_path}: {k} bytes differ at {off}')


def _layout(items, heads):
    """Plan one output file from [(src_path, key)]: header blob, header, copy runs, data size.

    The tensors are laid out in source order (header order does not matter to safetensors), so adjacent
    source tensors merge into one run [src, src_offset, dst_offset, length] (dst_offset counts from the
    start of the file). `heads` caches the source headers.
    """
    for p, _ in items:
        if p not in heads:
            heads[p] = read_header(p)
    items = sorted(items, key=lambda it: (str(it[0]), heads[it[0]][0][it[1]]['data_offsets'][0]))
    new, offset, runs = {}, 0, []
    for p, k in items:
        a, b = heads[p][0][k]['data_offsets']
        new[k] = {**heads[p][0][k], 'data_offsets': [offset, offset + b - a]}
        start = heads[p][1] + a
        if runs and runs[-1][0] == p and runs[-1][1] + runs[-1][3] == start:
            runs[-1][3] += b - a
        else:
            runs.append([p, start, offset, b - a])
        offset += b - a
    blob = json.dumps(new, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    for r in runs:
        r[2] += 8 + len(blob)
    return blob, new, runs, offset


def _check_written(dst_path, new, data_size):
    """Header as planned, sizes match dtype x shape, file size as planned (no re-read of the sources)."""
    nh, nbase = read_header(dst_path)
    nh.pop('__metadata__', None)
    if sorted(nh) != sorted(new) or any((nh[k]['dtype'], nh[k]['shape'], nh[k]['data_offsets'])
                                        != (new[k]['dtype'], new[k]['shape'], new[k]['data_offsets']) for k in new):
        raise RuntimeError(f'{dst_path}: header differs from the plan')
    for k in new:
        a, b = new[k]['data_offsets']
        if DSIZE.get(new[k]['dtype'], 0) * math.prod(new[k]['shape']) != b - a:
            raise RuntimeError(f'{dst_path}: {k} size does not match dtype x shape')
    if Path(dst_path).stat().st_size != nbase + data_size:
        raise RuntimeError(f'{dst_path}: file size {Path(dst_path).stat().st_size} != {nbase + data_size}')


def write_local(out='/tmp/kleea-pack/klee3bpw', root='/kaggle/input', threads=48, piece=32 << 20,
                report_s=20):
    """Write the whole packed hybrid into `out` at serve time, copying 32 MB pieces in parallel.

    For notebooks that attach the full klee parts (arc3-klee3bpw-p1..p4) and the Intel model instead of the
    packed parts: only the tensors the hybrid uses are read (about 65 GB instead of 98 GB of whole shards).
    Every output file is laid out first (header written, size preallocated); then `threads` workers copy
    `piece`-sized ranges from the sources to their places, so one large file is read by many threads at once
    (one stream on the network mount reads only tens of MB/s). Progress is printed every `report_s` seconds.
    `out` should be on the local disk; its parent then serves as assemble.py's `root`.

    Returns:
        A report with the files written, their sizes and run counts, and the seconds taken.

    Raises:
        RuntimeError: Too little disk space, or a written file that does not match its plan.
        IOError: A short read from a source.
    """
    import os
    import shutil as _sh
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    p = plan(root, n_parts=4)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # files: name -> [kind, header, data size, file size, pieces left, runs]
    heads, files, pieces = {}, {}, []
    for part in p['parts']:
        for name, kind, payload, _ in part:
            dst = out / name
            if kind == 'copy':
                size = p['files'][name].stat().st_size
                runs, new, data, blob = [[p['files'][name], 0, 0, size]], None, None, b''
            else:
                items = ([(p['files'][name], k) for k in payload] if kind == 'slim'
                         else [(p['intel'] / f, k) for f, k in payload])
                blob, new, runs, data = _layout(items, heads)
                size = 8 + len(blob) + data
            with open(dst, 'wb') as f:
                if blob:
                    f.write(struct.pack('<Q', len(blob)) + blob)
                f.truncate(size)
            n0 = len(pieces)
            for src, so, do, length in runs:
                for o in range(0, length, piece):
                    pieces.append((src, so + o, dst, do + o, min(piece, length - o)))
            files[name] = [kind, new, data, size, len(pieces) - n0, len(runs)]
    total = sum(f[3] for f in files.values())
    if _sh.disk_usage(out).free < 2e9:
        raise RuntimeError(f'{out}: under 2 GB free after preallocating {total / 1e9:.0f} GB')
    print(f'packing {len(files)} files, {total / 1e9:.1f} GB in {len(pieces)} pieces with {threads} threads '
          f'(planned in {time.time() - t0:.0f} s)', flush=True)
    lock, local, stop = threading.Lock(), threading.local(), threading.Event()
    state = dict(bytes=0, files=0)

    def handle(path, mode):
        h = local.__dict__.setdefault('h', {})
        if (path, mode) not in h:
            h[(path, mode)] = open(path, mode)
        return h[(path, mode)]

    def copy(pc):
        src, so, dst, do, n = pc
        fi, fo = handle(src, 'rb'), handle(dst, 'r+b')
        fi.seek(so)
        buf = fi.read(n)
        if len(buf) != n:
            raise IOError(f'short read in {src} at {so}: {len(buf)} of {n}')
        fo.seek(do)
        fo.write(buf)
        with lock:
            state['bytes'] += n
            f = files[dst.name]
            f[4] -= 1
            if f[4] == 0:
                state['files'] += 1

    def worker(chunk):
        try:
            for pc in chunk:
                copy(pc)
        finally:
            for h in local.__dict__.pop('h', {}).values():
                h.close()

    def progress():
        while not stop.wait(report_s):
            el = time.time() - t0
            print(f'  packing {el:.0f} s: {state["bytes"] / 1e9:.1f} / {total / 1e9:.1f} GB, '
                  f'{state["files"]} / {len(files)} files, {state["bytes"] / 1e6 / max(1.0, el):.0f} MB/s',
                  flush=True)

    threading.Thread(target=progress, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            # every worker takes every threads-th piece, so all of them spread over all files
            list(ex.map(worker, [pieces[i::threads] for i in range(threads)]))
    finally:
        stop.set()
    report = dict(files={}, swapped_tensors=p['swapped'], layers=p['layers'], total_gb=round(total / 1e9, 2))
    for name, (kind, new, data, size, _, n_runs) in files.items():
        if kind == 'copy':
            if (out / name).stat().st_size != size:
                raise RuntimeError(f'{name}: copy size differs')
        else:
            _check_written(out / name, new, data)
        report['files'][name] = dict(kind=kind, gb=round(size / 1e9, 3), runs=n_runs)
    for name in p['small']:
        _sh.copyfile(p['files'][name], out / name)
    (out / 'model.safetensors.index.json').write_text(json.dumps(p['index']))
    (out / 'config.humming.json').write_text(json.dumps(p['config'], indent=1))
    (out / 'EXPECTED_SPECS.json').write_text(json.dumps(p['specs']))
    report.update(small_files=p['small'], module_specs=dict(collections.Counter(str(v) for v in p['specs'].values())),
                  seconds=round(time.time() - t0, 1), gb_per_s=round(total / 1e9 / max(1e-9, time.time() - t0), 3),
                  pid=os.getpid())
    (out / 'PACK_REPORT.local.json').write_text(json.dumps(report, indent=1))
    print(f'packed {len(files)} files, {total / 1e9:.1f} GB in {report["seconds"]} s '
          f'({report["gb_per_s"]} GB/s) to {out}', flush=True)
    return report


def write_part(part, n_parts=4, root='/kaggle/input', out='/kaggle/working/klee3bpw'):
    """Write part `part` (1-based) of the packed hybrid into `out`; the last part also gets the metadata."""
    p = plan(root, n_parts)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mine = p['parts'][part - 1]
    report = dict(part=part, n_parts=n_parts, files={}, swapped_tensors=p['swapped'], layers=p['layers'],
                  parts_gb=[round(sum(o[3] for o in q) / 1e9, 2) for q in p['parts']])
    for name, kind, payload, size in mine:
        src = p['files'].get(name)
        if kind == 'copy':
            shutil.copyfile(src, out / name)
            if (out / name).stat().st_size != src.stat().st_size:
                raise RuntimeError(f'{name}: copy size differs')
        elif kind == 'slim':
            _write_tensors(out / name, [(src, k) for k in payload])
        else:
            _write_tensors(out / name, [(p['intel'] / f, k) for f, k in payload])
        report['files'][name] = dict(kind=kind, gb=round((out / name).stat().st_size / 1e9, 3))
        print(f'part {part}: {name} ({kind}) {report["files"][name]["gb"]} GB', flush=True)
    if part == n_parts:
        for name in p['small']:
            shutil.copyfile(p['files'][name], out / name)
        (out / 'model.safetensors.index.json').write_text(json.dumps(p['index']))
        (out / 'config.humming.json').write_text(json.dumps(p['config'], indent=1))
        (out / 'EXPECTED_SPECS.json').write_text(json.dumps(p['specs']))
        counts = collections.Counter(str(v) for v in p['specs'].values())
        report.update(small_files=p['small'], module_specs=dict(counts))
        print('module precision counts:', dict(counts))
    (out / f'PACK_REPORT.p{part}.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=1))
    return report
