"""Hybrid klee/Intel checkpoint: routed experts of selected layers from the Intel W4A16 model.

klee100 3bpw stores routed experts as W3 g128 (layers 0-11, 36-47) and W2 g64 (12-35). The hybrid keeps
everything from klee except the routed experts of `layers` (default 12-35), which are taken from
Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (same AutoRound 0.15, packing auto_round:auto_gptq, sym int, W4 g128).

Both checkpoints keep each layer's experts in shards that also hold other tensors, and SGLang's safetensors
iterators yield every tensor of every listed shard. The index weight_map produced here is therefore the source
of truth, and the server must run with the tensor-filter hook (patches/tensor-filter): for each shard of this
directory it yields only the names the index maps to that shard, and raises if one is missing.

`hybridize(model_dir, intel_dir, layers)` (after assemble.assemble):
  1. key check: per layer, the Intel expert keys == the klee expert keys (qweight/qzeros/scales of every
     expert and projection); Intel shard names must not collide with klee names;
  2. links the Intel shards and points those keys at them in model.safetensors.index.json;
  3. shape check from the safetensors headers of every shard: the precision of every quantized
     language-model module (bits = qzeros cols * 32 / scales cols, in = qweight rows * 32 / bits,
     group = in / scales rows) must be consistent within a fused SGLang module, the swapped layers must be
     mid_spec, the rest klee's own (klee_spec), and the input/output sizes of each swapped module must equal klee's;
  4. writes EXPECTED_SPECS.json (SGLang module -> [bits, group] or None) for the preflight and
     HYBRID_SOURCE.json (per-shard tensor counts and bytes by source).
Returns the extra Humming `dynamic` rule (checkpoint prefix) for the swapped experts.
"""
import collections
import json
import os
import re
import struct
from pathlib import Path

LM = 'model.language_model.layers.'
EXPERT = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(qweight|qzeros|scales|g_idx)$')
FUSE = [(re.compile(r'(.*\.linear_attn)\.in_proj_(qkv|z)$'), r'\1.in_proj_qkvz'),
        (re.compile(r'(.*\.linear_attn)\.in_proj_(b|a)$'), r'\1.in_proj_ba'),
        (re.compile(r'(.*\.self_attn)\.(q|k|v)_proj$'), r'\1.qkv_proj'),
        (re.compile(r'(.*\.mlp\.shared_expert)\.(gate|up)_proj$'), r'\1.gate_up_proj'),
        (re.compile(r'(.*\.mlp\.experts)\.\d+\.(gate|up|down)_proj$'), r'\1')]
DSIZE = {'I32': 4, 'F16': 2, 'BF16': 2, 'F32': 4, 'I64': 8, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'I16': 2, 'U16': 2}


def fused(n):
    for p, r in FUSE:
        if p.match(n):
            return p.sub(r, n)
    return n


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return h


def module_spec(t):
    """(bits, group, in, out) from the GPTQ tensors of one module, or raise."""
    qw, qz, sc = t['qweight']['shape'], t['qzeros']['shape'], t['scales']['shape']
    bits = qz[1] * 32 / sc[1]
    assert bits == int(bits) and int(bits) in (2, 3, 4, 8), (qw, qz, sc)
    bits = int(bits)
    n_in = qw[0] * 32 / bits
    assert n_in == int(n_in) and qw[1] == sc[1] and n_in % sc[0] == 0, (qw, qz, sc)
    return bits, int(n_in) // sc[0], int(n_in), sc[1]


def klee_spec(layer):
    """klee100's own routed-expert precision (bits, group) for `layer`: W2 g64 in 12-35, W3 g128 elsewhere."""
    return (2, 64) if 12 <= layer < 36 else (3, 128)


def hybridize(model_dir, intel_dir, layers=range(12, 36), mid_spec=(4, 128)):
    """intel_dir: one directory or a list of directories (each with model.safetensors.index.json) that provide
    the routed experts of `layers` (Intel W4: mid_spec (4,128); our W3 MSE-clip shards: mid_spec (3,128)).
    `layers` may be any set: the default swaps klee's W2 middle; layers 0-11 + 36-47 keep that W2 middle
    and swap the W3 ends. Layers not swapped must keep klee's own precision (klee_spec)."""
    srcs = [Path(d).absolute() for d in ([intel_dir] if isinstance(intel_dir, (str, Path)) else intel_dir)]
    mp, intel = Path(model_dir).absolute(), srcs[0]
    layers = set(layers)
    idx_path = mp / 'model.safetensors.index.json'
    index = json.loads(idx_path.read_text())
    wmap = index['weight_map']
    iwmap, src_of = {}, {}
    for d in srcs:
        for k, f in json.loads((d / 'model.safetensors.index.json').read_text())['weight_map'].items():
            if f in src_of and src_of[f] != d:
                raise RuntimeError(f'shard name {f} in two source dirs')
            iwmap[k] = f; src_of[f] = d

    sel = lambda m: {k for k in m if (e := EXPERT.match(k)) and int(e.group(1)) in layers}
    kk, ik = sel(wmap), sel(iwmap)
    assert kk, 'no klee expert keys for the selected layers'
    if kk != ik:
        raise RuntimeError(f'expert keys differ: klee-only {sorted(kk - ik)[:4]} intel-only {sorted(ik - kk)[:4]} '
                           f'({len(kk)} vs {len(ik)})')
    ifiles = sorted({iwmap[k] for k in ik})
    klee_files = set(wmap.values())
    for f in ifiles:
        if f in klee_files and not ((mp / f).is_symlink() and str((mp / f).resolve()).startswith(str(src_of[f].resolve()))):
            raise RuntimeError(f'Intel shard name {f} collides with a klee shard')
    old_src = {k: wmap[k] for k in kk}
    for f in ifiles:
        dst = mp / f
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src_of[f] / f)
    for k in kk:
        wmap[k] = iwmap[k]
    idx_path.unlink()
    idx_path.write_text(json.dumps(index))

    # headers of every shard the index uses; only the tensors the index maps to that shard count
    heads = {f: header(mp / f) for f in sorted(set(wmap.values()))}
    missing = [k for k, f in wmap.items() if k not in heads[f]]
    if missing:
        raise RuntimeError(f'{len(missing)} index keys not in their shard, e.g. {missing[:3]}')
    per_mod = collections.defaultdict(dict)
    for k, f in wmap.items():
        if k.startswith(LM):
            m, leaf = k.rsplit('.', 1)
            per_mod[m][leaf] = heads[f][k]
    spec = {}
    for m, t in per_mod.items():
        if 'qweight' in t:
            b, g, n_in, n_out = module_spec(t)
            spec[m] = (b, g, n_in, n_out)
        elif 'weight' in t:
            spec[m] = None
    expect = collections.defaultdict(set)
    for m, s in spec.items():
        expect[fused(m)].add(None if s is None else s[:2])
    mixed = {k: v for k, v in expect.items() if len(v) > 1}
    if mixed:
        raise RuntimeError(f'fused modules with mixed precision: {list(mixed.items())[:4]}')
    expect = {k: next(iter(v)) for k, v in expect.items()}
    lay = lambda m: int(m[len(LM):].split('.', 1)[0])
    bad = []
    for m, s in expect.items():
        if m.endswith('.mlp.experts'):
            want = tuple(mid_spec) if lay(m) in layers else klee_spec(lay(m))
            if s != want:
                bad.append((m, s, want))
    if bad:
        raise RuntimeError(f'unexpected expert precision: {bad[:6]}')
    # swapped modules: same in/out as klee had (klee header of the original shard;
    # shards slimmed by slim_klee.py keep their original header in KLEE_HEADERS.<part>.json)
    kheads = {f: header(mp / f) for f in sorted(set(old_src.values()))}
    for sidecar in sorted(mp.glob('KLEE_HEADERS.*.json')):
        for f, h in json.loads(sidecar.read_text()).items():
            if f in kheads:
                kheads[f] = {**h, **kheads[f]}
    for k in sorted(kk):
        if k.endswith('.scales'):
            m = k.rsplit('.', 1)[0]
            kt = {leaf: kheads[old_src[f'{m}.{leaf}']][f'{m}.{leaf}'] for leaf in ('qweight', 'qzeros', 'scales')}
            if module_spec(kt)[2:] != spec[m][2:]:
                raise RuntimeError(f'{m}: klee in/out {module_spec(kt)[2:]} != Intel {spec[m][2:]}')

    def nbytes(h):
        return DSIZE.get(h['dtype'], 0) * int(__import__('math').prod(h['shape']))
    src = collections.defaultdict(lambda: [0, 0])
    for k, f in wmap.items():
        lk = mp / f
        s = src['source' if lk.is_symlink() and Path(os.readlink(lk)).parent in srcs else 'klee']
        s[0] += 1; s[1] += nbytes(heads[f][k])
    dtypes = collections.Counter((('source' if k in kk else 'klee'), heads[wmap[k]][k]['dtype'])
                                 for k in wmap if k.endswith('.scales') and '.mlp.experts.' in k)
    counts = collections.Counter(str(v) for v in expect.values())
    report = dict(layers=sorted(layers), sources=[str(d) for d in srcs], mid_spec=list(mid_spec), source_files=ifiles, swapped_tensors=len(kk),
                  by_source={k: dict(tensors=v[0], gb=round(v[1] / 1e9, 2)) for k, v in src.items()},
                  expert_scale_dtypes={f'{a}:{b}': n for (a, b), n in dtypes.items()}, modules=dict(counts))
    (mp / 'EXPECTED_SPECS.json').write_text(json.dumps({k: (list(v) if v else None) for k, v in expect.items()}))
    (mp / 'HYBRID_SOURCE.json').write_text(json.dumps(report, indent=1))
    print('hybrid:', json.dumps(report))
    rx = re.escape(LM) + '(' + '|'.join(str(l) for l in sorted(layers)) + r')\.mlp\.experts$'
    return '+:' + rx, {'bits': int(mid_spec[0]), 'group_size': int(mid_spec[1]), 'sym': True}
