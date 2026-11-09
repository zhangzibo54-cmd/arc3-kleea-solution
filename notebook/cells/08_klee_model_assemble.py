# ============================== KLEE 3BPW MODEL (only used when KLEE) ==============================
# klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP: routed experts W3 g128 (layers 0-11, 36-47) / W2 g64 (12-35),
# QSA/indexer/GDN projections W8 g128, everything else BF16. Mirrored as zhangzibo54/arc3-klee3bpw-p1..p4;
# shard 1 (102 GB BF16 n-gram/PLE table) is not mirrored: assemble.py links the identical Intel PLE-only files.
# Served with `--quantization humming` (SGLang reads per-layer precision only from GPTQModel-style `dynamic` rules,
# built from AutoRound's extra_config by tools/humming_config.py -> config.humming.json in p4).
#  1. assemble a symlinked model dir in /tmp (PLE keys -> Intel files); config.json = Intel architecture config +
#     klee quantization_config (config.humming.json); klee's own layer_types names are unknown to Pennyroyal
#  2. make the `dynamic` rules independent of the module-prefix style: the checkpoint says
#     `model.language_model.layers.N`, SGLang's runtime prefix may be `model.layers.N` (hf_to_sglang mapper).
#     A rule that does not match would silently treat a quantized module as BF16 (or vice versa).
#  3. klee_humming_preflight (called by the launcher inside the venv, before loading 50 GB):
#     resolves every language-model module with SGLang's own HummingConfig, for each prefix style, and compares
#     with the AutoRound precision of the checkpoint -> stops before the load on any mismatch.
import base64 as _base64, hashlib as _hashlib, json as _json, re as _re, sys as _sys, subprocess as _subprocess
from pathlib import Path as _Path

KLEE = bool(V.get('KLEE', False))
KLEE_TOKENIZER_MATCH = True
_LM_LIT = r'model\.language_model\.layers\.'
_LM_ANY = r'(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.'

if KLEE:
    _assemble_path = _Path('/tmp/klee_assemble.py')
    _assemble_path.write_bytes(code_file('assemble.py'))
    _sys.path.insert(0, '/tmp'); _sys.modules.pop('klee_assemble', None)
    from klee_assemble import assemble as _assemble

    INTEL_MODEL_DIR = str(_Path(MODEL_DIR).resolve())
    if V.get('KLEE_PACK_LOCAL', False):
        # hybrid A packed at serve time (bench/klee_model/pack_hybrid.py write_local): only the tensors the
        # hybrid uses are read from the full klee parts + Intel model, written to the local disk, then served
        # as the packed parts (KLEE_HYBRID False)
        _Path('/tmp/klee_pack.py').write_bytes(code_file('pack_hybrid.py'))
        _sys.modules.pop('klee_pack', None)
        from klee_pack import write_local as _write_local
        _pack = _write_local(out='/tmp/kleea-pack/klee3bpw', threads=96)
        RESULTS['klee_pack'] = {k: v for k, v in _pack.items() if k != 'files'}
        MODEL_DIR = _assemble(INTEL_MODEL_DIR, out='/tmp/klee3bpw-model', root='/tmp/kleea-pack')
    else:
        MODEL_DIR = _assemble(INTEL_MODEL_DIR, out='/tmp/klee3bpw-model')
    # hybrid (variant flag KLEE_HYBRID): routed experts of layers 12-35 from the Intel W4 model instead of klee W2;
    # needs the tensor-filter hook (cell_tensorfilter.py) because both checkpoints share tensor names per shard
    KLEE_HYBRID = bool(V.get('KLEE_HYBRID', False))
    _hyb_rule = None
    if KLEE_HYBRID:
        _Path('/tmp/klee_hybrid.py').write_bytes(code_file('hybrid.py'))
        _sys.modules.pop('klee_hybrid', None)
        from klee_hybrid import hybridize as _hybridize
        # HYBRID_SOURCE: 'intel' (Intel W4 g128 experts) or 'w3mid' (our W3 g128 MSE-clip experts, notebooks
        # arc3-w3mid-q1/q2 attached as inputs: directories named w3mid with model.safetensors.index.json)
        _src = V.get('HYBRID_SOURCE', 'intel')
        if _src == 'w3mid':
            _dirs = sorted({str(p.parent) for p in _Path('/kaggle/input').rglob('w3mid/model.safetensors.index.json')})
            assert len(_dirs) == 2, f'expected the two w3mid parts (arc3-w3mid-q1/q2), found {_dirs}'
            _hyb_rule = _hybridize(MODEL_DIR, _dirs, layers=range(12, 36), mid_spec=(3, 128))
        else:
            _hyb_rule = _hybridize(MODEL_DIR, INTEL_MODEL_DIR, layers=range(12, 36))
        print('hybrid source:', _src)
    _mp = _Path(MODEL_DIR)
    _cfg = _json.loads((_mp / 'config.json').read_text())
    _q = _cfg['quantization_config']
    # Architecture fields come from the Intel config (validated on this Pennyroyal build); the klee config uses
    # upstream names that Pennyroyal's Qwen4Exp does not know (layer_types 'qwen_sparse_attention' -> KeyError in
    # qwen3_5.get_layer). Only the quantization_config is taken from klee. Differences are printed for the record.
    _icfg = _json.loads((_Path(INTEL_MODEL_DIR) / 'config.json').read_text())
    def _flat(d, p=''):
        out = {}
        for k, v in d.items():
            if k == 'quantization_config': continue
            out.update(_flat(v, p + k + '.') if isinstance(v, dict) else {p + k: v})
        return out
    _fk, _fi = _flat(_cfg), _flat(_icfg)
    _diff = sorted(k for k in set(_fk) | set(_fi) if _fk.get(k, '<absent>') != _fi.get(k, '<absent>'))
    print(f'klee vs Intel config: {len(_diff)} differing keys (outside quantization_config)')
    for _k in _diff[:60]:
        _a, _b = repr(_fk.get(_k, '<absent>')), repr(_fi.get(_k, '<absent>'))
        print(f'  {_k}: klee={_a[:160]} | intel={_b[:160]}')
    RESULTS['klee_config_diff'] = {k: [str(_fk.get(k, '<absent>'))[:200], str(_fi.get(k, '<absent>'))[:200]] for k in _diff}
    _cfg = dict(_icfg, quantization_config=_q)
    if isinstance(_cfg.get('text_config'), dict) and 'quantization_config' in _cfg['text_config']:
        _cfg['text_config'] = dict(_cfg['text_config'], quantization_config=_q)
    assert _q.get('dynamic') and _q.get('quant_method') == 'auto-round', 'assembled config is not config.humming.json'
    if _hyb_rule is not None:  # first '+' rule wins in Humming; the swapped experts get W4 g128
        _q['dynamic'] = {_hyb_rule[0]: _hyb_rule[1], **{k: v for k, v in _q['dynamic'].items() if k != _hyb_rule[0]}}
    # keep the verified rules (checkpoint prefix) for the preflight, then generalize the prefix
    (_mp / 'HUMMING_RULES_CKPT.json').write_text(_json.dumps(_q))
    _dyn = {k.replace(_LM_LIT, _LM_ANY): v for k, v in _q['dynamic'].items()}
    assert len(_dyn) == len(_q['dynamic'])
    _q['dynamic'] = _dyn
    (_mp / 'config.json').unlink(missing_ok=True)
    (_mp / 'config.json').write_text(_json.dumps(_cfg, indent=1))
    print('quantization_config (top level):', {k: v for k, v in _q.items() if k != 'dynamic'})
    print(f'Humming dynamic rules: {len(_dyn)}'); [print('  ', k, v) for k, v in _dyn.items()]

    # PLE: only symlinks into the Intel dir (loaded to host memory by --ple-offload-embedding)
    _ple = _json.loads((_mp / 'PLE_SOURCE.json').read_text())['files']
    _idx = _json.loads((_mp / 'model.safetensors.index.json').read_text())['weight_map']
    _ple_t = [k for k, f in _idx.items() if f in set(_ple)]
    assert _ple_t and all((_mp / f).is_symlink() and str((_mp / f).resolve()).startswith(INTEL_MODEL_DIR) for f in _ple)
    _shards = sorted(set(_idx.values()))
    _missing = [f for f in _shards if not (_mp / f).is_file()]
    assert not _missing, f'unresolved shards: {_missing}'
    print(f'shards: {len(_shards)} ({len(_ple)} Intel PLE files, {len(_ple_t)} PLE tensors); '
          f'mtp tensors in index: {sum(k.startswith("mtp.") for k in _idx)} (target load skips them; draft = albucino)')
    print('text_config PLE dtype:', _cfg.get('text_config', {}).get('ple_embedding_dtype', 'default'),
          '| architectures:', _cfg.get('architectures'))

    def _sha(p):
        h = _hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
        return h.hexdigest()
    for _n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'):
        _a, _b = _mp / _n, _Path(INTEL_MODEL_DIR) / _n
        _same = _a.is_file() and _b.is_file() and _sha(_a) == _sha(_b)
        print(f'{_n}: klee {"present" if _a.is_file() else "MISSING"}, same as Intel: {_same}')
        if _n == 'tokenizer.json': KLEE_TOKENIZER_MATCH = _same
    _Path(ENV_OVERRIDES.get('HUMMING_CACHE_DIR', '/kaggle/working/humming_cache')).mkdir(parents=True, exist_ok=True)
    RESULTS['klee'] = dict(model_dir=MODEL_DIR, ple_files=len(_ple), shards=len(_shards),
                           tokenizer_match=KLEE_TOKENIZER_MATCH, dynamic_rules=len(_dyn), hybrid=KLEE_HYBRID)
    if KLEE_HYBRID:
        RESULTS['klee']['hybrid_source'] = _json.loads((_mp / 'HYBRID_SOURCE.json').read_text())

_PREFLIGHT = r'''
import collections, importlib, inspect, json, pkgutil, re, sys, traceback
from pathlib import Path
mp = Path(sys.argv[1])
cfg = json.loads((mp / 'config.json').read_text()); qc = cfg['quantization_config']
ref = json.loads((mp / 'HUMMING_RULES_CKPT.json').read_text())
wmap = json.loads((mp / 'model.safetensors.index.json').read_text())['weight_map']
LM = 'model.language_model.layers.'
FUSE = [(re.compile(r'(.*\.linear_attn)\.in_proj_(qkv|z)$'), r'\1.in_proj_qkvz'),
        (re.compile(r'(.*\.linear_attn)\.in_proj_(b|a)$'), r'\1.in_proj_ba'),
        (re.compile(r'(.*\.self_attn)\.(q|k|v)_proj$'), r'\1.qkv_proj'),
        (re.compile(r'(.*\.mlp\.shared_expert)\.(gate|up)_proj$'), r'\1.gate_up_proj'),
        (re.compile(r'(.*\.mlp\.experts)\.\d+\.(gate|up|down)_proj$'), r'\1')]
def fused(n):
    for p, r in FUSE:
        if p.match(n): return p.sub(r, n)
    return n
def mirror(c, prefix):   # tools/humming_config.py: verified against AutoRound extra_config, 0 mismatches
    if any(e in prefix for e in (c.get('modules_to_not_convert') or [])) or 'lm_head' in prefix: return None
    d = c.get('dynamic', {})
    if any(k[:1] == '-' and re.match(k[2:], prefix) for k in d): return None
    l = next(({**c, **o} for k, o in d.items() if k[:1] == '+' and re.match(k[2:], prefix)), c)
    return (int(l['bits']), int(l['group_size']))
q = {k.rsplit('.', 1)[0] for k in wmap if k.endswith('.qweight')}
w = {k.rsplit('.', 1)[0] for k in wmap if k.endswith('.weight')}
mods = sorted({fused(m) for m in q | w if m.startswith(LM)})
expect = {m: mirror(ref, m) for m in mods}
spec_file = mp / 'EXPECTED_SPECS.json'
if spec_file.exists():  # hybrid: precision derived from the tensor shapes of every shard (klee_hybrid.py)
    shape = {k: (tuple(v) if v else None) for k, v in json.loads(spec_file.read_text()).items()}
    diff = [(m, expect[m], shape.get(m, '<absent>')) for m in mods if shape.get(m, '<absent>') != expect[m]]
    print(f'rules vs tensor shapes: {len(mods)} modules, {len(diff)} mismatches', diff[:6])
    if diff:
        print('PREFLIGHT MISMATCH'); sys.exit(0)
    expect = {m: shape[m] for m in mods}
print('checkpoint modules:', len(mods), collections.Counter(str(v) for v in expect.values()))

import sglang.srt.layers.quantization as Q
HC = None
try:
    HC = Q.get_quantization_config('humming')
except Exception:
    for m in pkgutil.walk_packages(Q.__path__, Q.__name__ + '.'):
        if 'humming' in m.name:
            HC = getattr(importlib.import_module(m.name), 'HummingConfig', None) or HC
print('HummingConfig:', HC, inspect.getsourcefile(HC))
try: print('override_quantization_method ->', HC.override_quantization_method(qc, 'humming'))
except Exception as e: print('override_quantization_method error:', e)
hc = HC.from_config(qc)
meths = [n for n in dir(hc) if not n.startswith('__') and callable(getattr(hc, n))]
print('methods:', meths)
for n in ('is_layer_skipped', 'get_layer_weight_schema', 'get_layer_config', 'get_quant_method'):
    if hasattr(hc, n): print(' ', n, inspect.signature(getattr(hc, n)))
try:  # model class: prefix mapper
    from sglang.srt.models.registry import ModelRegistry
    cls = ModelRegistry.resolve_model_cls(cfg['architectures'])[0]
    mapper = getattr(cls, 'hf_to_sglang_mapper', None)
    print('model class:', cls.__name__, '| hf_to_sglang_mapper:', getattr(mapper, 'orig_to_new_prefix', mapper))
    src = inspect.getsource(sys.modules[cls.__module__])
    print('add_prefix names:', sorted(set(re.findall(r'add_prefix\(\s*["\']([\w\.]+)["\']', src)))[:40])
except Exception as e:
    print('model class lookup failed:', repr(e)[:300])

def bits_of(r):
    for a in ('bits', 'weight_bits', 'num_bits', 'b_bits'):
        v = getattr(r, a, None) if not isinstance(r, dict) else r.get(a)
        if isinstance(v, int): break
    else:
        m = re.search(r'uint(\d)', repr(r)); v = int(m.group(1)) if m else None
    g = getattr(r, 'group_size', None) if not isinstance(r, dict) else r.get('group_size')
    if g is None:
        m = re.search(r'group_size\D{0,3}(\d+)', repr(r)); g = int(m.group(1)) if m else None
    return (v, g)
TWO = len(inspect.signature(hc.is_layer_skipped).parameters) >= 2   # Pennyroyal: (config, prefix)
def resolve(prefix):
    a = (qc, prefix) if TWO else (prefix,)
    if hc.is_layer_skipped(*a): return None
    return bits_of(hc.get_layer_weight_schema(*a))
try:
    for _m in [m for m in mods if m.endswith('.mlp.experts')][:1] + [m for m in mods if m.endswith('.mlp.experts')][-1:]:
        _a = (qc, 'model.layers.' + _m[len(LM):]) if TWO else ('model.layers.' + _m[len(LM):],)
        print('schema repr sample', _m, repr(hc.get_layer_weight_schema(*_a))[:600])
except Exception as e:
    print('schema sample failed:', repr(e)[:300])

status = 'UNVERIFIED'
try:
    styles = ['model.layers.', 'model.language_model.layers.', 'language_model.model.layers.']
    bad = {}
    for s in styles:
        bad[s] = [(m, expect[m], r) for m in mods for r in [resolve(s + m[len(LM):])]
                  if r != expect[m] and not (r is not None and None in r)]   # unparsed schemas are not mismatches
        print(f'style {s:32s}: {len(mods)} modules, {len(bad[s])} mismatches', bad[s][:4])
    parsed = [resolve('model.layers.' + m[len(LM):]) for m in mods]
    if all(r is None or None in r for r in parsed):
        status = 'UNVERIFIED'   # schema attributes not recognized; the rule-vs-shape check above still holds
        print('schema bits/group not parseable; see schema repr sample')
    else:
        status = 'OK' if not any(bad.values()) else 'MISMATCH'
except Exception:
    traceback.print_exc(limit=3)
    s = inspect.getsource(HC).splitlines()
    print('--- HummingConfig source (dynamic / skip / schema lines) ---')
    for i, l in enumerate(s):
        if re.search(r'def |dynamic|skip|schema|prefix|mapper', l): print(f'{i:4d} {l[:150]}')
print('PREFLIGHT', status)
'''

def klee_humming_preflight(python, run_env, model_dir):
    """Run in the Pennyroyal venv before the server loads weights; raise on a precision mismatch."""
    p = _Path('/tmp/klee_preflight.py'); p.write_text(_PREFLIGHT)
    r = _subprocess.run([python, str(p), model_dir], capture_output=True, text=True, env=run_env, timeout=900)
    out = (r.stdout + ('\n' + r.stderr[-3000:] if r.returncode else ''))
    _Path('/kaggle/working/klee_preflight.log').write_text(r.stdout + '\n' + r.stderr)
    print(out[-12000:])
    status = 'ERROR' if r.returncode else (out.rsplit('PREFLIGHT ', 1)[-1].split()[0] if 'PREFLIGHT ' in out else '?')
    RESULTS.setdefault('klee', {})['preflight'] = status
    if status == 'MISMATCH':
        raise RuntimeError('Humming dynamic rules resolve to the wrong precision; see /kaggle/working/klee_preflight.log')
    print('klee preflight:', status, '(UNVERIFIED/ERROR = resolution API not recognized; load continues, '
          'a wrong rule fails at load or in the needle/sanity check)')
