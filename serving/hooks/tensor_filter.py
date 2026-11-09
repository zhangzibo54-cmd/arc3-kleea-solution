

# tensor_filter hook (arc3 bench): with SGLANG_INDEX_FILTER_DIR=<model dir>, every safetensors weights iterator
# yields, for the shards of that directory, only the tensors that model.safetensors.index.json maps to that shard
# (hybrid checkpoints whose shards share tensor names), and raises if a mapped tensor is missing.
# Shards outside the directory (draft model) pass through unchanged. Lines start with "[tensorfilter]".
def _arc3_tensor_filter_install():
    import json as _j, os as _o, sys as _s
    root = _o.environ.get('SGLANG_INDEX_FILTER_DIR')
    if not root:
        return
    root = _o.path.abspath(root)
    wmap = _j.load(open(_o.path.join(root, 'model.safetensors.index.json')))['weight_map']
    allow = {}
    for k, f in wmap.items():
        allow.setdefault(f, set()).add(k)
    import sglang.srt.model_loader.weight_utils as _wu
    names = sorted(n for n in dir(_wu) if n.endswith('weights_iterator') and 'safetensors' in n)

    def make(orig, fname):
        if getattr(orig, '_arc3_filtered', False):
            return orig
        def it(hf_weights_files, *a, **kw):
            files = list(hf_weights_files)
            ours = [f for f in files if _o.path.dirname(_o.path.abspath(f)) == root]
            others = [f for f in files if f not in ours]
            if others:
                yield from orig(others, *a, **kw)
            kept_all = dropped_all = 0
            for f in ours:
                base = _o.path.basename(f)
                want, got, dropped = allow.get(base, set()), set(), 0
                for name, t in orig([f], *a, **kw):
                    if name in want:
                        got.add(name)
                        yield name, t
                    else:
                        dropped += 1
                miss = want - got
                print(f'[tensorfilter] {fname} {base}: kept {len(got)} dropped {dropped} missing {len(miss)}',
                      flush=True)
                if miss:
                    raise RuntimeError(f'[tensorfilter] {base}: {len(miss)} indexed tensors not yielded, '
                                       f'e.g. {sorted(miss)[:3]}')
                kept_all += len(got); dropped_all += dropped
            if ours:
                print(f'[tensorfilter] OK {fname}: {len(ours)} shards, kept {kept_all}, dropped {dropped_all}',
                      flush=True)
        it._arc3_filtered = True
        return it

    wrapped = []
    for n in names:
        w = make(getattr(_wu, n), n)
        setattr(_wu, n, w)
        wrapped.append(n)
        for mod in list(_s.modules.values()):   # modules that imported the function by name (loader.py ...)
            d = getattr(mod, '__dict__', None)
            if d is not None and n in d and getattr(d[n], '__module__', '') == _wu.__name__ and d[n] is not w:
                try:
                    d[n] = w
                except Exception:
                    pass
    g = globals()
    for n in names:
        g[n] = getattr(_wu, n)
    print(f'[tensorfilter] installed in pid {_o.getpid()}: {wrapped}; {len(allow)} shards indexed under {root}',
          flush=True)


_arc3_tensor_filter_install()
