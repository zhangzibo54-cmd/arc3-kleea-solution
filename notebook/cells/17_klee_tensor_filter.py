# bench hook: index-driven tensor filter for the hybrid klee/Intel checkpoint (variant flag KLEE_HYBRID).
# Appends patches/tensor-filter/tensor_filter_hook.py to sglang/srt/model_loader/loader.py: with
# SGLANG_INDEX_FILTER_DIR=MODEL_DIR every safetensors iterator yields, per shard of MODEL_DIR, only the tensors the
# index maps to that shard (klee W2 experts of layers 12-35 and the non-expert Intel tensors are dropped) and raises
# if a mapped tensor is missing. klee_hybrid_check_load() (after READY) requires "[tensorfilter] OK" in serve.log.
import base64, glob, hashlib, time
from pathlib import Path
KLEE_HYBRID = bool(V.get('KLEE_HYBRID', False))
_tf_src = code_file('tensor_filter.py')
assert hashlib.md5(_tf_src).hexdigest() == '5e73ec123e1078bc6a7bc0709e72a07b', 'tensor_filter hook corrupted'

def _install_tensor_filter(venv):
    f = Path(sorted(glob.glob(f"{venv}/lib/python*/site-packages/sglang/srt/model_loader/loader.py"))[0])
    if b'tensor_filter hook' not in f.read_bytes():
        f.write_bytes(f.read_bytes() + _tf_src)
    print('tensor_filter installed in', f)

_apply_nv_prev_tf = apply_nvfp4_patch
def apply_nvfp4_patch(venv, python, run_env):
    _apply_nv_prev_tf(venv, python, run_env)
    if KLEE_HYBRID:
        _install_tensor_filter(venv)
        run_env['SGLANG_INDEX_FILTER_DIR'] = MODEL_DIR

def klee_hybrid_check_load(log_path, timeout_s=45 * 60):
    """Wait until the main model is loaded; raise unless the tensor filter ran and found every tensor."""
    if not KLEE_HYBRID:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        text = Path(log_path).read_text(errors='ignore') if Path(log_path).exists() else ''
        lines = [l for l in text.splitlines() if '[tensorfilter]' in l]
        if any('[tensorfilter] OK' in l for l in lines) or proc.poll() is not None:
            break
        time.sleep(10)
    print('\n'.join(l for l in lines if 'kept' not in l or 'OK' in l))
    shard_lines = [l for l in lines if ' kept ' in l and 'OK' not in l]
    print(f'{len(shard_lines)} shard lines; first/last:', shard_lines[:1], shard_lines[-1:])
    RESULTS.setdefault('klee', {})['tensor_filter'] = [l for l in lines if 'OK' in l or 'installed' in l]
    if not any('[tensorfilter] OK' in l for l in lines) or any('not yielded' in l for l in lines):
        raise RuntimeError('hybrid checkpoint: tensor filter did not run or a tensor is missing; see serve.log')
print('KLEE_HYBRID =', KLEE_HYBRID)
