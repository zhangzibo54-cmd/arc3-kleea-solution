# ============================== NVFP4 KV PATCH (only used when PATCH_NVFP4) ==============================
# Source: github.com/cookys/sglang-qsa-nvfp4-kv (Apache-2.0, see NOTICE), developed on Pennyroyal at fb1216c6c.
# Ported to Pennyroyal d00d88e (the wheel used here): 4/5 target files are byte-identical; the chunked-prefill
# history gather in qwen_sparse_attn_backend.py was refactored upstream (single gather_index), so its two anchors
# were re-written for that code. Local fixes (v2): history gather over the joined index (one gather + one
# dequant), and decode/verify dequantize only the batch*stride rows actually read (not the CUDA-graph capacity). Everything else is the original applier, which refuses a tree it does not match.
import base64, json, glob, subprocess
PATCH_NVFP4 = bool(V.get('PATCH_NVFP4', False))
NVFP4_DIR = Path('/tmp/nvfp4patch')
_NVFP4_FILES = {n: code_file(n).decode() for n in ['apply_qsa_nvfp4_patches.py', 'qsa_nvfp4_kv.py', 'test_qsa_nvfp4_gather.py', 'test_concurrent_needle.py', 'NOTICE']}

NVFP4_DIR.mkdir(parents=True, exist_ok=True)
for _name, _text in _NVFP4_FILES.items():
    (NVFP4_DIR / _name).write_text(_text)

def apply_nvfp4_patch(venv, python, run_env):
    srt = sorted(glob.glob(f"{venv}/lib/python*/site-packages/sglang/srt"))[0]
    env_ = dict(os.environ, SGLANG_SRT=srt)
    for args_ in (['--check'], []):
        r = subprocess.run([sys.executable, str(NVFP4_DIR / 'apply_qsa_nvfp4_patches.py'), *args_],
                           env=env_, capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-3000:])
        if r.returncode != 0 and 'already' not in (r.stdout + r.stderr):
            raise RuntimeError('NVFP4 patch failed: ' + ' '.join(args_))
    # standalone GPU unit test (gather/dequant byte-exactness, NaN-scratch guard) before the server takes the GPU
    r = subprocess.run([python, str(NVFP4_DIR / 'test_qsa_nvfp4_gather.py')], capture_output=True, text=True,
                       env=run_env, timeout=1200)
    print('gather test rc', r.returncode); print(r.stdout[-3000:]); print(r.stderr[-3000:])
    RESULTS['nvfp4_gather_test_rc'] = r.returncode
    if r.returncode != 0:
        raise RuntimeError('NVFP4 gather unit test failed')

def run_needle_test(n):
    r = subprocess.run([sys.executable, str(NVFP4_DIR / 'test_concurrent_needle.py'), str(SERVED_MODEL_PORT), str(n), '9000'],
                       capture_output=True, text=True, timeout=1800)
    print(r.stdout[-5000:], r.stderr[-2000:])
    RESULTS['needle_test'] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
print('PATCH_NVFP4 =', PATCH_NVFP4)

# bench hook: optional int8 Mamba-checkpoint support for Qwen4-Exp PLE side states (patches/int8-ckpt-ple/apply_int8_ckpt_ple.py)
INT8_CKPT_PLE = bool(V.get('INT8_CKPT_PLE', False)); import base64 as _b64, hashlib as _hl; _i8src = code_file('int8_ckpt_ple.py'); assert _hl.md5(_i8src).hexdigest() == 'e54f898aac008c89bc05bfeba68e6679', 'int8 patch corrupted'; open('/tmp/i8p.py', 'wb').write(_i8src); _apply_nv_orig = apply_nvfp4_patch; apply_nvfp4_patch = lambda venv, python, run_env: (_apply_nv_orig(venv, python, run_env), print(subprocess.run([sys.executable, '/tmp/i8p.py'], env=dict(os.environ, SGLANG_SRT=sorted(glob.glob(venv + '/lib/python*/site-packages/sglang/srt'))[0]), capture_output=True, text=True)) if INT8_CKPT_PLE else None); print('INT8_CKPT_PLE =', INT8_CKPT_PLE, 'md5 ok')
