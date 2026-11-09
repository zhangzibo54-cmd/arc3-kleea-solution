# bench hook: index-driven tensor filter for the hybrid klee/Intel checkpoint (variant flag KLEE_HYBRID).
# Appends patches/tensor-filter/tensor_filter_hook.py to sglang/srt/model_loader/loader.py: with
# SGLANG_INDEX_FILTER_DIR=MODEL_DIR every safetensors iterator yields, per shard of MODEL_DIR, only the tensors the
# index maps to that shard (klee W2 experts of layers 12-35 and the non-expert Intel tensors are dropped) and raises
# if a mapped tensor is missing. klee_hybrid_check_load() (after READY) requires "[tensorfilter] OK" in serve.log.
import base64, glob, hashlib, time
from pathlib import Path
KLEE_HYBRID = bool(V.get('KLEE_HYBRID', False))
_tf_src = base64.b64decode('CgojIHRlbnNvcl9maWx0ZXIgaG9vayAoYXJjMyBiZW5jaCk6IHdpdGggU0dMQU5HX0lOREVYX0ZJTFRFUl9ESVI9PG1vZGVsIGRpcj4sIGV2ZXJ5IHNhZmV0ZW5zb3JzIHdlaWdodHMgaXRlcmF0b3IKIyB5aWVsZHMsIGZvciB0aGUgc2hhcmRzIG9mIHRoYXQgZGlyZWN0b3J5LCBvbmx5IHRoZSB0ZW5zb3JzIHRoYXQgbW9kZWwuc2FmZXRlbnNvcnMuaW5kZXguanNvbiBtYXBzIHRvIHRoYXQgc2hhcmQKIyAoaHlicmlkIGNoZWNrcG9pbnRzIHdob3NlIHNoYXJkcyBzaGFyZSB0ZW5zb3IgbmFtZXMpLCBhbmQgcmFpc2VzIGlmIGEgbWFwcGVkIHRlbnNvciBpcyBtaXNzaW5nLgojIFNoYXJkcyBvdXRzaWRlIHRoZSBkaXJlY3RvcnkgKGRyYWZ0IG1vZGVsKSBwYXNzIHRocm91Z2ggdW5jaGFuZ2VkLiBMaW5lcyBzdGFydCB3aXRoICJbdGVuc29yZmlsdGVyXSIuCmRlZiBfYXJjM190ZW5zb3JfZmlsdGVyX2luc3RhbGwoKToKICAgIGltcG9ydCBqc29uIGFzIF9qLCBvcyBhcyBfbywgc3lzIGFzIF9zCiAgICByb290ID0gX28uZW52aXJvbi5nZXQoJ1NHTEFOR19JTkRFWF9GSUxURVJfRElSJykKICAgIGlmIG5vdCByb290OgogICAgICAgIHJldHVybgogICAgcm9vdCA9IF9vLnBhdGguYWJzcGF0aChyb290KQogICAgd21hcCA9IF9qLmxvYWQob3Blbihfby5wYXRoLmpvaW4ocm9vdCwgJ21vZGVsLnNhZmV0ZW5zb3JzLmluZGV4Lmpzb24nKSkpWyd3ZWlnaHRfbWFwJ10KICAgIGFsbG93ID0ge30KICAgIGZvciBrLCBmIGluIHdtYXAuaXRlbXMoKToKICAgICAgICBhbGxvdy5zZXRkZWZhdWx0KGYsIHNldCgpKS5hZGQoaykKICAgIGltcG9ydCBzZ2xhbmcuc3J0Lm1vZGVsX2xvYWRlci53ZWlnaHRfdXRpbHMgYXMgX3d1CiAgICBuYW1lcyA9IHNvcnRlZChuIGZvciBuIGluIGRpcihfd3UpIGlmIG4uZW5kc3dpdGgoJ3dlaWdodHNfaXRlcmF0b3InKSBhbmQgJ3NhZmV0ZW5zb3JzJyBpbiBuKQoKICAgIGRlZiBtYWtlKG9yaWcsIGZuYW1lKToKICAgICAgICBpZiBnZXRhdHRyKG9yaWcsICdfYXJjM19maWx0ZXJlZCcsIEZhbHNlKToKICAgICAgICAgICAgcmV0dXJuIG9yaWcKICAgICAgICBkZWYgaXQoaGZfd2VpZ2h0c19maWxlcywgKmEsICoqa3cpOgogICAgICAgICAgICBmaWxlcyA9IGxpc3QoaGZfd2VpZ2h0c19maWxlcykKICAgICAgICAgICAgb3VycyA9IFtmIGZvciBmIGluIGZpbGVzIGlmIF9vLnBhdGguZGlybmFtZShfby5wYXRoLmFic3BhdGgoZikpID09IHJvb3RdCiAgICAgICAgICAgIG90aGVycyA9IFtmIGZvciBmIGluIGZpbGVzIGlmIGYgbm90IGluIG91cnNdCiAgICAgICAgICAgIGlmIG90aGVyczoKICAgICAgICAgICAgICAgIHlpZWxkIGZyb20gb3JpZyhvdGhlcnMsICphLCAqKmt3KQogICAgICAgICAgICBrZXB0X2FsbCA9IGRyb3BwZWRfYWxsID0gMAogICAgICAgICAgICBmb3IgZiBpbiBvdXJzOgogICAgICAgICAgICAgICAgYmFzZSA9IF9vLnBhdGguYmFzZW5hbWUoZikKICAgICAgICAgICAgICAgIHdhbnQsIGdvdCwgZHJvcHBlZCA9IGFsbG93LmdldChiYXNlLCBzZXQoKSksIHNldCgpLCAwCiAgICAgICAgICAgICAgICBmb3IgbmFtZSwgdCBpbiBvcmlnKFtmXSwgKmEsICoqa3cpOgogICAgICAgICAgICAgICAgICAgIGlmIG5hbWUgaW4gd2FudDoKICAgICAgICAgICAgICAgICAgICAgICAgZ290LmFkZChuYW1lKQogICAgICAgICAgICAgICAgICAgICAgICB5aWVsZCBuYW1lLCB0CiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgZHJvcHBlZCArPSAxCiAgICAgICAgICAgICAgICBtaXNzID0gd2FudCAtIGdvdAogICAgICAgICAgICAgICAgcHJpbnQoZidbdGVuc29yZmlsdGVyXSB7Zm5hbWV9IHtiYXNlfToga2VwdCB7bGVuKGdvdCl9IGRyb3BwZWQge2Ryb3BwZWR9IG1pc3Npbmcge2xlbihtaXNzKX0nLAogICAgICAgICAgICAgICAgICAgICAgZmx1c2g9VHJ1ZSkKICAgICAgICAgICAgICAgIGlmIG1pc3M6CiAgICAgICAgICAgICAgICAgICAgcmFpc2UgUnVudGltZUVycm9yKGYnW3RlbnNvcmZpbHRlcl0ge2Jhc2V9OiB7bGVuKG1pc3MpfSBpbmRleGVkIHRlbnNvcnMgbm90IHlpZWxkZWQsICcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZidlLmcuIHtzb3J0ZWQobWlzcylbOjNdfScpCiAgICAgICAgICAgICAgICBrZXB0X2FsbCArPSBsZW4oZ290KTsgZHJvcHBlZF9hbGwgKz0gZHJvcHBlZAogICAgICAgICAgICBpZiBvdXJzOgogICAgICAgICAgICAgICAgcHJpbnQoZidbdGVuc29yZmlsdGVyXSBPSyB7Zm5hbWV9OiB7bGVuKG91cnMpfSBzaGFyZHMsIGtlcHQge2tlcHRfYWxsfSwgZHJvcHBlZCB7ZHJvcHBlZF9hbGx9JywKICAgICAgICAgICAgICAgICAgICAgIGZsdXNoPVRydWUpCiAgICAgICAgaXQuX2FyYzNfZmlsdGVyZWQgPSBUcnVlCiAgICAgICAgcmV0dXJuIGl0CgogICAgd3JhcHBlZCA9IFtdCiAgICBmb3IgbiBpbiBuYW1lczoKICAgICAgICB3ID0gbWFrZShnZXRhdHRyKF93dSwgbiksIG4pCiAgICAgICAgc2V0YXR0cihfd3UsIG4sIHcpCiAgICAgICAgd3JhcHBlZC5hcHBlbmQobikKICAgICAgICBmb3IgbW9kIGluIGxpc3QoX3MubW9kdWxlcy52YWx1ZXMoKSk6ICAgIyBtb2R1bGVzIHRoYXQgaW1wb3J0ZWQgdGhlIGZ1bmN0aW9uIGJ5IG5hbWUgKGxvYWRlci5weSAuLi4pCiAgICAgICAgICAgIGQgPSBnZXRhdHRyKG1vZCwgJ19fZGljdF9fJywgTm9uZSkKICAgICAgICAgICAgaWYgZCBpcyBub3QgTm9uZSBhbmQgbiBpbiBkIGFuZCBnZXRhdHRyKGRbbl0sICdfX21vZHVsZV9fJywgJycpID09IF93dS5fX25hbWVfXyBhbmQgZFtuXSBpcyBub3QgdzoKICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBkW25dID0gdwogICAgICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgICAgICBwYXNzCiAgICBnID0gZ2xvYmFscygpCiAgICBmb3IgbiBpbiBuYW1lczoKICAgICAgICBnW25dID0gZ2V0YXR0cihfd3UsIG4pCiAgICBwcmludChmJ1t0ZW5zb3JmaWx0ZXJdIGluc3RhbGxlZCBpbiBwaWQge19vLmdldHBpZCgpfToge3dyYXBwZWR9OyB7bGVuKGFsbG93KX0gc2hhcmRzIGluZGV4ZWQgdW5kZXIge3Jvb3R9JywKICAgICAgICAgIGZsdXNoPVRydWUpCgoKX2FyYzNfdGVuc29yX2ZpbHRlcl9pbnN0YWxsKCkK')
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
