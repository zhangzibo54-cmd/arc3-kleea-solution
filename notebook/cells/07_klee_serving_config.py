# ============================== KLEE 3BPW SERVING CONFIG ==============================
# ISTA variant (serving/ista/build_ista_notebook.py): routed experts from the ISTA GGUF via the ista_moe hook (OFF: W3 baseline)
# Bench variant `klee_w3mid_r25_m125` (github.com/zhangzibo54-cmd/arc3-fast-inference, bench/cell_config.py): the v4 game
# config (NVFP4 main KV + native-FP4 XQA, draft KV fp8, MEMFRAC 0.96, 25 running, 125 Mamba slots
# extra_buffer, no cap2, no skip-decode-lock, chunk 8192, prefill max 3 requests) with the main model replaced by
# klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP served by Humming (bench arc3-klee-3bpw-humming-speed-bench).
V = {'PATCH_NVFP4': True, 'KLEE': True, 'KLEE_HYBRID': False, 'KLEE_PACK_LOCAL': False, 'ISTA': False}  # hybrid A, packed (arc3-kleea-p1..p4): klee100 3bpw with Intel W4 experts in layers 12-35 (W3 ends)
KLEE = False
RESULTS = {}
CFG_OVERRIDES = {'CTX': 148*1024, 'KVDTYPE': 'fp8_e4m3', 'DRAFT_KVDTYPE': 'fp8_e4m3', 'MAXREQ': 20, 'CUDAGRAPH_MAXBS': 20, 'MAMBA_CACHE': 100, 'MAMBA_RADIX': 'extra_buffer', 'MEMFRAC': 0.96, 'CHUNK': 8192, 'MAX_PREFILL': 16384, 'GRAPH_BS': [1, 2, 4, 8, 12, 14, 15, 16, 17, 18, 19, 20]}
ENV_OVERRIDES = {'HUMMING_CACHE_DIR': '/kaggle/working/humming_cache'}
EXTRA_ARGS = ['--prefill-max-requests', '3', '--weight-loader-prefetch-num-threads', '16']
print('CFG_OVERRIDES', CFG_OVERRIDES); print('ENV_OVERRIDES', ENV_OVERRIDES); print('EXTRA_ARGS', EXTRA_ARGS)
