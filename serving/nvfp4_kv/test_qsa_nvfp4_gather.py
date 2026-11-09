"""Standalone GPU test for qsa_nvfp4_kv.compact_and_dequant / gather_history_fp4.

Reference = torch gather of NVFP4KVQuantizeUtil.dequantize(whole small pool),
i.e. the same FlashInfer dequant kernel, so valid rows must match EXACTLY.
Strided tail rows must be exactly 0, and a NaN-poisoned scratch must not leak.
"""
import sys

import torch

from sglang.srt.layers.attention import qsa_nvfp4_kv as m
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

torch.manual_seed(0)
dev = torch.device("cuda:0")
N, H, D = 8192, 1, 256  # pool rows, kv heads per rank, head_dim (Flash-Next TP=2)
R, L = 6, 4096  # req_to_token rows / max context
PAGE = 64


class FakeReqPool:
    def __init__(self):
        self.req_to_token = torch.randint(0, N, (R, L), dtype=torch.int32, device=dev)


class FakeBackend:
    _get_fa2_scratch = QwenSparseAttnBackend._get_fa2_scratch

    def __init__(self):
        self._fa2_scratch = {}
        self.req_to_token_pool = FakeReqPool()


def make_pool(gs_val=1.0):
    k = torch.randn(N, H, D, dtype=torch.bfloat16, device=dev) * 3
    v = torch.randn(N, H, D, dtype=torch.bfloat16, device=dev) * 3
    gs = torch.tensor([gs_val], dtype=torch.float32, device=dev)
    k_fp4, k_sf, _ = NVFP4KVQuantizeUtil.quantize(k, gs)
    v_fp4, v_sf, _ = NVFP4KVQuantizeUtil.quantize(v, gs)
    return k_fp4.view(torch.uint8), v_fp4.view(torch.uint8), k_sf, v_sf, gs


def run_case(topk, batch, out_dtype, poison):
    k_fp4, v_fp4, k_sf, v_sf, gs = make_pool()
    view = m.QSAFP4KVView(k_fp4, v_fp4, k_sf, v_sf, gs, gs)
    k_ref = NVFP4KVQuantizeUtil.dequantize(k_fp4, k_sf, gs)
    v_ref = NVFP4KVQuantizeUtil.dequantize(v_fp4, v_sf, gs)
    be = FakeBackend()
    rtt = be.req_to_token_pool.req_to_token
    req_indices = torch.randperm(R, device=dev)[:batch].to(torch.int32)
    seq_lens = torch.randint(topk // 2, L, (batch,), dtype=torch.int32, device=dev)
    # indices: valid prefix (< seq_len), then some >= seq_len, then -1 tail
    indices = torch.full((batch, topk), -1, dtype=torch.int32, device=dev)
    valid_n = []
    for b in range(batch):
        n_valid = int(torch.randint(1, min(topk, int(seq_lens[b])), (1,)))
        n_oob = min(3, topk - n_valid)
        indices[b, :n_valid] = torch.randperm(int(seq_lens[b]), device=dev)[:n_valid]
        indices[b, n_valid : n_valid + n_oob] = int(seq_lens[b]) + 5
        valid_n.append(n_valid)
    pages_per_row = (topk + PAGE - 1) // PAGE
    stride = pages_per_row * PAGE
    cu_strided = torch.arange(batch + 1, dtype=torch.int32, device=dev) * stride
    if poison:
        # pre-fill every u8 scratch this call will use with 0xFF (e4m3 NaN)
        for dim in (D // 2, D // 16):
            a, b_ = be._get_fa2_scratch(batch * stride, H, dim, torch.uint8, dev)
            a.fill_(0xFF)
            b_.fill_(0xFF)
    pk, pv = m.compact_and_dequant(
        be, view, batch * stride, req_indices, indices, seq_lens, cu_strided,
        batch, topk, zero_fill_cols=stride, out_dtype=out_dtype,
    )
    assert pk.shape == (batch * stride, H, D) and pk.dtype == out_dtype, (pk.shape, pk.dtype)
    # CUDA-graph sizing: scratch capacity larger than batch*stride must give identical used rows
    pkg, pvg = m.compact_and_dequant(
        be, view, 3 * batch * stride, req_indices, indices, seq_lens, cu_strided,
        batch, topk, zero_fill_cols=stride, out_dtype=out_dtype,
    )
    assert pkg.shape[0] >= batch * stride
    assert torch.equal(pkg[: batch * stride], pk) and torch.equal(pvg[: batch * stride], pv), "graph-capacity rows differ"
    assert torch.isfinite(pk.float()).all() and torch.isfinite(pv.float()).all(), "non-finite in scratch"
    for b in range(batch):
        nv = valid_n[b]
        slots = rtt[req_indices[b], indices[b, :nv].long()].long()
        exp_k = k_ref[slots].to(out_dtype)
        exp_v = v_ref[slots].to(out_dtype)
        got_k = pk[b * stride : b * stride + nv]
        got_v = pv[b * stride : b * stride + nv]
        assert torch.equal(got_k, exp_k), f"K mismatch b={b}"
        assert torch.equal(got_v, exp_v), f"V mismatch b={b}"
        tail_k = pk[b * stride + nv : (b + 1) * stride].float()
        tail_v = pv[b * stride + nv : (b + 1) * stride].float()
        assert (tail_k == 0).all() and (tail_v == 0).all(), f"tail not zero b={b}"
    # compact (varlen) layout: cu_k = prefix sum of valid counts, no zero fill
    counts = torch.tensor(valid_n, dtype=torch.int32, device=dev)
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    cu_k[1:] = torch.cumsum(counts, 0)
    pk2, pv2 = m.compact_and_dequant(
        be, view, batch * topk, req_indices, indices, seq_lens, cu_k, batch, topk,
        out_dtype=out_dtype,
    )
    for b in range(batch):
        nv = valid_n[b]
        slots = rtt[req_indices[b], indices[b, :nv].long()].long()
        s = int(cu_k[b])
        assert torch.equal(pk2[s : s + nv], k_ref[slots].to(out_dtype)), f"compact K b={b}"
        assert torch.equal(pv2[s : s + nv], v_ref[slots].to(out_dtype)), f"compact V b={b}"
    # chunked-prefill history gather
    seq_list = [int(x) for x in seq_lens.tolist()]
    kc, vc = m.gather_history_fp4(view, rtt, req_indices.tolist(), seq_list)
    exp = torch.cat([k_ref[rtt[int(req_indices[b]), : seq_list[b]].long()] for b in range(batch)])
    assert torch.equal(kc, exp) and kc.dtype == torch.bfloat16, "history K"
    print(f"PASS topk={topk} batch={batch} out={out_dtype} poison={poison} valid={valid_n} stride={stride}")


def scale_source_equivalence():
    # Writing with a host float 1.0 vs the preloaded 1-elem GPU tensor must
    # produce identical packed bytes (what the _quantized_scales change relies on).
    x = torch.randn(512, H, D, dtype=torch.bfloat16, device=dev)
    a = NVFP4KVQuantizeUtil.quantize(x, 1.0)
    b = NVFP4KVQuantizeUtil.quantize(x, torch.ones(1, dtype=torch.float32, device=dev))
    assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
    assert torch.equal(a[1].view(torch.uint8), b[1].view(torch.uint8))
    print("PASS scale-source equivalence (host 1.0 == gpu ones)")


def method_guard():
    q = m.QSANVFP4KVCacheMethod(12, "cuda:0")
    assert q.layer_global_scale(q.k_scales_gpu, 47).numel() == 1, "guarded lookup must be 1-elem"
    assert q.dequant_workspace_dtype() is None
    try:
        q.dequantize_kv_tensor(torch.zeros(1, 1, 1, dtype=torch.uint8, device=dev), None, 0)
    except RuntimeError as e:
        assert "whole-pool" in str(e)
    else:
        raise AssertionError("plain read must fail fast")
    print("PASS method guards")


if __name__ == "__main__":
    method_guard()
    scale_source_equivalence()
    for out_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        run_case(2051, 3, out_dtype, poison=False)
        run_case(2051, 4, out_dtype, poison=True)
        run_case(128, 2, out_dtype, poison=True)
    print("ALL PASS")
