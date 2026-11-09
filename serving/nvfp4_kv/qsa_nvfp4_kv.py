"""NVFP4 KV cache on the QSA (Qwen4-Exp sparse attention) path.

Why this exists
---------------
SGLang's stock ``nvfp4`` KV recipe (``NVFP4KVCacheMethod``) declares two
consumers that the QSA backend never uses: FlashInfer prefill reading an FP8
*dequant workspace* covering the whole pool, and TRT-LLM MHA decode consuming
native packed FP4 + block scales.  ``QwenSparseAttnBackend`` instead gathers
a sparse top-k subset of KV rows with a Triton kernel (``_compact_kv``) and
hands the packed rows to FlashInfer's paged decode (XQA on SM120).  With a
packed-FP4 pool that gather receives ``torch.float4_e2m1fn_x2`` tensors and
Triton 3.7.1 (and Triton main as of 2026-09-21) has no entry for that dtype in
``triton/_utils.py::type_canonicalisation_dict`` -> ``KeyError:
'float4_e2m1fn_x2'`` on the first forward pass.

What it does
------------
* ``QSANVFP4KVCacheMethod`` -- the NVFP4 method with PLAIN-only attention
  access rules, so the pool allocates packed FP4 + per-block FP8 scales and no
  full-size FP8 dequant workspace (the workspace would cost 2 B/elem, i.e.
  more than the fp4 data + scales it shadows).  Quantization, storage layout,
  per-layer global scales and slot moves are inherited unchanged.
  Plain whole-pool reads (``get_key_buffer``) are refused loudly instead of
  silently dequantizing ~1M rows per call.
* ``try_fp4_view`` / ``compact_and_dequant`` / ``gather_history_fp4`` -- the
  gather-dequant helpers the patched backend calls.  The stock compaction
  kernel is a pure gather/copy (no arithmetic on the values), so it runs
  unchanged over the packed FP4 bytes viewed as ``uint8`` (dim/2 bytes per
  row) and a second time over the per-block scale bytes (dim/16 per row); the
  compacted rows are then dequantized with FlashInfer's ``nvfp4_kv_dequantize``
  (the same kernel SGLang's own FP4 prefill path uses), giving a bf16 scratch
  that FlashInfer's paged decode consumes exactly like a bf16 pool.

Lineage: the approach (two uint8 gather passes + nvfp4_kv_dequantize, PLAIN
access rules, host-scalar write-scale fix) is the "dspark" patch from
Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks (MiaAI Lab, MIT) as ported to SM120
by redashes1984/sglang-flashnext-nvfp4kv-sm120.  This rewrite adapts it to the
fb1216c6c fork, adds zero-filled strided scratch (upstream sgl-project PR
#38851's finding: stale bytes past ``valid_count`` in the page-aligned scratch
reach the paged kernel, and stale e4m3 bytes can be NaN), guards the global
scale lookup, gates the method swap on QSA models only, and makes plain
whole-pool reads fail fast.

All of this is inert unless ``--kv-cache-dtype nvfp4`` on a Qwen QSA model.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccess,
    KVCacheAttentionAccessKind,
    KVCacheAttentionPhase,
    KVCacheBackendMatcher,
    NVFP4KVCacheMethod,
)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

_BF16 = torch.bfloat16
_U8 = torch.uint8
_FP8 = torch.float8_e4m3fn

# Scratch dtype handed to the paged attention kernel after dequant.
#   bf16 (default): exact dequant of the stored fp4 x block-scale x global-scale
#                   value; needs the XQA kv_cache_bf16 JIT variant.
#   fp8:            one extra e4m3 rounding on top of the fp4 quantization, but
#                   reuses the XQA kv_cache_e4m3 variant the fp8 production cell
#                   already compiled (same semantics as SGLang's own
#                   NVFP4KVCacheMethod.dequantize_prev_kv, which also lands in fp8).
_SCRATCH_DTYPE_ENV = "SGLANG_QSA_NVFP4_SCRATCH_DTYPE"


def scratch_dtype() -> torch.dtype:
    v = os.environ.get(_SCRATCH_DTYPE_ENV, "bf16").strip().lower()
    if v in ("bf16", "bfloat16"):
        return _BF16
    if v in ("fp8", "fp8_e4m3", "e4m3", "float8_e4m3fn"):
        return _FP8
    raise ValueError(f"{_SCRATCH_DTYPE_ENV} must be bf16 or fp8, got {v!r}")


def _plain_access(phase: KVCacheAttentionPhase) -> KVCacheAttentionAccess:
    return KVCacheAttentionAccess(
        phase,
        KVCacheAttentionAccessKind.PLAIN,
        KVCacheBackendMatcher(any_backend=True),
        storage_dtype=_U8,
        attention_kv_dtype=_BF16,
        scale_recipe="nvfp4",
    )


class QSANVFP4KVCacheMethod(NVFP4KVCacheMethod):
    """NVFP4 KV cache as consumed by the QSA backend (gather-dequant reads).

    Differs from ``NVFP4KVCacheMethod`` only in its declared attention
    accesses: PLAIN for both phases and every backend.  Consequently
    ``dequant_workspace_dtype()`` is None, so the pool allocates no FP8
    dequant workspace, and ``compute_cell_size`` excludes it automatically.
    """

    name = "nvfp4"  # keep the registry name: pool/backends key on it

    def __init__(self, num_layers: int, device: str):
        super().__init__(num_layers, device)
        # Pre-allocated so the out-of-range fallback never allocates during
        # CUDA-graph capture.
        self._ones_scale = torch.ones(1, dtype=torch.float32, device=device)

    def attention_accesses(self) -> tuple[KVCacheAttentionAccess, ...]:
        return (
            _plain_access(KVCacheAttentionPhase.PREFILL),
            _plain_access(KVCacheAttentionPhase.DECODE),
        )

    def layer_global_scale(
        self, scales_gpu: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        """1-element fp32 device tensor; never an empty slice.

        ``k_scales_gpu`` is pre-sized to the number of *effective* layers but
        indexed by the *global* layer id; ``load_scales_from_model`` resizes it
        when it finds the attention modules.  If it did not (unknown module
        layout), an unguarded slice past the end is an EMPTY tensor, which
        ``nvfp4_kv_quantize`` would accept silently.  Fall back to 1.0, which
        is also what the write side stores when the checkpoint carries no
        k_scale/v_scale.
        """
        if 0 <= layer_id < scales_gpu.numel():
            return scales_gpu[layer_id : layer_id + 1]
        return self._ones_scale

    def dequantize_kv_tensor(
        self,
        fp4_tensor: torch.Tensor,
        scales: torch.Tensor,
        layer_id: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # PLAIN access makes MHATokenToKVPool._get_key_buffer route here.  A
        # whole-pool dequant (~1M rows x 256 x bf16 = ~0.5 GB per layer per
        # call) would "work" and be misread as "fp4 is slow".  Every QSA call
        # site is patched to use try_fp4_view instead; anything else reaching
        # this is a bug we want to see.
        raise RuntimeError(
            "qsa_nvfp4_kv: plain whole-pool KV read requested on an NVFP4 QSA "
            f"pool (layer {layer_id}); use try_fp4_view()/compact_and_dequant() "
            "on the sparse gather path instead."
        )


class QSAFP4KVView:
    """Packed FP4 + per-block-scale buffers of one QSA full-attention layer."""

    __slots__ = ("k_fp4", "v_fp4", "k_sf", "v_sf", "k_gs", "v_gs", "head_num", "head_dim", "device")

    def __init__(self, k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs):
        self.k_fp4 = k_fp4.view(_U8)  # [rows, head_num, head_dim // 2]
        self.v_fp4 = v_fp4.view(_U8)
        self.k_sf = k_sf.view(_U8)  # [rows, head_num, head_dim // 16] (e4m3 bytes)
        self.v_sf = v_sf.view(_U8)
        self.k_gs = k_gs  # 1-element fp32 global scale, on device
        self.v_gs = v_gs
        self.head_num = self.k_fp4.shape[1]
        self.head_dim = self.k_fp4.shape[2] * 2
        self.device = self.k_fp4.device

    def dequant_rows(self, k_rows, k_sf_rows, v_rows, v_sf_rows, out_dtype):
        k = NVFP4KVQuantizeUtil.dequantize(k_rows, k_sf_rows.view(_FP8), self.k_gs)
        v = NVFP4KVQuantizeUtil.dequantize(v_rows, v_sf_rows.view(_FP8), self.v_gs)
        if out_dtype is not _BF16:
            k = k.to(out_dtype)
            v = v.to(out_dtype)
        return k, v


def try_fp4_view(pool, layer_id: int) -> Optional[QSAFP4KVView]:
    """Return the layer's FP4 buffers, or None for a non-NVFP4 pool."""
    full_pool = getattr(pool, "full_kv_pool", pool)
    quant_method = getattr(full_pool, "quant_method", None)
    if not isinstance(quant_method, QSANVFP4KVCacheMethod):
        return None
    k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer_id)
    k_gs = quant_method.layer_global_scale(quant_method.k_scales_gpu, layer_id)
    v_gs = quant_method.layer_global_scale(quant_method.v_scales_gpu, layer_id)
    return QSAFP4KVView(k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs)


_DECODE_ENV = "SGLANG_QSA_NVFP4_DECODE"


def decode_mode() -> str:
    """Decode/verify path for an NVFP4 pool on the paged (XQA) backend.

    ``xqa4`` (default): compact the packed FP4 bytes + block scales and run
    FlashInfer XQA with native NVFP4 KV (``kv_cache_sf``; SM120).  Kernel
    bench: 0.066 ms vs 0.497 ms (dequant) vs 0.113 ms (fp8) per layer at
    bs10 x 4 verify rows.  ``dequant``: the previous compact + dequant to a
    bf16/fp8 scratch + XQA on that scratch.
    """
    v = os.environ.get(_DECODE_ENV, "xqa4").strip().lower()
    if v not in ("xqa4", "dequant"):
        raise ValueError(f"{_DECODE_ENV} must be xqa4 or dequant, got {v!r}")
    return v


def compact_fp4(
    backend,
    fp4_kv: QSAFP4KVView,
    scratch_capacity: int,
    req_indices,
    topk_indices,
    sequence_lens,
    cu_seqlens_k,
    batch: int,
    topk: int,
    zero_fill_cols: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse gather of packed FP4 rows and their block scales (no dequant).

    Runs the stock compaction kernel twice -- once over the packed FP4 bytes,
    once over the per-block scale bytes.  Kernel-only, so CUDA-graph
    capture/replay is safe.

    ``zero_fill_cols`` > 0 selects the page-aligned strided layout used by the
    paged-decode path: slots in [valid_count, stride) are stored as zero in
    both passes (a zero scale byte decodes to 0, never a stale e4m3 NaN).
    0 keeps the compact varlen layout untouched.

    Returns:
        (k_fp4, v_fp4, k_sf, v_sf) uint8 scratch views covering the rows the
        consumer reads: [rows, heads, head_dim / 2] and [rows, heads, head_dim / 16].
    """
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton,
    )

    req_to_token = backend.req_to_token_pool.req_to_token
    n, d = fp4_kv.head_num, fp4_kv.head_dim
    # The scratch cache is keyed by (heads, dim, dtype, device): the FP4 data
    # (dim/2, u8) and scale (dim/16, u8) buffers never collide with each other
    # or with the bf16/fp8 buffers of another cell.
    pk_fp4, pv_fp4 = backend._get_fa2_scratch(scratch_capacity, n, d // 2, _U8, fp4_kv.device)
    pk_sf, pv_sf = backend._get_fa2_scratch(scratch_capacity, n, d // 16, _U8, fp4_kv.device)
    common = (req_to_token, req_indices, topk_indices, sequence_lens, cu_seqlens_k)
    qwen_sparse_kv_extraction_compact_triton(
        fp4_kv.k_fp4, fp4_kv.v_fp4, *common, pk_fp4, pv_fp4, batch, topk,
        zero_fill_cols=zero_fill_cols,
    )
    qwen_sparse_kv_extraction_compact_triton(
        fp4_kv.k_sf, fp4_kv.v_sf, *common, pk_sf, pv_sf, batch, topk,
        zero_fill_cols=zero_fill_cols,
    )
    # Only the first batch*stride (paged) / batch*topk (varlen) rows are read;
    # the scratch itself is sized for the CUDA-graph maximum.  `batch` is static
    # per captured graph, so the slice is graph-safe.
    used = batch * zero_fill_cols if zero_fill_cols > 0 else min(scratch_capacity, batch * topk)
    return pk_fp4[:used], pv_fp4[:used], pk_sf[:used], pv_sf[:used]


def xqa4_decode(trtllm_decode, q, fp4_kv, packed, page, block_tables, valid_counts,
                stride, softmax_scale, workspace):
    """FlashInfer XQA paged decode straight on compacted NVFP4 rows.

    ``packed`` is ``compact_fp4``'s output in the page-aligned strided layout.
    The per-layer global scales stay device tensors (graph-safe): logits use
    ``softmax_scale * k_global``, the output ``v_global``.
    """
    k_fp4, v_fp4, k_sf, v_sf = packed
    n, d = fp4_kv.head_num, fp4_kv.head_dim

    def hnd(x, width):
        return x.view(-1, page, n, width).permute(0, 2, 1, 3)

    return trtllm_decode(
        query=q.contiguous(),
        kv_cache=(hnd(k_fp4, d // 2), hnd(v_fp4, d // 2)),
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=valid_counts,
        max_seq_len=stride,
        bmm1_scale=fp4_kv.k_gs * softmax_scale,
        bmm2_scale=fp4_kv.v_gs,
        kv_cache_sf=(hnd(k_sf.view(_FP8), d // 16), hnd(v_sf.view(_FP8), d // 16)),
    )


def compact_and_dequant(
    backend,
    fp4_kv: QSAFP4KVView,
    scratch_capacity: int,
    req_indices,
    topk_indices,
    sequence_lens,
    cu_seqlens_k,
    batch: int,
    topk: int,
    zero_fill_cols: int = 0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse gather + dequant for the QSA decode/verify paths.

    ``compact_fp4`` followed by a dequant of the compacted rows to a bf16 (or
    fp8) scratch that a non-FP4 kernel can consume.  Used by the varlen
    fallback and by ``SGLANG_QSA_NVFP4_DECODE=dequant``.
    """
    if out_dtype is None:
        out_dtype = scratch_dtype()
    k_fp4, v_fp4, k_sf, v_sf = compact_fp4(
        backend, fp4_kv, scratch_capacity, req_indices, topk_indices, sequence_lens,
        cu_seqlens_k, batch, topk, zero_fill_cols=zero_fill_cols,
    )
    return fp4_kv.dequant_rows(k_fp4, k_sf, v_fp4, v_sf, out_dtype)


def gather_history_fp4(
    fp4_kv: QSAFP4KVView, req_to_token, req_indices, sequence_lens, gather_index=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked-prefill history gather for an NVFP4 pool.

    Pennyroyal d00d88e joins the per-request token INDICES first and gathers once
    (one index_select, no torch.cat of gathered K/V).  Same here for the packed FP4
    bytes and the E4M3 block scales: one gather each over the joined index, then a
    single dequant to bf16 (what ``sparse_gqa_fwd_interface_triton_ck`` consumes).
    Peak transient memory: packed rows (~0.28x bf16) + one bf16 copy, instead of
    per-request bf16 parts + a second bf16 copy from torch.cat.
    Output order equals the per-request concatenation, so results are identical.
    """
    if gather_index is None:
        parts = [req_to_token[req_indices[i], :seq_len].long() for i, seq_len in enumerate(sequence_lens)]
        gather_index = parts[0] if len(parts) == 1 else torch.cat(parts)
    k = NVFP4KVQuantizeUtil.dequantize(
        fp4_kv.k_fp4.index_select(0, gather_index),
        fp4_kv.k_sf.index_select(0, gather_index).view(_FP8), fp4_kv.k_gs,
    )
    v = NVFP4KVQuantizeUtil.dequantize(
        fp4_kv.v_fp4.index_select(0, gather_index),
        fp4_kv.v_sf.index_select(0, gather_index).view(_FP8), fp4_kv.v_gs,
    )
    return k, v


__all__ = [
    "QSANVFP4KVCacheMethod",
    "QSAFP4KVView",
    "compact_and_dequant",
    "compact_fp4",
    "decode_mode",
    "xqa4_decode",
    "gather_history_fp4",
    "scratch_dtype",
    "try_fp4_view",
]
