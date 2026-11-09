#!/usr/bin/env python3
"""Apply the NVFP4-KV x QSA patches to an SGLang srt tree (fb1216c6c fork).

Idempotent anchor/replace patcher (same shape as dspark's), so it can be
re-run after a reinstall and refuses to apply against a tree whose anchors
moved.  Usage:

    SGLANG_SRT=<.../sglang/srt> python3 apply_qsa_nvfp4_patches.py [--check]

--check only verifies every anchor matches exactly once (no writes).
"""
import ast
import os
import pathlib
import shutil
import sys

SRT = pathlib.Path(os.environ.get("SGLANG_SRT", "/tmp/sgl-intel/venv/lib/python3.12/site-packages/sglang/srt"))
HERE = pathlib.Path(__file__).resolve().parent
MARKER = "qsa_nvfp4_kv"
CHECK = "--check" in sys.argv


def patch(path: pathlib.Path, replacements):
    s = path.read_text()
    if MARKER in s:
        print(f"{path.relative_to(SRT)}: already patched")
        return
    for anchor, replacement in replacements:
        count = s.count(anchor)
        assert count == 1, f"{path}: anchor matched {count} times (want 1):\n{anchor[:300]}"
        s = s.replace(anchor, replacement, 1)
    ast.parse(s)
    if CHECK:
        print(f"{path.relative_to(SRT)}: anchors OK")
        return
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)
    path.write_text(s)
    print(f"{path.relative_to(SRT)}: patched (backup {orig.name})")


# --------------------------------------------------------------------------
# 1. New module
# --------------------------------------------------------------------------
MODULE_SRC = HERE / "qsa_nvfp4_kv.py"
MODULE_DST = SRT / "layers" / "attention" / "qsa_nvfp4_kv.py"

# --------------------------------------------------------------------------
# 2. qsa/sparse_attn.py: ZERO_FILL + int64 offsets + out-dtype cast in _compact_kv
#    (upstream sgl-project PR #38851 shape; zero_fill_cols=0 keeps the existing
#    callers byte-identical in behaviour)
# --------------------------------------------------------------------------
SPARSE = SRT / "layers" / "attention" / "qsa" / "sparse_attn.py"

KERNEL_SIG_ANCHOR = """    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)
"""
KERNEL_SIG_REPLACEMENT = """    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # qsa_nvfp4_kv: 64-bit element offsets (slot * heads * dim overflows int32
    # once the pool holds > 2^31 / (heads * dim) tokens; upstream PR #38851).
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole
        # pages, so every slot in [valid_count, pad_cols) must hold zeros, never
        # stale bytes (stale e4m3 scale bytes can be NaN). The store covers the
        # full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    out_dtype = out_k.dtype.element_ty
    tl.store(
        out_k + dst,
        tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )
    tl.store(
        out_v + dst,
        tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )
"""

WRAPPER_ANCHOR = """def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk
):
    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )
"""
WRAPPER_REPLACEMENT = """def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    \"\"\"Gather the selected KV rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the page-aligned strided layout: every row
    in [valid_count, zero_fill_cols) of each request is stored as zero
    (qsa_nvfp4_kv; upstream PR #38851). 0 = the compact varlen layout, in
    which only valid rows are written (unchanged behaviour).
    \"\"\"
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = int(zero_fill_cols) > 0
    grid_cols = max(topk, zero_fill_cols) if zero_fill else topk
    _compact_kv[(batch, heads, triton.cdiv(grid_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        zero_fill_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        num_warps=8,
    )
"""

# --------------------------------------------------------------------------
# 3. qwen_sparse_attn_backend.py: route the three KV-pool readers through the
#    fp4 view when the pool is NVFP4.
# --------------------------------------------------------------------------
BACKEND = SRT / "layers" / "attention" / "qwen_sparse_attn_backend.py"

IMPORT_ANCHOR = """from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
"""
IMPORT_REPLACEMENT = IMPORT_ANCHOR + (
    "from sglang.srt.layers.attention import qsa_nvfp4_kv  # NVFP4 KV on the QSA path\n"
)

# 3a. chunked-prefill history gather (forward_extend, prefix_lens != 0)
EXTEND_CHUNK_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = [
            k_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
"""
EXTEND_CHUNK_REPLACEMENT = """        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is not None:
            k_cat, v_cat = qsa_nvfp4_kv.gather_history_fp4(
                fp4_kv, req_to_token, req_indices, sequence_lens
            )
        else:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
            k_parts = [
                k_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            v_parts = [
                v_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            k_cat = torch.cat(k_parts)
            v_cat = torch.cat(v_parts)
"""
EXTEND_CK_ANCHOR = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
"""
EXTEND_CK_REPLACEMENT = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            k_cat,
            v_cat,
"""

# 3b. trtllm/XQA paged sparse decode
TRTLLM_HEAD_ANCHOR = """        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
"""
TRTLLM_HEAD_REPLACEMENT = """        fp4_kv = qsa_nvfp4_kv.try_fp4_view(self.token_to_kv_pool, layer.layer_id)
        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
"""
TRTLLM_EXTRACT_ANCHOR = """        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_strided,
            packed_k,
            packed_v,
            batch,
            topk,
        )
        num_kv_heads = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]
"""
TRTLLM_EXTRACT_REPLACEMENT = """        if fp4_kv is not None and qsa_nvfp4_kv.decode_mode() == "xqa4":
            # NVFP4 pool, native: gather packed bytes + block scales (zero-filled
            # to the page-aligned stride) and run XQA on NVFP4 KV directly.
            packed = qsa_nvfp4_kv.compact_fp4(
                self,
                fp4_kv,
                max(capacity_rows, batch) * stride,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_strided,
                batch,
                topk,
                zero_fill_cols=stride,
            )
            if self._trtllm_workspace is None:
                self._trtllm_workspace = torch.zeros(
                    128 * 1024 * 1024, dtype=torch.uint8, device=device
                )
            output = qsa_nvfp4_kv.xqa4_decode(
                trtllm_decode,
                q,
                fp4_kv,
                packed,
                page,
                block_tables,
                valid_counts,
                stride,
                layer.scaling,
                self._trtllm_workspace,
            )
            return output.reshape(q.shape[0], -1)
        if fp4_kv is not None:
            # NVFP4 pool (SGLANG_QSA_NVFP4_DECODE=dequant): gather packed bytes +
            # block scales, dequantize the compacted rows, and let the paged
            # decode consume the scratch as a bf16 (or fp8) cache.
            packed_k, packed_v = qsa_nvfp4_kv.compact_and_dequant(
                self,
                fp4_kv,
                max(capacity_rows, batch) * stride,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_strided,
                batch,
                topk,
                zero_fill_cols=stride,
            )
            num_kv_heads = fp4_kv.head_num
            head_dim = fp4_kv.head_dim
        else:
            packed_k, packed_v = self._get_fa2_scratch(
                max(capacity_rows, batch) * stride,
                k_buffer.shape[1],
                k_buffer.shape[2],
                k_buffer.dtype,
                k_buffer.device,
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_strided,
                packed_k,
                packed_v,
                batch,
                topk,
            )
            num_kv_heads = k_buffer.shape[1]
            head_dim = k_buffer.shape[2]
"""

# 3c. _forward_paged_attention head: do not touch get_key_buffer on an fp4 pool
PAGED_HEAD_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""
PAGED_HEAD_REPLACEMENT = """        pool = self.token_to_kv_pool
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is None:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
        else:
            # Both consumers below resolve the fp4 view themselves; a plain
            # get_key_buffer on an NVFP4 QSA pool is refused (whole-pool dequant).
            k_buffer = v_buffer = None
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            if fp4_kv is not None:
                k_buffer = pool.get_key_buffer(layer.layer_id)
                v_buffer = pool.get_value_buffer(layer.layer_id)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""

# 3d. varlen (FA) fallback in _forward_paged_attention
VARLEN_ANCHOR = """        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
        )
"""
VARLEN_REPLACEMENT = """        if fp4_kv is not None:
            packed_k, packed_v = qsa_nvfp4_kv.compact_and_dequant(
                self,
                fp4_kv,
                scratch_capacity,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                batch,
                topk,
                out_dtype=q.dtype,
            )
        else:
            packed_k, packed_v = self._get_fa2_scratch(
                scratch_capacity,
                k_buffer.shape[1],
                k_buffer.shape[2],
                k_buffer.dtype,
                k_buffer.device,
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                packed_k,
                packed_v,
                batch,
                topk,
            )
"""

# --------------------------------------------------------------------------
# 4. kv_cache_configurator.py: swap in the QSA method for QSA models only
# --------------------------------------------------------------------------
CONFIGURATOR = SRT / "mem_cache" / "kv_cache_configurator.py"
BUILD_FP4_ANCHOR = """        quant_method = get_kv_cache_quant_method(
            quant_name,
            num_layers=num_layers,
            device=self.device,
        )
        quant_method.load_scales_from_model(self.model)
        return quant_method
"""
BUILD_FP4_REPLACEMENT = """        if quant_name == "nvfp4" and is_qwen_qsa(self.model_config.hf_config):
            # qsa_nvfp4_kv: the QSA backend gathers sparse rows with Triton and
            # dequantizes on the way in; it never uses the FP8 dequant
            # workspace or native-FP4 decode the stock method declares.
            from sglang.srt.layers.attention.qsa_nvfp4_kv import (
                QSANVFP4KVCacheMethod,
            )

            quant_method = QSANVFP4KVCacheMethod(
                num_layers=num_layers, device=self.device
            )
        else:
            quant_method = get_kv_cache_quant_method(
                quant_name,
                num_layers=num_layers,
                device=self.device,
            )
        quant_method.load_scales_from_model(self.model)
        return quant_method
"""
CONFIGURATOR_IMPORT_ANCHOR = """from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
"""
CONFIGURATOR_IMPORT_REPLACEMENT = """from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
"""

# --------------------------------------------------------------------------
# 5. memory_pool.py: host-scalar write scales on nvfp4 -> per-layer GPU scales
#    (HybridLinearKVPool.set_kv_buffer defaults k_scale=v_scale=1.0, a Python
#    float; NVFP4KVQuantizeUtil.quantize then builds a device tensor from it
#    on every write, which is illegal inside CUDA-graph capture, and it also
#    bypasses the per-layer global scales the read side uses.)
# --------------------------------------------------------------------------
POOL = SRT / "mem_cache" / "memory_pool.py"
QUANT_SCALES_ANCHOR = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale
"""
QUANT_SCALES_REPLACEMENT = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        # qsa_nvfp4_kv: HybridLinearKVPool.set_kv_buffer passes host floats
        # (default 1.0) instead of None. On nvfp4 treat any non-device scale
        # as unset so the write uses the same preloaded per-layer GPU scale the
        # read side dequantizes with, and no tensor is allocated during
        # CUDA-graph capture. A guarded lookup never returns an empty slice.
        use_gpu = hasattr(self.quant_method, "k_scales_gpu")
        host_scalar = getattr(self.quant_method, "name", None) == "nvfp4" and not (
            torch.is_tensor(k_scale) and k_scale.is_cuda
        )
        if use_gpu and (k_scale is None or host_scalar):
            lookup = getattr(self.quant_method, "layer_global_scale", None)
            if lookup is not None:
                k_scale = lookup(self.quant_method.k_scales_gpu, global_layer_id)
                v_scale = lookup(self.quant_method.v_scales_gpu, global_layer_id)
            else:
                k_scale = self.quant_method.k_scales_gpu[
                    global_layer_id : global_layer_id + 1
                ]
                v_scale = self.quant_method.v_scales_gpu[
                    global_layer_id : global_layer_id + 1
                ]
        return k_scale, v_scale
"""

# --------------------------------------------------------------------------
# 6. server_args.py: the KV4/backend guard does not apply to QSA hybrids
#    (attention_registry builds QwenSparseAttnBackend for full-attention layers
#    regardless of --prefill/--decode-attention-backend; the guard only ever
#    forced an unrelated backend split).
# --------------------------------------------------------------------------
SERVER_ARGS = SRT / "server_args.py"
KV4_ANCHOR = """            if cfg.kv_cache_dtype == "nvfp4" and not (
                is_sm100_supported() or is_sm120_supported()
            ):
                raise RuntimeError(
                    "--kv-cache-dtype=nvfp4 requires Blackwell SM100 or SM120. "
                    "Use --kv-cache-dtype=fp4_mx_block16 for the block-size-16 FP4 recipe."
                )
"""
KV4_REPLACEMENT = KV4_ANCHOR + """            if cfg.kv_cache_dtype == "nvfp4":
                # qsa_nvfp4_kv: on a Qwen QSA hybrid the full-attention layers
                # always run QwenSparseAttnBackend (gather-dequant), so the
                # prefill/decode backend whitelist below is not what reads the
                # FP4 pool.
                try:
                    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

                    _is_qsa = is_qwen_qsa(self.get_model_config().hf_config)
                except Exception:
                    _is_qsa = False
                if _is_qsa:
                    logger.info(
                        "Attention: KV4 (nvfp4) on a Qwen QSA hybrid; full-attention "
                        "layers use the QSA gather-dequant path (qsa_nvfp4_kv)."
                    )
                    return
"""



PR_A_OLD='        k_buffer = pool.get_key_buffer(layer.layer_id)\n        v_buffer = pool.get_value_buffer(layer.layer_id)\n        if diagnostics is not None:\n            diagnostics.set_phase(layer.layer_id, "before_req_indices_to_list")\n'
PR_A_NEW='        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)\n        if fp4_kv is None:\n            k_buffer = pool.get_key_buffer(layer.layer_id)\n            v_buffer = pool.get_value_buffer(layer.layer_id)\n        if diagnostics is not None:\n            diagnostics.set_phase(layer.layer_id, "before_req_indices_to_list")\n'
PR_B_OLD='        output = sparse_gqa_fwd_interface_triton_ck(\n            q.contiguous(),\n            k_buffer.index_select(0, gather_index),\n            v_buffer.index_select(0, gather_index),\n'
PR_B_NEW='        if fp4_kv is not None:\n            k_cat, v_cat = qsa_nvfp4_kv.gather_history_fp4(\n                fp4_kv, req_to_token, req_indices, sequence_lens, gather_index\n            )\n        else:\n            k_cat = k_buffer.index_select(0, gather_index)\n            v_cat = v_buffer.index_select(0, gather_index)\n        output = sparse_gqa_fwd_interface_triton_ck(\n            q.contiguous(),\n            k_cat,\n            v_cat,\n'

def main():
    if not CHECK:
        if MODULE_DST.exists() and MODULE_DST.read_text() == MODULE_SRC.read_text():
            print(f"{MODULE_DST.relative_to(SRT)}: already installed")
        else:
            shutil.copy2(MODULE_SRC, MODULE_DST)
            print(f"{MODULE_DST.relative_to(SRT)}: installed")
    patch(SPARSE, [
        (KERNEL_SIG_ANCHOR, KERNEL_SIG_REPLACEMENT),
        (WRAPPER_ANCHOR, WRAPPER_REPLACEMENT),
    ])
    patch(BACKEND, [
        (IMPORT_ANCHOR, IMPORT_REPLACEMENT),
        (PR_A_OLD, PR_A_NEW),
        (PR_B_OLD, PR_B_NEW),
        (TRTLLM_HEAD_ANCHOR, TRTLLM_HEAD_REPLACEMENT),
        (TRTLLM_EXTRACT_ANCHOR, TRTLLM_EXTRACT_REPLACEMENT),
        (PAGED_HEAD_ANCHOR, PAGED_HEAD_REPLACEMENT),
        (VARLEN_ANCHOR, VARLEN_REPLACEMENT),
    ])
    patch(CONFIGURATOR, [
        (CONFIGURATOR_IMPORT_ANCHOR, CONFIGURATOR_IMPORT_REPLACEMENT),
        (BUILD_FP4_ANCHOR, BUILD_FP4_REPLACEMENT),
    ])
    patch(POOL, [(QUANT_SCALES_ANCHOR, QUANT_SCALES_REPLACEMENT)])
    patch(SERVER_ARGS, [(KV4_ANCHOR, KV4_REPLACEMENT)])
    print("ALL-OK" if CHECK else "ALL-PATCHED")


if __name__ == "__main__":
    sys.exit(main())
