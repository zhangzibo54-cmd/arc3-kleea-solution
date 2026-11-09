"""Per-layer KV precision for the QSA full-attention layers (ARC3_QSA_FP4_LAYERS).

The full-attention KV of a Qwen QSA hybrid lives in one MHATokenToKVPool inside
HybridLinearKVPool, one dtype for every layer. With ARC3_QSA_FP4_LAYERS set, that
pool is replaced by SplitFullKVPool: an fp8 sub-pool for the other full-attention
layers and an NVFP4 sub-pool (QSANVFP4KVCacheMethod) for the listed ones. Both
sub-pools have the same size and share the token slot indices, so req_to_token,
the radix tree, the scheduler and the MTP draft pool are untouched. No kernel
changes: the QSA backend (with the qsa_nvfp4_kv patch applied) already picks the
FP4 path per layer through qsa_nvfp4_kv.try_fp4_view, which this module makes
answer per layer instead of per pool.

ARC3_QSA_FP4_LAYERS: comma-separated global layer ids or ranges ("3,7,12-35"),
"all" for every full-attention layer; empty or unset = off (stock fp8 pool).
Ids that are not full-attention layers are ignored. The server keeps
--kv-cache-dtype fp8_e4m3.

Installed by appending a hook to sglang/srt/mem_cache/memory_pool.py that calls
install() (see arc3_kvmix_install.py).
"""
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

ENV = "ARC3_QSA_FP4_LAYERS"


def parse_layers(text: Optional[str], full_attention_layer_ids: list[int]) -> list[int]:
    """The full-attention layer ids `text` selects, in order; [] when off."""
    text = (text or "").strip().lower()
    if not text:
        return []
    if text == "all":
        return list(full_attention_layer_ids)
    chosen = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            chosen.update(range(lo, hi + 1))
        else:
            chosen.add(int(part))
    return [i for i in full_attention_layer_ids if i in chosen]


def fp4_bytes_saved_per_token(num_heads: int, head_dim: int, v_head_dim: int, num_fp4_layers: int) -> int:
    """Bytes per token an NVFP4 layer saves against fp8 (1 byte per element).

    NVFP4 stores half a byte per element plus one E4M3 scale byte per 16
    elements, for K and V; the QSA method allocates no dequant workspace."""
    elements = num_heads * (head_dim + v_head_dim)
    fp4 = elements // 2 + elements // 16
    return num_fp4_layers * (elements - fp4)


class SplitRouting:
    """Maps a local full-attention index to (sub-pool index, index inside it).

    Attributes:
        is_fp4: Per local index, whether the layer lives in the NVFP4 sub-pool.
        sub_index: Per local index, the layer's index inside its sub-pool.
    """

    def __init__(self, full_attention_layer_ids: list[int], fp4_layer_ids: list[int]):
        """Build the tables for the global ids of the full-attention layers."""
        fp4 = set(fp4_layer_ids)
        self.is_fp4 = [gid in fp4 for gid in full_attention_layer_ids]
        counts = [0, 0]
        self.sub_index = []
        for flag in self.is_fp4:
            self.sub_index.append(counts[flag])
            counts[flag] += 1
        self.num_fp8, self.num_fp4 = counts


def _build_split_class(KVCache):
    class SplitFullKVPool(KVCache):
        """Full-attention KV split into an fp8 and an NVFP4 sub-pool by layer."""

        def __init__(self, routing: SplitRouting, fp8_pool, fp4_pool):
            # no super().__init__: every buffer belongs to a sub-pool
            self.routing = routing
            self.fp8_pool, self.fp4_pool = fp8_pool, fp4_pool
            self._pools = (fp8_pool, fp4_pool)
            base = fp8_pool if fp8_pool is not None else fp4_pool
            for name in ("size", "page_size", "dtype", "store_dtype", "device", "head_num",
                         "head_dim", "v_head_dim", "start_layer", "kv_cache_layout"):
                setattr(self, name, getattr(base, name, None))
            self.layer_num = routing.num_fp8 + routing.num_fp4
            self.end_layer = self.layer_num - 1
            self.quant_method = base.quant_method
            self.layer_transfer_counter = None
            self.mem_usage = sum(p.mem_usage for p in self._pools if p is not None)

        def route(self, local_id: int):
            """(sub-pool, index inside it) of a local full-attention index."""
            return self._pools[self.routing.is_fp4[local_id]], self.routing.sub_index[local_id]

        # --- per-layer access ---------------------------------------------------
        def get_key_buffer(self, layer_id, *args):
            pool, sub = self.route(layer_id)
            return pool.get_key_buffer(sub, *args)

        def get_value_buffer(self, layer_id, *args):
            pool, sub = self.route(layer_id)
            return pool.get_value_buffer(sub, *args)

        def get_kv_buffer(self, layer_id):
            pool, sub = self.route(layer_id)
            return pool.get_kv_buffer(sub)

        def get_raw_kv_buffer(self, layer_id):
            pool, sub = self.route(layer_id)
            return pool.get_raw_kv_buffer(sub)

        def get_kv_scale_buffer(self, layer_id):
            pool, sub = self.route(layer_id)
            return pool.get_kv_scale_buffer(sub)

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None,
                          layer_id_override=None, **kwargs):
            local = layer_id_override if layer_id_override is not None else layer.layer_id
            pool, sub = self.route(local)
            if pool is self.fp4_pool:
                # NVFP4 writes use the layer's preloaded GPU global scale (qsa_nvfp4_kv)
                k_scale = v_scale = None
            return pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale,
                                      layer_id_override=sub, **kwargs)

        # --- whole-pool operations: every sub-pool ----------------------------------
        def _each(self):
            return [p for p in self._pools if p is not None]

        def move_kv_cache(self, tgt_loc, src_loc):
            for p in self._each():
                p.move_kv_cache(tgt_loc, src_loc)

        def get_kv_size_bytes(self):
            k = v = 0
            for p in self._each():
                pk, pv = p.get_kv_size_bytes()
                k, v = k + pk, v + pv
            return k, v

        def get_kv_buffer_shape(self):
            return self._each()[0].get_kv_buffer_shape()

        def get_contiguous_buf_infos(self):
            ptrs, lens, item = [], [], []
            for p in self._each():
                a, b, c = p.get_contiguous_buf_infos()
                ptrs += a
                lens += b
                item += c
            return ptrs, lens, item

        def register_layer_transfer_counter(self, counter):
            self.layer_transfer_counter = counter
            for p in self._each():
                p.register_layer_transfer_counter(counter)

        @property
        def post_capture_active(self) -> bool:
            return any(getattr(p, "post_capture_active", False) for p in self._each())

        @property
        def post_capture_backed_bytes(self) -> int:
            return sum(getattr(p, "post_capture_backed_bytes", 0) for p in self._each())

        def _finalize_backing_tokens(self, n):
            for p in self._each():
                p._finalize_backing_tokens(n)

        def maybe_get_custom_mem_pool(self):
            return self._each()[0].maybe_get_custom_mem_pool()

        def get_cpu_copy(self, indices, mamba_indices=None):
            return [p.get_cpu_copy(indices) for p in self._each()]

        def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
            for p, part in zip(self._each(), kv_cache_cpu):
                p.load_cpu_copy(part, indices)

    return SplitFullKVPool


def install_pool(memory_pool_module) -> None:
    """Patch HybridLinearKVPool to build a SplitFullKVPool when ARC3_QSA_FP4_LAYERS selects layers."""
    mp = memory_pool_module
    Hybrid = mp.HybridLinearKVPool
    if getattr(Hybrid, "_arc3_kvmix_installed", False):
        return
    SplitFullKVPool = _build_split_class(mp.KVCache)
    mp.SplitFullKVPool = SplitFullKVPool
    original_init = Hybrid.__init__

    def __init__(self, *args, **kwargs):
        import torch

        full_ids = list(kwargs.get("full_attention_layer_ids") or [])
        fp4_ids = parse_layers(os.environ.get(ENV), full_ids)
        # only the target model's multi-layer fp8 pool is split; the MTP draft pool
        # (one attention layer, its own --speculative-draft-kv-cache-dtype) stays as it is
        if (not fp4_ids or len(full_ids) < 2 or kwargs.get("use_mla")
                or kwargs.get("full_kv_pool") is not None or kwargs.get("dtype") != torch.float8_e4m3fn):
            return original_init(self, *args, **kwargs)
        from sglang.srt.layers.attention.qsa_nvfp4_kv import QSANVFP4KVCacheMethod

        routing = SplitRouting(full_ids, fp4_ids)
        common = dict(size=kwargs["size"], page_size=kwargs["page_size"], head_num=kwargs["head_num"],
                      head_dim=kwargs["head_dim"], device=kwargs["device"],
                      enable_memory_saver=kwargs.get("enable_memory_saver", False),
                      enable_kv_cache_copy=kwargs.get("enable_kv_cache_copy", False))
        if kwargs.get("post_capture_active"):
            common["post_capture_active"] = True
        fp8_pool = (mp.MHATokenToKVPool(dtype=kwargs["dtype"], layer_num=routing.num_fp8, **common)
                    if routing.num_fp8 else None)
        # global scales are indexed by global layer id; sized to cover all of them
        method = QSANVFP4KVCacheMethod(num_layers=max(full_ids) + 1, device=kwargs["device"])
        fp4_pool = mp.MHATokenToKVPool(dtype=torch.float4_e2m1fn_x2, layer_num=routing.num_fp4,
                                       quant_method=method, **common)
        kwargs["full_kv_pool"] = SplitFullKVPool(routing, fp8_pool, fp4_pool)
        log.warning("arc3 kvmix: %d full-attention layers fp8, %d NVFP4 %s",
                    routing.num_fp8, routing.num_fp4, fp4_ids)
        return original_init(self, *args, **kwargs)

    Hybrid.__init__ = __init__
    Hybrid._arc3_kvmix_installed = True


def install_view(qsa_nvfp4_kv_module) -> None:
    """Make qsa_nvfp4_kv.try_fp4_view answer per layer on a SplitFullKVPool."""
    q = qsa_nvfp4_kv_module
    if getattr(q, "_arc3_kvmix_installed", False):
        return
    original = q.try_fp4_view

    def try_fp4_view(pool, layer_id: int):
        full = getattr(pool, "full_kv_pool", pool)
        routing = getattr(full, "routing", None)
        if routing is None:
            return original(pool, layer_id)
        local = pool._transfer_full_attention_id(layer_id) if full is not pool else layer_id
        if not routing.is_fp4[local]:
            return None
        sub_pool, sub = full.route(local)
        k_fp4, v_fp4, k_sf, v_sf = sub_pool.get_raw_kv_buffer(sub)
        method = sub_pool.quant_method
        k_gs = method.layer_global_scale(method.k_scales_gpu, layer_id)
        v_gs = method.layer_global_scale(method.v_scales_gpu, layer_id)
        return q.QSAFP4KVView(k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs)

    q.try_fp4_view = try_fp4_view
    q._arc3_kvmix_installed = True


def draft_adjusted_saving(saved: int, num_layers: int, draft_layers: int) -> int:
    """The per-token saving to subtract before SGLang adds the EAGLE/MTP draft share.

    DefaultPoolConfigurator scales the cell size by (1 + draft_layers / num_layers)
    for the draft KV. The draft pool stays fp8, so its share must be computed from
    the fp8 size: subtracting saved * n / (n + d) leaves exactly that share. With the
    full saving the pool came out ~0.5 GB too large and the server ran out of
    activation memory under load (all25 x 20 streams, vast JP 2026-10-29).
    """
    if draft_layers <= 0 or num_layers <= 0:
        return saved
    return saved * num_layers // (num_layers + draft_layers)


def _draft_layers(kvc) -> int:
    spec = getattr(kvc, "spec_algorithm", None)
    if spec is None or not (spec.is_eagle() or spec.is_standalone()):
        return 0
    return int(getattr(getattr(kvc, "spec_aux_config", None), "eagle_draft_num_layers", 0) or 0)


def install_cell_size(pool_configurator_module) -> None:
    """Subtract what the NVFP4 layers save from the per-token KV size the pool is sized with."""
    cls = pool_configurator_module.DefaultPoolConfigurator  # hybrid linear models size with it
    if getattr(cls, "_arc3_kvmix_installed", False):
        return
    original = cls._compute_cell_size

    def _compute_cell_size(self, kvc, num_layers):
        cell = original(self, kvc, num_layers)
        mc = kvc.model_config
        # Same layer source as DefaultPoolConfigurator.__init__ (model_config has no
        # full_attention_layer_ids of its own).
        mambaish = pool_configurator_module.mambaish_config(mc)
        lo, hi = kvc.layer_info.start_layer, kvc.layer_info.end_layer
        full_ids = [i for i in (mambaish.full_attention_layer_ids if mambaish else []) if lo <= i < hi]
        fp4_ids = parse_layers(os.environ.get(ENV), full_ids)
        if fp4_ids and not kvc.use_mla_backend and not getattr(kvc, "is_draft_worker", False):
            get_parallel = pool_configurator_module.get_parallel
            heads = mc.get_num_kv_heads(get_parallel().attn_tp_size, get_parallel().attn_dcp_size)
            saved = fp4_bytes_saved_per_token(heads, mc.head_dim, mc.v_head_dim, len(fp4_ids))
            saved = draft_adjusted_saving(saved, num_layers, _draft_layers(kvc))
            log.warning("arc3 kvmix: cell size %d -> %d bytes/token", cell, cell - saved)
            cell -= saved
        return cell

    cls._compute_cell_size = _compute_cell_size
    cls._arc3_kvmix_installed = True


CALIB_ENV = "ARC3_KV_CALIB_DIR"
# log2 bins of the 16-element block amax: bin i holds [2**(lo+i), 2**(lo+i+1)); bin 0 also holds zeros
CALIB_LOG2_LO, CALIB_LOG2_HI = -40, 24


class _CalibRecorder:
    """Prefill K/V statistics of the QSA full-attention layers for NVFP4 calibration.

    Per (layer, "k"/"v"): a histogram of the 16-element block amax (log2 bins)
    and the running amax, over every prefill token; per prefill call the
    block-amax quantiles and the per-channel absmax (range stability across
    requests, fixed outlier channels); plus up to `max_chunks`
    (q, k, v) segments of `chunk_tokens` tokens per layer (the first sequence of
    every `every`-th prefill call), for offline attention-error checks. Files
    in `out_dir`: hist.pt (rewritten every 240 calls and at exit) and
    sample_L<layer>_<i>.pt.
    """

    def __init__(self, out_dir: str, max_chunks: int = 8, chunk_tokens: int = 1024, every: int = 8):
        import atexit
        from pathlib import Path

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_chunks, self.chunk_tokens, self.every = max_chunks, chunk_tokens, every
        self.hist, self.amax, self.calls, self.saved = {}, {}, {}, {}
        self.per_call, self.channels = {}, {}
        self.total_calls = 0
        atexit.register(self.dump)

    def observe(self, q, k, v, layer, forward_batch) -> None:
        import torch

        if k is None or v is None or not k.is_cuda or torch.cuda.is_current_stream_capturing():
            return
        mode = forward_batch.forward_mode
        if any(getattr(mode, name, lambda: False)() for name in ("is_target_verify", "is_draft_extend")):
            return
        lid, tokens = layer.layer_id, k.shape[0]
        if tokens == 0:
            return
        for name, x in (("k", k), ("v", v)):
            blocks = x.reshape(tokens, -1, 16).float().abs().amax(-1)
            bins = torch.floor(torch.log2(blocks.clamp_min(2.0 ** CALIB_LOG2_LO)))
            bins = (bins.clamp_(CALIB_LOG2_LO, CALIB_LOG2_HI) - CALIB_LOG2_LO).long().flatten()
            h = torch.bincount(bins, minlength=CALIB_LOG2_HI - CALIB_LOG2_LO + 1)
            key = (lid, name)
            self.hist[key] = self.hist[key] + h if key in self.hist else h
            m = blocks.max()
            self.amax[key] = torch.maximum(self.amax[key], m) if key in self.amax else m
            # range stability across prefill calls: block-amax quantiles and per-channel absmax per call
            flat = blocks.flatten()
            row = torch.stack([m, *torch.quantile(flat, flat.new_tensor([0.5, 0.99])),
                               x.float().abs().mean(), flat.new_tensor(float(tokens))])
            self.per_call.setdefault(key, []).append(row)
            self.channels.setdefault(key, []).append(x.reshape(tokens, -1).abs().amax(0).half())
        n = self.calls.get(lid, 0)
        self.calls[lid] = n + 1
        if n % self.every == 0 and self.saved.get(lid, 0) < self.max_chunks:
            self._save_sample(q, k, v, layer, forward_batch, tokens)
        self.total_calls += 1
        if self.total_calls % 240 == 0:
            self.dump()

    def _save_sample(self, q, k, v, layer, forward_batch, tokens: int) -> None:
        import torch

        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        seg = min(int(lens[0]) if lens is not None and len(lens) else tokens, self.chunk_tokens)
        if seg < 64:
            return
        i = self.saved.get(lid := layer.layer_id, 0)
        torch.save({
            "q": q[:seg].reshape(seg, layer.tp_q_head_num, -1).to(torch.bfloat16).cpu(),
            "k": k[:seg].reshape(seg, layer.tp_k_head_num, -1).to(torch.bfloat16).cpu(),
            "v": v[:seg].reshape(seg, layer.tp_v_head_num, -1).to(torch.bfloat16).cpu(),
            "scaling": float(layer.scaling),
        }, self.out_dir / f"sample_L{lid}_{i}.pt")
        self.saved[lid] = i + 1

    def dump(self) -> None:
        import torch

        if not self.hist:
            return
        torch.save({
            "log2_lo": CALIB_LOG2_LO,
            "hist": {f"{l}_{n}": h.cpu() for (l, n), h in self.hist.items()},
            "amax": {f"{l}_{n}": float(m) for (l, n), m in self.amax.items()},
            "calls": dict(self.calls),
            # per prefill call: [amax, block-amax p50, block-amax p99, mean |x|, tokens]
            "per_call": {f"{l}_{n}": torch.stack(r).cpu() for (l, n), r in self.per_call.items()},
            # per prefill call: absmax of every (head, channel), flattened
            "channels": {f"{l}_{n}": torch.stack(r).cpu() for (l, n), r in self.channels.items()},
        }, self.out_dir / "hist.pt")


def install_calib(backend_module) -> None:
    """Record prefill K/V statistics in QwenSparseAttnBackend.forward_extend when ARC3_KV_CALIB_DIR is set."""
    out = os.environ.get(CALIB_ENV, "").strip()
    cls = backend_module.QwenSparseAttnBackend
    if not out or getattr(cls, "_arc3_kvcalib_installed", False):
        return
    recorder = _CalibRecorder(out)
    original = cls.forward_extend

    def forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs):
        recorder.observe(q, k, v, layer, forward_batch)
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)

    cls.forward_extend = forward_extend
    cls._arc3_kvcalib_installed = True
    log.warning("arc3 kvcalib: recording prefill K/V statistics to %s", out)
