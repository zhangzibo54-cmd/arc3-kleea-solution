"""Split-K BF16 GEMM for decode-sized M (verify batches of 40-96 rows) on SM120.

    y = lowm_splitk_linear(x, w, cfg)    # x [M, K] bf16, w [N, K] bf16 -> y [M, N] bf16

The weight read dominates at small M, so the kernel is laid out to stream W from
DRAM with every SM busy:

- swap AB: each program computes acc[BN, BM] = W_tile[BN, BK] @ X_tile[BM, BK]^T,
  so the token dimension M is the MMA's narrow side;
- a grid of (N / BN) x SPLIT programs, each reading one N-tile over a K-range;
- SPLIT > 1 adds the partial sums with fp32 atomics into a workspace that lives
  in L2; the last program to finish an N-tile (per-tile counter, acq_rel)
  converts that tile to bf16 and zeroes the workspace and the counter again, so
  one launch does the whole GEMM and the buffers are clean for the next call
  (CUDA-graph safe). Atomic order makes the fp32 sum order nondeterministic.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _lowm_splitk_kernel(
    x_ptr, w_ptr, y_ptr, ws_ptr, cnt_ptr,
    M, N, K, k_per_split, split, scale,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr, SILU: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    mask_n = offs_n < N
    mask_m = offs_m < M
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    w_ptrs = w_ptr + offs_n[:, None] * K + (k_start + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * K + (k_start + offs_k)[None, :]
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    # K % BK == 0 (checked by the wrapper), so only N and M need masks.
    for _ in range(k_start, k_end, BK):
        w = tl.load(w_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_first")
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
        w_ptrs += BK
        x_ptrs += BK
    out_off = offs_m[None, :] * N + offs_n[:, None]
    out_mask = mask_m[None, :] & mask_n[:, None]
    if ATOMIC:
        tl.atomic_add(ws_ptr + out_off, acc, mask=out_mask, sem="relaxed")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if done == split - 1:
            total = tl.load(ws_ptr + out_off, mask=out_mask, other=0.0, cache_modifier=".cg")
            if SILU:
                total = total.to(y_ptr.dtype.element_ty).to(tl.float32) * scale
                total = total * tl.sigmoid(total)
            tl.store(y_ptr + out_off, total.to(y_ptr.dtype.element_ty), mask=out_mask)
            tl.store(ws_ptr + out_off, tl.zeros((BN, BM), dtype=tl.float32), mask=out_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)
    else:
        if SILU:
            acc = acc.to(y_ptr.dtype.element_ty).to(tl.float32) * scale
            acc = acc * tl.sigmoid(acc)
        tl.store(y_ptr + out_off, acc.to(y_ptr.dtype.element_ty), mask=out_mask)


_WORKSPACE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(m: int, n: int, n_tiles: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed fp32 partial-sum buffer and per-tile counters, kept per shape."""
    key = (m, n, n_tiles, str(device))
    if key not in _WORKSPACE:
        _WORKSPACE[key] = (torch.zeros(m * n, dtype=torch.float32, device=device),
                           torch.zeros(n_tiles, dtype=torch.int32, device=device))
    return _WORKSPACE[key]


def lowm_splitk_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    cfg: dict,
    silu_scale: float | None = None,
) -> torch.Tensor:
    """Compute x @ w.T with one launch, optionally followed by silu(y * silu_scale).

    Args:
        x: Activations [M, K], bf16, contiguous.
        w: Weight [N, K], bf16, contiguous (nn.Linear layout).
        cfg: Tile config with keys BN, BK, SPLIT, num_warps, num_stages. SPLIT
            is rounded so that every split covers a whole number of BK steps.
        silu_scale: If set, the output is silu(bf16(x @ w.T) * silu_scale),
            the HC mix down step (`F.silu(F.linear(x, w_down) / hc)`).

    Returns:
        y [M, N] in x's dtype.
    """
    m, k = x.shape
    n = w.shape[0]
    bn, bk = cfg["BN"], cfg["BK"]
    assert k % bk == 0, (k, bk)
    steps = k // bk
    steps_per_split = triton.cdiv(steps, min(cfg["SPLIT"], steps))
    split = triton.cdiv(steps, steps_per_split)
    n_tiles = triton.cdiv(n, bn)
    bm = max(16, triton.next_power_of_2(m))
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    ws, cnt = _workspace(m, n, n_tiles, x.device) if split > 1 else (y, y)
    _lowm_splitk_kernel[(n_tiles, split)](
        x, w, y, ws, cnt, m, n, k, steps_per_split * bk, split,
        1.0 if silu_scale is None else silu_scale,
        BM=bm, BN=bn, BK=bk, ATOMIC=split > 1, SILU=silu_scale is not None,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return y
