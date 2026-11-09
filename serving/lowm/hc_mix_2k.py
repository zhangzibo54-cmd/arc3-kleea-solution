"""HyperConnection mix in two launches for verify batches of 40-96 rows.

    out = hc_mix_2k(x_normed, w_down, w_up, hc, hs, cfg_down, cfg_up)

Same result as `GatedResidual._mix_compute` (server: torch.compile chain of
down GEMM + split-K reduce, silu, up GEMM, sigmoid-mul-mean above 16 rows):

1. down: t = silu(x @ W_down^T / hc), the split-K kernel of
   lowm_splitk_gemm.py with a silu epilogue (one launch, split-K fix-up in
   the kernel);
2. up: out_j = mean_g(sigmoid(t @ W_up[g * hs + j]^T) * x[g * hs + j]), one
   program per block of BJ hidden columns, loading the hc weight rows
   {g * hs + j} together so the mean over g happens in the epilogue.

Unlike Pennyroyal's persistent fused kernel there is no grid barrier: the
launch boundary orders the two phases, and each phase gets its own grid.
"""
import torch
import triton
import triton.language as tl

from lowm_splitk_gemm import lowm_splitk_linear


@triton.jit
def _hc_up_mix_kernel(
    t_ptr, w_up_ptr, x_ptr, out_ptr,
    M, HS, R, inv_hc,
    BM: tl.constexpr, HC: tl.constexpr, BJ: tl.constexpr, BR: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    offs_j = pid * BJ + tl.arange(0, BJ)
    offs_g = tl.arange(0, HC)
    offs_r = tl.arange(0, BR)
    mask_m = offs_m < M
    mask_j = offs_j < HS
    rows = tl.reshape(offs_g[:, None] * HS + offs_j[None, :], (HC * BJ,))
    mask_rows = tl.reshape(tl.broadcast_to(mask_j[None, :], (HC, BJ)), (HC * BJ,))
    acc = tl.zeros((HC * BJ, BM), dtype=tl.float32)
    for r0 in range(0, R, BR):
        r = r0 + offs_r
        mask_r = r < R
        w = tl.load(w_up_ptr + rows[:, None] * R + r[None, :], mask=mask_rows[:, None] & mask_r[None, :],
                    other=0.0, eviction_policy="evict_first")
        t = tl.load(t_ptr + offs_m[:, None] * R + r[None, :], mask=mask_m[:, None] & mask_r[None, :], other=0.0)
        acc = tl.dot(w, tl.trans(t), acc)
    # the server rounds the up GEMM output to bf16 before the sigmoid
    gate = tl.sigmoid(tl.reshape(acc.to(x_ptr.dtype.element_ty).to(tl.float32), (HC, BJ, BM)))
    xg = tl.load(x_ptr + offs_m[None, None, :] * (HC * HS) + offs_g[:, None, None] * HS + offs_j[None, :, None],
                 mask=mask_j[None, :, None] & mask_m[None, None, :], other=0.0).to(tl.float32)
    out = tl.sum(gate * xg, axis=0) * inv_hc
    tl.store(out_ptr + offs_m[None, :] * HS + offs_j[:, None], out.to(out_ptr.dtype.element_ty),
             mask=mask_j[:, None] & mask_m[None, :])


def hc_up_mix(t: torch.Tensor, w_up: torch.Tensor, x: torch.Tensor, hc: int, hs: int, cfg: dict) -> torch.Tensor:
    """Up GEMM with the sigmoid-mul-mean epilogue; t [M, R], w_up [hc * hs, R], x [M, hc * hs]."""
    m, r = t.shape
    out = torch.empty(m, hs, dtype=x.dtype, device=x.device)
    bm = max(16, triton.next_power_of_2(m))
    _hc_up_mix_kernel[(triton.cdiv(hs, cfg["BJ"]),)](
        t, w_up, x, out, m, hs, r, 1.0 / hc,
        BM=bm, HC=hc, BJ=cfg["BJ"], BR=cfg["BR"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out


def hc_mix_2k(
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    cfg_down: dict,
    cfg_up: dict,
) -> torch.Tensor:
    """Return the HC mixed input [M, hs] for the normed hyper input x [M, hc * hs].

    Args:
        x: Normed hyper input, [M, hc * hs], M <= 128.
        w_down: [lowrank, hc * hs].
        w_up: [hc * hs, lowrank].
        hc: Number of residual streams.
        hs: Hidden size.
        cfg_down: Split-K tile config of the down GEMM (lowm_splitk_linear).
        cfg_up: BJ, BR, num_warps, num_stages of the up kernel.
    """
    t = lowm_splitk_linear(x, w_down, cfg_down, silu_scale=1.0 / hc)
    return hc_up_mix(t, w_up, x, hc, hs, cfg_up)
