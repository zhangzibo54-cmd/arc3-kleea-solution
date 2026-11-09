"""HyperConnection combine fused with the next block's hc_norm (grouped Gemma RMSNorm).

    r_new, r_new_normed = hc_combine_norm(block_out, r, r_normed, inject_w, norm_w, hc, hs, eps, cfg)

In every Qwen4Exp layer, `attn_hyper_connection.combine` writes the new
4-stream residual and `mlp_hyper_connection.mix` immediately reads it again for
`hc_norm` (Gemma RMSNorm per 2560-wide group); the same happens between the MLP
combine and the next layer's attention mix when no PLE add sits in between.
The fused kernel writes the new residual and its normed copy in one pass:

    a[m, c]      = 2 * sigmoid(dot(r_normed[m], inject_w[c]) / hc)        (fp32)
    r_new[m, c]  = bf16(r[m, c] + a[m, c] * block_out[m])                 (group c, hs wide)
    normed[m, c] = bf16(r_new * rsqrt(mean(r_new^2) + eps) * (1 + norm_w[c]))

One program per (row, group): it needs only its own gate a[m, c] and its own
group's sum of squares, so no cross-program reduction is needed. With NORM=False
the same kernel is a plain combine (the server's two-kernel order, for timing).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _hc_combine_norm_kernel(
    blk_ptr, r_ptr, rn_ptr, inj_ptr, nw_ptr, out_r_ptr, out_n_ptr,
    HS, inv_hc, eps,
    HC: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, NORM: tl.constexpr,
):
    m = tl.program_id(0)
    c = tl.program_id(1)
    width = HC * HS
    offs_d = tl.arange(0, BLOCK_D)
    dot = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for d0 in range(0, width, BLOCK_D):
        d = d0 + offs_d
        mask = d < width
        rn = tl.load(rn_ptr + m * width + d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(inj_ptr + c * width + d, mask=mask, other=0.0).to(tl.float32)
        dot += rn * w
    a = 2.0 * tl.sigmoid(tl.sum(dot) * inv_hc)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < HS
    base = m * width + c * HS
    r = tl.load(r_ptr + base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    blk = tl.load(blk_ptr + m * HS + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    r_new = (r + a * blk).to(out_r_ptr.dtype.element_ty)
    tl.store(out_r_ptr + base + offs_h, r_new, mask=mask_h)
    if NORM:
        x = r_new.to(tl.float32)
        rs = tl.rsqrt(tl.sum(x * x) / HS + eps)
        nw = tl.load(nw_ptr + c * HS + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        tl.store(out_n_ptr + base + offs_h, (x * rs * (1.0 + nw)).to(out_n_ptr.dtype.element_ty), mask=mask_h)


@triton.jit
def _group_gemma_norm_kernel(x_ptr, nw_ptr, out_ptr, HS, eps, HC: tl.constexpr, BLOCK_H: tl.constexpr):
    m = tl.program_id(0)
    c = tl.program_id(1)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < HS
    base = m * HC * HS + c * HS
    x = tl.load(x_ptr + base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    rs = tl.rsqrt(tl.sum(x * x) / HS + eps)
    nw = tl.load(nw_ptr + c * HS + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offs_h, (x * rs * (1.0 + nw)).to(out_ptr.dtype.element_ty), mask=mask_h)


def hc_combine_norm(
    block_out: torch.Tensor,
    r: torch.Tensor,
    r_normed: torch.Tensor,
    inject_w: torch.Tensor,
    norm_w: torch.Tensor | None,
    hc: int,
    hs: int,
    eps: float,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combine, and (if norm_w is given) the next hc_norm, in one launch.

    Args:
        block_out: Block output [M, hs].
        r: Residual before the block [M, hc * hs].
        r_normed: hc_norm(r) as returned by the block's mix [M, hc * hs].
        inject_w: Inject weight [hc, hc * hs].
        norm_w: Next block's hc_norm weight [hc * hs]; None for combine only.
        hc: Number of residual streams.
        hs: Hidden size (group size of the norm).
        eps: RMSNorm epsilon.
        cfg: BLOCK_D (dot chunk) and num_warps.

    Returns:
        (new residual, its normed copy or None).
    """
    m = block_out.shape[0]
    out_r = torch.empty_like(r)
    out_n = torch.empty_like(r) if norm_w is not None else None
    _hc_combine_norm_kernel[(m, hc)](
        block_out, r, r_normed, inject_w, norm_w if norm_w is not None else inject_w, out_r,
        out_n if out_n is not None else out_r, hs, 1.0 / hc, eps,
        HC=hc, BLOCK_H=triton.next_power_of_2(hs), BLOCK_D=cfg["BLOCK_D"], NORM=norm_w is not None,
        num_warps=cfg["num_warps"])
    return out_r, out_n


def group_gemma_norm(x: torch.Tensor, norm_w: torch.Tensor, hc: int, hs: int, eps: float, num_warps: int) -> torch.Tensor:
    """The server's hc_norm (GroupedGemmaRMSNorm, group = hs) as a separate launch."""
    out = torch.empty_like(x)
    _group_gemma_norm_kernel[(x.shape[0], hc)](x, norm_w, out, hs, eps, HC=hc,
                                                BLOCK_H=triton.next_power_of_2(hs), num_warps=num_warps)
    return out


def reference(block_out, r, r_normed, inject_w, norm_w, hc, hs, eps):
    """fp32 reference: GatedResidual._combine_compute, then GroupedGemmaRMSNorm."""
    a = 2 * torch.sigmoid(r_normed.float() @ inject_w.float().T / hc)
    r_new = (r.float().unflatten(-1, (hc, hs)) + block_out.float().unsqueeze(-2) * a.unsqueeze(-1)).flatten(-2)
    g = r_new.unflatten(-1, (hc, hs))
    normed = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2) * (1 + norm_w.float())
    return r_new, normed
