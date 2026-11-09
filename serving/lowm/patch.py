"""Small-kernel patches for hybrid A decode on SM120 (installed as `arc3_lowm.patch` in the SGLang venv).

Each part is a default-off switch read at import time in every SGLang process:

- ARC3_LOWM_GEMM=1: BF16 linears of the shapes measured in
  results/kernel-prof/lowm-splitk-v1 (router, shared expert gate_up / down, PLE
  key / value) at 33-128 rows go to the split-K kernel (lowm_splitk_gemm.py)
  instead of cuBLAS + split-K reduce. Up to 32 rows SGLang's own low-M kernel
  stays.
- ARC3_LOWM_HC_MIX=1: GatedResidual.mix at 17-128 rows runs hc_mix_2k (split-K
  down with silu epilogue + up with sigmoid-mul-mean epilogue) instead of the
  torch.compile chain.
- ARC3_LOWM_HC_COMBINE_NORM=1 (needs HC_MIX): GatedResidual.combine also
  writes the next GatedResidual's hc_norm of the new residual (one launch,
  hc_combine_norm.py). The next mix uses it only when it receives that very
  tensor and is that module, so a PLE add in between (new tensor) falls back
  to the normal norm.

The hooks appended to sglang's unquant.py and hyperconnection.py call
install_linear / install_hc; nothing changes while every switch is off.
"""
import os
import sys
import weakref

import torch

from .hc_combine_norm import hc_combine_norm
from .hc_mix_2k import hc_mix_2k
from .lowm_splitk_gemm import lowm_splitk_linear

_ON = lambda name: os.environ.get(name, "0") == "1"  # noqa: E731
GEMM, HC_MIX, HC_COMBINE_NORM = _ON("ARC3_LOWM_GEMM"), _ON("ARC3_LOWM_HC_MIX"), _ON("ARC3_LOWM_HC_COMBINE_NORM")
HC_COMBINE_NORM = HC_COMBINE_NORM and HC_MIX


def _cfg(bn, bk, split, stages):
    return dict(BN=bn, BK=bk, SPLIT=split, num_warps=4, num_stages=stages)


# (n, k) -> {row bucket: split-K config}; buckets 64 (33-64 rows) and 128 (65-128 rows),
# best configs at M = 40 / 72 in lowm-splitk-v1 (4 warps only: the split-K fix-up is verified there)
_GEMM_CFG = {
    (512, 2560): {64: _cfg(32, 64, 12, 4), 128: _cfg(32, 64, 12, 3)},     # router gate
    (1280, 2560): {64: _cfg(32, 64, 9, 3), 128: _cfg(32, 64, 9, 3)},      # shared expert gate_up
    (2560, 640): {64: _cfg(32, 128, 1, 3), 128: _cfg(32, 128, 1, 3)},     # shared expert down
    (10240, 2560): {64: _cfg(64, 128, 1, 4), 128: _cfg(64, 128, 1, 3)},   # PLE key
    (2560, 2560): {64: _cfg(32, 128, 2, 3), 128: _cfg(32, 64, 2, 4)},     # PLE value
}
_MIX_DOWN = {64: _cfg(32, 64, 19, 4), 128: _cfg(32, 64, 19, 4)}
_MIX_UP = {64: dict(BJ=16, BR=64, num_warps=8, num_stages=3), 128: dict(BJ=16, BR=32, num_warps=8, num_stages=3)}
_COMBINE = dict(BLOCK_D=2048, num_warps=16)


def _bucket(rows: int, lo: int) -> int | None:
    if rows < lo or rows > 128:
        return None
    return 64 if rows <= 64 else 128


def install_linear(cls) -> None:
    """Wrap UnquantizedLinearMethod.apply (ARC3_LOWM_GEMM)."""
    if not GEMM or getattr(cls, "_arc3_lowm", False):
        return
    orig = cls.apply

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if (bias is None and x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
                and w.dim() == 2 and w.is_contiguous()):
            table = _GEMM_CFG.get(tuple(w.shape))
            rows = x.numel() // x.shape[-1]
            b = _bucket(rows, 33) if table else None
            if b is not None:
                x2 = x.reshape(rows, x.shape[-1])
                if not x2.is_contiguous():
                    x2 = x2.contiguous()
                return lowm_splitk_linear(x2, w, table[b]).view(*x.shape[:-1], w.shape[0])
        return orig(self, layer, x, bias)

    cls.apply = apply
    cls._arc3_lowm = True
    print(f"ARC3_LOWM linear hook installed (pid {os.getpid()})", file=sys.stderr, flush=True)


_CREATED: list = []      # weakrefs to mix-capable GatedResidual modules, in construction order
_STASH: list = [None]    # (residual tensor, next module, normed residual) left by the last fused combine


def _next_mixer(module):
    idx = module._arc3_idx
    if idx + 1 < len(_CREATED):
        nxt = _CREATED[idx + 1]()
        if nxt is not None and hasattr(nxt, "input_mix_weight_down"):
            return nxt
    return None


def _mix_ok(module, x) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and _bucket(x.shape[0], 17) is not None
            and module.config.hc_per_branch_norm and module.hc_count in (2, 4, 8)  # tl.arange needs a power of 2
            and module.input_mix_weight_down.weight.dtype == torch.bfloat16
            and module.input_mix_weight_up.weight.dtype == torch.bfloat16)


def install_hc(cls) -> None:
    """Wrap GatedResidual.__init__ / mix / combine (ARC3_LOWM_HC_MIX, ARC3_LOWM_HC_COMBINE_NORM)."""
    if not HC_MIX or getattr(cls, "_arc3_lowm", False):
        return
    orig_init, orig_mix, orig_combine = cls.__init__, cls.mix, cls.combine

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._arc3_idx = len(_CREATED)
        _CREATED.append(weakref.ref(self))

    def mix(self, hyper_input):
        if not _mix_ok(self, hyper_input):
            return orig_mix(self, hyper_input)
        stash, _STASH[0] = _STASH[0], None
        if stash is not None and stash[0] is hyper_input and stash[1] is self:
            normed = stash[2]
        else:
            normed = self.hc_norm(hyper_input)
        b = _bucket(hyper_input.shape[0], 17)
        out = hc_mix_2k(normed.contiguous(), self.input_mix_weight_down.weight, self.input_mix_weight_up.weight,
                        self.hc_count, self.hidden_size, _MIX_DOWN[b], _MIX_UP[b])
        return out.to(self.params_dtype), (hyper_input, normed)

    def combine(self, block_output, residuals):
        hyper_input, hyper_input_normed = residuals
        nxt = _next_mixer(self) if HC_COMBINE_NORM else None
        if (nxt is None or not _mix_ok(nxt, hyper_input) or block_output.dtype != torch.bfloat16
                or hyper_input.dtype != torch.bfloat16 or hyper_input_normed.dtype != torch.bfloat16
                or self.block_inject_weight.weight.dtype != torch.bfloat16 or block_output.shape[0] == 0):
            return orig_combine(self, block_output, residuals)
        out_r, out_n = hc_combine_norm(
            block_output.contiguous(), hyper_input.contiguous(), hyper_input_normed.contiguous(),
            self.block_inject_weight.weight, nxt.hc_norm.weight, self.hc_count, self.hidden_size,
            nxt.hc_norm.variance_epsilon, _COMBINE)
        _STASH[0] = (out_r, nxt, out_n)
        return out_r

    cls.__init__, cls.mix, cls.combine = __init__, mix, combine
    cls._arc3_lowm = True
    print(f"ARC3_LOWM HC hooks installed (combine+norm {HC_COMBINE_NORM}, pid {os.getpid()})",
          file=sys.stderr, flush=True)
