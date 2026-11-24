"""Fused short conv + post-op for the K / V prefill prologue of sliding-window layers (imported on the GPU path).

A copy of sglang.kernels.ops.mamba.inkling2_sconv._sconv_prefill_kernel (static weights, residual; the same FP32
conv code, enable_fp_fusion=False like stock) whose BF16 output feeds, in registers, the op stock runs next:
    POST = 1 (K): QKL2Clip without scale / output scale:  ss = sum(y^2); norm = ss * rsqrt(ss) (0 if ss == 0);
                  y * min(1, thr / (norm + eps))
    POST = 2 (V): RMSNorm without weight: y * rsqrt(ss / 128 + eps)
One program = 16 rows x one 128-wide head (BN = head_dim), so the head reduction stays in the program. The row sum
order differs from the CuTe kernels (FP32 order class); every rounding point (BF16 conv output, BF16 result) is kept.
"""

import triton
import triton.language as tl

from sglang.kernels.numerics import rsqrt_approx_f32


@triton.jit
def _sconv_post_kernel(
    X, Weight, Cache, Indices, Mask, Cu, Seq, Y, n_rows,
    SX: tl.constexpr, SW0: tl.constexpr, SW1: tl.constexpr, SC0: tl.constexpr, SC1: tl.constexpr,
    SC2: tl.constexpr, D: tl.constexpr, W: tl.constexpr, SCALE: tl.constexpr, POST: tl.constexpr, THR, EPS,
    BM: tl.constexpr, BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    valid = (rows < n_rows)[:, None] & (col < D)[None, :]
    rows = tl.minimum(rows, n_rows - 1)
    seq = tl.load(Seq + rows)
    bos = tl.load(Cu + seq)
    slot = tl.load(Indices + seq).to(tl.int64)
    has_cache = (tl.load(Mask + seq) != 0) & (slot > 0)
    current = tl.load(X + rows[:, None] * SX + col[None, :], valid, other=0).to(tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = W - 1 - tap
        src = rows - lag
        from_x = (src >= bos)[:, None]
        value = tl.load(X + src[:, None] * SX + col[None, :], valid & from_x, other=0).to(tl.float32)
        prefix = tl.load(
            Cache + slot[:, None] * SC0 + (src - bos + W - 1)[:, None] * SC1 + col[None, :] * SC2,
            valid & ~from_x & has_cache[:, None],
            other=0,
        ).to(tl.float32)
        value = tl.where(from_x, value, prefix)
        weight = (
            tl.load(Weight + col * SW0 + tap * SW1, col < D, other=0).to(X.dtype.element_ty).to(tl.float32)
        )[None, :]
        acc = tl.fma(value, weight, acc)
    acc = acc + current
    acc = acc * SCALE
    y = acc.to(tl.bfloat16).to(tl.float32)  # the stock conv output (BF16)
    ss = tl.sum(y * y, 1)
    if POST == 1:
        norm = tl.where(ss > 0.0, ss * rsqrt_approx_f32(ss), 0.0)
        coeff = tl.minimum(1.0, tl.div_rn(tl.full(norm.shape, THR, tl.float32), norm + EPS),
                           propagate_nan=tl.PropagateNan.ALL)  # stock minimumf propagates NaN
    else:
        coeff = rsqrt_approx_f32(tl.div_rn(ss, tl.full(ss.shape, BN, tl.float32)) + EPS)
    out = y * coeff[:, None]
    offs = rows[:, None].to(tl.int64) * D + col[None, :]
    tl.store(Y + offs, out.to(tl.bfloat16), valid)
