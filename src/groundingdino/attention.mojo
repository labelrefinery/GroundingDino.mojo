"""Scaled dot-product multi-head attention, shared by BERT, the text enhancer and the decoder.

This is `GroundingDinoMultiheadAttention` / `BertSelfAttention`: separate q/k/v/out
linear layers, `scores / sqrt(head_dim)`, an additive mask, softmax, then the output
projection. Masks are additive (`0` to keep, a large negative to drop) exactly like
transformers' converted attention masks.
"""

from std.math import sqrt, exp

from .tensor import FP, Tensor, VW, matmul_nt

comptime MASK_NEG = Float32(-1.0e30)
"""Additive value for masked positions; softmax drives these to exactly zero."""


def _dot(a: FP, b: FP, n: Int) -> Float32:
    var acc = SIMD[DType.float32, VW](0)
    var t = 0
    while t + VW <= n:
        acc = a.unsafe_load[width=VW](t).fma(b.unsafe_load[width=VW](t), acc)
        t += VW
    var s = acc.reduce_add()
    while t < n:
        s += a[unsafe_offset=t] * b[unsafe_offset=t]
        t += 1
    return s


def _axpy(dst: FP, src: FP, scale: Float32, n: Int):
    var sv = SIMD[DType.float32, VW](scale)
    var t = 0
    while t + VW <= n:
        dst.unsafe_store(t, src.unsafe_load[width=VW](t).fma(sv, dst.unsafe_load[width=VW](t)))
        t += VW
    while t < n:
        dst[unsafe_offset=t] = dst[unsafe_offset=t] + scale * src[unsafe_offset=t]
        t += 1


def attention_core(
    q: Tensor, k: Tensor, v: Tensor, num_heads: Int, mask: Tensor, use_mask: Bool
) raises -> Tensor:
    """Multi-head attention over already-projected `q`/`k`/`v` of shape `(T, D)`.

    Args:
        q: Queries, shape `(Tq, D)`.
        k: Keys, shape `(Tk, D)`.
        v: Values, shape `(Tk, D)`.
        num_heads: Number of attention heads; `D` must be divisible by it.
        mask: Additive mask of shape `(Tq, Tk)`, ignored when `use_mask` is False.
        use_mask: Whether `mask` participates.

    Returns:
        The context of shape `(Tq, D)` (before the output projection).
    """
    var tq = q.dim(0)
    var tk = k.dim(0)
    var d = q.dim(1)
    if d % num_heads != 0:
        raise Error("attention_core: d_model not divisible by num_heads")
    var hd = d // num_heads
    var scale = 1.0 / sqrt(Float32(hd))

    var out = Tensor.zeros([tq, d])
    var scores = Tensor.zeros([tk])
    var sp = scores.ptr()
    var qp = q.ptr()
    var kp = k.ptr()
    var vp = v.ptr()
    var op = out.ptr()
    var mp = mask.ptr()

    for h in range(num_heads):
        var off = h * hd
        for i in range(tq):
            var qrow = qp + i * d + off
            var mrow = mp + i * tk
            var mx = MASK_NEG
            for j in range(tk):
                var s = _dot(qrow, kp + j * d + off, hd) * scale
                if use_mask:
                    s += mrow[unsafe_offset=j]
                sp[unsafe_offset=j] = s
                if s > mx:
                    mx = s
            var total: Float32 = 0.0
            for j in range(tk):
                var e = exp(sp[unsafe_offset=j] - mx)
                sp[unsafe_offset=j] = e
                total += e
            var inv = 1.0 / total
            var orow = op + i * d + off
            for j in range(tk):
                var p = sp[unsafe_offset=j] * inv
                if p != 0.0:
                    _axpy(orow, vp + j * d + off, p, hd)
    # `scores` is only ever touched through `sp`, so keep it alive past the loops:
    # Mojo destroys values at their last *use*, which would otherwise be `.ptr()`.
    _ = scores^
    return out^


def multihead_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    weights: Dict[String, Tensor],
    prefix: String,
    num_heads: Int,
    mask: Tensor,
    use_mask: Bool,
) raises -> Tensor:
    """Projected multi-head attention using the exported `<prefix>.{q,k,v,out}` layers."""
    var q = matmul_nt(queries, weights[prefix + ".q.w"], weights[prefix + ".q.b"])
    var k = matmul_nt(keys, weights[prefix + ".k.w"], weights[prefix + ".k.b"])
    var v = matmul_nt(values, weights[prefix + ".v.w"], weights[prefix + ".v.b"])
    var ctx = attention_core(q, k, v, num_heads, mask, use_mask)
    return matmul_nt(ctx, weights[prefix + ".out.w"], weights[prefix + ".out.b"])
