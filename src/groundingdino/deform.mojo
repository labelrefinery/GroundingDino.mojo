"""Multi-scale deformable attention (Deformable DETR), used by both encoder and decoder.

Each query predicts `n_heads * n_levels * n_points` sampling offsets around its
reference point and a softmax over those samples; the value sequence is bilinearly
sampled at those locations. The bilinear sampler reproduces
`F.grid_sample(..., mode="bilinear", padding_mode="zeros", align_corners=False)`:
`x_pixel = location * width - 0.5`, with out-of-range corners contributing zero.
"""

from std.math import floor

from .attention import _axpy
from .tensor import FP, Tensor, keep_alive, matmul_nt, softmax_rows_


def _sample_bilinear(
    value: FP, start: Int, height: Int, width: Int, stride: Int, channel_off: Int,
    head_dim: Int, x: Float32, y: Float32, weight: Float32, dst: FP
):
    """Accumulate `weight * bilinear(value, x, y)` into `dst` for one head's channels."""
    var x0 = Int(floor(x))
    var y0 = Int(floor(y))
    var wx1 = x - Float32(x0)
    var wy1 = y - Float32(y0)
    var wx0 = 1.0 - wx1
    var wy0 = 1.0 - wy1
    for dy in range(2):
        var yy = y0 + dy
        if yy < 0 or yy >= height:
            continue
        var wy = wy0 if dy == 0 else wy1
        for dx in range(2):
            var xx = x0 + dx
            if xx < 0 or xx >= width:
                continue
            var wxx = wx0 if dx == 0 else wx1
            var corner = weight * wy * wxx
            if corner == 0.0:
                continue
            var src = value.unsafe_offset((start + yy * width + xx) * stride + channel_off)
            _axpy(dst, src, corner, head_dim)


def deformable_attention(
    query: Tensor,
    position: Tensor,
    use_position: Bool,
    value_seq: Tensor,
    reference: Tensor,
    heights: List[Int],
    widths: List[Int],
    starts: List[Int],
    weights: Dict[String, Tensor],
    prefix: String,
    num_heads: Int,
    n_points: Int,
) raises -> Tensor:
    """Multi-scale deformable attention.

    Args:
        query: `(Q, D)` query features.
        position: `(Q, D)` position embedding added to the query before projection.
        use_position: Whether `position` participates.
        value_seq: `(S, D)` flattened multi-level value sequence.
        reference: `(Q, L, 2)` normalized points, or `(Q, L, 4)` reference boxes.
        heights: Per-level heights.
        widths: Per-level widths.
        starts: Per-level start index into `value_seq`.
        weights: The exported weight dictionary.
        prefix: Name prefix for `.samp` / `.attnw` / `.value` / `.outp`.
        num_heads: Attention head count.
        n_points: Sampling points per head and level.

    Returns:
        `(Q, D)` attention output after the output projection.
    """
    var q_len = query.dim(0)
    var d_model = query.dim(1)
    var num_levels = len(heights)
    var head_dim = d_model // num_heads
    var ref_dim = reference.dim(2)

    var hidden = Tensor(copy=query)
    if use_position:
        for i in range(hidden.numel()):
            hidden[i] = hidden[i] + position[i]

    var value = matmul_nt(value_seq, weights[prefix + ".value.w"], weights[prefix + ".value.b"])
    var offsets = matmul_nt(hidden, weights[prefix + ".samp.w"], weights[prefix + ".samp.b"])
    var attn = matmul_nt(hidden, weights[prefix + ".attnw.w"], weights[prefix + ".attnw.b"])
    softmax_rows_(attn, num_levels * n_points)

    var out = Tensor.zeros([q_len, d_model])
    var vp = value.ptr()
    var op = out.ptr()
    var offp = offsets.ptr()
    var ap = attn.ptr()
    var rp = reference.ptr()

    for q in range(q_len):
        var off_base = q * num_heads * num_levels * n_points * 2
        var attn_base = q * num_heads * num_levels * n_points
        var ref_base = q * num_levels * ref_dim
        for h in range(num_heads):
            var dst = op.unsafe_offset(q * d_model + h * head_dim)
            for l in range(num_levels):
                var width = widths[l]
                var height = heights[l]
                var ref_x = rp[unsafe_offset=ref_base + l * ref_dim]
                var ref_y = rp[unsafe_offset=ref_base + l * ref_dim + 1]
                for p in range(n_points):
                    var oi = off_base + ((h * num_levels + l) * n_points + p) * 2
                    var dx = offp[unsafe_offset=oi]
                    var dy = offp[unsafe_offset=oi + 1]
                    var loc_x: Float32
                    var loc_y: Float32
                    if ref_dim == 2:
                        loc_x = ref_x + dx / Float32(width)
                        loc_y = ref_y + dy / Float32(height)
                    else:
                        var ref_w = rp[unsafe_offset=ref_base + l * ref_dim + 2]
                        var ref_h = rp[unsafe_offset=ref_base + l * ref_dim + 3]
                        loc_x = ref_x + dx / Float32(n_points) * ref_w * 0.5
                        loc_y = ref_y + dy / Float32(n_points) * ref_h * 0.5
                    var weight = ap[unsafe_offset=attn_base + (h * num_levels + l) * n_points + p]
                    _sample_bilinear(
                        vp, starts[l], height, width, d_model, h * head_dim, head_dim,
                        loc_x * Float32(width) - 0.5, loc_y * Float32(height) - 0.5,
                        weight, dst,
                    )
    keep_alive(value)
    keep_alive(offsets)
    keep_alive(attn)
    return matmul_nt(out, weights[prefix + ".outp.w"], weights[prefix + ".outp.b"])
