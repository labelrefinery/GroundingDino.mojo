"""Vision neck: 1x1/3x3 input projections with GroupNorm, sine position embeddings.

Turns the three Swin levels into the four d_model=256 levels the encoder consumes
(the fourth is a stride-2 3x3 convolution on the *unprojected* last Swin level, as in
`GroundingDinoModel.forward`), attaches sine position embeddings plus the learned
per-level embedding, and flattens everything into one `(S, 256)` sequence.

Batch size is 1 and no image padding is used, so `pixel_mask` is all ones: the sine
embedding's cumulative sums reduce to row/column indices and all valid ratios are 1.
"""

from std.math import cos, sin, sqrt

from .io import Config
from .swin import FeatureMap
from .tensor import Tensor, matmul_nt

comptime GROUP_NORM_EPS = Float32(1e-5)
comptime GROUP_NORM_GROUPS = 32


def group_norm(x: Tensor, gamma: Tensor, beta: Tensor, groups: Int) raises -> Tensor:
    """`nn.GroupNorm(groups, C)` over a `(H * W, C)` feature map (batch size 1)."""
    var rows = x.dim(0)
    var channels = x.dim(1)
    if channels % groups != 0:
        raise Error("group_norm: channels not divisible by groups")
    var per_group = channels // groups
    var out = Tensor.zeros([rows, channels])
    var src = x.ptr()
    var dst = out.ptr()
    var count = Float32(rows * per_group)
    for g in range(groups):
        var c0 = g * per_group
        var mean: Float32 = 0.0
        for r in range(rows):
            for c in range(c0, c0 + per_group):
                mean += src[unsafe_offset=r * channels + c]
        mean /= count
        var acc: Float32 = 0.0
        for r in range(rows):
            for c in range(c0, c0 + per_group):
                var d = src[unsafe_offset=r * channels + c] - mean
                acc += d * d
        acc /= count
        var inv_std = 1.0 / sqrt(acc + GROUP_NORM_EPS)
        for r in range(rows):
            for c in range(c0, c0 + per_group):
                var i = r * channels + c
                dst[unsafe_offset=i] = (src[unsafe_offset=i] - mean) * inv_std * gamma[c] + beta[c]
    return out^


def conv3x3_stride2(feature: FeatureMap, w: Tensor, b: Tensor) raises -> FeatureMap:
    """3x3 stride-2 padding-1 convolution over a `(H * W, C_in)` map."""
    var height = feature.height
    var width = feature.width
    var cin = feature.channels
    var cout = w.dim(0)
    var out_h = (height + 2 - 3) // 2 + 1
    var out_w = (width + 2 - 3) // 2 + 1
    var out = Tensor.zeros([out_h * out_w, cout])
    var src = feature.data.ptr()
    for oy in range(out_h):
        for ox in range(out_w):
            var row = oy * out_w + ox
            for oc in range(cout):
                var acc = b[oc]
                for ky in range(3):
                    var iy = oy * 2 + ky - 1
                    if iy < 0 or iy >= height:
                        continue
                    for kx in range(3):
                        var ix = ox * 2 + kx - 1
                        if ix < 0 or ix >= width:
                            continue
                        var base_w = ((oc * cin) * 3 + ky) * 3 + kx
                        var base_x = (iy * width + ix) * cin
                        for ic in range(cin):
                            acc += src[unsafe_offset=base_x + ic] * w[base_w + ic * 9]
                out.set2(row, oc, acc)
    return FeatureMap(out^, out_h, out_w, cout)


def sine_position_embedding(
    height: Int, width: Int, d_model: Int, temperature: Float32
) raises -> Tensor:
    """`GroundingDinoSinePositionEmbedding` for an all-valid pixel mask, as `(H * W, d_model)`.

    Channels `[0, d_model/2)` carry the y encoding and `[d_model/2, d_model)` the x one,
    matching the `cat((pos_y, pos_x))` ordering of the reference implementation.
    """
    var half = d_model // 2
    var scale = Float32(6.283185307179586)
    var eps = Float32(1e-6)
    var dim_t = Tensor.zeros([half])
    for i in range(half):
        dim_t[i] = temperature ** (Float32(2 * (i // 2)) / Float32(half))

    var out = Tensor.zeros([height * width, d_model])
    for y in range(height):
        var y_embed = Float32(y + 1) / (Float32(height) + eps) * scale
        for x in range(width):
            var x_embed = Float32(x + 1) / (Float32(width) + eps) * scale
            var row = y * width + x
            for i in range(half):
                var py = y_embed / dim_t[i]
                var px = x_embed / dim_t[i]
                if i % 2 == 0:
                    out.set2(row, i, sin(py))
                    out.set2(row, half + i, sin(px))
                else:
                    out.set2(row, i, cos(py))
                    out.set2(row, half + i, cos(px))
    return out^


@fieldwise_init
struct MultiScaleFeatures(Copyable, Movable):
    """The flattened multi-level vision sequence handed to the encoder."""

    var features: Tensor
    """`(S, d_model)` concatenation of every level."""
    var position: Tensor
    """`(S, d_model)` sine position embedding plus the learned level embedding."""
    var heights: List[Int]
    var widths: List[Int]
    var starts: List[Int]

    def num_levels(self) -> Int:
        return len(self.heights)

    def length(self) -> Int:
        return self.features.dim(0)


def project_levels(
    levels: List[FeatureMap], weights: Dict[String, Tensor], cfg: Config
) raises -> MultiScaleFeatures:
    """Project the Swin levels to d_model, add the extra stride-64 level, and flatten."""
    var projected = List[FeatureMap]()
    for level in range(len(levels)):
        ref src = levels[level]
        var name = "proj" + String(level)
        # A 1x1 convolution over (H*W, C_in) is exactly a linear layer once the
        # trailing kernel dimensions are dropped from the (C_out, C_in, 1, 1) weight.
        ref kernel = weights[name + ".conv.w"]
        var w2 = kernel.reshaped([kernel.dim(0), kernel.dim(1)])
        var conv = matmul_nt(src.data, w2, weights[name + ".conv.b"])
        var normed = group_norm(
            conv, weights[name + ".gn.w"], weights[name + ".gn.b"], GROUP_NORM_GROUPS
        )
        projected.append(FeatureMap(normed^, src.height, src.width, cfg.d_model))

    for level in range(len(levels), cfg.num_levels):
        var name = "proj" + String(level)
        # The first extra level convolves the *unprojected* last backbone map.
        var use_backbone = level == len(levels)
        var conv = (
            conv3x3_stride2(
                levels[len(levels) - 1], weights[name + ".conv.w"], weights[name + ".conv.b"]
            )
            if use_backbone
            else conv3x3_stride2(
                projected[level - 1], weights[name + ".conv.w"], weights[name + ".conv.b"]
            )
        )
        var normed = group_norm(
            conv.data, weights[name + ".gn.w"], weights[name + ".gn.b"], GROUP_NORM_GROUPS
        )
        projected.append(FeatureMap(normed^, conv.height, conv.width, cfg.d_model))

    var total = 0
    var heights = List[Int]()
    var widths = List[Int]()
    var starts = List[Int]()
    for level in range(len(projected)):
        starts.append(total)
        heights.append(projected[level].height)
        widths.append(projected[level].width)
        total += projected[level].height * projected[level].width

    var features = Tensor.zeros([total, cfg.d_model])
    var position = Tensor.zeros([total, cfg.d_model])
    ref level_embed = weights["level_embed"]
    for level in range(len(projected)):
        ref p = projected[level]
        var pos = sine_position_embedding(
            p.height, p.width, cfg.d_model, cfg.pos_temperature
        )
        var base = starts[level]
        for r in range(p.height * p.width):
            for c in range(cfg.d_model):
                features.set2(base + r, c, p.data.at2(r, c))
                position.set2(base + r, c, pos.at2(r, c) + level_embed.at2(level, c))
    return MultiScaleFeatures(features^, position^, heights^, widths^, starts^)
