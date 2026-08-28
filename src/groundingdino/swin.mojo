"""Swin-T backbone: patch embedding, shifted-window attention, patch merging.

Mirrors `SwinBackbone` as Grounding DINO drives it -- `always_partition=True` (the
window is never shrunk to the input resolution) and
`output_hidden_states_before_downsampling=True` (each stage reports its features
*before* patch merging). Stages 2, 3 and 4 are returned, at strides 8/16/32 with
192/384/768 channels, each passed through its `hidden_states_norms` LayerNorm.

Relative position bias arrives from the exporter already expanded to
`(num_heads, 49, 49)`, so nothing here needs the relative position index table.
"""

from std.math import exp, sqrt

from .attention import _axpy, _dot
from .io import Config
from .tensor import (
    Tensor, add_, gelu_, keep_alive, layernorm_named, matmul_nt, matmul_nt_nobias
)

comptime SWIN_EPS = Float32(1e-5)
"""`nn.LayerNorm` default eps, which is what SwinEmbeddings/SwinLayer use."""


@fieldwise_init
struct FeatureMap(Copyable, Movable, Writable):
    """A `(height * width, channels)` feature map with its spatial shape."""

    var data: Tensor
    var height: Int
    var width: Int
    var channels: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FeatureMap(", self.channels, "x", self.height, "x", self.width, ")")


def patch_embed(
    pixels: Tensor, weights: Dict[String, Tensor], cfg: Config
) raises -> FeatureMap:
    """4x4 stride-4 convolution over `(3, H, W)`, zero-padding H/W up to a multiple of 4."""
    var channels_in = pixels.dim(0)
    var height = pixels.dim(1)
    var width = pixels.dim(2)
    var pad_h = (4 - height % 4) % 4
    var pad_w = (4 - width % 4) % 4
    var out_h = (height + pad_h) // 4
    var out_w = (width + pad_w) // 4
    var dim = cfg.swin_embed_dim

    ref w = weights["swin.patch.w"]
    ref b = weights["swin.patch.b"]
    var out = Tensor.zeros([out_h * out_w, dim])
    for oy in range(out_h):
        for ox in range(out_w):
            var row = oy * out_w + ox
            for oc in range(dim):
                var acc = b[oc]
                for ic in range(channels_in):
                    for ky in range(4):
                        var iy = oy * 4 + ky
                        if iy >= height:
                            continue
                        for kx in range(4):
                            var ix = ox * 4 + kx
                            if ix >= width:
                                continue
                            var wi = ((oc * channels_in + ic) * 4 + ky) * 4 + kx
                            acc += pixels.at3(ic, iy, ix) * w[wi]
                out.set2(row, oc, acc)
    var normed = layernorm_named(out, weights, "swin.embed_norm", SWIN_EPS)
    return FeatureMap(normed^, out_h, out_w, dim)


def _region(index: Int, extent: Int, window: Int, shift: Int) -> Int:
    var r = 0
    if index >= extent - window:
        r += 1
    if index >= extent - shift:
        r += 1
    return r


def window_attention(
    hidden: Tensor,
    height: Int,
    width: Int,
    channels: Int,
    num_heads: Int,
    shift: Int,
    window: Int,
    weights: Dict[String, Tensor],
    prefix: String,
) raises -> Tensor:
    """(Shifted) window multi-head self-attention over a `(H * W, C)` feature map."""
    var pad_bottom = (window - height % window) % window
    var pad_right = (window - width % window) % window
    var hp = height + pad_bottom
    var wp = width + pad_right
    var nwh = hp // window
    var nww = wp // window
    var num_windows = nwh * nww
    var area = window * window

    # Gather the cyclically shifted, padded windows into one contiguous (nW*49, C) block
    # so the q/k/v projections are a single big matmul.
    var src_index = List[Int](length=num_windows * area, fill=-1)
    var packed = Tensor.zeros([num_windows * area, channels])
    var hp_ptr = hidden.ptr()
    var pk = packed.ptr()
    for wi in range(num_windows):
        var bh = wi // nww
        var bw = wi % nww
        for p in range(area):
            var y = bh * window + p // window
            var x = bw * window + p % window
            var sy = (y + shift) % hp
            var sx = (x + shift) % wp
            var row = wi * area + p
            if sy < height and sx < width:
                var src = (sy * width + sx) * channels
                src_index[row] = sy * width + sx
                for c in range(channels):
                    pk[unsafe_offset=row * channels + c] = hp_ptr[unsafe_offset=src + c]

    var q = matmul_nt(packed, weights[prefix + ".q.w"], weights[prefix + ".q.b"])
    var k = matmul_nt(packed, weights[prefix + ".k.w"], weights[prefix + ".k.b"])
    var v = matmul_nt(packed, weights[prefix + ".v.w"], weights[prefix + ".v.b"])

    # Cyclic-shift attention mask: positions from different shift regions never mix.
    var regions = List[Int](length=num_windows * area, fill=0)
    if shift > 0:
        for wi in range(num_windows):
            var bh = wi // nww
            var bw = wi % nww
            for p in range(area):
                var y = bh * window + p // window
                var x = bw * window + p % window
                regions[wi * area + p] = (
                    _region(y, hp, window, shift) * 3 + _region(x, wp, window, shift)
                )

    var head_dim = channels // num_heads
    var scale = 1.0 / sqrt(Float32(head_dim))
    ref rpb = weights[prefix + ".rpb"]
    var ctx = Tensor.zeros([num_windows * area, channels])
    var scores = Tensor.zeros([area])
    var sp = scores.ptr()
    var qp = q.ptr()
    var kp = k.ptr()
    var vp = v.ptr()
    var cp = ctx.ptr()
    var rp = rpb.ptr()

    for wi in range(num_windows):
        var base = wi * area
        for h in range(num_heads):
            var off = h * head_dim
            var bias = rp.unsafe_offset(h * area * area)
            for i in range(area):
                var qrow = qp.unsafe_offset((base + i) * channels + off)
                var mx = Float32(-1.0e30)
                for j in range(area):
                    var s = (
                        _dot(qrow, kp.unsafe_offset((base + j) * channels + off), head_dim) * scale
                        + bias[unsafe_offset=i * area + j]
                    )
                    if shift > 0 and regions[base + i] != regions[base + j]:
                        s -= 100.0
                    sp[unsafe_offset=j] = s
                    if s > mx:
                        mx = s
                var total: Float32 = 0.0
                for j in range(area):
                    var e = exp(sp[unsafe_offset=j] - mx)
                    sp[unsafe_offset=j] = e
                    total += e
                var inv = 1.0 / total
                var orow = cp.unsafe_offset((base + i) * channels + off)
                for j in range(area):
                    _axpy(orow, vp.unsafe_offset((base + j) * channels + off), sp[unsafe_offset=j] * inv, head_dim)
    keep_alive(scores)
    keep_alive(q)
    keep_alive(k)
    keep_alive(v)

    var projected = matmul_nt(ctx, weights[prefix + ".o.w"], weights[prefix + ".o.b"])

    # Scatter back, dropping the padded positions and undoing the cyclic shift.
    var out = Tensor.zeros([height * width, channels])
    var op = out.ptr()
    var pp = projected.ptr()
    for row in range(num_windows * area):
        var dst = src_index[row]
        if dst >= 0:
            for c in range(channels):
                op[unsafe_offset=dst * channels + c] = pp[unsafe_offset=row * channels + c]
    keep_alive(projected)
    return out^


def swin_block(
    feature: FeatureMap,
    num_heads: Int,
    shift: Int,
    window: Int,
    weights: Dict[String, Tensor],
    prefix: String,
) raises -> FeatureMap:
    """One Swin transformer block: (S)W-MSA then MLP, both post-norm residual."""
    var normed = layernorm_named(feature.data, weights, prefix + ".norm1", SWIN_EPS)
    var attn = window_attention(
        normed,
        feature.height,
        feature.width,
        feature.channels,
        num_heads,
        shift,
        window,
        weights,
        prefix,
    )
    add_(attn, feature.data)

    var normed2 = layernorm_named(attn, weights, prefix + ".norm2", SWIN_EPS)
    var mlp = matmul_nt(normed2, weights[prefix + ".fc1.w"], weights[prefix + ".fc1.b"])
    gelu_(mlp)
    var mlp2 = matmul_nt(mlp, weights[prefix + ".fc2.w"], weights[prefix + ".fc2.b"])
    add_(mlp2, attn)
    return FeatureMap(mlp2^, feature.height, feature.width, feature.channels)


def patch_merging(
    feature: FeatureMap, weights: Dict[String, Tensor], prefix: String
) raises -> FeatureMap:
    """Concatenate the four 2x2 sub-grids, LayerNorm, then a bias-free 4C -> 2C linear."""
    var height = feature.height
    var width = feature.width
    var channels = feature.channels
    var out_h = (height + 1) // 2
    var out_w = (width + 1) // 2
    var merged = Tensor.zeros([out_h * out_w, 4 * channels])
    var src = feature.data.ptr()
    var dst = merged.ptr()
    # torch does `for col in range(2) for row in range(2)`, i.e. (0,0), (1,0), (0,1), (1,1).
    var row_offsets = [0, 1, 0, 1]
    var col_offsets = [0, 0, 1, 1]
    for oy in range(out_h):
        for ox in range(out_w):
            var out_row = (oy * out_w + ox) * 4 * channels
            for q in range(4):
                var sy = 2 * oy + row_offsets[q]
                var sx = 2 * ox + col_offsets[q]
                if sy >= height or sx >= width:
                    continue
                var in_row = (sy * width + sx) * channels
                for c in range(channels):
                    dst[unsafe_offset=out_row + q * channels + c] = src[unsafe_offset=in_row + c]
    var normed = layernorm_named(merged, weights, prefix + ".down.norm", SWIN_EPS)
    var reduced = matmul_nt_nobias(normed, weights[prefix + ".down.reduction.w"])
    return FeatureMap(reduced^, out_h, out_w, 2 * channels)


def forward(
    pixels: Tensor, weights: Dict[String, Tensor], cfg: Config
) raises -> List[FeatureMap]:
    """Run the backbone; returns the stage-2/3/4 feature maps, LayerNorm'd."""
    var feature = patch_embed(pixels, weights, cfg)
    var window = cfg.window_size
    var outputs = List[FeatureMap]()

    for stage in range(len(cfg.swin_depths)):
        for block in range(cfg.swin_depths[stage]):
            var shift = 0 if block % 2 == 0 else window // 2
            feature = swin_block(
                feature,
                cfg.swin_heads[stage],
                shift,
                window,
                weights,
                "swin.s" + String(stage) + ".b" + String(block),
            )
        if stage >= 1:
            var name = "swin.out_norm" + String(stage - 1)
            var normed = layernorm_named(feature.data, weights, name, SWIN_EPS)
            outputs.append(
                FeatureMap(normed^, feature.height, feature.width, feature.channels)
            )
        if stage < len(cfg.swin_depths) - 1:
            feature = patch_merging(feature, weights, "swin.s" + String(stage))
    return outputs^
