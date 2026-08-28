"""Grounding DINO feature enhancer: the 6-layer vision/text fusion encoder.

Each layer runs, in this order (matching `GroundingDinoEncoderLayer.forward`):

1. `fusion_layer` -- pre-LayerNorm on both streams, then bidirectional
   image<->text cross-attention whose deltas are scaled by the learned
   `vision_param` / `text_param` vectors.
2. `text_enhancer_layer` -- text self-attention restricted to the phrase blocks,
   with sinusoidal position embeddings of the per-phrase position ids, plus an FFN.
3. `deformable_layer` -- multi-scale deformable self-attention over the four vision
   levels, plus an FFN.

Batch size 1 with no image padding means every vision/text key-padding mask is empty,
so only the phrase-block mask has any effect.
"""

from std.math import cos, sin, sqrt

from .attention import MASK_NEG, _axpy, _dot, attention_core
from .deform import deformable_attention
from .io import Config
from .tensor import (
    Tensor, add_, add_scaled_, keep_alive, layernorm_named, matmul_nt, relu_, softmax_rows_
)
from .vision import MultiScaleFeatures

comptime BI_CLAMP = Float32(50000.0)
"""`GroundingDinoBiMultiHeadAttention` clamps its logits to +/-50000 for fp16 safety."""


def sinusoidal_embedding(
    coords: Tensor, num_pos_feats: Int, temperature: Float32, truncate: Bool
) raises -> Tensor:
    """`encode_sinusoidal_position_embedding` for `(Q, n_coords)` normalized coordinates.

    Each coordinate gets `num_pos_feats` interleaved sin/cos components; for two or more
    coordinates the first two blocks are swapped to the DETR `[pos_y, pos_x, ...]` order.

    `truncate` reproduces a quirk of the reference implementation: it ends with
    `.to(pos_tensor.dtype)`, so when the caller passes *integer* position ids -- which
    `GroundingDinoEncoderLayer.get_text_position_embeddings` does -- every sin/cos value
    is truncated toward zero to an integer. The text enhancer therefore sees an almost
    entirely zero position embedding. Reproducing this is required for parity.
    """
    var q_len = coords.dim(0)
    var n_coords = coords.dim(1)
    var scale = Float32(6.283185307179586)
    var dim_t = Tensor.zeros([num_pos_feats])
    for i in range(num_pos_feats):
        dim_t[i] = temperature ** (Float32(2 * (i // 2)) / Float32(num_pos_feats))

    var out = Tensor.zeros([q_len, n_coords * num_pos_feats])
    for q in range(q_len):
        for c in range(n_coords):
            var slot = c
            if n_coords >= 2:
                if c == 0:
                    slot = 1
                elif c == 1:
                    slot = 0
            var v = coords.at2(q, c) * scale
            var base = slot * num_pos_feats
            for i in range(num_pos_feats):
                var e = v / dim_t[i]
                var value = sin(e) if i % 2 == 0 else cos(e)
                if truncate:
                    value = Float32(Int(value))
                out.set2(q, base + i, value)
    return out^


def reference_points(features: MultiScaleFeatures) raises -> Tensor:
    """`GroundingDinoEncoder.get_reference_points` with all valid ratios equal to 1."""
    var total = features.length()
    var num_levels = features.num_levels()
    var out = Tensor.zeros([total, num_levels, 2])
    for l in range(num_levels):
        var height = features.heights[l]
        var width = features.widths[l]
        var base = features.starts[l]
        for y in range(height):
            var ry = (Float32(y) + 0.5) / Float32(height)
            for x in range(width):
                var rx = (Float32(x) + 0.5) / Float32(width)
                var row = base + y * width + x
                for m in range(num_levels):
                    out.set3(row, m, 0, rx)
                    out.set3(row, m, 1, ry)
    return out^


def bi_attention(
    vision: Tensor, text: Tensor, weights: Dict[String, Tensor], prefix: String, num_heads: Int
) raises -> Tuple[Tensor, Tensor]:
    """Bidirectional image<->text cross-attention (`GroundingDinoBiMultiHeadAttention`)."""
    var s_len = vision.dim(0)
    var n_len = text.dim(0)
    var embed_dim = weights[prefix + ".vision_proj.w"].dim(0)
    var head_dim = embed_dim // num_heads
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    var vq = matmul_nt(vision, weights[prefix + ".vision_proj.w"], weights[prefix + ".vision_proj.b"])
    for i in range(vq.numel()):
        vq[i] = vq[i] * scale
    var tk = matmul_nt(text, weights[prefix + ".text_proj.w"], weights[prefix + ".text_proj.b"])
    var vv = matmul_nt(vision, weights[prefix + ".vv_proj.w"], weights[prefix + ".vv_proj.b"])
    var tv = matmul_nt(text, weights[prefix + ".vt_proj.w"], weights[prefix + ".vt_proj.b"])

    # logits[h, s, n]
    var logits = Tensor.zeros([num_heads * s_len, n_len])
    var lp = logits.ptr()
    var qp = vq.ptr()
    var kp = tk.ptr()
    var global_max = Float32(-3.0e38)
    for h in range(num_heads):
        var off = h * head_dim
        for s in range(s_len):
            var qrow = qp.unsafe_offset(s * embed_dim + off)
            var row = (h * s_len + s) * n_len
            for n in range(n_len):
                var v = _dot(qrow, kp.unsafe_offset(n * embed_dim + off), head_dim)
                lp[unsafe_offset=row + n] = v
                if v > global_max:
                    global_max = v

    # Vision->text softmax: subtract the global max, clamp, softmax over the text axis.
    var vision_probs = Tensor.zeros([num_heads * s_len, n_len])
    var vpp = vision_probs.ptr()
    for i in range(logits.numel()):
        var v = lp[unsafe_offset=i] - global_max
        if v < -BI_CLAMP:
            v = -BI_CLAMP
        elif v > BI_CLAMP:
            v = BI_CLAMP
        lp[unsafe_offset=i] = v
        vpp[unsafe_offset=i] = v
    softmax_rows_(vision_probs, n_len)

    # Text->vision softmax on the transposed logits, with its own per-row max.
    var text_probs = Tensor.zeros([num_heads * n_len, s_len])
    var tpp = text_probs.ptr()
    for h in range(num_heads):
        for n in range(n_len):
            var dst = (h * n_len + n) * s_len
            var mx = Float32(-3.0e38)
            for s in range(s_len):
                var v = lp[unsafe_offset=(h * s_len + s) * n_len + n]
                tpp[unsafe_offset=dst + s] = v
                if v > mx:
                    mx = v
            for s in range(s_len):
                var v = tpp[unsafe_offset=dst + s] - mx
                if v < -BI_CLAMP:
                    v = -BI_CLAMP
                elif v > BI_CLAMP:
                    v = BI_CLAMP
                tpp[unsafe_offset=dst + s] = v
    softmax_rows_(text_probs, s_len)

    var vision_ctx = Tensor.zeros([s_len, embed_dim])
    var vcp = vision_ctx.ptr()
    var tvp = tv.ptr()
    for h in range(num_heads):
        var off = h * head_dim
        for s in range(s_len):
            var dst = vcp.unsafe_offset(s * embed_dim + off)
            var row = (h * s_len + s) * n_len
            for n in range(n_len):
                _axpy(dst, tvp.unsafe_offset(n * embed_dim + off), vpp[unsafe_offset=row + n], head_dim)

    var text_ctx = Tensor.zeros([n_len, embed_dim])
    var tcp = text_ctx.ptr()
    var vvp = vv.ptr()
    for h in range(num_heads):
        var off = h * head_dim
        for n in range(n_len):
            var dst = tcp.unsafe_offset(n * embed_dim + off)
            var row = (h * n_len + n) * s_len
            for s in range(s_len):
                _axpy(dst, vvp.unsafe_offset(s * embed_dim + off), tpp[unsafe_offset=row + s], head_dim)

    var vision_out = matmul_nt(
        vision_ctx, weights[prefix + ".out_v.w"], weights[prefix + ".out_v.b"]
    )
    var text_out = matmul_nt(text_ctx, weights[prefix + ".out_t.w"], weights[prefix + ".out_t.b"])
    keep_alive(logits)
    keep_alive(vision_probs)
    keep_alive(text_probs)
    keep_alive(vq)
    keep_alive(tk)
    keep_alive(vv)
    keep_alive(tv)
    return (vision_out^, text_out^)


def fusion_layer(
    vision: Tensor, text: Tensor, weights: Dict[String, Tensor], prefix: String, cfg: Config
) raises -> Tuple[Tensor, Tensor]:
    """Pre-norm both streams, cross-attend, then add the layer-scaled deltas."""
    var v = layernorm_named(vision, weights, prefix + ".ln_v", cfg.layer_norm_eps)
    var t = layernorm_named(text, weights, prefix + ".ln_t", cfg.layer_norm_eps)
    var deltas = bi_attention(v, t, weights, prefix, cfg.encoder_heads // 2)
    add_scaled_(v, deltas[0], weights[prefix + ".vparam"])
    add_scaled_(t, deltas[1], weights[prefix + ".tparam"])
    return (v^, t^)


def text_enhancer(
    text: Tensor,
    position: Tensor,
    mask: Tensor,
    weights: Dict[String, Tensor],
    prefix: String,
    cfg: Config,
) raises -> Tensor:
    """Text self-attention over the phrase blocks, plus the half-width FFN."""
    var queries = Tensor(copy=text)
    add_(queries, position)
    var num_heads = cfg.encoder_heads // 2

    var q = matmul_nt(queries, weights[prefix + ".q.w"], weights[prefix + ".q.b"])
    var k = matmul_nt(queries, weights[prefix + ".k.w"], weights[prefix + ".k.b"])
    var v = matmul_nt(text, weights[prefix + ".v.w"], weights[prefix + ".v.b"])
    var ctx = attention_core(q, k, v, num_heads, mask, True)
    var attn = matmul_nt(ctx, weights[prefix + ".out.w"], weights[prefix + ".out.b"])

    add_(attn, text)
    var hidden = layernorm_named(attn, weights, prefix + ".ln_before", cfg.layer_norm_eps)

    var ffn = matmul_nt(hidden, weights[prefix + ".fc1.w"], weights[prefix + ".fc1.b"])
    relu_(ffn)
    var ffn2 = matmul_nt(ffn, weights[prefix + ".fc2.w"], weights[prefix + ".fc2.b"])
    add_(ffn2, hidden)
    return layernorm_named(ffn2, weights, prefix + ".ln_after", cfg.layer_norm_eps)


def deformable_layer(
    vision: Tensor,
    position: Tensor,
    reference: Tensor,
    features: MultiScaleFeatures,
    weights: Dict[String, Tensor],
    prefix: String,
    cfg: Config,
) raises -> Tensor:
    """Deformable self-attention over the vision levels, plus the FFN."""
    var attn = deformable_attention(
        vision, position, True, vision, reference,
        features.heights, features.widths, features.starts,
        weights, prefix, cfg.encoder_heads, cfg.encoder_n_points,
    )
    add_(attn, vision)
    var hidden = layernorm_named(attn, weights, prefix + ".ln1", cfg.layer_norm_eps)

    var ffn = matmul_nt(hidden, weights[prefix + ".fc1.w"], weights[prefix + ".fc1.b"])
    relu_(ffn)
    var ffn2 = matmul_nt(ffn, weights[prefix + ".fc2.w"], weights[prefix + ".fc2.b"])
    add_(ffn2, hidden)
    return layernorm_named(ffn2, weights, prefix + ".ln2", cfg.layer_norm_eps)


def forward(
    features: MultiScaleFeatures,
    text: Tensor,
    text_position_ids: Tensor,
    text_self_attn_mask: Tensor,
    weights: Dict[String, Tensor],
    cfg: Config,
) raises -> Tuple[Tensor, Tensor]:
    """Run the 6 encoder layers; returns `(vision (S, 256), text (N, 256))`."""
    var reference = reference_points(features)
    var n_len = text.dim(0)
    var coords = Tensor.zeros([n_len, 1])
    for i in range(n_len):
        coords.set2(i, 0, text_position_ids[i])
    var text_position = sinusoidal_embedding(coords, cfg.d_model, Float32(10000.0), True)

    var additive = Tensor.zeros([n_len, n_len])
    for i in range(additive.numel()):
        additive[i] = 0.0 if text_self_attn_mask[i] > 0.5 else MASK_NEG

    var vision_state = Tensor(copy=features.features)
    var text_state = Tensor(copy=text)
    for layer in range(cfg.encoder_layers):
        var prefix = "enc" + String(layer)
        var fused = fusion_layer(vision_state, text_state, weights, prefix + ".fu", cfg)
        vision_state = fused[0].copy()
        text_state = text_enhancer(
            fused[1], text_position, additive, weights, prefix + ".te", cfg
        )
        vision_state = deformable_layer(
            vision_state, features.position, reference, features,
            weights, prefix + ".df", cfg,
        )
    return (vision_state^, text_state^)
