"""Language-guided query selection and the 6-layer decoder with iterative box refinement.

Two-stage query selection (`GroundingDinoModel.forward` with `two_stage=True`) turns
every encoder pixel into an anchor proposal, scores it against the text with the
contrastive head, and keeps the top `num_queries` by max text logit. The decoder then
runs self-attention, text cross-attention and deformable vision cross-attention, and
after each layer refines the reference boxes through a shared 3-layer MLP in
inverse-sigmoid space.
"""

from std.math import exp, log

from .attention import multihead_attention
from .deform import deformable_attention
from .encoder import sinusoidal_embedding
from .io import Config
from .tensor import (
    Tensor, add_, inverse_sigmoid, keep_alive, layernorm_named, matmul_nt, relu_, sigmoid
)
from .vision import MultiScaleFeatures

comptime NEG_INF = Float32(-3.4028235e38)
"""Stands in for `float("-inf")` in the contrastive head's padding to max_text_len."""

comptime LOGIT_EPS = Float32(1e-5)
"""`torch.special.logit(x, eps=1e-5)` clamp used by every reference-point update."""


def positive_infinity() -> Float32:
    return exp(Float32(1.0e30))


def mlp_head(
    x: Tensor, weights: Dict[String, Tensor], prefix: String, num_layers: Int
) raises -> Tensor:
    """`GroundingDinoMLPPredictionHead`: ReLU between layers, none after the last."""
    var h = Tensor(copy=x)
    for i in range(num_layers):
        var name = prefix + ".l" + String(i)
        var y = matmul_nt(h, weights[name + ".w"], weights[name + ".b"])
        if i < num_layers - 1:
            relu_(y)
        h = y^
    return h^


def contrastive_logits(
    queries: Tensor, text: Tensor, max_text_len: Int
) raises -> Tensor:
    """`GroundingDinoContrastiveEmbedding`: query . text^T, padded to `max_text_len`."""
    var q_len = queries.dim(0)
    var n_len = text.dim(0)
    var d = queries.dim(1)
    var out = Tensor.full([q_len, max_text_len], NEG_INF)
    for q in range(q_len):
        for n in range(n_len):
            var acc: Float32 = 0.0
            for c in range(d):
                acc += queries.at2(q, c) * text.at2(n, c)
            out.set2(q, n, acc)
    return out^


@fieldwise_init
struct QuerySelection(Copyable, Movable):
    """Output of language-guided query selection."""

    var target: Tensor
    """`(num_queries, d_model)` decoder input embeddings."""
    var reference: Tensor
    """`(num_queries, 4)` initial reference boxes in [0, 1] cxcywh."""


def select_queries(
    vision: Tensor,
    text: Tensor,
    features: MultiScaleFeatures,
    weights: Dict[String, Tensor],
    cfg: Config,
) raises -> QuerySelection:
    """Score every encoder pixel against the text and keep the top `num_queries`."""
    var total = vision.dim(0)
    var d = cfg.d_model
    var inf = positive_infinity()

    # Anchor proposals: a grid cell centre plus a level-dependent width/height.
    var proposals = Tensor.zeros([total, 4])
    var valid = List[Bool](length=total, fill=True)
    for l in range(features.num_levels()):
        var height = features.heights[l]
        var width = features.widths[l]
        var base = features.starts[l]
        var wh = Float32(0.05) * (Float32(2.0) ** Float32(l))
        for y in range(height):
            for x in range(width):
                var row = base + y * width + x
                var gx = (Float32(x) + 0.5) / Float32(width)
                var gy = (Float32(y) + 0.5) / Float32(height)
                proposals.set2(row, 0, gx)
                proposals.set2(row, 1, gy)
                proposals.set2(row, 2, wh)
                proposals.set2(row, 3, wh)
                var ok = True
                for c in range(4):
                    var v = proposals.at2(row, c)
                    if v <= 0.01 or v >= 0.99:
                        ok = False
                valid[row] = ok

    var coord_base = Tensor.zeros([total, 4])
    for row in range(total):
        for c in range(4):
            if valid[row]:
                var p = proposals.at2(row, c)
                coord_base.set2(row, c, Float32(log_ratio(p)))
            else:
                coord_base.set2(row, c, inf)

    var masked = Tensor.zeros([total, d])
    for row in range(total):
        if valid[row]:
            for c in range(d):
                masked.set2(row, c, vision.at2(row, c))

    var projected = matmul_nt(masked, weights["enc_output.w"], weights["enc_output.b"])
    var object_query = layernorm_named(
        projected, weights, "enc_output_norm", cfg.layer_norm_eps
    )

    var logits = contrastive_logits(object_query, text, cfg.max_text_len)
    var delta = mlp_head(object_query, weights, "enc_bbox", 3)

    # Top-k by the best text logit, descending (ties broken by index, like torch.topk).
    var best = Tensor.zeros([total])
    for row in range(total):
        var m = NEG_INF
        for c in range(cfg.max_text_len):
            var v = logits.at2(row, c)
            if v > m:
                m = v
        best[row] = m

    var chosen = List[Int]()
    var taken = List[Bool](length=total, fill=False)
    for _ in range(cfg.num_queries):
        var best_idx = -1
        var best_val = NEG_INF
        for row in range(total):
            if taken[row]:
                continue
            if best_idx < 0 or best[row] > best_val:
                best_idx = row
                best_val = best[row]
        taken[best_idx] = True
        chosen.append(best_idx)

    var reference = Tensor.zeros([cfg.num_queries, 4])
    for q in range(cfg.num_queries):
        var row = chosen[q]
        for c in range(4):
            reference.set2(q, c, sigmoid(delta.at2(row, c) + coord_base.at2(row, c)))

    var target = Tensor.zeros([cfg.num_queries, d])
    ref query_embed = weights["query_pos_emb"]
    for q in range(cfg.num_queries):
        for c in range(d):
            target.set2(q, c, query_embed.at2(q, c))
    return QuerySelection(target^, reference^)


def log_ratio(p: Float32) -> Float32:
    """Unclamped inverse sigmoid `log(p / (1 - p))`, as used for the anchor proposals."""
    return log(p / (1.0 - p))


@fieldwise_init
struct DecoderOutput(Copyable, Movable):
    """Stacked per-layer decoder states."""

    var hidden: List[Tensor]
    """`decoder_layers` tensors of `(num_queries, d_model)`, each LayerNorm'd."""
    var reference: List[Tensor]
    """`decoder_layers` tensors of `(num_queries, 4)`."""


def forward(
    target: Tensor,
    init_reference: Tensor,
    vision: Tensor,
    text: Tensor,
    features: MultiScaleFeatures,
    weights: Dict[String, Tensor],
    cfg: Config,
) raises -> DecoderOutput:
    """Run the decoder, returning every layer's normalized hidden state and boxes."""
    var num_queries = target.dim(0)
    var d = cfg.d_model
    var num_levels = features.num_levels()

    var hidden = Tensor(copy=target)
    var reference = Tensor(copy=init_reference)
    var hidden_states = List[Tensor]()
    var references = List[Tensor]()

    var empty = Tensor.zeros([1])
    for layer in range(cfg.decoder_layers):
        var prefix = "dec" + String(layer)

        # All valid ratios are 1, so the per-level reference boxes are simply repeated.
        var reference_input = Tensor.zeros([num_queries, num_levels, 4])
        for q in range(num_queries):
            for l in range(num_levels):
                for c in range(4):
                    reference_input.set3(q, l, c, reference.at2(q, c))

        var first_level = Tensor.zeros([num_queries, 4])
        for q in range(num_queries):
            for c in range(4):
                first_level.set2(q, c, reference_input.at3(q, 0, c))
        var query_pos = sinusoidal_embedding(first_level, d // 2, Float32(10000.0), False)
        query_pos = mlp_head(query_pos, weights, "ref_head", 2)

        var queries = Tensor(copy=hidden)
        add_(queries, query_pos)
        var self_attn = multihead_attention(
            queries, queries, hidden, weights, prefix + ".sa", cfg.decoder_heads, empty, False
        )
        add_(self_attn, hidden)
        hidden = layernorm_named(self_attn, weights, prefix + ".sa_ln", cfg.layer_norm_eps)

        var text_queries = Tensor(copy=hidden)
        add_(text_queries, query_pos)
        var text_attn = multihead_attention(
            text_queries, text, text, weights, prefix + ".ca_t", cfg.decoder_heads, empty, False
        )
        add_(text_attn, hidden)
        hidden = layernorm_named(text_attn, weights, prefix + ".ca_t_ln", cfg.layer_norm_eps)

        var cross = deformable_attention(
            hidden, query_pos, True, vision, reference_input,
            features.heights, features.widths, features.starts,
            weights, prefix + ".ca", cfg.decoder_heads, cfg.decoder_n_points,
        )
        add_(cross, hidden)
        hidden = layernorm_named(cross, weights, prefix + ".ca_ln", cfg.layer_norm_eps)

        var ffn = matmul_nt(hidden, weights[prefix + ".fc1.w"], weights[prefix + ".fc1.b"])
        relu_(ffn)
        var ffn2 = matmul_nt(ffn, weights[prefix + ".fc2.w"], weights[prefix + ".fc2.b"])
        add_(ffn2, hidden)
        hidden = layernorm_named(ffn2, weights, prefix + ".ln_final", cfg.layer_norm_eps)

        # Iterative box refinement in inverse-sigmoid space (weights are shared).
        var delta = mlp_head(hidden, weights, "bbox" + String(layer), 3)
        var next_reference = Tensor.zeros([num_queries, 4])
        for q in range(num_queries):
            for c in range(4):
                next_reference.set2(
                    q, c,
                    sigmoid(delta.at2(q, c) + inverse_sigmoid(reference.at2(q, c), LOGIT_EPS)),
                )
        reference = next_reference^

        hidden_states.append(
            layernorm_named(hidden, weights, "dec_ln", cfg.layer_norm_eps)
        )
        references.append(Tensor(copy=reference))
    keep_alive(empty)
    return DecoderOutput(hidden_states^, references^)
