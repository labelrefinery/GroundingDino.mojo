"""BERT-base text encoder (12 layers, d=768) plus Grounding DINO's text projection.

Matches `BertModel` in eval mode: word + position + token-type embeddings, LayerNorm
(eps 1e-12), then 12 post-norm transformer layers with exact-erf GELU. The attention
mask is the phrase-block mask from `tokenizer.text_masks`, converted to an additive
mask here.
"""

from .attention import MASK_NEG, multihead_attention
from .io import Config
from .tensor import Tensor, add_, gelu_, layernorm_named, matmul_nt


def embed(
    ids: List[Int], position_ids: Tensor, weights: Dict[String, Tensor], cfg: Config
) raises -> Tensor:
    """word_embeddings[ids] + position_embeddings[pos] + token_type_embeddings[0], normed."""
    var n = len(ids)
    var d = cfg.text_hidden
    var out = Tensor.zeros([n, d])
    ref word = weights["bert.word_emb"]
    ref pos = weights["bert.pos_emb"]
    ref tok = weights["bert.tok_emb"]
    for i in range(n):
        var pi = Int(position_ids[i])
        for j in range(d):
            out.set2(i, j, word.at2(ids[i], j) + pos.at2(pi, j) + tok.at2(0, j))
    return layernorm_named(out, weights, "bert.emb_norm", cfg.text_layer_norm_eps)


def additive_mask(mask: Tensor) raises -> Tensor:
    """Turn a 1.0/0.0 keep-mask into the additive mask attention expects."""
    var out = Tensor(mask.shape.copy())
    for i in range(mask.numel()):
        out[i] = 0.0 if mask[i] > 0.5 else MASK_NEG
    return out^


def forward(
    ids: List[Int],
    position_ids: Tensor,
    self_attn_mask: Tensor,
    weights: Dict[String, Tensor],
    cfg: Config,
) raises -> Tensor:
    """Run the text encoder; returns the last hidden state of shape `(N, 768)`."""
    var hidden = embed(ids, position_ids, weights, cfg)
    var mask = additive_mask(self_attn_mask)
    var eps = cfg.text_layer_norm_eps

    for layer in range(cfg.text_layers):
        var p = "bert.l" + String(layer)
        var attn = multihead_attention(
            hidden, hidden, hidden, weights, p, cfg.text_heads, mask, True
        )
        add_(attn, hidden)
        hidden = layernorm_named(attn, weights, p + ".attn_norm", eps)

        var inter = matmul_nt(hidden, weights[p + ".inter.w"], weights[p + ".inter.b"])
        gelu_(inter)
        var ffn = matmul_nt(inter, weights[p + ".ffn_out.w"], weights[p + ".ffn_out.b"])
        add_(ffn, hidden)
        hidden = layernorm_named(ffn, weights, p + ".ffn_norm", eps)
    return hidden^


def project(hidden: Tensor, weights: Dict[String, Tensor]) raises -> Tensor:
    """Grounding DINO's `text_projection`: 768 -> 256."""
    return matmul_nt(hidden, weights["text_proj.w"], weights["text_proj.b"])
