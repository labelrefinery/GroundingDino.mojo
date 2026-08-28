#!/usr/bin/env python3
"""Export IDEA-Research/grounding-dino-tiny to the LFT1 container read by GroundingDino.mojo.

Everything is fp32 with flat names (see NAMING below), decoupled from the PyTorch
module tree.  Two transformations happen here rather than in Mojo:

* Swin's relative position bias is expanded from the ``(169, heads)`` table plus the
  static ``relative_position_index`` into a dense ``(heads, 49, 49)`` bias.
* Nothing else is folded: Swin has no BatchNorm, and the input projections' GroupNorm(32)
  cannot be folded into the preceding 1x1 conv (it normalizes per sample), so its affine
  parameters are exported as-is and GroupNorm runs in Mojo.

Usage:
    uv run --project tools python tools/export_weights.py --out data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from lft import MODEL_ID, write_lft

# NAMING -----------------------------------------------------------------------
# swin.patch.{w,b}                 patch embed conv 4x4 stride 4
# swin.embed_norm.{w,b}            LayerNorm after patch embed
# swin.s{S}.b{B}.{norm1,q,k,v,o,norm2,fc1,fc2}.{w,b}, swin.s{S}.b{B}.rpb (heads,49,49)
# swin.s{S}.down.{norm.{w,b},reduction.w}
# swin.out_norm{L}.{w,b}           hidden_states_norms for stage2/3/4 (L = 0..2)
# proj{L}.conv.{w,b}, proj{L}.gn.{w,b}     input_proj_vision (L = 0..3)
# level_embed                      (4, 256)
# bert.{word_emb,pos_emb,tok_emb}, bert.emb_norm.{w,b}
# bert.l{I}.{q,k,v,attn_out,attn_norm,inter,out,out_norm}.{w,b}
# text_proj.{w,b}
# enc{I}.te.{q,k,v,out,fc1,fc2,ln_before,ln_after}.{w,b}
# enc{I}.fu.{ln_v,ln_t,vision_proj,text_proj,vv_proj,vt_proj,out_v,out_t}.{w,b}
# enc{I}.fu.{vparam,tparam}
# enc{I}.df.{samp,attnw,value,outp,ln1,fc1,fc2,ln2}.{w,b}
# enc_output.{w,b}, enc_output_norm.{w,b}, enc_bbox.l{K}.{w,b}
# query_pos_emb                    (900, 256)
# dec{I}.sa.{q,k,v,out}.{w,b}, dec{I}.sa_ln.{w,b}
# dec{I}.ca_t.{q,k,v,out}.{w,b}, dec{I}.ca_t_ln.{w,b}
# dec{I}.ca.{samp,attnw,value,outp}.{w,b}, dec{I}.ca_ln.{w,b}
# dec{I}.{fc1,fc2,ln_final}.{w,b}
# dec_ln.{w,b}, ref_head.l{0,1}.{w,b}, bbox{I}.l{K}.{w,b}
# ------------------------------------------------------------------------------

WINDOW_SIZE = 7


def relative_position_index(window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Mirror of SwinRelativePositionBias._create_relative_position_index."""
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1
    return rel.sum(-1).numpy()


class Exporter:
    def __init__(self, model):
        self.model = model
        self.out: dict[str, np.ndarray] = {}
        self.rp_index = relative_position_index()

    def _np(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy()

    def put(self, name: str, tensor: torch.Tensor) -> None:
        self.out[name] = self._np(tensor)

    def lin(self, name: str, layer) -> None:
        self.put(f"{name}.w", layer.weight)
        if layer.bias is not None:
            self.put(f"{name}.b", layer.bias)

    def norm(self, name: str, layer) -> None:
        self.put(f"{name}.w", layer.weight)
        self.put(f"{name}.b", layer.bias)

    # -- pieces ---------------------------------------------------------------
    def swin(self, swin_backbone) -> None:
        swin = swin_backbone.swin
        self.put("swin.patch.w", swin.embeddings.patch_embeddings.projection.weight)
        self.put("swin.patch.b", swin.embeddings.patch_embeddings.projection.bias)
        self.norm("swin.embed_norm", swin.embeddings.norm)

        for s, stage in enumerate(swin.encoder.layers):
            for b, block in enumerate(stage.blocks):
                p = f"swin.s{s}.b{b}"
                self.norm(f"{p}.norm1", block.layernorm_before)
                self.lin(f"{p}.q", block.attention.q_proj)
                self.lin(f"{p}.k", block.attention.k_proj)
                self.lin(f"{p}.v", block.attention.v_proj)
                self.lin(f"{p}.o", block.attention.o_proj)
                table = self._np(block.attention.relative_position_bias.relative_position_bias_table)
                # (49*49, heads) -> (heads, 49, 49)
                bias = table[self.rp_index.reshape(-1)]
                area = WINDOW_SIZE * WINDOW_SIZE
                self.out[f"{p}.rpb"] = np.ascontiguousarray(
                    bias.reshape(area, area, -1).transpose(2, 0, 1)
                )
                self.norm(f"{p}.norm2", block.layernorm_after)
                self.lin(f"{p}.fc1", block.mlp.fc1)
                self.lin(f"{p}.fc2", block.mlp.fc2)
            if stage.downsample is not None:
                self.norm(f"swin.s{s}.down.norm", stage.downsample.norm)
                self.put(f"swin.s{s}.down.reduction.w", stage.downsample.reduction.weight)

        for level, stage_name in enumerate(swin_backbone.out_features):
            self.norm(f"swin.out_norm{level}", swin_backbone.hidden_states_norms[stage_name])

    def bert(self, text_backbone) -> None:
        emb = text_backbone.embeddings
        self.put("bert.word_emb", emb.word_embeddings.weight)
        self.put("bert.pos_emb", emb.position_embeddings.weight)
        self.put("bert.tok_emb", emb.token_type_embeddings.weight)
        self.norm("bert.emb_norm", emb.LayerNorm)
        for i, layer in enumerate(text_backbone.encoder.layer):
            p = f"bert.l{i}"
            self.lin(f"{p}.q", layer.attention.self.query)
            self.lin(f"{p}.k", layer.attention.self.key)
            self.lin(f"{p}.v", layer.attention.self.value)
            self.lin(f"{p}.attn_out", layer.attention.output.dense)
            self.norm(f"{p}.attn_norm", layer.attention.output.LayerNorm)
            self.lin(f"{p}.inter", layer.intermediate.dense)
            self.lin(f"{p}.out", layer.output.dense)
            self.norm(f"{p}.out_norm", layer.output.LayerNorm)

    def encoder(self, encoder) -> None:
        for i, layer in enumerate(encoder.layers):
            te = layer.text_enhancer_layer
            p = f"enc{i}.te"
            self.lin(f"{p}.q", te.self_attn.query)
            self.lin(f"{p}.k", te.self_attn.key)
            self.lin(f"{p}.v", te.self_attn.value)
            self.lin(f"{p}.out", te.self_attn.out_proj)
            self.lin(f"{p}.fc1", te.fc1)
            self.lin(f"{p}.fc2", te.fc2)
            self.norm(f"{p}.ln_before", te.layer_norm_before)
            self.norm(f"{p}.ln_after", te.layer_norm_after)

            fu = layer.fusion_layer
            p = f"enc{i}.fu"
            self.norm(f"{p}.ln_v", fu.layer_norm_vision)
            self.norm(f"{p}.ln_t", fu.layer_norm_text)
            self.put(f"{p}.vparam", fu.vision_param)
            self.put(f"{p}.tparam", fu.text_param)
            self.lin(f"{p}.vision_proj", fu.attn.vision_proj)
            self.lin(f"{p}.text_proj", fu.attn.text_proj)
            self.lin(f"{p}.vv_proj", fu.attn.values_vision_proj)
            self.lin(f"{p}.vt_proj", fu.attn.values_text_proj)
            self.lin(f"{p}.out_v", fu.attn.out_vision_proj)
            self.lin(f"{p}.out_t", fu.attn.out_text_proj)

            df = layer.deformable_layer
            p = f"enc{i}.df"
            self.lin(f"{p}.samp", df.self_attn.sampling_offsets)
            self.lin(f"{p}.attnw", df.self_attn.attention_weights)
            self.lin(f"{p}.value", df.self_attn.value_proj)
            self.lin(f"{p}.outp", df.self_attn.output_proj)
            self.norm(f"{p}.ln1", df.self_attn_layer_norm)
            self.lin(f"{p}.fc1", df.fc1)
            self.lin(f"{p}.fc2", df.fc2)
            self.norm(f"{p}.ln2", df.final_layer_norm)

    def decoder(self, decoder) -> None:
        self.norm("dec_ln", decoder.layer_norm)
        for k, layer in enumerate(decoder.reference_points_head.layers):
            self.lin(f"ref_head.l{k}", layer)
        for i, layer in enumerate(decoder.layers):
            p = f"dec{i}"
            self.lin(f"{p}.sa.q", layer.self_attn.query)
            self.lin(f"{p}.sa.k", layer.self_attn.key)
            self.lin(f"{p}.sa.v", layer.self_attn.value)
            self.lin(f"{p}.sa.out", layer.self_attn.out_proj)
            self.norm(f"{p}.sa_ln", layer.self_attn_layer_norm)
            self.lin(f"{p}.ca_t.q", layer.encoder_attn_text.query)
            self.lin(f"{p}.ca_t.k", layer.encoder_attn_text.key)
            self.lin(f"{p}.ca_t.v", layer.encoder_attn_text.value)
            self.lin(f"{p}.ca_t.out", layer.encoder_attn_text.out_proj)
            self.norm(f"{p}.ca_t_ln", layer.encoder_attn_text_layer_norm)
            self.lin(f"{p}.ca.samp", layer.encoder_attn.sampling_offsets)
            self.lin(f"{p}.ca.attnw", layer.encoder_attn.attention_weights)
            self.lin(f"{p}.ca.value", layer.encoder_attn.value_proj)
            self.lin(f"{p}.ca.outp", layer.encoder_attn.output_proj)
            self.norm(f"{p}.ca_ln", layer.encoder_attn_layer_norm)
            self.lin(f"{p}.fc1", layer.fc1)
            self.lin(f"{p}.fc2", layer.fc2)
            self.norm(f"{p}.ln_final", layer.final_layer_norm)

    def run(self) -> dict[str, np.ndarray]:
        gd = self.model.model
        self.swin(gd.backbone.conv_encoder.model)
        for level, seq in enumerate(gd.input_proj_vision):
            self.lin(f"proj{level}.conv", seq[0])
            self.norm(f"proj{level}.gn", seq[1])
        self.put("level_embed", gd.level_embed)
        self.bert(gd.text_backbone)
        self.lin("text_proj", gd.text_projection)
        self.encoder(gd.encoder)
        self.lin("enc_output", gd.enc_output)
        self.norm("enc_output_norm", gd.enc_output_norm)
        for k, layer in enumerate(gd.encoder_output_bbox_embed.layers):
            self.lin(f"enc_bbox.l{k}", layer)
        self.put("query_pos_emb", gd.query_position_embeddings.weight)
        self.decoder(gd.decoder)
        for i, head in enumerate(self.model.bbox_embed):
            for k, layer in enumerate(head.layers):
                self.lin(f"bbox{i}.l{k}", layer)

        cfg = self.model.config
        self.out["__config__"] = np.array(
            [
                cfg.d_model,
                cfg.encoder_layers,
                cfg.decoder_layers,
                cfg.encoder_attention_heads,
                cfg.decoder_attention_heads,
                cfg.encoder_ffn_dim,
                cfg.decoder_ffn_dim,
                cfg.encoder_n_points,
                cfg.decoder_n_points,
                cfg.num_feature_levels,
                cfg.num_queries,
                cfg.max_text_len,
                cfg.positional_embedding_temperature,
                cfg.layer_norm_eps,
                cfg.text_config.hidden_size,
                cfg.text_config.num_hidden_layers,
                cfg.text_config.num_attention_heads,
                cfg.text_config.layer_norm_eps,
                cfg.backbone_config.embed_dim,
                WINDOW_SIZE,
            ],
            dtype=np.float32,
        )
        # Swin depths and per-stage head counts as their own tensors (variable length).
        self.out["__swin_depths__"] = np.array(cfg.backbone_config.depths, dtype=np.float32)
        self.out["__swin_heads__"] = np.array(cfg.backbone_config.num_heads, dtype=np.float32)
        return self.out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = Exporter(model).run()
    total = write_lft(args.out / "weights.lft", tensors)
    print(f"weights.lft: {len(tensors)} tensors, {total / 1e6:.2f}M values ({total * 4 / 1e6:.0f} MB)")

    processor = AutoProcessor.from_pretrained(args.model_id)
    vocab_src = Path(processor.tokenizer.vocab_file)
    shutil.copyfile(vocab_src, args.out / "vocab.txt")
    print(f"vocab.txt: {sum(1 for _ in open(args.out / 'vocab.txt', encoding='utf-8'))} tokens")


if __name__ == "__main__":
    main()
