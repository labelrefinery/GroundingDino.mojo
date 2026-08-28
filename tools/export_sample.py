#!/usr/bin/env python3
"""Export one (image, prompt) pair plus every reference intermediate to an LFT1 sample.

The Mojo side loads the sample and compares stage by stage; because the preprocessed
`pixel_values` and `input_ids` travel inside the sample, parity is measured on exactly
the tensors transformers used (the Mojo PPM reader / resizer is exercised separately in
standalone mode).

Usage:
    uv run --project tools python tools/export_sample.py \
        --image photo.jpg --prompt "car . person . truck ." --index 0 --out data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lft import MODEL_ID, write_lft

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


def save_ppm(image: Image.Image, path: Path) -> None:
    """Write a binary P6 PPM — the standalone input format of the Mojo CLI."""
    image.convert("RGB").save(path, format="PPM")


@torch.no_grad()
def export(
    image_path: Path,
    prompt: str,
    out: Path,
    index: int,
    model_id: str,
    shortest_edge: int = 800,
    longest_edge: int = 1333,
) -> None:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers.models.grounding_dino.modeling_grounding_dino import (
        generate_masks_with_special_tokens_and_transfer_map,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.size = {"shortest_edge": shortest_edge, "longest_edge": longest_edge}
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    gd = model.model
    captured: dict[str, object] = {}

    def hook(name):
        def fn(module, args, output):
            captured[name] = output

        return fn

    handles = [
        gd.text_backbone.register_forward_hook(hook("bert")),
        gd.text_projection.register_forward_hook(hook("text_features")),
        gd.backbone.conv_encoder.register_forward_hook(hook("swin")),
        gd.encoder.register_forward_hook(hook("encoder")),
        gd.decoder.register_forward_hook(hook("decoder")),
    ]
    for level, seq in enumerate(gd.input_proj_vision):
        handles.append(seq.register_forward_hook(hook(f"proj{level}")))

    outputs = model(**inputs)
    for h in handles:
        h.remove()

    input_ids = inputs["input_ids"]
    self_attn_mask, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)

    def f32(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.asarray(t, dtype=np.float32)

    def sq(t) -> np.ndarray:
        return f32(t)[0]

    tensors: dict[str, np.ndarray] = {
        "pixel_values": sq(inputs["pixel_values"]),
        "input_ids": sq(input_ids),
        "token_type_ids": sq(inputs["token_type_ids"]),
        "attention_mask": sq(inputs["attention_mask"]),
        "text_self_attn_mask": sq(self_attn_mask),
        "position_ids": sq(position_ids),
        "bert_hidden": sq(captured["bert"].last_hidden_state),
        "text_features": sq(captured["text_features"]),
        "enc_vision": sq(captured["encoder"].last_hidden_state_vision),
        "enc_text": sq(captured["encoder"].last_hidden_state_text),
        "dec_hidden": sq(outputs.intermediate_hidden_states),
        "dec_refpoints": sq(outputs.intermediate_reference_points),
        "init_ref": sq(outputs.init_reference_points),
        "logits": sq(outputs.logits),
        "pred_boxes": sq(outputs.pred_boxes),
        "target_size": np.array([image.height, image.width], dtype=np.float32),
        "__thresholds__": np.array([BOX_THRESHOLD, TEXT_THRESHOLD], dtype=np.float32),
    }
    for level, (feature_map, _mask) in enumerate(captured["swin"]):
        tensors[f"swin_feat{level}"] = sq(feature_map)
    for level in range(len(gd.input_proj_vision)):
        tensors[f"proj{level}"] = sq(captured[f"proj{level}"])

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    tensors["det_boxes"] = f32(results["boxes"]).reshape(-1, 4)
    tensors["det_scores"] = f32(results["scores"]).reshape(-1)

    probs = torch.sigmoid(outputs.logits)[0]
    scores = probs.max(-1)[0]
    keep = scores > BOX_THRESHOLD
    posmap = (probs[keep] > TEXT_THRESHOLD)[:, : input_ids.shape[1]]
    tensors["det_posmap"] = f32(posmap).reshape(-1, input_ids.shape[1])

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sample_{index}.lft"
    total = write_lft(path, tensors)
    save_ppm(image, out / f"sample_{index}.ppm")
    (out / f"sample_{index}.txt").write_text(prompt + "\n", encoding="utf-8")

    labels = results["text_labels"]
    (out / f"sample_{index}.labels").write_text("|".join(labels) + "\n", encoding="utf-8")
    print(f"{path.name}: {image.width}x{image.height} -> {tuple(tensors['pixel_values'].shape)}")
    print(f"  prompt: {prompt!r}  tokens: {input_ids.shape[1]}  values: {total / 1e6:.2f}M")
    print(f"  detections ({len(labels)}):")
    for score, box, label in zip(tensors["det_scores"], tensors["det_boxes"], labels):
        print(f"    {label:<20} {score:.3f}  [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--shortest-edge", type=int, default=800)
    ap.add_argument("--longest-edge", type=int, default=1333)
    args = ap.parse_args()
    export(
        args.image,
        args.prompt,
        args.out,
        args.index,
        args.model_id,
        args.shortest_edge,
        args.longest_edge,
    )


if __name__ == "__main__":
    main()
