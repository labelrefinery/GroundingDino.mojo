#!/usr/bin/env python3
"""Dump per-sublayer encoder/decoder intermediates for one sample, to localize divergence.

Writes `data/debug.lft` alongside `data/sample_N.lft`; only used while porting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lft import MODEL_ID, write_lft


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--shortest-edge", type=int, default=256)
    ap.add_argument("--longest-edge", type=int, default=400)
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.size = {
        "shortest_edge": args.shortest_edge,
        "longest_edge": args.longest_edge,
    }
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=args.prompt, return_tensors="pt")

    captured: dict[str, np.ndarray] = {}

    def grab(name, pick=None):
        def fn(module, a, output):
            out = output if pick is None else pick(output)
            if isinstance(out, torch.Tensor):
                captured[name] = out.detach().cpu().float().numpy()[0]
            else:
                for k, v in enumerate(out):
                    if isinstance(v, torch.Tensor):
                        captured[f"{name}_{k}"] = v.detach().cpu().float().numpy()[0]

        return fn

    handles = []
    gd = model.model
    for i, layer in enumerate(gd.encoder.layers):
        handles.append(
            layer.fusion_layer.register_forward_hook(
                grab(f"fu{i}", lambda o: (o[0][0], o[1][0]))
            )
        )
        handles.append(
            layer.text_enhancer_layer.register_forward_hook(grab(f"te{i}", lambda o: o[0]))
        )
        handles.append(
            layer.deformable_layer.register_forward_hook(grab(f"df{i}", lambda o: o[0]))
        )
    for i, layer in enumerate(gd.decoder.layers):
        handles.append(layer.register_forward_hook(grab(f"dec{i}", lambda o: o[0])))

    model(**inputs)
    for h in handles:
        h.remove()

    args.out.mkdir(parents=True, exist_ok=True)
    total = write_lft(args.out / "debug.lft", captured)
    print(f"debug.lft: {len(captured)} tensors, {total / 1e6:.2f}M values")
    for k in sorted(captured):
        print(f"  {k}: {captured[k].shape}")


if __name__ == "__main__":
    main()
