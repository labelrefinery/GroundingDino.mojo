#!/usr/bin/env python3
"""Run grounding-dino-tiny end-to-end with `transformers`, for eyeballing against Mojo.

This writes nothing; it is the "what should the answer be" companion to
`pixi run mojo run -O3 -I src src/main.mojo ... --image photo.ppm --prompt '...'`.
Use `export_sample.py` when you want a parity fixture instead.

    PYTHONPATH=tools uv run --project tools python tools/reference.py \
        --image tools/images/excavator.jpg --prompt "excavator . crane . worker ."
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from lft import MODEL_ID


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=args.prompt, return_tensors="pt")

    start = time.perf_counter()
    outputs = model(**inputs)
    elapsed = time.perf_counter() - start

    result = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    size = tuple(inputs["pixel_values"].shape[-2:])
    print(f"{args.image} | {image.width} x {image.height} -> {size[1]} x {size[0]}")
    print(f"  prompt: {args.prompt!r}  forward: {elapsed:.2f} s")
    for score, box, label in zip(result["scores"], result["boxes"], result["text_labels"]):
        coords = ", ".join(f"{v:.1f}" for v in box.tolist())
        print(f"  {label:<20} score {score.item():.4f}  box [{coords}]")


if __name__ == "__main__":
    main()
