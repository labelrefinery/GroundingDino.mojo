# GroundingDino.mojo

[![CI](https://github.com/labelrefinery/GroundingDino.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/labelrefinery/GroundingDino.mojo/actions/workflows/ci.yml)

Pure-[Mojo](https://www.modular.com/mojo) inference implementation of **Grounding DINO**
(tiny) — *Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*
(Liu et al., ECCV 2024, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)), ported from
the `transformers` implementation of
[IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny).

Give it an image and a text prompt (`"excavator . crane . worker ."`) and it returns boxes,
scores and the phrase each box points at — no MAX, no Python interop at inference time, one
hand-written float32 tensor library, and numerical parity with PyTorch verified stage by
stage.

## Pipeline role

This is the **first stage of an auto-distillation / auto-labeling pipeline**: an
open-vocabulary 2D detector that turns a text prompt into boxes on camera frames, which
then bootstrap training data for the LiDAR side of the same project —
[CenterPillars](https://github.com/labelrefinery/CenterPillars.py) (detection) →
[OfflinePoly](https://github.com/labelrefinery/OfflinePoly.mojo) (offline tracking) →
[LabelFormer](https://github.com/labelrefinery/LabelFormer.mojo) (trajectory refinement).
Prompting for construction vocabulary (`excavator`, `crane`, `worker`) is exactly the case
a closed-vocabulary COCO detector cannot serve.

## Layout

- `src/groundingdino/` — the package:
  - `tensor` — owned f32 tensor plus the blocked SIMD matmul, LayerNorm, softmax, GELU
  - `io` — LFT1 weight/sample container reader and the model config
  - `tokenizer` — BERT uncased WordPiece and Grounding DINO's phrase-block text masks
  - `bert` — the 12-layer, d=768 text encoder and the 768→256 text projection
  - `attention` — scaled dot-product multi-head attention (BERT / text enhancer / decoder)
  - `swin` — Swin-T backbone: patch embedding, shifted-window attention, patch merging
  - `vision` — input projections + GroupNorm, sine position embeddings, level flattening
  - `deform` — multi-scale deformable attention with the bilinear `grid_sample` semantics
  - `encoder` — the 6-layer feature enhancer (fusion / text enhancer / deformable)
  - `decoder` — language-guided query selection and the 6-layer decoder with box refinement
  - `model` — the end-to-end forward pass, detection heads and post-processing
  - `image` — binary PPM reading, resizing and ImageNet normalization
- `src/main.mojo` — CLI: parity mode and standalone detection
- `tests/test_ops.mojo` — 26 hand-computed unit tests
- `tools/` — a `uv` project that exports the HF checkpoint and the parity fixtures

## Supported Mojo versions

Both **stable Mojo 1.0** and the **Modular nightly** are supported and tested in CI (unit
tests on Linux and macOS-arm64):

| pixi environment | compiler | run it |
|---|---|---|
| `default` | nightly (`modular` ≥ 26.6 nightly) | `pixi run test` / `pixi run infer` |
| `stable` | `mojo-compiler == 1.0.0` | `pixi run -e stable test` |

## Setup

Requires [pixi](https://pixi.sh) for Mojo and [uv](https://docs.astral.sh/uv/) for the
export tools. `pixi install` pulls the toolchain.

Export the weights, the tokenizer vocabulary and the parity fixtures (this downloads the
~700 MB checkpoint from HuggingFace; `data/` is gitignored):

```sh
uv sync --project tools
PYTHONPATH=tools uv run --project tools python tools/export_weights.py --out data
PYTHONPATH=tools uv run --project tools python tools/export_sample.py \
    --image tools/images/street.jpg    --prompt "car . person . truck ."      --index 0
PYTHONPATH=tools uv run --project tools python tools/export_sample.py \
    --image tools/images/excavator.jpg --prompt "excavator . crane . worker ." --index 1
PYTHONPATH=tools uv run --project tools python tools/export_sample.py \
    --image tools/images/cats.jpg      --prompt "a cat . a remote control ."   --index 2
```

`weights.lft` is 693 MB of fp32 in the LFT1 container (`b"LFT1" | u32 n | per tensor:
u32 name_len | name utf8 | u32 ndim | u32 shape[] | f32 C-order data`), the same format the
sibling LabelFormer.mojo uses. Two things happen at export time rather than in Mojo: Swin's
relative position bias is expanded from its `(169, heads)` table into a dense
`(heads, 49, 49)` bias, and every layer keeps its own name in a flat scheme decoupled from
the PyTorch module tree. Nothing is folded — Swin has no BatchNorm, and the input
projections' `GroupNorm(32)` normalizes per sample so it cannot be folded into the
preceding 1×1 convolution; it runs in Mojo instead.

## Run

```sh
pixi run test     # 26 op unit tests
pixi run infer    # replay the fixtures, print per-stage parity vs transformers
pixi run detect   # standalone detection on a PPM, writing detections.csv
```

Or directly:

```sh
# parity mode
pixi run mojo run -O3 -I src src/main.mojo data/weights.lft data/vocab.txt data/sample_0.lft

# standalone: PPM in, CSV out (columns: image,label,score,x1,y1,x2,y2)
pixi run mojo run -O3 -I src src/main.mojo data/weights.lft data/vocab.txt \
    --image photo.ppm --prompt "excavator . crane . person ." --csv out.csv \
    --box-threshold 0.35 --text-threshold 0.25
```

The standalone path takes a **binary PPM (P6)**; JPEG/PNG decoding is out of scope. Convert
with `convert photo.jpg photo.ppm`, `ffmpeg -i photo.jpg photo.ppm`, or
`python -c "from PIL import Image; Image.open('photo.jpg').convert('RGB').save('photo.ppm')"`.

Example, on the public-domain construction photo in `tools/images/`:

```
data/sample_1.ppm | 1280 x 730 | 2 detections in 10.3 s
   worker      score 0.730   box  27.2 617.7  45.9 664.5
   excavator   score 0.736   box 123.6 472.3 402.4 657.5
```

## Parity vs PyTorch

`pixi run infer` reports every stage and ends in a single `PARITY: PASS/FAIL` line. Worst
case over the three fixtures (COCO street scene at 800×1066, the construction photo at
760×1333, COCO cats at 800×1066):

| gate | stage | max abs diff | max abs reference | criterion |
|---|---|---|---|---|
| (a) | tokenizer ids, phrase mask, position ids | exact | — | identical |
| (b) | BERT last hidden state | 5.1e-6 | 3.6 | ≤ 1e-3 |
| (b) | text features (after the 768→256 projection) | 1.1e-4 | 133.2 | ≤ 1e-3 |
| (c) | Swin stage-2/3/4 features | 3.2e-4 | 16.7 | ≤ 1e-3 |
| (c) | the four projected d=256 levels | 3.2e-4 | 10.3 | ≤ 1e-3 |
| (d) | encoder vision output | 1.3e-5 | 1.4 | ≤ 1e-3 |
| (d) | encoder text output | 3.7e-5 | 4.5 | ≤ 1e-3 |
| (e) | decoder hidden, layer 1 of 6 | 3.7e-5 | 3.4 | ≤ 1e-3 |
| (e) | decoder hidden, layer 6 of 6 | 9.5e-3 | 3.4 | ≤ 1% of reference |
| (e) | decoder reference boxes, layer 6 | 7.9e-4 | 1.0 | ≤ 1e-3 |
| (e) | classification logits | 1.0e-2 | 9.5 | ≤ 1% of reference |
| (e) | predicted boxes | 5.8e-4 | 1.0 | ≤ 1e-3 |
| (f) | final detections | score 8.1e-4, box 0.10 px | — | same count and labels |

Two things in that table deserve explanation, because both are properties of the model
rather than of this port.

**The decoder amplifies float32 noise.** Gate (e) is measured with the decoder re-run from
the reference's encoder outputs and initial boxes, which isolates it. Its *first* layer
agrees to 3.7e-5 — the same order as every other stage — but each of the six iterative
refinement rounds feeds the updated boxes back into the deformable sampling locations, and
measured layer by layer that multiplies the error by roughly 2.5×: 3e-5 → 5e-5 → 4e-4 →
5e-4 → 2.5e-3 → 9.5e-3. Even at the last layer only ~0.3% of elements exceed 1e-3, and the
predicted boxes themselves stay under 1e-3, so the last layer is held to a relative
criterion and the per-element over-tolerance count is printed alongside.

**Query selection can swap a few rank slots.** Language-guided query selection ranks every
encoder pixel (~20k of them) by its best text logit and keeps the top 900. On these
fixtures the tightest gap inside the top 900 is ~1e-6 while float32 accumulation noise is
~2e-5, so 4–20 of the 900 queries land one slot away from where PyTorch put them. The
report prints how many queries agree; the detections are unaffected because they come from
the high-scoring end of the ranking.

**One quirk worth recording for anyone else porting this model.** In
`GroundingDinoEncoderLayer.get_text_position_embeddings`, the sinusoidal embedding of the
text position ids is computed in float and then run through
`.to(pos_tensor.dtype)` — and `pos_tensor` there is the *integer* position id tensor. Every
sin/cos value is therefore truncated toward zero to an integer, so the text enhancer sees an
almost entirely zero position embedding. Reproducing that truncation is required for parity;
without it the text stream diverges by ~0.5 from the first encoder layer onward.

## Timing

One image, single-threaded, Apple M4 (24 GB), `mojo run -O3`:

| image | preprocessed size | forward |
|---|---|---|
| COCO street scene | 800 × 1066 | 8.6 s |
| construction photo | 760 × 1333 | 10.4 s |
| COCO cats | 800 × 1066 | 8.7 s |

Loading `weights.lft` takes another ~0.3 s. Roughly 700 GFLOP go into one image, most of it
in the encoder's feed-forward networks and the vision/text fusion projections; the matmul is
a hand-blocked 6×4 micro-kernel vectorized over the contiguous reduction dimension and
sustains ~100 GFLOP/s on one M4 performance core. `parallelize` is **not** present in
`std.algorithm` on the Mojo nightly this repo targets (1.1.0.dev2026082807), so nothing here
is threaded — see the recommendations below.

## Limitations

- **Tiny only.** The Swin-B `grounding-dino-base` checkpoint uses the same architecture with
  different depths/heads/embed_dim, all of which already come from `__config__`; the export
  script should work unchanged, but it has not been run or verified.
- **PPM input.** JPEG/PNG decoding is out of scope for the standalone path. The Mojo-side
  resize is a plain `align_corners=False` bilinear resample, whereas the HF processor
  resamples through PIL/torchvision with antialiasing, so standalone scores differ from the
  fixture path in the third decimal (0.736 vs 0.725 on the construction photo). The parity
  fixtures carry the already-preprocessed `pixel_values` precisely so that this is not
  conflated with a porting error.
- **CPU, float32, batch size 1, single-threaded.** No image padding is supported, which is
  also why every key-padding mask reduces to a no-op here.
- **ASCII prompts.** The tokenizer implements the lowercase/punctuation/WordPiece path that
  Grounding DINO's prompt convention needs; accent stripping and the CJK spacing rule are
  not implemented.
- The parity CI job is manual-only (`workflow_dispatch`) because it needs the 693 MB export.

## Licenses

- This repository: MIT (see `LICENSE`).
- Upstream weights: `IDEA-Research/grounding-dino-tiny` is **Apache-2.0**; see its
  [model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny) for the training
  data and intended use. No weights are redistributed here — `tools/export_weights.py`
  downloads them from HuggingFace at export time.
- The port follows the `transformers` implementation
  (`transformers/models/grounding_dino/modeling_grounding_dino.py`, Apache-2.0).
- Sample images: see `tools/images/CREDITS.md`.
