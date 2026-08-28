"""GroundingDino.mojo CLI: stage-by-stage parity mode and standalone detection.

Parity mode (the default) replays an exported fixture and prints the max absolute
difference against the transformers reference for every stage, ending in a single
`PARITY: PASS/FAIL` line:

    pixi run mojo run -I src src/main.mojo data/weights.lft data/vocab.txt data/sample_0.lft

Standalone mode reads a binary PPM, tokenizes the prompt itself, and writes detections:

    pixi run mojo run -I src src/main.mojo data/weights.lft data/vocab.txt \
        --image photo.ppm --prompt "excavator . crane . person ." --csv out.csv
"""

from std.sys import argv
from std.time import perf_counter_ns

from groundingdino.image import preprocess
from groundingdino.io import Config, decode_config, load_lft
from groundingdino.model import Detection, decode_and_predict, forward, postprocess
from groundingdino.tensor import Tensor, max_abs, max_abs_diff
from groundingdino.tokenizer import load_vocab, load_vocab_tokens, text_masks, tokenize

comptime SHORTEST_EDGE = 800
comptime LONGEST_EDGE = 1333
comptime TOLERANCE = Float32(1.0e-3)
"""Absolute tolerance for a tensor stage to count as passing (differences are reported
alongside the magnitude of the reference tensor so they can be read relatively)."""

comptime RELATIVE_TOLERANCE = Float32(1.0e-2)
"""Criterion for the decoder's final layer, whose iterative box refinement amplifies
float32 noise by roughly 2.5x per layer (measured layer by layer on the fixtures)."""

comptime SCORE_TOLERANCE = Float32(5.0e-3)
"""Tolerance on a detection's confidence."""

comptime BOX_TOLERANCE = Float32(1.0)
"""Tolerance on a detection's box corners, in pixels of the original image."""


def report(name: String, got: Tensor, want: Tensor, mut all_pass: Bool) raises:
    var diff = max_abs_diff(got, want)
    var magnitude = max_abs(want)
    var ok = diff <= TOLERANCE
    if not ok:
        all_pass = False
    print("  ", name, "max|diff| =", diff, " max|ref| =", magnitude, " ", "PASS" if ok else "FAIL")


def report_slice(
    name: String,
    got: Tensor,
    want: Tensor,
    rows: Int,
    cols: Int,
    stride: Int,
    offset: Int,
    relative: Bool,
    mut all_pass: Bool,
) raises:
    """Compare the leading `(rows, cols)` block of a flat reference tensor.

    `relative` switches the criterion from an absolute 1e-3 to 1% of the reference
    magnitude, which is what the decoder's *final* layer is held to -- see the comment
    at the gate (e) call sites.
    """
    var diff = Float32(0.0)
    var magnitude = Float32(0.0)
    var over = 0
    for r in range(rows):
        for c in range(cols):
            var w = want[offset + r * stride + c]
            var e = got.at2(r, c) - w
            if e < 0.0:
                e = -e
            if e > diff:
                diff = e
            if e > TOLERANCE:
                over += 1
            var a = w
            if a < 0.0:
                a = -a
            if a > magnitude:
                magnitude = a
    var limit = RELATIVE_TOLERANCE * magnitude if relative else TOLERANCE
    var ok = diff <= limit
    if not ok:
        all_pass = False
    print(
        "  ", name, "max|diff| =", diff, " max|ref| =", magnitude,
        " over-tol elements:", over, "of", rows * cols, " ", "PASS" if ok else "FAIL",
    )


def flatten_chw(want: Tensor) raises -> Tensor:
    """Reshape a reference `(C, H, W)` map into the `(H * W, C)` layout used here."""
    var channels = want.dim(0)
    var height = want.dim(1)
    var width = want.dim(2)
    var out = Tensor.zeros([height * width, channels])
    for c in range(channels):
        for y in range(height):
            for x in range(width):
                out.set2(y * width + x, c, want.at3(c, y, x))
    return out^


def run_parity(
    weights: Dict[String, Tensor],
    cfg: Config,
    vocab_path: String,
    sample_path: String,
    mut all_pass: Bool,
) raises:
    var sample = load_lft(sample_path)
    var prompt_path = String(sample_path.removesuffix(".lft")) + ".txt"
    var f = open(prompt_path, "r")
    var prompt = String(f.read().strip())
    f.close()

    var vocab = load_vocab(vocab_path)
    var vocab_tokens = load_vocab_tokens(vocab_path)
    var ids = tokenize(prompt, vocab)

    print(sample_path, "| prompt:", prompt)
    ref reference_ids = sample["input_ids"]
    var ids_ok = len(ids) == reference_ids.numel()
    if ids_ok:
        for i in range(len(ids)):
            if ids[i] != Int(reference_ids[i]):
                ids_ok = False
    if not ids_ok:
        all_pass = False
    print("   (a) tokenizer ids  ", len(ids), "tokens  ", "PASS" if ids_ok else "FAIL")

    var masks = text_masks(ids)
    report("(a) text self-attn mask", masks[0], sample["text_self_attn_mask"], all_pass)
    report("(a) text position ids ", masks[1], sample["position_ids"], all_pass)

    var start = perf_counter_ns()
    var out = forward(sample["pixel_values"], ids, weights, cfg)
    var elapsed = Float64(perf_counter_ns() - start) / 1e9

    report("(b) bert_hidden       ", out.bert_hidden, sample["bert_hidden"], all_pass)
    report("(b) text_features     ", out.text_features, sample["text_features"], all_pass)
    for l in range(len(out.levels)):
        report(
            "(c) swin_feat" + String(l) + "        ",
            out.levels[l].data,
            flatten_chw(sample["swin_feat" + String(l)]),
            all_pass,
        )
    for l in range(cfg.num_levels):
        var height = out.features.heights[l]
        var width = out.features.widths[l]
        var got = Tensor.zeros([height * width, cfg.d_model])
        for r in range(height * width):
            for c in range(cfg.d_model):
                got.set2(r, c, out.features.features.at2(out.features.starts[l] + r, c))
        report(
            "(c) proj" + String(l) + "            ",
            got,
            flatten_chw(sample["proj" + String(l)]),
            all_pass,
        )
    report("(d) enc_vision        ", out.enc_vision, sample["enc_vision"], all_pass)
    report("(d) enc_text          ", out.enc_text, sample["enc_text"], all_pass)

    # Gate (e) isolates the decoder: it is re-run from the reference's encoder outputs
    # and initial boxes. Layer 0 shows the implementation error; by layer 6 the
    # iterative box refinement has amplified it ~300x (each refinement moves the
    # deformable sampling locations, which moves the next hidden state), so the last
    # layer is held to a relative criterion instead.
    var num_queries = cfg.num_queries
    ref want_init = sample["init_ref"]
    var swapped = 0
    for q in range(num_queries):
        var differs = False
        for c in range(4):
            var e = out.init_reference.at2(q, c) - want_init.at2(q, c)
            if e < 0.0:
                e = -e
            if e > TOLERANCE:
                differs = True
        if differs:
            swapped += 1
    print(
        "   (e) query selection    ", num_queries - swapped, "of", num_queries,
        "queries in the same rank slot",
    )

    var pinned = decode_and_predict(
        out.target, want_init, sample["enc_vision"], sample["enc_text"],
        out.features, weights, cfg,
    )
    ref want_hidden = sample["dec_hidden"]
    ref want_reference = sample["dec_refpoints"]
    var last = cfg.decoder_layers - 1
    report_slice(
        "(e) dec_hidden[0]     ", pinned.decoder.hidden[0], want_hidden,
        num_queries, cfg.d_model, cfg.d_model, 0, False, all_pass,
    )
    report_slice(
        "(e) dec_hidden[last]  ", pinned.decoder.hidden[last], want_hidden,
        num_queries, cfg.d_model, cfg.d_model, last * num_queries * cfg.d_model,
        True, all_pass,
    )
    report_slice(
        "(e) dec_reference[last]", pinned.decoder.reference[last], want_reference,
        num_queries, 4, 4, last * num_queries * 4, False, all_pass,
    )
    # The logits are padded to max_text_len with -inf, so only the real tokens compare.
    report_slice(
        "(e) logits            ", pinned.logits, sample["logits"],
        num_queries, len(ids), cfg.max_text_len, 0, True, all_pass,
    )
    report_slice(
        "(e) pred_boxes        ", pinned.boxes, sample["pred_boxes"],
        num_queries, 4, 4, 0, False, all_pass,
    )

    ref thresholds = sample["__thresholds__"]
    var detections = postprocess(
        out.logits, out.pred_boxes, ids, vocab_tokens,
        Int(sample["target_size"][0]), Int(sample["target_size"][1]),
        thresholds[0], thresholds[1],
    )
    ref want_boxes = sample["det_boxes"]
    ref want_scores = sample["det_scores"]
    var count_ok = len(detections) == want_scores.numel()
    var score_diff = Float32(0.0)
    var box_diff = Float32(0.0)
    if count_ok:
        for i in range(len(detections)):
            var e = detections[i].score - want_scores[i]
            if e < 0.0:
                e = -e
            if e > score_diff:
                score_diff = e
            var coords = [
                detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2
            ]
            for c in range(4):
                var b = coords[c] - want_boxes.at2(i, c)
                if b < 0.0:
                    b = -b
                if b > box_diff:
                    box_diff = b
    var labels_ok = True
    var labels_path = String(sample_path.removesuffix(".lft")) + ".labels"
    try:
        var lf = open(labels_path, "r")
        var joined = String(lf.read().strip())
        lf.close()
        var want_labels = joined.split("|")
        labels_ok = len(want_labels) == len(detections)
        if labels_ok:
            for i in range(len(detections)):
                if detections[i].label != String(want_labels[i]):
                    labels_ok = False
    except:
        print("       (no", labels_path, "-- label comparison skipped)")

    var detections_ok = (
        count_ok
        and labels_ok
        and score_diff <= SCORE_TOLERANCE
        and box_diff <= BOX_TOLERANCE
    )
    if not detections_ok:
        all_pass = False
    print(
        "   (f) detections        ", len(detections), "of", want_scores.numel(),
        " labels", "match" if labels_ok else "DIFFER",
        " max|score diff| =", score_diff, " max|box diff| =", box_diff, "px  ",
        "PASS" if detections_ok else "FAIL",
    )
    for d in detections:
        print("        ", d.label, "  score", d.score, "  box", d.x1, d.y1, d.x2, d.y2)
    print("   forward time:", elapsed, "s")


def run_standalone(
    weights: Dict[String, Tensor],
    cfg: Config,
    vocab_path: String,
    image_path: String,
    prompt: String,
    csv_path: String,
    box_threshold: Float32,
    text_threshold: Float32,
) raises:
    var vocab = load_vocab(vocab_path)
    var vocab_tokens = load_vocab_tokens(vocab_path)
    var ids = tokenize(prompt, vocab)
    var prepared = preprocess(image_path, SHORTEST_EDGE, LONGEST_EDGE)

    var start = perf_counter_ns()
    var out = forward(prepared[0], ids, weights, cfg)
    var elapsed = Float64(perf_counter_ns() - start) / 1e9
    var detections = postprocess(
        out.logits, out.pred_boxes, ids, vocab_tokens,
        prepared[1], prepared[2], box_threshold, text_threshold,
    )

    print(image_path, "|", prepared[2], "x", prepared[1], "|", len(detections), "detections in", elapsed, "s")
    for d in detections:
        print("  ", d.label, "  score", d.score, "  box", d.x1, d.y1, d.x2, d.y2)

    if csv_path != "":
        var rows = String("image,label,score,x1,y1,x2,y2\n")
        for d in detections:
            rows += (
                image_path + "," + d.label + "," + String(d.score) + ","
                + String(d.x1) + "," + String(d.y1) + "," + String(d.x2) + "," + String(d.y2) + "\n"
            )
        var f = open(csv_path, "w")
        f.write(rows)
        f.close()
        print("wrote", csv_path)


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("usage: mojo run -I src src/main.mojo <weights.lft> <vocab.txt> <sample.lft>...")
        print("       mojo run -I src src/main.mojo <weights.lft> <vocab.txt> \\")
        print("           --image photo.ppm --prompt 'car . person .' [--csv out.csv]")
        return

    var weights_path = String(args[1])
    var vocab_path = String(args[2])

    var samples = List[String]()
    var image_path = String("")
    var prompt = String("")
    var csv_path = String("")
    var box_threshold = Float32(0.35)
    var text_threshold = Float32(0.25)
    var i = 3
    while i < len(args):
        var a = String(args[i])
        if a == "--image" and i + 1 < len(args):
            image_path = String(args[i + 1])
            i += 2
        elif a == "--prompt" and i + 1 < len(args):
            prompt = String(args[i + 1])
            i += 2
        elif a == "--csv" and i + 1 < len(args):
            csv_path = String(args[i + 1])
            i += 2
        elif a == "--box-threshold" and i + 1 < len(args):
            box_threshold = Float32(Float64(String(args[i + 1])))
            i += 2
        elif a == "--text-threshold" and i + 1 < len(args):
            text_threshold = Float32(Float64(String(args[i + 1])))
            i += 2
        else:
            samples.append(a)
            i += 1

    var load_start = perf_counter_ns()
    var weights = load_lft(weights_path)
    var cfg = decode_config(weights)
    print(
        "loaded", len(weights), "tensors in",
        Float64(perf_counter_ns() - load_start) / 1e9, "s |", cfg,
    )

    if image_path != "":
        if prompt == "":
            raise Error("--image requires --prompt")
        run_standalone(
            weights, cfg, vocab_path, image_path, prompt, csv_path,
            box_threshold, text_threshold,
        )
        return

    var all_pass = True
    for sample in samples:
        run_parity(weights, cfg, vocab_path, String(sample), all_pass)
    print("PARITY:", "PASS" if all_pass else "FAIL")
