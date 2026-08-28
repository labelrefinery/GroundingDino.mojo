"""End-to-end Grounding DINO forward pass and detection post-processing.

`forward` keeps every stage's output so `main.mojo` can report parity stage by stage;
`postprocess` reproduces `GroundingDinoProcessor.post_process_grounded_object_detection`
(sigmoid, max over the padded text axis, cxcywh -> xyxy in pixels, then the token
positions above `text_threshold` decoded back into a phrase).
"""

from .bert import forward as bert_forward, project
from .decoder import (
    DecoderOutput, contrastive_logits, forward as decoder_forward, mlp_head, select_queries
)
from .encoder import forward as encoder_forward
from .io import Config
from .swin import FeatureMap, forward as swin_forward
from .tensor import Tensor, inverse_sigmoid, sigmoid
from .tokenizer import text_masks
from .vision import MultiScaleFeatures, project_levels

comptime BOX_THRESHOLD = Float32(0.35)
comptime TEXT_THRESHOLD = Float32(0.25)


struct Outputs(Movable):
    """Every stage of the forward pass, kept for the parity report."""

    var bert_hidden: Tensor
    var text_features: Tensor
    var levels: List[FeatureMap]
    var features: MultiScaleFeatures
    var enc_vision: Tensor
    var enc_text: Tensor
    var init_reference: Tensor
    var target: Tensor
    var decoder: DecoderOutput
    var logits: Tensor
    var pred_boxes: Tensor

    def __init__(
        out self,
        var bert_hidden: Tensor,
        var text_features: Tensor,
        var levels: List[FeatureMap],
        var features: MultiScaleFeatures,
        var enc_vision: Tensor,
        var enc_text: Tensor,
        var init_reference: Tensor,
        var target: Tensor,
        var decoder: DecoderOutput,
        var logits: Tensor,
        var pred_boxes: Tensor,
    ):
        self.bert_hidden = bert_hidden^
        self.text_features = text_features^
        self.levels = levels^
        self.features = features^
        self.enc_vision = enc_vision^
        self.enc_text = enc_text^
        self.init_reference = init_reference^
        self.target = target^
        self.decoder = decoder^
        self.logits = logits^
        self.pred_boxes = pred_boxes^

    def __init__(out self, *, deinit move: Self):
        self.bert_hidden = move.bert_hidden^
        self.text_features = move.text_features^
        self.levels = move.levels^
        self.features = move.features^
        self.enc_vision = move.enc_vision^
        self.enc_text = move.enc_text^
        self.init_reference = move.init_reference^
        self.target = move.target^
        self.decoder = move.decoder^
        self.logits = move.logits^
        self.pred_boxes = move.pred_boxes^


def forward(
    pixels: Tensor, ids: List[Int], weights: Dict[String, Tensor], cfg: Config
) raises -> Outputs:
    """Run the whole model on one preprocessed image and one tokenized prompt."""
    var unused = Tensor.zeros([1])
    return forward_with(pixels, ids, weights, cfg, unused, False)


def forward_with(
    pixels: Tensor,
    ids: List[Int],
    weights: Dict[String, Tensor],
    cfg: Config,
    override_reference: Tensor,
    use_override: Bool,
) raises -> Outputs:
    """As `forward`, but optionally starting the decoder from supplied initial boxes.

    The parity harness uses the override to separate two questions: whether language-
    guided query selection picked the same proposals (it can differ by a couple of
    slots, because neighbouring top-k scores are closer together than float32 noise)
    and whether the decoder itself is faithful.
    """
    var masks = text_masks(ids)
    var bert_hidden = bert_forward(ids, masks[1], masks[0], weights, cfg)
    var text = project(bert_hidden, weights)

    var levels = swin_forward(pixels, weights, cfg)
    var features = project_levels(levels, weights, cfg)
    var enc = encoder_forward(features, text, masks[1], masks[0], weights, cfg)

    var selection = select_queries(enc[0], enc[1], features, weights, cfg)
    var initial = (
        Tensor(copy=override_reference) if use_override else selection.reference.copy()
    )
    var head = decode_and_predict(
        selection.target, initial, enc[0], enc[1], features, weights, cfg
    )

    return Outputs(
        bert_hidden^, text^, levels^, features^,
        enc[0].copy(), enc[1].copy(), selection.reference.copy(), selection.target.copy(),
        head.decoder.copy(), head.logits.copy(), head.boxes.copy(),
    )


struct Prediction(Movable):
    """Decoder states plus the final classification logits and boxes."""

    var decoder: DecoderOutput
    var logits: Tensor
    var boxes: Tensor

    def __init__(out self, var decoder: DecoderOutput, var logits: Tensor, var boxes: Tensor):
        self.decoder = decoder^
        self.logits = logits^
        self.boxes = boxes^

    def __init__(out self, *, deinit move: Self):
        self.decoder = move.decoder^
        self.logits = move.logits^
        self.boxes = move.boxes^


def decode_and_predict(
    target: Tensor,
    initial_reference: Tensor,
    enc_vision: Tensor,
    enc_text: Tensor,
    features: MultiScaleFeatures,
    weights: Dict[String, Tensor],
    cfg: Config,
) raises -> Prediction:
    """Decoder plus the last layer's detection heads.

    Only the last decoder layer's heads are evaluated -- the earlier ones exist for the
    auxiliary training losses and are unused at inference.
    """
    var decoder = decoder_forward(
        target, initial_reference, enc_vision, enc_text, features, weights, cfg
    )
    var last = cfg.decoder_layers - 1
    var logits = contrastive_logits(decoder.hidden[last], enc_text, cfg.max_text_len)
    var delta = mlp_head(decoder.hidden[last], weights, "bbox" + String(last), 3)
    var num_queries = decoder.hidden[last].dim(0)
    var boxes = Tensor.zeros([num_queries, 4])
    for q in range(num_queries):
        for c in range(4):
            var reference = decoder.reference[last - 1].at2(q, c)
            boxes.set2(q, c, sigmoid(delta.at2(q, c) + inverse_sigmoid(reference, 1e-5)))
    return Prediction(decoder^, logits^, boxes^)


@fieldwise_init
struct Detection(Copyable, Movable):
    """One kept detection in pixel coordinates."""

    var score: Float32
    var x1: Float32
    var y1: Float32
    var x2: Float32
    var y2: Float32
    var label: String
    var query: Int


def decode_tokens(tokens: List[String]) raises -> String:
    """Join WordPiece pieces back into a phrase, matching `tokenizer.batch_decode`."""
    var out = String("")
    for i in range(len(tokens)):
        var piece = tokens[i]
        if piece.startswith("##"):
            out += String(piece.removeprefix("##"))
        elif i == 0:
            out += piece
        else:
            out += " " + piece
    return out^


def postprocess(
    logits: Tensor,
    boxes: Tensor,
    ids: List[Int],
    vocab_tokens: List[String],
    height: Int,
    width: Int,
    box_threshold: Float32,
    text_threshold: Float32,
) raises -> List[Detection]:
    """Threshold, rescale to pixels and attach the phrase each detection points at."""
    var num_queries = logits.dim(0)
    var max_text_len = logits.dim(1)
    var seq_len = len(ids)
    var out = List[Detection]()
    for q in range(num_queries):
        var score = Float32(0.0)
        for c in range(max_text_len):
            var p = sigmoid(logits.at2(q, c))
            if p > score:
                score = p
        if score <= box_threshold:
            continue

        # get_phrases_from_posmap zeroes position 0 and everything from max_text_len - 1.
        var tokens = List[String]()
        for c in range(1, seq_len):
            if c >= max_text_len - 1:
                break
            if sigmoid(logits.at2(q, c)) > text_threshold:
                tokens.append(vocab_tokens[ids[c]])

        var cx = boxes.at2(q, 0)
        var cy = boxes.at2(q, 1)
        var w = boxes.at2(q, 2)
        var h = boxes.at2(q, 3)
        out.append(
            Detection(
                score,
                (cx - 0.5 * w) * Float32(width),
                (cy - 0.5 * h) * Float32(height),
                (cx + 0.5 * w) * Float32(width),
                (cy + 0.5 * h) * Float32(height),
                decode_tokens(tokens),
                q,
            )
        )
    return out^
