"""Hand-computed unit tests for every op the Grounding DINO forward pass is built from.

Run with `pixi run test` (i.e. `mojo run -I src tests/test_ops.mojo`).
"""

from std.math import sqrt
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

from groundingdino.attention import attention_core
from groundingdino.decoder import contrastive_logits, log_ratio, mlp_head
from groundingdino.deform import _sample_bilinear
from groundingdino.encoder import sinusoidal_embedding
from groundingdino.image import Image, normalize, read_ppm, resize_bilinear, target_size
from groundingdino.model import decode_tokens
from groundingdino.swin import patch_merging
from groundingdino.tensor import (
    Tensor, add_, gelu_, inverse_sigmoid, layernorm, matmul_nt, matmul_nt_nobias,
    max_abs, max_abs_diff, relu_, sigmoid, softmax_rows_
)
from groundingdino.tokenizer import (
    basic_tokenize, is_special, text_masks, tokenize, wordpiece
)
from groundingdino.vision import group_norm, sine_position_embedding

comptime TOL = Float64(1e-6)


def _fill(mut t: Tensor, values: List[Float32]):
    for i in range(len(values)):
        t[i] = values[i]


def test_matmul_small() raises:
    """2x3 @ (2x3)^T + bias, computed by hand."""
    var x = Tensor.zeros([2, 3])
    _fill(x, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var w = Tensor.zeros([2, 3])
    _fill(w, [1.0, 0.0, -1.0, 0.5, 0.5, 0.5])
    var b = Tensor.zeros([2])
    _fill(b, [10.0, -1.0])
    var y = matmul_nt(x, w, b)
    assert_equal(y.dim(0), 2)
    assert_equal(y.dim(1), 2)
    assert_almost_equal(Float64(y.at2(0, 0)), 1.0 - 3.0 + 10.0, atol=TOL)
    assert_almost_equal(Float64(y.at2(0, 1)), 3.0 - 1.0, atol=TOL)
    assert_almost_equal(Float64(y.at2(1, 0)), 4.0 - 6.0 + 10.0, atol=TOL)
    assert_almost_equal(Float64(y.at2(1, 1)), 7.5 - 1.0, atol=TOL)


def test_matmul_blocking_edges() raises:
    """Rows and columns that do not divide the 6x4 micro-kernel must still be exact."""
    var m = 13
    var k = 11
    var n = 7
    var x = Tensor.zeros([m, k])
    for i in range(m):
        for j in range(k):
            x.set2(i, j, Float32((i * 7 + j * 3) % 5) - 2.0)
    var w = Tensor.zeros([n, k])
    for i in range(n):
        for j in range(k):
            w.set2(i, j, Float32((i * 5 + j) % 4) - 1.5)
    var b = Tensor.zeros([n])
    for i in range(n):
        b[i] = Float32(i) * 0.25
    var y = matmul_nt(x, w, b)
    for i in range(m):
        for j in range(n):
            var expected = Float64(b[j])
            for t in range(k):
                expected += Float64(x.at2(i, t)) * Float64(w.at2(j, t))
            assert_almost_equal(Float64(y.at2(i, j)), expected, atol=TOL)


def test_matmul_nobias() raises:
    var x = Tensor.zeros([1, 2])
    _fill(x, [3.0, 4.0])
    var w = Tensor.zeros([1, 2])
    _fill(w, [3.0, 4.0])
    var y = matmul_nt_nobias(x, w)
    assert_almost_equal(Float64(y.at2(0, 0)), 25.0, atol=TOL)


def test_layernorm() raises:
    """Row [1, 2, 3] has mean 2 and biased variance 2/3."""
    var x = Tensor.zeros([1, 3])
    _fill(x, [1.0, 2.0, 3.0])
    var gamma = Tensor.full([3], 2.0)
    var beta = Tensor.full([3], 1.0)
    var y = layernorm(x, gamma, beta, 1e-5)
    var inv = 1.0 / sqrt(2.0 / 3.0 + 1e-5)
    assert_almost_equal(Float64(y.at2(0, 0)), -inv * 2.0 + 1.0, atol=1e-5)
    assert_almost_equal(Float64(y.at2(0, 1)), 1.0, atol=1e-5)
    assert_almost_equal(Float64(y.at2(0, 2)), inv * 2.0 + 1.0, atol=1e-5)


def test_softmax_rows() raises:
    var x = Tensor.zeros([2, 2])
    _fill(x, [0.0, 0.0, 0.0, 1000.0])
    softmax_rows_(x, 2)
    assert_almost_equal(Float64(x.at2(0, 0)), 0.5, atol=TOL)
    assert_almost_equal(Float64(x.at2(0, 1)), 0.5, atol=TOL)
    assert_almost_equal(Float64(x.at2(1, 0)), 0.0, atol=TOL)
    assert_almost_equal(Float64(x.at2(1, 1)), 1.0, atol=TOL)


def test_activations() raises:
    var x = Tensor.zeros([3])
    _fill(x, [-1.0, 0.0, 2.0])
    var y = Tensor(copy=x)
    relu_(y)
    assert_almost_equal(Float64(y[0]), 0.0, atol=TOL)
    assert_almost_equal(Float64(y[2]), 2.0, atol=TOL)
    var g = Tensor(copy=x)
    gelu_(g)
    assert_almost_equal(Float64(g[1]), 0.0, atol=TOL)
    # gelu(-1) = -0.15865525, gelu(2) = 1.9544997 (exact erf form)
    assert_almost_equal(Float64(g[0]), -0.15865525, atol=1e-6)
    assert_almost_equal(Float64(g[2]), 1.9544997, atol=1e-6)


def test_sigmoid_roundtrip() raises:
    assert_almost_equal(Float64(sigmoid(0.0)), 0.5, atol=TOL)
    assert_almost_equal(Float64(sigmoid(-40.0)), 0.0, atol=1e-12)
    var p = Float32(0.3)
    assert_almost_equal(Float64(sigmoid(inverse_sigmoid(p, 1e-5))), 0.3, atol=1e-6)
    # The eps clamp is what keeps logit(0) and logit(1) finite.
    assert_true(inverse_sigmoid(0.0, 1e-5) > -12.0)
    assert_almost_equal(Float64(log_ratio(0.5)), 0.0, atol=TOL)


def test_attention_core_uniform() raises:
    """Zero queries and keys make attention a plain average of the values."""
    var q = Tensor.zeros([1, 2])
    var k = Tensor.zeros([2, 2])
    var v = Tensor.zeros([2, 2])
    _fill(v, [0.0, 4.0, 2.0, 8.0])
    var mask = Tensor.zeros([1, 2])
    var out = attention_core(q, k, v, 1, mask, False)
    assert_almost_equal(Float64(out.at2(0, 0)), 1.0, atol=TOL)
    assert_almost_equal(Float64(out.at2(0, 1)), 6.0, atol=TOL)


def test_attention_core_mask() raises:
    """A large negative additive mask removes a key entirely."""
    var q = Tensor.zeros([1, 2])
    var k = Tensor.zeros([2, 2])
    var v = Tensor.zeros([2, 2])
    _fill(v, [0.0, 4.0, 2.0, 8.0])
    var mask = Tensor.zeros([1, 2])
    mask[1] = -1.0e30
    var out = attention_core(q, k, v, 1, mask, True)
    assert_almost_equal(Float64(out.at2(0, 0)), 0.0, atol=TOL)
    assert_almost_equal(Float64(out.at2(0, 1)), 4.0, atol=TOL)


def test_group_norm() raises:
    """Two groups of one channel each normalize independently to mean 0."""
    var x = Tensor.zeros([2, 2])
    _fill(x, [1.0, 10.0, 3.0, 30.0])
    var gamma = Tensor.full([2], 1.0)
    var beta = Tensor.zeros([2])
    var y = group_norm(x, gamma, beta, 2)
    assert_almost_equal(Float64(y.at2(0, 0)), -1.0, atol=1e-4)
    assert_almost_equal(Float64(y.at2(1, 0)), 1.0, atol=1e-4)
    assert_almost_equal(Float64(y.at2(0, 1)), -1.0, atol=1e-4)
    assert_almost_equal(Float64(y.at2(1, 1)), 1.0, atol=1e-4)


def test_sine_position_embedding() raises:
    """Channel 1 of a 1x1 map is cos(2*pi * 1/(1+eps) / 1) = cos(2*pi) = 1."""
    var pos = sine_position_embedding(1, 1, 4, 20.0)
    assert_equal(pos.dim(0), 1)
    assert_equal(pos.dim(1), 4)
    assert_almost_equal(Float64(pos.at2(0, 1)), 1.0, atol=1e-5)
    assert_almost_equal(Float64(pos.at2(0, 3)), 1.0, atol=1e-5)


def test_sinusoidal_embedding_swaps_xy() raises:
    """With two or more coordinates, x and y blocks are emitted in DETR order."""
    var coords = Tensor.zeros([1, 4])
    _fill(coords, [0.25, 0.75, 0.5, 0.5])
    var e = sinusoidal_embedding(coords, 2, 10000.0, False)
    assert_equal(e.dim(1), 8)
    var single_y = Tensor.zeros([1, 1])
    single_y[0] = 0.75
    var y_only = sinusoidal_embedding(single_y, 2, 10000.0, False)
    assert_almost_equal(Float64(e.at2(0, 0)), Float64(y_only.at2(0, 0)), atol=TOL)
    assert_almost_equal(Float64(e.at2(0, 1)), Float64(y_only.at2(0, 1)), atol=TOL)


def test_sinusoidal_embedding_truncation() raises:
    """Integer position ids truncate the embedding, as transformers does."""
    var coords = Tensor.zeros([1, 1])
    coords[0] = 1.0
    var e = sinusoidal_embedding(coords, 4, 10000.0, True)
    for i in range(4):
        var v = Float64(e.at2(0, i))
        assert_true(v == Float64(Int(v)))


def test_bilinear_sampler() raises:
    """Sampling the centre of a 2x2 map averages the four corners; outside gives zero."""
    var value = Tensor.zeros([4, 1])
    _fill(value, [0.0, 2.0, 4.0, 6.0])
    var dst = Tensor.zeros([1])
    _sample_bilinear(value.ptr(), 0, 2, 2, 1, 0, 1, 0.5, 0.5, 1.0, dst.ptr())
    assert_almost_equal(Float64(dst[0]), 3.0, atol=TOL)
    var outside = Tensor.zeros([1])
    _sample_bilinear(value.ptr(), 0, 2, 2, 1, 0, 1, -5.0, -5.0, 1.0, outside.ptr())
    assert_almost_equal(Float64(outside[0]), 0.0, atol=TOL)


def test_contrastive_logits_padding() raises:
    """Positions past the prompt length are filled with -inf so sigmoid gives 0."""
    var q = Tensor.zeros([1, 2])
    _fill(q, [1.0, 0.0])
    var text = Tensor.zeros([2, 2])
    _fill(text, [1.0, 0.0, 0.0, 1.0])
    var logits = contrastive_logits(q, text, 5)
    assert_almost_equal(Float64(logits.at2(0, 0)), 1.0, atol=TOL)
    assert_almost_equal(Float64(logits.at2(0, 1)), 0.0, atol=TOL)
    assert_almost_equal(Float64(sigmoid(logits.at2(0, 4))), 0.0, atol=TOL)


def test_tokenizer_basic_split() raises:
    var pieces = basic_tokenize("Excavator . Crane .")
    assert_equal(len(pieces), 4)
    assert_equal(pieces[0], String("excavator"))
    assert_equal(pieces[1], String("."))
    assert_equal(pieces[2], String("crane"))


def test_wordpiece_greedy() raises:
    """Longest-match-first, with continuation pieces prefixed by ##."""
    var vocab = Dict[String, Int]()
    vocab["ex"] = 1
    vocab["##ca"] = 2
    vocab["##vator"] = 3
    vocab["[UNK]"] = 100
    var pieces = wordpiece("excavator", vocab)
    assert_equal(len(pieces), 3)
    assert_equal(pieces[0], String("ex"))
    assert_equal(pieces[1], String("##ca"))
    assert_equal(pieces[2], String("##vator"))
    var unknown = wordpiece("zzz", vocab)
    assert_equal(len(unknown), 1)
    assert_equal(unknown[0], String("[UNK]"))


def test_tokenize_adds_special_tokens() raises:
    var vocab = Dict[String, Int]()
    vocab["cat"] = 4937
    vocab["."] = 1012
    vocab["[UNK]"] = 100
    var ids = tokenize("cat .", vocab)
    assert_equal(len(ids), 4)
    assert_equal(ids[0], 101)
    assert_equal(ids[1], 4937)
    assert_equal(ids[2], 1012)
    assert_equal(ids[3], 102)
    assert_true(is_special(101))
    assert_true(is_special(1012))
    assert_true(not is_special(4937))


def test_text_masks_blocks_phrases() raises:
    """[CLS] cat . dog . [SEP] -- the two phrases must not see each other."""
    var ids: List[Int] = [101, 4937, 1012, 3899, 1012, 102]
    var masks = text_masks(ids)
    assert_almost_equal(Float64(masks[0].at2(1, 1)), 1.0, atol=TOL)
    assert_almost_equal(Float64(masks[0].at2(1, 3)), 0.0, atol=TOL)
    assert_almost_equal(Float64(masks[0].at2(3, 3)), 1.0, atol=TOL)
    assert_almost_equal(Float64(masks[0].at2(0, 1)), 0.0, atol=TOL)
    # Position ids restart inside every phrase.
    assert_almost_equal(Float64(masks[1][1]), 0.0, atol=TOL)
    assert_almost_equal(Float64(masks[1][3]), 0.0, atol=TOL)


def test_decode_tokens() raises:
    var pieces: List[String] = ["ex", "##ca", "##vator", "arm"]
    assert_equal(decode_tokens(pieces), String("excavator arm"))


def test_patch_merging_order() raises:
    """Concatenation order is (0,0), (1,0), (0,1), (1,1), as `for col ... for row ...` gives."""
    var weights = Dict[String, Tensor]()
    weights["m.down.norm.w"] = Tensor.full([4], 1.0)
    weights["m.down.norm.b"] = Tensor.zeros([4])
    # 4C -> 2C reduction that simply selects concatenation slots 1 and 2.
    var reduction = Tensor.zeros([2, 4])
    reduction.set2(0, 1, 1.0)
    reduction.set2(1, 2, 1.0)
    weights["m.down.reduction.w"] = reduction^
    var feature = Tensor.zeros([4, 1])
    _fill(feature, [10.0, 20.0, 30.0, 40.0])  # (0,0)=10 (0,1)=20 (1,0)=30 (1,1)=40

    from groundingdino.swin import FeatureMap

    var merged = patch_merging(FeatureMap(feature^, 2, 2, 1), weights, "m")
    assert_equal(merged.height, 1)
    assert_equal(merged.width, 1)
    assert_equal(merged.channels, 2)
    # Slot 1 must be (1,0) = 30 and slot 2 must be (0,1) = 20, so out[0] > out[1].
    assert_true(merged.data.at2(0, 0) > merged.data.at2(0, 1))


def test_mlp_head_relu_between_layers() raises:
    var weights = Dict[String, Tensor]()
    var w0 = Tensor.zeros([1, 1])
    w0[0] = -1.0
    weights["h.l0.w"] = w0^
    weights["h.l0.b"] = Tensor.zeros([1])
    var w1 = Tensor.zeros([1, 1])
    w1[0] = 1.0
    weights["h.l1.w"] = w1^
    weights["h.l1.b"] = Tensor.zeros([1])
    var x = Tensor.zeros([1, 1])
    x[0] = 5.0
    var y = mlp_head(x, weights, "h", 2)
    # -5 goes through ReLU to 0, so the head outputs 0 rather than -5.
    assert_almost_equal(Float64(y.at2(0, 0)), 0.0, atol=TOL)


def test_target_size_respects_longest_edge() raises:
    var square = target_size(500, 500, 800, 1333)
    assert_equal(square[0], 800)
    assert_equal(square[1], 800)
    # 480x640 scales the short side to 800 without exceeding 1333 on the long side.
    var landscape = target_size(480, 640, 800, 1333)
    assert_equal(landscape[0], 800)
    assert_equal(landscape[1], 1066)
    # A very wide image is capped by longest_edge instead.
    var wide = target_size(200, 2000, 800, 1333)
    assert_true(wide[1] <= 1333)


def test_resize_and_normalize() raises:
    """Upsampling a constant image is constant; normalization matches (v/255 - m) / s."""
    var pixels = Tensor.full([3, 2, 2], 255.0)
    var image = Image(pixels^, 2, 2)
    var big = resize_bilinear(image, 4, 4)
    assert_equal(big.height, 4)
    assert_almost_equal(Float64(big.data.at3(0, 2, 2)), 255.0, atol=1e-4)
    var normalized = normalize(big)
    assert_almost_equal(
        Float64(normalized.at3(0, 0, 0)), (1.0 - 0.485) / 0.229, atol=1e-5
    )
    assert_almost_equal(
        Float64(normalized.at3(2, 3, 3)), (1.0 - 0.406) / 0.225, atol=1e-5
    )


def test_ppm_roundtrip() raises:
    """The PPM reader handles comments and returns channel-first pixel values."""
    var path = String("/tmp/groundingdino_test.ppm")
    var f = open(path, "w")
    f.write(String("P6\n# a comment\n2 1\n255\n"))
    f.close()
    var g = open(path, "a")
    var payload = List[UInt8]()
    payload.append(255)
    payload.append(0)
    payload.append(0)
    payload.append(0)
    payload.append(128)
    payload.append(255)
    g.write_bytes(payload)
    g.close()

    var image = read_ppm(path)
    assert_equal(image.width, 2)
    assert_equal(image.height, 1)
    assert_almost_equal(Float64(image.data.at3(0, 0, 0)), 255.0, atol=TOL)
    assert_almost_equal(Float64(image.data.at3(1, 0, 1)), 128.0, atol=TOL)
    assert_almost_equal(Float64(image.data.at3(2, 0, 1)), 255.0, atol=TOL)


def test_tensor_helpers() raises:
    var a = Tensor.zeros([2, 2])
    _fill(a, [1.0, -2.0, 3.0, -4.0])
    var b = Tensor.zeros([2, 2])
    add_(b, a)
    assert_almost_equal(Float64(max_abs(b)), 4.0, atol=TOL)
    assert_almost_equal(Float64(max_abs_diff(a, b)), 0.0, atol=TOL)
    var reshaped = a.reshaped([4])
    assert_equal(reshaped.rank(), 1)
    assert_almost_equal(Float64(reshaped[3]), -4.0, atol=TOL)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
