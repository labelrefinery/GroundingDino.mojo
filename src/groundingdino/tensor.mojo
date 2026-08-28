"""Minimal owned float32 tensor plus the ops the Grounding DINO forward pass needs.

Storage is a raw `alloc`ed buffer rather than a `List` so the hot kernels can take
`Pointer[Float32, MutUntrackedOrigin]` arguments directly.

The matmul is the reason this file exists: Grounding DINO tiny is ~173M parameters
and one 800x1200 image is several hundred GFLOP, so `matmul_nt` is a hand-blocked
6x4 micro-kernel vectorized over the contiguous reduction dimension (~100 GFLOP/s on
an Apple M4 P-core). Note that `parallelize` is *not* present in `std.algorithm` on
the Mojo nightly this repo targets (1.1.0.dev2026082807), so the kernel is
single-threaded; the register blocking is what buys the speed.
"""

from std.math import sqrt, exp, erf, log

comptime VW = 4
"""SIMD width used by the matmul micro-kernel (NEON f32 lane count)."""

comptime FP = Pointer[Float32, MutUntrackedOrigin]
"""Raw mutable pointer into a tensor buffer."""


struct Tensor(Copyable, Movable, Writable):
    """Row-major float32 tensor with an explicit shape."""

    var shape: List[Int]
    var size: Int
    var data: Self.FP

    comptime FP = FP

    def __init__(out self, shape: List[Int]):
        var n = 1
        for d in shape:
            n *= d
        self.shape = shape.copy()
        self.size = n
        self.data = alloc[Float32](n if n > 0 else 1)

    @staticmethod
    def zeros(shape: List[Int]) raises -> Self:
        var t = Self(shape)
        for i in range(t.size):
            t.data[unsafe_offset=i] = 0.0
        return t^

    @staticmethod
    def full(shape: List[Int], value: Float32) raises -> Self:
        var t = Self(shape)
        for i in range(t.size):
            t.data[unsafe_offset=i] = value
        return t^

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.size = copy.size
        self.data = alloc[Float32](copy.size if copy.size > 0 else 1)
        for i in range(copy.size):
            self.data[unsafe_offset=i] = copy.data[unsafe_offset=i]

    def __init__(out self, *, deinit move: Self):
        self.shape = move.shape^
        self.size = move.size
        self.data = move.data

    def __deinit__(deinit self):
        self.data.unsafe_free()

    def numel(self) -> Int:
        return self.size

    def rank(self) -> Int:
        return len(self.shape)

    def dim(self, i: Int) -> Int:
        return self.shape[i]

    def ptr(self) -> Self.FP:
        return self.data

    def __getitem__(self, i: Int) -> Float32:
        return self.data[unsafe_offset=i]

    def __setitem__(mut self, i: Int, v: Float32):
        self.data[unsafe_offset=i] = v

    def at2(self, i: Int, j: Int) -> Float32:
        return self.data[unsafe_offset=i * self.shape[1] + j]

    def set2(mut self, i: Int, j: Int, v: Float32):
        self.data[unsafe_offset=i * self.shape[1] + j] = v

    def at3(self, i: Int, j: Int, k: Int) -> Float32:
        return self.data[unsafe_offset=(i * self.shape[1] + j) * self.shape[2] + k]

    def set3(mut self, i: Int, j: Int, k: Int, v: Float32):
        self.data[unsafe_offset=(i * self.shape[1] + j) * self.shape[2] + k] = v

    def row(self, i: Int) -> Self.FP:
        """Pointer to row ``i`` of a rank-2 tensor."""
        return self.data + i * self.shape[1]

    def reshaped(self, shape: List[Int]) raises -> Self:
        """A copy of this tensor with a new shape of the same element count."""
        var n = 1
        for d in shape:
            n *= d
        if n != self.size:
            raise Error("reshaped: element count mismatch")
        var t = Self(shape)
        for i in range(self.size):
            t.data[unsafe_offset=i] = self.data[unsafe_offset=i]
        return t^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Tensor(shape=[")
        for i in range(len(self.shape)):
            if i > 0:
                writer.write(", ")
            writer.write(self.shape[i])
        writer.write("], numel=", self.size, ")")


def _row_tail(
    i0: Int, rows: Int, j: Int, n: Int, k: Int, x: FP, w: FP, bias: FP, has_bias: Bool, dst: FP
):
    """Scalar fallback for the ragged edges of the blocked kernel."""
    for r in range(rows):
        var i = i0 + r
        var acc = SIMD[DType.float32, VW](0)
        var xp = x.unsafe_offset(i * k)
        var wp = w.unsafe_offset(j * k)
        var t = 0
        while t + VW <= k:
            acc = xp.unsafe_load[width=VW](t).fma(wp.unsafe_load[width=VW](t), acc)
            t += VW
        var s = acc.reduce_add()
        while t < k:
            s += xp[unsafe_offset=t] * wp[unsafe_offset=t]
            t += 1
        if has_bias:
            s += bias[unsafe_offset=j]
        dst[unsafe_offset=i * n + j] = s


def _micro_kernel(
    m0: Int, m1: Int, n: Int, k: Int, x: FP, w: FP, bias: FP, has_bias: Bool, dst: FP
):
    """Blocked 6x4 micro-kernel: ``dst[i, j] = dot(x[i, :], w[j, :]) + bias[j]``."""
    var i = m0
    while i + 6 <= m1:
        var xp0 = x.unsafe_offset((i + 0) * k)
        var xp1 = x.unsafe_offset((i + 1) * k)
        var xp2 = x.unsafe_offset((i + 2) * k)
        var xp3 = x.unsafe_offset((i + 3) * k)
        var xp4 = x.unsafe_offset((i + 4) * k)
        var xp5 = x.unsafe_offset((i + 5) * k)
        var j = 0
        while j + 4 <= n:
            var wp0 = w.unsafe_offset((j + 0) * k)
            var wp1 = w.unsafe_offset((j + 1) * k)
            var wp2 = w.unsafe_offset((j + 2) * k)
            var wp3 = w.unsafe_offset((j + 3) * k)
            var a00 = SIMD[DType.float32, VW](0)
            var a01 = SIMD[DType.float32, VW](0)
            var a02 = SIMD[DType.float32, VW](0)
            var a03 = SIMD[DType.float32, VW](0)
            var a10 = SIMD[DType.float32, VW](0)
            var a11 = SIMD[DType.float32, VW](0)
            var a12 = SIMD[DType.float32, VW](0)
            var a13 = SIMD[DType.float32, VW](0)
            var a20 = SIMD[DType.float32, VW](0)
            var a21 = SIMD[DType.float32, VW](0)
            var a22 = SIMD[DType.float32, VW](0)
            var a23 = SIMD[DType.float32, VW](0)
            var a30 = SIMD[DType.float32, VW](0)
            var a31 = SIMD[DType.float32, VW](0)
            var a32 = SIMD[DType.float32, VW](0)
            var a33 = SIMD[DType.float32, VW](0)
            var a40 = SIMD[DType.float32, VW](0)
            var a41 = SIMD[DType.float32, VW](0)
            var a42 = SIMD[DType.float32, VW](0)
            var a43 = SIMD[DType.float32, VW](0)
            var a50 = SIMD[DType.float32, VW](0)
            var a51 = SIMD[DType.float32, VW](0)
            var a52 = SIMD[DType.float32, VW](0)
            var a53 = SIMD[DType.float32, VW](0)
            var t = 0
            while t + VW <= k:
                var xv0 = xp0.unsafe_load[width=VW](t)
                var xv1 = xp1.unsafe_load[width=VW](t)
                var xv2 = xp2.unsafe_load[width=VW](t)
                var xv3 = xp3.unsafe_load[width=VW](t)
                var xv4 = xp4.unsafe_load[width=VW](t)
                var xv5 = xp5.unsafe_load[width=VW](t)
                var wv0 = wp0.unsafe_load[width=VW](t)
                var wv1 = wp1.unsafe_load[width=VW](t)
                var wv2 = wp2.unsafe_load[width=VW](t)
                var wv3 = wp3.unsafe_load[width=VW](t)
                a00 = xv0.fma(wv0, a00)
                a01 = xv0.fma(wv1, a01)
                a02 = xv0.fma(wv2, a02)
                a03 = xv0.fma(wv3, a03)
                a10 = xv1.fma(wv0, a10)
                a11 = xv1.fma(wv1, a11)
                a12 = xv1.fma(wv2, a12)
                a13 = xv1.fma(wv3, a13)
                a20 = xv2.fma(wv0, a20)
                a21 = xv2.fma(wv1, a21)
                a22 = xv2.fma(wv2, a22)
                a23 = xv2.fma(wv3, a23)
                a30 = xv3.fma(wv0, a30)
                a31 = xv3.fma(wv1, a31)
                a32 = xv3.fma(wv2, a32)
                a33 = xv3.fma(wv3, a33)
                a40 = xv4.fma(wv0, a40)
                a41 = xv4.fma(wv1, a41)
                a42 = xv4.fma(wv2, a42)
                a43 = xv4.fma(wv3, a43)
                a50 = xv5.fma(wv0, a50)
                a51 = xv5.fma(wv1, a51)
                a52 = xv5.fma(wv2, a52)
                a53 = xv5.fma(wv3, a53)
                t += VW
            var s00 = a00.reduce_add()
            var s01 = a01.reduce_add()
            var s02 = a02.reduce_add()
            var s03 = a03.reduce_add()
            var s10 = a10.reduce_add()
            var s11 = a11.reduce_add()
            var s12 = a12.reduce_add()
            var s13 = a13.reduce_add()
            var s20 = a20.reduce_add()
            var s21 = a21.reduce_add()
            var s22 = a22.reduce_add()
            var s23 = a23.reduce_add()
            var s30 = a30.reduce_add()
            var s31 = a31.reduce_add()
            var s32 = a32.reduce_add()
            var s33 = a33.reduce_add()
            var s40 = a40.reduce_add()
            var s41 = a41.reduce_add()
            var s42 = a42.reduce_add()
            var s43 = a43.reduce_add()
            var s50 = a50.reduce_add()
            var s51 = a51.reduce_add()
            var s52 = a52.reduce_add()
            var s53 = a53.reduce_add()
            while t < k:
                var xs0 = xp0[unsafe_offset=t]
                var xs1 = xp1[unsafe_offset=t]
                var xs2 = xp2[unsafe_offset=t]
                var xs3 = xp3[unsafe_offset=t]
                var xs4 = xp4[unsafe_offset=t]
                var xs5 = xp5[unsafe_offset=t]
                var ws0 = wp0[unsafe_offset=t]
                var ws1 = wp1[unsafe_offset=t]
                var ws2 = wp2[unsafe_offset=t]
                var ws3 = wp3[unsafe_offset=t]
                s00 += xs0 * ws0
                s01 += xs0 * ws1
                s02 += xs0 * ws2
                s03 += xs0 * ws3
                s10 += xs1 * ws0
                s11 += xs1 * ws1
                s12 += xs1 * ws2
                s13 += xs1 * ws3
                s20 += xs2 * ws0
                s21 += xs2 * ws1
                s22 += xs2 * ws2
                s23 += xs2 * ws3
                s30 += xs3 * ws0
                s31 += xs3 * ws1
                s32 += xs3 * ws2
                s33 += xs3 * ws3
                s40 += xs4 * ws0
                s41 += xs4 * ws1
                s42 += xs4 * ws2
                s43 += xs4 * ws3
                s50 += xs5 * ws0
                s51 += xs5 * ws1
                s52 += xs5 * ws2
                s53 += xs5 * ws3
                t += 1
            if has_bias:
                var bv0 = bias[unsafe_offset=j + 0]
                s00 += bv0
                s10 += bv0
                s20 += bv0
                s30 += bv0
                s40 += bv0
                s50 += bv0
                var bv1 = bias[unsafe_offset=j + 1]
                s01 += bv1
                s11 += bv1
                s21 += bv1
                s31 += bv1
                s41 += bv1
                s51 += bv1
                var bv2 = bias[unsafe_offset=j + 2]
                s02 += bv2
                s12 += bv2
                s22 += bv2
                s32 += bv2
                s42 += bv2
                s52 += bv2
                var bv3 = bias[unsafe_offset=j + 3]
                s03 += bv3
                s13 += bv3
                s23 += bv3
                s33 += bv3
                s43 += bv3
                s53 += bv3
            var o0 = (i + 0) * n + j
            dst[unsafe_offset=o0 + 0] = s00
            dst[unsafe_offset=o0 + 1] = s01
            dst[unsafe_offset=o0 + 2] = s02
            dst[unsafe_offset=o0 + 3] = s03
            var o1 = (i + 1) * n + j
            dst[unsafe_offset=o1 + 0] = s10
            dst[unsafe_offset=o1 + 1] = s11
            dst[unsafe_offset=o1 + 2] = s12
            dst[unsafe_offset=o1 + 3] = s13
            var o2 = (i + 2) * n + j
            dst[unsafe_offset=o2 + 0] = s20
            dst[unsafe_offset=o2 + 1] = s21
            dst[unsafe_offset=o2 + 2] = s22
            dst[unsafe_offset=o2 + 3] = s23
            var o3 = (i + 3) * n + j
            dst[unsafe_offset=o3 + 0] = s30
            dst[unsafe_offset=o3 + 1] = s31
            dst[unsafe_offset=o3 + 2] = s32
            dst[unsafe_offset=o3 + 3] = s33
            var o4 = (i + 4) * n + j
            dst[unsafe_offset=o4 + 0] = s40
            dst[unsafe_offset=o4 + 1] = s41
            dst[unsafe_offset=o4 + 2] = s42
            dst[unsafe_offset=o4 + 3] = s43
            var o5 = (i + 5) * n + j
            dst[unsafe_offset=o5 + 0] = s50
            dst[unsafe_offset=o5 + 1] = s51
            dst[unsafe_offset=o5 + 2] = s52
            dst[unsafe_offset=o5 + 3] = s53
            j += 4
        while j < n:
            _row_tail(i, 6, j, n, k, x, w, bias, has_bias, dst)
            j += 1
        i += 6
    while i < m1:
        var j = 0
        while j < n:
            _row_tail(i, 1, j, n, k, x, w, bias, has_bias, dst)
            j += 1
        i += 1


def matmul_nt(x: Tensor, w: Tensor, bias: Tensor) raises -> Tensor:
    """``(M, K) @ (N, K)^T + (N,)`` with torch-layout weight ``w`` (rows are outputs)."""
    if x.rank() != 2 or w.rank() != 2:
        raise Error("matmul_nt: x and w must be rank 2")
    var m = x.dim(0)
    var k = x.dim(1)
    var n = w.dim(0)
    if w.dim(1) != k:
        raise Error("matmul_nt: inner dimension mismatch")
    if bias.numel() != n:
        raise Error("matmul_nt: bias size mismatch")
    var dst = Tensor([m, n])
    _micro_kernel(0, m, n, k, x.ptr(), w.ptr(), bias.ptr(), True, dst.ptr())
    return dst^


def matmul_nt_nobias(x: Tensor, w: Tensor) raises -> Tensor:
    """``(M, K) @ (N, K)^T`` with torch-layout weight ``w``."""
    if x.rank() != 2 or w.rank() != 2:
        raise Error("matmul_nt_nobias: x and w must be rank 2")
    var m = x.dim(0)
    var k = x.dim(1)
    var n = w.dim(0)
    if w.dim(1) != k:
        raise Error("matmul_nt_nobias: inner dimension mismatch")
    var dst = Tensor([m, n])
    _micro_kernel(0, m, n, k, x.ptr(), w.ptr(), x.ptr(), False, dst.ptr())
    return dst^


def linear(x: Tensor, weights: Dict[String, Tensor], name: String) raises -> Tensor:
    """Apply the exported linear layer ``name`` (``name.w`` / ``name.b``) to ``x``."""
    return matmul_nt(x, weights[name + ".w"], weights[name + ".b"])


def add_(mut x: Tensor, y: Tensor) raises:
    """In-place ``x += y`` (element counts must match)."""
    if x.numel() != y.numel():
        raise Error("add_: size mismatch")
    var p = x.ptr()
    var q = y.ptr()
    for i in range(x.numel()):
        p[unsafe_offset=i] = p[unsafe_offset=i] + q[unsafe_offset=i]


def add_scaled_(mut x: Tensor, y: Tensor, scale: Tensor) raises:
    """In-place ``x += scale * y`` broadcasting ``scale`` over the last dimension."""
    var d = scale.numel()
    if x.numel() != y.numel() or x.numel() % d != 0:
        raise Error("add_scaled_: size mismatch")
    var p = x.ptr()
    var q = y.ptr()
    var s = scale.ptr()
    for i in range(x.numel()):
        p[unsafe_offset=i] = p[unsafe_offset=i] + s[unsafe_offset=i % d] * q[unsafe_offset=i]


def relu_(mut x: Tensor):
    var p = x.ptr()
    for i in range(x.numel()):
        if p[unsafe_offset=i] < 0.0:
            p[unsafe_offset=i] = 0.0


def gelu_(mut x: Tensor):
    """Exact erf GELU, matching ``nn.functional.gelu`` (BERT / Swin default)."""
    var p = x.ptr()
    var inv_sqrt2 = Float32(0.70710678118654752440)
    for i in range(x.numel()):
        var v = p[unsafe_offset=i]
        p[unsafe_offset=i] = v * 0.5 * (1.0 + erf(v * inv_sqrt2))


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float32) raises -> Tensor:
    """LayerNorm over the last dimension, matching ``torch.nn.LayerNorm`` in eval mode."""
    var d = x.dim(x.rank() - 1)
    if gamma.numel() != d or beta.numel() != d:
        raise Error("layernorm: gamma/beta size mismatch")
    var rows = x.numel() // d
    var out = Tensor(x.shape.copy())
    var src = x.ptr()
    var dst = out.ptr()
    var g = gamma.ptr()
    var b = beta.ptr()
    var inv_d = 1.0 / Float32(d)
    for r in range(rows):
        var base = r * d
        var mean: Float32 = 0.0
        for j in range(d):
            mean += src[unsafe_offset=base + j]
        mean *= inv_d
        var acc: Float32 = 0.0
        for j in range(d):
            var diff = src[unsafe_offset=base + j] - mean
            acc += diff * diff
        acc *= inv_d
        var inv_std = 1.0 / sqrt(acc + eps)
        for j in range(d):
            dst[unsafe_offset=base + j] = (
                (src[unsafe_offset=base + j] - mean) * inv_std * g[unsafe_offset=j]
                + b[unsafe_offset=j]
            )
    return out^


def layernorm_named(
    x: Tensor, weights: Dict[String, Tensor], name: String, eps: Float32
) raises -> Tensor:
    return layernorm(x, weights[name + ".w"], weights[name + ".b"], eps)


def softmax_rows_(mut x: Tensor, row_len: Int) raises:
    """In-place softmax over contiguous groups of ``row_len`` values."""
    if x.numel() % row_len != 0:
        raise Error("softmax_rows_: length mismatch")
    var p = x.ptr()
    var rows = x.numel() // row_len
    for r in range(rows):
        var base = r * row_len
        var mx = p[unsafe_offset=base]
        for j in range(1, row_len):
            var v = p[unsafe_offset=base + j]
            if v > mx:
                mx = v
        var total: Float32 = 0.0
        for j in range(row_len):
            var e = exp(p[unsafe_offset=base + j] - mx)
            p[unsafe_offset=base + j] = e
            total += e
        var inv = 1.0 / total
        for j in range(row_len):
            p[unsafe_offset=base + j] = p[unsafe_offset=base + j] * inv


def sigmoid(v: Float32) -> Float32:
    if v >= 0.0:
        return 1.0 / (1.0 + exp(-v))
    var e = exp(v)
    return e / (1.0 + e)


def inverse_sigmoid(v: Float32, eps: Float32) -> Float32:
    """``torch.special.logit(x, eps)``: clamp to [eps, 1-eps] then ``log(x / (1 - x))``."""
    var x = v
    if x < eps:
        x = eps
    if x > 1.0 - eps:
        x = 1.0 - eps
    return log(x / (1.0 - x))


def keep_alive(x: Tensor):
    """Extend a tensor's lifetime past the last use of a raw pointer into it.

    Mojo destroys a value at its last *use*, and `x.ptr()` is a use of `x` -- so a
    local tensor read only through its pointer is freed underneath that pointer.
    Calling this at the end of such a function moves the last use to the right place.
    """
    pass


def max_abs_diff(a: Tensor, b: Tensor) raises -> Float32:
    """Largest absolute element-wise difference; raises when the sizes differ."""
    if a.numel() != b.numel():
        raise Error(
            "max_abs_diff: size mismatch " + String(a.numel()) + " vs " + String(b.numel())
        )
    var m: Float32 = 0.0
    for i in range(a.numel()):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def max_abs(a: Tensor) -> Float32:
    """Largest absolute element, used to report differences relative to magnitude."""
    var m: Float32 = 0.0
    for i in range(a.numel()):
        var v = a[i]
        if v < 0.0:
            v = -v
        if v > m:
            m = v
    return m
