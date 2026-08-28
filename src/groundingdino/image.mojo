"""Standalone image input: binary PPM (P6) reading, resizing and ImageNet normalization.

JPEG/PNG decoding is out of scope, so the standalone CLI takes a binary PPM. Convert
with any of:

    convert photo.jpg photo.ppm            # ImageMagick
    ffmpeg -i photo.jpg photo.ppm
    python -c "from PIL import Image; Image.open('photo.jpg').convert('RGB').save('photo.ppm')"

The resize follows `get_size_with_aspect_ratio(shortest_edge=800, longest_edge=1333)`
and then a plain `align_corners=False` bilinear resample. The HF processor resamples
through PIL/torchvision, which applies an antialiasing filter when downscaling, so the
pixels here are close but not bit-identical -- which is exactly why the parity fixtures
carry the already-preprocessed `pixel_values` instead of going through this path.
"""

from .tensor import Tensor

comptime IMAGENET_MEAN = SIMD[DType.float32, 4](0.485, 0.456, 0.406, 0.0)
comptime IMAGENET_STD = SIMD[DType.float32, 4](0.229, 0.224, 0.225, 1.0)


@fieldwise_init
struct Image(Copyable, Movable, Writable):
    """An 8-bit RGB image stored as `(3, H, W)` float values in [0, 255]."""

    var data: Tensor
    var height: Int
    var width: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Image(", self.width, "x", self.height, ")")


def _skip_ws_and_comments(bytes: List[UInt8], mut off: Int):
    while off < len(bytes):
        var c = bytes[off]
        if c == 35:  # '#'
            while off < len(bytes) and bytes[off] != 10:
                off += 1
        elif c == 32 or c == 9 or c == 10 or c == 13:
            off += 1
        else:
            return


def _read_int(bytes: List[UInt8], mut off: Int) raises -> Int:
    _skip_ws_and_comments(bytes, off)
    var value = 0
    var digits = 0
    while off < len(bytes) and bytes[off] >= 48 and bytes[off] <= 57:
        value = value * 10 + Int(bytes[off]) - 48
        off += 1
        digits += 1
    if digits == 0:
        raise Error("PPM: expected an integer in the header")
    return value


def read_ppm(path: String) raises -> Image:
    """Read a binary P6 PPM with maxval 255."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) < 10 or bytes[0] != 80 or bytes[1] != 54:
        raise Error("not a binary PPM (P6): " + path)
    var off = 2
    var width = _read_int(bytes, off)
    var height = _read_int(bytes, off)
    var maxval = _read_int(bytes, off)
    if maxval != 255:
        raise Error("PPM: only maxval 255 is supported, got " + String(maxval))
    off += 1  # exactly one whitespace byte separates the header from the payload
    if off + 3 * width * height > len(bytes):
        raise Error("PPM: truncated pixel data")

    var data = Tensor.zeros([3, height, width])
    for y in range(height):
        for x in range(width):
            var base = off + (y * width + x) * 3
            for c in range(3):
                data.set3(c, y, x, Float32(Int(bytes[base + c])))
    return Image(data^, height, width)


def target_size(height: Int, width: Int, shortest: Int, longest: Int) -> Tuple[Int, Int]:
    """`transformers.image_transforms.get_size_with_aspect_ratio`."""
    var size = shortest
    var raw_size = Float32(0.0)
    var has_raw = False
    var min_original = Float32(height if height < width else width)
    var max_original = Float32(height if height > width else width)
    if max_original / min_original * Float32(size) > Float32(longest):
        raw_size = Float32(longest) * min_original / max_original
        size = Int(raw_size + 0.5)
        has_raw = True

    if (height <= width and height == size) or (width <= height and width == size):
        return (height, width)
    if width < height:
        var out_w = size
        var scale = raw_size if has_raw else Float32(size)
        return (Int(scale * Float32(height) / Float32(width)), out_w)
    var out_h = size
    var scale2 = raw_size if has_raw else Float32(size)
    return (out_h, Int(scale2 * Float32(width) / Float32(height)))


def resize_bilinear(image: Image, out_h: Int, out_w: Int) raises -> Image:
    """`align_corners=False` bilinear resample with edge clamping."""
    var out = Tensor.zeros([3, out_h, out_w])
    var scale_y = Float32(image.height) / Float32(out_h)
    var scale_x = Float32(image.width) / Float32(out_w)
    for oy in range(out_h):
        var sy = (Float32(oy) + 0.5) * scale_y - 0.5
        if sy < 0.0:
            sy = 0.0
        var y0 = Int(sy)
        var y1 = y0 + 1
        if y1 > image.height - 1:
            y1 = image.height - 1
        var wy = sy - Float32(y0)
        for ox in range(out_w):
            var sx = (Float32(ox) + 0.5) * scale_x - 0.5
            if sx < 0.0:
                sx = 0.0
            var x0 = Int(sx)
            var x1 = x0 + 1
            if x1 > image.width - 1:
                x1 = image.width - 1
            var wx = sx - Float32(x0)
            for c in range(3):
                var v00 = image.data.at3(c, y0, x0)
                var v01 = image.data.at3(c, y0, x1)
                var v10 = image.data.at3(c, y1, x0)
                var v11 = image.data.at3(c, y1, x1)
                var top = v00 + (v01 - v00) * wx
                var bottom = v10 + (v11 - v10) * wx
                out.set3(c, oy, ox, top + (bottom - top) * wy)
    return Image(out^, out_h, out_w)


def normalize(image: Image) raises -> Tensor:
    """Rescale by 1/255 then apply the ImageNet mean/std, giving `(3, H, W)`."""
    var out = Tensor.zeros([3, image.height, image.width])
    for c in range(3):
        var m = IMAGENET_MEAN[c]
        var s = IMAGENET_STD[c]
        for y in range(image.height):
            for x in range(image.width):
                out.set3(c, y, x, (image.data.at3(c, y, x) / 255.0 - m) / s)
    return out^


def preprocess(path: String, shortest: Int, longest: Int) raises -> Tuple[Tensor, Int, Int]:
    """Read a PPM and produce `(pixel_values (3, H, W), original_height, original_width)`."""
    var image = read_ppm(path)
    var size = target_size(image.height, image.width, shortest, longest)
    var resized = resize_bilinear(image, size[0], size[1])
    return (normalize(resized), image.height, image.width)
