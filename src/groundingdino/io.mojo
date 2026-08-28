"""Reader for the LFT1 tensor container written by `tools/export_weights.py`.

Layout (little-endian):
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)

`weights.lft` is ~693 MB, so the payload is bulk-copied through a bitcast pointer
rather than decoded value by value.
"""

from .tensor import Tensor


def _u32(bytes: List[UInt8], off: Int) -> Int:
    var v: UInt32 = 0
    v |= UInt32(bytes[off])
    v |= UInt32(bytes[off + 1]) << 8
    v |= UInt32(bytes[off + 2]) << 16
    v |= UInt32(bytes[off + 3]) << 24
    return Int(v)


def load_lft(path: String) raises -> Dict[String, Tensor]:
    """Parse an LFT1 file into a name -> Tensor dictionary."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    var total = len(bytes)
    if (
        total < 8
        or bytes[0] != 76
        or bytes[1] != 70
        or bytes[2] != 84
        or bytes[3] != 49
    ):
        raise Error("not an LFT1 file: " + path)
    var base = bytes.unsafe_ptr()

    var out = Dict[String, Tensor]()
    var n_tensors = _u32(bytes, 4)
    var off = 8
    for _ in range(n_tensors):
        var name_len = _u32(bytes, off)
        off += 4
        var name = String("")
        for i in range(name_len):
            name += chr(Int(bytes[off + i]))
        off += name_len

        var ndim = _u32(bytes, off)
        off += 4
        var shape = List[Int]()
        var numel = 1
        for _ in range(ndim):
            var d = _u32(bytes, off)
            off += 4
            shape.append(d)
            numel *= d
        if ndim == 0:
            shape.append(1)

        var t = Tensor(shape)
        var src = (base + off).bitcast[Float32]()
        var dst = t.ptr()
        for i in range(numel):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
        off += 4 * numel
        out[name] = t^
    if off != total:
        raise Error("trailing bytes in " + path)
    return out^


@fieldwise_init
struct Config(Copyable, Movable, Writable):
    """Grounding DINO hyper-parameters decoded from the `__config__` tensor."""

    var d_model: Int
    var encoder_layers: Int
    var decoder_layers: Int
    var encoder_heads: Int
    var decoder_heads: Int
    var encoder_ffn_dim: Int
    var decoder_ffn_dim: Int
    var encoder_n_points: Int
    var decoder_n_points: Int
    var num_levels: Int
    var num_queries: Int
    var max_text_len: Int
    var pos_temperature: Float32
    var layer_norm_eps: Float32
    var text_hidden: Int
    var text_layers: Int
    var text_heads: Int
    var text_layer_norm_eps: Float32
    var swin_embed_dim: Int
    var window_size: Int
    var swin_depths: List[Int]
    var swin_heads: List[Int]

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Config(d_model=", self.d_model,
            ", enc=", self.encoder_layers,
            ", dec=", self.decoder_layers,
            ", queries=", self.num_queries,
            ", levels=", self.num_levels,
            ", bert=", self.text_layers, "x", self.text_hidden,
            ", swin=", self.swin_embed_dim, ")",
        )


def decode_config(tensors: Dict[String, Tensor]) raises -> Config:
    ref c = tensors["__config__"]
    if c.numel() != 20:
        raise Error("bad __config__ tensor")
    ref depths = tensors["__swin_depths__"]
    ref heads = tensors["__swin_heads__"]
    var d = List[Int]()
    for i in range(depths.numel()):
        d.append(Int(depths[i]))
    var h = List[Int]()
    for i in range(heads.numel()):
        h.append(Int(heads[i]))
    return Config(
        d_model=Int(c[0]),
        encoder_layers=Int(c[1]),
        decoder_layers=Int(c[2]),
        encoder_heads=Int(c[3]),
        decoder_heads=Int(c[4]),
        encoder_ffn_dim=Int(c[5]),
        decoder_ffn_dim=Int(c[6]),
        encoder_n_points=Int(c[7]),
        decoder_n_points=Int(c[8]),
        num_levels=Int(c[9]),
        num_queries=Int(c[10]),
        max_text_len=Int(c[11]),
        pos_temperature=c[12],
        layer_norm_eps=c[13],
        text_hidden=Int(c[14]),
        text_layers=Int(c[15]),
        text_heads=Int(c[16]),
        text_layer_norm_eps=c[17],
        swin_embed_dim=Int(c[18]),
        window_size=Int(c[19]),
        swin_depths=d^,
        swin_heads=h^,
    )
