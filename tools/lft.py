"""LFT1 tensor container (shared with LabelFormer.mojo / the Mojo side of this repo).

Layout (little-endian):
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def write_lft(path: Path, tensors: dict[str, np.ndarray]) -> int:
    """Write the tensors to ``path``; returns the total number of float values."""
    total = 0
    with open(path, "wb") as f:
        f.write(b"LFT1")
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            total += arr.size
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            if arr.ndim:
                f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(arr.tobytes())
    return total


def read_lft(path: Path) -> dict[str, np.ndarray]:
    """Parse an LFT1 file (used by the tools' own round-trip checks)."""
    buf = Path(path).read_bytes()
    assert buf[:4] == b"LFT1", path
    (n,) = struct.unpack_from("<I", buf, 4)
    off = 8
    out: dict[str, np.ndarray] = {}
    for _ in range(n):
        (name_len,) = struct.unpack_from("<I", buf, off)
        off += 4
        name = buf[off : off + name_len].decode()
        off += name_len
        (ndim,) = struct.unpack_from("<I", buf, off)
        off += 4
        shape = struct.unpack_from(f"<{ndim}I", buf, off) if ndim else ()
        off += 4 * ndim
        count = int(np.prod(shape)) if ndim else 1
        arr = np.frombuffer(buf, dtype=np.float32, count=count, offset=off).reshape(shape or (1,))
        off += 4 * count
        out[name] = arr
    return out
