"""The packed-nibble genome representation used by the benchmark."""

import numpy as np

# This intentionally matches zarr_compression_benchmark.py: A=0, C=1, G=2,
# T=3, and N (including unsupported IUPAC bases) = 4.  The first base is the
# high nibble of each byte.
_BASE_CODES = np.full(256, 4, dtype=np.uint8)
for _base, _code in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3), (b"N", 4)):
    _BASE_CODES[_base[0]] = _code
    _BASE_CODES[_base.lower()[0]] = _code


def encode_4bit(sequence: str) -> bytes:
    """Encode two bases per byte, padding a final odd base with zero."""
    values = _BASE_CODES[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if len(values) % 2:
        values = np.pad(values, (0, 1), constant_values=0)
    return ((values[0::2] << 4) | values[1::2]).tobytes()


def decode_4bit(payload: bytes, logical_length: int) -> str:
    """Decode packed bytes. Mainly useful for validation and consumers."""
    symbols = np.frombuffer(b"ACGTN", dtype="S1")
    packed = np.frombuffer(payload, dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = packed >> 4
    codes[1::2] = packed & 0x0F
    # Unknown codes are deliberately represented as N on decoding.
    codes[codes > 4] = 4
    return b"".join(symbols[codes[:logical_length]]).decode("ascii")
