"""Binary encoding primitives shared by the on-disk format.

Posting lists, term dictionaries and the stored document store are all built on
a small set of variable-length integer encodings. Keeping them in one place
means the readers and writers agree on the byte layout by construction.

Encodings
---------
``uvarint``
    LEB128 unsigned variable-length integer. Small numbers take one byte, which
    is what we want for the delta-encoded document ids and term frequencies that
    dominate a posting list.
``varint``
    Signed variable-length integer using zig-zag mapping, so small-magnitude
    negative numbers stay small.
Fixed widths
    ``u32``/``u64`` little-endian, used for file offsets and checksums where a
    predictable size matters more than compactness.
"""

from __future__ import annotations

import struct
import zlib
from typing import Tuple

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


def zigzag_encode(value: int) -> int:
    """Map a signed integer onto an unsigned one with small magnitudes small."""
    return (value << 1) ^ (value >> 63) if value < 0 else value << 1


def zigzag_decode(value: int) -> int:
    """Inverse of :func:`zigzag_encode`."""
    return (value >> 1) ^ -(value & 1)


def encode_uvarint(value: int) -> bytes:
    """Encode a non-negative integer as an unsigned LEB128 varint."""
    if value < 0:
        raise ValueError("uvarint cannot encode negative values")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_uvarint(buf: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode an unsigned varint from ``buf`` at ``offset``.

    Returns ``(value, next_offset)``. Raises :class:`ValueError` if the buffer
    ends in the middle of a varint.
    """
    result = 0
    shift = 0
    pos = offset
    while True:
        if pos >= len(buf):
            raise ValueError("truncated uvarint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("uvarint overflows 64 bits")


def encode_varint(value: int) -> bytes:
    """Encode a signed integer (zig-zag + uvarint)."""
    return encode_uvarint(zigzag_encode(value))


def decode_varint(buf: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode a signed varint; returns ``(value, next_offset)``."""
    raw, pos = decode_uvarint(buf, offset)
    return zigzag_decode(raw), pos


def crc32(data: bytes) -> int:
    """Return the CRC-32 of ``data`` as an unsigned 32-bit integer."""
    return zlib.crc32(data) & 0xFFFFFFFF


class ByteWriter:
    """Append-only buffer with helpers for the encodings above."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def __len__(self) -> int:
        return len(self._buf)

    def tell(self) -> int:
        """Current length, i.e. the offset the next write will start at."""
        return len(self._buf)

    def raw(self, data: bytes) -> "ByteWriter":
        self._buf += data
        return self

    def u32(self, value: int) -> "ByteWriter":
        self._buf += _U32.pack(value)
        return self

    def u64(self, value: int) -> "ByteWriter":
        self._buf += _U64.pack(value)
        return self

    def uvarint(self, value: int) -> "ByteWriter":
        self._buf += encode_uvarint(value)
        return self

    def varint(self, value: int) -> "ByteWriter":
        self._buf += encode_varint(value)
        return self

    def bytes(self, data: bytes) -> "ByteWriter":
        """Length-prefixed byte string."""
        self._buf += encode_uvarint(len(data))
        self._buf += data
        return self

    def string(self, text: str) -> "ByteWriter":
        """Length-prefixed UTF-8 string."""
        return self.bytes(text.encode("utf-8"))

    def getvalue(self) -> bytes:
        return bytes(self._buf)


class ByteReader:
    """Cursor over a bytes buffer, mirroring :class:`ByteWriter`."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, buf: bytes, offset: int = 0) -> None:
        self._buf = buf
        self._pos = offset

    @property
    def pos(self) -> int:
        return self._pos

    def seek(self, offset: int) -> None:
        self._pos = offset

    def at_end(self) -> bool:
        return self._pos >= len(self._buf)

    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def raw(self, n: int) -> bytes:
        if self._pos + n > len(self._buf):
            raise ValueError("read past end of buffer")
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def u32(self) -> int:
        (value,) = _U32.unpack_from(self._buf, self._pos)
        self._pos += 4
        return value

    def u64(self) -> int:
        (value,) = _U64.unpack_from(self._buf, self._pos)
        self._pos += 8
        return value

    def uvarint(self) -> int:
        value, self._pos = decode_uvarint(self._buf, self._pos)
        return value

    def varint(self) -> int:
        value, self._pos = decode_varint(self._buf, self._pos)
        return value

    def bytes(self) -> bytes:
        n = self.uvarint()
        return self.raw(n)

    def string(self) -> str:
        return self.bytes().decode("utf-8")
