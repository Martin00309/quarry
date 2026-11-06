"""Tests for the binary encoding primitives."""

from __future__ import annotations

import pytest

from quarry.coding import (
    ByteReader,
    ByteWriter,
    crc32,
    decode_uvarint,
    decode_varint,
    encode_uvarint,
    encode_varint,
    zigzag_decode,
    zigzag_encode,
)


@pytest.mark.parametrize("value", [0, 1, 127, 128, 255, 300, 16384, 2**31, 2**63 - 1])
def test_uvarint_round_trip(value):
    encoded = encode_uvarint(value)
    decoded, offset = decode_uvarint(encoded)
    assert decoded == value
    assert offset == len(encoded)


def test_uvarint_is_compact_for_small_numbers():
    assert len(encode_uvarint(0)) == 1
    assert len(encode_uvarint(127)) == 1
    assert len(encode_uvarint(128)) == 2


def test_uvarint_rejects_negative():
    with pytest.raises(ValueError):
        encode_uvarint(-1)


def test_decode_uvarint_truncated():
    with pytest.raises(ValueError):
        decode_uvarint(b"\x80")  # continuation bit set but no following byte


@pytest.mark.parametrize("value", [0, -1, 1, -127, 127, -128, 128, -(2**40), 2**40])
def test_signed_varint_round_trip(value):
    decoded, _ = decode_varint(encode_varint(value))
    assert decoded == value


@pytest.mark.parametrize("value", [0, -1, 1, -1000, 1000])
def test_zigzag_is_invertible(value):
    assert zigzag_decode(zigzag_encode(value)) == value


def test_zigzag_keeps_small_magnitudes_small():
    assert zigzag_encode(-1) == 1
    assert zigzag_encode(1) == 2


def test_writer_reader_mixed_round_trip():
    writer = ByteWriter()
    writer.u32(42).u64(10**12).uvarint(300).varint(-7).string("héllo").bytes(b"\x00\x01")
    reader = ByteReader(writer.getvalue())
    assert reader.u32() == 42
    assert reader.u64() == 10**12
    assert reader.uvarint() == 300
    assert reader.varint() == -7
    assert reader.string() == "héllo"
    assert reader.bytes() == b"\x00\x01"
    assert reader.at_end()


def test_reader_seek_and_remaining():
    writer = ByteWriter().uvarint(1).uvarint(2)
    reader = ByteReader(writer.getvalue())
    assert reader.remaining() == 2
    reader.uvarint()
    reader.seek(0)
    assert reader.uvarint() == 1


def test_reader_raises_past_end():
    reader = ByteReader(b"\x01")
    with pytest.raises(ValueError):
        reader.raw(5)


def test_crc32_changes_with_content():
    assert crc32(b"hello") != crc32(b"world")
    assert crc32(b"hello") == crc32(b"hello")
