#!/usr/bin/env python3
"""protocol.py — PC<->C3M5 replay wire format (mirror of firmware/replay_protocol.h).

Host -> board frame:
    A5 5A  LEN_L LEN_H  <512 bytes = 128 float32 LE>  CRC_L CRC_H
    CRC = CRC-16/CCITT-FALSE over the 512 payload bytes.
Board -> host: ASCII lines "RES <seq> <pred> <label> <conf%> <cycles> <ok|ALERT>".

Apache-2.0."""
from __future__ import annotations
import struct

SYNC0, SYNC1 = 0xA5, 0x5A
FEATURE_DIM = 128
PAYLOAD_LEN = FEATURE_DIM * 4  # 512


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def pack_frame(feat) -> bytes:
    """feat: iterable of 128 floats -> framed bytes ready for the wire."""
    payload = struct.pack("<128f", *feat)
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"expected {FEATURE_DIM} floats")
    return bytes([SYNC0, SYNC1]) + struct.pack("<H", PAYLOAD_LEN) + payload \
        + struct.pack("<H", crc16(payload))


def read_exact(readfn, n: int) -> bytes:
    """Read exactly n bytes using a blocking read(n)-style callable."""
    buf = b""
    while len(buf) < n:
        chunk = readfn(n - len(buf))
        if not chunk:
            raise EOFError("stream closed")
        buf += chunk
    return buf


def read_frame(readfn):
    """Read one frame from a blocking read(n) callable.
    Returns (feats: list[float] | None, crc_ok: bool). feats is None on desync."""
    # hunt for SYNC0 SYNC1
    while True:
        if read_exact(readfn, 1)[0] != SYNC0:
            continue
        if read_exact(readfn, 1)[0] == SYNC1:
            break
    (length,) = struct.unpack("<H", read_exact(readfn, 2))
    if length != PAYLOAD_LEN:
        return None, False
    payload = read_exact(readfn, PAYLOAD_LEN)
    (crc,) = struct.unpack("<H", read_exact(readfn, 2))
    ok = crc == crc16(payload)
    return list(struct.unpack("<128f", payload)), ok
