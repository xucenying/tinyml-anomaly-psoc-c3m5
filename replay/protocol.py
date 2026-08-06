#!/usr/bin/env python3
"""protocol.py — PC<->C3M5 replay wire format (mirror of firmware/replay_protocol.h).

Continuous ADC stream. Host -> board frame (one chunk = one ADC half-buffer):
    A5 5A  LEN_L LEN_H  <1024 bytes = 512 int16 LE ADC counts>  CRC_L CRC_H
    CRC = CRC-16/CCITT-FALSE over the 1024 payload bytes.
Board -> host: ASCII lines, one per chunk
    "WARM"  while the 1024-sample window is still filling, then
    "RES <seq> <pred> <label> <conf%> <fft_cyc> <inf_cyc> <ok|ALERT>".

The host sends a continuous stream of raw 12-bit ADC samples in HOP-sized chunks.
The board appends each chunk to a sliding 1024-sample window and classifies once
per chunk. Apache-2.0."""
from __future__ import annotations
import struct

SYNC0, SYNC1 = 0xA5, 0x5A
HOP = 512                          # samples per streamed chunk (ADC half-buffer)
PAYLOAD_LEN = HOP * 2              # 1024 bytes (int16 ADC counts)
ADC_MIN, ADC_MAX = -2048, 2047     # 12-bit signed


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def pack_frame(chunk) -> bytes:
    """chunk: iterable of HOP ADC counts -> framed int16 bytes for the wire.
    Values are rounded and clipped to the 12-bit signed range [-2048, 2047]."""
    ints = [max(ADC_MIN, min(ADC_MAX, int(round(v)))) for v in chunk]
    payload = struct.pack("<%dh" % HOP, *ints)
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"expected {HOP} samples")
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
    """Read one chunk frame from a blocking read(n) callable.
    Returns (chunk: list[int] | None, crc_ok: bool). chunk is None on desync."""
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
    return list(struct.unpack("<%dh" % HOP, payload)), ok
