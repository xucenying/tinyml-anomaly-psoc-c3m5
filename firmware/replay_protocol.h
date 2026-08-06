/*
 * replay_protocol.h — PC<->C3M5 raw-window streaming wire format (shared).
 *
 * CONTINUOUS ADC STREAM. Host -> board: a binary frame carrying ONE chunk of
 * RPL_HOP raw ADC samples (12-bit signed counts) — one ADC/DMA "half-buffer",
 * exactly how a real MCU delivers samples. The board appends each chunk to a
 * sliding 1024-sample window (hop = RPL_HOP) and, once the window is full,
 * classifies on every chunk: int16 -> float -> Hann + FFT (features.h) ->
 * INT8 quantize -> classify.
 *   byte 0      : SYNC0  0xA5
 *   byte 1      : SYNC1  0x5A
 *   byte 2..3   : payload length in bytes, little-endian (== 1024)
 *   byte 4..    : payload = RPL_HOP x int16 ADC counts (12-bit, [-2048,2047]), LE
 *   last 2 bytes: CRC-16/CCITT-FALSE over the payload bytes, little-endian
 *
 * Board -> host: one ASCII line per chunk (see main.cpp):
 *   during window fill:  "WARM\r\n"
 *   once classifying:    "RES <seq> <pred> <label> <conf%> <fft_cyc> <inf_cyc> <ok|ALERT>\r\n"
 *
 * Streaming raw ADC samples (not features, not windows, not model int8) lets the
 * board do its own windowing + FFT + quantize, exactly as from a live sensor.
 * At 12 kHz this is RPL_HOP/12000 s per chunk (~42.7 ms), 24 KB/s -> needs
 * ~921600 baud for real time. Apache-2.0.
 */
#pragma once
#include <stdint.h>

#define RPL_SYNC0        0xA5u
#define RPL_SYNC1        0x5Au
#define RPL_RAW_DIM      1024                     /* samples per FFT window */
#define RPL_HOP          512                      /* samples per streamed chunk */
#define RPL_PAYLOAD_LEN  (RPL_HOP * 2)            /* 1024 bytes (int16 ADC counts) */

/* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection. */
static inline uint16_t rpl_crc16(const uint8_t *p, uint32_t n)
{
    uint16_t crc = 0xFFFFu;
    for (uint32_t i = 0; i < n; ++i) {
        crc ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; ++b)
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u)
                                  : (uint16_t)(crc << 1);
    }
    return crc;
}
