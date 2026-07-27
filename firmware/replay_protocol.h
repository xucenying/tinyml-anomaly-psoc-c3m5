/*
 * replay_protocol.h — PC<->C3M5 raw-window streaming wire format (shared).
 *
 * Host -> board: a binary frame carrying ONE raw 1024-sample vibration window
 * (what a sensor+ADC would hand you). The board runs the FULL on-chip pipeline
 * on it: Hann + FFT feature extraction (features.h) -> INT8 quantize -> classify.
 *   byte 0      : SYNC0  0xA5
 *   byte 1      : SYNC1  0x5A
 *   byte 2..3   : payload length in bytes, little-endian (== 4096)
 *   byte 4..    : payload = 1024 x float32 raw samples, little-endian
 *   last 2 bytes: CRC-16/CCITT-FALSE over the payload bytes, little-endian
 *
 * Board -> host: one ASCII status line per processed frame (see main.cpp):
 *   "RES <seq> <pred> <label> <conf%> <fft_cyc> <inf_cyc> <ok|ALERT>\r\n"
 *
 * Raw float32 samples are sent (not features, not pre-quantized int8) so the
 * board performs the SAME FFT + quantization as the on-device benchmark — the
 * demo emulates a real sensor and exercises the whole optimized pipeline.
 * Apache-2.0.
 */
#pragma once
#include <stdint.h>

#define RPL_SYNC0        0xA5u
#define RPL_SYNC1        0x5Au
#define RPL_RAW_DIM      1024                     /* raw samples per window */
#define RPL_PAYLOAD_LEN  (RPL_RAW_DIM * 4)        /* 4096 bytes */

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
