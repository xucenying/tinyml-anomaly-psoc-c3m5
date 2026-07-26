/*
 * replay_protocol.h — PC<->C3M5 feature-streaming wire format (shared).
 *
 * Host -> board: a binary frame carrying one 128-float feature window.
 *   byte 0      : SYNC0  0xA5
 *   byte 1      : SYNC1  0x5A
 *   byte 2..3   : payload length in bytes, little-endian (== 512)
 *   byte 4..    : payload = 128 x float32, little-endian (the feature vector)
 *   last 2 bytes: CRC-16/CCITT-FALSE over the payload bytes, little-endian
 *
 * Board -> host: one ASCII status line per processed frame (see main.cpp):
 *   "RES <seq> <pred> <label> <conf%> <cycles> <ok|ALERT>\r\n"
 *
 * Float32 (not pre-quantized int8) is sent so the board performs the same
 * quantization as the on-device benchmark — inference numbers stay comparable.
 * Apache-2.0.
 */
#pragma once
#include <stdint.h>

#define RPL_SYNC0        0xA5u
#define RPL_SYNC1        0x5Au
#define RPL_FEATURE_DIM  128
#define RPL_PAYLOAD_LEN  (RPL_FEATURE_DIM * 4)   /* 512 bytes */

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
