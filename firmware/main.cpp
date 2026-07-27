/*
 * main.cpp — CWRU bearing-fault classifier on PSOC Control C3M5 (KIT_PSC3M5_EVK)
 * INT8 MLP (128 FFT features -> 96 -> 48 -> 10 classes), TFLite-Micro.
 * Runs 10 embedded validation vectors (one per class) and reports
 * prediction, confidence, and DWT cycle count per inference. Apache-2.0.
 */
#include "cy_pdl.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "mtb_hal.h"

#include <cstdio>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Pick ONE: model_data.h (INT8, 24 KB) or model_data_fp32.h (FP32, 72 KB) */
#include "model_data.h"
#include "test_vectors.h"      /* 10 validation feature vectors */
#include "fft_selftest.h"      /* on-board FFT feature extraction test */
#include "replay_protocol.h"   /* PC<->board streaming wire format */

#include <cstring>

/* ---- class table (id -> label); "normal" is the healthy class ---- */
static const char *const kClassLabels[10] = {
    "b_007", "b_014", "b_021", "ir_007", "ir_014", "ir_021",
    "normal", "or_007", "or_014", "or_021"};
static constexpr int kNormalClass = 6;

/* ---- alert-logic tuning (see run_replay) ---- */
static constexpr int kConfThreshPct = 60;  /* min confidence to trust a call     */
static constexpr int kDebounceFault = 3;   /* consecutive faults to latch ALERT  */
static constexpr int kDebounceClear = 5;   /* consecutive normals to clear ALERT */

static cy_stc_scb_uart_context_t DEBUG_UART_context;
static mtb_hal_uart_t            DEBUG_UART_hal_obj;

constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

/* --- minimal DWT cycle counter (full harness lands in step 1.9) --- */
static inline void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_now(void) { return DWT->CYCCNT; }

static void halt(const char *msg) { printf("FATAL: %s\r\n", msg); for (;;) {} }

/* --- board LED: on = fault latched (guarded so it compiles on any BSP) --- */
static inline void led_set(bool on) {
#if defined(CYBSP_USER_LED_PORT) && defined(CYBSP_USER_LED_PIN)
  #if defined(CYBSP_LED_STATE_ON)
    Cy_GPIO_Write(CYBSP_USER_LED_PORT, CYBSP_USER_LED_PIN,
                  on ? CYBSP_LED_STATE_ON : CYBSP_LED_STATE_OFF);
  #else
    Cy_GPIO_Write(CYBSP_USER_LED_PORT, CYBSP_USER_LED_PIN, on ? 0u : 1u);
  #endif
#else
    (void)on;   /* no user LED on this BSP — status still goes out over UART */
#endif
}

/* --- CAN-FD alert frame. Real TX is board-specific; stub prints the frame
 *     it would send so the pipeline is provable without a bus. Define
 *     CANFD_ENABLE + wire mtb_hal_canfd to send for real. --- */
static void emit_canfd_alert(int fault_id, int conf_pct) {
    uint8_t d[4] = {0xFA, (uint8_t)fault_id, (uint8_t)conf_pct, 0x01};
    printf("CANFD id=0x7DF dlc=4 data=%02X %02X %02X %02X\r\n",
           d[0], d[1], d[2], d[3]);
#if defined(CANFD_ENABLE)
    /* TODO: mtb_hal_canfd_transmit(&canfd_obj, &frame); */
#endif
}

/* --- blocking single-byte UART read (shares the retarget-io SCB) --- */
static inline uint8_t uart_get_byte(void) {
    uint32_t b;
    do { b = Cy_SCB_UART_Get(DEBUG_UART_HW); } while (b == CY_SCB_UART_RX_NO_DATA);
    return (uint8_t)b;
}

/* Read one protocol frame into feat[128]. Returns true on a CRC-valid frame,
 * false on a CRC mismatch (caller reports and keeps streaming). Blocks until a
 * well-formed SYNC/len header arrives, so a stalled host just pauses the loop. */
static bool read_frame(float *feat) {
    /* hunt for SYNC0 SYNC1 */
    for (;;) {
        if (uart_get_byte() != RPL_SYNC0) continue;
        if (uart_get_byte() == RPL_SYNC1) break;
    }
    uint16_t len = (uint16_t)uart_get_byte();
    len |= (uint16_t)uart_get_byte() << 8;
    if (len != RPL_PAYLOAD_LEN) return false;   /* desync; resync on next call */

    uint8_t *raw = (uint8_t *)feat;             /* 512 bytes, LE float32 in place */
    for (uint16_t i = 0; i < RPL_PAYLOAD_LEN; ++i) raw[i] = uart_get_byte();

    uint16_t crc = (uint16_t)uart_get_byte();
    crc |= (uint16_t)uart_get_byte() << 8;
    return crc == rpl_crc16(raw, RPL_PAYLOAD_LEN);
}

/* Continuous replay: read frame -> quantize -> invoke -> alert state machine.
 * Never returns. Alert latches after kDebounceFault consecutive confident
 * faults and clears after kDebounceClear consecutive normals (hysteresis kills
 * single-frame blips). */
static void run_replay(tflite::MicroInterpreter &interpreter,
                       TfLiteTensor *in, TfLiteTensor *out,
                       float in_scale, int in_zp,
                       float out_scale, int out_zp) {
    static float feat[RPL_FEATURE_DIM];
    uint32_t seq = 0;
    int fault_run = 0, normal_run = 0;
    bool alert = false;

    printf("\r\n=== replay mode: streaming frames (115200 8N1) ===\r\n");
    led_set(false);

    for (;;) {
        if (!read_frame(feat)) { printf("ERR crc\r\n"); continue; }

        if (in->type == kTfLiteInt8) {
            for (int i = 0; i < RPL_FEATURE_DIM; ++i) {
                float q = feat[i] / in_scale + (float)in_zp;
                if (q > 127.0f) q = 127.0f; else if (q < -128.0f) q = -128.0f;
                in->data.int8[i] = (int8_t)q;
            }
        } else {
            for (int i = 0; i < RPL_FEATURE_DIM; ++i) in->data.f[i] = feat[i];
        }

        uint32_t t0 = dwt_now();
        if (interpreter.Invoke() != kTfLiteOk) halt("Invoke failed");
        uint32_t dt = dwt_now() - t0;

        int best = 0;
        if (out->type == kTfLiteInt8) {
            for (int i = 1; i < 10; ++i)
                if (out->data.int8[i] > out->data.int8[best]) best = i;
        } else {
            for (int i = 1; i < 10; ++i)
                if (out->data.f[i] > out->data.f[best]) best = i;
        }
        int conf_pct = (out->type == kTfLiteInt8)
            ? (int)(((float)out->data.int8[best] - (float)out_zp) * out_scale * 100.0f)
            : (int)(out->data.f[best] * 100.0f);

        bool is_fault = (best != kNormalClass) && (conf_pct >= kConfThreshPct);
        if (is_fault) { fault_run++; normal_run = 0; }
        else          { normal_run++; fault_run = 0; }

        bool prev = alert;
        if (!alert && fault_run  >= kDebounceFault) alert = true;
        if ( alert && normal_run >= kDebounceClear) alert = false;
        if (alert != prev) {
            led_set(alert);
            if (alert) emit_canfd_alert(best, conf_pct);
        }

        printf("RES %lu %d %-7s %3d%% %lu %s\r\n",
               (unsigned long)seq++, best, kClassLabels[best], conf_pct,
               (unsigned long)dt, alert ? "ALERT" : "ok");
    }
}

int main(void)
{
    if (cybsp_init() != CY_RSLT_SUCCESS) { CY_ASSERT(0); }
    __enable_irq();

    if (Cy_SCB_UART_Init(DEBUG_UART_HW, &DEBUG_UART_config,
                         &DEBUG_UART_context) != CY_SCB_UART_SUCCESS) { CY_ASSERT(0); }
    Cy_SCB_UART_Enable(DEBUG_UART_HW);
    if (mtb_hal_uart_setup(&DEBUG_UART_hal_obj, &DEBUG_UART_hal_config,
                           &DEBUG_UART_context, NULL) != CY_RSLT_SUCCESS) { CY_ASSERT(0); }
    if (cy_retarget_io_init(&DEBUG_UART_hal_obj) != CY_RSLT_SUCCESS) { CY_ASSERT(0); }

    dwt_init();

    printf("\x1b[2J\x1b[;H");
    printf("=== CWRU bearing-fault classifier: PSOC Control C3M5 ===\r\n");
    printf("model: %u bytes\r\n", (unsigned)g_model_data_len);

    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) halt("schema mismatch");

    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter interpreter(model, resolver,
                                                tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) halt("AllocateTensors failed");
    printf("arena used: %u / %d bytes\r\n\r\n",
           (unsigned)interpreter.arena_used_bytes(), kTensorArenaSize);

    TfLiteTensor *in  = interpreter.input(0);
    TfLiteTensor *out = interpreter.output(0);
    const float in_scale  = in->params.scale;
    const int   in_zp     = in->params.zero_point;
    const float out_scale = out->params.scale;
    const int   out_zp    = out->params.zero_point;

    /* on-board FFT feature extraction test (raw window -> FFT -> classify) */
    fft_selftest(interpreter, in, out);

    int correct = 0;
    uint64_t total_cycles = 0;

    for (int t = 0; t < kNumTests; ++t) {
        /* feed input: quantize for INT8 models, copy for FP32 models */
        if (in->type == kTfLiteInt8) {
            for (int i = 0; i < kFeatureDim; ++i) {
                float q = kTestVectors[t][i] / in_scale + (float)in_zp;
                if (q > 127.0f) q = 127.0f;
                if (q < -128.0f) q = -128.0f;
                in->data.int8[i] = (int8_t)q;
            }
        } else {
            for (int i = 0; i < kFeatureDim; ++i)
                in->data.f[i] = kTestVectors[t][i];
        }

        uint32_t t0 = dwt_now();
        if (interpreter.Invoke() != kTfLiteOk) halt("Invoke failed");
        uint32_t dt = dwt_now() - t0;
        total_cycles += dt;

        int best = 0;
        if (out->type == kTfLiteInt8) {
            for (int i = 1; i < kNumTests; ++i)
                if (out->data.int8[i] > out->data.int8[best]) best = i;
        } else {
            for (int i = 1; i < kNumTests; ++i)
                if (out->data.f[i] > out->data.f[best]) best = i;
        }
        int conf_pct = (out->type == kTfLiteInt8)
            ? (int)(((float)out->data.int8[best] - (float)out_zp) * out_scale * 100.0f)
            : (int)(out->data.f[best] * 100.0f);
        bool ok = (best == kTestExpected[t]);
        correct += ok;

        printf("[%s] true=%-7s pred=%-7s conf=%3d%%  cycles=%lu (%lu us)\r\n",
               ok ? "PASS" : "FAIL", kTestLabels[t], kTestLabels[best], conf_pct,
               (unsigned long)dt, (unsigned long)(dt / 240));
    }

    printf("\r\n%d/%d correct, avg %lu cycles/inference (%lu us @240MHz)\r\n",
           correct, kNumTests,
           (unsigned long)(total_cycles / kNumTests),
           (unsigned long)(total_cycles / kNumTests / 240));
    printf("=== classifier test complete ===\r\n");

    /* Hand off to continuous replay: PC streams feature windows, board
     * classifies each and raises a debounced fault alert (LED + CAN-FD). */
    run_replay(interpreter, in, out, in_scale, in_zp, out_scale, out_zp);
    return 0;   /* unreachable */
}
