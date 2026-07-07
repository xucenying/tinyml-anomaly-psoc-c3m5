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

#include "model_data.h"     /* INT8 CWRU classifier (24 KB) */
#include "test_vectors.h"   /* 10 validation feature vectors */

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
    printf("model: %u bytes INT8\r\n", (unsigned)g_model_data_len);

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

    int correct = 0;
    uint64_t total_cycles = 0;

    for (int t = 0; t < kNumTests; ++t) {
        /* quantize the float feature vector to int8 (on-device, like live path) */
        for (int i = 0; i < kFeatureDim; ++i) {
            float q = kTestVectors[t][i] / in_scale + (float)in_zp;
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            in->data.int8[i] = (int8_t)q;
        }

        uint32_t t0 = dwt_now();
        if (interpreter.Invoke() != kTfLiteOk) halt("Invoke failed");
        uint32_t dt = dwt_now() - t0;
        total_cycles += dt;

        int best = 0;
        for (int i = 1; i < kNumTests; ++i)
            if (out->data.int8[i] > out->data.int8[best]) best = i;
        int conf_pct = (int)(((float)out->data.int8[best] - (float)out_zp)
                             * out_scale * 100.0f);
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
    for (;;) {}
}
