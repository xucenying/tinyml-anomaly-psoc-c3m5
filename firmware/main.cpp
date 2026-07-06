/*
 * main.cpp — TFLite-Micro smoke test for PSOC Control C3M5 (KIT_PSC3M5_EVK)
 * Runs the TFLM "hello world" sine model: predicts sin(x) for x in [0, 2π).
 * Success = UART prints predicted vs true values. Apache-2.0.
 *
 * Status: NOT yet hardware-tested. Bring build errors back to Claude.
 */
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <cstdio>
#include <cmath>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   /* generated: convert_tflite_to_c.py -> g_model_data[] */

/* Tensor arena: working memory for activations. Sine model needs ~2-4 KB;
 * our real anomaly model will need more. 64 KB SRAM total — budget carefully. */
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

int main(void)
{
    if (cybsp_init() != CY_RSLT_SUCCESS) { for (;;) {} }
    __enable_irq();

    /* UART for printf via KitProg3 (adjust TX/RX macros if BSP names differ) */
    cy_retarget_io_init(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX, 115200);

    printf("\r\n=== TFLM smoke test: PSOC Control C3M5 ===\r\n");

    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Schema mismatch: model %lu, runtime %d\r\n",
               (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        for (;;) {}
    }

    /* Register ONLY the ops the model uses — keeps flash small.
     * hello_world uses FullyConnected only. */
    static tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter interpreter(model, resolver,
                                                tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed (arena too small?)\r\n");
        for (;;) {}
    }
    printf("Arena used: %u / %d bytes\r\n",
           (unsigned)interpreter.arena_used_bytes(), kTensorArenaSize);

    TfLiteTensor *in  = interpreter.input(0);
    TfLiteTensor *out = interpreter.output(0);

    /* hello_world_int8: quantized int8 input/output */
    const float in_scale  = in->params.scale;
    const int   in_zp     = in->params.zero_point;
    const float out_scale = out->params.scale;
    const int   out_zp    = out->params.zero_point;

    for (int i = 0; i < 16; ++i) {
        const float x = (6.2831853f * i) / 16.0f;
        in->data.int8[0] = (int8_t)(x / in_scale + in_zp);

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Invoke failed at i=%d\r\n", i);
            for (;;) {}
        }
        const float y = (out->data.int8[0] - out_zp) * out_scale;
        /* int math for printf portability: values x100 */
        printf("x=%d.%02u  sin_pred=%s%d.%02u  sin_true=%s%d.%02u\r\n",
               (int)x, (unsigned)((x - (int)x) * 100),
               y < 0 ? "-" : "", (int)fabsf(y), (unsigned)((fabsf(y) - (int)fabsf(y)) * 100),
               sinf(x) < 0 ? "-" : "", (int)fabsf(sinf(x)),
               (unsigned)((fabsf(sinf(x)) - (int)fabsf(sinf(x))) * 100));
    }

    printf("=== smoke test complete ===\r\n");
    for (;;) {}
}
