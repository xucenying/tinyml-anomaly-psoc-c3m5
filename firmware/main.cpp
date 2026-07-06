/*
 * main.cpp — TFLite-Micro smoke test for PSOC Control C3M5 (KIT_PSC3M5_EVK)
 * Runs the TFLM "hello world" sine model: predicts sin(x) for x in [0, 2π).
 * UART init mirrors Infineon's mtb-example-ce240510-hello-world (PDL + HAL).
 * Apache-2.0.
 */
#include "cy_pdl.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "mtb_hal.h"

#include <cstdio>
#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   /* generated: convert_tflite_to_c.py -> g_model_data[] */

/* Retarget-IO (Debug UART) — same objects as the stock Hello World example */
static cy_stc_scb_uart_context_t DEBUG_UART_context;
static mtb_hal_uart_t            DEBUG_UART_hal_obj;

/* Tensor arena: working memory for activations. Sine model needs ~2-4 KB;
 * our real anomaly model will need more. 64 KB SRAM total — budget carefully. */
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static void halt(const char *msg)
{
    printf("FATAL: %s\r\n", msg);
    for (;;) {}
}

int main(void)
{
    /* Board init (clocks, pins) */
    if (cybsp_init() != CY_RSLT_SUCCESS) { CY_ASSERT(0); }
    __enable_irq();

    /* Debug UART init — PDL first, then HAL wrapper, then retarget printf */
    if (Cy_SCB_UART_Init(DEBUG_UART_HW, &DEBUG_UART_config,
                         &DEBUG_UART_context) != CY_SCB_UART_SUCCESS) { CY_ASSERT(0); }
    Cy_SCB_UART_Enable(DEBUG_UART_HW);
    if (mtb_hal_uart_setup(&DEBUG_UART_hal_obj, &DEBUG_UART_hal_config,
                           &DEBUG_UART_context, NULL) != CY_RSLT_SUCCESS) { CY_ASSERT(0); }
    if (cy_retarget_io_init(&DEBUG_UART_hal_obj) != CY_RSLT_SUCCESS) { CY_ASSERT(0); }

    printf("\x1b[2J\x1b[;H");   /* clear terminal */
    printf("=== TFLM smoke test: PSOC Control C3M5 ===\r\n");

    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) halt("schema mismatch");

    /* Register ONLY the ops the model uses — keeps flash small.
     * hello_world uses FullyConnected only. */
    static tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter interpreter(model, resolver,
                                                tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk)
        halt("AllocateTensors failed (arena too small?)");

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
        const float x = (6.2831853f * (float)i) / 16.0f;
        in->data.int8[0] = (int8_t)(x / in_scale + (float)in_zp);

        if (interpreter.Invoke() != kTfLiteOk) halt("Invoke failed");

        const float y  = ((float)out->data.int