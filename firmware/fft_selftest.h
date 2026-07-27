/*
 * fft_selftest.h — end-to-end on-board test: raw window -> FFT features
 * -> classify. Times the FFT step (feature extraction) and the inference
 * step separately with the DWT counter.
 *
 * Build the plain-C FFT and the CMSIS-DSP FFT (define FE_USE_CMSIS) and
 * compare the "fft" cycle column — that's the CMSIS-DSP rung.
 *
 * Call fft_selftest(interpreter, in, out) after AllocateTensors().
 * Requires features.h, raw_vectors.h, and a dwt_now() in scope. Apache-2.0.
 */
#ifndef FFT_SELFTEST_H
#define FFT_SELFTEST_H

#include "features.h"
#include "raw_vectors.h"

/* self-contained cycle read (DWT already enabled by dwt_init() in main) */
static inline uint32_t fft_dwt_now(void) { return DWT->CYCCNT; }

static void fft_selftest(tflite::MicroInterpreter &interpreter,
                         TfLiteTensor *in, TfLiteTensor *out)
{
    const float in_scale  = in->params.scale;
    const int   in_zp     = in->params.zero_point;

    fe_init();   /* build Hann table + FFT instance once */

    static float feat[128];
    int correct = 0;
    uint64_t fft_cyc = 0, inf_cyc = 0;

#if defined(FE_USE_CMSIS)
    printf("\r\n=== on-board FFT self-test (CMSIS-DSP) ===\r\n");
#else
    printf("\r\n=== on-board FFT self-test (plain-C FFT) ===\r\n");
#endif

    for (int t = 0; t < kNumRaw; ++t) {
        /* 1) raw window -> 128 features (timed) */
        uint32_t a = fft_dwt_now();
        fe_extract(kRawWindows[t], feat);
        uint32_t fdt = fft_dwt_now() - a;
        fft_cyc += fdt;

        /* 2) quantize + invoke (timed) */
        for (int i = 0; i < 128; ++i) {
            float q = feat[i] / in_scale + (float)in_zp;
            if (q > 127.0f) q = 127.0f; else if (q < -128.0f) q = -128.0f;
            in->data.int8[i] = (int8_t)q;
        }
        uint32_t b = fft_dwt_now();
        if (interpreter.Invoke() != kTfLiteOk) { printf("Invoke failed\r\n"); return; }
        uint32_t idt = fft_dwt_now() - b;
        inf_cyc += idt;

        int best = 0;
        for (int i = 1; i < kNumRaw; ++i)
            if (out->data.int8[i] > out->data.int8[best]) best = i;
        bool ok = (best == kRawExpected[t]);
        correct += ok;

        printf("[%s] true=%-7s pred=%-7s  fft=%lu inf=%lu cyc\r\n",
               ok ? "PASS" : "FAIL", kRawLabels[t], kRawLabels[best],
               (unsigned long)fdt, (unsigned long)idt);
    }

    printf("\r\n%d/%d correct | avg FFT %lu cyc (%lu us) | avg inference %lu cyc (%lu us)\r\n",
           correct, kNumRaw,
           (unsigned long)(fft_cyc / kNumRaw), (unsigned long)(fft_cyc / kNumRaw / 240),
           (unsigned long)(inf_cyc / kNumRaw), (unsigned long)(inf_cyc / kNumRaw / 240));
    printf("=== FFT self-test complete ===\r\n");
}

#endif /* FFT_SELFTEST_H */
