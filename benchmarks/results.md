# Optimization ladder — measured on KIT_PSC3M5_EVK (Cortex-M33 @ 240 MHz), CONFIG=Debug

Model: CWRU 10-class MLP 128-96-48-10. All stages: 10/10 test vectors correct.

| stage | kernels | model | avg cycles/inf | latency | arena | model size |
|---|---|---|---|---|---|---|
| int8_ref | TFLM reference | INT8 | 326,236 | 1,359 µs | 2,228 B | 24,152 B |
| int8_cmsisnn | CMSIS-NN | INT8 | **97,147** | **404 µs** | 2,324 B | 24,152 B |
| fp32_ref | TFLM reference | FP32 | TBD | TBD | TBD | 72,236 B |
| int8_cmsisnn -Os (Release) | CMSIS-NN | INT8 | 97,147 | 404 µs | 2,324 B | 24,152 B |
| int8_cmsisnn -O2 | CMSIS-NN | INT8 | **84,374** | **351 µs** | 2,324 B | 24,152 B |


**Headline: 3.87x total (326,236 -> 84,374 cycles): CMSIS-NN 3.36x, then -O2 +13%. Same model, same 10/10 accuracy.**
Off-device (PC, quantize.py): FP32→INT8 = 3.0x smaller, 0.00 pp accuracy drop.
Finding: -Og vs -Os = identical cycles (neither enables speed transforms); -O2 = -13% cycles (unrolling etc.). -Os saves 16% flash.
TODO: per-op profiling; Corstone-300 column (stretch).
