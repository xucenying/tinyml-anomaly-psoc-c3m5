# Optimization ladder — KIT_PSC3M5_EVK (Cortex-M33 @ 240 MHz, FPU/softfp)

Model: CWRU 10-class bearing-fault MLP (128 FFT features → 96 → 48 → 10).
Every row: same 10 test vectors, 10/10 correct, DWT cycle counter, `-O2`.

## Main table (consistent config: -O2, softfp)

| stage | kernels | model | avg cycles | latency | arena | model size |
|---|---|---|---|---|---|---|
| fp32_ref | TFLM reference (FPU) | FP32 | 140,246 | 584 µs | 1,712 B | 72,236 B |
| int8_ref | TFLM reference | INT8 | 180,728 | 753 µs | 2,228 B | 24,152 B |
| int8_cmsisnn | **CMSIS-NN** | INT8 | **83,807** | **349 µs** | 2,324 B | 24,152 B |

## Key findings

1. **Quantization alone made it SLOWER: naive INT8 is 1.29x slower than FP32 on a
   core with an FPU** (753 vs 584 µs). Portable-C INT8 requantization is expensive.
2. **CMSIS-NN unlocks INT8**: 2.16x vs naive INT8, 1.67x vs FP32 — plus 3x smaller
   model (24 KB vs 72 KB) and INT8 accuracy identical to FP32 on validation.
3. **Compiler flags**: -Og ≈ -Os (no speed transforms) → -O2 = -13% cycles.
   Everything-default (-Og, ref kernels) to final (-O2, CMSIS-NN): 326,236 → 83,807
   = **3.89x end to end**.
4. FP32→INT8 conversion (PC, TFLite full-integer): 72,236 → 24,152 B, 0.00 pp
   accuracy drop (val).

## Reproduce

Debug/-Og historical rows and capture workflow: see git history of this file and
`harness/results_table.py`. Build config deltas per stage:
- tree: `firmware/tflm-tree-ref` vs `firmware/tflm-tree-cmsisnn` (as `tflite-micro/`)
- DEFINES: `CMSIS_NN` only with the cmsisnn tree
- model: `model_data.h` (INT8) vs `model_data_fp32.h`
- flags: `CFLAGS+=-O2 CXXFLAGS+=-O2`, `VFP_SELECT=softfp`

TODO (stretch): per-op profiling; Corstone-300 (M55/Ethos-U55) column; ExecuTorch attempt.

## Phase 3b: on-board feature extraction (FFT)

Full pipeline now on-device: raw 1024-sample window -> FFT features -> classify.
Held-out load 3, 9/10 correct (b_021 confused with b_014 — same in both the FFT
pipeline and the embedded-vector test, so the FFT is correct; it's a real model
limit on the held-out load).

| stage | avg cycles | latency |
|---|---|---|
| feature extraction (plain-C FFT) | 338,114 | 1,408 µs |
| inference (INT8 + CMSIS-NN, -O2) | 86,067 | 358 µs |
| **full pipeline (plain-C FFT)** | **~424,000** | **~1,766 µs** |

Finding: after CMSIS-NN sped up inference, the **FFT became the bottleneck**
(4x the inference cost) — the classic "bottleneck shifts" result. Next rung:
CMSIS-DSP FFT (-DFE_USE_CMSIS) vs this plain-C FFT.

## Phase 3b (cont.): CMSIS-DSP FFT rung

Same on-board pipeline, only the FFT implementation changed (plain-C radix-2
rFFT -> CMSIS-DSP arm_rfft_fast_f32). Same 9/10, same b_021->b_014 miss.

| FFT implementation | avg cycles | latency | speedup |
|---|---|---|---|
| plain-C radix-2 | 338,114 | 1,408 µs | 1.0x |
| **CMSIS-DSP arm_rfft_fast_f32** | **109,210** | **455 µs** | **3.1x** |

Full on-device pipeline (raw window -> features -> classify):

| pipeline | FFT | inference | total | speedup |
|---|---|---|---|---|
| plain-C FFT + CMSIS-NN | 1,408 µs | 358 µs | ~1,766 µs | 1.0x |
| **CMSIS-DSP FFT + CMSIS-NN** | 455 µs | 356 µs | **~811 µs** | **2.2x** |

CMSIS-DSP vendored minimally into the project (8 FFT source files + 3 tables:
TWIDDLECOEF_F32_512, BITREVIDX_FLT_512, TWIDDLECOEF_RFFT_F32_1024) to fit flash.
Bottleneck now balanced (FFT 56% / inference 44%) vs plain-C (FFT 80%).
