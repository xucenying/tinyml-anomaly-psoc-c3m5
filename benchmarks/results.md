# Optimization ladder — KIT_PSC3M5_EVK (Cortex-M33 @ 180 MHz, FPU/softfp)

Model: CWRU 10-class bearing-fault MLP (128 FFT features → 96 → 48 → 10).
Every row: same 10 test vectors, DWT cycle counter, `-O2`.
Cycle counts are the primary metric (DWT, exact). Latencies = cycles ÷ 180 MHz
(the PSOC Control C3M5 max CPU clock per the Infineon datasheet).

## Data methodology (revision 2 — 2026-07-28)

The pipeline was tightened for a fully defensible accuracy number. All results
below use this methodology:

- **Drive-end (DE) channel only**, every file treated as 12 kHz.
- **12-bit ADC simulation**: each raw sample is quantized to a 12-bit signed
  count at a fixed ±8 g full-scale (clip to [−2048, 2047]) *before* feature
  extraction, so the pipeline runs on the same integer measurements a real MCU
  ADC produces (`ml/preprocess.py`).
- **Split = per-file 70/30 temporal, no overlap**: for every recording the first
  70 % (in time) trains and the last 30 % tests; any window straddling the 70 %
  border is dropped, so **no train window shares a single raw sample with any
  test window**. Normalization (mean/std) is fit on train windows only.
- Train 8,258 windows · test 3,494 windows.

**Accuracy (confirmed on the real trained model, 2026-08-06):** FP32 100.00 %
val accuracy, INT8 100.00 % val accuracy — 0.00 pp drop from quantization
(`ml/quantize.py`), plus 10/10 on the one-window-per-class board proxy. Board
FFT-on-counts matches the PC feature pipeline to 2.4e-7. The 12-bit ADC
quantization did **not** reduce accuracy.

**Honest caveat on the split.** This per-file temporal split removes the window-
overlap leakage, but train and test still come from the *same recording, load,
and bearing* — separated only in time. It is therefore an **easier** test than
the previous leave-one-load-out split (which held out a whole operating
condition and scored 98.34 %). 100 % here reflects the easier task, not a
stronger generalization claim. Leave-one-load-out remains the harder benchmark;
this split is what was requested for the ADC/no-overlap study.

**Cycle counts are unchanged by this revision.** Latency depends on the model
architecture (128→96→48→10) and the kernels, not on the training data, and the
FFT cost is data-independent. The architecture is identical, so every cycle
figure below carries over; reflash the final config to reconfirm.

## Main table (consistent config: -O2, softfp)

| stage | kernels | model | avg cycles | latency | arena | model size |
|---|---|---|---|---|---|---|
| fp32_ref | TFLM reference (FPU) | FP32 | 140,246 | 779 µs | 1,712 B | 72,236 B |
| int8_ref | TFLM reference | INT8 | 180,728 | 1,004 µs | 2,228 B | 24,152 B |
| int8_cmsisnn | **CMSIS-NN** | INT8 | **83,807** | **466 µs** | 2,324 B | 24,152 B |

## Key findings

1. **Quantization alone made it SLOWER: naive INT8 is 1.29x slower than FP32 on a
   core with an FPU** (1,004 vs 779 µs). Portable-C INT8 requantization is expensive.
2. **CMSIS-NN unlocks INT8**: 2.16x vs naive INT8, 1.67x vs FP32 — plus 3x smaller
   model (24 KB vs 72 KB) and INT8 accuracy identical to FP32 on validation.
3. FP32→INT8 conversion (PC, TFLite full-integer): 72,236 → 24,152 B, 0.00 pp
   accuracy drop (val).

## Reproduce

All rows are built and measured with `-O2`. Historical rows and capture workflow:
see git history of this file and `harness/results_table.py`. Build config deltas
per stage:
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
| feature extraction (plain-C FFT) | 338,114 | 1,878 µs |
| inference (INT8 + CMSIS-NN, -O2) | 86,067 | 478 µs |
| **full pipeline (plain-C FFT)** | **~424,000** | **~2,356 µs** |

Finding: after CMSIS-NN sped up inference, the **FFT became the bottleneck**
(4x the inference cost) — the classic "bottleneck shifts" result. Next rung:
CMSIS-DSP FFT (-DFE_USE_CMSIS) vs this plain-C FFT.

## Phase 3b (cont.): CMSIS-DSP FFT rung

Same on-board pipeline, only the FFT implementation changed (plain-C radix-2
rFFT -> CMSIS-DSP arm_rfft_fast_f32). Same 9/10, same b_021->b_014 miss.

| FFT implementation | avg cycles | latency | speedup |
|---|---|---|---|
| plain-C radix-2 | 338,114 | 1,878 µs | 1.0x |
| **CMSIS-DSP arm_rfft_fast_f32** | **109,210** | **607 µs** | **3.1x** |

Full on-device pipeline (raw window -> features -> classify):

| pipeline | FFT | inference | total | speedup |
|---|---|---|---|---|
| plain-C FFT + CMSIS-NN | 1,878 µs | 478 µs | ~2,356 µs | 1.0x |
| **CMSIS-DSP FFT + CMSIS-NN** | 607 µs | 475 µs | **~1,082 µs** | **2.2x** |

CMSIS-DSP vendored minimally into the project (8 FFT source files + 3 tables:
TWIDDLECOEF_F32_512, BITREVIDX_FLT_512, TWIDDLECOEF_RFFT_F32_1024) to fit flash.
Bottleneck now balanced (FFT 56% / inference 44%) vs plain-C (FFT 80%).
