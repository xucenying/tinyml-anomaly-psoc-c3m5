# On-MCU bearing-fault detection on the Infineon PSOC&nbsp;Control&nbsp;C3M5 (Arm Cortex-M33)

A complete, cycle-accurate optimization study: the *same* TinyML model taken from
portable-C reference kernels down to a CMSIS-NN + CMSIS-DSP build on a bare
microcontroller, with every speedup measured on real silicon.

**Headline:** raw vibration window in → fault class out, entirely on the chip, in
**~811 µs** — a **2.2× faster full pipeline** and a **3.89× faster inference path**
versus the naive build, with the model shrunk 3× to fit flash. No cloud, no host.

> Target: Infineon **KIT_PSC3M5_EVK** — Arm **Cortex-M33 @ 240 MHz**, DSP extension,
> FPU, 256 KB flash / 64 KB SRAM. This is, as far as I can find, the **first public
> TensorFlow-Lite-Micro + CMSIS-NN port to the PSOC Control C3** — Infineon's own ML
> tooling targets only PSoC 6 / Edge.

---

## The problem

A bearing on a motor fails gradually, and the early warning is in the vibration
spectrum. Detecting it on the motor's own microcontroller — instead of streaming
data to a server — means the alert fires in under a millisecond and works with no
network. The hard part is that a microcontroller has almost no memory or compute,
so *how* you build the model decides whether it runs at all.

This project trains one classifier on the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter)
(10 classes: normal + inner/outer/ball faults at 3 severities) and then optimizes
its **on-device** execution, measuring each rung with the Cortex-M **DWT cycle
counter** (cycle-accurate, not a stopwatch).

## What was optimized, and the evidence

Every row below is the same 10 held-out test vectors, 10/10 correct, `-O2`,
softfp/FPU, timed on the board.

### Rung set 1 — inference (the neural net)

| build | kernels | precision | avg cycles | latency | model size |
|---|---|---|---|---|---|
| reference | TFLM portable C (FPU) | FP32 | 140,246 | 584 µs | 72 KB |
| reference | TFLM portable C | INT8 | 180,728 | 753 µs | 24 KB |
| **CMSIS-NN** | **Arm CMSIS-NN** | **INT8** | **83,807** | **349 µs** | **24 KB** |

From the everything-default build (portable kernels, no `-O2`) to the final
CMSIS-NN `-O2` build: **326,236 → 83,807 cycles = 3.89× end to end.**

### Rung set 2 — feature extraction (the FFT that feeds the net)

| FFT implementation | avg cycles | latency | speedup |
|---|---|---|---|
| plain-C radix-2 rFFT | 338,114 | 1,408 µs | 1.0× |
| **CMSIS-DSP `arm_rfft_fast_f32`** | **109,210** | **455 µs** | **3.1×** |

### Full on-chip pipeline (raw window → features → class)

| pipeline | FFT | inference | total | speedup |
|---|---|---|---|---|
| plain-C FFT + CMSIS-NN | 1,408 µs | 358 µs | ~1,766 µs | 1.0× |
| **CMSIS-DSP FFT + CMSIS-NN** | 455 µs | 356 µs | **~811 µs** | **2.2×** |

## Why the numbers matter (the non-obvious findings)

These are the results a judge — or another embedded developer — actually learns
from:

1. **Quantization *alone* made it slower.** Naive INT8 ran **1.29× slower** than
   FP32 on this core (753 vs 584 µs), because portable-C INT8 requantization is
   expensive and the C33 already has an FPU. INT8 is only a win once CMSIS-NN's
   optimized kernels do the requantization — *then* it's 1.67× faster than FP32 and
   3× smaller. The common advice "just quantize to INT8 for speed" is wrong on an
   FPU-equipped Cortex-M unless you also switch kernels.

2. **The bottleneck moved.** Once CMSIS-NN made inference cheap (349 µs), the FFT
   became **4× the cost of inference** and dominated the pipeline. Speeding up one
   stage exposed the next — so the FFT had to be optimized too (CMSIS-DSP), which
   rebalanced the pipeline to 56% FFT / 44% inference. Optimization is a moving
   target, and the data shows exactly where it moved.

3. **Compiler flags are a free rung.** `-Og`/`-Os` do no speed transforms here;
   `-O2` alone cut 13% of cycles at zero code change.

## Reusable artifacts (what you can lift for your own board)

- **First TFLM + CMSIS-NN port to PSOC Control C3** — the ModusToolbox build
  integration that makes it work (`firmware/README-tflm-port.md`): the `CY_IGNORE`
  auto-discovery fixes, vendor patches, and the exact `DEFINES`.
- **Minimal vendored CMSIS-DSP FFT** (`firmware/cmsis-dsp/`) — 8 source files + a
  3-table trim that keeps the 1024-pt real FFT under the 256 KB flash budget, with
  full integration notes (`firmware/cmsis-dsp/INTEGRATION.md`).
- **Leakage-free ML pipeline** (`ml/`) — leave-one-load-out split (train on motor
  loads 0/1/2, test on load 3) so the reported accuracy is honest, not inflated by
  windows from the same run leaking across train/test.
- **Cycle-accurate benchmark harness** (`harness/`) — DWT counter C + Python
  results tooling, reusable on any Cortex-M.
- **Replay + alert demo** (`replay/`) — streams recorded vibration to the board,
  runs a debounced fault-alert state machine (LED + UART alert). Verified on the
  real C3M5.

## Repository layout

```
ml/            training, preprocessing, INT8 quantization, test-vector export
firmware/      on-device code: FFT features, TFLM trees (ref & CMSIS-NN), CMSIS-DSP
harness/       DWT cycle-counter benchmark tooling
replay/        host-side replay rig + fault-injection demo
benchmarks/    results.md — the full measured tables and method
00-research-and-plan.md   background, dataset choice, strategy
```

## Reproduce

Full method and every historical row: [`benchmarks/results.md`](benchmarks/results.md).
Build deltas per rung (same project, swap three things):

- **kernels**: use `firmware/tflm-tree-ref` or `firmware/tflm-tree-cmsisnn` as
  `tflite-micro/`; add `CMSIS_NN` to `DEFINES` only for the CMSIS-NN tree.
- **precision**: `model_data.h` (INT8) vs `model_data_fp32.h` (FP32).
- **FFT**: add `FE_USE_CMSIS` + the `cmsis-dsp/` sources for the fast FFT
  (see `firmware/cmsis-dsp/INTEGRATION.md`); omit it for the plain-C FFT.
- **flags**: `CFLAGS+=-O2 CXXFLAGS+=-O2`, `VFP_SELECT=softfp`.

## Scope and limits

- **Domain is CWRU bearings.** The model classifies bearing faults from vibration;
  it is not a general anomaly detector. 9/10 on the fully held-out load (one ball
  fault confuses two severities — a real model limit, identical in both FFT paths,
  so it is not an FFT bug).
- **The techniques are standard; the contribution is the port and the measurement.**
  INT8, CMSIS-NN and CMSIS-DSP are the established Arm ML stack. What is new here is
  bringing that stack to a chip with no prior public ML example and surfacing the
  cycle-accurate, sometimes counterintuitive, before/after evidence.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Credit

CWRU Bearing Data Center (Case Western Reserve University) for the dataset.
Arm's [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) and
[CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP), and
[TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro).
