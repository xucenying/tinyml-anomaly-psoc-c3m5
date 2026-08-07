# On-MCU bearing-fault detection on the Infineon PSOC&nbsp;Control&nbsp;C3M5 (Arm Cortex-M33)

A complete, cycle-accurate optimization study: the *same* TinyML model taken from
portable-C reference kernels down to a CMSIS-NN + CMSIS-DSP build on a bare
microcontroller, with every speedup measured on real silicon.

**Headline:** raw vibration window in → fault class out, entirely on the chip, in
**~1,086 µs** — a **2.1× faster full pipeline** and a **2.16× faster inference path**
versus naive INT8, with the model shrunk 3× to fit flash. No cloud, no host.

> Target: Infineon **KIT_PSC3M5_EVK** — Arm **Cortex-M33 @ 180 MHz** 
> DSP extension, FPU, 256 KB flash / 64 KB SRAM.
> This is, as far as I can find, the **first public TensorFlow-Lite-Micro +
> CMSIS-NN port to the PSOC Control C3** — Infineon's own ML tooling (MTB-ML /
> DEEPCRAFT) targets only PSoC 6 / PSoC Edge.

---

## Why it should win

- **Technological implementation (measured, on Arm).** This is not "I used
  TinyML" — it is a controlled before/after ladder on real hardware, isolating
  one optimization at a time and reporting the exact cycle delta. That is the
  kind of work the Arm Developer Ecosystem team publishes itself (e.g.
  `sme-executorch-profiling`).
- **A genuine reusable artifact.** As far as we can find, this is the **first
  public TensorFlow-Lite-Micro + CMSIS-NN port to the PSOC Control C3**.
  Infineon's official `ml-tflite-micro` library is pre-compiled for PSoC 6 /
  PSoC Edge only, **not** the Cortex-M33-based Control C3 — so this port fills
  a real gap.
- **Counter-intuitive findings that teach.** Naive INT8 was **1.28× slower**
  than FP32 on this FPU-equipped core; INT8 only wins *through* CMSIS-NN.
  After CMSIS-NN sped up inference, the **FFT became the bottleneck** — the
  classic "bottleneck shifts" result — so it too was optimized with CMSIS-DSP.
- **Honesty as a feature.** Data leakage was found and fixed; a dataset
  sampling-rate quirk was disclosed; the 180 MHz clock was verified from the
  datasheet and all latencies corrected. The numbers are defensible.
- **It respects the day job.** The detector rides on the *same* silicon a
  motor controller already needs — the fast ADC, the DSP unit — leaving the
  real-time control loop intact. Physical AI that fits where the motor
  already is.

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

## Latency

Every row below is the same 10 held-out test vectors, `-O2`, softfp/FPU, timed on
the board. Latencies = cycles ÷ 180 MHz (the C3M5's max CPU clock).

### Rung set 1 — inference (the neural net)

| build | kernels | precision | avg cycles | latency | model size |
|---|---|---|---|---|---|
| reference | TFLM portable C (FPU) | FP32 | 140,437 | 780 µs | 72 KB |
| reference | TFLM portable C | INT8 | 179,889 | 999 µs | 24 KB |
| **CMSIS-NN** | **Arm CMSIS-NN** | **INT8** | **83,414** | **463 µs** | **24 KB** |

CMSIS-NN vs naive INT8 (both `-O2`): **179,889 → 83,414 cycles = 2.16× faster.**

### Rung set 2 — feature extraction (the FFT that feeds the net)

| FFT implementation | avg cycles | latency | speedup |
|---|---|---|---|
| plain-C radix-2 rFFT | 334,213 | 1,856 µs | 1.0× |
| **CMSIS-DSP `arm_rfft_fast_f32`** | **110,693** | **614 µs** | **3.0×** |

### Full on-chip pipeline (raw window → features → class)

| pipeline | FFT | inference | total | speedup |
|---|---|---|---|---|
| plain-C FFT + CMSIS-NN | 1,856 µs | 470 µs | ~2,327 µs | 1.0× |
| **CMSIS-DSP FFT + CMSIS-NN** | 614 µs | 471 µs | **~1,086 µs** | **2.1×** |

## Accuracy

### Classification accuracy

| stage | accuracy |
|---|---|
| fp32_ref | 100.00% |
| int8_ref | 100.00% |
| int8_cmsisnn | 100.00% |

Confirmed on the real trained model (`ml/quantize.py`): FP32 100.00% val
accuracy, INT8 100.00% val accuracy — **0.00 pp accuracy drop** from
quantization, model size 72,236 B → 24,152 B (3.0× smaller).

The 100.0% figure is on the 70/30 temporal split (see "Data methodology"
below). The harder leave-one-load-out split scores **98.34%** and is the
more meaningful generalization number; see "Scope and limits".

### Numerical accuracy (the FFT itself)

Classification accuracy only tells you the *model* is right — it doesn't
tell you the on-board *FFT* is computing the right numbers to begin with.
Speed isn't useful if the faster FFT is wrong, so each on-board
implementation's 128 output features were compared directly against the
Python reference FFT (same raw window in, same math, different code):

| FFT implementation | max diff vs Python reference |
|---|---|
| plain-C radix-2 rFFT | ~2.4e-7 |
| CMSIS-DSP `arm_rfft_fast_f32` | ~8.7e-6 |

For scale, the features themselves typically vary with a standard deviation
around 0.94 — so even the larger of the two differences (8.7e-6) is about
100,000× smaller than the feature values it's compared against: normal
floating-point rounding noise, not a computational error.

## Memory footprint

| stage | model (flash) | tensor arena (RAM) |
|---|---|---|
| fp32_ref | 72,236 B | 1,616 B |
| int8_ref | 24,152 B | 2,228 B |
| int8_cmsisnn | 24,152 B | 2,324 B |

INT8 quantization shrinks the model **3×** (72 KB → 24 KB). CMSIS-NN adds
only **+96 B** of arena over the reference INT8 kernels for its 2.16×
speedup — RAM cost is negligible next to the latency win. All three fit
comfortably inside the chip's 256 KB flash / 64 KB SRAM, leaving the
majority of both free for a motor-control application running alongside it.

Flash is also the binding constraint for the FFT rung: the CMSIS-DSP tables
for the 1024-point real FFT originally overflowed the 256 KB budget by
**21 KB** and had to be trimmed to 3 specific tables (see "Setup and
reproduce" step 2) to fit.

## Data methodology

- **Dataset:** [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
  (Case Western Reserve University) — accelerometer recordings from a motor
  test rig with bearing faults seeded by hand. **10 classes**: normal, plus
  inner-race, outer-race, and ball faults at 3 fault severities (0.007",
  0.014", 0.021" diameter). Each class was recorded at **4 motor loads**
  (0–3, roughly 0–3 horsepower, ~1,730–1,797 RPM), so every class has 4
  separate recordings, one per load.
- **Drive-end (DE) channel only**, only 12 kHz data is used.
- **12-bit ADC simulation**: each raw sample is quantized to a 12-bit signed count
  at a fixed ±8 g full-scale (clip to [−2048, 2047]) *before* feature extraction —
  the same integer format the board's own ADC produces.
- **Split = per-file 70/30 temporal, no overlap**: for every recording, the first
  70% (in time) trains and the last 30% tests; any window straddling that border
  is dropped, so no train window shares a raw sample with any test window.
- **Limitation:** this split removes window-overlap leakage, but train and test
  still come from the *same recording, load, and bearing* — separated only in
  time — so it's an **easier** test than a leave-one-load-out split (holding out
  a whole operating condition entirely), which scored **98.34%**. Full detail in
  [`benchmarks/results.md`](benchmarks/results.md).
- **Why leave-one-load-out isn't the main split here:** it's the harder,
  more rigorous benchmark, but it trains on only 3 of the 4 loads and leaves
  the model with no data at all from the held-out load — 25% less training
  data. It also means only the one held-out load's recordings have any test
  windows, so the live streaming demo could only inject faults from that
  single load. The 70/30 split gives every class and every load its own
  held-out test slice, so the model trains on all 4 loads and the demo can
  show any fault type at any load while still testing on unseen data. It was
  also the exact split specified for this 12-bit-ADC, no-overlap rebuild.
  Leave-one-load-out numbers are kept and reported (98.34%) for an honest
  upper bound on the harder generalization question.

## What the numbers show

1. **Quantization *alone* made it slower.** Naive INT8 ran **1.28× slower** than
   FP32 on this core (999 vs 780 µs), because portable-C INT8 requantization is
   expensive and the M33 already has an FPU. INT8 is only a win once CMSIS-NN's
   optimized kernels do the requantization — *then* it's 1.68× faster than FP32
   and 3× smaller. The common advice "just quantize to INT8 for speed" is wrong
   on an FPU-equipped Cortex-M unless you also switch kernels.

2. **The bottleneck moved.** Once CMSIS-NN made inference cheap (463 µs), the FFT
   became roughly **4× the cost of inference** and dominated the pipeline.
   Speeding up one stage exposed the next — so the FFT had to be optimized too
   (CMSIS-DSP), which rebalanced the pipeline to ~57% FFT / 43% inference.
   Optimization is a moving target, and the data shows exactly where it moved.

## Repository layout

```
ml/            training, preprocessing, INT8 quantization, header export
firmware/      on-device code: FFT features, TFLM trees (ref & CMSIS-NN), CMSIS-DSP
replay/        host-side continuous-ADC streaming + fault-injection demo
benchmarks/    results.md — the full measured tables and method
00-research-and-plan.md   background, dataset choice, strategy
```

## Setup and reproduce

### Prerequisites

- **Hardware:** Infineon `KIT_PSC3M5_EVK` (Cortex-M33), USB cable. *(Step 4 has
  a no-hardware path — a software board simulator — for anyone without the
  board.)*
- **Firmware toolchain:** ModusToolbox™ (includes `arm-none-eabi-gcc`).
- **ML toolchain:** Python **3.12** — TensorFlow doesn't yet publish wheels for
  newer Python versions, so a newer interpreter will fail at `pip install`.
- **The buildable firmware project lives in a separate repo:**
  [tinyml-anomaly-psoc-c3m5-fw](https://github.com/xucenying/tinyml-anomaly-psoc-c3m5-fw).
  This repo (`ml/`, `firmware/`, `harness` results, `replay/`) holds the
  training pipeline, the source trees to copy into the firmware project, and
  the results/writeup. Clone the firmware repo as a sibling of this one so
  the relative paths below (`../tinyml-anomaly-psoc-c3m5-fw/`) resolve:
  ```bash
  git clone https://github.com/xucenying/tinyml-anomaly-psoc-c3m5-fw.git
  cd tinyml-anomaly-psoc-c3m5-fw && make getlibs && cd ..
  ```
  `make getlibs` fetches Infineon's PDL/HAL/BSP libraries over the network and
  can take several minutes (longer the first time, or if you open the project
  in the ModusToolbox IDE / VS Code right after, which re-runs it while
  preparing the workspace) — this is normal, just let it finish.

### 1. Build the model (`ml/`)

Needs a machine with TensorFlow — only for this step. Reading the code and
running the live demo (step 4, below) do **not** need TensorFlow; the demo
uses the lighter `tflite-runtime` package instead.

```bash
cd ml
python --version              # confirm 3.12.x before continuing - see Prerequisites
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt   # tensorflow, numpy, scipy, matplotlib, requests
python download_data.py       # fetches the CWRU .mat files into data/cwru/ (~400 MB)
                               # prints "N/40 files present"; if N<40, just rerun -
                               # it skips files already downloaded and retries the rest
                               # (dropped connections on large files are common)
python preprocess.py          # DE-only, 12-bit ADC, per-file 70/30 no-overlap split
python train.py                # train MLP (128->96->48->10), write model_fp32.keras + norm.json
python quantize.py             # INT8 tflite + model_data.h + model_data_fp32.h
python make_vectors.py         # test_vectors.h + raw_vectors.h + norm.h (matched test windows)
python export_raw_windows.py   # (demo) all raw test windows as ADC counts
```

Then sync the generated headers into the buildable firmware project — a
separate repo, [tinyml-anomaly-psoc-c3m5-fw](https://github.com/xucenying/tinyml-anomaly-psoc-c3m5-fw)
(cloned as a sibling per "Prerequisites" above; still from inside `ml/`, so
`firmware/` is `../firmware/`):

```bash
cp ../firmware/{model_data.h,model_data_fp32.h,test_vectors.h,raw_vectors.h,norm.h} \
   ../../tinyml-anomaly-psoc-c3m5-fw/
cd ..
```

**The five benchmark configs** — flash each, run the on-board self-test, record
correct/10 and avg cycles:

| # | config | build knobs | expected cycles |
|---|--------|-------------|------------------|
| 1 | FP32 | `model_data_fp32.h`, ref tree, no `CMSIS_NN` | ~140,437 |
| 2 | INT8 plain | `model_data.h`, ref tree, no `CMSIS_NN` | ~179,889 |
| 3 | INT8 + CMSIS-NN | `model_data.h`, cmsisnn tree, `CMSIS_NN` | ~83,414 |
| 4 | + on-board plain-C FFT | rung 3 + `raw_vectors.h`, no `FE_USE_CMSIS` | FFT ~334,213 |
| 5 | + on-board CMSIS-DSP FFT | rung 4 + `FE_USE_CMSIS` + `cmsis-dsp/` | FFT ~110,693 |

**How to switch between rungs.** All of this happens inside the
[tinyml-anomaly-psoc-c3m5-fw](https://github.com/xucenying/tinyml-anomaly-psoc-c3m5-fw)
repo cloned in "Prerequisites" above — that's the actual buildable
ModusToolbox project (`firmware/` in *this* repo only holds the source *to be
copied into* the firmware repo; it isn't compiled directly). Each rung needs
three things changed together, all inside the firmware repo: which folder is
copied in as its `tflite-micro/`, one or two lines in its `Makefile`'s
`DEFINES+=`, and one or two lines near the top of its `main.cpp`. The table
above shows *what* changes; here's exactly *where*:

| # | copy this folder in | change in `Makefile` | change in `main.cpp` |
|---|---|---|---|
| 1 | `firmware/tflm-tree-ref/` → `tflite-micro/` | `DEFINES+=` has **no** `CMSIS_NN`, **no** `FE_USE_CMSIS` | `#include "model_data_fp32.h"` (not `model_data.h`); `#define INFERENCE_ONLY 1` |
| 2 | (same as rung 1, no change) | (same as rung 1) | `#include "model_data.h"` (back to INT8); `#define INFERENCE_ONLY 1` |
| 3 | `firmware/tflm-tree-cmsisnn/` → `tflite-micro/` (replaces the ref tree) | add `CMSIS_NN` to `DEFINES+=` | `#include "model_data.h"` (unchanged); `#define INFERENCE_ONLY 1` |
| 4 | (same tree as rung 3, no change) | keep `CMSIS_NN`; make sure `FE_USE_CMSIS` is **not** listed | (unchanged from rung 3); `#define INFERENCE_ONLY 0` — this turns on `fft_selftest.h` + the live-stream code |
| 5 | (same tree) **plus** `firmware/cmsis-dsp/` → `cmsis-dsp/` | add `FE_USE_CMSIS` to `DEFINES+=` | (unchanged from rung 4) |

After changing all three things for a rung: `make clean && make build && make
program`, then read the results off the serial terminal. **Heads up:** the
`Makefile` and `main.cpp` checked into the firmware repo are already set to
rung 5's config (the fastest, final build) — so working through the rungs in
order means editing *backward* from rung 5 to rung 1, not forward.

### 2. Port TFLite-Micro to the board (`firmware/`)

Infineon's official ML middleware (MTB-ML / DEEPCRAFT examples) supports PSoC 6
and PSoC Edge only — **not** PSOC Control C3, so this project ports TFLM from
source. The generated trees (`firmware/tflm-tree-ref/`, `firmware/tflm-tree-cmsisnn/`)
are already committed — only rebuild them if you need to regenerate from scratch:

```bash
# baseline tree (reference kernels)
python tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ../tflm-tree-ref \
  --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR="

# optimized tree (CMSIS-NN kernels)
python tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ../tflm-tree-cmsisnn \
  --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn"
```

Drop the chosen tree into a ModusToolbox project as `tflite-micro/` and add to
the project `Makefile`:

```make
# --- TFLite-Micro (manual integration; keep out of auto-discovery) ---
CY_IGNORE+=tflite-micro
SOURCES+=$(shell find tflite-micro -name '*.c' -o -name '*.cc')
INCLUDES+=tflite-micro tflite-micro/third_party/flatbuffers/include tflite-micro/third_party/gemmlowp tflite-micro/third_party/kissfft tflite-micro/third_party/ruy tflite-micro/third_party/cmsis_nn tflite-micro/third_party/cmsis_nn/Include
DEFINES+=TF_LITE_STATIC_MEMORY PROJECT_GENERATION
# when using tflm-tree-cmsisnn, additionally: DEFINES+=CMSIS_NN
CXXFLAGS+=-std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics
```

**`CY_IGNORE` is load-bearing.** ModusToolbox auto-discovery otherwise adds every
subfolder to the include path — including `flatbuffers/include/flatbuffers/`,
whose `string.h` then shadows the system `<string.h>` and breaks
`memcpy`/`strcmp`/`memset` across the entire build.

*Why both `CY_IGNORE` and `INCLUDES` mention tflite-micro:* they're two separate systems. `CY_IGNORE` only turns
off ModusToolbox's **automatic** folder scanner for `tflite-micro/` (the one
that was grabbing the wrong files and breaking the build). `INCLUDES` is a
**manual** list that always applies regardless of the scanner, so it's what
actually tells the compiler where TFLM's headers are. It's like telling a
robot vacuum "skip this room" and then sweeping that exact room yourself by
hand — same room, but handled by two different, independent mechanisms. Skip
either line and the build breaks: no `CY_IGNORE` brings back the original
bug, no `INCLUDES` means the compiler can't find TFLM at all.

**CMSIS-DSP FFT rung (`firmware/cmsis-dsp/`).** A minimal vendored subset of
[ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) v1.10.1 —
just the 8 source files the 1024-point real FFT needs — used by
`firmware/features.h` when built with `-DFE_USE_CMSIS` to swap in
`arm_rfft_fast_f32` in place of the plain-C radix-2 FFT. Measured on the
C3M5 @ 180 MHz: plain-C FFT 334,213 cyc (1,856 µs) → CMSIS-DSP 110,693 cyc
(614 µs) = **3.0×**; full on-chip pipeline (FFT + CMSIS-NN inference) **2.1×**
(~2,327 µs → ~1,086 µs).

Source files kept (`Source/`): `TransformFunctions` —
`arm_rfft_fast_f32.c`, `arm_rfft_fast_init_f32.c`, `arm_cfft_f32.c`,
`arm_cfft_init_f32.c`, `arm_cfft_radix8_f32.c`, `arm_bitreversal2.c`;
`CommonTables` — `arm_common_tables.c`, `arm_const_structs.c`; plus the full
`Include/` and `PrivateInclude/`.

Same `CY_IGNORE`-plus-manual-`INCLUDES` pattern as the TFLM port above, with
one addition — compiling only the specific FFT tables the 1024-point
transform needs (compiling *all* tables overflows the 256 KB flash by
~21 KB):

```make
CY_IGNORE+=cmsis-dsp
SOURCES+=$(shell find cmsis-dsp/Source -name '*.c')
INCLUDES+=cmsis-dsp/Include cmsis-dsp/PrivateInclude
DEFINES+=FE_USE_CMSIS
# compile only the tables the 1024-pt real FFT needs (saves ~60 KB flash)
DEFINES+=ARM_DSP_CONFIG_TABLES ARM_FFT_ALLOW_TABLES
DEFINES+=ARM_TABLE_TWIDDLECOEF_F32_512 ARM_TABLE_BITREVIDX_FLT_512 ARM_TABLE_TWIDDLECOEF_RFFT_F32_1024
```

Which three tables, and why exactly those: `arm_rfft_fast_init_f32(1024)`
internally calls `arm_cfft_init_f32(512)`, so the FFT needs the rfft-1024
twiddle factors, the cfft-512 twiddle factors, and the 512-entry
bit-reversal index table. Omit any one of the three and you get an
`undefined reference to '...'` at link time.

**Gotcha:** the raw upstream CMSIS-DSP library ships a `Testing/` folder with
startup files for other chips (ARMCM7 FVP) that ModusToolbox auto-discovery
tries to compile. That's why only this trimmed subset is vendored, and why
`deps/CMSIS-DSP.mtb` must **not** be present — a `.mtb` file re-pulls the
full upstream library into `mtb_shared/` and reintroduces the broken build.

**Build config: how `-O2` and the FPU are set.** All benchmark numbers in
this README are measured with `-O2` and the hardware FPU active. To set
these yourself in a ModusToolbox project's `Makefile`, edit the "Advanced
Configuration" section:

```make
# Select softfp or hardfp floating point. Default is softfp.
VFP_SELECT=softfp        # enables the hardware FPU (fpv5-sp-d16 on this core)
                          # under the softfp calling convention - this line
                          # already defaults to softfp in a fresh MTB project,
                          # just don't blank it out or set it to "none"

# Additional / custom C / C++ compiler flags:
CFLAGS+=-O2
CXXFLAGS+=-O2
```

(In this repo's own firmware `Makefile`, these lines are already present —
see `CXXFLAGS+=` / `CFLAGS+=` / `VFP_SELECT=` — so you don't need to add them
yourself if you're building this project as-is; this is for reference if
you're setting up a similar project from scratch.)

`CONFIG=Release` (also in the Makefile) sets a base optimization of `-Os`,
but since the `CFLAGS+=`/`CXXFLAGS+=` lines above are appended *after* that,
`-O2` ends up later on the actual compiler command line and wins (GCC always
uses whichever `-O` flag appears last). Verified directly from a real build:
running `make build VERBOSE=1` and inspecting the printed compiler command
shows `-Os ... -O2` (confirming `-O2` wins) and
`-mfloat-abi=softfp -mfpu=fpv5-sp-d16` (confirming the hardware FPU is
active, not software float emulation).

**Use the terminal, not the IDE's Debug/Release selector.** The Eclipse IDE
for ModusToolbox has its *own* active build configuration dropdown (Project →
Build Configurations → Set Active), separate from the Makefile's
`CONFIG=Release` line — and it can silently disagree with it if left on
"Debug." Building from the IDE's buttons uses whichever the IDE's own
selector says, not necessarily what's in the Makefile. **Always build and
flash from a terminal** instead, so the settings actually used are guaranteed
to be exactly what's written in the Makefile — no separate IDE selector to
second-guess:

```bash
make clean
make build
make program          # flashes the board
```

On Windows, run these from ModusToolbox's own **modus-shell** (search for it
in the Start menu) rather than plain PowerShell or Command Prompt — it comes
with `make`, `arm-none-eabi-gcc`, and the rest of the toolchain already on
its PATH, so you don't have to locate and add them yourself.

**Flash and validate on the board.** With the ModusToolbox project pointed at
the tree and model header you want to test, run the three commands above.
All results (PASS/FAIL, predictions, cycle counts) are printed by the
firmware itself and sent back over the board's **UART** (via `printf`,
retargeted to UART by the `retarget-io` middleware) — open a serial terminal
on the board's COM port at **921600 baud, 8N1** (8 data bits, no parity, 1
stop bit) to read them. The board's UART baud rate is set project-wide (not
per-test) via ModusToolbox's Device Configurator (`make config`, in
`bsps/TARGET_APP_KIT_PSC3M5_EVK/config/design.modus`) — 921600 was chosen so
the same rate works for both these self-tests and the live-streaming demo
below, which needs it to keep up with the 12 kHz ADC data rate. On boot, the firmware
self-tests: raw window → on-board FFT → classify, then a feature-vector
inference test, printing per-class PASS/FAIL and DWT cycle counts. For the
CMSIS-NN + CMSIS-DSP build you should see ~84,788 inference cycles and
~110,693 FFT cycles, matching the tables above. Full raw console output from
a clean run of all five flash configs plus the live demo is in the appendix
of [`benchmarks/results.md`](benchmarks/results.md).

### 3. Live demo — continuous ADC streaming (`replay/`)

A PC streams a **continuous 12-bit ADC sample stream** — in 512-sample chunks,
one per ADC/DMA half-buffer — to the C3M5 over UART, from the held-out TEST
region (last 30%) of the CWRU recordings, so nothing streamed was trained on.
The board **windows the stream itself** (sliding 1024, hop 512) and runs the
**entire pipeline on-device** — Hann + FFT, INT8 quantize, CMSIS-NN classify —
then raises a **debounced fault alert** (LED + UART print). The same host script
also runs against a software board simulator, so the whole chain — framing,
windowing, FFT, inference, alert logic — is testable with no hardware attached.

```
CWRU test region ─512-sample ADC chunks─►  UART/TCP  ──►  C3M5 (or board_sim)
  stream.py                                  slide 1024-window → FFT → INT8 infer
  (fault injection)   ◄──WARM / RES lines──  debounced alert → LED + UART
```

At 12 kHz a chunk is 512/12000 = 42.7 ms of audio-rate data (24 KB/s), which
needs **~921600 baud** to keep up in real time (115200 can't) — this is why
the board's UART is configured for 921600 baud project-wide (see "Flash and
validate on the board" above), not just for this demo.

**No hardware needed:**

```bash
python ml/preprocess.py     # one-time: builds files.json / classes.json
cd replay
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # numpy<2, tflite-runtime, scipy, pyserial
python stream.py --sim
```

Expect one `WARM` line while the first window fills, then healthy windows
classified `normal`, a fault injected mid-stream, `ALERT` latching ~2–3 chunks
later, and clearing once the fault passes. This is real captured output from
the actual board (`--serial COM5 --baud 921600`, 2026-08-07):

```
chunk  streamed      pred  conf  fft_cyc  inf_cyc  state
--------------------------------------------------------
    0    normal        --    --       --       --  WARM
    1    normal    normal   99%   112286    82960  ok
    7    normal    normal   99%   112329    82969  ok
    8    or_021    or_021   99%   112320    84396  ok (fault injected)
    9    or_021    or_021   99%   112329    84889  ok
      board| *** ALERT: or_021 (99%) ***
   10    or_021    or_021   99%   112329    84344  ALERT  <== ALERT
   ...
   27    or_021    or_021   99%   112341    85276  ALERT  <== ALERT
   28    normal    or_021   99%   112347    84864  ALERT  <== ALERT
   29    normal    normal   99%   112353    82958  ALERT  <== ALERT
   ...
   33    normal    normal   99%   112347    82885  ok
   34    normal    normal   99%   112344    82911  ok
   35    normal    normal   99%   112326    82943  ok
--------------------------------------------------------

chunks sent: 36   classified windows: 35   accuracy: 34/35 = 97.1%
fault injected at chunk 8 (or_021); ALERT at chunk 10  ->  latency 2 chunks (~85 ms @ 12 kHz)
false alerts during healthy stretch: 0
```

(The one miss, chunk 28, is the alert hysteresis coasting through the first
`normal` chunk right after the fault ends — not a misclassification of a
steady-state signal. Full unedited capture in the appendix of
[`benchmarks/results.md`](benchmarks/results.md).)

`python test_e2e.py` (still inside `replay/`) runs a no-sockets in-process
check of the whole chain. Try other faults: `python stream.py --sim --fault
ir_014 --normal 10 --fault-chunks 25 --tail 10`.

**On real hardware:** flash `firmware/main.cpp` (it runs the self-tests, then
enters the ADC-stream loop automatically), set the board UART to **921600 baud**,
find its serial port, then (still inside `replay/`, `pyserial` is already
covered by `requirements.txt` above):

```bash
python stream.py --serial COM5 --baud 921600 --interval 0.0427
```

**Wire protocol** — one binary frame per 512-sample ADC chunk:

| bytes | field |
|---|---|
| `A5 5A` | sync |
| 2 | payload length, little-endian (= 1024) |
| 1024 | payload: 512 × `int16` 12-bit ADC counts ([-2048, 2047]), little-endian |
| 2 | CRC-16/CCITT-FALSE over the payload, little-endian |

Board → host, one ASCII line per chunk: `WARM` while the first window fills,
then `RES <seq> <pred_idx> <label> <conf%> <fft_cyc> <inf_cyc> <ok|ALERT>`.

**Alert logic:** a window is a fault if the predicted class ≠ `normal` and
confidence ≥ 60%. `ALERT` latches after 3 consecutive fault frames and clears
after 5 consecutive normal frames — hysteresis that rejects single-frame blips
at the cost of a few frames of detection latency.

## Reusable artifacts (what you can lift for your own board)

- **First TFLM + CMSIS-NN port to PSOC Control C3** — the ModusToolbox build
  integration in "Setup and reproduce" step 2: the `CY_IGNORE` auto-discovery
  fixes, vendor patches, and exact `DEFINES`.
- **Minimal vendored CMSIS-DSP FFT** (`firmware/cmsis-dsp/`) — 8 source files + a
  3-table trim that keeps the 1024-pt real FFT under the 256 KB flash budget.
- **Leakage-free ML pipeline** (`ml/`) — per-file temporal split with straddler-
  window dropping, so no train/test raw-sample overlap.
- **Continuous-ADC replay + alert demo** (`replay/`) — streams raw 12-bit ADC
  samples to the board, which does its own windowing/FFT/inference and runs a
  debounced fault-alert state machine. Verified on the real C3M5.

## Scope and limits

- **Domain is CWRU bearings.** The model classifies bearing faults from
  vibration.
- **Accuracy depends on which split you look at.** ~100% on the per-file
  temporal split reported above (easier — same recording/load, split only in
  time); **98.34%** on the harder leave-one-load-out split (a whole operating
  condition held out entirely). See "Data methodology" above.
- **The techniques are standard; the contribution is the port and the
  measurement.** INT8, CMSIS-NN, and CMSIS-DSP are the established Arm ML
  stack. What's new here is bringing that stack to a chip with no prior public
  ML example and surfacing the cycle-accurate, sometimes counterintuitive,
  before/after evidence.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Credit

CWRU Bearing Data Center (Case Western Reserve University) for the dataset.
Arm's [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) and
[CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP), and
[TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro).
