# Porting TFLite-Micro to the PSOC Control C3M5 (KIT_PSC3M5_EVK)

Infineon's official ML middleware (MTB-ML / DEEPCRAFT examples) supports PSoC 6 and
PSOC Edge only — **not** PSOC Control C3. This is our own port. Status: **drafted,
not yet tested on hardware** — expect to iterate.

Two routes; try A first, fall back to B.

## Route A (recommended): generate a TFLM source tree, drop into ModusToolbox

TFLM ships a script that produces a clean, self-contained source tree — no bazel,
no downloads at build time. With `cmsis_nn` it also pulls in CMSIS-NN kernels
(that's our Phase-2 optimization step; start WITHOUT it for the baseline).

On your PC (needs Python 3.10+; run inside the cloned `tflite-micro` repo,
`..\arm-refs\tflite-micro`):

```bash
# baseline tree (reference kernels — Phase 2 stage "ref")
python tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ../tflm-tree-ref \
  --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR="

# optimized tree (CMSIS-NN kernels — Phase 2 stage "cmsisnn")
python tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  ../tflm-tree-cmsisnn \
  --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m33 OPTIMIZED_KERNEL_DIR=cmsis_nn"
```

Then in your ModusToolbox "Hello World" project for KIT_PSC3M5_EVK:

1. Copy `tflm-tree-ref/` into the project as `tflite-micro/`.
2. Edit the project `Makefile`:
   ```make
   SOURCES += $(wildcard tflite-micro/tensorflow/**/*.cc) $(wildcard tflite-micro/third_party/**/*.c*)
   INCLUDES += tflite-micro tflite-micro/third_party/flatbuffers/include \
               tflite-micro/third_party/gemmlowp tflite-micro/third_party/kissfft \
               tflite-micro/third_party/ruy
   DEFINES  += TF_LITE_STATIC_MEMORY
   CXXFLAGS += -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics
   ```
   (ModusToolbox auto-discovers sources in the project dir on recent versions —
   if so, only the INCLUDES/DEFINES/CXXFLAGS lines are needed. Check `Makefile`
   comments in your generated project.)
3. Replace `main.c` with our `main.cpp` (this folder) and add `model_data.h`
   (generate with `convert_tflite_to_c.py`, see below).
4. Build. First build will surface missing-include or flag issues — fix one at a
   time; bring errors back to Claude.

## Route B (fallback): community integration

`jeldriks/mtb-tflite-micro` on GitHub wraps TFLM for ModusToolbox as a library.
Not Infineon-official and may lag TFLM upstream, but designed for exactly this
use. If Route A's Makefile surgery fights back, try adding this via Library
Manager (Import → local/custom library) instead.

## Model for the smoke test

Use TFLM's own tiny "hello world" sine model (predicts sin(x); ~3 KB):

```bash
python convert_tflite_to_c.py \
  ../arm-refs/tflite-micro/tensorflow/lite/micro/examples/hello_world/models/hello_world_int8.tflite \
  model_data.h
```

Success = UART prints x, predicted sin(x), true sin(x) pairs. Then 1.3 is done
and Phase-1 continues with our real anomaly model in place of the sine model.

## Known risks (log findings here)

- [ ] Flash/RAM: TFLM core ≈ 40–100 KB flash depending on kernels registered —
      fits our 256 KB, but keep the op resolver minimal (only ops the model uses)
- [ ] `-std=c++17` collides with BSP C flags? Set per-language flags if so
- [ ] DWT lock on PSOC C3 (harness 1.9) — unrelated to TFLM but same session
- [ ] If linker overflows SRAM: shrink `kTensorArenaSize` in main.cpp first
