# Project: Arm Create AI Optimization Challenge 2026 — Physical AI track

Read `00-research-and-plan.md` first for full context. This file is the quick-resume brief for any Claude session.

## COMMUNICATION PREFERENCE (cenying)
Explain everything as simply as possible: short sentences, plain words, everyday analogies. Assume non-expert. Avoid jargon; when a technical term is unavoidable, define it in one line. Prefer "here's the simple version" over dense/precise phrasing.

## Goal
Win the Devpost "Arm Create: AI Optimization Challenge" (deadline **Aug 14, 2026, 4pm PDT**, submit by Aug 13).
Secondary: make the user (cenying) an expert in Arm, embedded AI, and Claude workflows.

## Concept (locked 2026-07-06)
On-MCU anomaly detection for motor/power systems on the **Infineon PSOC Control C3M5 EVK** (Arm Cortex-M33 @ 240 MHz, DSP ext, FPU, 256 KB flash / 64 KB SRAM).
Headline result = optimization ladder, same model, cycle-accurate before/after:
FP32 reference kernels → INT8 quantized → CMSIS-NN → (stretch) ExecuTorch Cortex-M backend.
Optional extension: Corstone-300 FVP (Cortex-M55 + Ethos-U55) "scales to NPU" comparison.
Demo: fault injection → on-MCU detection in ms → UART/CAN-FD alert. dsPIC boards (DM330029, dsPIC33CK PIM) are NOT Arm — supporting props only (motor drive stage).

## Hard constraints
- Public repo, **MIT or Apache-2.0 license visible in GitHub About section**
- Reproducible setup instructions, tested from clean machine
- Video ≤ 3 min, must show it running on the device
- Model must fit 256 KB flash / 64 KB SRAM (no LLMs — classic TinyML)

## Judging (100 pts)
Tech implementation 40 · WOW 25 · Impact (reusable artifacts) 20 · DevEx/docs 15.
Judges = Arm Developer Evangelists (Avin Zarlez, Michael Hall, Gabriel Peterson) who maintain
github.com/ArmDeveloperEcosystem — mirror their repo style (`sme-executorch-profiling`, `rnnoise-examples-for-pico-2`).

## Repo layout (this folder)
- `00-research-and-plan.md` — research, strategy, weekly schedule
- `datasets.md` — candidate dataset shortlist
- `harness/` — benchmark harness (DWT cycle counter C code + Python results tooling)
- `firmware/` — (future) ModusToolbox project for the C3M5

## Toolchain
ModusToolbox (Infineon), arm-none-eabi-gcc, TFLite-Micro + CMSIS-NN/CMSIS-DSP, Python train/quantize env.
Key repos: ARM-software/CMSIS-NN, ARM-software/CMSIS-DSP, tensorflow/tflite-micro, pytorch/executorch, Infineon/TARGET_KIT_PSC3M5_EVK.

## Status log (append entries here)
- 2026-07-10 (hardware): **REPLAY RIG RUNS ON THE REAL C3M5.** stream.py over USB-UART (COM, 115200) → board classified live: 100% window accuracy, ALERT latched 3 frames after or_021 fault onset, CAN-FD stub frame emitted (`FA 09 63 01`), cleared on return to normal, 0 false alarms. Host-path banner-read bug fixed (skip pre-amble until RES). **Caught a regression:** the flashed project was left on the REFERENCE tflm tree from the int8_ref benchmark run → 180,7xx cyc (~753 us), not the CMSIS-NN rung. Restored `hello-world/tflite-micro` = cmsisnn tree (127 cmsis_nn kernels) + added `CMSIS_NN` to Makefile DEFINES (Makefile.bak saved). -O2/softfp already set. Rebuild clean + reflash → expect ~83,807 cyc / 349 us. NOTE: keep the project on the cmsisnn tree for the demo/video; only swap to ref/fp32 trees for benchmarking.
- 2026-07-10: **PHASE 3 REPLAY RIG DONE (3.1–3.3).** Firmware `run_replay` in main.cpp: after the 10-vector self-test the board enters a streaming loop — reads binary frames over UART, quantizes, invokes, and runs a debounced alert state machine (fault≠normal & conf≥60%; latch after 3 faults, clear after 5 normals) → LED + CAN-FD stub. New `firmware/replay_protocol.h` (A5 5A | len | 128×f32 | CRC-16/CCITT). Host tooling in `replay/`: `protocol.py`, `stream.py` (scenario + fault injection + live dashboard, serial/TCP/--sim), `board_sim.py` (real INT8 tflite + identical alert FSM over TCP for no-hardware testing). Verified in sandbox: 100% window accuracy, ALERT latches 3 frames (~129 ms) after fault onset, 0 false alarms, clears on return to normal; CRC matches canonical 0x29B1. Sends float32 (not pre-quant int8) so on-device quantization == benchmark. Next: flash on real board (replace sim timing with DWT), then 3.4 live accel / 3.5 Corstone / 3.6 video.
- 2026-07-09 (evening): **PHASE 2 CORE COMPLETE. Final -O2 column: fp32 140,246 / int8_ref 180,728 / int8_cmsisnn 83,807 cyc (349 us).** Headline findings: naive INT8 1.29x SLOWER than FP32+FPU; CMSIS-NN = 2.16x vs naive INT8, 1.67x vs FP32, 3x smaller model; 3.89x vs all-default config. benchmarks/results.md rewritten as definitive table. Next: Phase 3 demo (replay rig, alert logic), optional per-op profiling / Corstone.
- 2026-07-09 (later): **fp32 row measured: 140,246 cyc / 584 us** (-O2, softfp/FPU, arena 1,712 B). KEY INSIGHT: FP32+FPU beats naive INT8 (326k) — INT8 only wins via CMSIS-NN. Pending for consistent -O2 column: re-run int8_cmsisnn under softfp (Run A) and int8_ref at -O2 (Run B, remove CMSIS_NN define with ref tree).
- 2026-07-09: **-O2 rung: 84,374 cyc / 351 us** (-13% vs -Os/-Og). Total ladder 3.87x. CFLAGS+=-O2 CXXFLAGS+=-O2 in project Makefile (overrides Release -Os). Remaining: fp32_ref on-device row, per-op profiling, then Phase 3 demo.
- 2026-07-08 (evening): **RUNG 2 DONE: CMSIS-NN = 3.36x.** 97,147 cyc / 404 us vs 326,236 / 1,359 us baseline. 10/10 both. Arena 2,324 B (+96 B). Required: cmsis_nn include paths + CMSIS_NN define (see README-tflm-port.md). Remaining rungs: fp32_ref on-device, Release(-O2) build, per-op profiling, Corstone-300 stretch. Then Phase 3 demo.
- 2026-07-08 (later): **Phase-1 COMPLETE. Baseline measured:** CWRU INT8 classifier on-device, 10/10 correct, avg 326,236 cyc = 1,359 us @240MHz (reference kernels, Debug build). Arena 2,228 B, model 24,152 B. DWT confirmed working (1.9 satisfied via inline counter). Numbers in benchmarks/results.md. Next: Phase 2 rung 1 = swap tflm-tree-cmsisnn into project, re-measure.
- 2026-07-08: **MILESTONE: TFLM smoke test PASSED on the C3M5.** Sine model inference on-device, arena 676 B, .text 68 KB (26% flash). First public TFLM run on PSOC Control C3. Working project: example_psoc/hello-world (Makefile lines 117-121 = integration block). Fix chain: CY_IGNORE auto-discovery shadowing -> vendor patches (kissfft guard) -> PROJECT_GENERATION define. Next: 1.4-1.8 (train + quantize anomaly model), user to register Devpost + join Arm Discord.
- 2026-07-06 (later): No Infineon ML example exists for PSOC Control C3 (MTB-ML/DEEPCRAFT = PSoC 6/Edge only). Created `firmware/` TFLM port scaffold: README-tflm-port.md (Route A: create_tflm_tree.py cortex-m33; Route B: jeldriks/mtb-tflite-micro), main.cpp (sine-model smoke test, NOT hardware-tested), convert_tflite_to_c.py (tested). Angle: first public TFLM+CMSIS-NN port to PSOC Control C3 = reusable artifact.
- 2026-07-06: Research done, plan written, concept locked. Next: register on 