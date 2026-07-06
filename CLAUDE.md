# Project: Arm Create AI Optimization Challenge 2026 — Physical AI track

Read `00-research-and-plan.md` first for full context. This file is the quick-resume brief for any Claude session.

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
- 2026-07-06: Research done, plan written, concept locked. Next: register on Devpost, join Arm Discord, flash blink on C3M5, run stock TFLM model.
- 2026-07-06: Git repo initialized (branch `main`, .gitignore added, initial commit b1ac1be). Remote/push pending: user to create GitHub repo (suggest `tinyml-anomaly-psoc-c3m5`), then `git remote add origin ... && git push -u origin main`. Must be public with Apache-2.0 visible in About before submission.
- 2026-07-06: Created `datasets.md` (re