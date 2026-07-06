# Task breakdown — beginner-grade steps (Phases 1–4)

Each task has a "done when" check. Do them in order; one variable at a time in Phase 2.

## Phase 1 — Proof of life (Jul 13–19)

- [ ] **1.1** ModusToolbox → Project Creator → board `KIT_PSC3M5_EVK` → "Hello World" example → Build → Program. *Done when: LED blinks.*
- [ ] **1.2** Find COM port (Device Manager → Ports), open serial terminal at 115200 baud. *Done when: text from the board appears.*
- [ ] **1.3** Find/build an Infineon ML code example for PSC3M5 (`mtb-example-ml-...`); else add TFLite-Micro middleware to Hello World. *Done when: UART prints an inference result.*
- [ ] **1.4** PC: `python -m venv .venv`, install `tensorflow numpy scipy matplotlib`. *Done when: `import tensorflow` works.*
- [ ] **1.5** Download MaFaulDa + CWRU; load one file, plot waveform. *Done when: signal visible.*
- [ ] **1.6** Preprocessing script: window → FFT/spectrogram → features+labels .npz. *Done when: correct feature shapes.*
- [ ] **1.7** Train tiny CNN: >90% acc, <200K params. *Done when: saved model + accuracy number.*
- [ ] **1.8** INT8 quantize (TFLite converter), accuracy drop <2%, export .tflite → C array. *Done when: model_data.h fits flash budget.*
- [ ] **1.9** Integrate harness/bench.h/.c; benchmark dummy loop. *Done when: sane cycle counts over UART (verifies DWT unlocked on PSOC C3).*

## Phase 2 — Optimization ladder (Jul 20–26)

- [ ] **2.1** FP32 model, reference kernels → `results_table.py --stage fp32_baseline`
- [ ] **2.2** INT8 model, reference kernels → stage `int8_ref` (isolates quantization gain)
- [ ] **2.3** CMSIS-NN kernels (`OPTIMIZED_KERNEL_DIR=cmsis_nn`) → stage `int8_cmsisnn` (the big jump)
- [ ] **2.4** Feature extraction: plain C FFT vs CMSIS-DSP FFT benchmark entries
- [ ] **2.5** Compiler flags: -O0 / -O2 / -Ofast stages
- [ ] **2.6** Pruning / weight clustering; keep only if accuracy holds
- [ ] **2.7** Tensor-arena high-water mark; shrink to minimum → RAM number
- [ ] **2.8** `results_table.py --render` → results.md with % improvements
- [ ] **2.9** Stretch: ExecuTorch .pte export + Cortex-M backend attempt; document outcome either way

## Phase 3 — Demo & extension (Jul 27–Aug 2)

- [ ] **3.1** Choose demo input: replay first; order mikroBUS accelerometer (~$20) now if going live
- [ ] **3.2** Replay path: PC streams samples over UART → board classifies continuously. *Done when: "normal" prints on normal data.*
- [ ] **3.3** Alert logic: fault type + confidence + LED + optional CAN-FD frame. *Done when: fault data triggers alert in ms.*
- [ ] **3.4** (live option) Accelerometer on small fan; coin on blade = imbalance; live detection
- [ ] **3.5** (optional) Same model on Corstone-300 FVP (M55/Ethos-U) → extra comparison column
- [ ] **3.6** Video storyline: hook 10s → problem 20s → demo 60s → results 40s → reusable artifacts 30s → close 10s

## Phase 4 — Package (Aug 3–9)

- [ ] **4.1** LICENSE (Apache-2.0), visible in GitHub About — hard requirement, verify it displays
- [ ] **4.2** README: pitch → results table → photo → quickstart → repro steps → harness reuse guide
- [ ] **4.3** `download_data.py`: one-command dataset fetch + preprocess (never commit raw data)
- [ ] **4.4** Clean-machine test of your own README; fix every snag (DevEx points live here)
- [ ] **4.5** Arm Performix run where applicable
- [ ] **4.6** Post in Arm Discord for feedback before submitting

## Phase 0 + 5 (reference)
Phase 0: Devpost registration, Arm Discord, GitHub push, ModusToolbox install, repo clones to `..\arm-refs\`.
Phase 5 (Aug 10–13): record ≤3-min video, write Devpost text, submit Aug 13.
