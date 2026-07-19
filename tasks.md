# Task breakdown — beginner-grade steps (Phases 1–4)

Each task has a "done when" check. Do them in order; one variable at a time in Phase 2.

## Phase 1 — Proof of life (Jul 13–19)

- [ ] **1.1** ModusToolbox → Project Creator → board `KIT_PSC3M5_EVK` → "Hello World" example → Build → Program. *Done when: LED blinks.*
- [ ] **1.2** Find COM port (Device Manager → Ports), open serial terminal at 115200 baud. *Done when: text from the board appears.*
- [x] **1.3** ~~Find Infineon ML example~~ (confirmed 2026-07-06: none exist for PSOC Control C3 — official ML flow covers PSoC 6/Edge only). Manual TFLM port instead: follow `firmware/README-tflm-port.md` (generate TFLM tree → graft into Hello World project → flash `main.cpp` sine-model smoke test). *Done when: UART prints predicted vs true sin(x).* Bonus: this makes us the first public TFLM port to PSOC Control C3 — reusable-artifact points.
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

- [x] **3.1** ~~Choose demo input~~ — replay path chosen (record → stream). Live accelerometer stays optional (3.4).
- [x] **3.2** ~~Replay path~~ DONE: `replay/stream.py` streams CWRU windows over UART/TCP; board (`firmware/main.cpp run_replay`) classifies each. Binary framed protocol (`replay_protocol.h` ⇄ `protocol.py`), CRC-16. Verified vs `board_sim.py`: normal→`normal`, 100% window accuracy.
- [x] **3.3** ~~Alert logic~~ DONE: fault-vs-normal + 60% confidence gate + debounce (3 fault / 5 normal, hysteresis) → LED + CAN-FD stub frame. Verified: fault injection latches ALERT in 3 frames (~129 ms), 0 false alarms, clears on return to normal.
- [ ] **3.4** (live option) Accelerometer on small fan; coin on blade = imbalance; live detection
- [ ] **3.5** (optional) Same model on Corstone-300 FVP (M55/Ethos-U) → extra comparison column
- [ ] **3.6** Video storyline: hook 10s → problem 20s → demo 60s → results 40s → reusable artifacts 30s → close 10s

## Phase 4 — Package (Aug 3–9)

- [ ] **4.1** LICENSE (Apache-2.0), visible in GitHub About — hard requirement, verify it displays
- [ ] **4.2** README: pitch → results table → photo → quickstart → repro steps → harness reuse guide
- [ ] **4.3** `download_data.py`: one-command dataset fetch + preprocess (never commit raw data)
- [ ] **4.4** Clean-machine test of your own README; fix every snag (DevEx points live here)
- [x] **4.5** ~~Arm Performix~~ — resolved 2026-07-12: Performix targets Neoverse servers (Graviton/Cobalt/Axion), not applicable to bare-metal Cortex-M. README will state this + that DWT cycle counting is the M-profile equivalent.
- [ ] **4.6