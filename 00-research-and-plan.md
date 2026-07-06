# Arm Create: AI Optimization Challenge — Research & Battle Plan

## 1. Challenge facts (verified 2026-07-06)

- **Deadline:** Aug 14, 2026 @ 4:00pm PDT (~5.5 weeks left)
- **Prizes:** $8,000 total — Overall $3,000 / Runner-up $2,000 / Best per track $1,000
- **Participants so far:** ~150 (low competition for 5 cash prizes)
- **Tracks:** Physical AI (robotics/embedded/edge), Cloud AI (Arm64 servers), Mobile AI (on-device)
- **Judges:** Avin Zarlez, Michael Hall, Gabriel Peterson — all Arm Developer Evangelists

### Judging criteria (100 pts)
| Criterion | Points | Implication |
|---|---|---|
| Technological implementation | 40 | Clean code, clearly Arm-leveraged, sound approach |
| "WOW" factor | 25 | Creative, communicates value in seconds |
| Potential impact | 20 | Reusable artifacts for other developers |
| UX / DevEx | 15 | Docs, easy to run and validate |

### What they explicitly want to see (pick 2–3, measure all)
Model size ↓ · model quality ↑ per size · latency/tokens-sec ↑ · inference server throughput ↑ · developer experience ↑ · **Arm-specific optimization of an existing framework/library/model**

### Hard submission requirements
- Public repo, **MIT or Apache 2.0 license visible in About section**
- Text description: overview, functionality/output, why it should win
- Step-by-step setup instructions to build/run/validate on Arm hardware
- Optional ≤3 min video (judges stop at 3:00) showing it running on-device
- Benchmarks: use **Arm Performix** where applicable

## 2. Hardware assessment

| Board | Core | Challenge-eligible? |
|---|---|---|
| PSOC Control C3M5 EVK (KIT_PSC3M5_EVK) | **Arm Cortex-M33 @ 240 MHz**, DSP ext., FPU, 256 KB flash / 64 KB SRAM | ✅ Physical AI target |
| DM330029 (dsPIC33CK LV Motor Control) | Microchip dsPIC, 16-bit | ❌ Not Arm — supporting role only |
| dsPIC33CK512MPT608 DP PIM | Microchip dsPIC, 16-bit | ❌ Not Arm |

No purchase required. Options:
- **Primary:** TinyML on the PSOC C3M5 (Cortex-M33 + DSP extension → CMSIS-NN INT8 kernels)
- **Free extension:** Arm FVP / Corstone-300 virtual hardware (Cortex-M55 + Ethos-U55) for a "scales up" story
- Constraint to design around: 256 KB flash / 64 KB SRAM → model must be tiny (quantized, pruned)

## 3. Recommended concept (Physical AI track)

**"Optimized on-MCU condition monitoring / anomaly detection for motor & power systems on Cortex-M33"**

Why this wins on the rubric:
- Plays to the C3M5's actual purpose (motor control / industrial) → authentic Physical AI story
- Optimization is the star: FP32 reference → INT8 quantized → CMSIS-NN accelerated, with **cycle-count / latency / flash / RAM before-after tables** (40 pts tech + measurable improvements)
- Reusable artifact: publish the optimized pipeline + benchmark harness as a template others can rerun (20 pts impact)
- WOW: live demo — induce a fault (imbalance, voltage sag), MCU flags it in ms, fully offline (25 pts)
- Optional twist: dsPIC board drives the motor; Arm chip does the AI — turns non-eligible hardware into demo props

Architecture:
```
Sensor (IMU / current sense / ADC)
  → feature extraction (CMSIS-DSP: FFT/MFCC)
  → INT8 model (TFLite-Micro or ExecuTorch, CMSIS-NN kernels)
  → decision + UART/CAN-FD alert
Benchmark harness: cycle counters (DWT), Arm Performix, memory report
```

Baseline vs optimized to report: reference kernels vs CMSIS-NN; FP32 vs INT8 (accuracy delta); pre/post pruning; -O levels & compiler flags.

## 3b. Review after studying official resources (2026-07-06)

The ArmDeveloperEcosystem GitHub org (46 repos) is maintained by the same Developer Evangelist team that is judging. What their recent activity reveals:

- **`rnnoise-examples-for-pico-2`** — RNNoise audio ML on RP2350 **Cortex-M33** (same core as your PSOC C3M5). Direct validation that Cortex-M33 TinyML demos are exactly in the judges' wheelhouse, and a proven reference for what fits in this class of chip.
- **`sme-executorch-profiling`** (Jun 2026, Apache-2.0) — operator-level ExecuTorch latency profiling. Judges are actively invested in **ExecuTorch + profiling/benchmarking**. A submission with an operator-level benchmark story speaks their language.
- **`workshop-ethos-u`**, **`Paddle-examples-for-AVH`** — Cortex-M + virtual hardware (AVH/Corstone) workflows are a recurring theme; the free "scales to Ethos-U NPU" extension is well aligned.
- Their example repos use **MIT / Apache-2.0 / BSD-3** licenses and clean README + benchmark structure — mirror that repo style.

Refinements to the concept (unchanged core, sharpened execution):
1. Keep motor/power anomaly detection on the C3M5 — nothing like it in their org; it uses the chip for its designed purpose (motor control), which differentiates from the sea of Pi/Android projects.
2. **Runtime comparison as a headline result:** TFLite-Micro reference kernels → +CMSIS-NN → ExecuTorch Cortex-M backend, same model, same input, cycle-accurate table. This echoes their own profiling repo and hits "Arm-specific optimization of an existing framework."
3. Structure the benchmark harness as a standalone reusable tool (their `sysreport`/`sme-executorch-profiling` pattern) → impact points.
4. **Join the [Arm Developer Program Discord](https://discord.com/invite/armsoftwaredev)** — the challenge runs workshops and office hours there; judges answer questions directly. Early, visible engagement helps.

Risk note: ExecuTorch on bare-metal Cortex-M33 (no Ethos-U) is less mature than TFLM+CMSIS-NN. Treat TFLM+CMSIS-NN as the guaranteed path; ExecuTorch comparison is the stretch goal — report it even if the result is "not yet viable at 64 KB SRAM" (that's still a useful finding for the community).

## 4. Toolchain to install

Development:
- **ModusToolbox** (Infineon IDE + SDK for PSOC) — or VS Code + ModusToolbox extension
- **Arm GNU Toolchain** (arm-none-eabi-gcc)
- Optional: **Zephyr RTOS** (has KIT_PSC3M5_EVK board support) — portability story
- Python: TensorFlow/PyTorch + quantization tooling (`tensorflow-model-optimization`, `torchao`)

Key GitHub repos:
- `ARM-software/CMSIS-NN` — optimized NN kernels for Cortex-M (the core of the optimization story)
- `ARM-software/CMSIS-DSP` — FFT/feature extraction
- `tensorflow/tflite-micro` — MCU inference runtime
- `pytorch/executorch` — alternative runtime, Arm/Cortex-M backend (judges' learning paths feature it)
- `ARM-software/ML-examples` — reference TinyML examples
- `Infineon/TARGET_KIT_PSC3M5_EVK` + Infineon code examples (`Infineon/` org)
- `ARM-software/AVH` / Corstone-300 FVP — free virtual Cortex-M55+Ethos-U

From the judges' own org (`ArmDeveloperEcosystem/`):
- `rnnoise-examples-for-pico-2` — ML on Cortex-M33, closest reference to your target
- `sme-executorch-profiling` — benchmark-harness structure to emulate
- `workshop-ethos-u`, `Paddle-examples-for-AVH` — Cortex-M + virtual hardware workflows

Claude setup:
- **Claude Code** for the firmware/repo work; add a `CLAUDE.md` with board specs, memory budget, build commands
- Connect the **GitHub connector/MCP** (engineering plugin — needs authorization in settings) for PR/issue workflow
- Already-installed skills that will carry the submission: `engineering:architecture` (ADR for design decisions), `engineering:documentation` (README/runbook — 15 DevEx pts), `engineering:testing-strategy`, `docx`/`pptx` for the write-up and video storyboard

## 5. Process (5.5 weeks)

| Week | Milestone |
|---|---|
| Jul 6–12 | Register on Devpost + join Arm Discord (workshops/office hours). Blink LED → run a stock TFLM model on C3M5. Pick dataset (e.g., motor vibration/current anomaly). Lock concept. |
| Jul 13–19 | Train + quantize model; get FP32 baseline running; build benchmark harness (DWT cycle counts). |
| Jul 20–26 | Optimization passes: INT8 + CMSIS-NN, pruning, memory layout, compiler flags. Record every number. |
| Jul 27–Aug 2 | Live demo rig (fault injection), CAN-FD/UART dashboard. Optional Corstone-300 "scales to NPU" comparison. |
| Aug 3–9 | Repo polish: license, README with reproducible benchmarks, Performix results, setup guide tested from scratch. |
| Aug 10–13 | 3-min video, Devpost write-up, submit **early** (Aug 13). |

## 6. Expertise roadmap (secondary goal)

1. Arm architecture: Cortex-M33 TRM, DSP extension, TrustZone → then M55/M85 (Helium/MVE), Ethos-U NPUs
2. learn.arm.com learning paths — start with "Visualize Ethos-U NPU performance with ExecuTorch" (linked from the challenge)
3. CMSIS-NN source reading — how INT8 kernels exploit SIMD on M-profile
4. Book: "TinyML" (Warden & Situnayake); EdX/Coursera TinyML specialization
5. Claude mastery: Claude Code workflows, CLAUDE.md conventions, skills/MCP — practiced live in this project

## Key links
- Challenge: https://arm-ai-optimization-challenge.devpost.com/
- Track details: https://arm-ai-optimization-challenge.devpost.com/details/trackdetails
- Rules: https://arm-ai-optimization-challenge.devpost.com/rules
- Arm Performix: https://developer.arm.com/servers-and-cloud-computing/arm-performix
- Kit docs: https://documentation.infineon.com/psoccontrolc3/docs/tcc1731244321580
