# Candidate datasets — motor/power anomaly detection on Cortex-M33

Goal: train small (≤50 KB post-INT8) anomaly/fault classifier; deploy on PSOC C3M5.
Note: the C3M5 EVK has **no onboard IMU/microphone**. Plan for one of: (a) mikroBUS accelerometer click board, (b) stream recorded data into the ADC/UART for the demo ("replay rig" — acceptable and common), (c) current-sense via the board's analog subsystem.

## Shortlist (ranked)

### 1. MaFaulDa — Machinery Fault Database ★ primary candidate
- ~1,951 multivariate time-series samples, 6 conditions: normal, horizontal/vertical misalignment, imbalance, underhang/overhang bearing faults
- SpectraQuest simulator: accelerometers + tachometer + microphone, 50 kHz
- Why: multiple fault types = richer demo (fault-type classification, not just binary); vibration-centric fits the motor-control story
- http://www02.smt.ufrj.br/~offshore/mfs/page_01.html

### 2. CWRU Bearing Dataset ★ benchmark companion
- The most-benchmarked bearing fault dataset (inner race / ball / outer race defects, multiple severities)
- Why: judges/reviewers can compare your accuracy against hundreds of published results → credibility. Use as secondary validation set
- https://engineering.case.edu/bearingdatacenter

### 3. Paderborn University (KAt) bearing dataset
- Unique: includes **motor current signals**, not just vibration → enables "no extra sensor" story (current sense via C3M5's programmable analog front-end, its designed strength)
- Larger download, more prep work
- https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter

### 4. AI4I 2020 Predictive Maintenance (UCI)
- Synthetic tabular data (torque, speed, temp, wear). Trivial to fit on MCU but weak WOW — no signal processing story. Fallback only
- https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### 5. MIMII / ToyADMOS (audio)
- Industrial machine sounds (fans, pumps, valves). Strong if pivoting to acoustic monitoring; needs microphone on the board
- https://zenodo.org/records/3384388

## Recommendation
- **Train on MaFaulDa** (6-class fault classifier + anomaly score), **validate on CWRU** for published-benchmark comparability.
- Feature pipeline: windowed FFT or mel-spectrogram via CMSIS-DSP → small CNN or depthwise-separable CNN → INT8.
- Stretch (if time): Paderborn current-signal model to showcase the C3M5 analog subsystem — strongest hardware-fit narrative.
- Demo rig: replay recorded waveforms through ADC/UART; if budget allows, one mikroBUS accelerometer click (~$20) on a small fan/motor for a live fault demo.

*Verify licenses permit redistribution of preprocessed samples in the repo; otherwise ship a download+preprocess script (better for reproducibility anyway).*
