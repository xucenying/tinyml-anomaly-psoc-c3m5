# Replay rig — stream raw sensor windows, full pipeline on-device

Phase 3.2 + 3.3 of the challenge build. A PC streams **raw 1024-sample vibration
windows** (held-out load 3 — data the model never trained on) to the Cortex-M33
over UART. The board runs the **entire pipeline itself** — Hann + FFT feature
extraction, INT8 quantize, CMSIS-NN classify — then raises a **debounced fault
alert** (LED + UART alert line). The same host tool runs against a **software
board simulator**, so the whole pipeline — framing, FFT, inference, alert logic —
is testable with no hardware attached.

```
raw_windows.npz ─raw 1024-sample frames─►  UART/TCP  ──►  C3M5 (or board_sim)
  stream.py                                   on-board FFT → INT8 CMSIS-NN infer
  (fault injection)   ◄──ASCII status──       debounced alert → LED + UART
```

Streaming raw samples (not pre-computed features) means the demo emulates a real
sensor and exercises the on-board FFT — the 3.1× CMSIS-DSP rung runs live, not
just in a self-test.

## Quick start (no hardware)

```bash
# one-time: export the raw held-out windows the demo streams
python ml/export_raw_windows.py            # writes ml/data/raw_windows.npz

cd replay
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install "numpy<2" tflite-runtime scipy         # + pyserial for a real board
python stream.py --sim
```

You should see healthy windows classified as `normal`, a fault injected mid-run,
and the board latch `ALERT` a few frames later, then clear once the fault stops.
Each line reports the on-board FFT and inference cycle counts separately:

```
 seq  streamed      pred  conf  fft_cyc  inf_cyc  state
   5    normal    normal   99%   109359    84137  ok
   6    or_021    or_021   99%   109213    83892  ok (fault injected)
   8    or_021    or_021   99%   109342    83841  ALERT  <== ALERT
   ...
fault injected at frame 6 (or_021); ALERT at frame 8  ->  detection latency 3 frames (~129 ms of signal)
false alerts during healthy stretch: 0
```

A no-sockets in-process check of the whole chain (protocol roundtrip + FFT +
classify + alert) is in `test_e2e.py`: `python test_e2e.py`.

Try other faults / timings:

```bash
python stream.py --sim --fault ir_014 --normal 10 --fault-frames 25 --tail 10
python stream.py --sim --interval 0.043      # pace at real-time window cadence
```

## On real hardware

1. Flash `firmware/main.cpp` (it runs the 10-vector self-test, then enters
   replay mode automatically).
2. Find the board's serial port (Device Manager → Ports, or `/dev/ttyACM*`).
3. Stream to it:

```bash
python stream.py --serial COM5        # or --serial /dev/ttyACM0
```

The board's LED lights while an alert is latched, and it prints a
`*** ALERT: <class> (<conf>%) ***` line over UART.

## Wire protocol

Host → board, one binary frame per **raw window** (`protocol.py` ⇄
`firmware/replay_protocol.h`, kept byte-for-byte identical):

| bytes | field |
|---|---|
| `A5 5A` | sync |
| 2 | payload length, little-endian (= 4096) |
| 4096 | payload: 1024 × `float32` raw samples, little-endian |
| 2 | CRC-16/CCITT-FALSE over the payload, little-endian |

Board → host, one ASCII line per frame (FFT and inference cycles reported
separately):

```
RES <seq> <pred_idx> <label> <conf%> <fft_cyc> <inf_cyc> <ok|ALERT>
```

Design notes: **raw float32 samples** are sent (not features, not pre-quantized
int8) so the board runs the whole pipeline — FFT feature extraction *and*
quantization — exactly as it would from a live sensor; nothing is moved
off-device. CRC-16 guards against UART noise; a bad frame is reported (`ERR crc`)
and the stream continues.

## Alert logic (firmware `run_replay`, mirrored in `board_sim.py`)

- A window is a **fault** if the predicted class ≠ `normal` **and** confidence
  ≥ `kConfThreshPct` (60%).
- `ALERT` **latches** after `kDebounceFault` (3) consecutive fault frames and
  **clears** after `kDebounceClear` (5) consecutive normal frames. The
  hysteresis rejects single-frame blips (no false alarms in the healthy stretch)
  at the cost of a few frames of detection latency — tune the constants for your
  trade-off.

## Files

| file | role |
|---|---|
| `protocol.py` | frame pack/unpack + CRC-16 (mirror of `replay_protocol.h`) |
| `board_sim.py` | host stand-in: real INT8 model + identical alert state machine, over TCP |
| `stream.py` | host: scenario builder, fault injection, live dashboard, latency report |

`board_sim.py` reports the measured on-device `int8_cmsisnn` cycle count
(`benchmarks/results.md`) with light jitter — it is a **simulation** of timing,
not a measurement. Real cycle counts come from the DWT counter on the board.

Apache-2.0.
