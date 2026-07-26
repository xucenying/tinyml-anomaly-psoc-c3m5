# Replay rig — stream recorded faults, detect on-device

Phase 3.2 + 3.3 of the challenge build. A PC streams recorded CWRU vibration
feature windows to the Cortex-M33 over UART; the board classifies each window
and raises a **debounced fault alert** (LED + CAN-FD frame). The same host tool
runs against a **software board simulator**, so the whole pipeline — framing,
quantization, inference, alert logic — is testable with no hardware attached.

```
features.npz ──frames──►  UART/TCP  ──►  C3M5 (or board_sim)
  stream.py                                 quantize → INT8 CMSIS-NN infer
  (fault injection)  ◄──ASCII status──      debounced alert → LED + CAN-FD
```

## Quick start (no hardware)

```bash
cd replay
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install "numpy<2" tflite-runtime               # + pyserial for a real board
python stream.py --sim
```

You should see healthy windows classified as `normal`, a fault injected mid-run,
and the board latch `ALERT` a few frames later, then clear once the fault stops:

```
 seq  streamed      pred  conf   cycles  state
   5    normal    normal   99%    83439  ok
   6    or_021    or_021   99%    83467  ok (fault injected)
   8    or_021    or_021   99%    83547  ALERT  <== ALERT
   ...
fault injected at frame 6 (or_021); ALERT at frame 8  ->  detection latency 3 frames (~129 ms of signal)
false alerts during healthy stretch: 0
```

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

The board's LED lights while an alert is latched; if you enable CAN-FD
(`-DCANFD_ENABLE`) it also emits an alert frame.

## Wire protocol

Host → board, one binary frame per feature window (`protocol.py` ⇄
`firmware/replay_protocol.h`, kept byte-for-byte identical):

| bytes | field |
|---|---|
| `A5 5A` | sync |
| 2 | payload length, little-endian (= 512) |
| 512 | payload: 128 × `float32`, little-endian |
| 2 | CRC-16/CCITT-FALSE over the payload, little-endian |

Board → host, one ASCII line per frame:

```
RES <seq> <pred_idx> <label> <conf%> <cycles> <ok|ALERT>
```

Design notes: **float32** is sent (not pre-quantized int8) so the board runs the
exact same quantization as the on-device benchmark — inference numbers stay
comparable and no work is moved off-device. 512 B/frame at 115200 baud ≈ 44 ms,
which matches the real CWRU window cadence (hop 512 @ ~12 kHz ≈ 43 ms), so it
streams at real-time. CRC-16 guards against UART noise; a bad frame is reported
(`ERR crc`) and the stream continues.

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
