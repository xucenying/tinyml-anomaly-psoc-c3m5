#!/usr/bin/env python3
"""In-process check of the continuous-ADC-stream demo: protocol roundtrip +
board sliding-window FFT + classify + debounced alert. No sockets. Apache-2.0."""
import json
from pathlib import Path
import numpy as np
from protocol import pack_frame, read_frame, PAYLOAD_LEN, HOP
from board_sim import Board
from stream import build_scenario

ROOT = Path(__file__).resolve().parent
files = json.loads((ROOT.parent / "ml/data/files.json").read_text())
rng = np.random.default_rng(1)
plan = build_scenario(files, "or_021", n_normal=8, n_fault=20, n_tail=8, rng=rng)
fault_onset = next(i for i, (lab, _) in enumerate(plan) if lab != "normal")

# protocol roundtrip on one chunk (bytes in == bytes out, CRC valid)
def reader_from(buf):
    pos = [0]
    def rd(n):
        b = buf[pos[0]:pos[0]+n]; pos[0] += n; return b
    return rd
c0 = plan[0][1]
frame = pack_frame(c0)
assert len(frame) == 2 + 2 + PAYLOAD_LEN + 2, "frame size"
back, ok = read_frame(reader_from(frame))
assert ok and np.allclose(back, np.round(c0)), "roundtrip"
print(f"protocol roundtrip OK  (frame {len(frame)} B, {HOP} int16 ADC samples/chunk, CRC valid)")

board = Board(ROOT.parent / "ml/model_int8.tflite")
windows = correct = false_alerts = 0
alert_at = None
print(f"{'chunk':>5} {'true':>8} {'pred':>8} {'state':>6}")
for i, (true_lab, chunk) in enumerate(plan):
    _, ok = read_frame(reader_from(pack_frame(chunk)))
    assert ok
    line = board.step(chunk).split()
    tag = "  <-- fault" if i == fault_onset else ""
    if line[0] != "RES":
        print(f"{i:>5} {true_lab:>8} {'--':>8} {line[0]:>6}{tag}")
        continue
    pred, state = line[3], line[7]
    windows += 1
    correct += (pred == true_lab)
    if state == "ALERT":
        if alert_at is None and i >= fault_onset: alert_at = i
        if i < fault_onset: false_alerts += 1
    print(f"{i:>5} {true_lab:>8} {pred:>8} {state:>6}{tag}")

print(f"\nclassified windows: {windows} | accuracy {correct}/{windows} = "
      f"{100*correct/max(windows,1):.0f}%")
if alert_at is not None:
    lat = alert_at - fault_onset
    print(f"fault at chunk {fault_onset} -> ALERT at {alert_at} = {lat} chunks "
          f"(~{lat*HOP/12000*1000:.0f} ms @ 12 kHz)")
else:
    print("NO ALERT")
print(f"false alerts during healthy stretch: {false_alerts}")
print("RESULT:", "PASS" if (alert_at is not None and false_alerts == 0) else "CHECK")
