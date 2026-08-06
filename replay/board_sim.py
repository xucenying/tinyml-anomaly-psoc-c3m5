#!/usr/bin/env python3
"""board_sim.py — host stand-in for the C3M5, for testing the replay rig with
no hardware attached. Receives RAW 1024-sample windows and runs the SAME full
pipeline as firmware/main.cpp: Hann + FFT features (mirrors features.h /
preprocess.py) -> INT8 quantize -> invoke -> debounced alert state machine.
Speaks the binary framing protocol over a TCP socket.

Reported cycle counts are the on-device measurements (benchmarks/results.md:
CMSIS-DSP FFT + int8_cmsisnn inference) with light jitter, clearly a simulation
— replace with a real board for true numbers.

Usage:  python board_sim.py [--port 9000] [--model ../ml/model_int8.tflite]
Apache-2.0."""
from __future__ import annotations
import argparse, json, socket, struct, sys
from pathlib import Path

import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:                                   # fall back to full TF
    import tensorflow.lite as tflite                  # type: ignore

from protocol import read_frame, SYNC0

ROOT = Path(__file__).resolve().parent
NORMAL_CLASS = 6            # "normal" — must match firmware kNormalClass
CONF_THRESH = 60           # kConfThreshPct
DEBOUNCE_FAULT = 3         # kDebounceFault
DEBOUNCE_CLEAR = 5         # kDebounceClear
SIM_FFT_CYCLES = 109210    # measured CMSIS-DSP FFT avg (benchmarks/results.md)
SIM_INF_CYCLES = 83807     # measured int8_cmsisnn inference avg


# --- feature extraction: mirrors ml/preprocess.py + firmware/features.h ---
_WIN, _NBINS, _HOP = 1024, 128, 512
_HANN = np.hanning(_WIN).astype(np.float32)


def extract_features(window, mean, std):
    """raw 1024 samples -> 128 standardized features (== on-board fe_extract)."""
    x = np.asarray(window, dtype=np.float32) * _HANN
    mag = np.abs(np.fft.rfft(x))[1:513]              # drop DC -> 512 bins
    pooled = mag.reshape(_NBINS, 4).mean(axis=1)     # 512 -> 128
    return ((np.log1p(pooled) - mean) / std).astype(np.float32)


class Board:
    """Mirrors firmware: FFT features -> quantize -> invoke -> alert FSM."""
    def __init__(self, model_path: Path):
        self.it = tflite.Interpreter(model_path=str(model_path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.in_s, self.in_zp = self.inp["quantization"]
        self.out_s, self.out_zp = self.out["quantization"]
        labels = json.loads((ROOT.parent / "ml/data/classes.json").read_text())
        self.labels = labels
        norm = json.loads((ROOT.parent / "ml/data/norm.json").read_text())
        self.mean, self.std = float(norm["mean"]), float(norm["std"])
        self.seq = 0
        self.fault_run = self.normal_run = 0
        self.alert = False
        self.rng = np.random.default_rng(0)
        self.ring = np.zeros(_WIN, dtype=np.float32)   # sliding window
        self.filled = 0

    def infer(self, feat):
        q = np.clip(np.round(feat / self.in_s + self.in_zp), -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp["index"], q[None, :])
        self.it.invoke()
        raw = self.it.get_tensor(self.out["index"])[0].astype(np.int32)
        best = int(np.argmax(raw))
        conf = int((raw[best] - self.out_zp) * self.out_s * 100)
        return best, conf

    def step(self, chunk) -> str:
        # slide the 1024-window: drop oldest HOP, append newest HOP (mirrors firmware)
        self.ring = np.concatenate([self.ring[_HOP:],
                                    np.asarray(chunk, dtype=np.float32)])
        if self.filled < _WIN:
            self.filled += _HOP
        if self.filled < _WIN:
            return "WARM\r\n"                                   # window not full yet
        feat = extract_features(self.ring, self.mean, self.std)  # on-board FFT
        best, conf = self.infer(feat)
        is_fault = best != NORMAL_CLASS and conf >= CONF_THRESH
        if is_fault:
            self.fault_run += 1; self.normal_run = 0
        else:
            self.normal_run += 1; self.fault_run = 0
        if not self.alert and self.fault_run >= DEBOUNCE_FAULT:
            self.alert = True
        if self.alert and self.normal_run >= DEBOUNCE_CLEAR:
            self.alert = False
        fft_cyc = SIM_FFT_CYCLES + int(self.rng.integers(-500, 500))
        inf_cyc = SIM_INF_CYCLES + int(self.rng.integers(-400, 400))
        line = (f"RES {self.seq} {best} {self.labels[best]:<7} {conf:3d}% "
                f"{fft_cyc} {inf_cyc} {'ALERT' if self.alert else 'ok'}\r\n")
        self.seq += 1
        return line


def serve(port: int, model_path: Path):
    board = Board(model_path)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print(f"[board_sim] model={model_path.name} listening on 127.0.0.1:{port}",
          file=sys.stderr, flush=True)
    conn, addr = srv.accept()
    print(f"[board_sim] host connected from {addr}", file=sys.stderr, flush=True)
    conn.sendall(b"=== board_sim ready (CMSIS-DSP FFT + int8_cmsisnn) ===\r\n")
    readfn = lambda n: conn.recv(n)
    try:
        while True:
            chunk, ok = read_frame(readfn)
            if chunk is None:
                conn.sendall(b"ERR desync\r\n"); continue
            if not ok:
                conn.sendall(b"ERR crc\r\n"); continue
            conn.sendall(board.step(chunk).encode())
    except (EOFError, ConnectionError):
        print("[board_sim] host disconnected", file=sys.stderr, flush=True)
    finally:
        conn.close(); srv.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--model", default=str(ROOT.parent / "ml/model_int8.tflite"))
    a = ap.parse_args()
    serve(a.port, Path(a.model))
