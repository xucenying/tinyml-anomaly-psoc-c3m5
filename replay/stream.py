#!/usr/bin/env python3
"""stream.py — PC host for the replay rig. Streams recorded CWRU feature windows
to the C3M5 (or the board simulator), injects a fault mid-run, and shows a live
detection dashboard.

Transports:
    --serial COM5           real board over USB-UART (needs pyserial)
    --tcp 127.0.0.1:9000    an already-running board_sim
    --sim                   auto-launch board_sim on localhost (no hardware)

Scenario (default): stream healthy windows, switch to a fault class to mimic a
developing fault, then return to healthy. The board raises a debounced ALERT;
this host measures detection latency and flags any false alarms.

Examples:
    python stream.py --sim
    python stream.py --sim --fault ir_014 --normal 10 --fault-frames 25
    python stream.py --serial COM5
Apache-2.0."""
from __future__ import annotations
import argparse, json, socket, subprocess, sys, time
from pathlib import Path

import numpy as np
from protocol import pack_frame

ROOT = Path(__file__).resolve().parent


# ---------- transports ----------
class Transport:
    def write(self, b: bytes): ...
    def readline(self) -> str: ...
    def close(self): ...


class TcpTransport(Transport):
    def __init__(self, host: str, port: int):
        self.s = socket.create_connection((host, port), timeout=10)
        self.buf = b""

    def write(self, b): self.s.sendall(b)

    def readline(self):
        while b"\n" not in self.buf:
            chunk = self.s.recv(4096)
            if not chunk:
                raise EOFError("board closed connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def close(self): self.s.close()


class SerialTransport(Transport):
    def __init__(self, port: str, baud: int):
        import serial  # pyserial
        self.s = serial.Serial(port, baud, timeout=10)

    def write(self, b): self.s.write(b)

    def readline(self):
        return self.s.readline().decode(errors="replace").strip()

    def close(self): self.s.close()


# ---------- data ----------
def load_windows():
    d = np.load(ROOT.parent / "ml/data/features.npz")
    X, y = d["X"].astype(np.float32), d["y"]
    labels = json.loads((ROOT.parent / "ml/data/classes.json").read_text())
    by_class = {lab: X[y == i] for i, lab in enumerate(labels)}
    return labels, by_class


def build_scenario(by_class, normal_n, fault_label, fault_n, tail_n, seed):
    rng = np.random.default_rng(seed)
    def take(lab, n):
        pool = by_class[lab]
        idx = rng.integers(0, len(pool), size=n)
        return [(lab, pool[i]) for i in idx]
    return (take("normal", normal_n)
            + take(fault_label, fault_n)
            + take("normal", tail_n))


# ---------- run ----------
def run(tr: Transport, plan, labels, fault_label):
    # No upfront banner read: a real board stays silent in read_frame until it
    # receives a frame (only the sim greets on connect). We stream first, then
    # skip any pre-amble lines (boot self-test, banners, ERR/ALERT) until "RES".
    hdr = f"{'seq':>4} {'streamed':>9} {'pred':>9} {'conf':>5} {'cycles':>8}  state"
    print(hdr); print("-" * len(hdr))

    fault_onset = next(i for i, (lab, _) in enumerate(plan) if lab != "normal")
    alert_at = None
    correct = 0
    false_alerts = 0

    for i, (true_lab, feat) in enumerate(plan):
        tr.write(pack_frame(feat))
        # RES <seq> <pred_idx> <label> <conf%> <cycles> <ok|ALERT>
        parts = tr.readline().split()
        while not parts or parts[0] != "RES":
            if parts:                      # surface board pre-amble / diagnostics
                print(f"     board| {' '.join(parts)}")
            parts = tr.readline().split()
        pred_lab = parts[3]
        conf = parts[4]
        cycles = int(parts[5])
        state = parts[6]
        correct += (pred_lab == true_lab)
        if state == "ALERT":
            if alert_at is None and i >= fault_onset:
                alert_at = i
            if i < fault_onset:
                false_alerts += 1
        flag = "  <== ALERT" if state == "ALERT" else ""
        marker = " (fault injected)" if i == fault_onset else ""
        print(f"{i:>4} {true_lab:>9} {pred_lab:>9} {conf:>5} {cycles:>8}  {state}{flag}{marker}")

    print("-" * len(hdr))
    total = len(plan)
    print(f"\nframes: {total}   window pred-accuracy: {correct}/{total} "
          f"= {100*correct/total:.1f}%")
    if alert_at is not None:
        lat = alert_at - fault_onset + 1
        # ~43 ms per real CWRU window (hop 512 @ ~12 kHz)
        print(f"fault injected at frame {fault_onset} ({fault_label}); "
              f"ALERT at frame {alert_at}  ->  detection latency {lat} frames "
              f"(~{lat*43} ms of signal)")
    else:
        print(f"fault injected at frame {fault_onset} ({fault_label}); "
              f"no ALERT raised (check thresholds)")
    print(f"false alerts during healthy stretch: {false_alerts}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serial", metavar="PORT", help="board COM/tty port")
    g.add_argument("--tcp", metavar="HOST:PORT", help="running board_sim")
    g.add_argument("--sim", action="store_true", help="auto-launch board_sim")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fault", default="or_021", help="fault class to inject")
    ap.add_argument("--normal", type=int, default=8, help="healthy frames before")
    ap.add_argument("--fault-frames", type=int, default=20, dest="fault_n")
    ap.add_argument("--tail", type=int, default=8, help="healthy frames after")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between frames (0 = as fast as possible)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sim-port", type=int, default=9007)
    a = ap.parse_args()

    labels, by_class = load_windows()
    if a.fault not in by_class or a.fault == "normal":
        sys.exit(f"--fault must be one of {[l for l in labels if l!='normal']}")
    plan = build_scenario(by_class, a.normal, a.fault, a.fault_n, a.tail, a.seed)

    proc = None
    try:
        if a.sim:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "board_sim.py"), "--port", str(a.sim_port)],
                stderr=subprocess.PIPE, text=True)
            # wait for the sim to start listening
            for _ in range(50):
                try:
                    tr = TcpTransport("127.0.0.1", a.sim_port); break
                except OSError:
                    time.sleep(0.1)
            else:
                sys.exit("board_sim did not start")
        elif a.tcp:
            host, port = a.tcp.split(":"); tr = TcpTransport(host, int(port))
        else:
            tr = SerialTransport(a.serial, a.baud)

        # optional pacing wrapper
        if a.interval > 0:
            orig = tr.write
            def paced(b, _o=orig): _o(b); time.sleep(a.interval)
            tr.write = paced  # type: ignore

        run(tr, plan, labels, a.fault)
        tr.close()
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
# end of stream.py
