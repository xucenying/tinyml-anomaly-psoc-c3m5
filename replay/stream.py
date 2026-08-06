#!/usr/bin/env python3
"""stream.py — PC host for the replay rig. Streams a CONTINUOUS 12-bit ADC sample
stream (in 512-sample chunks = ADC/DMA half-buffers) to the C3M5 (or the board
simulator). The board windows the stream itself (sliding 1024, hop 512), runs the
full on-chip pipeline (FFT + INT8 classify), and raises a debounced ALERT.

Samples are drawn from the held-out TEST region (last 30%) of the CWRU recordings,
so nothing was trained on. A fault is injected by switching the source recording
mid-stream. This host prints a live dashboard and measures detection latency.

Transports:
    --serial COM5           real board over USB-UART (use --baud 921600 for 12 kHz)
    --tcp 127.0.0.1:9000    an already-running board_sim
    --sim                   auto-launch board_sim on localhost (no hardware)

At 12 kHz each chunk is 512/12000 = 42.7 ms of signal; pass --interval 0.0427 to
pace the stream at real time (needs ~921600 baud on serial).

Examples:
    python stream.py --sim
    python stream.py --serial COM5 --baud 921600 --interval 0.0427
    python stream.py --sim --fault ir_014 --normal 8 --fault-chunks 24
Apache-2.0."""
from __future__ import annotations
import argparse, json, socket, subprocess, sys, time
from pathlib import Path

import numpy as np
from protocol import pack_frame, HOP

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ml"))
from preprocess import de_signal, adc_quantize   # noqa: E402  (shared ADC pipeline)

FS = 12000                      # assumed sample rate (Hz)
CWRU = ROOT.parent / "ml" / "data" / "cwru"


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


# ---------- data: continuous ADC sample streams ----------
def class_stream(label, n_samples, files, rng):
    """A contiguous run of n_samples 12-bit ADC counts from the TEST region
    (last 30%) of a recording of `label`."""
    cand = [f for f in files if f.startswith(label + "_")]
    if not cand:
        sys.exit(f"no recording for class '{label}'")
    f = cand[int(rng.integers(len(cand)))]
    sig = adc_quantize(de_signal(CWRU / f))
    test = sig[int(0.70 * len(sig)):]          # held-out region only
    if len(test) < n_samples:
        test = np.tile(test, int(np.ceil(n_samples / len(test))))
    start = int(rng.integers(0, len(test) - n_samples + 1))
    return test[start:start + n_samples].astype(np.float32)


def build_scenario(files, fault_label, n_normal, n_fault, n_tail, rng):
    """Return a list of (truth_label, chunk[HOP]) — healthy, then fault, then
    healthy — as a continuous sample stream chopped into HOP-sized chunks."""
    def chunks(label, nchunks):
        s = class_stream(label, nchunks * HOP, files, rng)
        return [(label, s[k * HOP:(k + 1) * HOP]) for k in range(nchunks)]
    return chunks("normal", n_normal) + chunks(fault_label, n_fault) + chunks("normal", n_tail)


# ---------- run ----------
def run(tr: Transport, plan, fault_label):
    hdr = (f"{'chunk':>5} {'streamed':>9} {'pred':>9} {'conf':>5} "
           f"{'fft_cyc':>8} {'inf_cyc':>8}  state")
    print(hdr); print("-" * len(hdr))

    fault_onset = next(i for i, (lab, _) in enumerate(plan) if lab != "normal")
    alert_at = None
    correct = windows = false_alerts = 0

    for i, (true_lab, chunk) in enumerate(plan):
        tr.write(pack_frame(chunk))
        parts = tr.readline().split()
        # skip banner / diagnostics until a protocol line
        while not parts or parts[0] not in ("RES", "WARM", "ERR"):
            if parts:
                print(f"      board| {' '.join(parts)}")
            parts = tr.readline().split()

        if parts[0] != "RES":                 # WARM (window filling) or ERR
            marker = " (fault injected)" if i == fault_onset else ""
            print(f"{i:>5} {true_lab:>9} {'--':>9} {'--':>5} "
                  f"{'--':>8} {'--':>8}  {parts[0]}{marker}")
            continue

        pred_lab, conf = parts[3], parts[4]
        fft_cyc, inf_cyc, state = int(parts[5]), int(parts[6]), parts[7]
        windows += 1
        correct += (pred_lab == true_lab)
        if state == "ALERT":
            if alert_at is None and i >= fault_onset:
                alert_at = i
            if i < fault_onset:
                false_alerts += 1
        flag = "  <== ALERT" if state == "ALERT" else ""
        marker = " (fault injected)" if i == fault_onset else ""
        print(f"{i:>5} {true_lab:>9} {pred_lab:>9} {conf:>5} "
              f"{fft_cyc:>8} {inf_cyc:>8}  {state}{flag}{marker}")

    print("-" * len(hdr))
    ms = 1000.0 * HOP / FS
    print(f"\nchunks sent: {len(plan)}   classified windows: {windows}   "
          f"accuracy: {correct}/{windows} = {100*correct/max(windows,1):.1f}%")
    if alert_at is not None:
        lat = alert_at - fault_onset
        print(f"fault injected at chunk {fault_onset} ({fault_label}); "
              f"ALERT at chunk {alert_at}  ->  latency {lat} chunks "
              f"(~{lat*ms:.0f} ms @ {FS//1000} kHz)")
    else:
        print(f"fault injected at chunk {fault_onset} ({fault_label}); "
              f"no ALERT raised (check thresholds)")
    print(f"false alerts during healthy stretch: {false_alerts}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serial", metavar="PORT", help="board COM/tty port")
    g.add_argument("--tcp", metavar="HOST:PORT", help="running board_sim")
    g.add_argument("--sim", action="store_true", help="auto-launch board_sim")
    ap.add_argument("--baud", type=int, default=921600, help="serial baud (12 kHz needs ~921600)")
    ap.add_argument("--fault", default="or_021", help="fault class to inject")
    ap.add_argument("--normal", type=int, default=8, help="healthy chunks before")
    ap.add_argument("--fault-chunks", type=int, default=20, dest="fault_n")
    ap.add_argument("--tail", type=int, default=8, help="healthy chunks after")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between chunks (0 = as fast as possible; 0.0427 = 12 kHz real time)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sim-port", type=int, default=9007)
    a = ap.parse_args()

    labels = json.loads((ROOT.parent / "ml/data/classes.json").read_text())
    files = json.loads((ROOT.parent / "ml/data/files.json").read_text())
    if a.fault not in labels or a.fault == "normal":
        sys.exit(f"--fault must be one of {[l for l in labels if l!='normal']}")
    rng = np.random.default_rng(a.seed)
    plan = build_scenario(files, a.fault, a.normal, a.fault_n, a.tail, rng)

    proc = None
    try:
        if a.sim:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "board_sim.py"), "--port", str(a.sim_port)],
                stderr=subprocess.DEVNULL, text=True)
            for _ in range(150):
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

        if a.interval > 0:
            orig = tr.write
            def paced(b, _o=orig): _o(b); time.sleep(a.interval)
            tr.write = paced  # type: ignore

        run(tr, plan, a.fault)
        tr.close()
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
# end of stream.py
