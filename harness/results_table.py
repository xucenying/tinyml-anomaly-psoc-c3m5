#!/usr/bin/env python3
"""results_table.py — collect BENCH_CSV output from the board's UART (or a saved
log) and maintain a cumulative results table across optimization stages.

Usage:
  # from a serial port (reads until BENCH_CSV_END):
  python results_table.py --port COM5 --baud 115200 --stage int8_cmsisnn

  # from a saved UART log:
  python results_table.py --log capture.txt --stage fp32_baseline

  # render the cumulative markdown table (for README):
  python results_table.py --render

Stages accumulate in results.json; --render emits results.md with
per-stage columns and % improvement vs the first (baseline) stage.
Apache-2.0.
"""
import argparse, json, re, sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results.json"

def parse_bench_output(text: str) -> dict:
    rows, mem = {}, {}
    in_csv = False
    for line in text.splitlines():
        line = line.strip()
        if line == "BENCH_CSV_BEGIN":
            in_csv = True; continue
        if line == "BENCH_CSV_END":
            in_csv = False; continue
        if in_csv and "," in line and not line.startswith("name,"):
            f = line.split(",")
            rows[f[0]] = {"iterations": int(f[1]), "avg_cycles": int(f[2]),
                          "min_cycles": int(f[3]), "max_cycles": int(f[4]),
                          "avg_us": float(f[5])}
        m = re.match(r"BENCH_MEM (.+)", line)
        if m:
            mem = dict(kv.split("=") for kv in m.group(1).split())
            mem = {k: int(v) for k, v in mem.items()}
    return {"benchmarks": rows, "memory": mem}

def read_serial(port: str, baud: int) -> str:
    import serial  # pip install pyserial
    buf = []
    with serial.Serial(port, baud, timeout=60) as s:
        while True:
            line = s.readline().decode(errors="replace")
            if not line:
                sys.exit("timeout waiting for BENCH_CSV_END")
            buf.append(line)
            if "BENCH_CSV_END" in line and any("BENCH_MEM" in l for l in buf):
                break
    return "".join(buf)

def load() -> dict:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {"stages": {}, "order": []}

def render(data: dict) -> str:
    order = data["order"]
    if not order:
        return "no results yet\n"
    base = data["stages"][order[0]]
    names = list(base["benchmarks"])
    out = ["# Benchmark results", "",
           "| metric | " + " | ".join(order) + " | vs baseline |",
           "|---|" + "---|" * (len(order) + 1)]
    for n in names:
        cells, last = [], None
        for st in order:
            b = data["stages"][st]["benchmarks"].get(n)
            cells.append(f"{b['avg_cycles']:,} cyc ({b['avg_us']:.0f} µs)" if b else "—")
            last = b or last
        b0 = base["benchmarks"][n]["avg_cycles"]
        imp = f"**{(1 - last['avg_cycles']/b0)*100:+.1f}%**" if last else "—"
        out.append(f"| {n} | " + " | ".join(cells) + f" | {imp} |")
    out.append("")
    out.append("| memory | " + " | ".join(order) + " |")
    out.append("|---|" + "---|" * len(order))
    for key in ("flash_text", "ram_static"):
        cells = [f"{data['stages'][st]['memory'].get(key, 0):,} B" for st in order]
        out.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log"); ap.add_argument("--stage")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()

    data = load()
    if a.render:
        md = render(data)
        (RESULTS.parent / "results.md").write_text(md)
        print(md); return
    if not a.stage:
        ap.error("--stage required when capturing")
    text = read_serial(a.port, a.baud) if a.port else Path(a.log).read_text()
    parsed = parse_bench_output(text)
    if a.stage not in data["order"]:
        data["order"].append(a.stage)
    data["stages"][a.stage] = parsed
    RESULTS.write_text(json.dumps(data, indent=2))
    print(f"stage '{a.stage}': {len(parsed['benchmarks'])} benchmarks, "
          f"mem={parsed['memory']}")

if __name__ == "__main__":
    main()
