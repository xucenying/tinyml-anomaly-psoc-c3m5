#!/usr/bin/env python3
"""Download the classic CWRU bearing-fault subset (12 kHz drive-end, 0 HP / 1797 rpm).
10 classes: normal + {inner race, ball, outer race} x {0.007", 0.014", 0.021"}.
~100 MB total. Usage: python download_data.py   (creates data/cwru/*.mat)
Apache-2.0."""
import sys
from pathlib import Path
import requests

BASE = "https://engineering.case.edu/sites/default/files"
# class_name -> CWRU file number (12k drive end, 0 HP)
FILES = {
    "normal":  "97",
    "ir_007":  "105", "b_007": "118", "or_007": "130",
    "ir_014":  "169", "b_014": "185", "or_014": "197",
    "ir_021":  "209", "b_021": "222", "or_021": "234",
}

def main():
    out = Path(__file__).parent / "data" / "cwru"
    out.mkdir(parents=True, exist_ok=True)
    for cls, num in FILES.items():
        dst = out / f"{cls}_{num}.mat"
        if dst.exists() and dst.stat().st_size > 100_000:
            print(f"[skip] {dst.name} already present")
            continue
        url = f"{BASE}/{num}.mat"
        print(f"[get ] {cls}: {url}")
        r = requests.get(url, timeout=120)
        if r.status_code != 200 or len(r.content) < 100_000:
            print(f"  !! failed ({r.status_code}, {len(r.content)} bytes).")
            print("  If CWRU moved the files, download manually from")
            print("  https://engineering.case.edu/bearingdatacenter/download-data-file")
            print(f"  and save as {dst}")
            continue
        dst.write_bytes(r.content)
        print(f"  ok: {len(r.content)/1e6:.1f} MB")
    have = sorted(p.name for p in out.glob("*.mat"))
    print(f"\n{len(have)}/10 files present: {have}")
    sys.exit(0 if len(have) == 10 else 1)

if __name__ == "__main__":
    main()
