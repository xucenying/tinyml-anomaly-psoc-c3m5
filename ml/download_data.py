#!/usr/bin/env python3
"""Download CWRU 12 kHz drive-end bearing data for a leave-one-load-out split.

10 classes x 4 motor loads (0,1,2,3 HP) = 40 recordings. Training holds out one
whole load, so train and test never share a recording or an operating
condition (no data leakage). ~400 MB total.
Usage: python download_data.py    Apache-2.0."""
import sys
from pathlib import Path
import requests

BASE = "https://engineering.case.edu/sites/default/files"

# class -> file numbers for loads [0hp, 1hp, 2hp, 3hp]
FILES = {
    "normal": [97, 98, 99, 100],
    "ir_007": [105, 106, 107, 108],
    "b_007":  [118, 119, 120, 121],
    "or_007": [130, 131, 132, 133],
    "ir_014": [169, 170, 171, 172],
    "b_014":  [185, 186, 187, 188],
    "or_014": [197, 198, 199, 200],
    "ir_021": [209, 210, 211, 212],
    "b_021":  [222, 223, 224, 225],
    "or_021": [234, 235, 236, 237],
}

def main():
    out = Path(__file__).parent / "data" / "cwru"
    out.mkdir(parents=True, exist_ok=True)
    have = 0
    want = 0
    for cls, nums in FILES.items():
        for load, num in enumerate(nums):
            want += 1
            # filename encodes class and load, e.g. ir_007_load1_106.mat
            dst = out / f"{cls}_load{load}_{num}.mat"
            if dst.exists() and dst.stat().st_size > 100_000:
                have += 1
                continue
            url = f"{BASE}/{num}.mat"
            print(f"[get ] {cls} load{load}: {url}")
            try:
                r = requests.get(url, timeout=120)
            except Exception as e:
                print(f"  !! {e}")
                continue
            if r.status_code != 200 or len(r.content) < 100_000:
                print(f"  !! failed ({r.status_code}). Download {num}.mat manually from")
                print("     https://engineering.case.edu/bearingdatacenter/download-data-file")
                print(f"     and save as {dst}")
                continue
            dst.write_bytes(r.content)
            have += 1
            print(f"  ok: {len(r.content)/1e6:.1f} MB")
    print(f"\n{have}/{want} files present.")
    sys.exit(0 if have == want else 1)

if __name__ == "__main__":
    main()
