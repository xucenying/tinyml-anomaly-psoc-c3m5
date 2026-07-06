#!/usr/bin/env python3
"""convert_tflite_to_c.py — .tflite flatbuffer -> C header (replaces xxd).
Usage: python convert_tflite_to_c.py model.tflite model_data.h [symbol_name]
Apache-2.0."""
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    sym = sys.argv[3] if len(sys.argv) > 3 else "g_model_data"
    data = src.read_bytes()
    lines = [f'/* Generated from {src.name} ({len(data)} bytes). Do not edit. */',
             '#pragma once', '#include <cstdint>', '',
             f'alignas(16) const unsigned char {sym}[] = {{']
    for i in range(0, len(data), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i:i+12])
        lines.append(f"    {chunk},")
    lines += ["};", f"const unsigned int {sym}_len = {len(data)};", ""]
    dst.write_text("\n".join(lines))
    print(f"{dst}: {len(data)} bytes -> {sym}[]")

if __name__ == "__main__":
    main()
