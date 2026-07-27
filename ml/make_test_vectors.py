#!/usr/bin/env python3
"""Generate firmware/test_vectors.h: one held-out-load window per class,
normalized with the train-only stats (matches the deployed model).
Usage: python make_test_vectors.py    Apache-2.0."""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).parent
d = np.load(ROOT/"data"/"features.npz"); X,y,load = d["X"],d["y"],d["load"]
classes = json.loads((ROOT/"data"/"classes.json").read_text())
norm = json.loads((ROOT/"data"/"norm.json").read_text())
held = json.loads((ROOT/"data"/"fp32_acc.json").read_text())["held_out_load"]
Xn = (X - norm["mean"]) / norm["std"]
rng = np.random.default_rng(7)
lines = ['/* Generated from held-out load, normalized. Do not edit. */','#pragma once','',
         'constexpr int kNumTests = %d;' % len(classes),
         'constexpr int kFeatureDim = 128;',
         'const char* const kTestLabels[kNumTests] = {%s};' % ', '.join('"%s"'%c for c in classes),
         'const int kTestExpected[kNumTests] = {%s};' % ', '.join(str(i) for i in range(len(classes))),
         'const float kTestVectors[kNumTests][kFeatureDim] = {']
for ci in range(len(classes)):
    pool = np.where((y==ci) & (load==held))[0]
    i = rng.choice(pool)
    lines.append('  {%s},' % ', '.join('%.6ff'%v for v in Xn[i]))
lines += ['};','']
(ROOT.parent/"firmware"/"test_vectors.h").write_text('\n'.join(lines))
print("wrote ../firmware/test_vectors.h from held-out load", held, "-", len(classes), "vectors")
