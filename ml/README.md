# ML pipeline — CWRU bearing-fault classifier for the C3M5

Run order (from this folder, venv active):

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt                     # step 1.4
python download_data.py                             # step 1.5 (~100 MB)
python preprocess.py                                # step 1.6
python train.py                                     # step 1.7
python quantize.py                                  # step 1.8 -> ../firmware/model_data.h
```

- Features mirror the planned on-device chain: 1024-pt hann rfft -> 512 mags
  -> x4 pool -> 128 log features -> scalar standardize (params in data/norm.json).
- Model: MLP 128-96-48-10 (FULLY_CONNECTED+SOFTMAX only ≈ ops already in firmware).
- Known caveat: window-level split (same recording in train+val) — optimistic
  accuracy; load-condition split is the stretch upgrade.
- v2 (optional): MaFaulDa multi-fault set for the live demo richness.
