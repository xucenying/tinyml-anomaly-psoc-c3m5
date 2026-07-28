# Regenerate the model + headers (revision 2: 12-bit ADC, per-file 70/30 split)

Run these on a machine with TensorFlow (the sandbox can't fit TF). Steps 1 and 4
already ran and are validated; steps 2–3 need TF and produce the real model.

```bash
cd ml
python preprocess.py          # 1. DE-only, 12-bit ADC, per-file 70/30 no-overlap split  [done]
python train.py               # 2. train MLP, write model_fp32.keras + norm.json   (needs TF)
python quantize.py            # 3. INT8 tflite + model_data.h + model_data_fp32.h   (needs TF)
python make_vectors.py        # 4. test_vectors.h + raw_vectors.h + norm.h (matched) [validated]
python export_raw_windows.py  # (demo) raw test windows as ADC counts               [validated]
```

Then sync the regenerated headers into the buildable project:

```bash
cp ../firmware/{model_data.h,model_data_fp32.h,test_vectors.h,raw_vectors.h,norm.h} \
   ../../example_psoc/hello-world/
```

## The five runs to record (accuracy + cycles), same as before

Flash each config, run the on-board self-test, record correct/10 and avg cycles.
Cycles should match the prior numbers (architecture unchanged); accuracy is the
new number to capture.

| # | config | build knobs | expected |
|---|--------|-------------|----------|
| 1 | FP32 | `model_data_fp32.h`, ref tree, no `CMSIS_NN` | ~140,246 cyc |
| 2 | INT8 plain | `model_data.h`, ref tree, no `CMSIS_NN` | ~180,728 cyc |
| 3 | INT8 + CMSIS-NN | `model_data.h`, cmsisnn tree, `CMSIS_NN` | ~83,807 cyc |
| 4 | + on-board plain-C FFT | rung 3 + `raw_vectors.h`, no `FE_USE_CMSIS` | FFT ~338,114 cyc |
| 5 | + on-board CMSIS-DSP FFT | rung 4 + `FE_USE_CMSIS` + `cmsis-dsp/` | FFT ~109,210 cyc |

Surrogate (sklearn, matching arch) preview under the new data: **100 % test
accuracy, 10/10 one-per-class**. Confirm the INT8 number in runs 2–5 and the
FP32 number in run 1, then fill `benchmarks/results.md`.
