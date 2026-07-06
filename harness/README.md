# Benchmark harness — Cortex-M33 cycle-accurate measurement

Reusable, standalone measurement kit for the optimization ladder
(FP32 reference → INT8 → CMSIS-NN → ExecuTorch). Designed so anyone can
reproduce the numbers in our results table. Apache-2.0.

## Pieces
- `bench.h` / `bench.c` — on-target: DWT cycle counter, min/avg/max over N runs, CSV over UART, flash/RAM report from linker symbols
- `results_table.py` — host: captures UART CSV per optimization stage, accumulates `results.json`, renders `results.md` with % improvement vs baseline

## On-target usage
```c
#include "bench.h"

int main(void) {
    /* ...board + UART init (retarget-io)... */
    bench_init();

    BENCH_RUN("feature_fft_1024", 100, { run_fft(input, features); });
    BENCH_RUN("inference",        100, { tflm_invoke(); });

    bench_report_uart(240000000u);   /* C3M5 @ 240 MHz */
    bench_report_memory();
    for (;;) {}
}
```

## Host usage
```bash
pip install pyserial
python results_table.py --port COM5 --stage fp32_baseline
# ...reflash with INT8 build...
python results_table.py --port COM5 --stage int8
# ...reflash with CMSIS-NN kernels...
python results_table.py --port COM5 --stage int8_cmsisnn
python results_table.py --render     # -> results.md for the README
```

## Notes / TODO
- Verify DWT is unlocked on PSOC C3 (some parts gate DWT behind debug auth; if CYCCNT reads 0, check DAUTHCTRL / use SysTick fallback)
- Measure with `-O2` and disable interrupts around BENCH_RUN for min jitter, or report max separately to show real-time behavior
- Peak RAM (tensor arena high-water mark) still to add: instrument TFLM arena allocator
- Cross-check one stage against Arm Performix before publishing numbers
