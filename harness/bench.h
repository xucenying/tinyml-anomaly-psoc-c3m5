/*
 * bench.h — Cycle-accurate benchmark harness for Arm Cortex-M33 (PSOC Control C3M5)
 * Uses the DWT cycle counter (CYCCNT). Apache-2.0.
 *
 * Usage:
 *   bench_init();
 *   BENCH_RUN("inference_int8_cmsisnn", 100, { interpreter_invoke(); });
 *   bench_report_uart();   // CSV over UART -> results_table.py
 */
#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>
#include <stddef.h>

#define BENCH_MAX_ENTRIES 32
#define BENCH_NAME_LEN    48

typedef struct {
    char     name[BENCH_NAME_LEN];
    uint32_t iterations;
    uint64_t total_cycles;
    uint32_t min_cycles;
    uint32_t max_cycles;
} bench_entry_t;

void     bench_init(void);                 /* enable DWT->CYCCNT              */
uint32_t bench_cycles_now(void);           /* raw CYCCNT read                 */
void     bench_record(const char *name, uint32_t iterations,
                      uint64_t total, uint32_t min, uint32_t max);
void     bench_report_uart(uint32_t cpu_hz); /* CSV: name,iters,avg,min,max,us */
void     bench_report_memory(void);          /* flash/RAM usage from linker    */

/* Measure a code block over N iterations. */
#define BENCH_RUN(label, n, block)                                   \
    do {                                                             \
        uint64_t _tot = 0; uint32_t _min = 0xFFFFFFFFu, _max = 0;    \
        for (uint32_t _i = 0; _i < (n); ++_i) {                      \
            uint32_t _t0 = bench_cycles_now();                       \
            { block; }                                               \
            uint32_t _dt = bench_cycles_now() - _t0;                 \
            _tot += _dt;                                             \
            if (_dt < _min) _min = _dt;                              \
            if (_dt > _max) _max = _dt;                              \
        }                                                            \
        bench_record((label), (n), _tot, _min, _max);                \
    } while (0)

#endif /* BENCH_H */
