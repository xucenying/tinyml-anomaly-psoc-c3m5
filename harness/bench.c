/*
 * bench.c — DWT cycle-counter benchmark harness for Cortex-M33. Apache-2.0.
 * Works on any Cortex-M3/M4/M7/M33 with DWT present (not M0/M0+/M23).
 * printf must be retargeted to UART (ModusToolbox: retarget-io middleware).
 */
#include "bench.h"
#include <stdio.h>
#include <string.h>

/* CMSIS core registers (ARM CMSIS core_cm33.h provides these; fallback below) */
#ifndef DWT_BASE
#define DWT_BASE        (0xE0001000UL)
#define DWT_CTRL        (*(volatile uint32_t *)(DWT_BASE + 0x000))
#define DWT_CYCCNT      (*(volatile uint32_t *)(DWT_BASE + 0x004))
#define DEMCR           (*(volatile uint32_t *)0xE000EDFCUL)
#define DEMCR_TRCENA    (1UL << 24)
#define DWT_CTRL_CYCCNTENA (1UL << 0)
#endif

static bench_entry_t s_entries[BENCH_MAX_ENTRIES];
static uint32_t s_count = 0;

void bench_init(void)
{
    DEMCR |= DEMCR_TRCENA;        /* enable trace subsystem */
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
    s_count = 0;
}

uint32_t bench_cycles_now(void)
{
    return DWT_CYCCNT;
}

void bench_record(const char *name, uint32_t iterations,
                  uint64_t total, uint32_t min, uint32_t max)
{
    if (s_count >= BENCH_MAX_ENTRIES) return;
    bench_entry_t *e = &s_entries[s_count++];
    strncpy(e->name, name, BENCH_NAME_LEN - 1);
    e->name[BENCH_NAME_LEN - 1] = '\0';
    e->iterations = iterations;
    e->total_cycles = total;
    e->min_cycles = min;
    e->max_cycles = max;
}

void bench_report_uart(uint32_t cpu_hz)
{
    printf("BENCH_CSV_BEGIN\r\n");
    printf("name,iterations,avg_cycles,min_cycles,max_cycles,avg_us\r\n");
    for (uint32_t i = 0; i < s_count; ++i) {
        const bench_entry_t *e = &s_entries[i];
        uint32_t avg = (uint32_t)(e->total_cycles / e->iterations);
        /* avg_us = avg / (cpu_hz / 1e6), integer math to avoid float printf */
        uint32_t avg_us_x100 = (uint32_t)((e->total_cycles * 100ULL) /
                                          (e->iterations * (uint64_t)(cpu_hz / 1000000UL)));
        printf("%s,%lu,%lu,%lu,%lu,%lu.%02lu\r\n",
               e->name,
               (unsigned long)e->iterations,
               (unsigned long)avg,
               (unsigned long)e->min_cycles,
               (unsigned long)e->max_cycles,
               (unsigned long)(avg_us_x100 / 100),
               (unsigned long)(avg_us_x100 % 100));
    }
    printf("BENCH_CSV_END\r\n");
}

/* Flash/RAM figures come from linker symbols (GNU ld). Adjust names to your
 * linker script if needed. Emitted alongside the CSV for results_table.py. */
extern uint32_t __etext, __data_start__, __data_end__, __bss_start__, __bss_end__;

void bench_report_memory(void)
{
    uint32_t text  = (uint32_t)&__etext;
    uint32_t data  = (uint32_t)&__data_end__ - (uint32_t)&__data_start__;
    uint32_t bss   = (uint32_t)&__bss_end__  - (uint32_t)&__bss_start__;
    printf("BENCH_MEM flash_text=%lu data=%lu bss=%lu ram_static=%lu\r\n",
           (unsigned long)text, (unsigned long)data,
           (unsigned long)bss, (unsigned long)(data + bss));
}
