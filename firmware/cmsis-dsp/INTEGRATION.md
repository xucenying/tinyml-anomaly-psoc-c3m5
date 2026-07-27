# CMSIS-DSP integration (on-board FFT / fast feature-extraction rung)

Minimal vendored subset of ARM-software/CMSIS-DSP v1.10.1 — just the 8 source
files the 1024-point real FFT needs. Used by `../features.h` when built with
`-DFE_USE_CMSIS` to replace the plain-C radix-2 FFT with `arm_rfft_fast_f32`.

Measured on the PSOC Control C3M5 (Cortex-M33 @ 240 MHz):
plain-C FFT 338,114 cyc (1408 us) -> CMSIS-DSP 109,210 cyc (455 us) = **3.1x**.
Full on-chip pipeline (FFT + CMSIS-NN inference) 2.2x: ~1766 us -> ~811 us.

## Source files kept (Source/)
- TransformFunctions: arm_rfft_fast_f32.c, arm_rfft_fast_init_f32.c,
  arm_cfft_f32.c, arm_cfft_init_f32.c, arm_cfft_radix8_f32.c, arm_bitreversal2.c
- CommonTables: arm_common_tables.c, arm_const_structs.c
Plus full Include/ and PrivateInclude/.

## ModusToolbox Makefile block
Keep the folder out of auto-discovery, add its sources + includes explicitly,
and compile ONLY the three FFT tables the 1024-pt transform uses (compiling all
tables overflows the 256 KB flash by ~21 KB):

```make
CY_IGNORE+=cmsis-dsp
SOURCES+=$(shell find cmsis-dsp/Source -name '*.c')
INCLUDES+=cmsis-dsp/Include cmsis-dsp/PrivateInclude
DEFINES+=FE_USE_CMSIS
# compile only the tables the 1024-pt real FFT needs (saves ~60 KB flash)
DEFINES+=ARM_DSP_CONFIG_TABLES ARM_FFT_ALLOW_TABLES
DEFINES+=ARM_TABLE_TWIDDLECOEF_F32_512 ARM_TABLE_BITREVIDX_FLT_512 ARM_TABLE_TWIDDLECOEF_RFFT_F32_1024
```

Which tables? `arm_rfft_fast_init_f32(1024)` calls `arm_cfft_init_f32(512)`, so
the set is: rfft-1024 twiddles + cfft-512 twiddles + 512 bit-reversal indices.
Omit any one and you get an `undefined reference to '...'` at link time.

## Gotcha
The raw upstream library ships a `Testing/` folder with startup files for other
chips (ARMCM7 FVP) that auto-discovery tries to compile. That is why only this
trimmed subset is vendored and `deps/CMSIS-DSP.mtb` must NOT be present — a
`.mtb` re-pulls the full library into `mtb_shared/` and breaks the build.
