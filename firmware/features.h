/*
 * features.h — on-device feature extraction for the CWRU classifier.
 * Mirrors ml/preprocess.py EXACTLY so board features == PC features:
 *   1024-sample window -> Hann -> rFFT -> |mag| bins 1..512
 *   -> average-pool x4 -> 128 -> log1p -> standardize (norm.h).
 *
 * Two FFT paths, selected at compile time (for the plain-C vs CMSIS-DSP rung):
 *   -DFE_USE_CMSIS   -> arm_rfft_fast_f32 (CMSIS-DSP)
 *   (default)        -> portable radix-2 rFFT (reference)
 *
 * Apache-2.0.
 */
#ifndef FEATURES_H
#define FEATURES_H

#include <math.h>
#include <stdint.h>
#include "norm.h"   /* FE_WIN, FE_HOP, FE_NBINS, FE_MEAN, FE_STD */

#if (FE_WIN != 1024) || (FE_NBINS != 128)
#error "features.h hardcodes 1024-pt FFT -> 128 bins; regenerate to match."
#endif

/* np.hanning(N): 0.5 - 0.5*cos(2*pi*n/(N-1)), n=0..N-1  (note N-1) */
static float fe_hann[FE_WIN];

/* scratch buffers */
static float fe_win[FE_WIN];       /* windowed input                     */
static float fe_mag[512];          /* |FFT| of bins 1..512               */

/* ---------------- FFT paths ---------------- */
#if defined(FE_USE_CMSIS)
#include "arm_math.h"
static arm_rfft_fast_instance_f32 fe_rfft;
static float fe_fft[FE_WIN];       /* packed rfft_fast output            */

static inline void fe_fft_init(void) { arm_rfft_fast_init_f32(&fe_rfft, FE_WIN); }

/* fill fe_mag[0..511] with |bin 1..512| */
static inline void fe_fft_mag(const float *x) {
    arm_rfft_fast_f32(&fe_rfft, (float *)x, fe_fft, 0);
    /* packed layout: fe_fft[0]=Re[0](DC), fe_fft[1]=Re[512](Nyquist),
       fe_fft[2k],[2k+1]=Re,Im of bin k for k=1..511 */
    for (int k = 1; k <= 511; ++k) {
        float re = fe_fft[2 * k], im = fe_fft[2 * k + 1];
        fe_mag[k - 1] = sqrtf(re * re + im * im);
    }
    fe_mag[511] = fabsf(fe_fft[1]);     /* bin 512 (Nyquist) */
}

#else  /* ---- portable radix-2 rFFT (reference kernel for the comparison) ---- */
static float fe_re[FE_WIN], fe_im[FE_WIN];

static inline void fe_fft_init(void) { /* nothing to precompute */ }

static void fe_fft_mag(const float *x) {
    const int N = FE_WIN;
    for (int i = 0; i < N; ++i) { fe_re[i] = x[i]; fe_im[i] = 0.0f; }
    /* bit-reversal */
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = fe_re[i]; fe_re[i] = fe_re[j]; fe_re[j] = t;
                     t = fe_im[i]; fe_im[i] = fe_im[j]; fe_im[j] = t; }
    }
    /* Cooley-Tukey */
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * 3.14159265358979f / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                int a = i + k, b = i + k + len / 2;
                float xr = fe_re[b] * cr - fe_im[b] * ci;
                float xi = fe_re[b] * ci + fe_im[b] * cr;
                fe_re[b] = fe_re[a] - xr; fe_im[b] = fe_im[a] - xi;
                fe_re[a] += xr;           fe_im[a] += xi;
                float ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
            }
        }
    }
    for (int k = 1; k <= 512; ++k)
        fe_mag[k - 1] = sqrtf(fe_re[k] * fe_re[k] + fe_im[k] * fe_im[k]);
}
#endif

/* ---------------- public API ---------------- */
static inline void fe_init(void) {
    for (int n = 0; n < FE_WIN; ++n)
        fe_hann[n] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979f * n / (FE_WIN - 1));
    fe_fft_init();
}

/* raw 1024-sample window -> 128 standardized features (ready for the model) */
static void fe_extract(const float *win1024, float *out128) {
    for (int n = 0; n < FE_WIN; ++n) fe_win[n] = win1024[n] * fe_hann[n];
    fe_fft_mag(fe_win);                          /* -> fe_mag[0..511] */
    for (int b = 0; b < FE_NBINS; ++b) {         /* pool x4 -> 128    */
        float s = fe_mag[4*b] + fe_mag[4*b+1] + fe_mag[4*b+2] + fe_mag[4*b+3];
        float pooled = s * 0.25f;
        out128[b] = (log1pf(pooled) - FE_MEAN) / FE_STD;   /* log1p + standardize */
    }
}

#endif /* FEATURES_H */
