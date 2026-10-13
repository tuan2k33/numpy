// CPU-only micro-benchmark, single thread, AVX2+FMA. PROXY for the numpy kernel (no ufunc dispatch overhead modelled).
// A3  : three separate passes mul -> rint -> div (what PyArray_Round does today, with SIMD loops like numpy's)
// B0  : fused single pass, plain mul -> rint -> div   (shows pure fusion gain, same results as A3)
// B1  : fused single pass, exact algorithm (fma + tie correction + guard) [f64]
// f32 : A3f three float passes (today)   Bf: fused, double intermediate (simple variant validated exhaustively)
// build: gcc -O2 -mavx2 -mfma -ffp-contract=off bench_round.c -o bench_round -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define RN (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---------------- f64 ---------------- */
static void a3_f64(const double *x, double *y, size_t n, double p, int neg) {
    __m256d vp = _mm256_set1_pd(p);
    for (size_t i = 0; i < n; i += 4) { __m256d v = _mm256_loadu_pd(x + i); _mm256_storeu_pd(y + i, neg ? _mm256_div_pd(v, vp) : _mm256_mul_pd(v, vp)); }
    for (size_t i = 0; i < n; i += 4) _mm256_storeu_pd(y + i, _mm256_round_pd(_mm256_loadu_pd(y + i), RN));
    for (size_t i = 0; i < n; i += 4) { __m256d v = _mm256_loadu_pd(y + i); _mm256_storeu_pd(y + i, neg ? _mm256_mul_pd(v, vp) : _mm256_div_pd(v, vp)); }
}
static void b0_f64(const double *x, double *y, size_t n, double p, int neg) {
    __m256d vp = _mm256_set1_pd(p);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        __m256d s = neg ? _mm256_div_pd(v, vp) : _mm256_mul_pd(v, vp);
        s = _mm256_round_pd(s, RN);
        _mm256_storeu_pd(y + i, neg ? _mm256_mul_pd(s, vp) : _mm256_div_pd(s, vp));
    }
}
static void b1_f64(const double *x, double *y, size_t n, double p, int neg) {
    const __m256d vp = _mm256_set1_pd(p), lim = _mm256_set1_pd(9007199254740992.0);
    const __m256d half = _mm256_set1_pd(0.5), mhalf = _mm256_set1_pd(-0.5), one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    const __m256d absm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    for (size_t i = 0; i < n; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i), s, e;
        if (neg) { s = _mm256_div_pd(v, vp); e = _mm256_fnmadd_pd(s, vp, v); }
        else     { s = _mm256_mul_pd(v, vp); e = _mm256_fmsub_pd(v, vp, s); }
        __m256d nh = _mm256_round_pd(s, RN);
        __m256d diff = _mm256_sub_pd(s, nh);
        __m256d up = _mm256_and_pd(_mm256_cmp_pd(diff, half, _CMP_EQ_OQ), _mm256_cmp_pd(e, zero, _CMP_GT_OQ));
        __m256d dn = _mm256_and_pd(_mm256_cmp_pd(diff, mhalf, _CMP_EQ_OQ), _mm256_cmp_pd(e, zero, _CMP_LT_OQ));
        nh = _mm256_sub_pd(_mm256_add_pd(nh, _mm256_and_pd(up, one)), _mm256_and_pd(dn, one));
        __m256d res = neg ? _mm256_mul_pd(nh, vp) : _mm256_div_pd(nh, vp);
        __m256d ok = _mm256_cmp_pd(_mm256_and_pd(s, absm), lim, _CMP_LT_OQ);
        _mm256_storeu_pd(y + i, _mm256_blendv_pd(v, res, ok));
    }
}
/* ---------------- f32 ---------------- */
static void a3_f32(const float *x, float *y, size_t n, float p, int neg) {
    __m256 vp = _mm256_set1_ps(p);
    for (size_t i = 0; i < n; i += 8) { __m256 v = _mm256_loadu_ps(x + i); _mm256_storeu_ps(y + i, neg ? _mm256_div_ps(v, vp) : _mm256_mul_ps(v, vp)); }
    for (size_t i = 0; i < n; i += 8) _mm256_storeu_ps(y + i, _mm256_round_ps(_mm256_loadu_ps(y + i), RN));
    for (size_t i = 0; i < n; i += 8) { __m256 v = _mm256_loadu_ps(y + i); _mm256_storeu_ps(y + i, neg ? _mm256_mul_ps(v, vp) : _mm256_div_ps(v, vp)); }
}
static inline __m256d f32half(__m256d v, __m256d vp, int neg) {
    __m256d s = neg ? _mm256_div_pd(v, vp) : _mm256_mul_pd(v, vp);
    s = _mm256_round_pd(s, RN);
    return neg ? _mm256_mul_pd(s, vp) : _mm256_div_pd(s, vp);
}
static void bf_f32(const float *x, float *y, size_t n, double p, int neg) {
    __m256d vp = _mm256_set1_pd(p);
    for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
        __m128 rlo = _mm256_cvtpd_ps(f32half(_mm256_cvtps_pd(lo), vp, neg));
        __m128 rhi = _mm256_cvtpd_ps(f32half(_mm256_cvtps_pd(hi), vp, neg));
        _mm256_storeu_ps(y + i, _mm256_insertf128_ps(_mm256_castps128_ps256(rlo), rhi, 1));
    }
}


/* f32 native (8 lanes), fma tie-correction, guard at 2^24 */
static void bn_f32(const float *x, float *y, size_t n, float p, int neg) {
    const __m256 vp = _mm256_set1_ps(p), lim = _mm256_set1_ps(16777216.0f);
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i), s, e;
        if (neg) { s = _mm256_div_ps(v, vp); e = _mm256_fnmadd_ps(s, vp, v); }
        else     { s = _mm256_mul_ps(v, vp); e = _mm256_fmsub_ps(v, vp, s); }
        __m256 nh = _mm256_round_ps(s, RN);
        __m256 diff = _mm256_sub_ps(s, nh);
        __m256 up = _mm256_and_ps(_mm256_cmp_ps(diff, half, _CMP_EQ_OQ), _mm256_cmp_ps(e, zero, _CMP_GT_OQ));
        __m256 dn = _mm256_and_ps(_mm256_cmp_ps(diff, mhalf, _CMP_EQ_OQ), _mm256_cmp_ps(e, zero, _CMP_LT_OQ));
        nh = _mm256_sub_ps(_mm256_add_ps(nh, _mm256_and_ps(up, one)), _mm256_and_ps(dn, one));
        __m256 res = neg ? _mm256_mul_ps(nh, vp) : _mm256_div_ps(nh, vp);
        __m256 ok = _mm256_cmp_ps(_mm256_and_ps(s, absm), lim, _CMP_LT_OQ);
        _mm256_storeu_ps(y + i, _mm256_blendv_ps(v, res, ok));
    }
}

/* ---- B2: exact, with a cheap fast path; the fma/tie correction only runs for vectors that contain a tie or a guard lane ---- */
static void b2_f64(const double *x, double *y, size_t n, double p, int neg) {
    const __m256d vp = _mm256_set1_pd(p), lim = _mm256_set1_pd(9007199254740992.0);
    const __m256d half = _mm256_set1_pd(0.5), mhalf = _mm256_set1_pd(-0.5), one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    const __m256d absm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    for (size_t i = 0; i < n; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        __m256d s = neg ? _mm256_div_pd(v, vp) : _mm256_mul_pd(v, vp);
        __m256d nh = _mm256_round_pd(s, RN);
        __m256d diff = _mm256_sub_pd(s, nh);
        __m256d tie = _mm256_cmp_pd(_mm256_and_pd(diff, absm), half, _CMP_EQ_OQ);
        __m256d big = _mm256_cmp_pd(_mm256_and_pd(s, absm), lim, _CMP_NLT_UQ);   /* >= 2^53, inf or nan */
        if (__builtin_expect(_mm256_movemask_pd(_mm256_or_pd(tie, big)) == 0, 1)) {
            _mm256_storeu_pd(y + i, neg ? _mm256_mul_pd(nh, vp) : _mm256_div_pd(nh, vp));
        } else {
            __m256d e = neg ? _mm256_fnmadd_pd(s, vp, v) : _mm256_fmsub_pd(v, vp, s);
            __m256d up = _mm256_and_pd(_mm256_cmp_pd(diff, half, _CMP_EQ_OQ), _mm256_cmp_pd(e, zero, _CMP_GT_OQ));
            __m256d dn = _mm256_and_pd(_mm256_cmp_pd(diff, mhalf, _CMP_EQ_OQ), _mm256_cmp_pd(e, zero, _CMP_LT_OQ));
            nh = _mm256_sub_pd(_mm256_add_pd(nh, _mm256_and_pd(up, one)), _mm256_and_pd(dn, one));
            __m256d res = neg ? _mm256_mul_pd(nh, vp) : _mm256_div_pd(nh, vp);
            _mm256_storeu_pd(y + i, _mm256_blendv_pd(v, res, _mm256_cmp_pd(_mm256_and_pd(s, absm), lim, _CMP_LT_OQ)));
        }
    }
}
static void b2_f32(const float *x, float *y, size_t n, float p, int neg) {
    const __m256 vp = _mm256_set1_ps(p), lim = _mm256_set1_ps(16777216.0f);
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 s = neg ? _mm256_div_ps(v, vp) : _mm256_mul_ps(v, vp);
        __m256 nh = _mm256_round_ps(s, RN);
        __m256 diff = _mm256_sub_ps(s, nh);
        __m256 tie = _mm256_cmp_ps(_mm256_and_ps(diff, absm), half, _CMP_EQ_OQ);
        __m256 big = _mm256_cmp_ps(_mm256_and_ps(s, absm), lim, _CMP_NLT_UQ);
        if (__builtin_expect(_mm256_movemask_ps(_mm256_or_ps(tie, big)) == 0, 1)) {
            _mm256_storeu_ps(y + i, neg ? _mm256_mul_ps(nh, vp) : _mm256_div_ps(nh, vp));
        } else {
            __m256 e = neg ? _mm256_fnmadd_ps(s, vp, v) : _mm256_fmsub_ps(v, vp, s);
            __m256 up = _mm256_and_ps(_mm256_cmp_ps(diff, half, _CMP_EQ_OQ), _mm256_cmp_ps(e, zero, _CMP_GT_OQ));
            __m256 dn = _mm256_and_ps(_mm256_cmp_ps(diff, mhalf, _CMP_EQ_OQ), _mm256_cmp_ps(e, zero, _CMP_LT_OQ));
            nh = _mm256_sub_ps(_mm256_add_ps(nh, _mm256_and_ps(up, one)), _mm256_and_ps(dn, one));
            __m256 res = neg ? _mm256_mul_ps(nh, vp) : _mm256_div_ps(nh, vp);
            _mm256_storeu_ps(y + i, _mm256_blendv_ps(v, res, _mm256_cmp_ps(_mm256_and_ps(s, absm), lim, _CMP_LT_OQ)));
        }
    }
}

typedef void (*fn64)(const double *, double *, size_t, double, int);
typedef void (*fn32)(const float *, float *, size_t, float, int);
typedef void (*fn32d)(const float *, float *, size_t, double, int);
static volatile double sink;

static double bench64(fn64 f, const double *x, double *y, size_t n, double p, int neg) {
    size_t reps = (size_t)(4e7 / n) + 1; double best = 1e9;
    for (int b = 0; b < 7; b++) { double t0 = now(); for (size_t r = 0; r < reps; r++) { f(x, y, n, p, neg); sink = y[r % n]; } double t = (now() - t0) / reps / n * 1e9; if (t < best) best = t; }
    return best;
}
static double bench32(fn32 f, const float *x, float *y, size_t n, float p, int neg) {
    size_t reps = (size_t)(4e7 / n) + 1; double best = 1e9;
    for (int b = 0; b < 7; b++) { double t0 = now(); for (size_t r = 0; r < reps; r++) { f(x, y, n, p, neg); sink = y[r % n]; } double t = (now() - t0) / reps / n * 1e9; if (t < best) best = t; }
    return best;
}
static double bench32d(fn32d f, const float *x, float *y, size_t n, double p, int neg) {
    size_t reps = (size_t)(4e7 / n) + 1; double best = 1e9;
    for (int b = 0; b < 7; b++) { double t0 = now(); for (size_t r = 0; r < reps; r++) { f(x, y, n, p, neg); sink = y[r % n]; } double t = (now() - t0) / reps / n * 1e9; if (t < best) best = t; }
    return best;
}

static void fill64(double *x, size_t n, int mode, int d) {
    srand(1);
    for (size_t i = 0; i < n; i++) {
        if (mode == 0) x[i] = ((double)rand() / RAND_MAX - 0.5) * 2000.0;                      /* generic random */
        else { long k = rand() % 200000 - 100000; x[i] = ((double)k + 0.5) / pow(10.0, d >= 0 ? d : 0) * (d >= 0 ? 1 : 1); if (d < 0) x[i] = ((double)k + 0.5) * pow(10.0, -d); } /* every lane a tie */
    }
}
int main(void) {
    size_t sizes[] = {24, 1000, 10000, 100000, 1000000, 8000000};
    int ds[] = {2, -2};
    for (int mode = 0; mode < 2; mode++)
    for (int di = 0; di < 2; di++) {
        int d = ds[di], neg = d < 0; double p = pow(10.0, abs(d));
        printf("\n=== data: %s   decimals=%d ===\n", mode ? "EVERY LANE A TIE (worst case)" : "random", d);
        printf("  f64     n      A3(today)   B0(plain)   B1(exact,full)  B2(exact,fast path)   B2/A3\n");
        for (int si = 0; si < 6; si++) {
            size_t n = sizes[si];
            double *x = aligned_alloc(64, n * 8 + 64), *y = aligned_alloc(64, n * 8 + 64);
            fill64(x, n, mode, d);
            double a = bench64(a3_f64, x, y, n, p, neg), b0 = bench64(b0_f64, x, y, n, p, neg), b1 = bench64(b1_f64, x, y, n, p, neg), b2 = bench64(b2_f64, x, y, n, p, neg);
            printf("  %9zu   %7.3f     %7.3f     %7.3f         %7.3f           %5.2fx\n", n, a, b0, b1, b2, b2 / a);
            free(x); free(y);
        }
        printf("  f32     n      A3f(today)  Bn(exact,full)  B2f(exact,fast path)   B2f/A3f\n");
        for (int si = 0; si < 6; si++) {
            size_t n = sizes[si];
            double *xd = aligned_alloc(64, n * 8 + 64); float *x = aligned_alloc(64, n * 4 + 64), *y = aligned_alloc(64, n * 4 + 64);
            fill64(xd, n, mode, d); for (size_t i = 0; i < n; i++) x[i] = (float)xd[i];
            double a = bench32(a3_f32, x, y, n, (float)p, neg), bn = bench32(bn_f32, x, y, n, (float)p, neg), b2 = bench32(b2_f32, x, y, n, (float)p, neg);
            printf("  %9zu   %7.3f     %7.3f         %7.3f              %5.2fx\n", n, a, bn, b2, b2 / a);
            free(x); free(y); free(xd);
        }
    }
    return 0;
}
