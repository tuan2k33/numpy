// Exhaustive check (CPU only): float32 x all 2^32 bit patterns x decimals in [-8,7].
// alg(): double-intermediate + fma tie correction (the design in ROUND_FUSED_PLAN.md section 9.2).
// ref(): exact integer arithmetic (u64/u128), independent of floating point division.
// build: gcc -O2 -fopenmp -mfma -mavx2 -ffp-contract=off exh32.c -o exh32 -lm
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
typedef unsigned __int128 u128;
static const double P10[9] = {1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
static const uint64_t U10[9] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
#define LIM 9007199254740992.0

static inline float alg(float x, int d) {      // NATIVE float32 arithmetic (what the 8-lane kernel does): fmaf tie correction, guard at 2^24
    float p = (float)P10[d >= 0 ? d : -d], s, e;
    if (d >= 0) { s = x * p; if (!(fabsf(s) < 16777216.0f)) return x; e = fmaf(x, p, -s); }
    else        { s = x / p; if (!(fabsf(s) < 16777216.0f)) return x; e = fmaf(-s, p, x); }
    float nh = rintf(s), diff = s - nh;
    if (diff == 0.5f && e > 0.0f) nh += 1.0f;
    else if (diff == -0.5f && e < 0.0f) nh -= 1.0f;
    return d >= 0 ? nh / p : nh * p;
}

static inline void decomp(float f, uint64_t *M, int *E) {   // positive finite f = M * 2^E
    uint32_t b; memcpy(&b, &f, 4);
    int ex = (b >> 23) & 255; uint32_t fr = b & 0x7fffff;
    if (ex == 0) { *M = fr; *E = -149; } else { *M = fr | 0x800000; *E = ex - 150; }
}

static float u128_to_f32(u128 t) {                           // correctly rounded (half-even), software
    if (t == 0) return 0.0f;
    int bits = 128; u128 top = (u128)1 << 127;
    while (!(t & top)) { top >>= 1; bits--; }
    if (bits <= 24) return (float)(uint32_t)t;
    int sh = bits - 24;
    uint64_t mant = (uint64_t)(t >> sh);
    u128 rem = t & (((u128)1 << sh) - 1), half = (u128)1 << (sh - 1);
    if (rem > half || (rem == half && (mant & 1))) mant++;
    return ldexpf((float)mant, sh);
}

static inline float ref(float xf, int d) {
    uint32_t b; memcpy(&b, &xf, 4);
    int neg = b >> 31;
    float ax = fabsf(xf);
    uint64_t m; int e2; decomp(ax, &m, &e2);
    if (m == 0) return xf;
    float zero = neg ? -0.0f : 0.0f;
    float out;
    if (d >= 0) {
        if (e2 >= 0) return xf;                              // already an integer
        uint64_t p = U10[d]; int sh = -e2;
        uint64_t num = m * p, n;
        if (sh >= 64) n = 0;
        else {
            uint64_t q = num >> sh, rem = num & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
            n = q + (rem > half || (rem == half && (q & 1)));
        }
        if (n == 0) return zero;
        float c0 = (float)((double)n / (double)p);
        float cs[3] = {c0, nextafterf(c0, INFINITY), nextafterf(c0, 0.0f)};
        float best = c0; u128 bestd = ~(u128)0; uint64_t bestM = 0;
        for (int i = 0; i < 3; i++) {
            float c = cs[i]; if (!(c > 0) || !isfinite(c)) continue;
            uint64_t M; int E; decomp(c, &M, &E);
            u128 A, B, dist;
            if (E >= 0) { A = ((u128)M * p) << E; B = n; }
            else        { A = (u128)M * p; B = ((u128)n) << (-E); }
            dist = A > B ? A - B : B - A;
            if (dist < bestd || (dist == bestd && (M & 1) == 0 && (bestM & 1))) { bestd = dist; best = c; bestM = M; }
        }
        out = best;
    } else {
        uint64_t pk = U10[-d]; u128 n;
        if (e2 >= 0) {
            u128 X = (u128)m << e2, q = X / pk, rem = X % pk;
            n = q + (rem * 2 > pk || (rem * 2 == pk && (q & 1)));
        } else {
            int sh = -e2;
            if (sh >= 100) n = 0;
            else {
                u128 den = (u128)pk << sh, q = (u128)m / den, rem = (u128)m % den;
                n = q + (rem * 2 > den || (rem * 2 == den && (q & 1)));
            }
        }
        if (n == 0) return zero;
        out = u128_to_f32(n * pk);
    }
    return neg ? -out : out;
}

int main(int argc, char **argv) {
    int dlo = -8, dhi = 7;
    uint64_t lo = 0, hi = 1ULL << 32;
    if (argc > 1) { dlo = dhi = atoi(argv[1]); }
    if (argc > 3) { lo = strtoull(argv[2], 0, 0); hi = strtoull(argv[3], 0, 0); }
    // self test on known values
    struct { float x; int d; float want; } t[] = {{2.675f, 2, 2.67f}, {2.5f, 0, 2.0f}, {3.5f, 0, 4.0f}, {0.125f, 2, 0.12f}, {1250.f, -2, 1200.f}, {-0.3f, 0, -0.0f}};
    for (unsigned i = 0; i < sizeof t / sizeof *t; i++) {
        float a = alg(t[i].x, t[i].d), r = ref(t[i].x, t[i].d);
        printf("selftest x=%g d=%d: alg=%g ref=%g (expect %g)\n", t[i].x, t[i].d, a, r, t[i].want);
    }
    for (int d = dlo; d <= dhi; d++) {
        double t0 = omp_get_wtime();
        uint64_t bad = 0, firstb[4] = {0}; int nf = 0;
        #pragma omp parallel for schedule(static, 1<<16) reduction(+:bad)
        for (uint64_t i = lo; i < hi; i++) {
            uint32_t b = (uint32_t)i;
            if (((b >> 23) & 255) == 255) continue;           // inf/nan: alg returns x by the guard
            float x; memcpy(&x, &b, 4);
            float a = alg(x, d), r = ref(x, d);
            uint32_t ab, rb; memcpy(&ab, &a, 4); memcpy(&rb, &r, 4);
            if (ab != rb) {
                bad++;
                #pragma omp critical
                { if (nf < 4) firstb[nf++] = b; }
            }
        }
        printf("d=%2d: mismatches=%llu  (%.1fs)", d, (unsigned long long)bad, omp_get_wtime() - t0);
        for (int k = 0; k < nf; k++) {
            uint32_t b = (uint32_t)firstb[k]; float x; memcpy(&x, &b, 4);
            printf("  [x=%.9g alg=%.9g ref=%.9g]", x, alg(x, d), ref(x, d));
        }
        printf("\n"); fflush(stdout);
    }
    return 0;
}
