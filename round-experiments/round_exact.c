/*
 * Prototype: round(x, d) for double, |d| <= 22, correct for EVERY input
 * (half-even on the exact binary value, same as Python's round(float, d)).
 * No FMA, no Dekker, no x*10^d rounding error: the scaling is done in exact
 * 128-bit integer arithmetic.
 *
 *   round_exact : always the integer path (slow, scalar)
 *   round_fast  : mul/div + rint (what SIMD does today) and only the inputs that
 *                 are within the error bound of a half-way point go through the
 *                 integer path
 * build: gcc -O2 -shared -fPIC round_exact.c -o libround_exact.so -lm
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef unsigned __int128 u128;
static const double P10[23] = {1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,
    1e12,1e13,1e14,1e15,1e16,1e17,1e18,1e19,1e20,1e21,1e22};
static u128 U10[23];
static int inited;
static void init(void) { U10[0] = 1; for (int i = 1; i < 23; i++) U10[i] = U10[i-1] * 10; inited = 1; }

/* *ok = 0: outside the supported domain (d < 0 and |x| >= 2^74) */
double round_exact(double x, int d, int *ok)
{
    if (!inited) init();
    *ok = 1;
    if (!isfinite(x) || x == 0.0) return x;
    uint64_t bits; memcpy(&bits, &x, 8);
    int sign = (int)(bits >> 63), be = (int)((bits >> 52) & 0x7ff);
    uint64_t fr = bits & ((1ULL << 52) - 1), m; int e;
    if (be == 0) { m = fr; e = -1074; } else { m = fr | (1ULL << 52); e = be - 1075; }
    double r;
    if (d >= 0) {
        if (e >= 0) return x;                       /* integer valued */
        int s = -e;
        u128 N = (u128)m * U10[d], n;               /* < 2^127 */
        if (s >= 128) n = 0;
        else {
            n = N >> s;
            u128 rem = N & (((u128)1 << s) - 1), half = (u128)1 << (s - 1);
            if (rem > half || (rem == half && (n & 1))) n++;
        }
        if (n >= ((u128)1 << 53)) return x;          /* x already has no digits at this scale */
        r = (double)(uint64_t)n / P10[d];            /* n, 10^d exact: correctly rounded */
    } else {
        int k = -d; u128 P = U10[k], n;
        if (e >= 0) {
            if (e > 74) { *ok = 0; return x; }
            u128 M = (u128)m << e;
            n = M / P; u128 rem = M % P;
            if (rem * 2 > P || (rem * 2 == P && (n & 1))) n++;
        } else {
            int s = -e;
            if (s > 53) n = 0;                       /* |x| < 2^-1, divided by P >= 10 */
            else {
                u128 D = P << s; n = (u128)m / D; u128 rem = (u128)m % D;
                if (rem * 2 > D || (rem * 2 == D && (n & 1))) n++;
            }
        }
        r = (double)(n * P);                         /* exact integer -> nearest double */
    }
    return sign ? -r : r;
}

double round_fast(double x, int d, long *nfallback, long *nbad)
{
    double p = P10[d >= 0 ? d : -d];
    double t = d >= 0 ? x * p : x / p, at = fabs(t);
    if (at < 1125899906842624.0 /* 2^50 */) {
        double rt = rint(t), f = fabs(t - rt);
        if (fabs(f - 0.5) > at * 2.3e-16) return d >= 0 ? rt / p : rt * p;
    }
    int ok; double r = round_exact(x, d, &ok);
    (*nfallback)++; if (!ok) (*nbad)++;
    return r;
}

void run_exact(const double *x, double *out, long n, int d, long *nbad)
{ int ok; *nbad = 0; for (long i = 0; i < n; i++) { out[i] = round_exact(x[i], d, &ok); if (!ok) (*nbad)++; } }
void run_fast(const double *x, double *out, long n, int d, long *nfallback, long *nbad)
{ *nfallback = 0; *nbad = 0; for (long i = 0; i < n; i++) out[i] = round_fast(x[i], d, nfallback, nbad); }
void run_plain(const double *x, double *out, long n, int d)   /* the current numpy formula, scalar */
{ double p = P10[d >= 0 ? d : -d]; if (d >= 0) for (long i = 0; i < n; i++) out[i] = rint(x[i] * p) / p;
  else for (long i = 0; i < n; i++) out[i] = rint(x[i] / p) * p; }
