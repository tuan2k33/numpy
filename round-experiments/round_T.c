/*
 * Prototype of "compare x with the double nearest to the neighbouring decimal midpoint".
 * One path, no fma, no branches in the loop body (selects only).
 *
 *   t = x*p (or x/p), n = rint(t)                    estimate, within +-1 of the answer
 *   T = fl((n +- 0.5) / p)  (or * p)                  the midpoint on the side of t: ONE division
 *   x > T -> n+1   x < T -> n    x == T -> tie, take the even one
 *   result = n' / p  (or * p)
 * valid for 10^|d| exact (float32: |d|<=10, float64: |d|<=22) and |t| < 2^22 / 2^50
 * build: gcc -O3 -mavx2 -ffp-contract=off -fno-math-errno -fno-trapping-math -shared -fPIC round_T.c -o libround_T.so -lm
 */
#include <math.h>
static const float  PF[11] = {1e0f,1e1f,1e2f,1e3f,1e4f,1e5f,1e6f,1e7f,1e8f,1e9f,1e10f};
static const double PD[23] = {1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12,1e13,1e14,1e15,
                              1e16,1e17,1e18,1e19,1e20,1e21,1e22};

#define GEN(NAME, T_, PTAB, LIM, RINT, FABS, COPYSIGN)                                                         \
void NAME##_plain(const T_ *x, T_ *out, long n, int d) {                                             \
    T_ p = PTAB[d >= 0 ? d : -d];                                                                    \
    if (d >= 0) for (long i = 0; i < n; i++) out[i] = RINT(x[i] * p) / p;                            \
    else        for (long i = 0; i < n; i++) out[i] = RINT(x[i] / p) * p;                            \
}                                                                                                    \
void NAME##_T1(const T_ *x, T_ *out, long n, int d) {      /* one midpoint */                        \
    T_ p = PTAB[d >= 0 ? d : -d];                                                                    \
    if (d >= 0) {                                                                                    \
        for (long i = 0; i < n; i++) {                                                               \
            T_ xv = x[i], t = xv * p, r = RINT(t);                                                   \
            T_ s = (t >= r) ? (T_)0.5 : (T_)-0.5;                                                    \
            T_ T = (r + s) / p;                                                                      \
            T_ odd = (r * (T_)0.5 != RINT(r * (T_)0.5)) ? (T_)1 : (T_)0;                             \
            T_ eq = (xv == T) ? (T_)1 : (T_)0, gt = (xv > T) ? (T_)1 : (T_)0, lt = (xv < T) ? (T_)1 : (T_)0; \
            T_ step = (s > 0) ? (gt + eq * odd) : -(lt + eq * odd);                                  \
            T_ res = COPYSIGN((r + step) / p, xv);                                                   \
            out[i] = (FABS(t) < (T_)LIM) ? res : xv;                                                 \
        }                                                                                            \
    } else {                                                                                         \
        for (long i = 0; i < n; i++) {                                                               \
            T_ xv = x[i], t = xv / p, r = RINT(t);                                                   \
            T_ s = (t >= r) ? (T_)0.5 : (T_)-0.5;                                                    \
            T_ T = (r + s) * p;                                                                      \
            T_ odd = (r * (T_)0.5 != RINT(r * (T_)0.5)) ? (T_)1 : (T_)0;                             \
            T_ eq = (xv == T) ? (T_)1 : (T_)0, gt = (xv > T) ? (T_)1 : (T_)0, lt = (xv < T) ? (T_)1 : (T_)0; \
            T_ step = (s > 0) ? (gt + eq * odd) : -(lt + eq * odd);                                  \
            T_ res = COPYSIGN((r + step) * p, xv);                                                   \
            out[i] = (FABS(t) < (T_)LIM) ? res : COPYSIGN(r * p, xv);                                \
        }                                                                                            \
    }                                                                                                \
}                                                                                                    \
void NAME##_T2(const T_ *x, T_ *out, long n, int d) {      /* both midpoints, for timing only, d>=0 */ \
    T_ p = PTAB[d >= 0 ? d : -d];                                                                    \
    for (long i = 0; i < n; i++) {                                                                   \
        T_ xv = x[i], t = xv * p, r = RINT(t);                                                       \
        T_ Tlo = (r - (T_)0.5) / p, Thi = (r + (T_)0.5) / p;                                         \
        T_ odd = (r * (T_)0.5 != RINT(r * (T_)0.5)) ? (T_)1 : (T_)0;                                 \
        T_ up = (xv > Thi) ? (T_)1 : (xv == Thi ? odd : (T_)0);                                      \
        T_ dn = (xv < Tlo) ? (T_)1 : (xv == Tlo ? odd : (T_)0);                                      \
        T_ nn = r + up - dn;                                                                         \
        out[i] = (FABS(t) < (T_)LIM) ? nn / p : xv;                                                  \
    }                                                                                                \
}

GEN(f32, float,  PF, 4194304.0f,         rintf, fabsf, copysignf)
GEN(f64, double, PD, 1125899906842624.0, rint,  fabs,  copysign)
