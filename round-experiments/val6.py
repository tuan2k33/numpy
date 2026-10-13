import math, numpy as np, time, warnings
from fractions import Fraction
warnings.simplefilter('ignore')
f32 = np.float32; f16 = np.float16
def ref_low(x, d, dt):
    q = Fraction(float(x)) * Fraction(10)**d
    t = Fraction(round(q)) / Fraction(10)**d
    if abs(t) >= 65520: return dt(np.inf if t > 0 else -np.inf)
    c = dt(float(t)); cs = [c, np.nextafter(c, dt(np.inf)), np.nextafter(c, dt(-np.inf))]
    return min(cs, key=lambda v: (abs(Fraction(float(v)) - t) if np.isfinite(v) else Fraction(10**9), v != c))
def alg32(x16, d):
    """NATIVE float32 lanes (what a vcvtph2ps -> f32 kernel would do), then ONE cast to f16.
    fma emulated through float64, which is exact for these operand widths."""
    x = f32(x16); p = f32(10.0**abs(d))
    if d >= 0:
        s = x*p; e = float(x)*float(p) - float(s)
    else:
        s = x/p; e = float(x) - float(s)*float(p)
    if not abs(s) < f32(16777216.0): return x16
    nh = f32(np.rint(s)); diff = s - nh
    if diff == f32(0.5) and e > 0: nh = nh + f32(1)
    elif diff == f32(-0.5) and e < 0: nh = nh - f32(1)
    return f16(nh/p if d >= 0 else nh*p)
def alg_dbl_simple(x16, d):
    x = float(x16); p = 10.0**abs(d); s = x*p if d >= 0 else x/p
    nh = np.rint(s); return f16(nh/p if d >= 0 else nh*p)
vals = np.arange(65536, dtype=np.uint16).view(np.float16); vals = vals[np.isfinite(vals)]
t0 = time.time(); bad32 = badd = 0; ex32 = exd = None
for d in range(-8, 8):
    for x in vals:
        r = ref_low(x, d, f16)
        a = alg32(x, d); b = alg_dbl_simple(x, d)
        if not (r == a or (np.isnan(r) and np.isnan(a))): bad32 += 1; ex32 = ex32 or (x, d, r, a)
        if not (r == b or (np.isnan(r) and np.isnan(b))): badd += 1; exd = exd or (x, d, r, b)
print(f'f16 exhaustive ({len(vals)} x 16 d)  native-f32 lanes -> one f16 cast : mismatches {bad32}', ex32 or '')
print(f'f16 exhaustive ({len(vals)} x 16 d)  double simple (no fma)          : mismatches {badd}', exd or '', f'[{time.time()-t0:.0f}s]')
