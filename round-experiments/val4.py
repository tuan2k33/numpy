import math, random, sys, time
import numpy as np
from fractions import Fraction
rng = random.Random(2026)
LIM = 2.0**53
def alg(x, d):
    """candidate algorithm in f64 (x is a python float)"""
    if d >= 0:
        p = 10.0**d; s = x*p
        if not abs(s) < LIM: return x
        e = math.fma(x, p, -s)
    else:
        p = 10.0**(-d); s = x/p
        if not abs(s) < LIM: return x
        e = math.fma(-s, p, x)/p
    nh = math.copysign(float(round(s)), s); r = (s - nh) + e
    if r > 0.5 or (r == 0.5 and nh % 2 != 0): nh += 1
    elif r < -0.5 or (r == -0.5 and nh % 2 != 0): nh -= 1
    return nh/p if d >= 0 else nh*p
def same(a, b): return a == b and math.copysign(1, a) == math.copysign(1, b) or (a != a and b != b)

# T0: table exactness
print('T0 10^k exact in f64 for k<=22:', all(float(10**k) == 10.0**k and Fraction(10.0**k) == 10**k for k in range(23)),
      '| in f32 for k<=10:', all(Fraction(float(np.float32(10**k))) == 10**k for k in range(11)), '| f32 10^11 exact:', Fraction(float(np.float32(10**11))) == 10**11)

# T2: extremes
specials = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, -1.7976931348623157e308,
            2.0**52, 2.0**53, 2.0**53+2, 2.0**63, 0.5, -0.5, 1.5, 2.5, 0.49999999999999994, 4503599627370497.5]
bad = 0; ex = None
for x in specials:
    for d in range(-8, 8):
        a = alg(x, d); r = round(x, d) if math.isfinite(x) else x
        if not same(a, r): bad += 1; ex = ex or (x, d, r, a)
print('T2a specials/edges:', 'OK' if bad == 0 else f'{bad} differ, e.g. {ex}')
bad = 0; ex = None; N = 400000
for _ in range(N):
    x = rng.choice((1, -1)) * 2.0**rng.uniform(-1074, 1023); d = rng.randint(-8, 7)
    a = alg(x, d); r = round(x, d)
    if not same(a, r): bad += 1; ex = ex or (x, d, r, a)
print(f'T2b log-uniform over the whole f64 range ({N}):', 'OK' if bad == 0 else f'{bad} differ, e.g. {ex}')

# T3: float16 exhaustive
def ref_low(x, d, dt):
    q = Fraction(float(x)) * Fraction(10)**d
    t = Fraction(round(q)) / Fraction(10)**d
    if abs(t) >= (65520 if dt is np.float16 else Fraction(2)**128 - Fraction(2)**103): return dt(np.inf if t > 0 else -np.inf)
    c = dt(float(t)); cs = [c, np.nextafter(c, dt(np.inf)), np.nextafter(c, dt(-np.inf))]
    best = min(cs, key=lambda v: (abs(Fraction(float(v)) - t) if np.isfinite(v) else Fraction(10**9), v != c))
    return best
def alg_low(x, d, dt): return dt(alg(float(x), d))
f16 = np.arange(65536, dtype=np.uint16).view(np.float16); f16 = f16[np.isfinite(f16)]
t0 = time.time(); bad = 0; ex = None
for d in range(-8, 8):
    for x in f16:
        r = ref_low(x, d, np.float16); a = alg_low(x, d, np.float16)
        if not (r == a or (np.isnan(r) and np.isnan(a))): bad += 1; ex = ex or (x, d, r, a)
print(f'T3 float16 EXHAUSTIVE ({len(f16)} values x 16 decimals): ', 'OK' if bad == 0 else f'{bad} differ, e.g. {ex}', f'[{time.time()-t0:.0f}s]')

# T4: float32 adversarial + random
def ties32(d):
    k = rng.randint(-10**8, 10**8) if d >= 0 else rng.randint(-10**6, 10**6)
    x = np.float32(float(Fraction(2*k+1, 2) / Fraction(10)**d))
    return x
N = 60000; bad = 0; ex = None
for i in range(N):
    d = rng.randint(-8, 7); m = i % 3
    if m == 0: x = np.float32(rng.uniform(-1, 1) * 10.0**rng.randint(-8, 12))
    elif m == 1: x = ties32(d)
    else: x = np.nextafter(ties32(d), np.float32(np.inf if rng.random() < .5 else -np.inf))
    if not np.isfinite(x): continue
    r = ref_low(x, d, np.float32); a = alg_low(x, d, np.float32)
    if r != a: bad += 1; ex = ex or (x, d, r, a)
print(f'T4 float32 random+ties+1ulp ({N}):', 'OK' if bad == 0 else f'{bad} differ, e.g. {ex}')
