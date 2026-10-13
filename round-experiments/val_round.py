# Throwaway validation of the *math* (not an implementation): is "fma tie-break" rounding
# == Python's correctly-rounded round(x, d) for d in [-8, 7]?
import math, random, struct, numpy as np
from fractions import Fraction
rng = random.Random(1)
LIM = 2.0**52

def old(x, d):                       # current numpy algorithm (f64)
    if d >= 0:
        p = 10.0**d; return round(x*p)/p          # python round() on float = half-even
    p = 10.0**(-d); return round(x/p)*p

def new(x, d):                       # fma tie-break
    if d >= 0:
        p = 10.0**d; s = x*p
        if not abs(s) < LIM: return x
        e = math.fma(x, p, -s)
    else:
        p = 10.0**(-d); s = x/p
        if not abs(s) < LIM: return x
        e = math.fma(-s, p, x)                    # x - s*p, exact residual
    fl = math.floor(s)
    if s - fl == 0.5:
        n = fl + 1 if e > 0 else (fl if e < 0 else (fl if fl % 2 == 0 else fl + 1))
    else:
        n = round(s)
    return n/p if d >= 0 else n*p

def sample(kind):
    d = rng.randint(-8, 7)
    if kind == 0:   x = rng.uniform(-1, 1) * 10.0**rng.randint(-8, 12)
    elif kind == 1: x = float(Fraction(2*rng.randint(-10**9, 10**9)+1, 2) / Fraction(10)**d)  # exact half-way decimal, nearest double
    else:
        x = float(Fraction(2*rng.randint(-10**9, 10**9)+1, 2) / Fraction(10)**d)
        x = math.nextafter(x, math.inf if rng.random() < .5 else -math.inf)         # 1 ulp off a tie
    return x, d

for kind, name in ((0, 'random'), (1, 'exact half-way decimals'), (2, '1 ulp next to half-way')):
    N = 300000; bo = bn = 0; ex = None
    for _ in range(N):
        x, d = sample(kind)
        ref = round(x, d)
        o = old(x, d); n = new(x, d)
        bo += (o != ref); bn += (n != ref)
        if n != ref and ex is None: ex = (x, d, ref, n)
    print(f'f64 {name:26s}: old != python.round: {bo:6d}/{N} ({100*bo/N:.2f}%)   new != python.round: {bn}', ex or '')

# float32 / float16 via exact float64 product, then one cast
def ref_low(x, d, dt):               # correctly rounded dt result of the exact decimal rounding of x
    q = Fraction(x) * Fraction(10)**d
    n = round(q)                      # Fraction round = half-even, exact
    t = Fraction(n) / Fraction(10)**d
    c = dt(float(t)); cands = [c, np.nextafter(c, dt(np.inf)), np.nextafter(c, dt(-np.inf))]
    return min(cands, key=lambda v: (abs(Fraction(float(v)) - t), v != c))

def new_low(x, d, dt):
    p = 10.0**abs(d); xf = float(x)
    s = xf*p if d >= 0 else xf/p     # d>=0: exact in f64 for f32/f16 inputs
    n = round(s)
    return dt(n/p if d >= 0 else n*p)

for dt, nm, lo, hi in ((np.float32, 'f32', -6, 8), (np.float16, 'f16', -3, 4)):
    N = 40000; bad = badnewd = 0; ex = None
    for _ in range(N):
        d = rng.randint(-8, 7)
        x = dt(rng.uniform(-1, 1) * 10.0**rng.randint(lo, hi))
        if d < 0 and nm == 'f32': pass
        s = float(x) * 10.0**d if d >= 0 else None
        r = ref_low(x, d, dt); m = new_low(x, d, dt)
        if r != m:
            bad += 1; ex = ex or (x, d, r, m)
    print(f'{nm} (f64 intermediate, one cast): mismatches vs exact reference: {bad}/{N}', ex or '')
