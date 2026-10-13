import math, random
from fractions import Fraction
rng = random.Random(7)
LIM = 2.0**53
def new2(x, d):
    if d >= 0:
        p = 10.0**d; s = x*p
        if not abs(s) < LIM: return x
        e = math.fma(x, p, -s)
    else:
        p = 10.0**(-d); s = x/p
        if not abs(s) < LIM: return x
        e = math.fma(-s, p, x)
    nh = float(round(s))                  # half-even
    r = (s - nh) + (e if d >= 0 else e/p)
    if r > 0.5 or (r == 0.5 and nh % 2 != 0): nh += 1
    elif r < -0.5 or (r == -0.5 and nh % 2 != 0): nh -= 1
    return nh/p if d >= 0 else nh*p
def sample(kind):
    d = rng.randint(-8, 7)
    if kind == 0: return rng.uniform(-1, 1) * 10.0**rng.randint(-8, 17), d
    if kind == 3:                                   # target the s in [2^52, 2^53) band
        x = rng.uniform(2.0**52, 2.0**53) / (10.0**d if d >= 0 else 10.0**d); return x * rng.choice((1, -1)), d
    x = float(Fraction(2*rng.randint(-10**9, 10**9)+1, 2) / Fraction(10)**d)
    if kind == 2: x = math.nextafter(x, math.inf if rng.random() < .5 else -math.inf)
    return x, d
for kind, name in ((0,'random (to 1e17)'),(1,'exact half-way'),(2,'1ulp off half-way'),(3,'s in [2^52,2^53)')):
    N = 300000; b = 0; ex = None
    for _ in range(N):
        x, d = sample(kind); ref = round(x, d); n = new2(x, d)
        if n != ref and not (n != n and ref != ref): b += 1; ex = ex or (x, d, ref, n)
    print(f'{name:22s} mismatches vs python.round: {b}/{N}', ex or '')
