import math, random
from fractions import Fraction
rng = random.Random(99)
LIM = 2.0**53
def alg2(x, d):
    if d >= 0:
        p = 10.0**d; s = x*p
        if not abs(s) < LIM: return x
        e = math.fma(x, p, -s)
    else:
        p = 10.0**(-d); s = x/p
        if not abs(s) < LIM: return x
        e = math.fma(-s, p, x)            # sign only: no division
    nh = math.copysign(float(round(s)), s)   # rint, half-even
    diff = s - nh
    if diff == 0.5 and e > 0: nh += 1
    elif diff == -0.5 and e < 0: nh -= 1
    return nh/p if d >= 0 else nh*p
def same(a, b): return (a == b and math.copysign(1, a) == math.copysign(1, b)) or (a != a and b != b)
def sample(kind):
    d = rng.randint(-8, 7)
    if kind == 0: return rng.uniform(-1, 1) * 10.0**rng.randint(-8, 17), d
    if kind == 3: return rng.choice((1,-1)) * rng.uniform(2.0**52, 2.0**53) / 10.0**d, d
    if kind == 4: return rng.choice((1,-1)) * 2.0**rng.uniform(-1074, 1023), d
    x = float(Fraction(2*rng.randint(-10**9, 10**9)+1, 2) / Fraction(10)**d)
    if kind == 2: x = math.nextafter(x, math.inf if rng.random() < .5 else -math.inf)
    return x, d
for kind, name in ((0,'random to 1e17'),(1,'exact half-way'),(2,'1ulp off half-way'),(3,'s in [2^52,2^53)'),(4,'log-uniform full range')):
    N = 400000; b = 0; ex = None
    for _ in range(N):
        x, d = sample(kind); ref = round(x, d); n = alg2(x, d)
        if not same(n, ref): b += 1; ex = ex or (x, d, ref, n)
    print(f'{name:24s} simplified alg vs python.round: {b}/{N}', ex or '')
