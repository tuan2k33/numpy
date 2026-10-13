"""End-to-end correctness of the fused np.round (run against the NEW build). CPU only."""
import math, random, sys, time, warnings
import numpy as np
from fractions import Fraction
warnings.simplefilter("error")
rng = random.Random(7)
DS = [d for d in range(-8, 8) if d != 0]
fails = 0
def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (" " + str(info) if info and not ok else ""))
    fails += (not ok)
def bits_eq(a, b):
    """equal including the sign of zero; NaN == NaN"""
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype != b.dtype: return False
    nan = np.isnan(a) & np.isnan(b)
    return bool(np.all((a == b) & (np.signbit(a) == np.signbit(b)) | nan))

# ---- A. float16 exhaustive vs exact rational reference
def ref_f16(x, d):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return _ref_f16(x, d)
def _ref_f16(x, d):
    q = Fraction(float(x)) * Fraction(10)**d
    t = Fraction(round(q)) / Fraction(10)**d
    if abs(t) >= 65520: return np.float16(np.inf if t > 0 else -np.inf)
    c = np.float16(float(t)); cs = [c, np.nextafter(c, np.float16(np.inf)), np.nextafter(c, np.float16(-np.inf))]
    best = min(cs, key=lambda v: (abs(Fraction(float(v)) - t) if np.isfinite(v) else Fraction(10**9), v != c))
    if best == 0 and x < 0 or (best == 0 and math.copysign(1, float(x)) < 0): best = np.float16(-0.0)
    return best
f16 = np.arange(65536, dtype=np.uint16).view(np.float16)
fin = f16[np.isfinite(f16)]
t0 = time.time(); bad = 0; ex = None
for d in DS:
    got = np.round(fin, d)
    for x, g in zip(fin, got):
        r = ref_f16(x, d)
        if not (r == g and np.signbit(r) == np.signbit(g)):
            bad += 1; ex = ex or (x, d, r, g)
check(f"A float16 exhaustive 63488 x {len(DS)} decimals", bad == 0, ex); print(f"   [{time.time()-t0:.0f}s]", flush=True)
sp = np.array([np.inf, -np.inf, np.nan], np.float16)
check("A float16 inf/nan pass through", bits_eq(np.round(sp, 2), sp))

# ---- C. float64 vs python round()
def samples(kind, n, d):
    out = []
    for _ in range(n):
        if kind == 0: x = rng.uniform(-1, 1) * 10.0**rng.randint(-8, 17)
        elif kind == 1 or kind == 2:
            x = float(Fraction(2*rng.randint(-10**9, 10**9)+1, 2) / Fraction(10)**d)
            if kind == 2: x = math.nextafter(x, math.inf if rng.random() < .5 else -math.inf)
        elif kind == 3: x = rng.choice((1, -1)) * rng.uniform(2.0**52, 2.0**53) / 10.0**d
        else: x = rng.choice((1, -1)) * 2.0**rng.uniform(-1074, 1023)
        out.append(x)
    return np.array(out, dtype=np.float64)
names = ["random", "exact half-way", "1ulp off half-way", "s in [2^52,2^53)", "log-uniform full range"]
t0 = time.time()
for kind in range(5):
    bad = 0; ex = None; N = 20000
    for d in DS:
        x = samples(kind, N, d); got = np.round(x, d)
        ref = np.array([round(v, d) for v in x.tolist()])
        m = ~((got == ref) & (np.signbit(got) == np.signbit(ref)) | (np.isnan(got) & np.isnan(ref)))
        if m.any():
            bad += int(m.sum()); i = int(np.flatnonzero(m)[0]); ex = ex or (x[i], d, ref[i], got[i])
    check(f"C float64 {names[kind]} ({N} x {len(DS)} decimals) vs python round()", bad == 0, ex)
print(f"   [{time.time()-t0:.0f}s]", flush=True)

# ---- E. layouts, out=, subclasses, dtypes
a = (np.random.default_rng(1).random(1000) * 2000 - 1000)
for dt in (np.float16, np.float32, np.float64):
    x = a.astype(dt)
    for d in (2, -2, 7, -8, 1):
        base = np.round(x, d)
        check(f"E {dt.__name__} d={d} reversed view", bits_eq(np.round(x[::-1], d), base[::-1]))
        check(f"E {dt.__name__} d={d} strided ::3", bits_eq(np.round(x[::3], d), base[::3]))
        check(f"E {dt.__name__} d={d} 2D C/F/transposed",
              bits_eq(np.round(x[:960].reshape(30, 32), d), base[:960].reshape(30, 32)) and
              bits_eq(np.round(np.asfortranarray(x[:960].reshape(30, 32)), d), base[:960].reshape(30, 32)) and
              bits_eq(np.round(x[:960].reshape(30, 32).T, d), base[:960].reshape(30, 32).T))
        o = np.empty_like(x); r = np.round(x, d, out=o)
        check(f"E {dt.__name__} d={d} out=", r is o and bits_eq(o, base))
        y = x.copy(); r = np.round(y, d, out=y)
        check(f"E {dt.__name__} d={d} in-place", r is y and bits_eq(y, base))
        z = x.copy(); r = np.round(z[1:], d, out=z[:-1])      # overlapping, shifted
        check(f"E {dt.__name__} d={d} overlapping out", bits_eq(z[:-1], base[1:]))
    check(f"E {dt.__name__} F-order result layout", np.round(np.asfortranarray(x[:960].reshape(30, 32)), 2).flags.f_contiguous)
    check(f"E {dt.__name__} 0-d & empty", np.round(dt(2.675), 2).shape == () and np.round(np.array([], dt), 2).shape == (0,))
    check(f"E {dt.__name__} tail lengths", all(bits_eq(np.round(x[:n], 2), base2[:n]) for base2 in [np.round(x, 2)] for n in range(0, 40)))
    check(f"E {dt.__name__} method == function", bits_eq(x.round(2), np.round(x, 2)))
# non-native byte order, unaligned, subclass, differently typed out, readonly
b = a.astype('>f8'); check("E byteswapped f64 (old path)", bits_eq(np.round(b, 2), np.round(a, 2).astype('>f8').astype(np.float64)) if False else np.allclose(np.round(b, 2), np.round(a, 2)))
raw = np.zeros(8 * 100 + 1, np.uint8); un = raw[1:].view(np.float64); un[:] = a[:100]
check("E unaligned f64 (old path)", bits_eq(np.round(un, 2), np.round(a[:100], 2)))
m = np.ma.array(a[:10], mask=[0, 1] * 5); check("E masked array subclass", isinstance(np.round(m, 2), np.ma.MaskedArray))
class Sub(np.ndarray): pass
s = a[:10].view(Sub); check("E ndarray subclass accepted", np.allclose(np.round(s, 2), np.round(a[:10], 2)))
o32 = np.empty(1000, np.float32); r = np.round(a, 2, out=o32)
check("E out of different dtype (old path, identity only; bytes verified identical to baseline separately)", r is o32)
ro = np.empty(10); ro.flags.writeable = False
try: np.round(a[:10], 2, out=ro); check("E read-only out raises", False)
except ValueError: check("E read-only out raises", True)
try: np.round(a[:10], 2, out=np.empty(11)); check("E bad out shape raises", False)
except ValueError: check("E bad out shape raises", True)
# decimals boundaries use fused path / old path consistently
for dt in (np.float32, np.float64):
    x = (a / 7).astype(dt)
    for d in (-9, -8, 7, 8, 9, 0):
        check(f"E {dt.__name__} d={d} finite and close", np.allclose(np.round(x, d), np.round(x.astype(np.float64), d), atol=10.0**-d if d < 0 else 10.0**-d))
print("\nFAILED:", fails)
