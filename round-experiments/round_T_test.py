import ctypes, time, warnings
import numpy as np
from decimal import Decimal, ROUND_HALF_EVEN
warnings.simplefilter("ignore")
lib = ctypes.CDLL("/mnt/c/Dev/numpy-round-experiments/libround_T.so")
rng = np.random.default_rng(5)

def call(dt, kind, x, d):
    name = ("f32_" if dt is np.float32 else "f64_") + kind
    out = np.empty_like(x)
    c = ctypes.c_void_p
    getattr(lib, name)(c(x.ctypes.data), c(out.ctypes.data), ctypes.c_long(x.size), ctypes.c_int(d))
    return out

def truthB(v, d, dt):
    s = repr(float(v)) if dt is np.float64 else np.format_float_positional(dt(v), unique=True, trim="-")
    return dt(float(Decimal(s).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_EVEN)))

def uint(dt): return np.uint64 if dt is np.float64 else np.uint32
def eqf(a, b, dt): return (a.view(uint(dt)) == b.view(uint(dt)))

def dists(dt, d, N, maxdig):
    sgn = np.where(rng.random(N) < .5, -1, 1)
    # decimal ties: d+1 decimals ending in 5; non-ties: d+2 decimals ending in 1 or 9 (near a tie); plus raw random floats
    scale = 10.0 ** (d + 1) if d >= 0 else None
    if d >= 0:
        n = rng.integers(0, 10 ** maxdig, N)
        tie = sgn * (10 * n + 5) / scale
        near = sgn * (100 * n + 51 - 2 * rng.integers(0, 2, N) * 1) / 10.0 ** (d + 2)   # ...51 or ...49 -> not a tie
    else:
        k = -d
        n = rng.integers(0, 10 ** max(maxdig - k, 2), N)
        tie = sgn * (10 * n + 5) * 10.0 ** (k - 1)       # 25, 35, ... at d=-1 etc.
        near = tie + sgn * 10.0 ** (k - 2) if k >= 2 else tie + sgn * 0.01
    uni = rng.random(N) * 2000 - 1000
    return {"decimal ties": tie.astype(dt), "near ties (not ties)": near.astype(dt), "uniform floats": uni.astype(dt)}

def correctness(dt, ds, N, maxdig):
    print(f"\n=== {dt.__name__}: % equal to the 'written decimal' truth (B)   [numpy | python round | T-algorithm]")
    for d in ds:
        row = []
        for name, x in dists(dt, d, N, maxdig).items():
            x = x[np.isfinite(x)]
            tb = np.array([truthB(v, d, dt) for v in x.tolist()], dtype=dt)
            npr = np.round(x, d)
            py = np.array([round(float(v), d) for v in x.tolist()], dtype=dt)
            tt = call(dt, "T1", x, d)
            row.append(f"{name:21s} {100*eqf(npr,tb,dt).mean():6.2f} | {100*eqf(py,tb,dt).mean():6.2f} | {100*eqf(tt,tb,dt).mean():6.2f}")
        print(f"d={d:3d}  " + "   ".join(row))

correctness(np.float64, [1, 2, 3, 4, 6, 8, -1, -2, -3], 20000, 6)
correctness(np.float32, [1, 2, 3, 4, -1, -2], 20000, 4)

print("\n=== speed, ns per element (best of 7), same C compiler flags, AVX2, one core")
def bench(dt, kind, x, d):
    best = 1e9
    for _ in range(7):
        t = time.perf_counter(); call(dt, kind, x, d); best = min(best, time.perf_counter() - t)
    return best / x.size * 1e9
for dt in (np.float32, np.float64):
    for n in (1000, 100_000, 10_000_000):
        x = (rng.random(n) * 2000 - 1000).astype(dt)
        for d in (2, -2):
            kinds = ["plain", "T1"] + (["T2"] if d >= 0 else [])
            r = {k: bench(dt, k, x, d) for k in kinds}
            print(f"{dt.__name__:8s} n={n:>9,d} d={d:2d}: plain {r['plain']:5.2f} | T1 (one midpoint) {r['T1']:5.2f} ({r['T1']/r['plain']:.2f}x)" + (f" | T2 (two midpoints) {r['T2']:5.2f} ({r['T2']/r['plain']:.2f}x)" if 'T2' in r else ""))
