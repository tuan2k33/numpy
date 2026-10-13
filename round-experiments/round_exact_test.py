import ctypes, time, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
lib = ctypes.CDLL("/mnt/c/Dev/numpy-round-experiments/libround_exact.so")
dp = ctypes.POINTER(ctypes.c_double); lp = ctypes.POINTER(ctypes.c_long)
def ptr(a): return a.ctypes.data_as(dp)
def exact(x, d):
    out = np.empty_like(x); nb = ctypes.c_long(); lib.run_exact(ptr(x), ptr(out), x.size, d, ctypes.byref(nb)); return out, nb.value
def fast(x, d):
    out = np.empty_like(x); nf = ctypes.c_long(); nb = ctypes.c_long()
    lib.run_fast(ptr(x), ptr(out), x.size, d, ctypes.byref(nf), ctypes.byref(nb)); return out, nf.value, nb.value
def plain(x, d):
    out = np.empty_like(x); lib.run_plain(ptr(x), ptr(out), x.size, d); return out
def pyref(x, d):
    return np.array([round(v, d) for v in x.tolist()])
rng = np.random.default_rng(11)
N = 40000
def samples(d):
    uni = rng.random(N) * 2000 - 1000
    mag = np.where(rng.random(N) < .5, -1, 1) * 10.0 ** rng.uniform(-8, 8, N)
    k = rng.integers(-3000, 3000, N).astype(np.float64)
    p = 10.0 ** abs(d)
    tie = (k + 0.5) / p if d >= 0 else (k + 0.5) * p            # nearest double to a decimal tie
    near = np.nextafter(tie, np.inf * np.where(rng.random(N) < .5, -1, 1))   # one ulp off a tie
    ints = rng.integers(-10**6, 10**6, N).astype(np.float64) / p if d >= 0 else rng.integers(-10**6, 10**6, N).astype(np.float64) * p
    return {"uniform": uni, "log-uniform": mag, "near-tie": tie, "tie+-1ulp": near, "decimal-looking": ints}
bits = lambda a: a.view(np.uint64)
tot = {"exact": 0, "fast": 0, "plain(numpy formula)": 0}; cells = 0; fb = 0; elems = 0; oks = 0
worst = {}
for d in list(range(-22, 0)) + list(range(1, 23)):
    for name, x in samples(d).items():
        x = x[np.isfinite(x)]
        ref = pyref(x, d)
        e, nb = exact(x, d); f, nf, nb2 = fast(x, d); pl = plain(x, d)
        be = int((bits(e) != bits(ref)).sum()); bf = int((bits(f) != bits(ref)).sum()); bp = int((bits(pl) != bits(ref)).sum())
        tot["exact"] += be; tot["fast"] += bf; tot["plain(numpy formula)"] += bp; fb += nf; elems += x.size; oks += nb
        worst.setdefault(name, [0, 0, 0]); worst[name][0] += be; worst[name][1] += bf; worst[name][2] += bp
print(f"checked {elems} values over |d| in 1..22 (both signs), 5 distributions")
print("mismatches vs Python round():", tot)
print("by distribution (exact, fast, plain):", {k: tuple(v) for k, v in worst.items()})
print(f"fast path sent to the integer path: {fb} values ({100*fb/elems:.4f}%), outside integer-path domain: {oks}")
# timing, scalar C loops, 2M uniform values
x = rng.random(2_000_000) * 2000 - 1000
def tm(f, rep=5):
    best = 1e9
    for _ in range(rep):
        t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
    return best / x.size * 1e9
for d in (2, -2, 6):
    print(f"d={d:3d}: plain scalar {tm(lambda: plain(x,d)):5.2f} ns/elem | fast+fallback {tm(lambda: fast(x,d)):5.2f} | always-exact {tm(lambda: exact(x,d)):6.2f} | np.round (this numpy) {tm(lambda: np.round(x,d)):5.2f}")
