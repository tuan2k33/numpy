"""Benchmark np.round on whatever numpy is on PYTHONPATH. CPU only, single process.

usage: PYTHONPATH=<site dir> taskset -c 4 python bench_np_round.py out.json [quick]
Reports min and median seconds per call over repeats. Compare two runs with compare_np_round.py.
"""
import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
SIZES = [24, 1000, 10_000, 100_000, 1_000_000, 8_000_000]
if quick:
    SIZES = [24, 10_000, 1_000_000]
DTYPES = [np.float16, np.float32, np.float64]
DECIMALS = [2, -2, 6]
KINDS = ["random", "ties"]
REPEATS = 15


def make(dt, n, d, kind):
    rng = np.random.default_rng(12345)
    if kind == "random":
        x = rng.random(n) * 2000.0 - 1000.0
    else:  # every element is a decimal half-way point at this d: (k + 0.5) / 10**d
        k = rng.integers(-50_000, 50_000, n).astype(np.float64)
        x = (k + 0.5) / 10.0**d if d >= 0 else (k + 0.5) * 10.0 ** (-d)
    if dt is np.float16:
        x = np.clip(x, -60000, 60000)
    return x.astype(dt)


def timeit(f, target=0.02):
    f()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        dt = time.perf_counter() - t0
        if dt >= target or n >= 1_000_000:
            break
        n *= 2
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        ts.append((time.perf_counter() - t0) / n)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


res = {"numpy": np.__version__, "path": np.__file__, "results": {}}
for dt in DTYPES:
    for d in DECIMALS:
        for kind in KINDS:
            for n in SIZES:
                if dt is np.float16 and n > 1_000_000:
                    continue
                a = make(dt, n, d, kind)
                mn, md = timeit(lambda: np.round(a, d))
                key = f"{dt.__name__}|d={d}|{kind}|n={n}"
                res["results"][key] = {"min": mn, "median": md}
                print(f"{key:44s} min {mn*1e6:11.2f} us   median {md*1e6:11.2f} us", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
