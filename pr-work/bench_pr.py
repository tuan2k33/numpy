"""Benchmarks for gh-32595: buffered broadcasting over a short trailing axis.

Run once per build, then compare (use the numpy dev environment's python):
    source /mnt/c/Dev/projects/numpy/.venv/bin/activate
    NP=/mnt/c/Dev/projects/numpy/build-install/usr/lib/python3/dist-packages
    PYTHONPATH=$NP python -P bench_pr.py run --json pr.json
    PYTHONPATH=<main build dist-packages> python -P bench_pr.py run --json main.json
    python bench_pr.py compare main.json pr.json      # markdown table

Profile the hot path:
    PYTHONPATH=$NP perf record --call-graph dwarf -o perf.data -- \\
        python -P bench_pr.py profile
    perf report -i perf.data --stdio -g none --sort=overhead,symbol | head -40

The same workload as an asv benchmark, to paste after ``Broadcast`` in
benchmarks/benchmarks/bench_ufunc.py and run with
``spin bench --compare main -t bench_ufunc.BroadcastTrailing``:

    class BroadcastTrailing(Benchmark):
        params = [1, 3, 10, 100]
        param_names = ['trailing']

        def setup(self, trailing):
            self.d = np.ones((3000000 // trailing, trailing), dtype=np.float64)
            self.e = np.ones((trailing,), dtype=np.float64)

        def time_broadcast(self, trailing):
            self.d - self.e
"""
import argparse
import json
import sys
import timeit

TRAILING = (1, 3, 10, 30, 100, 300, 1000)
CASES = (
    # (label, dtype, total number of elements)
    ("float64, 24 MB", "float64", 3_000_000),
    ("float32, 1.2 MB", "float32", 300_000),
)


def best_time(func, number, repeat):
    return min(timeit.repeat(func, number=number, repeat=repeat)) / number


def inplace_add_time(np, rows, trailing, dtype, repeat):
    a = np.ones((rows, trailing), dtype=dtype)  # C-contiguous
    v = np.zeros(trailing, dtype=dtype)
    number = max(1, int(1_000_000 // a.size))
    for _ in range(5):
        a += v
    return best_time(lambda: a.__iadd__(v), number, repeat)


def run(args):
    import numpy as np

    results = {"numpy": np.__version__,
               "python": sys.version.split()[0], "times": {}}
    print(f"numpy {results['numpy']}, python {results['python']}")
    for label, dtype, total in CASES:
        print(f"\na += v, a C-contiguous (rows, trailing), {label}")
        print(f"{'trailing':>9} {'time':>12} {'ns/element':>11}")
        for trailing in TRAILING:
            rows = total // trailing
            t = inplace_add_time(np, rows, trailing, dtype, args.repeat)
            results["times"][f"{dtype} {total} trailing={trailing}"] = t
            print(f"{trailing:>9} {t * 1e3:>9.3f} ms "
                  f"{t * 1e9 / (rows * trailing):>11.3f}")

    t = inplace_add_time(np, 2730, 3, "float32", args.repeat)
    results["times"]["float32 (2730, 3) one buffer"] = t
    print(f"\nfloat32 (2730, 3), exactly one default buffer: {t * 1e6:.2f} us")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nsaved to {args.json}")


def compare(args):
    with open(args.base) as f:
        base = json.load(f)
    with open(args.new) as f:
        new = json.load(f)
    print(f"| case | {base['numpy']} | {new['numpy']} | ratio |")
    print("| --- | --- | --- | --- |")
    for key, t0 in base["times"].items():
        t1 = new["times"].get(key)
        if t1 is not None:
            print(f"| {key} | {t0 * 1e3:.3f} ms | {t1 * 1e3:.3f} ms "
                  f"| {t1 / t0:.2f} |")


def profile(args):
    import numpy as np

    a = np.ones((100_000, 3), dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    for _ in range(args.loops):
        a += v


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run", help="time the workloads with the importable numpy")
    p.add_argument("--json", help="also save the results to this file")
    p.add_argument("--repeat", type=int, default=7)
    p = sub.add_parser("compare", help="markdown table comparing two runs")
    p.add_argument("base")
    p.add_argument("new")
    p = sub.add_parser("profile", help="tight loop to record with perf")
    p.add_argument("--loops", type=int, default=20_000)
    args = parser.parse_args()
    {"run": run, "compare": compare, "profile": profile}[args.mode](args)


if __name__ == "__main__":
    main()
