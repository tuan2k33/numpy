# Benchmarks — informal cross-venv timing log

All timing columns are **milliseconds (ms)**, formatted `min / median`
over 20-30 repeats (`N=2_000_000` for phase 2-3 ops, `N=500_000` for
phase 4 ops). The last two columns are `fork nomask − upstream` and
`fork nomask − fork mask`; both use the same ms unit. Latest measurement per
op only.

In each timing cell, `/` separates the minimum and median of the measured
runs. The table keeps `masked` next to the fork's no-mask baseline for quick
comparison.

Cross-venv (`fork nomask − upstream`) deltas are noisy (debug vs. release
build confound) — a single run's delta isn't signal, only a consistent
one-directional shift across reruns is. The `fork nomask − fork mask`
column is same-build, so it isn't subject to that confound; ranges shown
were reran 3x for `add`/`less`/`sin`
(phase 3) and `reduce`/`accumulate`/`reduceat`/`outer`/`at`/`sum`/`std`
(phase 4) and stayed positive (masked slower) every single run.

| phase | op | upstream | fork nomask | fork mask | fork nomask − upstream | fork nomask − fork mask |
|---|---|---|---|---|---|---|
| 2 | copy | 1.24 / 1.81 | 1.49 / 1.73 | — | -0.08 | — |
| 2 | view / `a[:]` / `a[2:N:3]` / reshape / ravel / squeeze | ~0 | ~0 | — | ~0 | — |
| 2 | sort | 13.34 / 14.89 | 13.88 / 16.76 | — | +1.87 | — |
| 2 | partition | 4.66 / 5.38 | 4.68 / 5.32 | — | -0.06 | — |
| 3 | add (operator) | 2.56 / 2.79 | 2.66 / 2.92 | 3.25 / 3.57 | +0.13 | +0.43–0.65 |
| 3 | np.add | 2.48 / 2.92 | 2.51 / 2.67 | — | -0.25 | — |
| 3 | multiply | 2.62 / 2.98 | 2.54 / 2.77 | — | -0.21 | — |
| 3 | less (comparison) | 1.17 / 1.51 | 1.27 / 1.39 | 1.58 / 2.01 | -0.12 | +0.47–1.13 |
| 3 | equal (comparison) | 1.14 / 1.48 | 1.12 / 1.69 | — | +0.21 | — |
| 3 | sin | 19.16 / 21.70 | 19.09 / 20.24 | 19.35 / 21.91 | -1.46 | ~0 (noisy, flips sign) |
| 3 | sqrt | 2.80 / 3.31 | 2.74 / 2.88 | — | -0.43 | — |
| 3 | divmod (multi-out) | 8.82 / 9.64 | 10.07 / 10.56 | — | +0.92 | — |
| 3 | in-place `+=` | 3.24 / 3.55 | 3.29 / 3.70 | — | +0.15 | — |
| 3 | `np.add(out=)` | 2.30 / 2.63 | 2.67 / 3.06 | — | +0.43 | — |
| 4 | `add.reduce` | 0.10 / 0.14 | 0.10 / 0.11 | 0.09 / 0.10 | -0.03 | ~0 (-0.03–+0.004) |
| 4 | `add.reduce(axis=1)` | 0.12 / 0.16 | 0.12 / 0.13 | — | -0.03 | — |
| 4 | `add.accumulate` | 1.88 / 1.98 | 2.02 / 2.09 | 2.93 / 3.12 | +0.11 | +1.04–1.32 |
| 4 | `add.reduceat` | 0.10 / 0.10 | 0.11 / 0.12 | 0.11 / 0.13 | +0.02 | +0.03–0.11 |
| 4 | `add.outer` | 4.46 / 4.91 | 4.68 / 5.70 | 5.37 / 6.05 | +0.79 | +0.39–1.78 |
| 4 | `add.at` (50k scattered) | 0.30 / 0.43 | 0.50 / 0.89 | 1.65 / 2.72 | +0.46 | +0.95–1.42 |
| 4 | `.sum()` | 0.09 / 0.12 | 0.11 / 0.12 | 0.13 / 0.16 | ~0 | ~0 (0.001–0.009) |
| 4 | `.mean()` | 0.10 / 0.11 | 0.11 / 0.11 | — | ~0 | — |
| 4 | `.std()` | 0.42 / 0.66 | 0.62 / 1.05 | 1.88 / 3.14 | +0.39 | +0.54–1.24 |
| 4 | `.var()` | 0.46 / 0.64 | 0.60 / 0.96 | — | +0.32 | — |
| 4 | `.max()` | 0.06 / 0.06 | 0.10 / 0.13 | — | +0.07 | — |
| 4 | `.argmax()` | 0.07 / 0.08 | 0.11 / 0.17 | — | +0.09 | — |
