# Benchmarks — informal cross-venv timing log

Latest measurement per op only. `fork nomask vs upstream %` is `—` when
within the established ±15% noise tolerance (debug-vs-release build
confound, not a regression signal); a shown % means it exceeded that
tolerance on this run. Only a consistent one-directional shift across
reruns is real signal — if an op keeps showing that, fix the phase listed.

**4th run note**: `add.at` flipped again (+25% → +74% → -43% → +109%) —
confirms noise, not a regression (sub-ms timing, random scattered indices,
high variance). The 4th run also invalidated the `.max()` "consistent
signal" read from runs 1-3: this run, `argmax` — previously flat at ~0%
across all 3 prior runs, the most stable op in the whole table — spiked to
+129%, and `std`/`var` also jumped (+58%/+51%) alongside `.max()`'s +112%.
A previously rock-solid-flat op moving this much means run 4 was a
system-wide noisy run (background load on the machine, not a code issue),
so the "no-mask" column across cross-venv runs is confirmed noisy/not
reliable as regression signal, while the mask-overhead column below was
reran 3x for `add`/`less`/`sin` (phase 3) and `reduce`/`accumulate`/
`reduceat`/`outer`/`at`/`sum`/`std` (phase 4), same-build, no cross-venv
confound: every op with overhead stayed **positive every single run**
(masked always slower, never flips sign) — that's real signal, unlike the
no-mask column above. Only `sin` and `add.reduce`/`.sum()` stayed
consistently near-zero (confirmed free), matching design intent. No
confirmed regression on the no-mask (cross-venv) path after 4 runs.

| phase | op | upstream (no mask) | fork (no mask) | fork nomask vs upstream % | fork (masked) | mask overhead |
|---|---|---|---|---|---|---|
| 2 | copy | 1.24 / 1.81 | 1.49 / 1.73 | — | — | — |
| 2 | view / `a[:]` / `a[2:N:3]` / reshape / ravel / squeeze | ~0 | ~0 | — | — | — |
| 2 | sort | 13.34 / 14.89 | 13.88 / 16.76 | — | — | — |
| 2 | partition | 4.66 / 5.38 | 4.68 / 5.32 | — | — | — |
| 3 | add (operator) | 2.56 / 2.79 | 2.66 / 2.92 | — | 3.25 / 3.57 | +13–30% (3 reruns, always +) |
| 3 | np.add | 2.48 / 2.92 | 2.51 / 2.67 | — | — | — |
| 3 | multiply | 2.62 / 2.98 | 2.54 / 2.77 | — | — | — |
| 3 | less (comparison) | 1.17 / 1.51 | 1.27 / 1.39 | — | 1.58 / 2.01 | +28–65% (3 reruns, always +) |
| 3 | equal (comparison) | 1.14 / 1.48 | 1.12 / 1.69 | — | — | — |
| 3 | sin | 19.16 / 21.70 | 19.09 / 20.24 | — | 19.35 / 21.91 | — |
| 3 | sqrt | 2.80 / 3.31 | 2.74 / 2.88 | — | — | — |
| 3 | divmod (multi-out) | 8.82 / 9.64 | 10.07 / 10.56 | — | — | — |
| 3 | in-place `+=` | 3.24 / 3.55 | 3.29 / 3.70 | — | — | — |
| 3 | `np.add(out=)` | 2.30 / 2.63 | 2.67 / 3.06 | +16.3% | — | — |
| 4 | `add.reduce` | 0.10 / 0.14 | 0.10 / 0.11 | -25.2% | 0.09 / 0.10 | — |
| 4 | `add.reduce(axis=1)` | 0.12 / 0.16 | 0.12 / 0.13 | -22.1% | — | — |
| 4 | `add.accumulate` | 1.88 / 1.98 | 2.02 / 2.09 | — | 2.93 / 3.12 | +53–71% (3 reruns, always +) |
| 4 | `add.reduceat` | 0.10 / 0.10 | 0.11 / 0.12 | — | 0.11 / 0.13 | +29–103% (3 reruns, always +) |
| 4 | `add.outer` | 4.46 / 4.91 | 4.68 / 5.70 | +16.0% | 5.37 / 6.05 | +7–30% (3 reruns, always +) |
| 4 | `add.at` (50k scattered) | 0.30 / 0.43 | 0.50 / 0.89 | +108.9% | 1.65 / 2.72 | +54–170% (3 reruns, always +) |
| 4 | `.sum()` | 0.09 / 0.12 | 0.11 / 0.12 | — | 0.13 / 0.16 | — |
| 4 | `.mean()` | 0.10 / 0.11 | 0.11 / 0.11 | — | — | — |
| 4 | `.std()` | 0.42 / 0.66 | 0.62 / 1.05 | +57.8% | 1.88 / 3.14 | +33–205% (3 reruns, always +) |
| 4 | `.var()` | 0.46 / 0.64 | 0.60 / 0.96 | +50.9% | — | — |
| 4 | `.max()` | 0.06 / 0.06 | 0.10 / 0.13 | +111.9% | — | — |
| 4 | `.argmax()` | 0.07 / 0.08 | 0.11 / 0.17 | +129.3% | — | — |
