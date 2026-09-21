# Benchmarks — informal cross-venv timing log

Times are **milliseconds**, `min / median` over 20–30 repeats; latest
measurement per op only. Sizes: `N=2_000_000` (phases 2–3), `500_000`
(phase 4), `1_000_000` (phases 6–8), 500x500 / 300x300 matrices and a
100k-point signal (phase 9). Columns: upstream (release wheel in a separate
venv), fork without a mask, fork with a mask, `fork nomask − upstream`, and the
mask overhead `fork mask − fork nomask`.

**How to read it.** The fork build is debugoptimized and the upstream wheel is a
release build, so cross-venv deltas are noise unless they shift in one
direction across reruns (`~0` = within ±15%). The mask-overhead column compares
one build with itself, so it is the real signal; `add`, `less`, `sin` (phase 3)
and `reduce`/`accumulate`/`reduceat`/`outer`/`at`/`sum`/`std` (phase 4) were
rerun 3x and stayed positive (masked slower) every time.

| phase | op | upstream | fork nomask | fork mask | fork nomask − upstream | mask overhead (mask − nomask) |
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
| 6 | `a[idx]` (200k fancy) | 1.16 / 1.46 | 0.85 / 1.73 | 2.03 / 3.07 | ~0 (noisy, flips sign) | +0.67–1.18 |
| 6 | `a[sel]` (bool, ~500k picked) | 4.90 / 5.05 | 5.02 / 5.56 | 9.89 / 10.29 | +0.12 | +4.87–5.04 |
| 6 | `a[idx] = v` | 0.65 / 0.78 | 0.52 / 0.68 | 0.91 / 1.36 | ~0 (noisy, flips sign) | +0.39–0.52 |
| 6 | `a[sel] = v` | 4.65 / 4.80 | 4.88 / 5.62 | 9.76 / 10.58 | +0.23 | +4.88–5.17 |
| 6 | `a.flat[idx]` | 4.91 / 6.45 | 3.05 / 5.48 | 7.18 / 8.82 | ~0 (noisy, flips sign) | +3.91–4.13 |
| 7 | `a.take(idx)` (200k) | 0.97 / 1.45 | 0.27 / 0.81 | 1.41 / 2.25 | ~0 (noisy, flips sign) | +0.88–1.14 |
| 7 | `a.repeat(2)` | 1.97 / 2.26 | 1.71 / 2.36 | 3.35 / 4.23 | -0.26 | +1.64–1.87 |
| 7 | `np.concatenate([a, b])` | 1.70 / 2.33 | 1.74 / 2.28 | 2.58 / 2.87 | +0.04 | +0.84–0.85 |
| 7 | `np.where(cond, a, b)` | 4.06 / 4.29 | 4.09 / 4.42 | 8.13 / 8.50 | +0.03 | +4.04–4.16 |
| 7 | `np.choose(sel, [a, b])` | 7.09 / 7.94 | 7.13 / 8.35 | 14.56 / 15.66 | +0.04 | +6.98–7.43 |
| 7 | `a.put(idx, v)` | 0.86 / 1.20 | 1.31 / 2.55 | 1.69 / 2.39 | ~0 (noisy, flips sign) | +0.38–0.70 |
| 7 | `np.copyto(a, b, where=cond)` | 5.04 / 5.48 | 4.77 / 4.97 | 9.80 / 10.28 | -0.27 | +4.99–5.03 |
| 8 | `a.astype(float64)` | 0.29 / 0.50 | 0.34 / 0.65 | 0.74 / 1.09 | ~0 (noisy, flips sign) | +0.34–0.39 |
| 8 | `np.asarray(a, dtype=float64)` | 0.30 / 0.49 | 0.37 / 0.54 | 0.81 / 1.13 | ~0 (noisy, flips sign) | +0.25–0.44 |
| 8 | `a.view(uint32)` | ~0 | ~0 | 0.04 / 0.04 | ~0 | +0.04 |
| 8 | `np.array(a, ndmin=3)` | 0.11 / 0.13 | 0.10 / 0.13 | 0.24 / 0.51 | ~0 | +0.12–0.14 |
| 8 | `np.array([a, b])` | 0.74 / 1.21 | 0.60 / 0.96 | 1.22 / 1.58 | ~0 (noisy, flips sign) | +0.52–0.62 |
| 9 | `a @ b` (500x500) | 1.28 / 9.89 (BLAS) | 115.36 / 117.84 | 116.09 / 121.12 | n/a (fork build has no optimized BLAS) | +0.73–0.90 |
| 9 | `np.dot(a, b)` (500x500) | 1.53 / 5.48 (BLAS) | 116.32 / 118.26 | 116.05 / 119.97 | n/a (no BLAS) | ~0 (noisy, flips sign) |
| 9 | `np.einsum('ij,jk->ik')` (500x500) | 26.11 / 27.15 | 26.38 / 30.54 | 59.71 / 62.24 | ~0 | +31–33 |
| 9 | `np.linalg.inv` (300x300) | 0.94 / 1.70 (BLAS) | 14.63 / 15.50 | 14.30 / 15.01 | n/a (no BLAS) | ~0 (noisy, flips sign) |
| 9 | `np.correlate(v, k, 'same')` (100k, 64) | 2.22 / 2.33 | 4.06 / 4.26 | 12.39 / 12.67 | +1.8 (debug build) | +8.3 |

Phase 9 note: the fork's local build is not linked against an optimized BLAS
(the upstream wheel is), so matmul/dot/inv cross-venv deltas are meaningless;
only the same-build masked-vs-unmasked column counts. `einsum` and
`correlate` derive their mask by running the same operation on the masks
(twice for two masked operands), hence ~2.2x and ~3x; the matrix products and
`numpy.linalg` only pay for a couple of `any()` reductions.

Phase 10 (repr, `filled`, pickle, `isin`) was not benchmarked: it adds no cost to
the unmasked hot paths (a NULL check in `__reduce_ex__`, and Python-side
branches on `mask is None` in `repr` and `isin`).
