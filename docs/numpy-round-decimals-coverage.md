# `np.round(a, b)` coverage of `b` for float16 / float32 / float64

Measured 2026-10-08 on `main` and on branch `fix/round-float16` (PR A, numpy 2.6.0.dev0, x86-64 Linux, AVX2).
Reference = Python `round(float(x), b)` (correct rounding of the exact binary value, half-even), cast to the dtype.
float16: every finite value (63488). float32 / float64: 100 000 random samples per cell (uniform [-1000, 1000] and log-uniform 1e-8..1e8) unless stated. float32 / float64 code is identical in `main` and PR A.

## 1. The three formats and the scale factor `p = 10**|b|`

| | float16 | float32 | float64 |
|---|---|---|---|
| significant bits | 11 | 24 | 53 |
| significant decimal digits | ~3.3 | ~7.2 | ~15.9 |
| largest finite value | 65504 | 3.4e38 | 1.8e308 |
| smallest positive value | 6e-8 | 1.4e-45 | 4.9e-324 |
| `np.round` computes in | `main`: float16, PR A: float32 | float32 | float64 |
| `10**|b|` exact in that type | `main`: `|b| <= 4` (inf from 5); PR A: `|b| <= 10` | `|b| <= 10` (5^10 < 2^24) | `|b| <= 22` (5^22 < 2^53) |
| `10**|b|` finite in that type | `main`: `|b| <= 4`; PR A: `|b| <= 38` | `|b| <= 38` | `|b| <= 308` |
| `x * 10**b` exact in a wider type | `|b| <= 5` (in float32) | `|b| <= 12` (in double) | never (needs a two-product) |

## 2. How accurate `np.round` is today (measured)

| | float16 `main` | float16 PR A | float32 (`main` = PR A) | float64 (`main` = PR A) |
|---|---|---|---|---|
| exact (0 wrong) | only `b = 0` | **every `b` in [-38, 33]** (all 63488 values) | uniform samples: `b` in [-38, -3]; `b = -2, -1`: 0.03% / 0.28% wrong (log-uniform) | uniform samples: `b` in [-308, -1] and [1, 7] |
| small errors | | | `b = 1`: 0.02%, `b = 2`: 0.24%, `b = 3`: 2% (uniform); `b >= 5`: 6-18% | `b = 8..11`: 0-3.7%; `b = 12`: 3.9%; `b >= 15`: 4-17%, mostly 1 ulp |
| overflow to `inf` | | `b = 34..38`: 1968 .. 29282 of 63488 | `b = 38`: 99.7% | `b = 308`: 99.8% |
| `NaN` for every element | `|b| >= 5` | `|b| >= 39` | `|b| >= 39` | `b >= 310` |
| near-tie values (nearest float of `(k+0.5)/10**b`) | | exact | 40-56% differ from Python for `b > 0` | 40-57% differ from Python for `b > 0`; 0% for `b` in [-12, -1] |

"Wrong" = bit pattern differs from the reference. Near ties are inputs where the decimal is half-way, e.g. `np.round(2.675, 2)` is 2.68 (numpy) vs 2.67 (Python).

## 3. Target of a single-path exact algorithm (proposal, NOT implemented or validated)

| | float16 | float32 | float64 |
|---|---|---|---|
| compute in | float32 | double, final division in float32 | float64 with a two-product (fma or Dekker) |
| exact by construction for | `|b| <= 10` | `|b| <= 10` | `|b| <= 22` |
| beyond that | measured exact up to `b` = 33 today | slightly wrong is acceptable | slightly wrong is acceptable |
| out-of-range guard | `|x*p| >= 2^24` -> return `x`; `p` = inf -> `x` or `+-0` | `|x*p| >= 2^24` -> return `x` | `|x*p| >= 2^53` -> return `x` |

The "exact by construction" limits come from bit counting (section 1), not from an exhaustive run.
