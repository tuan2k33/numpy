# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Scratch validation and benchmark scripts for the fused `np.round` work in the numpy fork at `/mnt/c/Dev/projects/numpy` (branch `perf/fused-round` = merge 3 ufunc passes into one SIMD pass + float16 without per-pass float16 round trips; branch `perf/fused-round-exact` = parked variant with exact half-way handling, no FMA). This directory is **not a git repository**; the design, decisions and measured results live in `/mnt/c/Dev/projects/numpy/ROUND_FUSED_PLAN.md` (sections 9 and 10). Everything here runs on CPU only; no GPU.

The numpy fork's own rules apply: remote `fork` is the push target, `origin` (upstream) is never pushed, no AI `Co-Authored-By` trailers on the user's commits/PRs, and numpy's AI policy forbids AI-written issue/PR text, so comments on GitHub are written by the user.

## Where the builds are

- `/home/tuannq/work/np-base`: rsync copy of the repo on the native ext4 disk (building from `/mnt/c` is far slower). After editing sources in the repo, copy the changed files into the same relative path here, then:
  ```bash
  cd /home/tuannq/work/np-base
  export PATH=/mnt/c/Dev/projects/numpy/.venv/bin:$PATH
  spin build -C build-rel -j 16 -- -Dbuildtype=release     # release is required; spin's default here was debugoptimized
  ```
  The install tree is `build-rel-install/usr/lib/python3/dist-packages` (numpy 2.6.0.dev0, new code).
- `/home/tuannq/work/baseline-rel-install`: saved release build of unmodified `main` for A/B comparison. Do not overwrite.
- `/home/tuannq/venvs/numpy-upstream/bin/python`: Python 3.14 with stock numpy 2.5.3. Run scripts with `PYTHONPATH=<install>/usr/lib/python3/dist-packages` to pick a build.
- The repo's own `build/` and `build-install/` hold the user's `refactor/ndarray-mask` build: never rebuild into them.

## Running things

Always run Python **from a directory outside the numpy repo** (e.g. `cd /tmp`): inside the repo root the in-tree `numpy/` shadows the installed one and fails with `cannot import name 'version'`.

```bash
cd /tmp
P=/home/tuannq/work/np-base/build-rel-install/usr/lib/python3/dist-packages
PYTHONPATH=$P /home/tuannq/venvs/numpy-upstream/bin/python /mnt/c/Dev/numpy-round-experiments/t_half_v1.py
```

| Script | What it checks |
|---|---|
| `t_identical.py dump out.npz` | Dumps `np.round` results (float32/64, 30 decimals incl. ±2^31, specials, layouts) to compare bit-for-bit between two builds; the comparison itself is a small inline script (load both npz, compare uint views, ignore NaN payload). v1 must be bit-identical to baseline. |
| `t_half_v1.py` | float16: 36 decimals × all 65536 patterns equal "float32 math, one cast"; prints the two #13699 examples. |
| `t_correct.py` | Correctness of the **exact** variant (`perf/fused-round-exact`): float16 exhaustive vs rational reference, float64 vs Python `round()` on tie-heavy samples, layouts/out/subclass. It is expected to fail its exactness checks on the v1 build (v1 keeps the old tie behavior); only its layout/`out=` part applies to v1. |
| `bench_np_round.py out.json [quick]` + `compare_np_round.py base.json new.json` | Real `np.round` timings (3 dtypes × decimals {2,-2,6} × random/tie data × sizes 24…8e6). `base_rel.json` is the baseline, `new_v1.json` the v1 result. |
| `val*.py` | Python-only validation of the exact algorithm (fma and no-fma/Dekker variants) against `round()`; throwaway, slow, not an implementation. |
| `exh32.c`, `exh32s.c`, `exh32n.c`, `bench_round.c` | C harnesses, see below. |

C harnesses (`gcc` is available; CPU is 16 threads, has `avx2`/`fma`/`f16c`, no AVX-512, so the AVX-512 target is compiled but never executed here):

```bash
gcc -O2 -fopenmp -mfma -mavx2 -ffp-contract=off exh32.c -o exh32 -lm   # ./exh32 [d [lo hi]]
gcc -O2 -mavx2 -mfma -ffp-contract=off bench_round.c -o bench_round -lm
```
`exh32*` check every float32 bit pattern for decimals -8…7 against an exact 64/128-bit integer reference (about 5 minutes per variant); `exh32` = double + fma tie fix, `exh32s` = plain double, `exh32n` = native float with `fmaf`. `bench_round` is a single-thread AVX2 proxy for the kernel (does not model ufunc dispatch overhead).

## Gotchas

- Benchmark noise is ±10–20% run to run; judge speed with alternating baseline/new runs pinned with `taskset -c 6` and min-vs-min, not a single pass (a lone 1.18× once turned out to be noise).
- Full `pytest numpy -n 12` (and even `-n 6`) pushed WSL (12 GB RAM) to its limit and killed the session twice. Use fewer workers, `--ignore=numpy/typing --ignore=numpy/f2py`, and a memory watchdog; the culprit test group is not identified yet. Run pytest directly from the install dir (`build-install`-style), not `spin test`, whose staleness check is very slow on `/mnt/c`.
- `calculation.c` must keep `#define _UMATHMODULE` at the top, otherwise `PyUFunc_GiveFloatingpointErrors` goes through an uninitialised API table and segfaults as soon as an FP flag is raised.
- The `-ffp-contract=off` meson flag is only needed by the exact (Dekker) variant, not by v1.
