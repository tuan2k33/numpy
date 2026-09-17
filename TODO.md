# TODO — native mask support in `ndarray` core

Research fork: give `ndarray` a built-in, opt-in `mask` so operators only need
`if (mask)` instead of `numpy.ma`'s per-function Python wrappers. Target
**numpy `main`/dev only** (currently `2.6.0.dev0`); no backport to released
branches. Track upstream and rebase periodically rather than diverging long-term.

## Housekeeping

- [ ] Pull from `origin/main` every day — this is a long-running fork of a
      fast-moving codebase, drift compounds fast if skipped.

## Settled design decisions

- `PyArrayObject` gets one new field: `PyArrayObject *mask;` — `NULL` by
  default. Same self-referential-pointer pattern the struct already uses for
  `base`; no size/recursion issue.
- `mask` is a **plain `ndarray` of dtype `bool`** (byte per element), *not*
  bit-packed in storage. This makes it inherit numpy's existing view/stride
  machinery for free — slicing, transposing, broadcasting the parent array
  slices/transposes/broadcasts the mask the same way, no custom bit-stride
  logic needed.
- Invariant: `mask->mask` must always be `NULL` (no masked masks). Enforce at
  every point a mask gets attached.
- Refcounting on `mask` follows the existing convention for `base`
  (`Py_INCREF`/`Py_DECREF` at the same points).
- **No mask (`mask == NULL`)**: goes through the exact current code path,
  unchanged. Only added cost is one pointer NULL-check per call, outside any
  loop. Must measure as zero regression vs. upstream.
- **Has mask, contiguous, AVX-512 available**: fast path. Load data + byte
  mask contiguously, convert byte-mask → k-register in-register
  (`VPMOVB2M`-class instruction, ~free), then masked load/compute/store via
  AVX-512 predicated ops. Single pass, no separate NA-scan pass.
- **Has mask, non-contiguous or no AVX-512**: fallback through the generic
  strided loop (`NpyIter`), mask read via ordinary strided/gather access at
  byte granularity — never attempt bit-level gather.
- **Dispatch check lives in one shared place**: `umath/ufunc_object.c` +
  `umath/dispatching.c` (NEP 43 DType-based dispatch), not duplicated per
  Python function the way `numpy.ma` does it. This is what makes every ufunc
  get mask support "for free" instead of hand-writing each one.
- Reuse numpy's existing `NPY_CPU_DISPATCH` / universal SIMD infra
  (`numpy/_core/src/common/simd/`) to add the AVX-512 masked variant per loop
  rather than building runtime CPU dispatch from scratch.

## Phases

- [ ] **0 — Setup**
  - [ ] New branch off `main` dedicated to this (don't reuse
        `enh/mean-var-non-legacy-dtype`)
  - [ ] Confirm build config / CI baseline against numpy `main`
- [ ] **1 — Struct & invariants**
  - [ ] Add `mask` field to `PyArrayObject`
  - [ ] Refcount handling (mirror `base`)
  - [ ] Enforce `mask->mask == NULL` invariant
- [ ] **2 — Ufunc dispatch (arithmetic, trig, comparisons)**
  - [ ] `umath/ufunc_object.c` — central `if (has_mask)` branch point
  - [ ] `umath/dispatching.c` — masked loop variant selection (NEP 43)
  - [ ] `umath/loops*.c.src` — masked-AVX512 loop + strided fallback loop,
        starting with binary arithmetic on `float64`/`int64` as proof of
        concept before generalizing
- [ ] **3 — Reductions / accumulate**
  - [ ] `umath/reduction.c`, `umath/ufunc_object.c`
        (`reduce`/`accumulate`/`reduceat`/`outer`/`at`)
  - [ ] `multiarray/calculation.c` (`.sum()`, `.mean()`, `.argmax()`...)
- [ ] **4 — View propagation**
  - [ ] `multiarray/shape.c` (reshape/ravel/squeeze/transpose)
  - [ ] `multiarray/getset.c` (`.T` and friends)
  - [ ] `multiarray/ctors.c` (`broadcast_to`)
- [ ] **5 — Indexing**
  - [ ] `multiarray/mapping.c` — basic indexing (view, propagate mask
        trivially), advanced/fancy + boolean indexing (copy — build new mask
        explicitly), assignment through indexing
- [ ] **6 — Sort / search**
  - [ ] `npysort/*.c.src`, `multiarray/item_selection.c`
        (`sort`/`argsort`/`partition`/`take`/`put`/`choose`/`repeat`)
- [ ] **7 — Combine / split**
  - [ ] `multiarray/multiarraymodule.c` (`concatenate`)
  - [ ] `multiarray/item_selection.c` (`repeat`, `choose`)
- [ ] **8 — Linear algebra**
  - [ ] `umath/matmul.c.src`
  - [ ] `numpy/linalg/umath_linalg.c.src` (likely: refuse/raise on masked
        input rather than trying to propagate through LAPACK calls)
- [ ] **9 — Casting**
  - [ ] `multiarray/convert_datatype.c`, `multiarray/convert.c`
- [ ] **10 — Python-level surface**
  - [ ] `numpy/_core/arrayprint.py` (repr/str show masked cells)
  - [ ] `numpy/lib/_arraysetops_impl.py` (`unique`/`isin` mask-awareness)
  - [ ] `multiarray/methods.c` (`__reduce__`/pickle, `tobytes`/`tofile`)
  - [ ] `multiarray/buffer.c` (buffer protocol — decide: expose data only,
        or refuse when masked)
- [ ] **11 — Benchmarking**
  - [ ] No-mask path: confirm zero measurable regression vs. upstream `main`
  - [ ] Contiguous masked path: confirm near-native speed vs. plain op
        (AVX-512 available)
  - [ ] Non-contiguous masked path: measure, document expected slowdown
- [ ] **12 — Testing**
  - [ ] Mask never silently lost across every op in phases 2–10
        (the `numpy.ma`-style leak bugs this design is meant to avoid)
  - [ ] `mask->mask == NULL` invariant enforced everywhere a mask is attached
  - [ ] ASAN/UBSAN clean on the new field's lifetime (alloc/dealloc/view
        chains)
