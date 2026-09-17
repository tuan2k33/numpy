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

## Existing test suites to reuse as templates / regression baseline

No dedicated test suite exists yet (feature isn't written). These existing
numpy test files cover the same ground and should be mined per phase — either
as a pattern to copy for the masked case, or as the no-mask regression
baseline that must keep passing unchanged:

| Phase | Template / baseline file | Why |
|---|---|---|
| 2 — ufunc dispatch | `numpy/_core/tests/test_umath.py` (5474 lines), `test_ufunc.py` (3523 lines) | Per-ufunc correctness tests; run unmodified against `mask=None` as the zero-regression check; mine for cases to duplicate with a masked operand |
| 3 — reductions | `numpy/_core/tests/test_umath.py` (reduce/accumulate sections), `numpy/ma/tests/test_core.py` (`MaskedArray.sum`/`mean`/etc semantics — closest prior art for what "reduce over a gap" should mean) | `numpy.ma`'s reduction semantics are the direct precedent to compare against/diverge from deliberately |
| 4 — view propagation | `numpy/_core/tests/test_shape_base.py`, `numpy/ma/tests/test_subclassing.py` (469 lines — how `numpy.ma` keeps `.mask` attached through view ops, and where it historically didn't) | `test_subclassing.py` is effectively a list of past leak bugs to not repeat |
| 5 — indexing | `numpy/_core/tests/test_indexing.py` (1717 lines) | Most thorough existing coverage of basic/advanced/boolean indexing edge cases |
| 6 — sort/search | `numpy/_core/tests/test_item_selection.py` (178 lines) | Covers `take`/`put`/`choose`/`repeat`; small, good starting template |
| 9 — casting | `numpy/_core/tests/test_multiarray.py` (12122 lines, has `astype`/casting sections) | Largest file — grep for `astype`/`can_cast` sections rather than reading whole file |
| 10 — Python surface | `numpy/ma/tests/test_core.py` (6276 lines), `test_old_ma.py` (939 lines), `test_mrecords.py` (513 lines), `test_deprecations.py`, `test_regression.py` | The full `numpy.ma` suite — closest thing to "what a masked-array test suite looks like end to end"; also the best source of regression cases for exactly the leak/surprise bugs this design is meant to avoid |

`numpy/ma/tests/` in particular is worth a full pass before writing any new
test: most of its ~7800 lines encode a real bug or surprising-behavior
decision made over `numpy.ma`'s lifetime — deciding *on purpose* whether the
mask-in-core design repeats or fixes each one is more valuable than writing
tests from scratch.

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
