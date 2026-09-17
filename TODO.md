# TODO — native mask support in `ndarray` core

Research fork: give `ndarray` a built-in, opt-in `mask` so operators only need
`if (mask)` instead of `numpy.ma`'s per-function Python wrappers. Target
**numpy `main`/dev only** (currently `2.6.0.dev0`); no backport to released
branches. Track upstream and rebase periodically rather than diverging long-term.

## Housekeeping

- [ ] Pull from `origin/main` every day and before every push — this is a long-running fork of a fast-moving codebase, drift compounds fast if skipped.

## Regression baseline: Phase 0 vs. Phase 1

`mask == NULL` must keep every existing code path byte-identical to
upstream. Tracking the actual pass counts here after each phase, not just
inside the phase's checklist, so drift is easy to spot at a glance:

| Phase | `test_multiarray.py` | `test_indexing.py` | Combined total | Match previous row? |
|---|---|---|---|---|
| 0 — baseline (no code changes) | 14810 passed, 17 skipped, 12 deselected | 106 passed | 14916 passed, 17 skipped, 12 deselected | — |
| 1 — `mask` field added | *(run combined, no per-file split)* | *(run combined, no per-file split)* | 14916 passed, 17 skipped, 12 deselected | ✅ identical |

Add one row per phase from here on, run against the same two files at
minimum (more as later phases touch more test files per the mapping table
below).

## Scope: which problem this solves

Two different usecases get called "masked array" and demand different designs:

1. **Genuinely missing data** — the value never existed / is unknown. MISSING
   semantics (R/SQL `NULL`). This is what the `nulldtype` bitpattern project
   (sibling, separate repo) solves — there, "hiding" a value **destroys** the
   old one (not observable afterwards), which is correct for this case: the
   whole point is nobody should be able to recover a value nobody ever knew.
2. **Complete data, selectively hidden** — the value is real and known, just
   excluded from a given view/computation for some reason: license/access
   control (redact a column for one audience, not another), an algorithm that
   needs to vary which elements are visible (dropout, k-fold
   cross-validation, random subsampling), etc.

**This fork targets case 2.** Case 1 is deliberately out of scope for now —
revisit later if it turns out to matter here too. This is exactly why
flag-layout (not bitpattern) is the right shape for this work: nothing about
`T`'s value range is sacrificed, hiding is non-destructive/reversible, and
because `mask` lives on the array *object* (not the data buffer), several
views can share one data buffer under *different* masks at once — e.g. two
users seeing different redactions of the same underlying array, or two CV
folds masking the same dataset differently — with no data copy. Worth
testing explicitly once views are implemented (phase 2): construct two views
over the same `base` data with different masks and confirm they don't
interfere with each other.

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

## Goal: no rebuild required for plain-array users

Anyone *not* touching masked arrays should be unaffected without rebuilding
anything of theirs:

- **Packages precompiled against stock numpy** (pandas, scipy, any wheel
  using `PyArray_DATA`/accessor macros) must keep working unmodified against
  this fork's numpy. This falls out of the existing version-negotiation in
  `import_array()`: an extension declares "I need API version ≤ X", the
  runtime declares "I provide version ≥ X" — compatible, no rebuild. Must
  hold as an invariant through every phase: `mask` is purely additive, gated
  behind a version strictly newer than anything existing code could request
  (phase 1), and `mask == NULL` behavior is byte-identical to upstream.
- **Someone who wants masked arrays from Python**: no rebuild at all — just
  run on this fork's already-built numpy and call the new Python-level API.
  Same as adopting any new feature in a new numpy release.
- **Someone who wants to touch `mask` from their own C extension**: *they*
  rebuild *their* extension against this fork's headers — not numpy itself
  (already built). Never the other direction.

## Backward compatibility of adding a struct field

Checked against the actual header (`ndarraytypes.h:787-857`): this is safer
than a naive "patching a public C struct always breaks ABI" take suggests.
`PyArrayObject_fields` (the real struct) is already hidden behind an opaque
`PyArrayObject` typedef for any external code that sets
`NPY_NO_DEPRECATED_API` (recommended since NumPy 1.7, 2013) — such code only
touches the array via `PyArray_DATA`/`PyArray_NDIM`/etc., which resolve
through the `PyArray_API` function-pointer table at runtime, not hardcoded
offsets. There's direct precedent: `_buffer_info` was added at
`NPY_1_20_API_VERSION` and `mem_handler` at `NPY_1_22_API_VERSION`, both
gated the same way `mask` will be (see phase 1).

Net effect: source code using the accessor API is unaffected. Old
*precompiled* wheels that read fields directly (`arr->data` without the
deprecated-API guard) would break at runtime against a numpy built with the
new field — same risk numpy itself already accepted twice for `_buffer_info`
and `mem_handler`, not a new category of risk this project introduces.

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

- [x] **0 — Setup**
  - [x] New branch `refactor/ndarray-mask` off freshly-pulled `main`
        (didn't reuse `enh/mean-var-non-legacy-dtype`)
  - [x] Confirmed build config / baseline against numpy `main` — see the
        "Regression baseline" table above (row 0)
- [x] **1 — Struct & invariants** (done on `refactor/ndarray-mask`)
  - [x] Added `NPY_2_7_API_VERSION 0x00000017` (`numpyconfig.h`), bumped
        `C_API_VERSION` in `numpy/_core/meson.build` (header-only change,
        same md5 in `cversions.txt` reused — no C-API function table
        change, matching the "Version 19: only header additions" precedent).
  - [x] Added `mask` field to `PyArrayObject_fields` in `ndarraytypes.h`,
        gated `#if NPY_FEATURE_VERSION >= NPY_2_7_API_VERSION`, appended
        after `mem_handler` (append-only keeps existing field offsets
        stable — see backward-compat note below). Added `PyArray_MASK()`
        accessor next to `PyArray_BASE()`.
  - [x] Init `fa->mask = NULL` in `ctors.c` (`PyArray_NewFromDescr` path),
        unguarded like sibling fields (internal build always has the
        current feature version).
  - [x] Refcount handling: `Py_CLEAR(fa->mask)` added to
        `_clear_array_attributes` in `arrayobject.c` (same function `base`
        is cleared in, used by both dealloc and pickle `__setstate__`).
  - [x] Enforce `mask->mask == NULL` invariant: added internal (not yet
        public-C-API) `PyArray_SetMaskObject(arr, obj)` in `arrayobject.c`
        (declared in `arrayobject.h`), mirroring `PyArray_SetBaseObject`'s
        shape. Refuses: non-ndarray, non-bool dtype, `PyArray_MASK(obj) !=
        NULL` (the invariant), and shape mismatch. Not registered in
        `numpy_api.py` yet — internal-only until a later phase needs
        external callers.
  - [x] Verified: `spin build` clean, zero regression — see the "Regression
        baseline" table near the top of this file.
  - [x] Expose `mask` as a Python-level property (`arr.mask`), since phase 2+
        needs a way to attach masks from Python to test against. Added
        `array_mask_get`/`array_mask_set` in `getset.c` next to `base`'s
        getter (settable, unlike `base`, mirroring `dtype`'s
        getter+setter shape): getter returns `None` when unset;
        setter/`del arr.mask` both route through `PyArray_SetMaskObject`,
        so the invariant/dtype/shape checks apply from Python too, not
        just from C callers.
  - [x] Leak-checked (refcount + RSS, not just correctness): balanced
        `sys.getrefcount` across 1000x per error path (invariant/dtype/shape
        rejection), 100000x attach+explicit-clear, 100000x
        attach-then-dealloc-without-clearing, and mask replacement dropping
        the old mask's reference. RSS flat (0.0 MB delta) over 500000
        alternating attach/clear/dealloc iterations.
- [ ] **2 — Basic ops: mask propagation + reorder correctness (promoted
        ahead of dispatch — found by manually probing `copy`/`view`/
        `reshape`/`sort`/`partition` right after phase 1 landed)**
  - Current (untouched since phase 1) behavior, verified empirically:
    `copy()`/`view()`/`a[:]`/`reshape()` all **silently drop** the mask
    (new array's `mask` is `NULL`) — safe but wrong, and defeats the
    case-2 goal (redaction should survive a plain `.view()` unless the
    caller explicitly asks for a different mask). `sort()`/`partition()`
    are worse: they reorder **data** in place but leave **mask** exactly
    where it was — the mask now marks the wrong elements, silently
    (verified: `[3,1,4,1,5]` masked at indices `[1,4]`, after `.sort()` the
    same indices `[1,4]` are still marked even though the values that were
    there moved to indices `[0,1]`). This is a correctness bug once masks
    exist for real use, not just a missing-feature gap — hence jumping the
    queue ahead of ufunc dispatch.
  - [ ] `copy()`/`view()`/`a[:]` (`multiarray/ctors.c`, `mapping.c` basic
        indexing path): propagate the parent's mask to the new array by
        default (deep-copy the mask array for `copy()`; for a view, decide
        share-vs-copy against the phase-4 "two views, different masks over
        one buffer" test from the Scope section — a plain `.view()` with no
        new mask argument should probably still see the parent's
        redaction, not lose it)
  - [ ] `reshape()`/`ravel()`/`squeeze()` (`multiarray/shape.c`): reshape
        the mask identically alongside data, same shape transform applied
        to both
  - [ ] `sort()`/`argsort()`/`partition()`/`argpartition()`
        (`npysort/*.c.src`, `multiarray/item_selection.c`): must permute
        `mask` by the exact same permutation applied to `data` — not leave
        it stationary. Add a regression test asserting mask tracks value
        identity through a sort, mirroring the `[3,1,4,1,5]` case above.
- [ ] **3 — Ufunc dispatch (arithmetic, trig, comparisons)**
  - [ ] `umath/ufunc_object.c` — central `if (has_mask)` branch point
  - [ ] `umath/dispatching.c` — masked loop variant selection (NEP 43)
  - [ ] `umath/loops*.c.src` — masked-AVX512 loop + strided fallback loop,
        starting with binary arithmetic on `float64`/`int64` as proof of
        concept before generalizing
- [ ] **4 — Reductions / accumulate**
  - [ ] `umath/reduction.c`, `umath/ufunc_object.c`
        (`reduce`/`accumulate`/`reduceat`/`outer`/`at`)
  - [ ] `multiarray/calculation.c` (`.sum()`, `.mean()`, `.argmax()`...)
- [ ] **5 — Further view propagation** (basic reshape/view covered in
        phase 2 already; this is the rest)
  - [ ] `multiarray/getset.c` (`.T` and friends)
  - [ ] `multiarray/ctors.c` (`broadcast_to`)
- [ ] **6 — Indexing**
  - [ ] `multiarray/mapping.c` — basic indexing already covered in phase 2;
        advanced/fancy + boolean indexing (copy — build new mask
        explicitly), assignment through indexing
- [ ] **7 — Search** (sort/partition covered in phase 2; this is the rest)
  - [ ] `npysort/*.c.src`, `multiarray/item_selection.c`
        (`searchsorted`, `take`/`put`/`choose`/`repeat`)
- [ ] **8 — Combine / split**
  - [ ] `multiarray/multiarraymodule.c` (`concatenate`)
  - [ ] `multiarray/item_selection.c` (`repeat`, `choose`)
- [ ] **9 — Linear algebra**
  - [ ] `umath/matmul.c.src`
  - [ ] `numpy/linalg/umath_linalg.c.src` (likely: refuse/raise on masked
        input rather than trying to propagate through LAPACK calls)
- [ ] **10 — Casting**
  - [ ] `multiarray/convert_datatype.c`, `multiarray/convert.c`
- [ ] **11 — Python-level surface**
  - [ ] `numpy/_core/arrayprint.py` (repr/str show masked cells)
  - [ ] `numpy/lib/_arraysetops_impl.py` (`unique`/`isin` mask-awareness)
  - [ ] `multiarray/methods.c` (`__reduce__`/pickle, `tobytes`/`tofile`)
  - [ ] `multiarray/buffer.c` (buffer protocol — decide: expose data only,
        or refuse when masked)
- [ ] **12 — Benchmarking**
  - Relevant `asv` files: `benchmarks/benchmarks/bench_core.py`,
    `bench_indexing.py`, `bench_ufunc.py`, `bench_ufunc_strides.py`
    (contiguous vs. strided — the one closest to the fast/fallback path
    split), `bench_array_coercion.py`.
  - `spin bench -t <name>` is slow by design (asv calibrates + repeats each
    benchmark in its own subprocess; `bench_ufunc_strides` alone is a
    dtype × stride × op matrix, hundreds of parameter combos).
    **Phase 0 baseline: run full (no `-q`)** — done below, this is the
    number everything else compares against. **Phase 1 onward, during
    day-to-day development: `spin bench -q -t <name>`** (quick, one run, no
    calibration) for "did I break something" checks. Only drop back to a
    full (no `-q`) run for the real before/after comparison once a phase's
    masked path is actually implemented and ready to judge.
  - [ ] No-mask path: confirm zero measurable regression vs. upstream `main`
  - [ ] Contiguous masked path: confirm near-native speed vs. plain op
        (AVX-512 available)
  - [ ] Non-contiguous masked path: measure, document expected slowdown
- [ ] **13 — Testing**
  - [ ] Mask never silently lost across every op in phases 2–11
        (the `numpy.ma`-style leak bugs this design is meant to avoid)
  - [ ] `mask->mask == NULL` invariant enforced everywhere a mask is attached
  - [ ] ASAN/UBSAN clean on the new field's lifetime (alloc/dealloc/view
        chains)
