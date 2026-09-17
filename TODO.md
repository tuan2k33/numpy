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
| 2 — copy/view/reshape/sort mask propagation | *(run combined, no per-file split)* | *(run combined, no per-file split)* | 14916 passed, 17 skipped, 12 deselected | ✅ identical |
| 3 — ufunc dispatch (arithmetic/trig/comparisons) | *(run combined, no per-file split)* | *(run combined, no per-file split)* | 14916 passed, 17 skipped, 12 deselected | ✅ identical |

Add one row per phase from here on, run against the same two files at
minimum (more as later phases touch more test files per the mapping table
below). New masked-path behavior gets its own dedicated suite instead of a
row here: `numpy/_core/tests/test_mask.py` (40 tests as of phase 3). Phase 3
also reran `test_umath.py` + `test_ufunc.py` (the phase-3 baseline files from
the mapping table below) as an additional no-mask regression check: 5615
passed, 60 skipped, 7 xfailed, all pre-existing (not new).

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
- Invariant: `mask->mask` must always be `NULL` (no masked masks) — enforced
  by usage-tracking, not by dtype. An earlier version of this fix tried "an
  array of dtype `bool` can never itself carry a mask" as a structural
  shortcut, but that's actually wrong: mask arrays are always bool, but not
  every bool array *is* a mask — comparison ufuncs (`a < b`, phase 3) produce
  ordinary bool arrays that must still be allowed to carry their own mask
  when they aren't currently serving as anyone else's. The real invariant is
  usage-based: an array cannot receive a mask *while it is currently in use
  as some other array's mask*, tracked via the internal `NPY_ARRAY_IS_MASK`
  flag (`arrayobject.h`), set when attached and cleared when detached or
  replaced. A plain "reject if `obj->mask != NULL` at attach" check (checking
  only the incoming candidate) has a point-in-time hole: nothing stops
  attaching a mask to `obj` *after* it's already serving as another array's
  mask (`arr.mask.mask = x` would still nest). Checking the flag on the
  *receiver* (`arr`, not just the incoming `obj`) at attach time closes that
  hole — see phase 2/3 checklists. Not refcounted: sharing the exact same
  mask object (by identity, not via `.view()` of it) across more than one
  owner is an accepted edge case where detaching from one owner clears the
  flag even if another owner still references it — a niche, undesigned-for
  usage, not the `.view()`-based per-owner sharing phase 2 actually relies on.
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

## Mask semantics and propagation

- mask == NULL means unmasked / ordinary ndarray.
- mask[i] == False means visible; mask[i] == True means hidden.
- Mask affects observability, not computation: underlying data is always
  computed exactly as in normal NumPy.
- Every non-NULL mask is a plain bool ndarray with exactly the same shape
  as its owner and mask == NULL itself.
- Elementwise operations propagate masks by OR after normal NumPy
  broadcasting.
- Views transform the mask using exactly the same indexing/striding
  transformation as the data; masks are not eagerly copied.
- Operations that permute/select elements must apply the same permutation
  or index mapping to the mask.
- Reductions preserve existing NumPy output shape/type semantics. Their
  output mask is the OR reduction of the masks of all input elements
  contributing to each output element.
- If all contributing inputs are unmasked, the result mask remains NULL.
- Scalar results preserve their normal NumPy scalar type; masked scalar
  results carry mask state rather than becoming None.
- Mask propagation is separate from output/observability policy. Data is
  fully computed; masked values are suppressed only at defined output
  boundaries.
- No operation may change ndarray shape, dtype, broadcasting rules, or
  ordinary unmasked usage merely because masks are enabled.
- `.copy()` deep-copies the mask into an independent buffer, exactly like it
  deep-copies data; `.view()`/basic slicing instead transform the mask via
  the identical indexing/striding operation applied to data, sharing the
  underlying mask buffer (consistent with "not eagerly copied" below). Two
  independent views can still carry independently-assigned masks — rebinding
  `view.mask = new_arr` only affects that view — while in-place mutation of a
  *shared* mask buffer through one view is visible through any other view
  sharing it, mirroring ordinary data-view semantics.

## Unmasked representation

- A newly created ordinary ndarray has `mask == NULL`.
- `mask == NULL` is the canonical representation of an entirely unmasked
  array; it is semantically equivalent to an all-False mask, but an
  all-False mask must never be materialized merely to represent the
  unmasked state.
- External mask assignment (`arr.mask = m`, `PyArray_SetMaskObject`) must
  canonicalize an all-False `m` down to `NULL` — a one-time scan cost paid
  at assignment, not a maintained invariant re-checked per-op.
- Internal mask propagation (view/copy/elementwise/reduction results) should
  preserve `NULL` whenever the result is provably unmasked from its inputs'
  `NULL`-ness alone, and should avoid a full-mask scan solely to canonicalize
  an already-non-NULL mask that happens to be all-False — only the external
  assignment path pays that scan cost.
- Operations whose inputs all have `mask == NULL` must produce
  `mask == NULL`.
- `mask == NULL` is therefore both a semantic state and the fast path.

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
- [x] **2 — Basic ops: mask propagation + reorder correctness (promoted
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
  - [x] Hardened `PyArray_SetMaskObject`'s invariant enforcement
        (`arrayobject.c`): refuses attaching a mask to a receiver currently
        flagged `NPY_ARRAY_IS_MASK` (checked on `arr`, not just the incoming
        object's own `mask` field) — closes the point-in-time hole where
        `arr.mask.mask = x` could nest a mask after the initial attach
        already passed. **Revised in phase 3**: the first version of this
        used a blanket "bool-dtype arrays can never carry a mask" rule
        instead of a usage flag, which turned out to be wrong — comparison
        ufuncs produce bool arrays that must still be maskable; see phase 3
        checklist and the corrected "Settled design decisions" entry. Also
        canonicalizes an all-False incoming mask to `NULL` via
        `PyArray_CountNonzero` on assignment (see "Unmasked representation"
        above) — a one-time scan paid only here, not by internal propagation.
  - [x] `copy()`/`view()`/`a[:]`: propagated at a small number of choke
        points rather than duplicated per call site —
        `PyArray_NewCopy` (`convert.c`) deep-copies the mask for every
        `.copy()` call site (and for `shape.c`'s reshape-needs-a-copy
        fallback, for free); `PyArray_View`'s no-dtype-change path
        (`convert.c`) attaches a *view* of the mask (shares the buffer,
        not eagerly copied, matching "Mask semantics and propagation"
        above — confirmed two views over one buffer can carry
        independently-*reassigned* masks while still seeing shared
        in-place mask mutations, exactly like data sharing); basic
        indexing (slices/integers/newaxis/ellipsis, no fancy component)
        in `mapping.c`'s `array_subscript`/`array_item_asarray` replays
        the identical parsed indices against the mask. Dtype-changing
        `.view(new_dtype)` does not propagate the mask yet (element
        count/shape can change under a dtype view; no obviously-correct
        mask reshape for that case) — noted as a follow-up nuance, not a
        phase-2 blocker. Fancy/boolean indexing is still phase 6.
  - [x] `reshape()`/`ravel()`/`squeeze()` (`shape.c`): `_reshape_with_copy_arg`
        recurses `PyArray_Newshape` onto the mask itself (reusing its own
        view-if-possible/copy-if-needed logic rather than re-deriving
        stride math); `PyArray_Ravel`'s contiguous-view fast path and its
        `PyArray_Flatten` fallback each recurse into themselves on the
        mask; `PyArray_Squeeze`/`PyArray_SqueezeSelected` call
        `PyArray_RemoveAxesInPlace` on the mask with the same axis flags
        right after doing so for the data.
  - [x] `sort()`/`partition()` (`item_selection.c`): **correctness-first
        fallback, not the true in-place algorithm.** New
        `_sort_with_mask` computes an argsort permutation (ordering is
        defined by the real values — mask never affects comparison) and
        replays it onto both the data and the mask, one line along `axis`
        at a time, via existing `PyArray_TakeFrom`/`PyArray_CopyInto`
        (so every dtype's refcounting/byteswap/alignment is already
        handled correctly, no hand-rolled element copy). `PyArray_Sort`
        dispatches to it whenever `PyArray_MASK(op) != NULL`;
        `PyArray_Partition` falls back to a full `PyArray_Sort` when
        masked (a sort is a valid, if asymptotically slower, partition).
        This sacrifices sort's in-place no-extra-allocation property and
        partition's O(n) advantage when masked — threading a companion
        mask buffer through every swap of every backend
        (quicksort/timsort/heapsort × dtype in `npysort/*.c.src`) is a
        real performance project, deferred to phase 12/13, not required
        for phase 2's correctness goal. `argsort()`/`argpartition()`
        need no changes: they return an index array, not a masked array.
  - [x] Regression tests: `numpy/_core/tests/test_mask.py` (new file) —
        invariant/canonicalization tests, copy/view/slice mask
        propagation (including the shared-vs-independent-buffer
        distinction), reshape/ravel/squeeze, and the sort/partition
        mask-permutation cases (distinct-value arrays for an unambiguous
        mapping, plus a duplicate-value case that only checks the
        preserved mask count, since tied elements have no guaranteed
        relative order under a non-stable sort).
- [x] **3 — Ufunc dispatch (arithmetic, trig, comparisons)**
  - [x] **Correctness-first, black-box wrapper — not the planned AVX-512
        masked loop.** Like phase 2's sort/partition fallback, the real
        masked-SIMD loop work (`umath/dispatching.c` NEP 43 variant
        selection, `umath/loops*.c.src` masked/strided loops) is a
        performance project of its own, deferred to a later phase. Instead,
        `ufunc_generic_vectorcall` (`umath/ufunc_object.c`) — the single
        choke point *every* ufunc call and operator overload goes through
        (`a + b` vectorcalls the `add` ufunc object exactly like
        `np.add(a, b)`) — wraps the existing, untouched
        `ufunc_generic_fastcall`: if none of the inputs carry a mask, the
        original call happens with no other change (matching "no-mask path
        byte-identical" exactly); if any input is masked, the same call
        still runs unmodified to get the real, fully-computed result (mask
        never affects computation), then a new `_propagate_ufunc_result_mask`
        OR-combines the input masks via the existing `bitwise_or`/`PyNumber_Or`
        machinery (broadcasting handled by that same machinery, not
        hand-rolled), broadcasts the combined mask up to each output's exact
        shape (OR against an explicit same-shape zero array), and attaches
        it via `PyArray_SetMaskObject`. Handles multi-output ufuncs
        (`divmod`) and `out=`/in-place operators (`+=`) — the output object
        already gets fully overwritten either way, so replacing its mask
        the same way is safe. gufuncs (`ufunc->core_enabled`, e.g. `matmul`)
        are excluded (phase 9).
  - [x] **Found and fixed a real bug in phase 2's invariant while
        implementing this**: comparison ufuncs (`a < b`) produce bool-dtype
        output, but phase 2's "hardened" invariant was a blanket "no bool
        array can ever carry a mask" — which would have made comparison
        results permanently unmaskable, contradicting phase 3's own scope.
        Replaced with proper usage-tracking: a new internal
        `NPY_ARRAY_IS_MASK` flag (`multiarray/arrayobject.h`), set on an
        array while it's attached as *some* array's mask and cleared on
        detach/replace; `PyArray_SetMaskObject` now refuses attaching a
        mask to a receiver that currently has this flag set (not "is dtype
        bool"). See the corrected "Settled design decisions" entry for the
        full reasoning and the accepted non-refcounted-flag limitation.
  - [x] Known, documented gaps (not phase-3 blockers): `where=` (partial
        writes) skips mask propagation entirely rather than risk getting
        stale-element masking wrong; a 0-d result that decays to a plain
        Python/numpy scalar can't carry a mask (scalar types have no
        `mask` field) — that piece of mask information is dropped for that
        call.
  - [x] Regression tests: 14 new cases in `numpy/_core/tests/test_mask.py`
        (`TestMaskUfuncPropagation`) covering binary/unary/trig/comparison
        OR-propagation, one-sided-mask broadcasting, operator-vs-`np.add`
        equivalence, no-mask-stays-unmasked, in-place (`+=`), `out=`,
        multi-output (`divmod`), the `where=` gap, and result-mask
        independence (mutating a result's mask must not affect an input's).
        Reran `test_umath.py`/`test_ufunc.py` (this phase's baseline files)
        as well as the standing `test_multiarray.py`/`test_indexing.py`
        baseline — all identical to pre-phase-3, see the table above.
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
