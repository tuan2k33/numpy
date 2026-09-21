# TODO — native mask support in `ndarray` core

Research fork: give `ndarray` a built-in, opt-in `mask` so operators only need
`if (mask)` instead of `numpy.ma`'s per-function Python wrappers. Target
**numpy `main`/dev only** (currently `2.6.0.dev0`); no backport to released
branches. Track upstream and rebase periodically rather than diverging long-term.

Design rules live in [`DESIGN.md`](DESIGN.md). This file is the working
roadmap and validation record.

## Housekeeping

- [ ] After every phase implementation, cross-test (pipenv/devenv,
  mask/nomask), review [`DESIGN.md`](DESIGN.md), and check relevant NumPy
  NEPs/current `main` behavior for dispatch, dtype, iterator, and array API
  changes. Verify no regression, leakage, or plain-array behavior change.
- [ ] Pull from upstream NumPy's `origin/main` every day and before every
  push. Do not use the personal fork's `fork/main` as the synchronization
  source — this is a long-running fork of a fast-moving codebase, so drift
  compounds fast if skipped.

## Regression baseline

`mask == NULL` must keep every existing code path byte-identical to
upstream. Tracking the actual pass counts here after each phase, not just
inside the phase's checklist, so drift is easy to spot at a glance:

Current focus:

- **Phase 6:** implement advanced/fancy and boolean indexing.
- **Phase 7:** implement gather, scatter, and combine/split operations.
- Phase 5 is complete (incl. `astype`/`flatten`/`diagonal`/`real`/`imag`); keep the cross-test, design review, NEP review, and
  plain-array baseline as the gate for every change.

- **Phase 0 — baseline (no code changes)**
  - `test_multiarray.py`: 14810 passed, 17 skipped, 12 deselected.
  - `test_indexing.py`: 106 passed.
  - Combined: 14916 passed, 17 skipped, 12 deselected.
- **Phase 1 — `mask` field added**
  - Combined: 14916 passed, 17 skipped, 12 deselected.
  - Match with phase 0: ✅ identical.
- **Phase 2 — copy/view/reshape/sort mask propagation**
  - Combined: 14916 passed, 17 skipped, 12 deselected.
  - Match with the previous row: ✅ identical.
- **Phase 3 — ufunc dispatch**
  - Combined: 14916 passed, 17 skipped, 12 deselected.
  - Match with the previous row: ✅ identical.
- **Phase 4 — reductions and `where=` merge fix**
  - Combined: 14916 passed, 17 skipped, 12 deselected.
  - Match with the previous row: ✅ identical.
- **Phase 5 — transpose, broadcast, view and dtype-cast mask transport**
  - `test_multiarray.py` + `test_indexing.py` (`-m "not slow"`): 14932
    passed, 17 skipped, 12 deselected, 0 failed. The count is 16 higher than
    the rows above with test files byte-identical to the phase-0 base; it is
    a collection difference in this environment (not from a code change, no
    new skips/failures) — re-baseline against it from here on.
  - Plus `test_umath.py`, `test_ufunc.py`, `test_shape_base.py`,
    `numpy/lib/tests/test_stride_tricks.py`: 20931 passed, 77 skipped,
    12 deselected, 7 xfailed (pre-existing xfails), 0 failed.
  - `test_mask.py`: all phase-5 tests pass (the phase-6 WIP tests in the same
    file are excluded from this gate).
  - Match with the plain-array baseline: ✅ no regression.

Add one row per phase from here on, run against the same two files at
minimum (more as later phases touch more test files per the mapping table
below). New masked-path behavior gets its own dedicated suite instead of a
row here: `numpy/_core/tests/test_mask.py` (40 tests as of phase 3). Phase 3
also reran `test_umath.py` + `test_ufunc.py` (the phase-3 baseline files from
the mapping table below) as an additional no-mask regression check: 5615
passed, 60 skipped, 7 xfailed, all pre-existing (not new).

## Design reference

The scope, invariants, mask propagation rules, unmasked representation,
compatibility requirements, and NEP alignment are maintained in
[`DESIGN.md`](DESIGN.md). Update that file when a design decision changes;
keep this file focused on implementation phases and validation.

## Existing test suites to reuse as templates / regression baseline

No dedicated test suite exists yet (feature isn't written). These existing
numpy test files cover the same ground and should be mined per phase — either
as a pattern to copy for the masked case, or as the no-mask regression
baseline that must keep passing unchanged:

- **Phase 3 — ufunc dispatch**
  - Files: `numpy/_core/tests/test_umath.py` (5474 lines) and
    `test_ufunc.py` (3523 lines).
  - Use them for per-ufunc correctness and the unmodified `mask=None`
    regression check. Mine them for masked-operand cases.
- **Phase 4 — reductions**
  - Files: reduction/accumulate sections in
    `numpy/_core/tests/test_umath.py` and
    `numpy/ma/tests/test_core.py`.
  - The `numpy.ma` reduction semantics are the direct precedent to compare
    against or deliberately diverge from.
- **Phase 5 — view propagation**
  - Files: `numpy/_core/tests/test_shape_base.py` and
    `numpy/ma/tests/test_subclassing.py` (469 lines).
  - The subclassing tests are effectively a list of past propagation bugs to
    avoid repeating.
- **Phase 6 — indexing**
  - File: `numpy/_core/tests/test_indexing.py` (1717 lines).
  - It has the most thorough coverage of basic, advanced, and boolean
    indexing edge cases.
- **Phase 7 — gather, scatter, and combine**
  - File: `numpy/_core/tests/test_item_selection.py` (178 lines).
  - It covers `take`/`put`/`choose`/`repeat` and is a small starting template.
- **Phase 8 — casting**
  - File: `numpy/_core/tests/test_multiarray.py` (12122 lines).
  - Grep for the `astype` and `can_cast` sections rather than reading the
    whole file.
- **Phase 9 — linear algebra**
  - Files: `umath/matmul.c.src` and `numpy/linalg/umath_linalg.c.src`.
  - Decide whether masked LAPACK input should be rejected or propagated.
- **Phase 10 — Python surface**
  - Files: `numpy/ma/tests/test_core.py` (6276 lines), `test_old_ma.py`
    (939 lines), `test_mrecords.py` (513 lines), `test_deprecations.py`, and
    `test_regression.py`.
  - The full `numpy.ma` suite is the closest example of an end-to-end masked
    array test suite and a source of leak/surprise regression cases.

`numpy/ma/tests/` in particular is worth a full pass before writing any new
test: most of its ~7800 lines encode a real bug or surprising-behavior
decision made over `numpy.ma`'s lifetime — deciding *on purpose* whether the
mask-in-core design repeats or fixes each one is more valuable than writing
tests from scratch.

## Phases

<details>
<summary>Completed phases 0–4</summary>

- [x] **0 — Setup**
  - [x] New branch `refactor/ndarray-mask` off freshly-pulled `main`
        (didn't reuse `enh/mean-var-non-legacy-dtype`)
  - [x] Confirmed build config / baseline against numpy `main` — see the
        "Regression baseline" table above (row 0)
- [x] **1 — Struct, invariants, and minimal Python API**
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
- [x] **2 — Basic ops**: mask propagation + reorder correctness (promoted
        ahead of dispatch — found by manually probing `copy`/`view`/
        `reshape`/`sort`/`partition` right after phase 1 landed)
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
        `.view(new_dtype)` is handled in phase 5: it is rejected for masked
        arrays; use `astype()` to preserve mask semantics.
        Fancy/boolean indexing is still phase 6.
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
        real performance project, not required for phase 2's correctness
        goal. `argsort()`/`argpartition()`
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
        are excluded (the linear-algebra phase).
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
  - [x] **Revised after phase 4's design discussion**: `where=` is no
        longer a blanket skip. It's now treated as a *selective write* —
        for both data and mask: `where=True` positions get the
        propagated/OR-combined mask, `where=False` positions keep `out`'s
        *old* mask (read directly off `PyArray_MASK(out)` after the call,
        since the wrapped call never touches `.mask`, only data — no need
        to snapshot it beforehand). Implemented via `merged =
        PyArray_Where(where, new_mask, old_mask)`, reusing `np.where`'s own
        broadcasting/selection instead of re-deriving the ufunc's
        iteration topology by hand. This only has a principled "old mask"
        when the caller passed a real `out=`; two narrower gaps remain:
        `where=` with `out=None` (freshly allocated, `where=False`
        positions are *uninitialized*, not "old, valid, differently-masked
        data" — no principled old mask to merge with) and `where=`
        combined with a multi-output ufunc (rare combination, not worth
        the complexity yet) both still skip propagation entirely, same as
        the original gap.
  - [x] Other known, documented gaps (not phase-3 blockers): a 0-d result
        that decays to a plain Python/numpy scalar can't carry a mask
        (scalar types have no `mask` field) — that piece of mask
        information is dropped for that call.
  - [x] Regression tests: 14 new cases in `numpy/_core/tests/test_mask.py`
        (`TestMaskUfuncPropagation`) covering binary/unary/trig/comparison
        OR-propagation, one-sided-mask broadcasting, operator-vs-`np.add`
        equivalence, no-mask-stays-unmasked, in-place (`+=`), `out=`,
        multi-output (`divmod`), the (then-blanket) `where=` gap, and
        result-mask independence (mutating a result's mask must not affect
        an input's). Reran `test_umath.py`/`test_ufunc.py` (this phase's
        baseline files) as well as the standing `test_multiarray.py`/
        `test_indexing.py` baseline — all identical to pre-phase-3, see the
        table above. **Updated in phase 4** once the `where=` gap was
        narrowed (see above): the old blanket-skip test was rewritten into
        4 tests covering the `merge_where` selective-write behavior (with
        and without a pre-existing `out` mask) and the two narrower
        remaining gaps (`out=None`, multi-output).
- [x] **4 — Reduction family and indexed updates**
  - [x] **Convention (settled after design discussion): run the identical
        reduce-like call again with `logical_or` substituted for the data
        ufunc, on the mask(s) instead of the data**, rather than
        hand-deriving mask combination per method — reuses each method's
        own correct axis/shape/segment/repeated-index/identity-exclusion
        logic instead of re-implementing it:
        `add.reduce(x)`→`logical_or.reduce(x.mask)`,
        `add.accumulate(x)`→`logical_or.accumulate(x.mask)` (cumulative
        OR — position i has incorporated everything up to i),
        `add.reduceat(x,ind)`→`logical_or.reduceat(x.mask,ind)` (OR within
        each segment), `add.outer(a,b)`→`logical_or.outer(a.mask,b.mask)`
        (outer's result shape is `a.shape+b.shape`, concatenation not
        broadcasting, so phase 3's zeros-broadcast trick doesn't apply —
        `outer()` itself already produces the right shape),
        `add.at(x,idx,y)`→`logical_or.at(x.mask,idx,y.mask)` (repeated
        indices accumulate correctly for free, since OR is
        associative/commutative/idempotent regardless of order, same
        reason `add.at` already handles repeats). `initial=NULL` is
        deliberately passed for the mask reduce (not the caller's own
        `initial`, a data identity, not a mask fact) so `logical_or`'s own
        identity (`False`) applies — an identity-filled slot contributes
        no real data and must never read as masked. Implemented in
        `umath/ufunc_object.c`: `_propagate_reduce_mask` /
        `_propagate_accumulate_mask` / `_propagate_reduceat_mask` (hooked
        into `PyUFunc_GenericReduction`'s switch, right after each of
        `PyUFunc_Reduce`/`Accumulate`/`Reduceat` computes `ret`),
        `_propagate_outer_mask` (hooked into `ufunc_outer`),
        `_propagate_at_mask` (hooked into `ufunc_at`, called directly
        rather than through the Python-visible method, matching phase 3's
        discipline of calling C API directly).
  - [x] `where=` for `reduce` (the only one of the five that accepts it):
        passed straight through unchanged to the mask's `logical_or.reduce`
        call — since `reduce` always fully (re)writes every output
        position regardless of `where=` (it only decides which *inputs*
        feed the combine, not which *outputs* get written), there's no
        "old mask to preserve" case here, unlike phase 3's elementwise
        `where=` gap. An element `where` excludes is correctly excluded
        from the mask combine too, for the same reason it's excluded from
        the data combine.
  - [x] Unary `.at()` (`ufunc->nin == 1`, e.g. `np.negative.at`) needs no
        mask update at all — recomputing an element from itself introduces
        no new masked information — so `_propagate_at_mask` is only called
        when `ufunc->nin == 2`.
  - [x] **`multiarray/calculation.c` needed zero changes** — traced every
        function first: `.sum()`/`.prod()`/`.any()`/`.all()`/`.max()`/
        `.min()` all route through `PyArray_GenericReduceFunction` →
        `PyUFunc_Reduce`, so they inherit mask propagation automatically.
        `.mean()` = `sum()` then `PyNumber_TrueDivide` by a plain Python
        float (already mask-aware from phase 3) — also free. `.std()`/
        `.var()` (`__New_PyArray_Std`) compose entirely from `PyArray_Mean`
        + `PyNumber_Subtract`/`multiply` (phase-3 ufuncs) +
        `PyArray_GenericReduceFunction(..., add)` (this phase's target) +
        a scalar multiply + `sqrt` — also free, with a deliberate,
        documented consequence: since `x - mean(x)` OR's every element's
        mask with the mean's (already axis-wide) mask, **if any element in
        an axis is masked, `std`/`var` for that *entire axis* reads
        masked** — correct given variance mixes every element in the axis
        together, not a bug.
  - [x] `.argmax()`/`.argmin()` need no changes — like `argsort` (phase 2),
        they return an index, not data, so the result carries no mask; and
        the masked element must still fully participate in the comparison
        (mask never affects computation) — confirmed by test, not a code
        change.
  - [x] Regression tests: 26 new cases in `numpy/_core/tests/test_mask.py`
        (`TestMaskReduceLike`) covering reduce (OR-of-contributing,
        axis-scoped, `where=`-exclusion, empty-slice-identity-not-masked,
        full-reduction scalar-decay gap), the calculation.c free-inherit
        confirmation (`sum`/`mean`/`max`/`min`/`any`/`all`), the
        `std`/`var` whole-axis-spread behavior, `argmax`/`argmin`
        mask-blind comparison, accumulate (cumulative OR), reduceat
        (per-segment OR), outer (concatenated-shape OR, one-sided
        broadcast), and `at` (touched-positions-only, second-operand
        propagation, repeated-index accumulation, unary no-op, no-mask
        fast path). Also fixed/extended 4 existing `TestMaskUfuncPropagation`
        cases for the `where=` merge-semantics change (see phase 3 above).
        Reran `test_umath.py`/`test_ufunc.py` (identical: 5615 passed, 60
        skipped, 7 xfailed) and the standing `test_multiarray.py`/
        `test_indexing.py` baseline (identical: 14916 passed, 17 skipped,
        12 deselected — see the table above).
  - [x] Leak-checked (RSS + refcount): RSS flat across 20000-iteration
        loops for each of reduce/accumulate/reduceat/outer/at/where-merge;
        refcount stable across 1000 repeated calls for the reduce path,
        the `.at()` in-place-mutate-existing-mask path, and the
        where-merge old-mask replacement path (old mask correctly
        dereferenced, not leaked, when replaced).

    </details>

- [x] **5 — Core transport completion** (follow-up to phase 2)
  - [x] `multiarray/shape.c` `PyArray_Transpose` (covers `.T`, `.mT`,
        `transpose`, `swapaxes`, `moveaxis`/`rollaxis`, `matrix_transpose`):
        the identical permutation is applied to the mask (a view).
  - [x] `numpy/lib/_stride_tricks_impl.py` (`broadcast_to`; also reached by
        `broadcast_arrays`): mask is `broadcast_to`'d too (read-only view).
  - [x] `.view()`: same-dtype views (including an explicit
        `a.view(a.dtype)`) share the mask; dtype-changing `.view()` on a
        masked array raises a clear error pointing at `astype()`. Unmasked
        arrays keep NumPy's normal dtype-view behavior.
  - [x] Pulled forward from phase 8 because phase 5's own error message
        sends users to `astype()`, and a workaround that silently drops the
        mask would be a redaction leak: shared helpers
        `PyArray_CopyMaskFrom`/`PyArray_ViewMaskFrom` (`arrayobject.c`) are
        used by `astype` (`methods.c`), `PyArray_CastToType`
        (`convert_datatype.c`) and `PyArray_FromArray`'s copy-and-cast path
        (`ctors.c`, i.e. `np.array/asarray(a, dtype=...)`). The mask is
        copied, not shared, exactly like the data. A no-op `astype` returns
        `self` (mask untouched).
  - [x] Same-class gaps found while re-auditing phase 5 (they were silently
        dropping the mask): `.flatten()` (`methods.c`, a copy: mask
        flattened with the same order), `.diagonal()` (`item_selection.c`,
        a view: same offset/axes on the mask), `.real`/`.imag` of complex
        arrays (`getset.c` `_get_part`, view: shares the mask) and `.imag`
        of a non-complex array (zeros with a copied mask).
  - [x] Tests: `TestMaskCoreTransport` in `test_mask.py`. Leak check (RSS +
        refcount, 30000 iterations each, error paths included): flat.
  - **Still dropping the mask after phase 5 (not phase 5's scope, decided
    per case):**
    - `.flat[...]` / `flatiter.copy()` (`iterators.c`) -> phase 6 (indexing).
    - `np.lib.stride_tricks.as_strided` / `sliding_window_view`: they build a
      new array from raw strides via `__array_interface__`, so there is no
      principled mask transform. Currently drops silently; needs a decision
      (raise for masked input, or document as convention).
- [ ] **6 — Indexing**
  - [ ] `multiarray/mapping.c` — basic indexing already covered in phase 2;
        advanced/fancy + boolean indexing (copy — build new mask
        explicitly), assignment through indexing
- [ ] **7 — Gather, scatter, and combine/split**
  - [ ] Depends on phase 6's index-mapping and assignment semantics.
  - [ ] `npysort/*.c.src`, `multiarray/item_selection.c`
        (`searchsorted`, `take`/`put`/`choose`/`repeat`)
  - [ ] `multiarray/multiarraymodule.c` (`concatenate`, split/stack APIs)
- [ ] **8 — Casting**
  - [ ] `multiarray/convert_datatype.c`, `multiarray/convert.c`
  - [ ] `astype`, `PyArray_CastToType` and `np.array/asarray(dtype=...)`
        already carry the mask (done in phase 5). Remaining: audit the other
        casting entry points (`can_cast`/`result_type` are mask-blind by
        design; `view` on structured/subarray dtypes; `astype` to a subarray
        dtype currently raises a shape-mismatch error instead of guessing)
        before implementing linear algebra.
- [ ] **9 — Linear algebra**
  - [ ] Decide whether masked `matmul`/LAPACK input is rejected, ignored, or
        propagated before changing the implementation.
  - [ ] `umath/matmul.c.src`
  - [ ] `numpy/linalg/umath_linalg.c.src`
- [ ] **10 — Python-level surface**
  - [ ] `numpy/_core/arrayprint.py` (repr/str show masked cells)
  - [ ] `numpy/lib/_arraysetops_impl.py` (`unique`/`isin` mask-awareness)
  - [ ] `multiarray/methods.c` (`__reduce__`/pickle, `tobytes`/`tofile`)
  - [ ] `multiarray/buffer.c` (buffer protocol — decide: expose data only,
        or refuse when masked)

No separate benchmarking or final testing phase is tracked here. The
cross-test requirement in the housekeeping section is the per-phase gate:
after each implementation step, rerun the relevant regression checks and the
plain-array baseline to confirm no masked-path or no-mask regression.

## Note / known limitation

<details>
<summary>Performance backlog (not an active phase)</summary>

This is a low-priority performance backlog, not part of phase 5's actual scope.

The current design throughout phases 1-4 is exactly:

```
data
  ↓
existing NumPy machinery
  ↓
correct result

mask
  ↓
existing NumPy machinery
  ↓
correct mask
```

In other words, every mask operation reuses an already-correct, tested piece
of NumPy (`logical_or.reduce`, `logical_or.accumulate`, `logical_or.at`,
`PyNumber_Or`, `PyArray_Where`, and so on) instead of hand-rolled loops.
This makes phases 1-4 easy to trust and cheap to leak-check.

Every item below trades that simplicity for speed by fusing mask logic into
shared, delicate, correctness-critical NumPy internals (`NpyIter`,
`PyArrayMapIterObject`, SIMD, and scalar dtype loops). This is not worth doing
until the feature is functionally complete and there is a measured need.
The earlier attempt to optimize `PyArray_Std` instead of the Python method
path is a concrete example of the risk. Revisit these items only after phases
5-10 are complete, not opportunistically mid-phase.

- **P0 `add.at` (~+54-170% overhead, worst in the table):** Remove the
  second `ufunc_at()` pass by fusing mask OR into the existing index loops in
  `ufunc_at__fast_iter`, `ufunc_at__slow_iter`, and `trivial_at_loop`.
  This is the highest-risk item because it touches shared `.at()` machinery,
  `PyArrayMapIterObject`, overlap-copy handling, and buffering.
- **P1 `add.accumulate` (~+53-71%):** Fuse the running mask OR into
  `PyUFunc_Accumulate` instead of calling `logical_or.accumulate()` separately.
  A validated 1-D fast path was reverted so this work can stay consolidated
  with the other high-risk optimizations.
- **P2 `.std()`/`.var()` (~+33-320%):** The earlier optimization targeted
  `PyArray_Std`, but Python's `ndarray.std()` and `ndarray.var()` use
  `numpy/_core/_methods.py`. It built cleanly but did not improve the real
  benchmark and was reverted.
  The correct future approach is to compute
  `logical_or.reduce(arr.mask, axis, where=where)` once, run the existing
  arithmetic chain with the mask stripped, and reattach the final mask.
  Handle `where=`, `keepdims`, `mean`, and complex-valued branches carefully.
- **P3 elementwise `add`/`less` (~+13-65%):** Fuse propagation into the
  SIMD or scalar inner loop in `umath/loops*.c.src` instead of calling
  `PyNumber_Or` after `ufunc_generic_fastcall` has produced the result.
  This is the real AVX-512 masked-loop work reserved for a dedicated
  optimization effort.
- **P4 `add.reduceat`/`add.outer` (~+7-103%):** Profile before choosing a
  direction. Determine whether the cost comes from the OR-mask pass or the
  surrounding dispatch machinery; do not optimize this by guesswork.

</details>