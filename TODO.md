# TODO — native mask support in `ndarray` core

Research fork: give `ndarray` a built-in, opt-in `mask` so operators only need
`if (mask)` instead of `numpy.ma`'s per-function Python wrappers. Target
**numpy `main`/dev only** (currently `2.6.0.dev0`); no backport to released
branches. Track upstream by merging `origin/main` regularly (see the standing rule
below) rather than diverging long-term.

Design rules live in [`DESIGN.md`](DESIGN.md). This file is the working
roadmap and validation record.

## Housekeeping

**Standing rule — sync upstream (not a checkbox; applies at all times):**
at the start of every session/phase and before every push, run
`git fetch origin main` → fast-forward local `main` to `origin/main` →
merge `main` into the feature branch (stash uncommitted WIP first, pop
after) → rebuild and rerun the plain-array baseline → confirm
`git rev-list --count HEAD..origin/main` is `0`. Never sync from the
personal fork's `fork/main`, never push `main`, and note that
`fork/<branch>` being up to date says nothing about upstream. This is a
long-running fork of a fast-moving codebase, so drift compounds fast if
skipped. Every phase checklist below starts with a sync item as a reminder.

- [ ] After every phase implementation, cross-test (pipenv/devenv,
  mask/nomask), review [`DESIGN.md`](DESIGN.md), and check relevant NumPy
  NEPs/current `main` behavior for dispatch, dtype, iterator, and array API
  changes. Verify no regression, leakage, or plain-array behavior change.

## Regression baseline

`mask == NULL` must keep every existing code path byte-identical to
upstream. Tracking the actual pass counts here after each phase, not just
inside the phase's checklist, so drift is easy to spot at a glance:

Current focus:

- **Phase 6:** done (advanced/fancy and boolean indexing, assignment, `.flat`).
- **Phase 7:** done (gather/scatter/combine, in-place mutator guards).
- **Phase 8:** done (casting audit: `view(dtype)`, `np.array(ndmin=)`, `np.array([masked, ...])`, subarray `astype`).
- **Phase 9:** done (contracting ops and `numpy.linalg`).
- **Phase 10:** done (repr/str, `ndarray.filled()`, `isin`, pickle, buffer/IO decisions; `unique` stays mask-blind by decision).
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
  - `test_multiarray.py` + `test_indexing.py` (`-m "not slow"`): 14937
    passed, 17 skipped, 12 deselected, 0 failed (measured after merging
    upstream `main` at `e5cae0a6e2`).
  - Why this is higher than the 14916 in the rows above: those were measured
    before the 2026-09-18 `pull --rebase` onto a newer upstream base, which
    brought upstream's own new tests into `test_multiarray.py` (16 more at
    `e765e6725e`, 5 more from the 20 commits merged afterwards). None are
    ours; no new skips or failures. Re-baseline against 14937 from here on.
  - Plus `test_umath.py`, `test_ufunc.py`, `test_shape_base.py`,
    `numpy/lib/tests/test_stride_tricks.py`: 6016 passed, 60 skipped,
    7 xfailed (pre-existing xfails), 0 failed.
  - `test_mask.py`: all phase-5 tests pass (the phase-6 WIP tests in the same
    file are excluded from this gate).
  - Match with the plain-array baseline: ✅ no regression.

- **Phase 6 — advanced/boolean indexing, assignment, `.flat`**
  - `test_multiarray.py` + `test_indexing.py` (`-m "not slow"`): 14937
    passed, 17 skipped, 12 deselected, 0 failed — ✅ identical to the phase 5
    row.
  - Plus `test_umath.py`, `test_ufunc.py`, `test_shape_base.py`,
    `test_stride_tricks.py`, `test_regression.py`, `test_item_selection.py`:
    21668 passed, 78 skipped, 12 deselected, 7 xfailed in one combined run
    (superset of the phase 5 groups), 0 failed.
  - `test_mask.py`: 114 passed.

- **Phase 7 — gather, scatter, combine, in-place mutator guards**
  - `test_multiarray.py` + `test_indexing.py` (`-m "not slow"`): 14937
    passed, 17 skipped, 12 deselected, 0 failed — ✅ identical to phases 5/6.
  - `test_umath.py`, `test_ufunc.py`, `test_shape_base.py`,
    `test_stride_tricks.py`: 6016 passed, 60 skipped, 7 xfailed — ✅ identical.
  - Widened gate (adds `test_regression`, `test_item_selection` and
    `numpy/lib/tests/{test_function_base,test_arraysetops,test_index_tricks,
    test_shape_base,test_twodim_base}.py`, all of which exercise
    take/put/where/concatenate heavily): 23466 passed, 151 skipped,
    15 deselected, 7 xfailed, 0 failed.
  - `test_mask.py`: 148 passed.

- **Phase 8 — casting audit (`view(dtype)`, `ndmin`, list-of-masked, subarray `astype`)**
  - Merged upstream `a5028e8c2c` first (behind 0).
  - `test_multiarray.py` + `test_indexing.py` (`-m "not slow"`): 14937
    passed, 17 skipped, 0 failed — ✅ identical to phases 5–7.
  - `test_umath.py`, `test_ufunc.py`, `test_shape_base.py`,
    `test_stride_tricks.py`: 6016 passed, 60 skipped, 7 xfailed — ✅ identical.
  - Widened gate (phase 7 set): 23466 passed, 151 skipped, 7 xfailed,
    0 failed — ✅ identical.
  - **Gate widened again to the whole suite** (`pytest numpy -m "not slow"`),
    because `numpy/ma` and `test_public_api` were not in the earlier gate and
    caught two phase 5 bugs (ma `.mask` collision in `_stride_tricks_impl.py`,
    stolen `@set_module` decorator on `as_strided`): 49348+ passed, 0 failed
    once run with `PYTHONPATH=<build-install dist-packages>` (the
    `test_cpu_features`/`test_limited_api`/`test_cython` tests spawn
    subprocesses that need it; without it they fail with "No module named
    numpy", which is environment, not a regression).
  - `test_mask.py`: 177 passed.

- **Phase 9 — linear algebra (matmul/dot/inner/einsum/correlate/linalg)**
  - Merged upstream first (behind 0).
  - `test_multiarray.py` + `test_indexing.py` (`-m "not slow"`): 14937
    passed, 17 skipped — ✅ identical to phases 5–8.
  - `test_umath.py`, `test_ufunc.py`, `test_shape_base.py`,
    `test_stride_tricks.py`: 6016 passed, 60 skipped, 7 xfailed — ✅ identical.
  - Widened gate: 23466 passed, 151 skipped, 7 xfailed — ✅ identical.
  - `numpy/linalg`: 520 passed, 1 skipped, 1 xfailed.
  - Whole suite (`pytest numpy -m "not slow" -n 4`, with `PYTHONPATH`):
    49419 passed, 1051 skipped, 57 xfailed, 1 xpassed, 0 failed.
  - `test_mask.py`: 208 passed.

- **Phase 10 — Python surface (repr, `filled`, pickle, `isin`)**
  - Merged upstream first (8 commits, clean; behind 0).
  - Whole suite (`pytest numpy -m "not slow" -n 4`, with `PYTHONPATH`):
    49472 passed, 1051 skipped, 57 xfailed, 1 xpassed, 0 failed.
    Without `numpy/ma`: 45038 passed, 1051 skipped, 55 xfailed, 1 xpassed,
    0 failed. (First run had 20 failures: the new `ndarray.filled` was picked
    up by `np.ma.filled`'s `hasattr(a, 'filled')` duck-typing and by the
    matrix "call every method" test; fixed in `ma/core.py` and
    `test_defmatrix.py`.)
  - `test_mask.py`: 261 passed.
  - The per-file baseline rows (multiarray/indexing, umath group) are
    subsumed by the whole-suite run.
  - **Cross-run against upstream** (2026-09-22): the same test tree run on a
    fresh build of `origin/main` @ 3526562baa (git worktree, `spin build`)
    and on this branch, `pytest numpy -m "not slow" -n 6`, per-test outcomes
    compared through junit XML. Upstream 49210 passed / 1052 skipped / 57
    xfailed / 1 xpassed; fork 49471 passed (the +261 are exactly the
    `test_mask.py` tests, nothing else) with identical skipped/xfailed/
    xpassed counts. Common tests: 50320, **0 status differences**, 0 tests
    missing on either side. Plain arrays behave identically to upstream.

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
        masked array is refined in phase 8 (same itemsize shares the mask,
        other itemsizes raise a clear error pointing at `astype()`). Unmasked
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
  - [x] `as_strided` (`numpy/lib/_stride_tricks_impl.py`): carries the mask
        by re-running `as_strided` on the mask with the byte strides divided
        by `itemsize`. Valid only when every view stride is a whole number of
        elements and the mask layout is proportional to the data layout;
        otherwise fail-open (see "Known limitations"). `sliding_window_view`
        calls `as_strided`, so it inherits this. `PyArray_CopyMaskFrom` lays
        the copied mask out like the destination data (same stride order) so
        the proportional case is the common one (`astype(order='F')`,
        `asfortranarray`, ...).
  - Remaining mask drops after phase 5 are tracked under "Known limitations
    (fail-open)" at the bottom of this file.
- [x] **6 — Indexing**
  - [x] Sync upstream first (standing rule in Housekeeping): `origin/main`
        merged, `HEAD..origin/main` is 0.
  - [x] `multiarray/mapping.c` getitem: advanced/fancy and boolean indexing
        return a copy, so the mask is copied the same way — `array_subscript`
        indexes the mask with the identical index object. The hook sits after
        `finish:` so it covers the single-boolean fast path, the 1-d fancy
        fast path and the general MapIter path (basic indexing already
        attached a mask view in phase 2). An all-False selection
        canonicalizes to `mask is None`.
  - [x] Assignment (`array_assign_mask_subscript`, `array_assign_item`):
        after the data assignment succeeds, the same assignment runs on the
        mask with the RHS's mask (`False` for an unmasked RHS); a masked RHS
        assigned into an unmasked array creates the mask. Convention: the
        assigned elements take the *RHS's* masked-ness (assigning new data
        unhides those elements). Assigning into an array that is itself a
        mask stays plain (a mask cannot carry a mask). If the destination's
        mask is read-only the assignment fails *before* touching the data
        instead of half-applying. Per the "Unmasked representation" rules,
        an internal assignment that leaves the mask all-False does not
        re-scan to canonicalize it to `None`.
  - [x] Dependency pulled forward from phase 7's `.flat`: `a.flat[...]` get
        and set (`iter_subscript`/`iter_ass_subscript` wrappers in
        `iterators.c`, applying the identical C-order flat index to the mask's
        flat iterator), `flatiter.copy()` and `np.asarray(a.flat)`.
  - [x] Fixed a reference leak in the WIP `success:` cleanup path of
        `array_assign_subscript` (the mask update now runs after all
        temporaries are released).
  - [x] Tests: `TestMaskIndexing` in `test_mask.py` (getitem, setitem,
        overlap/swap, read-only mask, in-place ops through an index,
        `.flat`). A 4000-case randomized differential check of get + set
        against a plain-array reference model found 0 mismatches. Leak check
        (RSS + refcount of both the destination and the RHS mask, 40000
        iterations each, error paths included): flat.
  - **Still not carried after phase 6** (fail-open, see "Known limitations"):
    structured dtypes (mask is per record; field views and field assignment
    do not carry/update it — see the table); `a.flat = v` (whole-array flat
    assignment); scalar results (`a[i]`, `a.flat[i]`) cannot carry a mask.
- [x] **7 — Gather, scatter, and combine/split**
  - [x] Sync upstream first (standing rule in Housekeeping): `origin/main`
        merged, `HEAD..origin/main` is 0.
  - [x] **Memory-safety fix found while probing:** in-place `a.shape = ...`,
        `a.strides = ...`, `a.dtype = ...` and `a.resize(...)` left the mask
        with the old shape/dtype (mask shape != array shape — the invariant
        every propagation path relies on). They now raise `ValueError` for an
        array that has a mask *or is serving as another array's mask*
        (`PyArray_FailIfMaskedInPlace`: shape/strides/dtype setters in
        `getset.c`, `PyArray_Resize_int` in `shape.c`). These setters are
        deprecated upstream (2.4/2.5); unmasked arrays are untouched.
  - [x] **Phase 3 follow-up:** `np.add(x, 1, out=o)` with unmasked inputs
        left `o`'s stale mask in place although `o` was fully overwritten.
        `_propagate_ufunc_result_mask` now treats "no masked input but a
        masked `out`" as an all-False result mask (so it clears, and merges
        with `where=`).
  - [x] Gather (`item_selection.c`): `PyArray_TakeFrom` (`take`, `compress`,
        `extract`, `take_along_axis`), `PyArray_Repeat` (`repeat`, `tile`),
        `PyArray_Choose` — the identical operation runs on the masks (same
        indices/axis/mode). `out=` gets the new mask; a stale mask on `out`
        is dropped when nothing masked was gathered. `np.choose`: result mask
        = mask of the chosen element OR the selector's own mask.
  - [x] `np.where(cond, x, y)` (`multiarraymodule.c`): mask = `where(cond,
        x.mask, y.mask)` OR the condition's own mask (a hidden condition hides
        the result); operands' shapes are broadcast like the data. Also gives
        `tril`/`triu`/`select`/`diag`-style helpers for free.
        `PyArray_WhereNoMask` is the mask-free core, used internally to merge
        masks (which cannot carry masks).
  - [x] Combine: `PyArray_ConcatenateInto` concatenates the masks (all-False
        for unmasked inputs) with the same axis; covers `axis=None`, `dtype=`,
        `out=` and, through them, `stack`/`vstack`/`hstack`/`dstack`/
        `column_stack`/`block`/`append`. `split`/`array_split`/`hsplit` are
        views, already correct from phases 2 and 5.
  - [x] Scatter — assigned elements take the masked-ness of the values (same
        convention as phase 6 indexing), by running the identical operation on
        the mask: `put` (`PyArray_PutTo`), `putmask` (`PyArray_PutMask`),
        `place` (`compiled_base.c`), `copyto` incl. `where=`
        (`multiarraymodule.c`), `fill`, `a.flat = v`, `a.real = v` /
        `a.imag = v`, and `fill_diagonal` via `.flat`. A masked value into an
        unmasked destination creates the mask. Shared helpers:
        `PyArray_MaskForUpdate` / `PyArray_FinishMaskUpdate` /
        `PyArray_FailUnlessMaskWriteable` (read-only masks fail before the
        data is touched).
  - [x] Convention: index-returning operations (`argsort`, `argpartition`,
        `lexsort`, `searchsorted`, `nonzero`/`argwhere`, and phase 4's
        `argmax`/`argmin`) are mask-blind — indices carry no mask and the
        mask never changes which index is returned. `npysort/*.c.src`
        therefore needed no change (`sort` was done in phase 2).
  - [x] Convention: selection *parameters* — `where=` of a ufunc, `np.putmask`'s
        and `np.copyto`'s condition, `take`'s indices — are plain data: their
        own masks are ignored. Data *inputs* that steer the result
        (`np.where`'s condition, `np.choose`'s selector) propagate their mask.
  - [x] Tests: `TestMaskGatherScatter` in `test_mask.py`. A 3000-case
        randomized differential check (each element tagged with a unique id,
        expected masks derived from the ids) over `take` (all modes/axes),
        `repeat`, `concatenate`, `stack`, `where`, `choose`, `put` and
        `copyto(where=)` found 0 mismatches. Leak check (RSS + refcounts of
        three live masks, 20000 iterations, 34 operations including error
        paths and `out=`): flat.
- [x] **8 — Casting**
  - [x] Sync upstream first (standing rule in Housekeeping): `origin/main`
        merged (`a5028e8c2c`), `HEAD..origin/main` was 0 before the work.
  - [x] Audit of every cast entry point (`probe8.py`-style sweep over
        `astype` with float/object/str/bool/complex/structured targets and
        `order`/`subok`/`casting`/`copy` variants, `np.array/asarray/
        asanyarray/ascontiguousarray/asfortranarray/require/astype/copy`,
        `__array__`, `copy.copy/deepcopy`, scalar constructors): all carried
        the mask already (phase 5) except the four below.
  - [x] `view(dtype)` (`convert.c`): a masked array may now be viewed as a
        dtype of the **same itemsize** (the shape is unchanged, so the mask
        is shared as a view, incl. structured targets); another itemsize
        changes the last axis and still raises (`astype()` instead). Also
        fixed a `type` reference leak on that error path.
  - [x] `np.array(a, ndmin=n)` (`multiarraymodule.c` `_prepend_ones`): the
        mask gets the same leading ones (a view when the data is a view).
  - [x] `np.array([a, b, ...])`, nested lists/tuples, with/without `dtype=`
        (`ctors.c`, `PyArray_FromAny` sequence path): if any leaf array is
        masked, a mask is filled alongside the data
        (`_assign_from_cache_masked`); unmasked leaves stay False; an
        all-False result stays unmasked.
  - [x] `astype` to a subarray dtype (`PyArray_CopyMaskFrom` generalised):
        the mask is broadcast over the appended dimensions (hidden element
        ⇒ every sub-element hidden). No longer a `ValueError`.
  - [x] Found by widening the gate to `numpy/ma` (see below): the phase 5
        Python code in `_stride_tricks_impl.py` read `array.mask`, which on a
        `MaskedArray`/`mvoid` is *ma's* mask, so `np.broadcast_to(ma_or_mvoid,
        ..., subok=True)` failed (`mvoid`) or silently re-set ma's mask. It
        now goes through the `np.ndarray.mask` descriptor (`_get_mask`/
        `_set_mask`); DESIGN.md records the rule.
  - [x] Tests: `TestMaskCasting`, `TestMaskSubclassNamespace` in
        `test_mask.py` (177 total); randomized differential check of
        `np.array(list)`/`ndmin`/subarray against masks built with NumPy,
        0 mismatches; RSS + refcount leak check (20000 iterations incl.
        error paths): flat.
- [x] **9 — Linear algebra**
  - [x] Sync upstream first (standing rule in Housekeeping): `origin/main`
        merged, `HEAD..origin/main` was 0 before the work.
  - [x] Decision (user): follow the existing convention — the whole array is
        computed as usual (masked cells included), the mask follows. No
        rejecting, no changes to `matmul.c.src`/`umath_linalg.c.src` (the
        kernels stay byte-identical). Rule: an output element is hidden iff a
        hidden input element *contributes* to it.
  - [x] Mask logic lives in `numpy/_core/_op_mask.py` (Python; only reached when
        an input or `out=` carries a mask). C hooks call it through
        `PyArray_PropagateOpMask` (`arrayobject.c`): `ufunc_generic_vectorcall`
        for gufuncs, and wrappers (`*_data` originals kept unmodified) around
        `PyArray_MatrixProduct2` (`dot`/`ndarray.dot`), `PyArray_InnerProduct`,
        `PyArray_Correlate`/`Correlate2` (`correlate`/`convolve`) and
        `array_einsum`. `tensordot`, `matrix_power`, `multi_dot`, `pinv`, ...
        need nothing: they are built from the above.
  - [x] Exact rules: `matmul` (`ra[..., :, None] | cb[..., None, :]` with
        `ra = any(mask_a, -1)`, `cb = any(mask_b, -2)`; 1-d operands drop that
        axis), `matvec`/`vecmat`/`vecdot`, `dot`, `inner`; `einsum` and
        `correlate`/`convolve` run the very same operation on the masks (a
        float32 count, `> 0`, per masked operand, others replaced by ones), so
        subscripts/ellipsis/`optimize=`/modes come for free.
  - [x] `numpy.linalg` gufuncs (`inv`, `eigh`/`eigvalsh`, `eig`, `svd`,
        `cholesky`, `qr`, `lstsq`, `det`, ...): conservative generic rule from
        the gufunc signature — every output element of a matrix is hidden if
        *any* element of that matrix is (each output depends on the whole
        input matrix), per matrix of a stack. `solve` is finer: hidden by
        column of `b`, but wholly hidden by any hidden element of `a`.
        `qr`: NumPy runs `qr_r_raw` in place on a copy of `a` and reads R from
        it, so that gufunc also updates `a`'s mask; R's strictly lower
        triangle is structurally zero and stays visible.
  - [x] Bugs found on the way: `multiply(a1, b2, out=cp0)` with 0-d masked
        operands raised `TypeError: mask must be an ndarray of dtype bool`
        (`np.cross` on masked vectors): or-ing two 0-d masks returns a numpy
        bool scalar. New `PyArray_MaskOr` (always an ndarray) replaces every
        `PyNumber_Or` on masks (ufunc, `where`, `choose`).
  - [x] Tests: `TestMaskLinalg` (brute-force reference for matmul incl.
        batch/broadcast/1-d, dot for six shape pairs, inner, tensordot,
        einsum incl. list form/`optimize`/`out`, correlate/convolve all modes,
        every linalg function above, stacks, subclass-with-own-`mask`).
        RSS + refcount leak check (3000 iterations incl. error paths): flat.
- [x] **10 — Python-level surface**
  - [x] Sync upstream first (standing rule in Housekeeping): `origin/main`
        merged (8 commits, clean), `HEAD..origin/main` is 0.
  - [x] `numpy/_core/arrayprint.py` (repr/str show masked cells): a hidden
        cell prints as `--` (right-aligned to the width of the visible
        cells), the hidden *value* is never printed, and it takes no part in
        the width/precision decisions (formatters are built from the visible
        cells only). Works for every dtype incl. structured/datetime/object/
        string, summarized output, 0-d (`array(--)`, `str` -> `--`), all-hidden
        and empty arrays, `array2string` options/`formatter` (hidden cells
        bypass user formatters), subclasses (`np.matrix`). Unmasked arrays take
        the old path untouched. Read through the base-class descriptor
        (`ndarray.mask.__get__`), so `numpy.ma` is unaffected.
  - [x] Explicit `filled(fill_value)`: new C method `ndarray.filled` in
        `methods.c` (`array_filled`): a new plain, unmasked, writable copy
        with hidden cells assigned `fill_value` (same conversion as
        `arr[i] = fill_value`, so `a.filled(np.nan)` on an int array raises;
        the value is validated even when nothing is hidden). Never a view,
        keeps the subclass and layout, `fill_value` is required (no implicit
        default). Docs in `_add_newdocs.py`, stub in `__init__.pyi`.
  - [x] `numpy/lib/_arraysetops_impl.py`: `isin` is mask-aware (rule below).
        `unique` and the set operations built on it (`intersect1d`,
        `setxor1d`, `union1d`, `setdiff1d`, `unique_*`) stay **mask-blind by
        decision** (user, 2026-09-21): they run on the whole data and return
        plain arrays; hidden values can show up in the result.
  - [x] `multiarray/methods.c` (`__reduce__`/pickle): a masked array pickles
        its mask as a sixth state item (unmasked arrays keep the exact
        5-item state, so their pickles are byte-identical and readable by any
        NumPy; a *masked* pickle needs this build). `__setstate__` validates
        the mask (bool ndarray of the array's shape, no mask of its own) and
        replaces any mask already on the object. Protocol 5 with a masked
        array uses the regular (in-band) reduce, since the `PickleBuffer`
        reconstruction has nowhere to carry a mask. Fixed while testing: the
        first failed `PyArg_ParseTuple` left its exception set when the
        fallback parse succeeded (`SystemError` on every 5-item state).
  - [x] `multiarray/buffer.c`, `tobytes`/`tofile`/`np.save`/`savetxt`/
        `tolist`: **decided (user): export the data only** — the raw values of
        hidden cells included, no mask — nothing to change in the code
        (fail-open, keeps `frombuffer`/`PickleBuffer`/third-party buffer
        consumers working). Use `filled()` first to export replacement values.
        `test_buffer_and_io_export_data_only` pins this.
  - Verification: see the phase 10 row in "Regression baseline"; RSS +
    refcount loop over pickle/unpickle (incl. rejected masks), `filled`
    (object marker), repr/str, `isin` (3300 iterations): flat.

No separate benchmarking or final testing phase is tracked here. The
cross-test requirement in the housekeeping section is the per-phase gate:
after each implementation step, rerun the relevant regression checks and the
plain-array baseline to confirm no masked-path or no-mask regression.

## Note / known limitation

### Known limitations (fail-open) — audited at the close of phase 10

Policy (see `DESIGN.md`): an operation without mask support behaves exactly
like it does for a plain array — the mask is dropped, nothing raises. This is
deliberate: the priority is that every op keeps working with no wrong result
or regression. Every such gap must be listed here so it can be audited later.

| Operation | What happens | Planned |
|---|---|---|
| `ndarray.setfield` and structured-field writes | data assigned, record mask unchanged (same per-record limitation as below) | out of scope |
| `out=` (ufunc, take, choose, concatenate) and masked-value assignment through a view whose own mask is `None` | the mask is *replaced* on the `out` object, so other views still holding the old mask object keep it; and a view with no mask (all-False slice) cannot write masked-ness back to its parent's mask | revisit (write into the existing mask buffer instead of replacing it) |
| structured dtypes (fields) | The mask is **per record** (one bool per array element), not per field; per-field masking (what `numpy.ma` does with a structured mask dtype) is out of scope. Consequences: field views `s["x"]` / `s[["x","y"]]` do **not** carry the record mask (a masked record is visible through the field view); `s["x"] = v` assigns the data and leaves the record mask unchanged. Whole-record operations (`s[i]` fancy/bool/slice, copy, sort, ...) work normally. | out of scope (revisit only if structured masking matters) |
| assignment to a broadcast view with a read-only mask (`np.broadcast_arrays` results: data warns, mask is read-only) | fails before touching data | revisit |
| `as_strided`/`sliding_window_view` with a stride that is not a whole number of elements, or a mask layout not proportional to the data | mask dropped | revisit (could conform the mask copy to the data layout) |
| `a[...] = [masked, ...]` (list/tuple RHS with masked arrays in a slice/int/ellipsis assignment) | data assigned, masks inside the list ignored (that path is the public `PyArray_CopyObject`, which has no mask hook). `a[...] = np.array([masked, ...])` carries them | revisit (route through `PyArray_FromAny` when the destination or a leaf is masked) |
| `np.frombuffer`, `getfield`, `__array_wrap__(x)` | result has no mask (the buffer/field view has no owner-mask relation) | by design; `getfield` follows the structured limitation |
| Buffer protocol, `tobytes`, `tofile`, `np.save`/`savetxt`, `tolist`, `memoryview` of a masked array | **Decided (user, phase 10): data only.** The raw data, hidden cells' values included, is exported and no mask travels with it (a saved/loaded array is unmasked). Pickle *does* carry the mask; use `filled()` before exporting replacement values | by design |
| `np.unique` (+ `return_index/inverse/counts`), `unique_*`, `intersect1d`, `setxor1d`, `union1d`, `setdiff1d` | **Decided (user, phase 10): mask-blind.** They flatten and de-duplicate, so no per-element mask relation exists; they run on the whole data and return plain arrays, hidden values included | by design |
| `np.isin(element, test_elements)` | mask follows the rule "hidden iff a hidden input can change the answer": hidden where `element` is hidden, and where `element` matches no *visible* test value while `test_elements` has a hidden cell (a hidden cell might have matched). A match against a visible value is a certain hit. Hidden test values are never compared as if they were known | none |
| Pickle of a masked array | 6-item `__setstate__` state (the sixth is the mask): reading it needs this build (older NumPy errors instead of silently dropping the mask); protocol 5 uses the in-band reduce for masked arrays, so no out-of-band buffers | by design |
| `view(dtype)` with a different itemsize on a masked array | raises `ValueError` (the last axis changes; no mask view describes it) | by design; `astype()` |
| `np.array(list_of_masked, dtype=object, ndmax=k)` with `ndmax` smaller than the depth | the masked arrays become *object elements* of the result, so the result has no mask (by construction, nothing per-element to mask); without `ndmax`, `dtype=object` carries the mask | by design |
| 0-d/scalar results (`np.add.reduce(x)`, `x.sum()`, `x.max()`, `x[0]`, `.flat[i]`, elementwise ufuncs on 0-d, ...) | **Convention (decided): a scalar carries no mask.** NumPy decays a 0-d result to a scalar and scalars cannot carry one, so the mask is dropped there. Keep it with `axis=`, `keepdims=True` or `out=` (a 0-d `out` array keeps its mask). Reduce path: `PyUFunc_Reduce` returns a 0-d ndarray, `_propagate_reduce_mask` attaches the mask to it, then `npy_apply_wrap(..., return_scalar)` decays it to a scalar. Elementwise path: the 0-d result is decayed inside `ufunc_generic_fastcall`, before mask propagation runs. Not chosen: NaN-for-masked (float-only, destroys the value, conflates masked with invalid/missing); retaining a 0-d result (would be a one-line `return_scalar &= mask == NULL` on the reduce path plus a re-wrap on the elementwise path) | **Final (user decision, 2026-09-21, after phase 9): every scalar-producing operation stays a plain scalar and loses the mask** — reductions, `x[i]`, `det`/`norm`/`trace`, `v @ v`, `vdot`, ... No 0-d retention, no masked scalar type, no NaN sentinel. Consequence to remember: e.g. `np.linalg.det(m)` of a matrix with a hidden cell returns an ordinary number. Use `axis=`/`keepdims=True`/`out=` to keep a mask, or `filled(fill_value)` (phase 10) to get an explicit plain array. Not to be re-opened unless the user asks |
| where= with `out=None` or multi-output ufuncs | mask propagation skipped | revisit |
| Scalar results of contracting ops (`v @ v`, `np.vdot`, `np.linalg.det`/`slogdet`/`norm`/`trace` of one matrix, 1-d `dot`/`inner`/`einsum(..., '->')`) | scalar convention (final, see the 0-d/scalar row): the mask is dropped, so `det` of a matrix with a hidden cell is a plain number. A stack (`det(stack)`) keeps it | none (decided) |
| Coarse linalg masks | every output of `inv`/`eig`/`svd`/`cholesky`/`qr`/`lstsq`... is hidden for the whole matrix if any element of it is hidden (an over-approximation of the true dependence for eigen-/singular-vectors is impossible to refine without knowing the algorithm) | by design |
| `vdot`, `np.cross`/any `out=` into a *view of an unmasked array* (`multiply(a1, b2, out=cp[..., 0])`) | `vdot` returns a scalar; the view's own mask cannot reach the parent (see the `out=` row above), so `np.cross` of masked vectors returns the right data with no mask | revisit with the `out=` row |

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