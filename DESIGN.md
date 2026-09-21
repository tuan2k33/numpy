# Design — native mask support in `ndarray` core

This document contains the design rules moved out of `TODO.md`. The TODO
file should contain the working roadmap; this file is the design reference.

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
testing explicitly across the core transport work (phases 2 and 5): construct
two views
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
- ndarray scalar results preserve their normal NumPy scalar type and carry
  mask state. Results that decay to a Python/NumPy scalar cannot carry a mask
  until scalar-mask support is designed.
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
- Transpose, axis moves, and `broadcast_to` apply the same shape and stride
  transformation to the mask. Broadcast masks are views and inherit the
  read-only behavior of the broadcast data view.
- Dtype-changing views of masked arrays are allowed only when the itemsize
  is unchanged (the shape is then unchanged and the mask is shared as a view);
  a different itemsize changes the last axis, which no view of the mask can
  describe, so it raises and points at `astype()`. Unmasked arrays retain
  NumPy's normal dtype-view behavior.
- Casting copies the mask like it copies the data (`astype`, `np.array`/
  `asarray`/`ascontiguousarray`/`require` with `dtype=`, `np.astype`); a
  cast to a subarray dtype (which appends dimensions) hides every
  sub-element of a hidden element. `np.array(a, ndmin=n)` prepends ones to
  the mask as it does to the data. `np.array([a, b, ...])` (nested
  lists/tuples of arrays) builds the result mask from the masked leaves;
  unmasked leaves are unmasked. `can_cast`/`result_type` are mask-blind.
- Python-level helpers inside NumPy must reach the built-in mask through
  `np.ndarray.mask` (base-class descriptor), never `obj.mask`: a subclass such
  as `numpy.ma.MaskedArray` defines its own, unrelated `mask`.

- Advanced/fancy and boolean indexing return copies, so the mask is copied
  too: the mask is indexed with the identical index object. Item assignment
  `a[idx] = rhs` makes the assigned elements take the *RHS's* masked-ness
  (unmasked RHS → unmasked; masked RHS into an unmasked array creates the
  mask): assigning new data unhides those elements. Assigning into an array
  that is itself serving as a mask is plain data assignment. If the
  destination's mask is read-only the assignment fails before the data is
  touched, never half-applied.

- Structured dtypes: the mask has one bool per array element, i.e. it masks
  whole *records*, not individual fields. Field views (`s["x"]`) currently do
  not carry it; see "Known limitations" in TODO.md.

- Scalars carry no mask (convention). Extracting a single element
  (`a[i]`) or reducing to a scalar (`a.sum()`) yields an ordinary NumPy
  scalar; the mask is metadata of arrays only. To keep it, use `axis=`,
  `keepdims=True` or a 0-d `out=` array. Masked elements are never turned into
  NaN or any other sentinel: hiding is non-destructive, and an explicit
  `filled()` (phase 10) is how a user asks for replacement values.

- Gather and combine operations (`take`, `repeat`, `choose`, `where`,
  `concatenate` and everything built on them) run the identical operation on
  the mask. Where a data input steers the result (`np.where`'s condition,
  `np.choose`'s selector) a hidden value hides the result. Scatter operations
  (`put`, `putmask`, `place`, `copyto`, `fill`, `flat`/`real`/`imag`
  assignment) follow the assignment rule: assigned elements take the
  masked-ness of the values.
- Index-returning operations (`argsort`, `searchsorted`, `nonzero`, `argmax`,
  ...) are mask-blind: indices carry no mask and the mask never changes which
  index is returned. Selection parameters (a ufunc's `where=`, `putmask`'s or
  `copyto`'s condition, `take`'s indices) are plain data whose own masks are
  ignored.
- The mask must always have exactly the array's shape. In-place changes to an
  array's shape, strides, dtype or size (`a.shape = ...`, `a.resize(...)`, ...)
  cannot honour that and are rejected for an array with a mask or serving as
  one; use `reshape`/`view`/`astype`/`np.resize`, which return new arrays.

## Fail-open policy for unsupported operations

While the feature is being built, an operation that has no mask support yet
behaves exactly like it does for a plain array: the result carries no mask
and nothing raises. This is deliberate — the first priority is that every
operation keeps working with no wrong result and no plain-array regression;
fail-closed (raising) would break internal NumPy code paths that copy or
view arrays. Every such gap is recorded in TODO.md under "Known limitations
(fail-open)" and must be audited before the work is considered complete.
Operations that do support masks must never silently produce a *wrong* mask —
dropping is allowed, mis-mapping is not (e.g. `as_strided` only carries the
mask when the byte strides map onto whole mask elements).

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

## Compatibility and API goal

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

The field addition is gated behind the newer feature/API version and appended
to the struct. Accessor-based extensions remain source-compatible. Old
precompiled wheels that directly access deprecated struct fields are an
accepted ABI risk, as with earlier NumPy struct additions.

## NEP alignment

- Ufunc integration follows NEP 43's DType-based dispatch model.
- CPU-specific loops must reuse NumPy's existing dispatch and SIMD
  infrastructure rather than introducing a parallel runtime dispatcher.
- Every phase must check the relevant NumPy NEPs and current `main` behavior
  before changing a shared dispatch, dtype, iterator, or array API path.
