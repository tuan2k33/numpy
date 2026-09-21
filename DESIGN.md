# Design — native mask support in `ndarray` core

Design rules and decisions. The roadmap, validation record and known gaps are
in [`TODO.md`](TODO.md).

## Scope: which problem this solves

Two different usecases get called "masked array":

1. **Genuinely missing data** — the value never existed (R/SQL `NULL`). The
   sibling `nulldtype` project solves this: hiding a value *destroys* it.
2. **Complete data, selectively hidden** — the value is real and known, just
   excluded from a view or computation: access control (redact a column for
   one audience), algorithms that vary which elements are visible (dropout,
   k-fold CV, subsampling).

**This fork targets case 2.** Hiding is non-destructive and reversible, no value
range of the dtype is sacrificed, and because `mask` lives on the array
*object*, several views of one buffer can carry *different* masks at once with
no data copy.

## Representation and invariants

- `PyArrayObject` gets one field, `PyArrayObject *mask`, `NULL` by default
  (same self-referential pattern as `base`), refcounted like `base`.
- The mask is a **plain bool `ndarray`** (a byte per element), so it inherits
  NumPy's view/stride/broadcast machinery; it always has exactly the owner's
  shape.
- `mask->mask` is always `NULL` (no masked masks). Enforced by *usage*, not by
  dtype: comparison results are bool arrays that must still be maskable. The
  internal flag `NPY_ARRAY_IS_MASK` marks an array while it serves as some
  array's mask, and `PyArray_SetMaskObject` refuses to attach a mask to a
  flagged receiver (checking only the incoming object has a point-in-time
  hole). The flag is not refcounted: sharing the identical mask object between
  owners is an accepted edge case; per-owner sharing goes through `.view()`.
- **Unmasked representation:** `mask == NULL` is the canonical, and the fast,
  state of an unmasked array. External assignment (`arr.mask = m`,
  `PyArray_SetMaskObject`) turns an all-False `m` into `NULL` (a one-time scan
  paid only there); internal propagation keeps `NULL` when the inputs are
  unmasked and never scans just to canonicalize. Operations whose inputs are all
  unmasked produce `mask == NULL`.
- **Black-box rule:** the mask never influences computation. The original code
  path (ufunc loop, reduction, copy, ...) runs unmodified on the whole array,
  masked cells included (errors and warnings unaffected), and the result's mask
  is computed and attached afterwards. `mask == NULL` costs one NULL check per
  call, outside any loop.
- A hook that lives in Python code inside NumPy must read the built-in mask as
  `np.ndarray.mask.__get__(obj)`, never `obj.mask`: a subclass such as
  `numpy.ma.MaskedArray` defines its own, unrelated `mask`.

## Semantics by operation family

`mask[i] == True` means hidden. An output element is hidden iff a hidden input
element contributes to it.

- **Elementwise ufuncs:** OR of the input masks after normal broadcasting, on
  the exact output shape; multi-output, `out=` and in-place work. `where=` with
  an explicit `out=` is a selective write of both data and mask; with `out=None`
  or a multi-output ufunc propagation is skipped (fail-open). Unmasked inputs
  into a masked `out=` clear its stale mask.
- **Reductions** (`reduce`, `accumulate`, `reduceat`, `outer`, `at`, and
  everything built on them: `sum`, `mean`, `max`, `std`, ...): the identical
  call with `logical_or` on the masks, so axis/segment/identity logic is reused;
  `std`/`var` hide a whole axis if any element of it is hidden. `argmax`/
  `argmin` and every other index-returning operation (`argsort`, `searchsorted`,
  `nonzero`, ...) are mask-blind.
- **Views and shape operations:** the identical transformation applied to the
  mask (`view`, basic slicing, transpose, `swapaxes`, `broadcast_to`,
  `reshape`/`ravel`/`squeeze`, `diagonal`, `real`/`imag`, `as_strided`).
  Views share the mask buffer; `.copy()` deep-copies it. Rebinding
  `view.mask = m` affects only that view, in-place mutation of a shared mask is
  seen by every view sharing it, like data. Broadcast masks are read-only views.
  `as_strided` carries the mask only when the strides map onto whole elements
  (dropping is allowed, mis-mapping is not).
- **Sorting:** the permutation is applied to data and mask (comparison uses the
  real values). `partition` is done as a full sort when masked.
- **Casting:** `astype`, `np.array/asarray(..., dtype=)`, `ascontiguousarray`,
  `require`, `np.astype` copy the mask; a cast to a subarray dtype hides every
  sub-element of a hidden element; `np.array(a, ndmin=n)` prepends ones;
  `np.array([a, b, ...])` builds the mask from the masked leaves. `view(dtype)`
  of a masked array is allowed only for the same itemsize (mask shared),
  otherwise it raises and points at `astype()`. `can_cast`/`result_type` are
  mask-blind.
- **Indexing:** fancy/boolean getitem indexes the mask with the same index.
  `a[idx] = rhs` gives the assigned elements the *RHS's* masked-ness (assigning
  new data unhides them; a masked RHS into an unmasked array creates the mask).
  Assigning into an array that serves as a mask is plain data assignment. A
  read-only destination mask fails before the data is touched.
- **Gather/combine** (`take`, `repeat`, `choose`, `where`, `concatenate` and
  everything built on them): the identical operation on the masks. A data input
  that steers the result (`np.where`'s condition, `np.choose`'s selector) hides
  the result where it is hidden. **Scatter** (`put`, `putmask`, `place`,
  `copyto`, `fill`, `flat`/`real`/`imag` assignment) follows the assignment
  rule. Selection *parameters* (`where=`, `putmask`'s and `copyto`'s condition,
  `take`'s indices) are plain data; their masks are ignored.
- **Contractions** (`matmul`, `dot`, `inner`, `einsum`, `correlate`,
  `numpy.linalg`): computed on the whole array, kernels untouched; logic in
  `numpy/_core/_op_mask.py`. Matrix products: `any` over the contributing
  row/column; `einsum`/`correlate`: the same operation on counts of the masks;
  LAPACK-backed functions depend on the whole matrix, so a hidden cell hides
  that matrix's whole output (per matrix of a stack); `solve` also distinguishes
  the columns of `b`.
- **In-place shape changes** (`a.shape = ...`, `strides`, `dtype`,
  `resize`) are rejected for an array with a mask or serving as one; use
  `reshape`/`view`/`astype`/`np.resize`.
- **Structured dtypes:** one bool per record, not per field. Field views do not
  carry it.
- **Scalars (final decision):** anything that decays to a scalar (`a[i]`, full
  reductions, `det`, `norm`, `v @ v`, `vdot`) is a plain NumPy scalar with no
  mask. Keep a mask with `axis=`, `keepdims=True` or a 0-d `out=`. Masked
  elements are never turned into NaN or any other sentinel.

## Python surface

- `repr`/`str` print a hidden cell as `--` and never its value; hidden cells
  are left out of width/precision decisions.
- `ndarray.filled(fill_value=<default>)` is the one way to get replacement
  values: a new plain, unmasked copy, the array is never changed. Without an
  argument it fills with the dtype's NA pattern from
  [`LAYOUTS.md`](LAYOUTS.md); dtypes that table rejects (`object`,
  `StringDType`, unsized `S`/`U`/`V`) raise unless a value is passed.
- Pickle carries the mask as an extra state item (unmasked pickles unchanged).
- The buffer protocol, `tobytes`/`tofile`/`np.save` and `tolist` export the
  data only, hidden values included, no mask.
- `isin` hides an answer when a hidden value could have changed it; `unique`
  and the set operations are deliberately mask-blind (they flatten and
  de-duplicate, so no per-element relation exists).

## Relation to `numpy.ma`

A comparison of 34 typical `numpy.ma` use cases with the built-in mask found 29
reproducible. Creation helpers (`masked_where`, `masked_invalid`,
`masked_equal`), arithmetic, comparison, ufuncs, `where`, slicing, boolean and
fancy indexing, `concatenate`, `take`, assignment and `filled` behave the same.
Where `ma` *skips* masked values (`sum`, `mean`, `min`, `max`, `cumsum`, `count`,
`compressed`, `average`, `median`, `sort` with masked last) the built-in
propagates instead, so those are composed from `filled(neutral)` and the mask
(`a.filled(0).sum(axis)` with mask `a.mask.all(axis)`, ...). Not reproducible, by
design: `ma.masked` and masked scalars, automatic masking of invalid results
(`sqrt(-1)`), hard masks, per-field masks, a `fill_value` *attribute* (there is
only the `filled` default).

## Fail-open policy

An operation without mask support behaves like it does for a plain array: the
mask is dropped and nothing raises. Raising would break internal NumPy paths
that copy or view arrays. Every gap is listed in `TODO.md` ("Known
limitations"). An operation that does support masks must never silently produce
a *wrong* mask.

## Compatibility

- **Extensions precompiled against stock NumPy** keep working unmodified: the
  field is purely additive, appended to the struct, gated behind
  `NPY_2_7_API_VERSION` (newer than anything existing code requests), and
  `mask == NULL` behaviour is byte-identical to upstream (checked per phase by
  running the whole suite against a fresh upstream build).
- **Python users** need no rebuild: use the new API on this fork's NumPy.
- **C extensions touching `mask`** rebuild against this fork's headers.
  Accessor-based extensions stay source-compatible; old wheels reading
  deprecated struct fields directly are an accepted ABI risk, as with earlier
  struct additions.

## NEP alignment and performance

Ufunc integration follows NEP 43 (DType-based dispatch); CPU-specific loops must
reuse NumPy's existing dispatch and SIMD infrastructure (`NPY_CPU_DISPATCH`,
`common/simd`) rather than a parallel dispatcher; every phase checks the
relevant NEPs and current `main` before touching a shared dispatch, dtype,
iterator or array-API path. A fused fast path (AVX-512 masked loads for
contiguous masked data, generic strided fallback otherwise, a single dispatch
check in `umath/ufunc_object.c`) is the long-term performance idea; today's
wrapper design is correctness-first (see the performance backlog in `TODO.md`).
