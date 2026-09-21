# TODO — native mask support in `ndarray` core

Research fork: give `ndarray` a built-in, opt-in `mask` so operators only need
`if (mask)` instead of `numpy.ma`'s per-function Python wrappers. Target
**numpy `main`/dev only** (currently `2.6.0.dev0`), no backport. Design rules:
[`DESIGN.md`](DESIGN.md). Timings: [`BENCHMARKS.md`](BENCHMARKS.md). This file
is the roadmap, the validation record and the list of known gaps.

## Status

Phases 0–10 are done. Whatever is next needs a decision from the user first
(candidates: the `out=` mask-replacement gap, per-field masks, the
performance backlog at the bottom).

| Phase | Scope | Main files |
|---|---|---|
| 0 | baseline, branch `refactor/ndarray-mask` | — |
| 1 | `mask` field, `NPY_2_7_API_VERSION`, `PyArray_SetMaskObject`, `arr.mask` property | `ndarraytypes.h`, `arrayobject.c`, `getset.c`, `ctors.c` |
| 2 | copy/view/slice/reshape/ravel/squeeze/sort/partition | `convert.c`, `shape.c`, `mapping.c`, `item_selection.c` |
| 3 | elementwise ufuncs, `where=`, `out=`, multi-output | `umath/ufunc_object.c` |
| 4 | reduce/accumulate/reduceat/outer/at (+ `sum/mean/std/...` for free) | `umath/ufunc_object.c` |
| 5 | transpose, `broadcast_to`, `view`, casting, `flatten`/`diagonal`/`real`/`imag`, `as_strided` | `shape.c`, `_stride_tricks_impl.py`, `convert_datatype.c`, `getset.c` |
| 6 | fancy/boolean indexing, assignment, `.flat` | `mapping.c`, `iterators.c` |
| 7 | take/repeat/choose/where/concatenate, put/putmask/place/copyto/fill, in-place shape guards | `item_selection.c`, `multiarraymodule.c`, `compiled_base.c` |
| 8 | casting audit: same-itemsize `view(dtype)`, `ndmin`, `np.array([masked, ...])`, subarray `astype` | `convert.c`, `ctors.c`, `multiarraymodule.c` |
| 9 | matmul/dot/inner/einsum/correlate, `numpy.linalg` | `_core/_op_mask.py`, `arrayobject.c`, `multiarraymodule.c` |
| 10 | repr/str, `filled()`, `isin`, pickle, buffer/IO decisions | `arrayprint.py`, `methods.c`, `_arraysetops_impl.py` |

## Housekeeping

**Standing rule — sync upstream** (at the start of every session/phase and
before every push): `git fetch origin main` → merge `origin/main` into the
feature branch (stash WIP first) → rebuild → rerun the gate → confirm
`git rev-list --count HEAD..origin/main` is `0`. Never sync from `fork/main`,
never push `main`; `fork/<branch>` being current says nothing about upstream.

**Gate after every change:** the plain-array baseline (mask == NULL) must
behave exactly like upstream.
1. Whole suite: `pytest numpy -m "not slow" -n 4` from
   `build-install/usr/lib/python3/dist-packages` with `.venv/bin` on `PATH` and
   `PYTHONPATH=$PWD` (without it `test_cpu_features`/`test_limited_api`/
   `test_cython` fail with "No module named numpy": environment, not a bug).
   The gate was widened from four test files to the whole suite in phase 8:
   `numpy/ma` and `test_public_api` caught two phase-5 bugs.
2. Cross-run against upstream: build `origin/main` in a git worktree
   (`git worktree add --detach ~/up-numpy <commit>`, `spin build`), run the same
   suite on both with `--junitxml`, compare per-test outcomes.
3. RSS + refcount leak loops for anything touching C.
4. Check the relevant NEPs and current `main` behaviour for dispatch/dtype/
   iterator changes; review `DESIGN.md`.

New masked behaviour is tested in `numpy/_core/tests/test_mask.py`.

## Regression baseline

| After phase | Plain-array baseline | `test_mask.py` |
|---|---|---|
| 0–4 | `test_multiarray` + `test_indexing` (`-m "not slow"`) 14916 passed, 17 skipped; `test_umath` + `test_ufunc` 5615 passed, 60 skipped, 7 xfailed | 40 (phase 3) |
| 5 | 14937 / 17 skipped (higher than 14916 only because a merge brought upstream's own new tests); `test_umath`, `test_ufunc`, `test_shape_base`, `test_stride_tricks` 6016 passed, 60 skipped, 7 xfailed | — |
| 6 | identical; wider run (+ `test_regression`, `test_item_selection`) 21668 passed | 114 |
| 7 | identical; widened gate (+ `test_function_base`, `test_arraysetops`, `test_index_tricks`, `test_shape_base`, `test_twodim_base`) 23466 passed, 151 skipped, 7 xfailed | 148 |
| 8 | identical; whole suite 49348+ passed, 0 failed | 177 |
| 9 | identical; whole suite 49419 passed, 1051 skipped, 57 xfailed, 1 xpassed; `numpy/linalg` 520 passed | 208 |
| 10 | whole suite 49513 passed, 1052 skipped, 57 xfailed, 1 xpassed, 0 failed (also after the zero default) | 287 |

Phase 10 cross-run against upstream (`origin/main` @ 84cd9b6d16, same test
tree, both runs `-n 6`): upstream 49226 passed, fork 49513 (+287 = exactly
`test_mask.py`), 50336 common tests, **0 status differences, 0 failures**. A
per-member comparison of every public `ndarray` method and attribute on 19
plain array kinds (2736 records, incl. pickle bytes of protocols 0–5, `dump`,
`tofile`, `repr`, `array2string`) is identical except the raw pointer bytes of
`object` arrays, which also differ between two runs of upstream itself.
Phase 10's first whole-suite run had 20 failures: the new `ndarray.filled` was
picked up by `np.ma.filled`'s `hasattr(a, 'filled')` duck-typing and by the
matrix "call every method" test; fixed in `ma/core.py`.

## Implementation notes per phase

Short; the design rules are in `DESIGN.md`, the git history has the details.

**1 — struct and API.** `PyArrayObject *mask` appended to
`PyArrayObject_fields` behind `NPY_FEATURE_VERSION >= NPY_2_7_API_VERSION`
(header-only, no C-API table change), `PyArray_MASK()` accessor, `Py_CLEAR` in
`_clear_array_attributes`. `PyArray_SetMaskObject(arr, obj)` (internal, steals
a reference) rejects non-ndarray, non-bool, shape mismatch, an `obj` that has
a mask, and a receiver flagged `NPY_ARRAY_IS_MASK`; an all-False mask becomes
`NULL`. `arr.mask` getter/setter/`del` in `getset.c`. Leak-checked.

**2 — copy/view/shape/sort.** Propagated at choke points: `PyArray_NewCopy`
copies the mask; `PyArray_View` (no dtype change) attaches a mask *view*;
basic indexing replays the parsed index on the mask; `reshape`/`ravel`/
`squeeze` recurse on the mask. `sort`/`partition` use `_sort_with_mask`: an
argsort permutation replayed on data and mask (correct, not the in-place
algorithm; partition falls back to sort). The blanket "bool arrays cannot
carry a mask" rule was wrong (comparison results are bool) and was replaced by
the usage flag `NPY_ARRAY_IS_MASK` (not refcounted: sharing the identical mask
object between owners is an accepted edge case).

**3 — elementwise ufuncs.** Black-box wrapper: `ufunc_generic_vectorcall` runs
the untouched `ufunc_generic_fastcall`, then `_propagate_ufunc_result_mask`
ORs the input masks (broadcast to the exact output shape) and attaches. Handles
multi-output, `out=`, in-place. `where=` with an explicit `out=` is a
selective write (`PyArray_Where(where, new, old_out_mask)`); `where=` with
`out=None` or multi-output skips propagation. The AVX-512 masked loop of the
original plan was deferred to the performance backlog.

**4 — reductions.** Run the *identical* call with `logical_or` on the mask(s):
`_propagate_{reduce,accumulate,reduceat,outer,at}_mask`. `initial=NULL` for the
mask so `logical_or`'s identity (`False`) applies. `where=` passes through for
`reduce`. Unary `.at()` needs nothing. `calculation.c` needed no change
(`sum/prod/max/min/any/all/mean/std/var` route through `PyUFunc_Reduce` or
compose from ufuncs); `std`/`var` hide a whole axis if any element of it is
hidden. `argmax`/`argmin` are mask-blind.

**5 — transport.** Same permutation on the mask for `PyArray_Transpose`;
`broadcast_to` (read-only mask view); dtype-changing `view` (refined in phase
8); `astype`/`np.array(dtype=)` copy the mask via `PyArray_CopyMaskFrom`
(laid out like the destination data); `flatten`, `diagonal`, `real`/`imag`;
`as_strided` re-runs itself on the mask with element strides (fail-open when
strides are not whole elements or the layout is not proportional).
`_stride_tricks_impl.py` must read the mask through `np.ndarray.mask`
(`_get_mask`/`_set_mask`), because `MaskedArray` has its own `mask`.

**6 — indexing.** Fancy/boolean getitem indexes the mask with the same index
object (hook after `finish:` in `array_subscript`). Assignment runs the same
assignment on the mask with the RHS's mask, after the data assignment
succeeds; a read-only destination mask fails *before* touching the data.
`.flat` get/set, `flatiter.copy()`. Verified by a 4000-case randomized
differential test.

**7 — gather/scatter/combine.** Gather (`take`, `repeat`, `choose`) and
combine (`concatenate` → `stack`/`vstack`/`append`/…, `np.where`) run the same
operation on the masks; scatter (`put`, `putmask`, `place`, `copyto`, `fill`,
`flat`/`real`/`imag` assignment) gives assigned elements the values'
masked-ness; helpers `PyArray_MaskForUpdate`/`FinishMaskUpdate`/
`FailUnlessMaskWriteable`. `PyArray_FailIfMaskedInPlace` rejects in-place
`a.shape=`/`strides=`/`dtype=`/`resize` on a masked array or a mask (the mask
would otherwise keep the old shape). `np.add(x, 1, out=o)` with unmasked
inputs now clears `o`'s stale mask. Index-returning operations are
mask-blind; selection *parameters* (`where=`, `putmask` condition, `take`
indices) are plain data. 3000-case randomized differential test.

**8 — casting.** Same-itemsize `view(dtype)` shares the mask, other itemsizes
raise (`astype`); `np.array(ndmin=)`, `np.array([masked, ...])` (leaf masks
copied through `_assign_from_cache_masked`) and subarray `astype` (mask
broadcast over the appended dims).

**9 — linear algebra.** Decision (user): compute on the whole array, the mask
follows; kernels untouched. `numpy/_core/_op_mask.py` (Python, reached only
when an input or `out=` is masked) called from C via `PyArray_PropagateOpMask`
(`ufunc_generic_vectorcall` for gufuncs; wrappers around
`PyArray_MatrixProduct2`, `InnerProduct`, `Correlate`/`Correlate2` and
`array_einsum`). Rule: an output element is hidden iff a hidden input element
contributes to it. `matmul`/`dot`/`inner`: `any` over rows/columns; `einsum`/
`correlate`: the same operation on float32 counts of the masks; `numpy.linalg`
gufuncs: the whole output matrix is hidden if any element of the input matrix
is (per matrix of a stack); `solve` distinguishes columns of `b`; `qr_r_raw`
also rewrites `a`'s mask. New `PyArray_MaskOr` (always an ndarray, replaces
`PyNumber_Or` on masks: two 0-d masks OR to a numpy bool scalar).

**10 — Python surface.**
- `repr`/`str` (`arrayprint.py`): a hidden cell prints as `--`, right-aligned
  to the visible width; the hidden value is never printed and is excluded from
  width/precision decisions; all dtypes, summarization, 0-d (`array(--)`),
  user formatters bypassed for hidden cells.
- `ndarray.filled(fill_value=0)` (`methods.c`): new plain, unmasked, writable
  copy; the value is assigned like `arr[i] = v` and validated even when nothing
  is hidden; keeps subclass and layout. **Default (user decision, 2026-09-22):
  the zero of the dtype, as `np.zeros` builds it** (0, `False`, `''`, `b''`,
  1970-01-01, all-zero records, `0` for object, `''` for `StringDType`), so
  every dtype has a default and nothing raises. (First version used the NA
  pattern of `LAYOUTS.md`; dropped: that table is for *missing* data, this
  branch hides *known* data, and it produced an out-of-range bool byte and a
  `uint8` 255.) `np.ma.filled` no longer mistakes the method for a masked
  array's.
- `isin`: hidden where `element` is hidden, and where `element` matches no
  *visible* test value while `test_elements` has a hidden cell.
- Pickle: masked arrays add the mask as a sixth `__setstate__` item (unmasked
  pickles are byte-identical to upstream); `__setstate__` validates it;
  protocol 5 uses the in-band reduce for masked arrays.
- Decided (user): `unique` and the set operations stay mask-blind;
  buffer protocol, `tobytes`/`tofile`/`np.save`/`tolist` export data only.

## Known limitations (fail-open)

Policy (`DESIGN.md`): an operation without mask support behaves like it does
for a plain array — the mask is dropped and nothing raises. Every gap is
listed here. Audited at the close of phase 10.

| Operation | What happens | Status |
|---|---|---|
| **Scalar results** (`x.sum()`, `x.max()`, `x[i]`, `.flat[i]`, `det`/`norm`/`trace` of one matrix, `v @ v`, `vdot`, 1-d `dot`/`inner`, ufuncs on 0-d, ...) | A scalar carries no mask: NumPy decays a 0-d result to a scalar. Keep the mask with `axis=`, `keepdims=True` or `out=`; `det` of a matrix with a hidden cell is a plain number (a stack keeps it). No 0-d retention, no masked scalar type, no NaN sentinel | **Final (user, 2026-09-21).** Not to be re-opened unless the user asks |
| `unique` (+ `return_*`), `unique_*`, `intersect1d`, `setxor1d`, `union1d`, `setdiff1d` | Mask-blind: run on the whole data, return plain arrays, hidden values included | Decided (user, phase 10) |
| Buffer protocol, `tobytes`, `tofile`, `np.save`/`savetxt`, `tolist`, `memoryview` | Export the raw data (hidden cells included), no mask; a saved/loaded array is unmasked. Pickle does carry the mask | Decided (user, phase 10) |
| Pickle of a masked array | 6-item state: older NumPy errors instead of dropping the mask; no out-of-band buffers with protocol 5 | By design |
| `np.frombuffer`, `getfield`, `__array_wrap__(x)` | Result has no mask (no owner-mask relation) | By design |
| Structured dtypes | The mask is **per record**, not per field (per-field masks are out of scope). Field views `s["x"]` do not carry it; `s["x"] = v` and `setfield` leave it unchanged. Whole-record operations work | Out of scope |
| Coarse linalg masks | Every output of `inv`/`eig`/`svd`/`cholesky`/`qr`/`lstsq`... is hidden for the whole matrix if any input element is | By design |
| `view(dtype)` with another itemsize | `ValueError` (the last axis changes); use `astype()` | By design |
| `np.array(list_of_masked, dtype=object, ndmax=k)`, `k` below the depth | The masked arrays become object *elements*; no mask | By design |
| `out=` (ufunc, take, choose, concatenate) and assignment through a view whose own mask is `None` | The mask *object* on `out` is replaced, so other views holding the old one keep it; a view with no mask cannot write masked-ness back to its parent | Revisit: write into the existing mask buffer |
| `vdot`, `np.cross` / any `out=` into a view of an unmasked array | Same cause: `np.cross` of masked vectors returns the right data with no mask | Revisit with the `out=` row |
| Assignment to a `broadcast_arrays` result (read-only mask) | Fails before touching the data | Revisit |
| `as_strided`/`sliding_window_view` with non-whole-element strides or a mask layout not proportional to the data | Mask dropped | Revisit (conform the mask copy to the data layout) |
| `a[...] = [masked, ...]` (list/tuple RHS) | Data assigned, masks inside the list ignored (public `PyArray_CopyObject` has no hook); `a[...] = np.array([masked, ...])` works | Revisit (route through `PyArray_FromAny`) |
| `where=` with `out=None`, or with a multi-output ufunc | Mask propagation skipped | Revisit |

## Performance backlog (not an active phase)

Everything so far reuses tested NumPy pieces (`logical_or.reduce`, `PyNumber_Or`,
`PyArray_Where`, ...) instead of hand-rolled loops: easy to trust and cheap to
leak-check. The items below trade that for speed by fusing mask logic into
shared, delicate internals (`NpyIter`, `PyArrayMapIterObject`, SIMD loops). Do
them only when there is a measured need, not opportunistically.

- **`add.at` (+54–170%, worst):** fuse the mask OR into the index loops of
  `ufunc_at__fast_iter`, `ufunc_at__slow_iter`, `trivial_at_loop` (highest risk:
  shared `.at()` machinery, overlap copies, buffering).
- **`add.accumulate` (+53–71%):** fuse the running OR into `PyUFunc_Accumulate`.
- **`.std()`/`.var()` (+33–320%):** they run in Python (`_core/_methods.py`), not
  `PyArray_Std` (an earlier attempt there had no effect and was reverted).
  Compute `logical_or.reduce(mask, axis, where=where)` once, run the arithmetic
  with the mask stripped, reattach; mind `where=`, `keepdims`, complex.
- **Elementwise `add`/`less` (+13–65%):** fuse into the SIMD/scalar inner loops
  in `umath/loops*.c.src` (the AVX-512 masked-loop work of the original plan).
- **`add.reduceat`/`add.outer` (+7–103%):** profile before choosing a direction.
