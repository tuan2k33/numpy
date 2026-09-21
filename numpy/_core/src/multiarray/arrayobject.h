#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip);

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op);

NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj);

/*
 * For use in __setstate__, where pickle gives us an instance on which we
 * have to replace all the actual data. Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
clear_array_attributes(PyArrayObject *self);

/*
 * Sets (or, with NULL/Py_None, clears) `arr`'s mask -- see the definition
 * in arrayobject.c for the full contract and the mask->mask invariant it
 * enforces. Not yet part of the public C-API (see TODO.md phase 1/2).
 */
NPY_NO_EXPORT int
PyArray_SetMaskObject(PyArrayObject *arr, PyObject *obj);

/*
 * Mask transport helpers for ops that produce a new array of the *same
 * shape* as `src`: no-ops when `src` is unmasked. `CopyMaskFrom` attaches an
 * independent copy of `src`'s mask to `dst` (for ops that copy the data),
 * `ViewMaskFrom` attaches a view sharing `src`'s mask buffer (for ops that
 * view the data). Return 0 on success, -1 with an exception set on failure.
 */
NPY_NO_EXPORT int
PyArray_CopyMaskFrom(PyArrayObject *dst, PyArrayObject *src);

NPY_NO_EXPORT int
PyArray_ViewMaskFrom(PyArrayObject *dst, PyArrayObject *src);

/*
 * This flag is used to mark arrays which we would like to, in the future,
 * turn into views. It causes a warning to be issued on the first attempt to
 * write to the array (but the write is allowed to succeed).
 *
 * This flag is for internal use only, and may be removed in a future release,
 * which is why the #define is not exposed to user code. Currently it is set
 * on arrays returned by ndarray.diagonal.
 */
static const int NPY_ARRAY_WARN_ON_WRITE = (1 << 31);


/*
 * These flags are used internally to indicate an array that was previously
 * a Python scalar (int, float, complex).  The dtype of such an array should
 * be considered as any integer, floating, or complex rather than the explicit
 * dtype attached to the array.
 *
 * These flags must only be used in local context when the array in question
 * is not returned.  Use three flags, to avoid having to double check the
 * actual dtype when the flags are used.
 */
static const int NPY_ARRAY_WAS_PYTHON_INT = (1 << 30);
static const int NPY_ARRAY_WAS_PYTHON_FLOAT = (1 << 29);
static const int NPY_ARRAY_WAS_PYTHON_COMPLEX = (1 << 28);
/*
 * Mark that this was a huge int which was turned into an object array (or
 * unsigned/non-default integer array), but then replaced by a temporary
 * array for further processing. This flag is only used in the ufunc machinery
 * where it is tricky to cover correctly all type resolution paths.
 */
static const int NPY_ARRAY_WAS_INT_AND_REPLACED = (1 << 27);
static const int NPY_ARRAY_WAS_PYTHON_LITERAL = (1 << 30 | 1 << 29 | 1 << 28);

/*
 * Set on a bool-dtype array while it is attached as *some* array's `mask`
 * (see `PyArray_SetMaskObject` in arrayobject.c). This is what actually
 * enforces "a mask can never itself have a mask": dtype alone can't do it,
 * since ordinary bool arrays (e.g. comparison-ufunc results) must still be
 * allowed to carry their own mask when they aren't currently serving as
 * anyone else's mask. Set when attached, cleared when detached/replaced.
 * Not refcounted -- if the same array object is deliberately shared as the
 * mask for more than one owner (attaching by object identity, not via a
 * `.view()` of it), detaching from one owner clears this flag even though
 * another owner may still reference it; that's an accepted limitation of
 * a niche, undesigned-for usage rather than the normal per-owner
 * `.view()`-based sharing phase 2 relies on.
 */
static const int NPY_ARRAY_IS_MASK = (1 << 24);

/*
 * Mark an array converted from an exact Python str.  Unlike the flags above
 * it does not participate in promotion (the array keeps its discovered
 * dtype); it only lets scalar-aware paths (ufuncs, copyto, where) convert
 * the operand again from the original object once the operation's
 * descriptors are resolved.  Not part of NPY_ARRAY_WAS_PYTHON_LITERAL,
 * whose consumers assume a numeric scalar.
 */
static const int NPY_ARRAY_WAS_PYTHON_STR = (1 << 25);

/*
 * This flag allows same kind casting, similar to NPY_ARRAY_FORCECAST.
 *
 * An array never has this flag set; they're only used as parameter
 * flags to the various FromAny functions.
 */
static const int NPY_ARRAY_SAME_KIND_CASTING = (1 << 26);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_ */
