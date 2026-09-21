#ifndef NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Embedded as the global_state field of multiarray_umath_state. */
typedef struct npy_global_state_struct {
    /*
     * Used to test the internal-only scaled float test dtype
     */
    npy_bool get_sfloat_dtype_initialized;

    /*
     * controls the global madvise hugepage setting
     */
    int madvise_hugepage;

    /*
     * used to detect module reloading in the reload guard
     */
    int reload_guard_initialized;

    /*
     * Holds the user-defined setting for whether or not to warn
     * if there is no memory policy set
     */
    int warn_if_no_mem_policy;
} npy_global_state_struct;

NPY_NO_EXPORT int
get_legacy_print_mode(void);

/* `np.where` without the mask propagation (used to combine masks, which
 * themselves cannot carry a mask). */
NPY_NO_EXPORT PyObject *
PyArray_WhereNoMask(PyObject *condition, PyObject *x, PyObject *y);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MULTIARRAYMODULE_H_ */
