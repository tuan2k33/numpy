#ifndef NUMPY_CORE_SRC_MULTIARRAY_ROUND_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ROUND_H_

#include "numpy/ndarraytypes.h"
#include "npy_cpu_dispatch.h"

/*
 * Fused kernels for ndarray.round(decimals), see round.dispatch.c.src.
 * Strides are in bytes.  ``p`` is 10**|decimals| and ``neg`` is non-zero for
 * negative decimals.
 */
#include "round.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void ROUND_HALF,
    (const char *src, npy_intp src_stride, char *dst, npy_intp dst_stride,
     npy_intp len, double p, int neg))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void ROUND_FLOAT,
    (const char *src, npy_intp src_stride, char *dst, npy_intp dst_stride,
     npy_intp len, double p, int neg))
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void ROUND_DOUBLE,
    (const char *src, npy_intp src_stride, char *dst, npy_intp dst_stride,
     npy_intp len, double p, int neg))

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ROUND_H_ */
