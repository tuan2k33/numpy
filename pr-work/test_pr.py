"""Regression tests for gh-32595.

Covers the two changes in the PR:
  1. buffer re-use in the plain buffered iternext (nditer_templ.c.src)
  2. filling a broadcast destination by doubling (lowlevel_strided_loops.c.src)

Written in NumPy's test style (pytest functions, numpy.testing helpers,
np.setbufsize restored in ``finally``) so they can be moved as they are:
  - test_broadcast_*   -> numpy/_core/tests/test_nditer.py
  - test_reduce_out_*  -> numpy/_core/tests/test_ufunc.py, class TestUfunc,
                          next to test_reducelike_out_promotes (add ``self``)

Note: np.setbufsize only accepts multiples of 16 (5 <= size <= 10e6), while
np.nditer(buffersize=...) accepts any size, so the nditer test is the one that
exercises buffer boundaries which do not line up with the rows.

Run against the PR build:
    cd /mnt/c/Dev/projects/numpy && source .venv/bin/activate
    spin test -- /mnt/c/Dev/projects/numpy-pr-work/test_pr.py -v

Expected: everything passes on the PR build.  On main, only the
test_reduce_out_* tests fail; that is the behaviour change noted in the PR.
"""
import math
import sys

import pytest

import numpy as np
from numpy.testing import HAS_REFCOUNT, assert_array_equal


def _with_bufsize(bufsize, func):
    old = np.getbufsize()
    try:
        np.setbufsize(bufsize)
        return func()
    finally:
        np.setbufsize(old)


@pytest.mark.parametrize("bufsize", [32, 128, 8192])
@pytest.mark.parametrize("trailing", [1, 3, 7, 50])
@pytest.mark.parametrize("dtype", ["f4", "f8", "i8", "c16"])
def test_broadcast_row_across_buffer_refills(dtype, trailing, bufsize):
    # A row broadcast over a C-contiguous array has identical buffer content
    # for every refill.  Compare against a materialised operand over many
    # refills, including rows longer than the buffer.
    rows = 1001
    a = np.arange(rows * trailing).reshape(rows, trailing).astype(dtype)
    v = np.arange(1, trailing + 1).astype(dtype)
    v_full = np.broadcast_to(v, a.shape).copy()

    def compute():
        inplace = a.copy()
        inplace += v
        return a + v, inplace

    res, inplace = _with_bufsize(bufsize, compute)
    assert_array_equal(res, a + v_full)
    assert_array_equal(inplace, a + v_full)


@pytest.mark.parametrize("buffersize", [3, 10, 100, 8192])
def test_broadcast_row_nditer_external_loop(buffersize):
    # Like test_iter_buffered_reduce_reuse_core, but for plain (non-reduce)
    # buffered iteration, with buffer sizes that do not divide the row length.
    rows, trailing = 1001, 3
    m = np.arange(rows * trailing, dtype="f8").reshape(rows, trailing)
    v = np.arange(1, trailing + 1, dtype="f8")
    it = np.nditer((m, v), op_flags=[["readonly"], ["readonly"]],
                   flags=["buffered", "external_loop"], buffersize=buffersize)
    with it:
        got = np.concatenate([a + b for a, b in it])
    assert_array_equal(got, (m + np.broadcast_to(v, m.shape).copy()).ravel())


@pytest.mark.parametrize("bufsize", [32, 128, 8192])
def test_broadcast_column_across_buffer_refills(bufsize):
    # The source pointer advances with every row, so the buffer must not be
    # re-used between refills.
    rows, trailing = 1001, 3
    a = np.arange(rows * trailing, dtype="f8").reshape(rows, trailing)
    col = np.arange(1, rows + 1, dtype="f8").reshape(rows, 1)
    res = _with_bufsize(bufsize, lambda: a + col)
    assert_array_equal(res, a + np.broadcast_to(col, a.shape).copy())


@pytest.mark.parametrize("bufsize", [32, 8192])
@pytest.mark.parametrize("a_dtype, v_dtype",
                         [("f8", "i4"), ("f4", "i8"), ("i8", "i2"), ("c16", "f4")])
def test_broadcast_row_with_cast_across_buffer_refills(a_dtype, v_dtype, bufsize):
    # The buffer is filled through a cast, so the doubling copies cast output.
    rows, trailing = 1001, 3
    a = np.arange(rows * trailing).reshape(rows, trailing).astype(a_dtype)
    v = np.array([1, 2, 3], dtype=v_dtype)
    v_full = np.broadcast_to(v, a.shape).copy()

    def compute():
        inplace = a.copy()
        inplace += v
        return a + v, inplace

    res, inplace = _with_bufsize(bufsize, compute)
    expected_inplace = a.copy()
    expected_inplace += v_full
    assert_array_equal(res, a + v_full)
    assert_array_equal(inplace, expected_inplace)


@pytest.mark.parametrize("bufsize", [32, 8192])
def test_broadcast_row_noncontiguous_and_3d(bufsize):
    a = np.arange(1001 * 6, dtype="f8").reshape(1001, 6)[:, ::2]
    v = np.array([0.5, 1.5, 2.5])
    b = np.arange(40 * 30 * 3, dtype="f8").reshape(40, 30, 3)
    w = np.arange(1, 31, dtype="f8").reshape(30, 1)

    res2d, res3d_row, res3d_col = _with_bufsize(
        bufsize, lambda: (a + v, b + v, b + w))
    assert_array_equal(res2d, a + np.broadcast_to(v, a.shape).copy())
    assert_array_equal(res3d_row, b + np.broadcast_to(v, b.shape).copy())
    assert_array_equal(res3d_col, b + np.broadcast_to(w, b.shape).copy())


@pytest.mark.parametrize("bufsize", [32, 8192])
def test_broadcast_rows_three_operands(bufsize):
    x = np.arange(1001 * 3, dtype="f8").reshape(1001, 3)
    y = np.array([1.0, 2.0, 3.0])
    z = np.array([0.0, 2.0, 4.0])
    res = _with_bufsize(bufsize, lambda: x * y + z)
    y_full = np.broadcast_to(y, x.shape).copy()
    z_full = np.broadcast_to(z, x.shape).copy()
    assert_array_equal(res, x * y_full + z_full)


@pytest.mark.parametrize("bufsize", [32, 8192])
def test_broadcast_row_object_dtype(bufsize):
    # Object arrays must not take the raw-byte doubling path; check values
    # and that the operand's references are balanced.
    rows = 1001
    a = np.arange(rows * 3, dtype=object).reshape(rows, 3)
    v = np.array([10**20, 2 * 10**20, 3 * 10**20], dtype=object)
    expected = a + np.broadcast_to(v, a.shape).copy()

    if HAS_REFCOUNT:
        before = [sys.getrefcount(x) for x in v]
    res = _with_bufsize(bufsize, lambda: a + v)
    if HAS_REFCOUNT:
        assert [sys.getrefcount(x) for x in v] == before
    assert_array_equal(res, expected)


@pytest.mark.parametrize("offsets, itemsize", [([0, 4], 8), ([0, 8], 12)])
def test_broadcast_row_structured_dtype(offsets, itemsize):
    # Contiguous fields and fields with a gap (not trivially copyable).
    dt = np.dtype({"names": ["a", "b"], "formats": ["i4", "i4"],
                   "offsets": offsets, "itemsize": itemsize})
    rows = 1001
    src = np.zeros(3, dtype=dt)
    src["a"] = [1, 2, 3]
    src["b"] = [10, 20, 30]
    dst = np.zeros((rows, 3), dtype=dt)
    np.copyto(dst, src)
    assert_array_equal(dst["a"], np.broadcast_to(src["a"], (rows, 3)))
    assert_array_equal(dst["b"], np.broadcast_to(src["b"], (rows, 3)))


@pytest.mark.parametrize("bufsize", [32, 1024])
@pytest.mark.parametrize("variant", ["plain", "initial", "keepdims", "where"])
def test_reduce_out_cast_keeps_computation_precision(variant, bufsize):
    # With a `dtype=` wider than `out`, the running result used to be written
    # to `out` and read back at every buffer refill, accumulating float32
    # rounding.  It is now kept in float64 and cast once, so the result stays
    # within one float32 ulp of the exact sum for any buffer size.
    x = np.full(100_000, 1e-3)
    kwargs = {}
    values = x
    if variant == "initial":
        kwargs["initial"] = 5.0
    elif variant == "keepdims":
        kwargs["keepdims"] = True
    elif variant == "where":
        mask = np.arange(x.size) % 2 == 0
        kwargs["where"] = mask
        values = x[mask]
    exact = math.fsum(values) + kwargs.get("initial", 0.0)
    out = np.zeros((1,) if variant == "keepdims" else (), dtype=np.float32)

    _with_bufsize(
        bufsize, lambda: np.add.reduce(x, dtype=np.float64, out=out, **kwargs))

    assert abs(out.item() - exact) <= float(np.spacing(np.float32(exact)))


# ---------------------------------------------------------------------------
# Buffer re-use safety on the plain (non-reduce) iternext.
# Non-reduce twins of test_iter_buffered_reduce_reuse and
# test_iter_buffered_reduce_reuse_core in numpy/_core/tests/test_nditer.py,
# which only exercise the reduce path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("buffersize", [0, 3, 7, 16, 27, 100])
def test_broadcast_reuse_stride_sweep(buffersize):
    # Sweep the strides of the non-broadcast operand while a second operand is
    # broadcast (zero strides).  Buffer re-use must not depend on the strides.
    base = np.arange(2 * 3**5)[3**5:3**5 + 1]
    itemsize = base.itemsize
    v = np.arange(1, 4, dtype="f8")
    v_full = np.broadcast_to(v, (3, 3, 3)).copy()

    for xs in range(-3**2, 3**2 + 1):
        for ys in range(xs, 3**2 + 1):
            strides = (xs * itemsize, ys * itemsize, itemsize)
            arr = np.lib.stride_tricks.as_strided(base, (3, 3, 3), strides)
            expected = arr.astype("f8") + v_full

            it = np.nditer(
                [arr, np.broadcast_to(v, (3, 3, 3)), None],
                flags=["buffered", "external_loop", "refs_ok"],
                op_flags=[["readonly"], ["readonly"],
                          ["writeonly", "allocate"]],
                op_dtypes=["f8", "f8", "f8"], buffersize=buffersize)
            with it:
                for x, y, z in it:
                    z[...] = x + y
                res = it.operands[2]
            assert_array_equal(res, expected, err_msg=f"xs={xs} ys={ys}")


@pytest.mark.parametrize("buffersize", [16, 30, 128])
@pytest.mark.parametrize("skip", [1, 5, 17, 64])
def test_broadcast_reuse_iterindex_jump(buffersize, skip):
    # Setting iterindex leaves a coreoffset, which must disable re-use.  Values
    # after the jump must match a straight iteration from the same index.
    rows, trailing = 200, 3
    a = np.arange(rows * trailing, dtype="f8").reshape(rows, trailing)
    v = np.arange(1, trailing + 1, dtype="f8")
    ref = (a + np.broadcast_to(v, a.shape).copy()).ravel()

    it = np.nditer([a, np.broadcast_to(v, a.shape)],
                   flags=["buffered", "multi_index", "refs_ok"],
                   op_flags=[["readonly"], ["readonly"]],
                   op_dtypes=["f8", "f8"], buffersize=buffersize)
    before = [sum(o.item() for o in next(it)) for _ in range(40)]
    assert_array_equal(before, ref[:40])

    it.iterindex = skip
    after = [sum(o.item() for o in next(it)) for _ in range(10)]
    assert_array_equal(after, ref[skip:skip + 10])


@pytest.mark.parametrize("buffersize", [0, 16, 32, 128])
def test_writemasked_broadcast_reuse_nditer(buffersize):
    # A writemasked operand is only partially written back, so the buffer must
    # not be re-used with whatever the inner loop left in it.  NumPy only
    # allows a broadcast writemasked operand when the mask broadcasts the same
    # way (one mask value per element), so mask and output share a shape.
    rows, trailing = 200, 3
    a = np.arange(rows * trailing, dtype="f8").reshape(rows, trailing)
    mask = np.array([[True, False, True]])

    expected = np.zeros((1, trailing), dtype=np.float32)
    for i in range(rows):
        for j in range(trailing):
            if mask[0, j]:
                expected[0, j] = a[i, j]

    out = np.zeros((1, trailing), dtype=np.float32)
    it = np.nditer(
        [a, np.broadcast_to(mask, (rows, trailing)), out],
        flags=["buffered", "reduce_ok", "refs_ok"],
        op_flags=[["readonly"], ["readonly", "arraymask"],
                  ["readwrite", "writemasked"]],
        op_dtypes=["f8", "?", "f8"], casting="same_kind",
        buffersize=buffersize)
    with it:
        for x, _m, z in it:
            z[...] = x
    assert_array_equal(out, expected)


@pytest.mark.parametrize("bufsize", [32, 128, 8192])
@pytest.mark.parametrize("trailing", [1, 3, 7])
def test_writemasked_broadcast_reuse_ufunc(trailing, bufsize):
    # `where=` makes the output writemasked while the broadcast operand's
    # buffer is re-used; masked-out elements must keep their original values.
    rows = 1001
    a = np.arange(rows * trailing, dtype="f8").reshape(rows, trailing)
    v = np.arange(1, trailing + 1, dtype="f8")
    mask = (np.arange(rows * trailing).reshape(rows, trailing) % 3) == 0
    v_full = np.broadcast_to(v, a.shape).copy()

    expected = a.copy()
    np.add(expected, v_full, out=expected, where=mask)

    def compute():
        got = a.copy()
        np.add(got, v, out=got, where=mask)
        return got

    assert_array_equal(_with_bufsize(bufsize, compute), expected)
