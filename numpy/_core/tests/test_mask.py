"""
Tests for the opt-in `ndarray.mask` (refactor/ndarray-mask branch).

See TODO.md at the repository root for the design this implements:
mask == NULL means "definitely unmasked" (the fast path, byte-identical to
upstream numpy); a non-NULL mask is a plain bool ndarray of the same shape
marking which elements are hidden from observability (not a MISSING-value
representation -- the underlying data is always real and fully computed).
"""
import numpy as np
import pytest


class TestMaskInvariants:
    """Phase 1/2: PyArray_SetMaskObject's validation and canonicalization."""

    def test_default_mask_is_none(self):
        a = np.arange(5)
        assert a.mask is None

    def test_set_and_clear(self):
        a = np.arange(5)
        m = np.array([False, True, False, False, True])
        a.mask = m
        assert np.array_equal(a.mask, m)
        a.mask = None
        assert a.mask is None

    def test_del_clears(self):
        a = np.arange(5)
        a.mask = np.array([True, False, False, False, False])
        del a.mask
        assert a.mask is None

    def test_non_ndarray_rejected(self):
        a = np.arange(5)
        with pytest.raises(TypeError):
            a.mask = [True, False, False, False, False]

    def test_non_bool_dtype_rejected(self):
        a = np.arange(5)
        with pytest.raises(TypeError):
            a.mask = np.zeros(5, dtype=np.int64)

    def test_shape_mismatch_rejected(self):
        a = np.arange(5)
        with pytest.raises(ValueError):
            a.mask = np.zeros(4, dtype=bool)

    def test_bool_array_cannot_carry_a_mask(self):
        # Closes the point-in-time hole: without this, `m = arr.mask;
        # m.mask = x` could nest a mask onto a mask after the fact, even
        # though attaching `m` to `arr` was valid at the time.
        m = np.array([True, False, True])
        with pytest.raises(TypeError):
            m.mask = np.array([False, True, False])

    def test_all_false_mask_canonicalized_to_none(self):
        a = np.arange(5)
        a.mask = np.zeros(5, dtype=bool)
        assert a.mask is None

    def test_mask_with_any_true_is_kept(self):
        a = np.arange(5)
        m = np.zeros(5, dtype=bool)
        m[2] = True
        a.mask = m
        assert a.mask is not None
        assert np.array_equal(a.mask, m)


class TestMaskCopyView:
    """Phase 2: copy()/view()/basic-slicing mask propagation."""

    def _masked(self):
        a = np.arange(10)
        a.mask = np.array(
            [False, True, False, False, True,
             False, False, False, False, False])
        return a

    def test_copy_propagates_mask_independently(self):
        a = self._masked()
        b = a.copy()
        assert np.array_equal(b.mask, a.mask)
        # A copy must not share the mask buffer either.
        b.mask[0] = True
        assert a.mask[0] == False

    def test_view_propagates_and_shares_mask(self):
        a = self._masked()
        v = a.view()
        assert np.array_equal(v.mask, a.mask)
        # `.view()` shares memory with the original -- the mask does too.
        v.mask[0] = True
        assert a.mask[0] == True

    def test_basic_slice_propagates_mask(self):
        a = self._masked()
        s = a[2:5]
        assert np.array_equal(s.mask, a.mask[2:5])

    def test_ellipsis_propagates_mask(self):
        a = self._masked()
        e = a[...]
        assert np.array_equal(e.mask, a.mask)

    def test_single_int_index_has_no_mask_attribute_issue(self):
        # a[i] for a 1-d array returns a scalar, not an ndarray -- just
        # confirm it doesn't crash when the parent is masked.
        a = self._masked()
        assert a[0] == 0

    def test_multi_dim_int_index_propagates_mask(self):
        a = np.arange(6).reshape(2, 3)
        a.mask = np.array([[False, True, False], [False, False, True]])
        row = a[0]
        assert np.array_equal(row.mask, a.mask[0])

    def test_unmasked_array_stays_unmasked_through_copy_view(self):
        a = np.arange(5)
        assert a.copy().mask is None
        assert a.view().mask is None
        assert a[1:3].mask is None

    def test_two_views_carry_independent_masks_over_shared_buffer(self):
        # From the "Scope" section of TODO.md: because `mask` lives on the
        # array object, not the data buffer, two views can share one data
        # buffer under *different* masks with no data copy and no
        # interference between them.
        base = np.arange(10)
        v1 = base.view()
        v2 = base.view()

        v1.mask = np.ones(10, dtype=bool)
        m2 = np.zeros(10, dtype=bool)
        m2[3] = True
        v2.mask = m2

        assert v1.mask.all()
        assert np.array_equal(v2.mask, m2)
        assert base.mask is None

        # Views still share the underlying data buffer.
        v1[0] = 999
        assert v2[0] == 999
        assert v1.mask is not v2.mask


class TestMaskReshape:
    """Phase 2: reshape()/ravel()/squeeze() mask propagation."""

    def _masked(self):
        a = np.arange(10)
        a.mask = np.array(
            [False, True, False, False, True,
             False, False, False, False, False])
        return a

    def test_reshape_propagates_mask(self):
        a = self._masked()
        r = a.reshape(2, 5)
        assert r.mask.shape == (2, 5)
        assert np.array_equal(r.mask, a.mask.reshape(2, 5))

    def test_ravel_propagates_mask(self):
        a = self._masked()
        r = a.reshape(2, 5).ravel()
        assert np.array_equal(r.mask, a.mask)

    def test_squeeze_propagates_mask(self):
        a = self._masked()
        sq = a.reshape(1, 10).squeeze()
        assert sq.mask.shape == (10,)
        assert np.array_equal(sq.mask, a.mask)

    def test_unmasked_reshape_stays_unmasked(self):
        a = np.arange(10)
        assert a.reshape(2, 5).mask is None
        assert a.ravel().mask is None


class TestMaskSortPartition:
    """
    Phase 2: sort()/partition() must permute the mask along with the data,
    not leave it stationary -- see TODO.md's original bug report (found by
    probing a `[3, 1, 4, 1, 5]` array: the mask stayed at indices [1, 4]
    while the values that used to be there moved elsewhere).

    These tests use all-distinct values so "which element the mask follows"
    is unambiguous -- with a default (non-stable) sort, tied values have no
    guaranteed relative order, so a duplicate-value case can only check
    that the total masked count is preserved, not which specific tied
    element carries the mask.
    """

    def test_sort_permutes_mask_with_data(self):
        a = np.array([3, 1, 4, 2, 5])
        mask = np.zeros(5, dtype=bool)
        mask[[1, 4]] = True  # marks the elements with value 1 and value 5

        a.mask = mask
        a.sort()

        assert np.array_equal(a, [1, 2, 3, 4, 5])
        # The mask must track the *values* 1 and 5, not the original
        # indices [1, 4] (which now hold 2 and 5).
        assert np.array_equal(a.mask, [True, False, False, False, True])

    def test_partition_permutes_mask_with_data(self):
        a = np.array([3, 1, 4, 2, 5])
        mask = np.zeros(5, dtype=bool)
        mask[[1, 4]] = True

        a.mask = mask
        a.partition(2)

        assert np.array_equal(a.mask, [v in (1, 5) for v in a])

    def test_sort_with_duplicate_values_preserves_mask_count(self):
        a = np.array([3, 1, 4, 1, 5])
        mask = np.zeros(5, dtype=bool)
        mask[[1, 4]] = True  # one of the two `1`s, and the `5`

        a.mask = mask
        a.sort()

        assert np.array_equal(a, [1, 1, 3, 4, 5])
        assert np.sum(a.mask) == 2
        # The `5` is a unique value, so it's unambiguous.
        assert a.mask[np.nonzero(a == 5)[0][0]]

    def test_sort_2d_along_axis_permutes_mask(self):
        a = np.array([[3, 1], [4, 1], [5, 2]])
        mask = np.array([[False, True], [False, False], [True, False]])
        a.mask = mask
        expected = np.sort(np.array([[3, 1], [4, 1], [5, 2]]), axis=0)
        a.sort(axis=0)
        assert np.array_equal(a, expected)
        # Column 0 was [3, 4, 5] with 5 masked -> sorted [3, 4, 5], mask
        # follows the 5 to the last row. Column 1 was [1, 1, 2] with the
        # first 1 masked -> sorted [1, 1, 2]; since both 1s are equal the
        # masked one's new position is whichever the (stable) sort put it,
        # so just check the total count of True values is preserved.
        assert np.array_equal(a.mask[:, 0], [False, False, True])
        assert np.sum(a.mask) == np.sum(mask)

    def test_unmasked_sort_partition_unaffected(self):
        a = np.array([3, 1, 4, 1, 5])
        a.sort()
        assert a.mask is None
        b = np.array([3, 1, 4, 1, 5])
        b.partition(2)
        assert b.mask is None
