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

    def test_bool_array_not_in_use_as_a_mask_can_carry_one(self):
        # Not every bool array is a mask (e.g. comparison-ufunc results,
        # phase 3) -- only an array *currently serving as someone else's
        # mask* is forbidden from carrying one, see
        # test_array_in_use_as_a_mask_cannot_carry_one below.
        m = np.array([True, False, True])
        m.mask = np.array([False, True, False])
        assert m.mask is not None

    def test_array_in_use_as_a_mask_cannot_carry_one(self):
        # Closes the point-in-time hole: without this, `m = arr.mask;
        # m.mask = x` could nest a mask onto a mask after the fact, even
        # though attaching `m` to `arr` was valid at the time.
        a = np.arange(3)
        m = np.array([True, False, True])
        a.mask = m
        with pytest.raises(TypeError):
            m.mask = np.array([False, True, False])

    def test_mask_can_carry_a_mask_after_being_detached(self):
        a = np.arange(3)
        m = np.array([True, False, True])
        a.mask = m
        a.mask = None  # detach; `m` is no longer in use as a's mask
        m.mask = np.array([False, True, False])
        assert m.mask is not None

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


class TestMaskUfuncPropagation:
    """
    Phase 3: elementwise ufunc calls (arithmetic, trig, comparisons, and
    operator overloads that route through the same ufunc call) OR their
    input masks together, broadcast to the output shape. Mask never
    affects computation -- the underlying data is always the same as the
    unmasked equivalent.
    """

    def test_add_ors_masks(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([10, 20, 30, 40])
        a.mask = np.array([True, False, False, False])
        b.mask = np.array([False, False, True, False])

        c = a + b
        assert np.array_equal(c, [11, 22, 33, 44])
        assert np.array_equal(c.mask, [True, False, True, False])

    def test_np_add_matches_operator(self):
        a = np.array([1.0, 2.0, 3.0])
        a.mask = np.array([True, False, False])
        b = np.array([1.0, 1.0, 1.0])

        c1 = a + b
        c2 = np.add(a, b)
        assert np.array_equal(c1.mask, c2.mask)
        assert np.array_equal(c1.mask, [True, False, False])

    def test_one_sided_mask_broadcasts_to_output_shape(self):
        a = np.array([1, 2, 3])
        a.mask = np.array([False, True, False])
        b = np.array([[10], [20]])  # unmasked, broadcasts a to (2, 3)

        c = a + b
        assert c.shape == (2, 3)
        assert np.array_equal(c.mask, [[False, True, False],
                                        [False, True, False]])

    def test_unary_ufunc_propagates_mask(self):
        a = np.array([1.0, 4.0, 9.0])
        a.mask = np.array([False, True, False])
        b = np.sqrt(a)
        assert np.array_equal(b, [1.0, 2.0, 3.0])
        assert np.array_equal(b.mask, [False, True, False])

    def test_trig_propagates_mask(self):
        a = np.array([0.0, np.pi / 2, np.pi])
        a.mask = np.array([True, False, False])
        b = np.sin(a)
        assert np.array_equal(b.mask, [True, False, False])

    def test_comparison_propagates_mask(self):
        # Comparisons produce bool-dtype output -- this is exactly the case
        # that broke a too-strict earlier version of the mask invariant
        # (see "Settled design decisions" in TODO.md): the bool result must
        # still be able to carry its own mask.
        a = np.array([1, 2, 3])
        b = np.array([3, 2, 1])
        a.mask = np.array([True, False, False])

        c = a < b
        assert np.array_equal(c, [True, False, False])
        assert np.array_equal(c.mask, [True, False, False])
        assert c.mask.dtype == bool

    def test_no_masked_inputs_leaves_result_unmasked(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        assert (a + b).mask is None
        assert (a < b).mask is None
        assert np.sin(a).mask is None

    def test_inplace_operator_updates_mask(self):
        a = np.array([1.0, 2.0, 3.0])
        a.mask = np.array([False, True, False])
        b = np.array([10.0, 10.0, 10.0])
        b.mask = np.array([False, False, True])

        a += b
        assert np.array_equal(a, [11.0, 12.0, 13.0])
        assert np.array_equal(a.mask, [False, True, True])

    def test_out_kwarg_updates_mask(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        a.mask = np.array([True, False, False])
        out = np.zeros(3, dtype=a.dtype)

        result = np.add(a, b, out=out)
        assert result is out
        assert np.array_equal(out.mask, [True, False, False])

    def test_result_mask_is_independent_new_array(self):
        # The result's mask must be its own buffer, not aliasing an input's.
        a = np.array([1, 2, 3])
        a.mask = np.array([True, False, False])
        b = np.array([4, 5, 6])

        c = a + b
        c.mask[0] = False
        assert a.mask[0] == True

    def test_where_kwarg_with_explicit_out_is_a_selective_write(self):
        # `where=` + a real `out=` is treated as a masked write: at
        # positions `where` excludes, `out` keeps both its old *data*
        # (guaranteed by the wrapped call itself) and its old *mask*
        # (merged here via `np.where(where, new_mask, old_mask)`).
        a = np.array([1, 2, 3])
        a.mask = np.array([True, False, False])
        b = np.array([4, 5, 6])
        out = np.zeros(3, dtype=int)
        r = np.add(a, b, out=out, where=np.array([True, True, False]))
        assert r is out
        assert np.array_equal(r, [5, 7, 0])
        # position 0: where=True -> new mask (True, from a)
        # position 1: where=True -> new mask (False)
        # position 2: where=False -> old mask (out had none -> False)
        assert np.array_equal(r.mask, [True, False, False])

    def test_where_kwarg_preserves_old_mask_at_excluded_positions(self):
        # Same as above, but `out` already had its own mask before the
        # call -- the excluded position must keep exactly that, not just
        # "unmasked".
        a = np.array([1, 2, 3])
        a.mask = np.array([True, False, False])
        b = np.array([4, 5, 6])
        out = np.array([10, 20, 30])
        out.mask = np.array([False, False, True])
        r = np.add(a, b, out=out, where=np.array([True, True, False]))
        assert np.array_equal(r, [5, 7, 30])
        # position 2 (where=False) must keep out's own old mask (True),
        # not the propagated-but-unused new_mask (False from a/b).
        assert np.array_equal(r.mask, [True, False, True])

    def test_where_kwarg_without_explicit_out_still_skips_propagation(self):
        # Documented, narrower gap: `where=` with a freshly allocated
        # `out=None` leaves excluded positions *uninitialized*, not "old,
        # valid, differently-masked data" -- there is no principled old
        # mask to merge with, so mask propagation is skipped entirely,
        # same as phase 3's original (broader) gap.
        a = np.array([1, 2, 3])
        a.mask = np.array([True, False, False])
        b = np.array([4, 5, 6])
        r = np.add(a, b, out=None, where=np.array([True, True, False]))
        assert np.array_equal(r[:2], [5, 7])
        assert r.mask is None

    def test_where_kwarg_with_multi_output_ufunc_skips_propagation(self):
        # Documented, narrower gap: `where=` combined with a multi-output
        # ufunc (e.g. `divmod`) is not handled -- rare combination.
        x = np.array([7, 8, 9])
        x.mask = np.array([True, False, False])
        y = np.array([2, 2, 2])
        with pytest.warns(UserWarning, match="uninitialized memory"):
            q, r = np.divmod(x, y, where=np.array([True, True, False]))
        assert q.mask is None
        assert r.mask is None

    def test_multi_output_ufunc_propagates_mask_to_each_output(self):
        x = np.array([7, 8, 9])
        x.mask = np.array([True, False, False])
        y = np.array([2, 2, 2])

        q, r = np.divmod(x, y)
        assert np.array_equal(q, [3, 4, 4])
        assert np.array_equal(r, [1, 0, 1])
        assert np.array_equal(q.mask, [True, False, False])
        assert np.array_equal(r.mask, [True, False, False])


class TestMaskReduceLike:
    """Phase 4 (see TODO.md): `reduce`/`accumulate`/`reduceat`/`outer`/
    `at` -- the ufunc methods, which don't go through
    `ufunc_generic_vectorcall` and so need their own mask handling.
    Convention: run the identical method again with `logical_or` on the
    mask(s) instead of `ufunc` on the data.
    """

    def test_reduce_mask_is_or_of_contributing_elements(self):
        # keepdims=True keeps the result a 0-d-avoiding real array; a full
        # reduction without it decays to a scalar, see the dedicated gap
        # test below (same root cause as phase 3's scalar-decay gap).
        x = np.array([1, 2, 3, 4])
        x.mask = np.array([False, True, False, False])
        r = np.add.reduce(x, keepdims=True)
        assert r[0] == 10
        assert bool(r.mask[0]) is True

    def test_reduce_no_mask_stays_unmasked(self):
        x = np.array([1, 2, 3, 4])
        r = np.add.reduce(x, keepdims=True)
        assert r.mask is None

    def test_full_reduce_decays_to_scalar_drops_mask(self):
        # Known, documented gap (same root cause as phase 3's ufunc
        # scalar-decay gap): a full reduction (no axis, keepdims=False)
        # returns a plain numpy scalar, which has no `mask` field.
        x = np.array([1, 2, 3, 4])
        x.mask = np.array([False, True, False, False])
        r = np.add.reduce(x)
        assert r == 10
        assert not hasattr(r, "mask")

    def test_reduce_with_axis_masks_only_affected_rows(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        x.mask = np.array([[False, False], [True, False], [False, False]])
        r = np.add.reduce(x, axis=1)
        assert np.array_equal(r, [3, 7, 11])
        assert np.array_equal(r.mask, [False, True, False])

    def test_reduce_where_excludes_masked_element_from_result_mask(self):
        # `where=` for reduce excludes *inputs* from the combine (every
        # output position is still fully (re)written), so applying the
        # same `where` to the mask combine correctly drops an excluded
        # masked element's contribution too.
        x = np.array([1, 2, 3])
        x.mask = np.array([False, True, False])
        r = np.add.reduce(
                x, where=np.array([True, False, True]), initial=0,
                keepdims=True)
        assert r[0] == 4  # 1 + 3, element 2 (value, masked) excluded
        # Excluding the only masked element leaves an all-False mask,
        # which canonicalizes back to NULL (see "Unmasked representation").
        assert r.mask is None

    def test_reduce_empty_slice_identity_is_not_masked(self):
        # An identity-filled slot (from `initial=`) contributes no real
        # data, so it must never read as masked, even though the *only*
        # element of the array is masked but excluded by `where=`.
        x = np.array([5])
        x.mask = np.array([True])
        r = np.add.reduce(x, where=np.array([False]), initial=0, keepdims=True)
        assert r[0] == 0
        assert r.mask is None

    def test_sum_mean_max_min_any_all_inherit_mask_from_reduce(self):
        # calculation.c's .sum()/.mean()/.max()/.min()/.any()/.all() all
        # route through PyArray_GenericReduceFunction -> PyUFunc_Reduce,
        # so they need no code of their own -- confirm that's really so.
        # axis= keeps the result a real array rather than decaying to a
        # scalar (see the dedicated scalar-decay gap test above).
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        x.mask = np.array([[False, True, False, False]])
        assert bool(x.sum(axis=1).mask[0]) is True
        assert bool(x.mean(axis=1).mask[0]) is True
        assert bool(x.max(axis=1).mask[0]) is True
        assert bool(x.min(axis=1).mask[0]) is True
        assert bool(np.any(x > 10, axis=1).mask[0]) is True
        assert bool(np.all(x > 0, axis=1).mask[0]) is True

    def test_std_var_mask_spreads_across_whole_axis(self):
        # Deliberate, documented consequence of composition (not a
        # separate code path): `x - mean(x)` OR's every element's mask
        # with the (already axis-wide) mean's mask, so if *any* element
        # in an axis is masked, std/var for that whole axis reads masked.
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        x.mask = np.array([[False, False], [True, False]])
        std = x.std(axis=1)
        var = x.var(axis=1)
        assert np.array_equal(std.mask, [False, True])
        assert np.array_equal(var.mask, [False, True])

    def test_argmax_argmin_ignore_mask_when_selecting(self):
        # argmax/argmin return an index, not data -- like argsort (phase
        # 2), the result carries no mask -- and the masked element must
        # still fully participate in the comparison (mask never affects
        # computation).
        x = np.array([1.0, 9.0, 3.0])
        x.mask = np.array([False, True, False])
        assert x.argmax() == 1  # still picks the masked, larger element
        assert not hasattr(x.argmax(), "mask")

    def test_accumulate_mask_is_cumulative_or(self):
        x = np.array([1, 2, 3, 4])
        x.mask = np.array([False, True, False, False])
        r = np.add.accumulate(x)
        assert np.array_equal(r, [1, 3, 6, 10])
        assert np.array_equal(r.mask, [False, True, True, True])

    def test_accumulate_no_mask_stays_unmasked(self):
        x = np.array([1, 2, 3, 4])
        r = np.add.accumulate(x)
        assert r.mask is None

    def test_accumulate_non_contiguous_mask_matches_contiguous(self):
        # Perf fast path (see BENCHMARKS.md P1) only applies to a 1-d,
        # C-contiguous mask; anything else falls back to the general
        # PyUFunc_Accumulate(logical_or, ...) path. Cross-check the
        # fallback against a manual `logical_or.accumulate` to confirm
        # both paths agree.
        y = np.arange(10)
        y.mask = (np.arange(10) % 3 == 0)
        yv = y[::2]  # non-unit stride -> non-contiguous mask view
        assert not yv.mask.flags["C_CONTIGUOUS"]
        r = np.add.accumulate(yv)
        assert np.array_equal(r.mask, np.logical_or.accumulate(yv.mask))

    def test_std_var_full_reduction_scalar_decay_does_not_crash(self):
        # Same documented scalar-decay gap as phases 3/4's reduce path:
        # a full (single-axis, 1-d input) reduction decays to a plain
        # scalar with no `mask` field -- must not crash attaching a mask.
        a = np.array([1.0, 2.0, 3.0])
        a.mask = np.array([True, False, False])
        s = a.std(axis=0)
        v = a.var(axis=0)
        assert not hasattr(s, "mask")
        assert not hasattr(v, "mask")

    def test_reduceat_mask_is_or_within_each_segment(self):
        x = np.array([1, 2, 3, 4, 5])
        x.mask = np.array([False, True, False, False, False])
        # segments: [0:2), [2:4), [4:5) -> sums [3, 7, 5]
        r = np.add.reduceat(x, [0, 2, 4])
        assert np.array_equal(r, [3, 7, 5])
        assert np.array_equal(r.mask, [True, False, False])

    def test_outer_mask_shape_matches_concatenated_shapes(self):
        a = np.array([1, 2])
        a.mask = np.array([False, True])
        b = np.array([10, 20, 30])
        r = np.add.outer(a, b)
        assert r.shape == (2, 3)
        assert np.array_equal(
                r.mask, [[False, False, False], [True, True, True]])

    def test_outer_one_sided_mask_broadcasts_correctly(self):
        a = np.array([1, 2])
        b = np.array([10, 20, 30])
        b.mask = np.array([False, True, False])
        r = np.add.outer(a, b)
        assert np.array_equal(
                r.mask, [[False, True, False], [False, True, False]])

    def test_outer_no_mask_stays_unmasked(self):
        a = np.array([1, 2])
        b = np.array([10, 20, 30])
        r = np.add.outer(a, b)
        assert r.mask is None

    def test_at_binary_ors_mask_at_touched_positions_only(self):
        a = np.array([1, 2, 3])
        a.mask = np.array([True, False, False])
        b = np.array([10, 20])
        np.add.at(a, [1, 2], b)
        assert np.array_equal(a, [1, 12, 23])
        # position 0 untouched -> keeps its own old mask.
        assert np.array_equal(a.mask, [True, False, False])

    def test_at_binary_propagates_mask_from_second_operand(self):
        a = np.array([1, 2, 3])
        b = np.array([10, 20])
        b.mask = np.array([True, False])
        np.add.at(a, [0, 1], b)
        assert np.array_equal(a, [11, 22, 3])
        assert np.array_equal(a.mask, [True, False, False])

    def test_at_binary_repeated_indices_accumulate_mask_correctly(self):
        a = np.array([0])
        a.mask = np.array([False])
        b = np.array([1, 2, 3])
        b.mask = np.array([False, True, False])
        np.add.at(a, [0, 0, 0], b)
        assert a[0] == 6
        # OR is associative/commutative/idempotent -- any masked
        # contribution anywhere in the run of repeats marks the result.
        assert bool(a.mask[0]) is True

    def test_at_unary_does_not_touch_mask(self):
        a = np.array([1, 2, 3])
        a.mask = np.array([False, True, False])
        np.negative.at(a, [0, 1])
        assert np.array_equal(a, [-1, -2, 3])
        assert np.array_equal(a.mask, [False, True, False])

    def test_at_no_mask_stays_unmasked(self):
        a = np.array([1, 2, 3])
        b = np.array([10, 20])
        np.add.at(a, [0, 1], b)
        assert a.mask is None


class TestMaskCoreTransport:
    """Phase 5: transpose-family and broadcast mask propagation."""

    def _masked_2d(self, dtype=np.intp):
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        a.mask = np.array([[False, True, False], [True, False, False]])
        return a

    def test_transpose_propagates_mask(self):
        a = self._masked_2d()
        assert np.array_equal(a.T.mask, a.mask.T)
        assert np.array_equal(a.transpose().mask, a.mask.T)

    def test_swapaxes_propagates_mask(self):
        a = self._masked_2d()
        result = a.swapaxes(0, 1)
        assert np.array_equal(result.mask, a.mask.swapaxes(0, 1))

    def test_moveaxis_propagates_mask(self):
        a = self._masked_2d()
        result = np.moveaxis(a, 0, 1)
        assert np.array_equal(result.mask, np.moveaxis(a.mask, 0, 1))

    def test_broadcast_to_propagates_mask(self):
        a = np.arange(3)
        a.mask = np.array([False, True, False])
        result = np.broadcast_to(a, (2, 3))
        assert result.shape == (2, 3)
        assert np.array_equal(result.mask, np.broadcast_to(a.mask, (2, 3)))

    def test_same_dtype_view_preserves_mask(self):
        a = self._masked_2d()
        result = a.view(a.dtype)
        assert np.array_equal(result.mask, a.mask)
        assert np.shares_memory(result.mask, a.mask)

    def test_unmasked_dtype_view_unchanged(self):
        a = np.arange(4, dtype=np.int32)
        assert a.view(np.float32).mask is None

    def test_astype_copies_mask(self):
        a = self._masked_2d()
        result = a.astype(np.float64)
        assert result.dtype == np.float64
        assert np.array_equal(result.mask, a.mask)
        assert not np.shares_memory(result.mask, a.mask)
        result.mask[0, 0] = True
        assert not a.mask[0, 0]

    def test_astype_keeps_mask_for_orders_and_no_copy(self):
        a = self._masked_2d()
        f = a.astype(np.int32, order='F')
        assert np.array_equal(f.mask, a.mask)
        assert f.flags.f_contiguous and f.mask.flags.f_contiguous
        assert np.array_equal(a.T.astype(np.int32).mask, a.mask.T)
        # No-op astype returns self, mask untouched.
        assert a.astype(a.dtype, copy=False) is a
        assert a.astype(a.dtype, copy=False).mask is a.mask

    def test_astype_unmasked_stays_unmasked(self):
        assert np.arange(4).astype(float).mask is None

    def test_dtype_conversion_constructors_keep_mask(self):
        a = self._masked_2d()
        assert np.array_equal(np.asarray(a, dtype=np.float32).mask, a.mask)
        assert np.array_equal(np.array(a, dtype=np.float32).mask, a.mask)

    def test_flatten_copies_mask(self):
        a = self._masked_2d()
        for order in "CFA":
            result = a.flatten(order)
            assert np.array_equal(result.mask, a.mask.flatten(order))
            assert not np.shares_memory(result.mask, a.mask)
        assert np.arange(3).flatten().mask is None

    def test_diagonal_propagates_mask(self):
        def full(arr):  # all-False canonicalizes to None
            return np.zeros(arr.shape, bool) if arr.mask is None else arr.mask

        a = np.arange(12).reshape(3, 4)
        a.mask = (a % 5 == 0)
        for offset in (-2, -1, 0, 1, 3):
            assert np.array_equal(full(a.diagonal(offset)),
                                  a.mask.diagonal(offset))
        assert np.array_equal(a.diagonal(0).mask, [True, True, True])
        b = np.arange(24).reshape(2, 3, 4)
        b.mask = (b % 3 == 0)
        assert np.array_equal(b.diagonal(-1, 2, 1).mask,
                              b.mask.diagonal(-1, 2, 1))
        assert np.arange(9).reshape(3, 3).diagonal().mask is None

    def test_as_strided_carries_mask(self):
        from numpy.lib.stride_tricks import as_strided
        a = self._masked_2d()
        # Default strides, and explicit byte strides (transposed / windowed).
        assert np.array_equal(as_strided(a, a.shape).mask, a.mask)
        result = as_strided(a, shape=(3, 2), strides=(8, 24))
        assert np.array_equal(result.mask, a.mask.T)
        result = as_strided(a, shape=(2, 2, 2), strides=(24, 8, 8))
        assert np.array_equal(result.mask[0, 0], a.mask[0, :2])
        assert np.array_equal(result.mask[0, 1], a.mask[0, 1:])
        # The mask buffer is shared like the data buffer is.
        assert np.shares_memory(result.mask, a.mask)

    def test_as_strided_masked_fortran_and_sliced(self):
        from numpy.lib.stride_tricks import as_strided
        a = np.asfortranarray(self._masked_2d())
        assert a.mask.flags.f_contiguous  # copy keeps mask layout like data
        result = as_strided(a, a.shape, a.strides)
        assert np.array_equal(result.mask, a.mask)
        b = self._masked_2d()[:, ::2]
        result = as_strided(b, b.shape, b.strides)
        assert np.array_equal(result.mask, b.mask)

    def test_as_strided_without_mapping_drops_mask(self):
        from numpy.lib.stride_tricks import as_strided
        a = self._masked_2d()
        # Stride not a whole number of elements: no mask mapping (fail-open).
        assert as_strided(a, shape=(3,), strides=(4,)).mask is None
        # Mask layout not proportional to the data layout: fail-open too.
        a.mask = np.asfortranarray(a.mask)
        assert as_strided(a, a.shape, a.strides).mask is None

    def test_as_strided_unmasked_stays_unmasked(self):
        from numpy.lib.stride_tricks import as_strided
        b = np.arange(6)
        assert as_strided(b, (3,), (16,)).mask is None

    def test_sliding_window_view_carries_mask(self):
        from numpy.lib.stride_tricks import sliding_window_view
        a = np.arange(10)
        a.mask = (a % 4 == 3)
        result = sliding_window_view(a, 3)
        assert result.shape == (8, 3)
        assert np.array_equal(result.mask, sliding_window_view(a.mask, 3))
        b = self._masked_2d()
        result = sliding_window_view(b, 2, axis=1)
        assert np.array_equal(result.mask,
                              sliding_window_view(b.mask, 2, axis=1))

    def test_real_imag_views_share_mask(self):
        a = self._masked_2d(np.complex128)
        for part in (a.real, a.imag):
            assert np.array_equal(part.mask, a.mask)
            assert np.shares_memory(part.mask, a.mask)

    def test_imag_of_real_array_keeps_mask(self):
        a = self._masked_2d(np.float64)
        assert np.array_equal(a.imag.mask, a.mask)
        assert np.array_equal(a.real.mask, a.mask)

    def test_same_itemsize_dtype_view_shares_mask(self):
        a = self._masked_2d(np.int32)
        for dt in (np.float32, np.uint32, [("x", "i4")]):
            v = a.view(dt)
            assert v.shape == a.shape
            assert np.array_equal(v.mask, a.mask)
            assert np.shares_memory(v.mask, a.mask)

    def test_itemsize_changing_view_is_rejected(self):
        a = self._masked_2d(np.int32)
        for dt in (np.int64, np.int16, np.int8):
            with pytest.raises(ValueError, match="keep the itemsize"):
                a.view(dt)
        # the failed views leave the original untouched
        assert a.mask is not None and a.mask.sum() == 2


class TestMaskIndexing:
    """Phase 6: advanced/fancy and boolean indexing, assignment, `.flat`."""

    def _a(self):
        a = np.arange(6)
        a.mask = np.array([False, True, False, True, False, False])
        return a

    def _b(self):
        b = np.arange(12).reshape(3, 4)
        b.mask = (b % 5 == 0) | (b == 7)
        return b

    # ----- getitem -----------------------------------------------------
    def test_integer_array_indexing_propagates_mask(self):
        a = self._a()
        result = a[np.array([4, 1, 3, 0])]
        assert np.array_equal(result, [4, 1, 3, 0])
        assert np.array_equal(result.mask, [False, True, True, False])
        assert not np.shares_memory(result.mask, a.mask)  # copy, like data

    def test_negative_and_repeated_indices(self):
        a = self._a()
        assert np.array_equal(a[[-5, 1, 1, -3]].mask, [True, True, True, True])
        assert np.array_equal(a[[]].shape, (0,))

    def test_broadcasted_integer_indexing_propagates_mask(self):
        a = np.arange(6).reshape(2, 3)
        a.mask = np.array([[False, True, False], [True, False, False]])
        rows = np.array([[0], [1]])
        cols = np.array([[1, 2, 0], [2, 0, 1]])
        result = a[rows, cols]
        expected = a.mask[rows, cols]
        assert result.shape == (2, 3)
        assert np.array_equal(result.mask, expected)
        assert np.array_equal(result.mask,
                              [[True, False, False], [False, True, False]])

    def test_boolean_indexing_propagates_selected_mask(self):
        a = self._a()
        result = a[np.array([True, False, True, True, False, False])]
        assert np.array_equal(result, [0, 2, 3])
        assert np.array_equal(result.mask, [False, False, True])

    def test_boolean_indexing_nd(self):
        b = self._b()
        sel = (b % 2 == 0)
        result = b[sel]
        assert result.ndim == 1
        assert np.array_equal(result.mask, b.mask[sel])
        # boolean index along the first axis only
        rows = np.array([True, False, True])
        assert np.array_equal(b[rows].mask, b.mask[rows])

    def test_mixed_fancy_and_slice(self):
        b = self._b()
        for idx in [(slice(None), [0, 2]), ([0, 2], slice(None)),
                    ([0, 2], slice(1, 3)), (Ellipsis, [3, 0]),
                    ([[0], [2]], [1, 3]), (1, [0, 1]), ([0, 1], 2)]:
            result = b[idx]
            expected = b.mask[idx]
            got = (result.mask if result.mask is not None
                   else np.zeros(result.shape, bool))
            assert np.array_equal(got, expected), idx

    def test_advanced_indexing_unmasked_input_stays_unmasked(self):
        a = np.arange(6)
        assert a[np.array([4, 1, 3])].mask is None
        assert a[a > 2].mask is None

    def test_all_false_selection_canonicalizes(self):
        a = self._a()
        assert a[[0, 2, 4]].mask is None

    def test_index_out_of_bounds_raises_and_leaves_state(self):
        a = self._a()
        before = a.mask.copy()
        with pytest.raises(IndexError):
            a[[0, 99]]
        assert np.array_equal(a.mask, before)

    def test_using_the_mask_as_an_index(self):
        a = self._a()
        result = a[a.mask]
        assert np.array_equal(result, [1, 3])
        assert np.array_equal(result.mask, [True, True])

    # ----- setitem -----------------------------------------------------
    def test_indexed_assignment_clears_mask_for_unmasked_rhs(self):
        a = self._a()
        a.mask = np.array([False, True, True, False, False, False])
        a[np.array([1, 2])] = np.array([10, 20])
        assert np.array_equal(a, [0, 10, 20, 3, 4, 5])
        # Internal ops do not re-scan to canonicalize an all-False mask.
        assert not np.asarray(a.mask).any()

    def test_indexed_assignment_keeps_other_mask_entries(self):
        a = self._a()
        a[[0, 1]] = 99
        assert np.array_equal(a.mask, [False, False, False, True, False, False])

    def test_indexed_assignment_propagates_mask_from_rhs(self):
        a = np.arange(6)
        rhs = np.array([10, 20])
        rhs.mask = np.array([True, False])
        a[np.array([1, 4])] = rhs
        assert np.array_equal(a, [0, 10, 2, 3, 20, 5])
        assert np.array_equal(a.mask, [False, True, False, False, False, False])

    def test_boolean_assignment_propagates_mask_from_rhs(self):
        a = np.arange(5)
        a.mask = np.array([True, False, False, False, False])
        rhs = np.array([10, 20])
        rhs.mask = np.array([False, True])
        a[np.array([False, True, True, False, False])] = rhs
        assert np.array_equal(a, [0, 10, 20, 3, 4])
        assert np.array_equal(a.mask, [True, False, True, False, False])

    def test_boolean_assignment_scalar_clears_selected(self):
        a = self._a()
        a[a.mask] = 0
        assert not np.asarray(a.mask).any()

    def test_slice_and_integer_assignment(self):
        b = self._b()
        b[0, :] = 0
        assert not b.mask[0].any()
        assert np.array_equal(b.mask[1:], (np.arange(12).reshape(3, 4) % 5 == 0)[1:]
                              | (np.arange(12).reshape(3, 4) == 7)[1:])
        c = self._a()
        c[1] = 5
        assert not c.mask[1] and c.mask[3]
        c[...] = 0
        assert not np.asarray(c.mask).any()

    def test_masked_rhs_into_unmasked_array_creates_mask(self):
        a = np.arange(6)
        rhs = np.array([7, 8, 9])
        rhs.mask = np.array([False, True, False])
        a[[0, 2, 4]] = rhs
        assert np.array_equal(a.mask, [False, False, True, False, False, False])
        b = np.arange(4)
        b[1] = rhs[1]  # scalar element: masked-ness cannot ride a scalar
        assert b.mask is None

    def test_masked_slice_assignment(self):
        a = np.arange(6)
        rhs = np.array([7, 8, 9])
        rhs.mask = np.array([True, False, True])
        a[1:4] = rhs
        assert np.array_equal(a.mask, [False, True, False, True, False, False])

    def test_assignment_swap_with_overlap(self):
        a = self._a()
        a[[0, 1]] = a[[1, 0]]
        assert np.array_equal(a, [1, 0, 2, 3, 4, 5])
        assert np.array_equal(a.mask, [True, False, False, True, False, False])

    def test_assigning_into_a_mask_is_plain(self):
        a = self._a()
        m = a.mask
        rhs = np.array([True, True])
        rhs.mask = np.array([True, False])
        m[[0, 2]] = rhs  # data assignment only: a mask cannot carry a mask
        assert m[0] and m[2]
        assert m.mask is None

    def test_read_only_mask_fails_before_touching_data(self):
        a = self._a()
        a.mask.flags.writeable = False
        with pytest.raises(ValueError, match="read-only"):
            a[[0, 1]] = 99
        assert np.array_equal(a, np.arange(6))

    def test_inplace_op_through_index_keeps_masks(self):
        a = self._a()
        a[[0, 1, 4]] += 100
        assert np.array_equal(a, [100, 101, 2, 3, 104, 5])
        assert np.array_equal(a.mask, [False, True, False, True, False, False])

    def test_structured_field_assignment_leaves_mask_alone(self):
        s = np.zeros(3, dtype=[("x", "i4"), ("y", "f8")])
        s.mask = np.array([False, True, False])
        s["x"] = 7
        assert np.array_equal(s.mask, [False, True, False])

    # ----- .flat -------------------------------------------------------
    def test_flat_slice_and_fancy_get(self):
        b = self._b()
        for idx in [slice(1, 9, 2), [0, 5, 7], slice(None)]:
            result = b.flat[idx]
            expected = b.mask.flat[idx]
            got = (result.mask if result.mask is not None
                   else np.zeros(result.shape, bool))
            assert np.array_equal(got, expected), idx

    def test_flat_bool_get_and_non_contiguous(self):
        b = self._b().T  # non-contiguous, C-order flat differs from memory
        sel = np.zeros(12, bool)
        sel[[0, 3, 6, 9]] = True
        result = b.flat[sel]
        assert np.array_equal(result.mask, b.mask.flat[sel])

    def test_flat_scalar_get_is_a_scalar(self):
        # Documented gap: a scalar cannot carry a mask.
        assert isinstance(self._a().flat[1], np.generic)

    def test_flat_assignment_updates_mask(self):
        b = self._b()
        b.flat[[0, 5]] = 1
        assert not b.mask[0, 0] and not b.mask[1, 1]
        rhs = np.array([1, 2])
        rhs.mask = np.array([True, False])
        c = np.arange(6).reshape(2, 3)
        c.flat[[1, 4]] = rhs
        assert np.array_equal(c.mask, [[False, True, False], [False, False, False]])

    def test_flat_copy_and_array(self):
        b = self._b()
        assert np.array_equal(b.flat.copy().mask, b.mask.flatten())
        assert np.array_equal(np.asarray(b.flat).mask, b.mask.flatten())
        t = b.T
        assert np.array_equal(t.flat.copy().mask, t.mask.flatten())
        assert np.array_equal(np.asarray(t.flat).mask, t.mask.flatten())


class TestMaskGatherScatter:
    """Phase 7: take/repeat/choose/where/concatenate/put/putmask/place/
    copyto/fill/flat-assignment, and in-place metadata mutators."""

    def _a(self):
        a = np.arange(6).reshape(2, 3)
        a.mask = np.array([[False, True, False], [True, False, False]])
        return a

    @staticmethod
    def _d():
        """Plain (unmasked) copy of the data of `_a()`, for conditions."""
        return np.arange(6).reshape(2, 3)

    @staticmethod
    def _full(x, shape=None):
        shape = x.shape if shape is None else shape
        return np.zeros(shape, bool) if x.mask is None else np.asarray(x.mask)

    # ----- gather ------------------------------------------------------
    def test_take_propagates(self):
        a = self._a()
        assert np.array_equal(a.take([4, 1, 3]).mask, [False, True, True])
        assert np.array_equal(a.take([2, 0], axis=1).mask, a.mask.take([2, 0], axis=1))
        assert np.array_equal(a.take([7, 1], mode="wrap").mask, [True, True])
        assert np.array_equal(a.take([9, 1], mode="clip").mask, [False, True])
        assert not np.shares_memory(a.take([1]).mask, a.mask)

    def test_take_out_replaces_mask(self):
        a = self._a()
        out = np.empty(3, a.dtype)
        r = a.take([4, 1, 3], out=out)
        assert r is out
        assert np.array_equal(out.mask, [False, True, True])
        # A stale mask on `out` disappears when nothing masked is taken.
        out2 = np.zeros(2, a.dtype)
        out2.mask = np.array([True, True])
        a.take([0, 2], out=out2)
        assert out2.mask is None

    def test_take_unmasked_and_scalar(self):
        assert np.arange(4).take([1, 2]).mask is None
        assert isinstance(self._a().take(1), np.generic)  # scalar: no mask

    def test_repeat_and_tile(self):
        a = self._a()
        assert np.array_equal(a.repeat(2).mask, a.mask.repeat(2))
        assert np.array_equal(a.repeat([1, 2, 1], axis=1).mask,
                              a.mask.repeat([1, 2, 1], axis=1))
        assert np.array_equal(np.tile(a, (2, 2)).mask, np.tile(a.mask, (2, 2)))
        assert np.arange(3).repeat(2).mask is None

    def test_compress_extract(self):
        a = self._a()
        assert np.array_equal(a.compress([True, False, True], axis=1).mask,
                              a.mask.compress([True, False, True], axis=1))
        sel = a.ravel() > 0
        assert np.array_equal(np.extract(sel, a).mask,
                              a.mask.ravel()[sel])

    def test_choose(self):
        a = self._a()
        b = np.arange(6).reshape(2, 3) * 10
        b.mask = np.array([[True, False, False], [False, False, True]])
        sel = np.array([[0, 1, 0], [1, 0, 1]])
        r = np.choose(sel, [a, b])
        assert np.array_equal(r, np.choose(sel, [np.asarray(a), np.asarray(b)]))
        expected = np.where(sel == 0, a.mask, b.mask)
        assert np.array_equal(self._full(r), expected)

    def test_choose_masked_selector_and_unmasked_choices(self):
        sel = np.array([0, 1, 0, 1])
        sel.mask = np.array([False, True, False, False])
        r = np.choose(sel, [np.arange(4), np.arange(4) * 2])
        assert np.array_equal(r.mask, [False, True, False, False])
        c = np.arange(4)
        c.mask = np.array([True, False, False, False])
        r = np.choose(np.zeros(4, int), [c, 5])  # scalar choice
        assert np.array_equal(r.mask, [True, False, False, False])

    def test_choose_out_clears_stale_mask(self):
        out = np.zeros(2, int)
        out.mask = np.array([True, True])
        np.choose(np.array([0, 1]), [np.arange(2), np.arange(2)], out=out)
        assert out.mask is None

    def test_where(self):
        a = self._a()
        cond = self._d() > 2
        r = np.where(cond, a, 0)
        assert np.array_equal(self._full(r), np.where(cond, a.mask, False))
        r = np.where(cond, 0, a)
        assert np.array_equal(self._full(r), np.where(cond, False, a.mask))
        # broadcast operands: unmasked y wider than masked x
        x = np.array([1, 2, 3]); x.mask = np.array([False, True, False])
        r = np.where(np.array([[True], [False]]), x, np.zeros((2, 3), int))
        assert r.shape == (2, 3)
        assert np.array_equal(r.mask, [[False, True, False], [False] * 3] )

    def test_where_masked_condition_hides_result(self):
        cond = np.array([True, False, True])
        cond.mask = np.array([False, True, False])
        r = np.where(cond, 1, 2)
        assert np.array_equal(r.mask, [False, True, False])

    def test_where_one_argument_and_unmasked(self):
        a = self._a()
        assert all(i.mask is None for i in np.where(a > 1))
        assert np.where(np.arange(3) > 1, 1, 2).mask is None

    def test_where_derived_helpers(self):
        a = self._a()
        assert np.array_equal(self._full(np.tril(a)),
                              np.where(np.tri(2, 3, dtype=bool), a.mask, False))
        r = np.select([self._d() > 3, self._d() <= 3], [a, -a])
        assert np.array_equal(self._full(r), a.mask)

    def test_index_returning_ops_are_mask_blind(self):
        a = self._a()
        assert np.argsort(a).mask is None
        assert np.arange(10).searchsorted(a).mask is None
        assert np.argwhere(a).mask is None
        assert all(i.mask is None for i in a.nonzero())

    # ----- combine -----------------------------------------------------
    def test_concatenate_axes(self):
        a = self._a()
        for axis in (0, 1, -1):
            r = np.concatenate([a, a], axis=axis)
            assert np.array_equal(r.mask,
                                  np.concatenate([a.mask, a.mask], axis=axis))
        r = np.concatenate([a, a], axis=None)
        assert np.array_equal(r.mask, np.concatenate([a.mask.ravel()] * 2))

    def test_concatenate_mixed_masked_unmasked(self):
        a = self._a()
        r = np.concatenate([np.zeros((1, 3), int), a])
        assert np.array_equal(r.mask, np.vstack([np.zeros((1, 3), bool), a.mask]))
        assert np.array_equal(np.concatenate([a, [[1, 2, 3]]]).mask,
                              np.vstack([a.mask, np.zeros((1, 3), bool)]))
        assert np.concatenate([np.arange(3), np.arange(3)]).mask is None

    def test_concatenate_dtype_and_out(self):
        a = self._a()
        r = np.concatenate([a, a], dtype=float)
        assert r.dtype == float
        assert np.array_equal(r.mask, np.vstack([a.mask, a.mask]))
        out = np.empty((4, 3), a.dtype)
        r = np.concatenate([a, a], out=out)
        assert r is out
        assert np.array_equal(out.mask, np.vstack([a.mask, a.mask]))
        stale = np.zeros((4, 3), a.dtype)
        stale.mask = np.ones((4, 3), bool)
        np.concatenate([np.zeros((2, 3), int)] * 2, out=stale)
        assert stale.mask is None

    def test_stacking_helpers(self):
        a = self._a()
        m = a.mask
        assert np.array_equal(np.stack([a, a]).mask, np.stack([m, m]))
        assert np.array_equal(np.vstack([a, a]).mask, np.vstack([m, m]))
        assert np.array_equal(np.hstack([a, a]).mask, np.hstack([m, m]))
        assert np.array_equal(np.dstack([a, a]).mask, np.dstack([m, m]))
        assert np.array_equal(np.column_stack([a, a]).mask, np.column_stack([m, m]))
        assert np.array_equal(np.block([[a, a], [a, a]]).mask,
                              np.block([[m, m], [m, m]]))
        assert np.array_equal(np.append(a, a, axis=0).mask, np.vstack([m, m]))

    def test_split_helpers(self):
        a = self._a()
        parts = np.split(a, 3, axis=1)
        for i, p in enumerate(parts):
            assert np.array_equal(self._full(p), a.mask[:, i:i + 1])
        assert np.array_equal(self._full(np.array_split(a, 2, axis=1)[0]),
                              a.mask[:, :2])

    # ----- scatter -----------------------------------------------------
    def test_put_takes_masked_ness_of_values(self):
        a = self._a()
        a.put([1, 4], [50, 60])
        assert np.array_equal(a.mask, [[False, False, False], [True, False, False]])
        b = np.arange(6)
        vals = np.array([7, 8]); vals.mask = np.array([True, False])
        b.put([2, 4], vals)
        assert np.array_equal(b.mask, [False, False, True, False, False, False])
        c = self._a()
        c.put([7], [9], mode="wrap")  # -> flat index 1
        assert not c.mask[0, 1]
        np.put(c, [0], [1])
        assert not c.mask[0, 0]

    def test_put_error_leaves_mask_alone(self):
        a = self._a()
        before = a.mask.copy()
        with pytest.raises(IndexError):
            a.put([99], [1])
        assert np.array_equal(a.mask, before)

    def test_putmask(self):
        a = self._a()
        np.putmask(a, self._d() >= 1, 0)
        assert not np.asarray(a.mask).any()
        b = np.arange(5)
        vals = np.array([10, 20]); vals.mask = np.array([False, True])
        np.putmask(b, np.array([True, False, True, True, False]), vals)
        # putmask cycles by absolute position: index n takes vals[n % 2]
        assert np.array_equal(b, [10, 1, 10, 20, 4])
        assert np.array_equal(b.mask, [False, False, False, True, False])

    def test_place(self):
        a = self._a()
        np.place(a, self._d() >= 1, [7, 8])
        assert not np.asarray(a.mask).any()
        b = np.arange(5)
        vals = np.array([10, 20]); vals.mask = np.array([True, False])
        np.place(b, np.array([False, True, True, True, False]), vals)
        assert np.array_equal(b, [0, 10, 20, 10, 4])
        assert np.array_equal(b.mask, [False, True, False, True, False])

    def test_copyto(self):
        a = self._a()
        np.copyto(a, 9, where=self._d() < 2)
        assert np.array_equal(a.mask, [[False, False, False], [True, False, False]])
        b = np.arange(4)
        src = np.array([5, 6, 7, 8]); src.mask = np.array([False, True, True, False])
        np.copyto(b, src, where=np.array([True, True, False, True]))
        assert np.array_equal(b, [5, 6, 2, 8])
        assert np.array_equal(b.mask, [False, True, False, False])
        c = self._a()
        np.copyto(c, np.zeros((2, 3), int))
        assert not np.asarray(c.mask).any()
        d = np.zeros(3)
        np.copyto(d, np.array([1.0, 2.0, 3.0]))
        assert d.mask is None

    def test_copyto_broadcast_masked_src(self):
        d = np.zeros((2, 3), int)
        src = np.array([1, 2, 3]); src.mask = np.array([False, True, False])
        np.copyto(d, src)
        assert np.array_equal(d.mask, [[False, True, False]] * 2)

    def test_fill(self):
        a = self._a()
        a.fill(5)
        assert not np.asarray(a.mask).any()
        b = np.arange(4)
        b.fill(0)
        assert b.mask is None
        z = np.array(1.0); z.mask = np.array(True)
        c = np.arange(3)
        c.fill(z)
        assert np.array_equal(c.mask, [True, True, True])

    def test_flat_whole_assignment(self):
        a = self._a()
        a.flat = 7
        assert not np.asarray(a.mask).any()
        b = np.arange(6).reshape(2, 3)
        vals = np.array([1, 2]); vals.mask = np.array([True, False])
        b.flat = vals  # cycles: [1, 2, 1, 2, 1, 2]
        assert np.array_equal(b.mask, [[True, False, True], [False, True, False]])
        c = self._a()
        np.fill_diagonal(c, 0)
        assert not c.mask[0, 0] and c.mask[0, 1]

    def test_real_imag_assignment(self):
        a = self._a().astype(complex)
        a.real = 1
        assert not np.asarray(a.mask).any()
        b = self._a().astype(float)
        b.real = 3
        assert not np.asarray(b.mask).any()

    def test_read_only_mask_blocks_scatter_before_data(self):
        for op in (lambda a: a.put([0], [9]),
                   lambda a: np.putmask(a, np.ones(a.shape, bool), 9),
                   lambda a: np.copyto(a, 9),
                   lambda a: a.fill(9),
                   lambda a: setattr(a, "flat", 9)):
            a = self._a()
            a.mask.flags.writeable = False
            with pytest.raises(ValueError, match="read-only"):
                op(a)
            assert np.array_equal(a, np.arange(6).reshape(2, 3))

    def test_assigning_into_a_mask_is_plain(self):
        a = self._a()
        m = a.mask
        m.fill(True)
        assert m.all() and m.mask is None

    # ----- ufunc out= (phase 3 follow-up) ------------------------------
    def test_ufunc_out_clears_stale_mask_for_unmasked_inputs(self):
        out = np.zeros(4)
        out.mask = np.array([True, False, True, False])
        np.add(np.arange(4.0), 1.0, out=out)
        assert out.mask is None

    def test_ufunc_out_where_clears_only_written_positions(self):
        out = np.zeros(4)
        out.mask = np.array([True, True, True, True])
        np.add(np.arange(4.0), 1.0, out=out,
               where=np.array([True, False, True, False]))
        assert np.array_equal(out.mask, [False, True, False, True])

    # ----- in-place metadata mutators cannot keep a mask consistent -----
    def test_inplace_shape_strides_dtype_resize_rejected(self):
        a = self._a()
        with pytest.warns(DeprecationWarning), \
                pytest.raises(ValueError, match="in place"):
            a.shape = (3, 2)
        with pytest.warns(DeprecationWarning), \
                pytest.raises(ValueError, match="in place"):
            a.dtype = np.int32
        with pytest.warns(DeprecationWarning), \
                pytest.raises(ValueError, match="in place"):
            a.strides = (24, 8)
        assert a.shape == (2, 3) and a.mask.shape == (2, 3)
        b = np.arange(6)
        b.mask = np.array([False, True, False, False, False, False])
        with pytest.raises(ValueError, match="in place"):
            b.resize(8, refcheck=False)
        assert b.shape == (6,)

    def test_inplace_mutators_rejected_on_an_array_serving_as_mask(self):
        a = self._a()
        m = a.mask
        with pytest.warns(DeprecationWarning), \
                pytest.raises(ValueError, match="in place"):
            m.shape = (6,)
        with pytest.warns(DeprecationWarning), \
                pytest.raises(ValueError, match="in place"):
            m.dtype = np.uint8
        assert a.mask.shape == a.shape

    def test_inplace_mutators_unchanged_for_unmasked(self):
        b = np.arange(6)
        with pytest.warns(DeprecationWarning):
            b.shape = (2, 3)
        assert b.shape == (2, 3)
        b = np.arange(6)
        b.resize(8, refcheck=False)
        assert b.shape == (8,)


class TestMaskCasting:
    """Phase 8: casting entry points beyond `astype` (done in phase 5)."""

    @staticmethod
    def _masked(dtype=np.int32, shape=(2, 3)):
        a = np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)
        m = np.zeros(shape, dtype=bool)
        m.flat[[1 % m.size, 3 % m.size]] = True
        a.mask = m
        return a

    @pytest.mark.parametrize("dtype", [float, object, str, bool, complex,
                                       [("x", "i4")]])
    def test_astype_keeps_mask_for_every_dtype_kind(self, dtype):
        a = self._masked()
        r = a.astype(dtype)
        assert np.array_equal(r.mask, a.mask)
        assert not np.shares_memory(r.mask, a.mask)

    def test_astype_to_subarray_dtype_expands_the_mask(self):
        a = self._masked()
        r = a.astype((np.int32, (2,)))
        assert r.shape == a.shape + (2,)
        # every sub-element of a hidden element is hidden
        assert np.array_equal(r.mask, np.repeat(a.mask[..., None], 2, axis=-1))
        assert np.array_equal(r.mask.shape, r.shape)
        # data cast like an unmasked array
        assert np.array_equal(np.asarray(r.view(np.ndarray)),
                              np.arange(6).reshape(2, 3, 1).repeat(2, -1))

    def test_astype_subarray_2d_subshape(self):
        a = self._masked(shape=(4,))
        r = a.astype((np.int32, (2, 3)))
        assert r.mask.shape == (4, 2, 3)
        assert np.array_equal(r.mask[..., 0, 0], a.mask)
        assert np.array_equal(r.mask.any(axis=(1, 2)), a.mask)

    def test_astype_order_variants(self):
        a = self._masked()
        for order in "CFAK":
            r = a.astype(float, order=order)
            assert np.array_equal(r.mask, a.mask)

    def test_astype_failure_leaves_source_intact(self):
        a = self._masked(np.float64)
        with pytest.raises(TypeError):
            a.astype(np.int8, casting="safe")
        assert a.mask.sum() == 2

    def test_asarray_family_with_dtype(self):
        a = self._masked()
        for f in (np.asarray, np.asanyarray, np.array, np.ascontiguousarray,
                  np.asfortranarray):
            r = f(a, dtype=np.float64)
            assert np.array_equal(r.mask, a.mask), f
        r = np.require(a, dtype=np.float64, requirements=["C"])
        assert np.array_equal(r.mask, a.mask)
        assert np.array_equal(np.astype(a, np.float32).mask, a.mask)

    def test_asarray_same_dtype_is_identity(self):
        a = self._masked()
        assert np.asarray(a) is a

    def test_array_ndmin_keeps_mask(self):
        a = self._masked()
        r = np.array(a, ndmin=4)  # copies
        assert r.shape == (1, 1, 2, 3)
        assert np.array_equal(r.mask, a.mask[None, None])
        assert not np.shares_memory(r.mask, a.mask)
        # copy=None: no copy needed, so both data and mask are views
        r = np.array(a, ndmin=3, copy=None)
        assert r.shape == (1, 2, 3)
        assert np.shares_memory(r, a)
        assert np.array_equal(r.mask, a.mask[None])
        assert np.shares_memory(r.mask, a.mask)

    def test_array_ndmin_order_f_and_1d(self):
        a = self._masked(shape=(6,))
        r = np.array(a, ndmin=2, order="F")
        assert r.shape == (1, 6)
        assert np.array_equal(r.mask, a.mask[None])
        z = self._masked(shape=(2, 3))
        r = np.array(z.T, ndmin=3)
        assert np.array_equal(r.mask, z.mask.T[None])

    def test_array_from_list_of_masked_arrays(self):
        a = self._masked()
        b = np.arange(6).reshape(2, 3)  # unmasked
        r = np.array([a, b, a])
        assert r.shape == (3, 2, 3)
        expected = np.stack([a.mask, np.zeros((2, 3), bool), a.mask])
        assert np.array_equal(r.mask, expected)
        # data of hidden cells is still there
        assert np.array_equal(r.view(np.ndarray)[1], b)

    def test_array_from_nested_list_with_dtype(self):
        a = self._masked(shape=(3,))
        r = np.array([[a, a], [a, a]], dtype=np.float64)
        assert r.shape == (2, 2, 3)
        assert np.array_equal(r.mask, np.broadcast_to(a.mask, (2, 2, 3)))
        assert r.dtype == np.float64

    def test_array_from_list_only_unmasked_stays_unmasked(self):
        b = np.arange(3)
        assert np.array([b, b]).mask is None
        assert np.array([[1, 2], [3, 4]]).mask is None

    def test_array_from_list_of_all_false_masked_arrays(self):
        a = np.arange(3)
        a.mask = np.zeros(3, bool)  # canonicalised to no mask
        assert np.array([a, a]).mask is None

    def test_array_from_list_mixed_with_scalar_mask_free(self):
        a = self._masked(shape=(3,))
        r = np.array([a, a[::-1]])
        assert np.array_equal(r.mask[1], a.mask[::-1])

    def test_array_from_list_ragged_still_errors(self):
        a = self._masked(shape=(3,))
        with pytest.raises(ValueError):
            np.array([a, a[:2]])
        # and does not leak / poison a later call
        assert np.array([a, a]).mask.shape == (2, 3)

    def test_setitem_from_list_of_masked_arrays_is_a_known_limitation(self):
        # `t[...] = [a, a]` goes through PyArray_CopyObject, which has no mask
        # hook: the data is assigned, the masks inside the list are ignored.
        # (`t[...] = np.array([a, a])` carries them.)
        a = self._masked(shape=(3,))
        t = np.zeros((2, 3))
        t[...] = [a, a]
        assert np.array_equal(t.view(np.ndarray), [[0, 1, 2]] * 2)
        assert t.mask is None
        t[...] = np.array([a, a])
        assert np.array_equal(t.mask, np.broadcast_to(a.mask, (2, 3)))

    def test_result_type_and_can_cast_are_mask_blind(self):
        a = self._masked()
        assert np.result_type(a, np.float64) == np.float64
        assert np.can_cast(a, np.float64)
        assert not np.can_cast(a.astype(np.float64), np.int8)

    def test_copy_variants_keep_mask(self):
        import copy
        a = self._masked()
        for r in (a.copy(), a.copy("F"), np.copy(a), copy.copy(a),
                  copy.deepcopy(a), np.array(a, copy=True),
                  a.__array__(copy=True)):
            assert np.array_equal(r.mask, a.mask)

    def test_scalar_conversions_carry_no_mask(self):
        a = self._masked(shape=(1,))
        a.mask = np.array([True])
        assert not hasattr(float(a[0]), "mask")
        assert not hasattr(a.item(), "mask")

    def test_structured_astype_and_view_are_per_record(self):
        s = np.zeros(3, dtype=[("a", "i4"), ("b", "f8")])
        s.mask = np.array([True, False, False])
        r = s.astype([("a", "i8"), ("b", "f4")])
        assert np.array_equal(r.mask, s.mask)
        with pytest.raises(ValueError, match="keep the itemsize"):
            s.view(np.uint8)
        # documented limitation: field views do not carry the record mask
        assert s["a"].mask is None

    def test_zero_d_and_empty_astype(self):
        z = np.array(3)
        z.mask = np.array(True)
        assert z.astype(float).mask == True  # noqa: E712
        e = np.zeros((0, 3))
        e.mask = np.zeros((0, 3), bool)
        assert e.astype(int).shape == (0, 3)


class TestMaskSubclassNamespace:
    """A subclass may define its own, unrelated `mask` attribute (numpy.ma)."""

    def test_stride_tricks_ignore_a_subclass_mask_attribute(self):
        class Sub(np.ndarray):
            @property
            def mask(self):
                raise AssertionError("subclass mask must not be touched")

            @mask.setter
            def mask(self, value):
                raise AssertionError("subclass mask must not be touched")

        a = np.arange(6.0).reshape(2, 3).view(Sub)
        assert np.broadcast_to(a, (4, 2, 3), subok=True).shape == (4, 2, 3)
        assert np.lib.stride_tricks.as_strided(
            a, shape=(2, 3), strides=a.strides, subok=True).shape == (2, 3)

    def test_builtin_mask_still_carried_on_a_subclass_with_its_own_mask(self):
        class Sub(np.ndarray):
            mask = "unrelated"

        a = np.arange(6.0).reshape(2, 3).view(Sub)
        np.ndarray.mask.__set__(a, np.array([[True, False, False]] * 2))
        r = np.broadcast_to(a, (2, 2, 3), subok=True)
        assert np.array_equal(np.ndarray.mask.__get__(r),
                              np.broadcast_to(np.ndarray.mask.__get__(a),
                                              (2, 2, 3)))

    def test_masked_array_broadcast_unchanged(self):
        m = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0], hard_mask=True)
        r = np.broadcast_to(m, (2, 3), subok=True)
        # exactly upstream's behaviour (np.broadcast_to never knew ma's mask)
        assert r.mask is np.ma.nomask
        v = np.ma.mvoid((1, 2), mask=(0, 1), dtype="i4,i4")
        assert np.broadcast_to(v, (), subok=True) is not None


def _hidden(shape, seed, p=0.25, dtype=float):
    """A float array with a random mask (at least one hidden element)."""
    rng = np.random.default_rng(seed)
    a = rng.random(shape).astype(dtype) + 1
    m = rng.random(shape) < p
    m.flat[0] = True
    a.mask = m
    return a


def _plain(a):
    """An unmasked copy of `a` (np.asarray/view would keep the mask)."""
    r = np.array(a)
    r.mask = None
    return r


class TestMaskLinalg:
    """Phase 9: contracting ops. The data is computed as usual on the whole
    array; an output element is hidden iff a hidden input element contributes."""

    @staticmethod
    def _matmul_ref(ma, mb):
        m, k = ma.shape[-2:]
        n = mb.shape[-1]
        out = np.zeros(np.broadcast_shapes(ma.shape[:-2], mb.shape[:-2]) + (m, n),
                       bool)
        for idx in np.ndindex(*out.shape[:-2]):
            ia = tuple(i if s > 1 else 0 for i, s in
                       zip(idx[len(idx) - ma.ndim + 2:], ma.shape[:-2]))
            ib = tuple(i if s > 1 else 0 for i, s in
                       zip(idx[len(idx) - mb.ndim + 2:], mb.shape[:-2]))
            for i in range(m):
                for j in range(n):
                    out[idx + (i, j)] = any(
                        ma[ia + (i, kk)] or mb[ib + (kk, j)] for kk in range(k))
        return out

    def test_matmul_2d_matches_definition(self):
        a, b = _hidden((4, 3), 1), _hidden((3, 5), 2)
        r = a @ b
        assert np.array_equal(r.mask, self._matmul_ref(a.mask, b.mask))
        assert np.array_equal(_plain(r), _plain(a) @ _plain(b))  # data untouched

    def test_matmul_only_one_operand_masked(self):
        a = _hidden((4, 3), 3)
        b = np.arange(15.).reshape(3, 5)
        assert np.array_equal((a @ b).mask, self._matmul_ref(a.mask, np.zeros((3, 5), bool)))
        assert np.array_equal((b.T @ a.T).mask,
                              self._matmul_ref(np.zeros((5, 3), bool), a.mask.T))

    def test_matmul_batched_and_broadcast(self):
        a, b = _hidden((2, 1, 4, 3), 4, p=0.1), _hidden((5, 3, 2), 5, p=0.1)
        r = a @ b
        assert r.shape == (2, 5, 4, 2)
        assert np.array_equal(r.mask, self._matmul_ref(a.mask, b.mask))

    def test_matmul_vectors_change_dimensionality(self):
        A, v = _hidden((4, 3), 6), _hidden((3,), 7)
        r = A @ v                                   # (4,)
        assert r.shape == (4,)
        assert np.array_equal(r.mask, self._matmul_ref(A.mask, v.mask[:, None])[:, 0])
        r = v @ A.T                                 # (4,)
        assert np.array_equal(r.mask, self._matmul_ref(v.mask[None], A.mask.T)[0])
        # vector @ vector is a scalar: a scalar carries no mask (convention)
        assert not hasattr(v @ v, "mask")

    def test_matmul_out_replaces_mask_and_unmasked_inputs_clear_it(self):
        a, b = _hidden((3, 3), 8), _hidden((3, 3), 9)
        out = np.empty((3, 3))
        assert np.matmul(a, b, out=out) is out
        assert np.array_equal(out.mask, self._matmul_ref(a.mask, b.mask))
        np.matmul(_plain(a), _plain(b), out=out)      # unmasked inputs
        assert out.mask is None

    def test_unmasked_matmul_stays_unmasked(self):
        a = np.arange(6.).reshape(2, 3)
        assert (a @ a.T).mask is None

    def test_vecdot_matvec_vecmat(self):
        A, B = _hidden((4, 3), 10), _hidden((4, 3), 11)
        v = _hidden((3,), 12)
        assert np.array_equal(np.vecdot(A, B).mask, A.mask.any(-1) | B.mask.any(-1))
        assert np.array_equal(np.matvec(A, v).mask, A.mask.any(-1) | v.mask.any())
        assert np.array_equal(np.vecmat(v, A.T).mask, v.mask.any() | A.mask.T.any(-2))
        assert np.array_equal(np.matvec(_plain(A), v).mask, np.full(4, v.mask.any()))

    @pytest.mark.parametrize("shapes", [((4, 3), (3, 5)), ((4, 3), (3,)),
                                        ((2, 4, 3), (3, 5)), ((2, 4, 3), (5, 3, 6)),
                                        ((3,), (3, 5)), ((3,), (2, 3, 5))])
    def test_dot_matches_definition(self, shapes):
        a, b = _hidden(shapes[0], 20), _hidden(shapes[1], 21)
        r = np.dot(a, b)
        exp = np.zeros(r.shape, bool)
        for ia in np.ndindex(*a.shape[:-1]):
            for ib in np.ndindex(*((b.shape[:-2] + b.shape[-1:]) if b.ndim > 1 else ())):
                bidx = ib[:-1] + (slice(None),) + ib[-1:] if b.ndim > 1 else (slice(None),)
                exp[ia + ib] = (a.mask[ia].any() or b.mask[bidx].any())
        assert np.array_equal(r.mask, exp)
        assert np.array_equal(_plain(r), np.dot(_plain(a), _plain(b)))
        assert np.array_equal(a.dot(b).mask, exp)

    def test_dot_with_scalar_and_out(self):
        a = _hidden((3, 3), 22)
        assert np.array_equal(np.dot(a, 2.0).mask, a.mask)
        out = np.empty((3, 3))
        np.dot(a, np.eye(3), out=out)
        assert np.array_equal(out.mask, np.broadcast_to(a.mask.any(-1)[:, None], (3, 3)))

    def test_inner_and_tensordot(self):
        a, b = _hidden((4, 3), 23), _hidden((5, 3), 24)
        r = np.inner(a, b)
        assert np.array_equal(r.mask, a.mask.any(-1)[:, None] | b.mask.any(-1)[None])
        t = np.tensordot(a, b, axes=([1], [1]))
        assert np.array_equal(t.mask, r.mask)
        c, d = _hidden((2, 3, 4), 25), _hidden((4, 3, 5), 26)
        t = np.tensordot(c, d, axes=([1, 2], [1, 0]))
        exp = (c.mask.any((1, 2))[:, None] | d.mask.any((0, 1))[None])
        assert np.array_equal(t.mask, exp)

    def test_einsum(self):
        a, b = _hidden((4, 3), 30), _hidden((3, 5), 31)
        r = np.einsum("ij,jk->ik", a, b)
        assert np.array_equal(r.mask, self._matmul_ref(a.mask, b.mask))
        assert np.array_equal(np.einsum(a, [0, 1], b, [1, 2], [0, 2]).mask, r.mask)
        t = np.einsum("ij->ji", a)
        assert np.array_equal(t.mask, a.mask.T)
        s = _hidden((3, 3), 32)
        assert np.array_equal(np.einsum("ii->i", s).mask, np.diagonal(s.mask))
        assert np.array_equal(np.einsum("ij->i", a).mask, a.mask.any(-1))
        # batched with an ellipsis
        c, d = _hidden((2, 4, 3), 33, p=0.1), _hidden((2, 3, 5), 34, p=0.1)
        assert np.array_equal(np.einsum("...ij,...jk->...ik", c, d).mask,
                              self._matmul_ref(c.mask, d.mask))
        # only one operand masked / result is a scalar
        assert np.array_equal(np.einsum("ij,jk->ik", a, _plain(b)).mask,
                              self._matmul_ref(a.mask, np.zeros((3, 5), bool)))
        assert not hasattr(np.einsum("ij,ij->", a, a), "mask")
        assert np.einsum("ij,jk->ik", _plain(a), _plain(b)).mask is None

    def test_einsum_optimize_and_out(self):
        a, b, c = _hidden((3, 4), 35), _hidden((4, 5), 36), _hidden((5, 2), 37)
        r = np.einsum("ij,jk,kl->il", a, b, c, optimize=True)
        exp = self._matmul_ref(self._matmul_ref(a.mask, b.mask), c.mask)
        assert np.array_equal(r.mask, exp)
        out = np.empty((3, 5))
        np.einsum("ij,jk->ik", a, b, out=out)
        assert np.array_equal(out.mask, self._matmul_ref(a.mask, b.mask))

    @pytest.mark.parametrize("mode", ["valid", "same", "full"])
    def test_correlate_convolve(self, mode):
        a, v = _hidden((7,), 40, p=0.2), _hidden((3,), 41, p=0.2)
        for f in (np.correlate, np.convolve):
            r = f(a, v, mode)
            # brute force: unit-impulse probe of every input element
            exp = np.zeros(r.shape, bool)
            for i in np.flatnonzero(a.mask):
                e = np.zeros(7); e[i] = 1
                exp |= f(e, np.ones(3), mode) != 0
            for i in np.flatnonzero(v.mask):
                e = np.zeros(3); e[i] = 1
                exp |= f(np.ones(7), e, mode) != 0
            assert np.array_equal(r.mask, exp), (f.__name__, mode)
            assert np.array_equal(_plain(r), f(_plain(a), _plain(v), mode))

    def test_cross_with_masked_vectors_no_longer_raises(self):
        # `multiply(a1, b2, out=cp0)` with 0-d masked operands and a 0-d
        # view as `out` used to raise (mask must be an ndarray of dtype bool)
        a, b = _hidden((3,), 42), _hidden((3,), 43)
        assert np.array_equal(_plain(np.cross(a, b)), np.cross(_plain(a), _plain(b)))

    def test_zero_d_or_of_masks_is_an_array(self):
        a = np.array(1.0)
        a.mask = np.array(True)
        b = np.array(2.0)
        b.mask = np.array(True)
        out = np.empty(())
        np.add(a, b, out=out)
        assert isinstance(out.mask, np.ndarray) and out.mask.shape == () and out.mask
        # `where`/`choose` on 0-d masked operands: no longer raise (their
        # 0-d results decay to scalars, which carry no mask)
        assert np.where(np.array(True), a, b) == 1.0
        assert np.choose(np.array(0), [a, b]) == 1.0

    # ---- numpy.linalg: every output depends on the whole matrix

    @staticmethod
    def _masked_at(a, *idx):
        m = np.zeros(a.shape, bool)
        for i in idx:
            m[i] = True
        a.mask = m
        return a

    @staticmethod
    def _spd(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.random((n, n))
        return x @ x.T + n * np.eye(n)

    def test_inv_cholesky_pinv_hide_the_whole_matrix(self):
        a = self._masked_at(self._spd(3, 50), (0, 1))
        for f in (np.linalg.inv, np.linalg.cholesky, np.linalg.pinv):
            r = f(a)
            assert r.mask.shape == (3, 3) and r.mask.all(), f.__name__
            assert np.array_equal(_plain(r), f(_plain(a))), f.__name__

    def test_stacked_matrices_are_independent(self):
        s = self._masked_at(
            np.stack([self._spd(3, 51), self._spd(3, 52), self._spd(3, 53)]),
            (1, 2, 0))
        r = np.linalg.inv(s)
        assert r.mask[1].all() and not r.mask[0].any() and not r.mask[2].any()
        d = np.linalg.det(s)
        assert np.array_equal(d.mask, [False, True, False])
        assert np.array_equal(np.linalg.eigvalsh(s).mask,
                              np.array([[False] * 3, [True] * 3, [False] * 3]))

    def test_eigh_svd_qr(self):
        a = self._masked_at(self._spd(3, 54), (2, 1))
        w, v = np.linalg.eigh(a)
        assert w.mask.shape == (3,) and w.mask.all() and v.mask.all()
        u, s, vh = np.linalg.svd(a)
        assert u.mask.all() and s.mask.all() and vh.mask.all()
        q, r = np.linalg.qr(a)
        assert q.mask.all()
        # R's strictly lower triangle is structurally zero: only the upper
        # triangle depends on (and hides) the data
        assert np.array_equal(r.mask, np.triu(np.ones((3, 3), bool)))
        assert np.array_equal(np.linalg.qr(a, mode="r").mask, r.mask)

    def test_solve_hides_by_column_of_b(self):
        a = self._spd(3, 55)
        b = self._masked_at(np.arange(1., 7.).reshape(3, 2), (0, 1))
        x = np.linalg.solve(a, b)
        assert np.array_equal(x.mask, np.broadcast_to([False, True], (3, 2)))
        # a hidden element of `a` hides every column
        self._masked_at(a, (1, 1))
        assert np.linalg.solve(a, np.ones((3, 2))).mask.all()
        assert np.linalg.solve(a, np.ones(3)).mask.all()

    def test_lstsq_and_matrix_power_and_multi_dot(self):
        a, b = _hidden((4, 3), 56, p=0.1), np.arange(4.)
        x, res, rank, sv = np.linalg.lstsq(a, b, rcond=None)
        assert x.mask.all() and sv.mask.all()
        sq = _hidden((3, 3), 57, p=0.1)
        assert np.array_equal(np.linalg.matrix_power(sq, 2).mask,
                              self._matmul_ref(sq.mask, sq.mask))
        c = _hidden((3, 4), 58, p=0.1)
        exp = self._matmul_ref(self._matmul_ref(sq.mask, c.mask),
                               np.zeros((4, 2), bool))
        assert np.array_equal(np.linalg.multi_dot([sq, c, np.ones((4, 2))]).mask, exp)

    def test_scalar_results_carry_no_mask(self):
        a = self._masked_at(self._spd(3, 59), (0, 0))
        # documented convention: a 0-d/scalar result cannot carry a mask
        assert not hasattr(np.linalg.det(a), "mask")
        assert not hasattr(np.trace(a), "mask")
        assert not hasattr(np.linalg.norm(a), "mask")

    def test_unmasked_linalg_untouched(self):
        a = self._spd(3, 60)
        assert np.linalg.inv(a).mask is None
        assert all(x.mask is None for x in np.linalg.svd(a))

    def test_gufunc_out_mask_is_cleared_by_unmasked_inputs(self):
        a = self._spd(3, 61)
        out = np.empty((3, 3))
        out.mask = np.ones((3, 3), bool)
        np.matmul(a, a, out=out)
        assert out.mask is None

    def test_gufunc_with_subclass_mask_attribute_is_left_alone(self):
        class Sub(np.ndarray):
            mask = "unrelated"
        a = np.eye(3).view(Sub)
        assert (a @ a).mask == "unrelated"


def _mk(values, mask, dtype=None):
    a = np.array(values, dtype=dtype)
    a.mask = np.array(mask, dtype=bool)
    return a


class TestMaskPythonSurface:
    """Phase 10: repr/str, filled(), pickle, isin, buffer/IO (data only)."""

    # -- repr / str -------------------------------------------------------

    def test_repr_and_str_show_hidden_cells(self):
        a = _mk([[0, 1, 2], [3, 4, 5]], [[0, 1, 0], [1, 0, 0]], np.int32)
        assert repr(a) == ("array([[0, --, 2],\n"
                           "       [--, 4, 5]], dtype=int32)")
        assert str(a) == "[[0 -- 2]\n [-- 4 5]]"

    def test_hidden_values_never_printed(self):
        a = _mk([123456, 7, 8], [1, 0, 0])
        assert "123456" not in repr(a)
        assert "123456" not in str(a)
        assert "123456" not in np.array2string(a)

    def test_hidden_cells_do_not_affect_widths(self):
        a = _mk([1e9, 1.5, 2.25], [1, 0, 0])
        b = np.array([1.5, 2.25])
        # same text as the visible cells alone, plus the aligned placeholder
        assert str(a).replace(" ", "") == "[--1.52.25]"
        assert str(b).replace(" ", "") == "[1.52.25]"

    @pytest.mark.parametrize("values, dtype", [
        ([True, False, True], None),
        ([1 + 2j, 3 - 1j, 0j], None),
        (["ab", "c", "def"], None),
        ([b"ab", b"c", b"d"], None),
        (["2020-01-01", "NaT", "2021-05-06"], "M8[D]"),
        ([1, 2, 3], "m8[s]"),
        ([1.5, 2.5, 3.5], np.float32),
        ([1, "a", None], object),
    ])
    def test_repr_all_dtypes(self, values, dtype):
        a = _mk(values, [0, 1, 0], dtype)
        assert "--" in repr(a)
        assert str(a).count("--") == 1

    def test_repr_structured(self):
        s = _mk([(1, 2.5), (3, 4.5), (5, 6.5)], [0, 1, 0],
                [("a", "i4"), ("b", "f8")])
        assert str(s).count("--") == 1
        assert "(3, 4.5)" not in repr(s)

    def test_repr_all_hidden_empty_and_0d(self):
        assert str(_mk([1.0, 2.0], [1, 1])) == "[-- --]"
        z = np.array(3)
        z.mask = np.array(True)
        assert repr(z) == "array(--)"
        assert str(z) == "--"
        assert str(np.array(3)) == "3"
        e = np.zeros((0, 3))
        e.mask = np.zeros((0, 3), bool)
        assert repr(e) == repr(np.zeros((0, 3)))

    def test_repr_summarized(self):
        a = np.arange(2000.).reshape(2, 1000)
        m = np.zeros((2, 1000), bool)
        m[0, 0] = m[1, 999] = True
        m[0, 500] = True        # hidden but inside the elided part
        a.mask = m
        text = repr(a)
        assert "..." in text
        assert text.count("--") == 2

    def test_repr_options(self):
        a = _mk([1.0, 2.0, 3.0], [0, 1, 0])
        assert np.array2string(a, separator=", ") == "[1., --, 3.]"
        assert (np.array2string(a, formatter={"float_kind": lambda x: f"<{x}>"})
                == "[<1.0>    -- <3.0>]")
        with np.printoptions(threshold=2, edgeitems=1):
            assert repr(a) == "array([1., ..., 3.], shape=(3,))"

    def test_repr_subclass_and_unmasked_unchanged(self):
        m = np.matrix([[1, 2], [3, 4]])
        m.mask = np.array([[False, True], [False, False]])
        assert repr(m) == "matrix([[1, --],\n        [3, 4]])"
        assert repr(np.arange(3)) == "array([0, 1, 2])"

    def test_repr_leaves_array_alone(self):
        a = _mk([1, 2, 3], [0, 1, 0])
        before = a.mask.copy()
        repr(a)
        str(a)
        assert np.array_equal(a.mask, before)
        assert np.array_equal(a.filled(0), [1, 0, 3])

    def test_subclass_with_own_mask_attribute(self):
        class Sub(np.ndarray):
            mask = "unrelated"
        a = np.arange(3).view(Sub)
        assert str(a) == "[0 1 2]"

    def test_repr_no_refcount_leak(self):
        import sys
        a = _mk([1, 2, 3], [0, 1, 0])
        m = np.ndarray.mask.__get__(a)
        base = sys.getrefcount(m)
        for _ in range(200):
            repr(a)
            str(a)
        assert sys.getrefcount(m) == base

    # -- filled() ---------------------------------------------------------

    def test_filled_basic(self):
        a = _mk([1.0, 2.0, 3.0], [0, 1, 0])
        f = a.filled(np.nan)
        assert np.array_equal(f, [1.0, np.nan, 3.0], equal_nan=True)
        assert f.mask is None
        assert not np.shares_memory(f, a)
        # non-destructive: the array keeps its data and its mask
        assert np.array_equal(np.asarray(a).view(np.ndarray)[1], 2.0)
        assert np.array_equal(a.mask, [False, True, False])

    def test_filled_result_is_writable_and_independent(self):
        a = _mk([1, 2, 3], [0, 1, 0])
        f = a.filled(0)
        f[0] = 99
        assert a[0] == 1

    def test_filled_unmasked_is_a_copy(self):
        a = np.arange(3)
        f = a.filled(-1)
        assert np.array_equal(f, a) and not np.shares_memory(f, a)

    def test_filled_keyword_and_errors(self):
        a = _mk([1, 2, 3], [0, 1, 0])
        assert np.array_equal(a.filled(fill_value=7), [1, 7, 3])
        with pytest.raises(TypeError):
            a.filled()
        with pytest.raises(ValueError):
            a.filled(np.nan)                   # not representable as int
        with pytest.raises(ValueError):
            np.arange(3).filled(np.nan)        # validated even without a mask
        with pytest.raises(OverflowError):
            _mk(np.arange(3, dtype=np.int8), [1, 0, 0]).filled(1000)

    def test_filled_dtypes(self):
        assert list(_mk([True, False], [1, 0]).filled(True)) == [True, False]
        assert list(_mk(["a", "bb"], [1, 0]).filled("zz")) == ["zz", "bb"]
        assert list(_mk([1, 2], [1, 0], object).filled(None)) == [None, 2]
        s = _mk([(1, 2.5), (3, 4.5)], [0, 1], [("a", "i4"), ("b", "f8")])
        assert s.filled((0, 0.0)).tolist() == [(1, 2.5), (0, 0.0)]
        d = _mk(["2020-01-01", "2021-01-01"], [1, 0], "M8[D]")
        assert str(d.filled(np.datetime64("NaT", "D"))[0]) == "NaT"

    def test_filled_layouts(self):
        a = _mk(np.arange(12.).reshape(3, 4), np.arange(12).reshape(3, 4) % 3 == 0)
        for view in (a.T, a[::2], a[:, ::-1], np.asfortranarray(a)):
            f = view.filled(-1.0)
            assert f.shape == view.shape and f.mask is None
            assert np.array_equal(f == -1.0, view.mask)
        z = np.array(1.0)
        z.mask = np.array(True)
        r = z.filled(5.0)
        assert r.shape == () and r == 5.0 and r.mask is None

    def test_filled_keeps_subclass(self):
        m = np.matrix([[1, 2], [3, 4]])
        m.mask = np.array([[False, True], [False, False]])
        f = m.filled(0)
        assert type(f) is np.matrix and f.tolist() == [[1, 0], [3, 4]]

    def test_ma_filled_ignores_the_ndarray_method(self):
        # `np.ma.filled` duck-types on a `filled` attribute; plain arrays now
        # have one with a different meaning and must still pass through
        a = np.arange(3)
        assert np.ma.filled(a, 99) is a
        m = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
        assert np.ma.filled(m, 99).tolist() == [1, 99, 3]

    def test_filled_no_refcount_leak(self):
        import sys
        a = _mk([1, 2, 3], [0, 1, 0], object)
        marker = object()
        base = sys.getrefcount(marker)
        for _ in range(200):
            a.filled(marker)
        assert sys.getrefcount(marker) == base

    # -- pickle / copy ----------------------------------------------------

    @pytest.mark.parametrize("protocol", range(6))
    def test_pickle_roundtrip_keeps_mask(self, protocol):
        import pickle
        a = _mk(np.arange(12.).reshape(3, 4),
                np.arange(12).reshape(3, 4) % 3 == 0)
        r = pickle.loads(pickle.dumps(a, protocol))
        assert np.array_equal(r, a)
        assert np.array_equal(r.mask, a.mask)
        assert r.mask is not a.mask
        r.mask[1, 1] = True                     # the copy is independent
        assert not a.mask[1, 1]

    @pytest.mark.parametrize("make", [
        lambda a: np.asfortranarray(a),
        lambda a: a[:, ::2],
        lambda a: a.T,
        lambda a: a[::-1],
        lambda a: a[1],
    ])
    def test_pickle_views_and_layouts(self, make):
        import pickle
        a = _mk(np.arange(12.).reshape(3, 4),
                np.arange(12).reshape(3, 4) % 3 == 0)
        v = make(a)
        for protocol in (2, 5):
            r = pickle.loads(pickle.dumps(v, protocol))
            assert np.array_equal(r, v)
            assert np.array_equal(r.mask, v.mask)

    def test_pickle_special_dtypes(self):
        import pickle
        for a in (_mk([1, "a", None], [0, 1, 0], object),
                  _mk([(1, 2.5), (3, 4.5)], [0, 1], [("a", "i4"), ("b", "f8")]),
                  _mk(["x", "yy"], [1, 0]),
                  _mk(np.array([1, 2], ">i4"), [0, 1])):
            for protocol in (2, 5):
                r = pickle.loads(pickle.dumps(a, protocol))
                # (a big-endian array comes back native, as without a mask)
                assert r.dtype == a.dtype.newbyteorder("=")
                assert r.tolist() == a.tolist()
                assert np.array_equal(r.mask, a.mask)
        z = np.array(3)
        z.mask = np.array(True)
        assert pickle.loads(pickle.dumps(z)).mask == True  # noqa: E712
        e = np.zeros((0, 3))
        assert pickle.loads(pickle.dumps(e)).mask is None

    def test_pickle_subclass(self):
        import pickle
        m = np.matrix([[1, 2], [3, 4]])
        m.mask = np.array([[False, True], [False, False]])
        r = pickle.loads(pickle.dumps(m, 5))
        assert type(r) is np.matrix and np.array_equal(r.mask, m.mask)

    def test_pickle_unmasked_format_unchanged(self):
        a = np.arange(4)
        state = a.__reduce__()[2]
        assert len(state) == 5
        assert len(a.filled(0).__reduce__()[2]) == 5
        m = _mk([1, 2], [0, 1])
        assert len(m.__reduce__()[2]) == 6
        # protocol 5 out-of-band buffers are only used for unmasked arrays
        assert m.__reduce_ex__(5)[0] is m.__reduce__()[0]
        assert a.__reduce_ex__(5)[0] is not a.__reduce__()[0]

    def test_setstate_validates_the_mask(self):
        a = _mk([1, 2, 3], [0, 1, 0])
        good = a.__reduce__()[2]
        b = np.empty(0)
        b.__setstate__(good)
        assert np.array_equal(b.mask, a.mask)
        for bad in (np.zeros(2, bool), np.zeros(3, int), [False, True, False]):
            with pytest.raises((TypeError, ValueError)):
                np.empty(0).__setstate__(good[:5] + (bad,))
        # an old 5-item state still loads and clears any mask already there
        c = _mk([9, 9], [1, 0])
        c.__setstate__(np.arange(3).__reduce__()[2])
        assert c.mask is None and c.shape == (3,)

    def test_copy_and_deepcopy_keep_mask(self):
        import copy
        a = _mk([1, 2, 3], [0, 1, 0])
        for c in (copy.copy(a), copy.deepcopy(a), a.copy()):
            assert np.array_equal(c.mask, a.mask) and c.mask is not a.mask

    # -- isin -------------------------------------------------------------

    def test_isin_element_mask_follows(self):
        e = _mk([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 0]])
        r = np.isin(e, [1, 4])
        assert r.tolist() == [[True, False, False], [True, False, False]]
        assert np.array_equal(r.mask, e.mask)
        r = np.isin(e, [1, 4], invert=True)
        assert np.array_equal(r.mask, e.mask)
        assert np.isin(np.array(3), [3]).mask is None

    def test_isin_hidden_test_values(self):
        e = np.array([1, 2, 3, 4])
        t = _mk([1, 4, 9], [0, 0, 1])
        r = np.isin(e, t)
        # 1 and 4 match a visible value (certain); 2 and 3 might match the hidden one
        assert np.array_equal(r.mask, [False, True, True, False])
        assert r.tolist() == [True, False, False, True]
        r2 = np.isin(e, t, invert=True)
        assert np.array_equal(r2.mask, r.mask)
        # a test value hidden nowhere: no mask at all
        assert np.isin(e, _mk([1, 4, 9], [0, 0, 0])).mask is None

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_matches_bruteforce(self, kind):
        rng = np.random.default_rng(5)
        for _ in range(100):
            el = rng.integers(0, 6, (3, 4))
            tt = rng.integers(0, 6, 5)
            em = rng.random((3, 4)) < 0.3
            tm = rng.random(5) < 0.3
            r = np.isin(_mk(el, em), _mk(tt, tm), kind=kind)
            expect = em | (~np.isin(el, tt[~tm]) & tm.any())
            got = np.zeros(el.shape, bool) if r.mask is None else r.mask
            assert np.array_equal(got, expect)
            assert np.array_equal(np.asarray(r).view(np.ndarray), np.isin(el, tt))

    def test_isin_plain_unchanged(self):
        r = np.isin(np.arange(5), [1, 2])
        assert r.mask is None
        assert np.isin([1, 2], {1}).tolist() == [False, False]

    # -- unique / set operations: mask-blind (documented) -----------------

    def test_unique_is_mask_blind(self):
        a = _mk([3, 1, 3, 2, 1], [1, 0, 0, 1, 0])
        u = np.unique(a)
        assert u.tolist() == [1, 2, 3] and u.mask is None
        _, idx, inv, cnt = np.unique(a, return_index=True, return_inverse=True,
                                     return_counts=True)
        assert idx.tolist() == [1, 3, 0] and cnt.tolist() == [2, 1, 2]

    # -- buffer protocol / IO: data only (documented) ---------------------

    def test_buffer_and_io_export_data_only(self):
        import io
        a = _mk(np.arange(4, dtype=np.int32), [0, 1, 0, 0])
        plain = np.arange(4, dtype=np.int32)
        assert a.tobytes() == plain.tobytes()
        assert bytes(memoryview(a)) == plain.tobytes()
        assert np.frombuffer(a, dtype=np.int32).mask is None
        buf = io.BytesIO()
        np.save(buf, a)
        buf.seek(0)
        loaded = np.load(buf)
        assert loaded.mask is None and np.array_equal(loaded, plain)
        assert a.tolist() == plain.tolist()

    def test_tofile_exports_data_only(self, tmp_path):
        a = _mk(np.arange(4, dtype=np.int32), [0, 1, 0, 0])
        path = tmp_path / "a.bin"
        a.tofile(path)
        assert path.read_bytes() == np.arange(4, dtype=np.int32).tobytes()
