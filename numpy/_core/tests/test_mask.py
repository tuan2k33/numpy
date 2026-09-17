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
