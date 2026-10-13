# Mảnh code từ các thí nghiệm gh-9750 (worktree đã xoá)

## 1. `numpy/ma/core.py` — gom chính sách vào một chỗ

Thay thế `__float__`/`__int__` cũ trong `MaskedArray`. Đo được: 0 test đỏ.

```python
    def _coerce_scalar(self, kind):
        """
        Convert a single element to a Python scalar of type `kind`.

        This is the one place that decides what converting a masked element
        means: it is an error where no value can stand in for "missing"
        (`int`, `bool`), and NaN where one can (`float`, `complex`).
        """
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "
                            "to Python scalars")
        if self._mask:
            if kind is float:
                warnings.warn("Warning: converting a masked element to nan.",
                              stacklevel=3)
                return np.nan
            if kind is complex:
                warnings.warn("Warning: converting a masked element to nan.",
                              stacklevel=3)
                return complex(np.nan, np.nan)
            raise MaskError(
                f'Cannot convert masked element to a Python {kind.__name__}.')
        return kind(self.item())

    def __float__(self):
        """
        Convert to float.

        """
        return self._coerce_scalar(float)

    def __int__(self):
        """
        Convert to int.

        """
        return self._coerce_scalar(int)

    def __complex__(self):
        """
        Convert to complex.

        """
        return self._coerce_scalar(complex)

    def __index__(self):
        """
        Convert to an index.

        """
        return self._coerce_scalar(operator.index)
```

## 2. `arraytypes.c.src` — thu hẹp `IsZeroDim`

Trong `UNICODE_setitem` (~:679) và `STRING_setitem` (~:776):

```c
-    if (PyArray_IsZeroDim(op)) {
+    if (PyArray_IsZeroDim(op) && PyArray_ISFLEXIBLE((PyArrayObject *)op)) {
         return convert_to_scalar_and_retry(descr, op, ov, STRING_setitem);
     }
```

Và trong `@NAME@_setitem` của complex (~:458), sửa `else if` tương tự.

## Kết quả đo

- `numpy/ma` + `test_multiarray` + `test_strings` + `test_scalarmath` + `test_indexing`: **0 failed, 22972 passed, 2 xpassed**
- Còn lỗ: `masked_array` 0-d **dtype chuỗi** có mask vẫn rò (`b\"xy\"` thay vì `--`)
- Chưa chữa: gh-9760 / gh-9761 (đường constructor của scalar, khác đường setitem)
