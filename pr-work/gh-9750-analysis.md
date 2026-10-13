# gh-9750 — Gán `np.ma.masked` vào mảng chuỗi cho ra `'0.0'`

Ghi chép phân tích. Mọi thí nghiệm chạy trong worktree riêng
`/mnt/c/Dev/projects/numpy-mutation` ở `origin/main` (`3deeb20adb`), **không đụng
gì tới nhánh PR** `perf/broadcast-buffer-fill`.

- Issue: https://github.com/numpy/numpy/issues/9750 — OPEN từ 2017-09-23, 0 comment
- Regression đến từ PR #8903, commit `c5f62da627` (2017-04-11)
- Test đánh dấu xfail: `numpy/ma/tests/test_core.py:5791` và `:5797`

---

## 1. `MaskedConstant` là cái gì

`np.ma.masked` là **singleton** của lớp `MaskedConstant`, định nghĩa trong
`numpy/ma/core.py`. Nó là một `MaskedArray` 0 chiều:

| Thành phần | Giá trị |
|---|---|
| `type` | `MaskedConstant` (kế thừa `MaskedArray` → `ndarray`) |
| `shape` | `()` — mảng 0 chiều |
| `dtype` | `float64` |
| `data` | `0.0` — **chỉ là chỗ độn, vô nghĩa** |
| `mask` | `True` |

Các dunder quan trọng:

```python
def __str__(self):   return str(masked_print_option._display)   # '--'
def __repr__(self):  return 'masked'
def __format__(self, spec): return object.__format__(self, spec)  # khong dinh dang
```

`MaskedConstant` không định nghĩa `__float__`/`__int__` riêng; chúng thừa kế từ
`MaskedArray` (`core.py:4545` và `:4558`), nơi có xử lý riêng cho phần tử bị mask:

```python
def __float__(self):
    ...
    elif self._mask:
        warnings.warn("Warning: converting a masked element to nan.", stacklevel=2)
        return np.nan

def __int__(self):
    ...
    elif self._mask:
        raise MaskError('Cannot convert masked element to a Python int.')
```

**Nhưng `MaskedArray` KHÔNG có `__bool__`, `__complex__`, `__index__`** —
`vars(np.ma.MaskedArray)` rỗng cho cả ba. Những phép ép kiểu đó rơi xuống dunder
của `ndarray`, vốn đọc thẳng buffer dữ liệu.

**Điểm mấu chốt:** ý nghĩa của object này nằm ở **mask**, không nằm ở `data`.
Bất kỳ đường chuyển đổi nào đọc thẳng buffer dữ liệu đều lấy nhầm `0.0`.

---

## 2. Đường đi của phép gán

```
a[()] = np.ma.masked
   -> PyArray_SETITEM
   -> STRING_setitem / UNICODE_setitem   (arraytypes.c.src:776 / :679)
```

Trong hai hàm đó, dòng đầu tiên là:

```c
if (PyArray_IsZeroDim(op)) {
    return convert_to_scalar_and_retry(descr, op, ov, STRING_setitem);
}
```

`convert_to_scalar_and_retry` (`arraytypes.c.src:172`):

```c
temp = PyArray_ToScalar(PyArray_BYTES((PyArrayObject *)op), (PyArrayObject *)op);
int res = setitem(descr, temp, ov);
```

`PyArray_BYTES` lấy **con trỏ thô tới buffer**, `PyArray_ToScalar` dựng scalar
numpy thuần từ đó. Lớp con bị lột, mask không nằm trong buffer nên biến mất.
`np.ma.masked` → `np.float64(0.0)` → gọi lại setitem → `PyObject_Str` → `'0.0'`.

Nếu không có nhánh đó, `PyObject_Str(np.ma.masked)` sẽ cho `'--'`.

**Nhưng nhánh đó không thừa.** Nó tồn tại để nội dung của mảng 0-d **dtype chuỗi**
được định tuyến xuống nhánh `PyBytes_Check` / `PyUnicode_Check` phía dưới:

```
np.array(b'N/A') --ToScalar--> np.bytes_(b'N/A') --PyBytes_Check--> b'N/A'   dung
np.array(b'N/A') --khong rut--> PyObject_Str(mang 0-d) --> "np.bytes_(b'N/A')" --> b'np.'   sai
```

`numpy.ma` dùng đúng đường này khi xử lý `fill_value`, nên bỏ nhánh đi là gãy.

Ghi chú sẵn có của maintainer ngay trên hàm (`arraytypes.c.src:166`):

> NOTE/TODO(seberg): This was important, but is now only used for *nested* 0-D
> arrays which makes it dubious whether it should remain used.

### Mỗi dtype đi một hàm setitem khác nhau

Không có một đường chung. `arraytypes.c.src` có nhiều hàm setitem, và **chỉ một
số hàm mới có nhánh `IsZeroDim`**:

| dtype đích | Hàm C | Có `IsZeroDim`? | Đường thực tế |
|---|---|---|---|
| `bool` | `@TYPE@_setitem:357` | không | `PyObject_IsTrue` → `ndarray.__bool__` |
| `int*` | `@TYPE@_setitem:357` | không | `PyNumber_Long` → `MaskedArray.__int__` |
| `float*` | `@TYPE@_setitem:357` | không | `PyFloat_AsDouble` → `MaskedArray.__float__` |
| `complex*` | `@NAME@_setitem:458` | **có** | rút scalar, không gọi giao thức |
| `S` / `U` | `STRING/UNICODE_setitem:776/:679` | **có** | rút scalar, không gọi giao thức |

### Muốn đúng thì cần đủ hai điều kiện

| | ① C không rút scalar | ② `numpy.ma` override dunder | Kết quả |
|---|---|---|---|
| `int` | ✅ | ✅ `__int__` | `MaskError` ✅ |
| `float` | ✅ | ✅ `__float__` | `nan` + cảnh báo ✅ |
| `bool` | ✅ | ❌ không có `__bool__` | rơi xuống `ndarray.__bool__` → `False` ❌ |
| `complex` | ❌ | không bao giờ tới | `0j` ❌ |
| `S` / `U` | ❌ | không bao giờ tới | `'0.0'` ❌ |

**Có `IsZeroDim` nghĩa là hỏng**, vì nó cắt đứt giao thức Python. Không có nó chỉ
là *điều kiện cần*: `bool` chứng minh điều đó — C đã gọi đúng `PyObject_IsTrue`,
nhưng `numpy.ma` không viết `__bool__` nên rơi tiếp xuống `ndarray` và đọc `data`.

Hệ quả: **hai nhóm lỗi độc lập, sửa ở hai nơi khác nhau.**

- Sửa bên C (bỏ hoặc thu hẹp `IsZeroDim`) → chữa `complex` và chuỗi.
- Sửa bên Python (thêm `__bool__` cho `MaskedArray`) → chữa `bool`.

Cũng đáng ghi nhận: `int` và `float` đúng **không nhờ C**, mà nhờ `numpy.ma` có
viết dunder và C chịu gọi giao thức.

---

## 3. Hành vi theo dtype đích (đo trên main)

```python
a = np.zeros((), dtype); a[()] = np.ma.masked
```

| dtype | Kết quả trên main | Cảnh báo | Kỳ vọng lý thuyết | Đánh giá | Nguyên nhân |
|---|---|---|---|---|---|
| `bool` | `False` | không | raise, hoặc ít nhất cảnh báo | **sai, im lặng** | thiếu `__bool__` |
| `int64` / `uint64` | raise `MaskError` | — | raise | đúng | — |
| `float16` / `float64` | `nan` | `UserWarning` | `nan` (nan là "không có giá trị") | đúng | — |
| `complex128` | `0j` | **không** | `nan+nanj` | **sai, im lặng** | `IsZeroDim` |
| `S10` / `U10` | `'0.0'` | không | `'--'` hoặc raise | **sai — gh-9750** | `IsZeroDim` |
| `object` | `masked` | — | giữ nguyên object | đúng | — |
| `datetime64` / `timedelta64` | raise `ValueError` | — | raise | đúng | — |

Nhìn bảng này thì gh-9750 chỉ là **một mặt** của vấn đề rộng hơn: numpy không có
quy ước thống nhất cho việc ép `masked` sang dtype khác. Ba nhóm hành vi cùng tồn
tại — raise, trả `nan` kèm cảnh báo, và âm thầm trả giá trị nền.

Nguy hiểm nhất là nhóm cuối (`bool`, `complex`, chuỗi): **giá trị đang bị che bị
phơi ra mà không một lời cảnh báo**. Với mảng nhiều phần tử thì lộ nguyên dữ liệu
thật chứ không phải `0.0`:

```python
d = np.zeros(3, 'U10')
d[:] = np.ma.masked_array([1.5, 2.5, 3.5], mask=[True, False, True])
# ['1.5' '2.5' '3.5']  <- 1.5 va 3.5 dang bi mask, van lo ra
```

---

## 4. Các test của `TestMaskedConstant` (test_core.py:5705)

| Test | Kiểm tra điều gì |
|---|---|
| `test_ufunc`, `test_operator` | Phép toán với `masked` trả lại `masked`, giữ singleton |
| `test_ctor` | `MaskedConstant()` mới không phải là singleton |
| `test_repr` | `repr(np.ma.masked) == 'masked'`; bản sao thì khác |
| `test_pickle` | Pickle/unpickle vẫn ra đúng singleton |
| `test_copy`, `test__copy`, `test_deepcopy` | Mọi kiểu copy đều trả về chính nó |
| `test_immutable` | Không cho ghi đè `data`, `mask`, kể cả qua view |
| `test_coercion_int` | Gán vào `int` → raise `MaskError`; `int(masked)` → raise |
| `test_coercion_float` | Gán vào `float` → `UserWarning` + `nan` |
| `test_coercion_unicode` | **xfail** — mong `'--'`, thực tế `'0.0'` |
| `test_coercion_bytes` | **xfail** — mong `b'--'`, thực tế `b'0.0'` |
| `test_subclass` | Lớp con của `MaskedConstant` không bị nhầm thành singleton |
| `test_attributes_readonly` | Không cho set thuộc tính |

Nhóm `test_coercion_*` là nhóm duy nhất nói về ép kiểu, và nó **thiếu hai ca**:
`bool` và `complex` — đúng hai ca đang sai im lặng mà không ai biết.

---

## 5. Ba thí nghiệm đã chạy

### Thí nghiệm 1 — xoá hẳn nhánh `IsZeroDim`

```c
/* xoa han khoi ca UNICODE_setitem va STRING_setitem */
```

| | Kết quả |
|---|---|
| 2 test xfail | → xpass |
| Khác | **5 failed** |

Gãy toàn bộ đường `fill_value` của `numpy.ma`
(`test_fillvalue_conversion`, `test_check_on_fields`, ...): `b'N/A'` thành `b'np.'`.
→ **Loại.**

### Thí nghiệm 2 — chỉ rút scalar khi là `ndarray` thuần

```c
if (PyArray_IsZeroDim(op) && Py_TYPE(op) == &PyArray_Type)
```

| | Kết quả |
|---|---|
| 2 test xfail | → xpass |
| Test suite | 0 failed, 21285 passed |
| Kiểm tra tay | **gãy**: `masked_array` 0-d dtype bytes, không mask → `b'np.bytes_('` |

Test suite xanh nhưng **không phủ ca đó**. Luật sai về bản chất: `MaskedArray`
cũng là lớp con, mà mảng masked 0-d mang giá trị thật thì vẫn cần đường scalar.
→ **Loại**, dù test xanh.

### Thí nghiệm 3 — chỉ rút scalar khi dtype là chuỗi

```c
if (PyArray_IsZeroDim(op) && PyArray_ISFLEXIBLE((PyArrayObject *)op))
```

Lập luận: nhánh scalar chỉ thực sự cần cho dtype chuỗi (để vào đúng
`PyBytes_Check`/`PyUnicode_Check`). Với dtype số thì `str(mảng 0-d)` và
`str(scalar)` cho ra chuỗi y hệt, nên rút hay không không quan trọng — trừ khi
lớp con có `__str__` riêng, lúc đó `str()` mới là đường đúng.

| Ca kiểm tra | Kết quả |
|---|---|
| `S10 <- np.ma.masked` | `b'--'` ✅ |
| `U10 <- np.ma.masked` | `'--'` ✅ |
| `S10 <- np.array(b'N/A')` (0-d thuần) | `b'N/A'` ✅ giữ nguyên |
| `U10 <- np.array(1.5)` (0-d thuần) | `'1.5'` ✅ giữ nguyên |
| `S10 <- ma 0-d bytes, không mask` | `b'xy'` ✅ |
| `S10 <- ma 0-d float, không mask` | `b'1.5'` ✅ |
| `S10 <- ma 0-d float, có mask` | `b'--'` ✅ |

Test: `numpy/ma` + `test_multiarray` + `test_strings` + `test_scalarmath` +
`test_datetime` → **0 failed, 23374 passed, đúng 2 xpassed**.

→ **Ứng viên khả thi.** Chưa chạy full suite.

---

## 6. Việc còn phải làm nếu muốn theo tới cùng

1. Chạy **full suite** trên thí nghiệm 3.
2. Bổ sung test cho hai ca chưa phủ: `masked_array` 0-d có/không mask với dtype
   chuỗi và dtype số — chính là ca đã lật đổ thí nghiệm 2.
3. Quyết định về mặt thiết kế: `'--'` có thật sự là đáp án đúng không, hay nên
   raise `MaskError` cho nhất quán với nhánh `int`? `'--'` là một chuỗi trông
   như dữ liệu, dễ bị hiểu nhầm là giá trị thật. Đây là câu hỏi cho maintainer,
   không nên tự quyết.
4. Cân nhắc gộp luôn `bool` và `complex` — cùng một lớp lỗi "im lặng trả giá trị
   nền", và đang không có test nào phủ. Lưu ý chúng sửa ở hai nơi khác nhau:
   `complex` cùng gốc `IsZeroDim` với chuỗi (sửa trong C), còn `bool` là do
   `MaskedArray` thiếu `__bool__` (sửa trong `numpy/ma/core.py`).
5. Gỡ nhãn `xfail` ở hai test, thêm release note.

**Lưu ý:** `numpy.ma` gần như không còn ai bảo trì, nên PR kiểu này dễ nằm lâu.
Nên mở issue/thảo luận chốt hướng ở bước 3 trước khi viết code hoàn chỉnh.
