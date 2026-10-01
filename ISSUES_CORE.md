# Open issues: `component: numpy._core` (numpy/numpy) — kiểm tra & reproduce

Snapshot 2026-10-01: 183 issue đang mở (77 Bug, 57 Enhancement).

**Phương pháp:** đọc nội dung + comment cuối từng issue, viết repro và chạy trên **numpy 2.5.3 stock** (Python 3.14, 1 luồng BLAS); với #7179 và #3634 có đối chiếu thêm source `main`. `main` hiện là 2.6.0.dev nên **mọi ứng viên 'Đã giải quyết' cần chạy lại trên bản build từ `main` trước khi comment đề xuất đóng**. Issue loại enh/thảo luận/docs/platform không có repro nên xếp 'Không repro được' (vẫn mở, không phải đã sửa). Nhóm gom theo tiêu đề nên gần đúng.

| Kết luận | Số lượng |
|---|---|
| ✅ Đã giải quyết | 9 |
| ◐ Một phần | 17 |
| ❌ Còn | 116 |
| 📝 Không repro được | 41 |

> Label `sustain-2026` (NumFOCUS Sustaining Open Source Series 2026) gắn trên #10332, #13547, #17118, #17175, #21676, #25910: nên tránh tự nhận/đóng khi chưa rõ sự kiện.

## arange / linspace (8)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #630 | Add step parameter to linspace (or endpoint parameter to arange) (Trac… | ❌ Còn | linspace vẫn không có `step` |
| #2457 | arange using float for step and integer dtype results in an array with… | ❌ Còn | `arange(0.,5.,.5,dtype=int)` vẫn ra toàn 0 |
| #8909 | behaviour of linspace with dtype Decimal | ◐ Một phần | Decimal input chạy được; input float vẫn tính bằng float |
| #9059 | BUG: np.arange treats start as stop, even when step is specified | ❌ Còn | `arange(10,None,2)` vẫn ra [0,2,4,6,8] |
| #10332 | BUG: arange behaves poorly on complex numbers | ❌ Còn | `arange(0j,10+0j,2+0j)` vẫn ra mảng rỗng (issue dành cho sustain-2026) |
| #16159 | Unexpected output from arange with dtype=int | ❌ Còn | `arange(-3,0,.5,dtype=int)` vẫn ra [-3..2] |
| #17155 | BUG: linspace on int64 overflows for large stop values | ❌ Còn | linspace int64 vẫn tràn thành -2^63 (+RuntimeWarning cast) |
| #18881 | linspace with int dtype sometimes doesn't include endpoints | ❌ Còn | endpoint vẫn mất chữ số cuối (…992) |

## Casting, overflow, số học (21)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #1991 | assignment of float to int array causes loss of decimal part (Trac #13… | ◐ Một phần | 300.4→uint8 nay OverflowError, nhưng `a[0]=5.4` (int) vẫn im lặng |
| #4126 | Integer overflow using numpy.dot | ❌ Còn | `dot(int16)` vẫn tràn (-32768); gần như by-design |
| #4260 | improve accuracy of logaddexp.reduce | 📝 Không repro được | enh về độ chính xác logaddexp.reduce, không có repro |
| #6846 | built-in ufunc(s) unexpectedly returns half-floats | ❌ Còn | `sin(int8)` vẫn ra float16 |
| #6854 | Bincount with integer weights should return integer array | ❌ Còn | `bincount` trọng số int vẫn ra float64 |
| #6971 | we should not silently truncate floats to ints | ❌ Còn | `a[1]=2.5` vẫn cắt thành 2, không cảnh báo |
| #8237 | Raise precision of numpy.std and numpy.var in float16 | ❌ Còn | var float16 vẫn ra inf (overflow reduce) |
| #8804 | No modular exponentiation? | ◐ Một phần | `pow(arr,10,3)` nay báo TypeError (hết bỏ qua modulus), chưa có modular pow |
| #8869 | numpy.mean along multiple axis gives wrong result for large arrays | ❌ Còn | mean float32 axis=(0,1) vẫn ra 0.4194304 thay vì 1.0 |
| #8987 | BUG: Integer overflow warning applies to scalars but not arrays | ❌ Còn | scalar cảnh báo overflow, mảng thì không |
| #9068 | np.ceil and np.floor are inconsistent with math.ceil and math.floor | ❌ Còn | `floor/ceil` vẫn trả float |
| #9733 | large int64 number compares to float number getting inaccurate result | ❌ Còn | `int64(1e16)==1e16+1.0` vẫn True |
| #13007 | float(np.complex128) silently drops imaginary part | ◐ Một phần | `float(complex128)` nay phát ComplexWarning, vẫn bỏ phần ảo |
| #13199 | var, std memory consumption | ❌ Còn | var vẫn cấp phát thêm đúng bằng 1 mảng (200MB/200MB) |
| #15630 | np.clip with complex input is untested and has odd behavior | ❌ Còn | `clip` complex vẫn cho kết quả lạ ([2.+0.j]) |
| #15856 | numpy has no boolean subtract functionality | ◐ Một phần | `bool - bool` từ deprecation thành TypeError; chưa có định nghĩa mới |
| #15981 | NumPy should warn (eventually raise an error?) on comparisons between … | ❌ Còn | so sánh complex vẫn trả True, không cảnh báo |
| #16903 | bincount fails for complex weights | ❌ Còn | `bincount` weights complex vẫn TypeError |
| #21462 | BUG: NumPy sometimes converts non-integer array-like to integer ones d… | 📝 Không repro được | cần nhánh ctors.c cụ thể; repro đơn giản không kích hoạt |
| #25621 | Deprecating in-place operations where the out-of-place equivalent woul… | ❌ Còn | `a+=int16(4096)` vẫn im lặng cắt về int8 |
| #26145 | BUG/question: Should np.abs always return a positive number? | ❌ Còn | `abs(int8 -128)` vẫn ra -128 |

## Rounding (5)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #3540 | subtypes and round() | ❌ Còn | `round(decimals=-1)` vẫn trả ndarray thay vì subclass |
| #6248 | ENH: Implement __round__ special method for ndarrays | ❌ Còn | `round(array)` vẫn TypeError |
| #9791 | BUG: np.round should fall back on `__round__` for object arrays | ❌ Còn | `np.round` object array vẫn không dùng `__round__` |
| #13699 | Surprising overflows in np.round of float16. | ❌ Còn | `round(float16(2.0),5)` vẫn ra nan/overflow |
| #15438 | TypeError when using np.around() on an integer array with in-place opt… | ❌ Còn | `around(int, decimals=-1, out=a)` vẫn UFuncTypeError |

## Structured dtype, record, np.void (11)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #1076 | numpy.rec.array is inconsistent on objects (Trac #478) | ❌ Còn | cặp subarray vẫn lệch cấp lồng; cặp scalar đã nhất quán |
| #1737 | record array: passing formats as tuple raises error (Trac #1139) | ❌ Còn | `formats=` dạng tuple vẫn ValueError (list thì ok) |
| #6743 | recfunctions.merge_arrays fails to merge some arrays with standard fie… | ❌ Còn | merge_arrays vẫn ValueError 'f0 occurs more than once' |
| #7552 | Make ndarray and np.void splattable | ❌ Còn | ndarray vẫn không có `keys` |
| #8969 | BUG: String fields in compound dtypes don't resize as might be expecte… | ❌ Còn | trường str vẫn thành '' (không resize) |
| #9313 | BUG: Subarrays casts truncate and zero-pad without error or warning | ❌ Còn | cast subarray→int vẫn lấy phần tử đầu (ra 1) |
| #11683 | core.records.fromfile fails without shape= | ✅ Đã giải quyết | repro chạy được từ 1.20 (PR #16675, file-like fromfile); lỗi trước 1.20 là TypeError khác lỗi gốc |
| #12207 | BUG: subclasses of np.void can cause a segfault | ✅ Đã giải quyết | biến thể `.dtype`: ValueError từ 2.3 (PR #28254 expire deprecation; #30179 xử lý tiếp); biến thể `_type_`: PR #12254; xem ISSUES_CLOSE_EVIDENCE.md |
| #13683 | np.full(..., dtype=structured-dtype) does not work as expected | ❌ Còn | `np.full` structured ra giá trị sai + cast warning |
| #15470 | np.rec.fromarrays(...) may fail if resulting array is going to be empt… | ❌ Còn | `rec.fromarrays` rỗng + subarray vẫn lỗi broadcast |
| #15849 | Conversion of nan in numpy.core.records.fromarrays | ❌ Còn | vẫn ValueError cannot convert NaN |

## dtype nói chung (12)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #1725 | Make the dtype object immutable and not coerce other types when compar… | ❌ Còn | đổi `dtype.names` vẫn lan sang view |
| #4317 | Pickling/unpickling a dtype resets isbuiltin flag | ❌ Còn | pickle/copy vẫn đặt isbuiltin=0 |
| #7829 | np.concatenate loses endianness / byte order | ❌ Còn | `concatenate` vẫn mất byte order '>i2'→int16 |
| #8849 | ENH: Implement np.iinfo(np.bool_) | ❌ Còn | `iinfo(np.bool_)` vẫn ValueError |
| #9049 | _dtype_from_pep3118 is overly strict on prefixes | ❌ Còn | vẫn lỗi (đổi thành ValueError 'not a valid PEP 3118') |
| #9496 | Can't change dtype for non-continuous array | ✅ Đã giải quyết | `.view(complex)` lỗi ≤1.22.4, chạy từ 1.23.5; PR #20722 |
| #15473 | Scalar constructors behave inconsistently on arrays | ❌ Còn | `np.int32(array)` vẫn không nhất quán |
| #16052 | Different behaviour in np.array between object/scalar 0d arrays in ite… | ❌ Còn | `np.array([np.array(obj)])` vẫn lồng thêm array |
| #16391 | Numpy timedelta64 NaT not converted to uint64 NaN | ❌ Còn | NaT→uint64 vẫn ra 9223372036854775808 |
| #16624 | TRACKING: Review and possibly address changes regard new dtypes | 📝 Không repro được | issue theo dõi/tracking |
| #17544 | cumsum() changes endianness | ❌ Còn | `cumsum(dtype='>i4')` vẫn ra int32 little-endian |
| #23500 | ENH: Support new-style custom dtypes in the buffer protocol | 📝 Không repro được | enh buffer protocol cho dtype mới |

## Subclass, memmap, pickle (8)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #2348 | empty_like not passing across attributes of memmap objects (Trac #1753… | ❌ Còn | `empty_like(memmap)` vẫn trả memmap với `_mmap=None` |
| #2402 | Instance attributes of ndarray subtypes get lost during pickling (Trac… | ❌ Còn | thuộc tính instance vẫn mất khi pickle subclass |
| #3143 | Memmap cannot use existing file handles. | ❌ Còn | `np.load(mmap_mode)` với file handle vẫn ValueError |
| #3758 | Exception "assignment destination is read-only" when reading from a re… | ❌ Còn | `take` với index read-only subclass vẫn ValueError |
| #11265 | BUG: PyArray_SetBaseObject is dangerous on any subclass that defines _… | 📝 Không repro được | vấn đề C API, cần extension để kiểm |
| #11388 | ENH: Memory map should warn on dtypes with objects | ❌ Còn | memmap dtype object vẫn không cảnh báo |
| #11437 | squeeze on memmap returns numpy.ndarray instead of memmap | ❌ Còn | `memmap.squeeze()` vẫn trả ndarray |
| #13172 | ENH: Add `madvise` for `memmap` objects | ❌ Còn | `np.memmap` chưa có `madvise` |

## Shape, reshape, broadcast, indexing (21)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #2486 | Broadcasting shape mismatch exception (Trac #1893) | ❌ Còn | vẫn chỉ ValueError, chưa có BroadcastError |
| #2667 | Writing to a transpose results in strange error (Trac #2075) | ❌ Còn | `a.T += b` vẫn báo lỗi nhưng đã ghi |
| #3016 | Contains method is not consistent for subarrays | ❌ Còn | `[0,1,2] in a` vẫn trả True |
| #3817 | reshape() should have a way to keep a dimension unchanged. | ❌ Còn | chưa có keepaxis (đề xuất thiết kế) |
| #4320 | small boolean assignment incosistencies | ❌ Còn | gán bool với trailing-ones / sequence vẫn TypeError |
| #7964 | BUG: np.reshape sometimes returns scalars | ❌ Còn | `reshape(np.int64(0),())` vẫn ra scalar |
| #8102 | "axes" argument for reshape/flatten/ravel | ❌ Còn | `reshape/flatten` chưa có tham số ndim |
| #8545 | Incorrect flags on result of elipsis indexed scalar | ❌ Còn | `int_(1)[...]` vẫn writeable |
| #8802 | Broadcasting Assigments In a Loop | 📝 Không repro được | đề xuất cảnh báo broadcast, không có repro |
| #8875 | Strange roll multiple shift behavior | ❌ Còn | roll nhiều shift không axis vẫn cộng dồn |
| #8899 | ENH: Possible improvements to `np.block` | ❌ Còn | `np.b_` vẫn chưa có |
| #8972 | ValueError reshaping empty arrays | ❌ Còn | `reshape(-1,0)` vẫn ValueError, thông báo khó hiểu |
| #9818 | Proposal: add an optional copy argument to np.reshape | ✅ Đã giải quyết | `reshape(copy=)` lỗi ≤2.0.2, chạy từ 2.1.3; PR #26292 |
| #11309 | Attempt to grow array by assigning a size-1 array to a size-0 slice do… | ❌ Còn | `a[5:5]=[123]` vẫn im lặng |
| #14168 | nditer usage with index and buffered flags | ❌ Còn | nditer buffered+multi_index vẫn luôn trả (0,0,0) |
| #14396 | asarray array disagrees with isscalar about what is a scalar (at least… | ❌ Còn | `asarray({1,2,3})` vẫn shape () |
| #15318 | Feature request: advanced slicing (on left hand side) | 📝 Không repro được | đề xuất tính năng |
| #15475 | ENH: Suggest "ignore" mode for ravel_multi_index | ❌ Còn | `ravel_multi_index` chưa có mode 'ignore' |
| #16179 | slightly confusing error message for when calling reshape() with an in… | ❌ Còn | thông báo vẫn in shape (2,3) thay vì (-1,2,3) |
| #17042 | DEP: Deprecate flatiter attributes (which produce confusing results) | ❌ Còn | `flatiter.index/coords` vẫn lệch 1 |
| #17175 | BUG: Boolean indexing broken in `np.flatiter` | ◐ Một phần | PR #28590 (2.4.0): list bool nay IndexError; `flat[True]` (0-d) mới chỉ DeprecationWarning; issue gắn sustain-2026 |

## Ufunc / gufunc (14)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #2543 | Redundant numeric type classes lead to unreliable behavior of isinstan… | ✅ Đã giải quyết | isinstance đúng ở 2.5.3 (Linux); PR #4547 (ABC, NumPy 1.9) theo comment maintainer; chưa thử Windows |
| #3994 | abs() is slow for complex, add abs2() | ❌ Còn | vẫn chưa có `abs2` |
| #7002 | Get rid of special scalar arithmetic. | 📝 Không repro được | thảo luận thiết kế |
| #8811 | Feature request: signal broadcasting is OK over core dimension | 📝 Không repro được | thảo luận thiết kế gufunc |
| #8867 | Discussion: Could there be a ufunc type hierarchy? | 📝 Không repro được | thảo luận thiết kế |
| #8975 | BUG: np.fmin behaves differently on object arrays | ❌ Còn | `fmin.reduce` object vẫn ra nan (float ra 1.0) |
| #8994 | ENH: make np.where a ufunc | ❌ Còn | `np.where` vẫn không phải ufunc |
| #9009 | ENH: Add a np.isnan loop for the object dtype (and possible `isfinite`… | ❌ Còn | `isnan` object vẫn TypeError |
| #11109 | UFunc userloop selection does not work when only output is void/user d… | 📝 Không repro được | cần C extension để repro |
| #11118 | Possible generalization of gufunc keepdims for multiple outputs | 📝 Không repro được | nhắc việc thiết kế |
| #11228 | Gufunc helper for sizes, flags and num_dims? | 📝 Không repro được | nhắc việc refactor |
| #12142 | Scalar compared with an obj implementing __array_ufunc__ becomes 0d-ar… | ❌ Còn | `float32 < obj` vẫn truyền 0-d ndarray vào __array_ufunc__ |
| #17359 | MAINT: Figure out alignment for complex loops | 📝 Không repro được | thảo luận alignment |
| #30413 | ENH: Allow ufuncs to request the iterator to provide contiguous arrays | 📝 Không repro được | enh, không phải bug |

## Reduction (6)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #834 | reduceat cornercase (Trac #236) | ❌ Còn | `add.reduceat(a,(1,1))` vẫn ra [1,10] |
| #835 | reduceat should handle outlier indices gracefully (Trac #237) | ❌ Còn | `reduceat` với index len(a) vẫn IndexError |
| #7179 | ENH: linalg.qr should be a gufunc (which will release the GIL) | ✅ Đã giải quyết | qr gufunc: stacked qr lỗi ở 1.21.6, chạy ở 1.22.4; PR #19151 |
| #9182 | ENH: argmax can be made faster for non-contiguous axes. | ❌ Còn | `argmax axis=0` vẫn copy toàn bộ (288MB) và chậm |
| #14371 | ENH: Adding where for argmin | ❌ Còn | `argmin` chưa có `where` |
| #18512 | ENH: Iterator does not block and NumPy has no transposed copy fast-pat… | ❌ Còn | argmax nx=1280 vẫn chậm gấp đôi 1281 (0.105s vs 0.053s) |

## Sort, search, set (9)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #641 | python3: regression for unique on dtype=object arrays with varying ite… | ❌ Còn | `unique` object trộn kiểu vẫn TypeError |
| #6061 | Calling searchsorted on an int haystack with str needle converts hayst… | ❌ Còn | `arange(1000).searchsorted('15')` vẫn ra 150 |
| #7051 | Sorting should work for array scalars. | ❌ Còn | `sort` 0-d vẫn AxisError |
| #8657 | Feature Request: add mergesorted and argmergesorted functions | ❌ Còn | chưa có mergesorted |
| #9331 | assert_array_equal does not compare NaNs as numbers when dtype is obje… | ❌ Còn | `assert_array_equal` object chứa nan vẫn fail |
| #11136 | unique() needlessly slow | ❌ Còn | `unique(axis=0)` chậm hơn view-void ~4.5x |
| #15008 | BUG: np.interp casts strings to floats | ❌ Còn | `interp('2',...)` vẫn ra 20.0 |
| #15499 | BUG: searchsorted with object arrays containing nan | ❌ Còn | `searchsorted` object chứa nan vẫn sai |
| #16377 | array_equal(a, b, equal_nan=True) throws errors for array with non-num… | ❌ Còn | `array_equal(equal_nan=True)` với str vẫn TypeError |

## File I/O (6)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #2230 | numpy.fromfile does not accept StringIO object (Trac #1634) | ❌ Còn | `fromfile(BytesIO)` vẫn lỗi (UnsupportedOperation: fileno) |
| #2682 | `npy_PyFile_Dup` should check file object subclass (Trac #2090) | ◐ Một phần | gzip nay ValueError thay vì rác [-1.], chưa phải lỗi sạch |
| #8458 | `np.fromfile` silently truncates integers > signed 64 bit when reading… | ❌ Còn | `fromfile` vẫn cắt số > int64 |
| #15442 | DEP: fromfile returns shorter array if `count` argument exceeds file c… | ◐ Một phần | crash đã sửa; `count` vượt file vẫn trả ngắn, không DEP |
| #17118 | zipfile with inner binary file could not be read numpy.fromfile | ❌ Còn | `fromfile(zipfile.Path)` vẫn AttributeError (issue sustain-2026) |
| #18435 | np.fromstring("\n", sep=" ") returns non-empty array | ❌ Còn | `fromstring('\n',sep=' ')` vẫn ra [-1.] |

## String, unicode, longdouble, printing (12)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #2376 | Support full long double precision in __format__ (Trac #1783) | ❌ Còn | format longdouble vẫn mất độ chính xác |
| #3481 | Multiplication a unicode/bytes string arrays by an integer fails | ◐ Một phần | `a+a` đã chạy; `2*a` vẫn TypeError |
| #7594 | Roundtripping error for unicode ndarrays with char.encode and char.dec… | 📝 Không repro được | repro đơn giản (utf-32-be) round-trip đúng; cần ca có null cuối |
| #7619 | BUG: .item() on a 0-dimensional datetime64[ns] array yields an integer | ❌ Còn | `.item()` datetime64[ns] vẫn ra int |
| #8089 | python incompatibility: bytes_ behaviour inconsistent with python | ❌ Còn | `bytes_` vẫn cắt null cuối |
| #8161 | BUG: datetime64 construction can underflow | ✅ Đã giải quyết | đúng ở mọi bản thử được (1.17.5–2.5.3); chưa xác định PR (ứng viên #13188) |
| #8224 | set_printoptions: custom formatter issues with string types | ❌ Còn | formatter str/bytes vẫn không tách riêng được |
| #11547 | printoptions "suppression" of scientific method is not doing anything … | ❌ Còn | `suppress` vẫn không áp dụng ≥1e8 (docs chưa nêu) |
| #16921 | BUG: String to complex longdouble conversions are  not full precision | ❌ Còn | clongdouble(str) vẫn qua float64 (trùng #26701) |
| #22098 | ENH: longdouble(int) performs poorly due to an unnecessary base 10 tra… | 📝 Không repro được | perf longdouble(int) không đo được (giới hạn 4300 chữ số của Python) |
| #25910 | ENH: Optimize np.strings.expandtabs to avoid unnecessary copies | 📝 Không repro được | tối ưu hiệu năng, sustain-2026 |
| #26701 | BUG: `clongdouble(str)` parses string to `float64` and loses precision… | ❌ Còn | clongdouble(str) vẫn qua float64 (trùng #16921) |

## Platform / kiến trúc (9)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #8097 | Crash when importing numpy from the Python C-API after calling Py_Fini… | 📝 Không repro được | cần chương trình nhúng Py_Finalize, chưa thử |
| #12638 | BUG: several test errors on SPARC | 📝 Không repro được | cần SPARC |
| #13809 | C Fortran ABI issues with character arguments | 📝 Không repro được | ABI Fortran, không repro bằng Python |
| #15754 | Underflow in npy_math_complex.c.src on ppc64le | 📝 Không repro được | cần ppc64le |
| #15763 | BUG: ppc64le uses double-double for np.float128, routines need adjustm… | 📝 Không repro được | cần ppc64le |
| #19243 | npy_cblas.h does not match libcblas (size_t vs int or long for CBLAS_I… | 📝 Không repro được | vấn đề header CBLAS, không repro bằng Python |
| #25460 | BUG: `np.log(max_value_for_longdouble)` fails on Linux aarch64 | 📝 Không repro được | cần aarch64 |
| #25869 | BUG: incompatible pointer type for `npy_int32t` and `npy_uint32` on i6… | 📝 Không repro được | cần i686 |
| #26746 | np.fmin returns inconsistent results for -0.0 and 0.0 across different… | ❌ Còn | fmin(±0.0) vẫn phụ thuộc thứ tự tham số (Linux) |

## Hiệu năng (6)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #619 | BLAS matrix product (dot) never used for ndim > 2 (tensordot does not … | 📝 Không repro được | chưa đo BLAS cho ndim>2; kết quả đúng |
| #2657 | Poor ndarray.take performance on Fortran order arrays (Trac #2065) | ❌ Còn | `take` F-order: 2.6ms vs 0.9µs C-order |
| #8480 | np.mean with axis parameter is slower than naive implementation | ❌ Còn | `mean(axis=(0,1))` 18.5ms vs naive 4.5ms |
| #11232 | Ufunc calls on scalars are very slow | ◐ Một phần | gap thu hẹp (np.sin 220ns vs math.sin 151ns) nhưng vẫn chậm hơn |
| #13229 | Complex matmul is very slow (and here's a 2x speedup) | ◐ Một phần | n=1024: matmul complex 0.20s vs 3-real 0.19s, không còn thấy speedup 2x |
| #13579 | Speed problem for searchsorted when different integer dtypes | ❌ Còn | `searchsorted` int8 với Python int vẫn chậm 100x |

## Correlate, einsum, cross, linalg-like (11)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #1530 | Add new convolve method for faster computation of even functions (Trac… | 📝 Không repro được | đề xuất tính năng |
| #1858 | Use FFT in np.correlate/convolve? (Trac #1260) | 📝 Không repro được | đề xuất dùng FFT, chưa có |
| #1905 | take with axis=1 from matrix with matrix indices gives row instead of … | ❌ Còn | `take(matrix, matrix([[0]]), axis=1)` vẫn trả hàng (1,3) ở 1.19.5, 2.1.3, 2.5.3 |
| #2310 | normalized cross-correlation (Trac #1714) | 📝 Không repro được | đề xuất tính năng (nên ở SciPy) |
| #2453 | einsum with arbitrary operations: chaining ufuncs using einsums `'ij,j… | 📝 Không repro được | đề xuất tính năng |
| #6631 | Request: currying `numpy.einsum` for repeated calls with the same subs… | 📝 Không repro được | đề xuất tính năng |
| #13233 | enhancement: axes keyword for cross | ❌ Còn | `cross` chưa có `axes` |
| #13718 | Splitting np.cross into np.cross and np.cross2d? | ◐ Một phần | NumPy 2.x bỏ cross 2D (ValueError), chưa có cross2d riêng |
| #13797 | Hermitian Transpose Syntax | ◐ Một phần | `.mT` đã có, `.mH` chưa |
| #15103 | Will tensordot support an 'out' argument? | ❌ Còn | `tensordot` chưa có `out` |
| #17286 | ENH: Implement a maxlag like feature for `np.correlate` | ❌ Còn | `correlate` chưa có maxlag |

## C API, bộ nhớ, buffer (9)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #3634 | Use the builtin BufferError for the buffer API | ◐ Một phần | buffer.c có 1 BufferError so với 12 ValueError |
| #6581 | Memory leak when array contains circular references | ❌ Còn | mảng chứa tham chiếu vòng vẫn rò (1 object còn lại) |
| #6811 | frombuffer segfault | ❌ Còn | `frombuffer(np.array(None),int64)[0]=1` vẫn segfault |
| #9311 | determining the datatype of recursive list is slow | ❌ Còn | `np.array(c)` với list đệ quy vẫn treo |
| #13547 | PyArray_FromAny ignores newtype with memoryview | ◐ Một phần | maintainer nói hành vi đúng; chỉ còn cleanup ở np.compress |
| #13831 | Supporting duck array coercion | 📝 Không repro được | đề xuất duck-array coercion |
| #21533 | Python API to allocator | 📝 Không repro được | đề xuất API allocator |
| #21676 | BUG: np.compress casting fails from narrower to wider type | ◐ Một phần | i4→f8 nay ok (kèm DeprecationWarning), f8→i4 đang deprecate |
| #27456 | ENH: Discussion: Integrating Numpy with CPython interpreter specializa… | 📝 Không repro được | thảo luận hiệu năng interpreter |

## Maintenance, docs, khác (15)

| Issue | Tiêu đề | Kết luận | Ghi chú |
|---|---|---|---|
| #6197 | Confusing warning for median of empty array. | ❌ Còn | `median([])` vẫn cảnh báo 'Mean of empty slice' + warning phụ |
| #9103 | MaskedArray min/max methods do not work for dtype object | ❌ Còn | `MaskedArray.min` object vẫn lỗi (TypeError) |
| #10290 | BUG: ptp fails on datetime types with out parameter | ❌ Còn | `ptp(out=)` datetime vẫn UFuncTypeError |
| #10296 | Audit `mem_overlap.c` to rationalize the use of unsigned integers. | 📝 Không repro được | audit code |
| #10801 | DOC: Document einsum's index parsing and mapping code | 📝 Không repro được | docs |
| #11266 | missing np.get_string_function (counterpart of np.set_string_function) | ✅ Đã giải quyết | `set_string_function` có ≤1.26.4, gỡ ở 2.0.2; PR #26611 |
| #11407 | np.exp raises AttributeError when called with large integer | ◐ Một phần | nay TypeError thay vì AttributeError, vẫn không ra inf |
| #11502 | ENH: for one variable data, np.cov should return either a scalar or a … | ❌ Còn | `cov` 1 hàng vẫn ra shape () |
| #13654 | MAINT: Fix undefined behaviour issues with memcpy | 📝 Không repro được | audit UB |
| #15567 | DOC: how to find your way in the code, for instance the dot function | 📝 Không repro được | docs |
| #15726 | Documentation for numpy.fromfunction induces an erroneous interpretati… | 📝 Không repro được | docs |
| #16469 | [Feature Request] Add alias of np.concatenate as np.concat  | ✅ Đã giải quyết | `np.concat` chưa có ≤1.26.4, có từ 2.0.2; PR #25086 |
| #16569 | BUG: `order='K'` behavior for `tobytes` | ❌ Còn | `tobytes('K')` vẫn không khớp `ravel('K')` |
| #18798 | accept a dtype argument in np.choose | ❌ Còn | `choose` chưa có `dtype` |
| #21457 | MAINT: Create static inline functions for builtin (numeric?) casts | 📝 Không repro được | refactor |
