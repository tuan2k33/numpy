# Bằng chứng: issue `component: numpy._core` có thể đã được giải quyết

Ngày kiểm: 2026-10-01. Đây là **dữ liệu để bạn tự viết comment** (chính sách AI của numpy, `doc/source/dev/ai_policy.rst`, không cho phép dùng AI viết thay trong issue/PR và không cho agent tự đăng). Không có comment nào đã được đăng.

**Cách kiểm:** cài từng phiên bản numpy từ wheel bằng `uv run --no-project --python <py> --with numpy==<ver>` (Linux x86_64) và chạy cùng một đoạn repro. Xác định PR bằng: (1) khoảng phiên bản mà hành vi đổi, (2) release notes, (3) diff/mô tả PR. Chưa chạy trên bản build từ `main` (2.6.0.dev); 2.5.3 là bản mới nhất đã thử.

**Đã loại khỏi danh sách sau khi kiểm lại:** #1905 (`take` trên `matrix` vẫn trả hàng (1,3) ở cả 1.19.5 và 2.5.3, nên chưa sửa; lần trước tôi kết luận nhầm) và #17175 (chỉ sửa một phần, xem cuối file).

| Issue | PR sửa | Phiên bản đổi | Độ chắc chắn |
|---|---|---|---|
| #7179 | #19151 | 1.21.6 → 1.22.4 | Cao |
| #9496 | #20722 | 1.22.4 → 1.23.5 | Cao |
| #9818 | #26292 | 2.0.2 → 2.1.3 | Cao |
| #11266 | #26611 | 1.26.4 → 2.0.2 | Cao (API bị gỡ) |
| #16469 | #25086 | 1.26.4 → 2.0.2 | Cao |
| #12207 | #28254 (+ #30179, #12254) | 2.2.6 → 2.3.2 | Trung bình-cao |
| #11683 | #16675 | 1.19.5 → 1.20.3 | Trung bình (nguyên nhân lỗi khác so với lỗi gốc của issue) |
| #8161 | chưa chắc; ứng viên #13188 | đã đúng từ 1.17.5 | Thấp |
| #2543 | #4547 | 1.9 (theo comment maintainer) | Thấp-trung bình, chưa kiểm lại trên Windows |

---

## #7179 — `linalg.qr` nên là gufunc

- **Issue nói:** `np.linalg.qr` không phải gufunc nên không chạy trên stack và không nhả GIL; đổi tên thành "qr should be a gufunc".
- **Repro:** `np.linalg.qr(np.ones((2,3,3)) + np.eye(3))` (mảng xếp chồng 3 chiều) và `hasattr(np.linalg._umath_linalg, 'qr_reduced')`.
- **Kết quả:**

  | numpy | stacked qr | có gufunc `qr_reduced` |
  |---|---|---|
  | 1.19.5 | LinAlgError | False |
  | 1.21.6 | LinAlgError | False |
  | 1.22.4 | shape (2,3,3) | True |
  | 1.23.5, 1.26.4, 2.0.2, 2.1.3 | shape (2,3,3) | True |
- **PR:** #19151 "ENH: Vectorising np.linalg.qr" (merged 2021-07-14), phát hành trong 1.22.
- **Source trên `main`:** [numpy/linalg/_linalg.py](numpy/linalg/_linalg.py) gọi `_umath_linalg.qr_r_raw`, `qr_complete`, `qr_reduced`.
- **Lưu ý:** phần "nhả GIL" trong tên cũ của issue tôi không đo được chắc chắn (kết quả đo thread bị nhiễu do BLAS đa luồng). Bằng chứng chính là qr thành gufunc.

## #9496 — không đổi dtype được cho mảng không liên tục

- **Repro:** `np.ones((3,4,2))[:, ::2, :].view(complex).shape`
- **Kết quả:** ValueError ở 1.19.5, 1.21.6, 1.22.4; `(3, 2, 1)` ở 1.23.5, 1.26.4, 2.x.
- **PR:** #20722 "ENH: Removed requirement for C-contiguity when changing to dtype of different size" (merged 2022-01-06), release notes 1.23.0 (mục về mảng không liên tục theo trục cuối).
- **Ghi chú:** timeline issue chỉ liệt kê PR cũ #9497 (đã đóng, không merge); đừng dẫn nhầm.

## #9818 — thêm tham số `copy` cho `np.reshape`

- **Repro:** `np.reshape(np.arange(6), (2,3), copy=False)`
- **Kết quả:** TypeError ở 1.19.5…2.0.2; `(2, 3)` ở 2.1.3.
- **PR:** #26292 "API: Add `shape` and `copy` arguments to `numpy.reshape`" (merge commit `2da02ea321`). Liên quan: #19173 (copy modes) và issue #11884 trùng chủ đề.

## #11266 — thiếu `np.get_string_function`

- **Repro:** `hasattr(np, 'set_string_function')`
- **Kết quả:** True ở 1.19.5…1.26.4; False ở 2.0.2 và mới hơn (cả `get_string_function` chưa từng có).
- **PR:** #26611 "remove set_string_function" (merge `a4cddb6048`, 2024-06-17). Vì API bị gỡ, nhu cầu của issue (khôi phục string function cũ) không còn.

## #16469 — thêm alias `np.concat`

- **Repro:** `hasattr(np, 'concat')`
- **Kết quả:** False ở 1.19.5…1.26.4; True ở 2.0.2 và mới hơn.
- **PR:** #25086 "API: Add Array API aliases (math, bitwise, linalg, misc)" (merged 2023-12-08). Diff có `concat = concatenate` trong numeric.py và `concat` trong `__all__`; release notes 2.0.0 liệt kê "Misc: concat, permute_dims, pow".

## #12207 — subclass của `np.void` gây segfault

- **Repro:** `class W(np.void): dtype = np.int32` rồi `np.dtype(W)` (biến thể `_type_` không còn crash ở 2.5.3).
- **Kết quả (cho biến thể `.dtype`, chạy với warnings=error):**

  | numpy | kết quả |
  |---|---|
  | 1.19.5 | không lỗi |
  | 1.21.6 … 2.2.6 | DeprecationWarning |
  | 2.3.2, 2.3.5, 2.4.x, 2.5.x | ValueError ("`.dtype` attribute ... is not a valid dtype instance") |
- **PR:** #28254 "MAINT: expire deprecations" (2025-02, NumPy 2.3.0). Release notes 2.3.0 ghi: thuộc tính `dtype` phải là dtype instance (deprecated từ 1.19). PR #30179 (2025-11-27) xử lý tiếp việc đệ quy và thêm test `class vdt(np.void): dtype = vdt`. PR #12254 (2018) xử lý phần ctypes `_type_`.
- **Lưu ý:** issue từng được seberg đóng rồi mở lại (2021-11-05) vì còn lỗi refcount; tôi chưa kiểm refcount/leak ở 2.5.3.

## #11683 — `records.fromfile` lỗi khi không có `shape=`

- **Repro:** ghi 10 bản ghi `f8,i4,S5` vào `TemporaryFile`, `seek(0)`, rồi `np.rec.fromfile(f, formats='f8,i4,S5', byteorder='<')`.
- **Kết quả:** TypeError ở 1.17.5, 1.19.5 (`expected str, bytes or os.PathLike object, not _io.BufferedRandom`); shape `(10,)` ở 1.20.3, 1.21.6, 1.23.5, 1.26.4, 2.1.3, 2.5.3.
- **PR:** #16675 "ENH: Add support for file like objects to np.core.records.fromfile" (merged 2020-08-13), release notes 1.20.0.
- **Lưu ý quan trọng:** TypeError ở 1.19.5 do `fromfile` không nhận file object, **khác** với lỗi ban đầu trong issue (2018, truy vết còn cắt cụt). Nên chỉ nói "repro trong issue hiện chạy được từ 1.20 nhờ #16675", không khẳng định đây là PR sửa lỗi gốc.

## #8161 — `datetime64` construction tràn/underflow

- **Repro:** `np.datetime64(np.iinfo(np.int64).min + 80000000000000, 'ns')`
- **Kết quả:** `1677-09-21T22:26:03.145224192` (đúng) ở 1.17.5, 1.19.5, 1.20.3, 1.21.6, 1.23.5, 1.26.4, 2.1.3, 2.5.3. Issue báo `2262-04-11...` ở 1.11.1. Không cài được 1.11–1.16 (thiếu Python 3.7/3.6), nên **không biết chính xác phiên bản đổi hành vi**.
- **PR ứng viên (chưa xác nhận):** #13188 "Simplify logic in convert_datetime_to_datetimestruct" (merged 2019-03-29) kèm commit `27efe4d1e2` thêm test round-trip các giới hạn datetime. Timeline issue chỉ có PR #11873 (đã đóng, không merge).
- **Gợi ý:** nếu muốn chắc, build hai commit trước/sau `15b14c5212` để so sánh.

## #2543 — isinstance không nhất quán với kiểu số của numpy

- **Repro:** `isinstance(np.abs(np.int32(5)), np.int32)`, `isinstance(np.int32(5), numbers.Integral)`.
- **Kết quả (Linux, 2.5.3):** cả hai đều True.
- **PR:** #4547 "add support for python ABCs" (merged 2014-03-25, commit `2d73ff34f4`), theo comment của maintainer: "we now register with Number, so it works fine beginning with 1.9".
- **Lưu ý:** issue gốc là Windows 32-bit (hai class `numpy.int32` khác nhau), tôi chưa kiểm được trên Windows.

---

## Không đưa vào danh sách đóng

- **#1905:** `np.take(np.matrix(...), np.matrix([[0]]), axis=1)` vẫn trả shape `(1, 3)` (hàng) ở 1.19.5, 2.1.3 và 2.5.3; issue mong đợi cột `(3, 1)`.
- **#17175** (label `sustain-2026`): PR #28590 (NumPy 2.4.0) làm `arr.flat[[True, False, ...]]` ném IndexError (2.3.5 vẫn coi như số nguyên), nhưng index bool 0 chiều `arr.flat[True]` mới chỉ DeprecationWarning ở 2.4.0 và 2.5.x. Mới sửa một phần; label cho thấy issue được giữ cho sự kiện NumFOCUS.

## Việc nên làm trước khi comment

1. Build `main` và chạy lại các repro trên đó (hiện mới tới 2.5.3).
2. Với #8161 và #2543, kiểm thêm (bisect, Windows) hoặc nói rõ độ chắc chắn thấp.
3. Tự viết comment bằng lời của bạn, nêu: phiên bản đã thử, đoạn repro ngắn, output, PR liên quan.
