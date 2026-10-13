# Issue về làm tròn (`round` / `rounding`) trong numpy/numpy

Snapshot 2026-10-06, chỉ issue **đang mở**. Tổng hợp từ 32 truy vấn tìm kiếm (`round`, `rounding`, `rounded`, `np.round`, `np.around`, `numpy.round`, `__round__`, `rint`, `ceil`, `floor`, `trunc`, `np.fix`, `half to even`, `ties to even`, `banker`, `decimals`, ...) trong tiêu đề và nội dung: **107 issue khớp từ khóa**, sau khi đọc đoạn văn quanh từ khóa còn **35** issue thật sự liên quan (xếp thành 4 nhóm bên dưới), 72 issue chỉ nhắc "round"/"around"/"round-trip" ngẫu nhiên (liệt kê ở cuối). 6 issue đầu của nhóm A là các issue gắn tag `component: numpy._core` đã có trong [ISSUES_CORE.md](ISSUES_CORE.md); phần còn lại không có tag đó.

**Giới hạn:** tìm theo từ khóa nên có thể sót issue không chứa các từ này; phân loại dựa vào việc đọc tiêu đề và đoạn văn quanh từ khóa. Ký hiệu: ✔ = tôi đã chạy repro trên numpy 2.5.3 (và bản build baseline/`perf/fused-round` khi liên quan); 📖 = chỉ đọc nội dung, chưa chạy. Chưa kiểm trên bản build từ `main`; chưa đăng gì lên GitHub.

## A. Hàm làm tròn: `round` / `around` / `rint` / `ceil` / `floor` / `trunc` / `fix` / `__round__` (15)

| Issue | Tiêu đề | Cập nhật | Trạng thái | Ghi chú |
|---|---|---|---|---|
| #3540 (core) | subtypes and round() | 2018-10-21 | ❌ Còn ✔ | Subclass mất ở một số tổ hợp `decimals`; nhánh `perf/fused-round` không đổi (chỉ `ndarray` chính xác mới vào đường mới). |
| #6248 (core) | ENH: Implement __round__ special method for ndarrays | 2022-12-14 | ❌ Còn ✔ | `round(array)` vẫn TypeError chung. |
| #9068 (core) | np.ceil and np.floor are inconsistent with math.ceil and math.floor | 2026-04-02 | ❌ Còn ✔ | `np.floor/ceil(1.5)` vẫn trả float; đề xuất hàm riêng `iceil/ifloor`. |
| #9791 (core) | BUG: np.round should fall back on `__round__` for object arrays | 2017-09-29 | ❌ Còn ✔ | Mảng object không dùng `__round__` (đường cũ, không đổi). |
| #13699 (core) | Surprising overflows in np.round of float16. | 2021-10-20 | ✅ Sửa (float16) ✔ | Nhánh `perf/fused-round` (chưa merge) sửa float16: `round(f16(2.0), 5)` ra `2.0` thay vì `nan`. float64/float32 giá trị lớn tràn vẫn như cũ (thuộc giai đoạn "round chuẩn"). |
| #15438 (core) | TypeError when using np.around() on an integer array with in-place option set | 2020-01-27 | ❌ Còn ✔ | `out` kiểu int vi phạm `same_kind`; doc ghi "cast if necessary". |
| #10574 | numpy.round may return a view of the input array | 2018-02-13 | ✅ Có vẻ đã hết ✔ | 2.5.3: `np.shares_memory(a, np.round(a))` là False với mảng int và float (test `test_round_copies` kiểm điều này). Chưa tìm PR sửa. |
| #11881 | numpy.round(i, decimals=d) wrong for np.int64 and d<0 | 2024-02-16 | ❌ Còn ✔ | `np.round(np.int64(2**63-1), -3)` ra `-9223372036854775808` (tràn ở đường số nguyên; không đụng tới). |
| #15896 | `round` produces a slightly incorrect result for large inputs | 2020-04-07 | ❌ Còn ✔ | `round(np.float64(5.1e73), -73)` ra `5.0000000000000004e+73`. `decimals=-73` ngoài khoảng `[-8, 7]` và `10^73` không chính xác trong double; thuật toán nhân/chia không sửa được. |
| #17100 | np.around and np.round supports complex numbers, but not np.trunc | 2020-08-20 | ❌ Còn ✔ | `np.round(complex)` chạy, `np.trunc(complex)` TypeError; thảo luận nghiêng về deprecate `round(complex)`. |
| #19464 | `np.trunc` is inconsistent with array-api | 2021-08-23 | ✅ Đã giải quyết ✔ | `np.trunc/floor/ceil/fix(int32)` ra `int32` từ 2.1.0 (2.0.2 còn float64). PR **#26766** "ENH: Support integer dtype inputs in rounding functions" (merged 2024-07-23, release notes 2.1.0). PR #19505 (cũ) đã đóng không merge. Ứng viên đề nghị đóng (cần chạy trên `main`). |
| #20514 | BUG: Rounding floats which are already equal to an integer changes the value | 2023-12-28 | ❌ Còn ✔ | `np.round(3061040371728385.0, 2)` ra `…385.5`, đổi giá trị của số vốn đã là số nguyên. Giai đoạn "round chuẩn" (guard `|x*p| ≥ 2^53` trả lại `x`) sửa được cho `decimals ∈ [-8, 7]`; issue còn nêu tới `decimals=15`. |
| #22522 | ENH: binary (arbitrary base) rounding | 2022-11-19 | 📝 Enh 📖 | Làm tròn theo số chữ số nhị phân. |
| #13100 | ENH: `np.fix` as ufunc | 2019-04-24 | 📝 Enh 📖 | `np.fix` thành ufunc (hiện là hàm Python). |
| #13375 | ENH: Add methods from the builtin float types to the numpy floating point types | 2026-01-21 | 📝 Tracking ✔ một phần | Checklist thêm method của `float` vào scalar numpy: trên 2.5.3 `np.float64` đã có `__trunc__`, `__round__`, `__floor__`, `__ceil__`, `is_integer`, `as_integer_ratio`, `hex`, `fromhex`. Chưa đọc hết các mục còn lại. |

Nhánh `perf/fused-round` (gộp ba ufunc thành một lượt SIMD, float16 bỏ cast thừa) giữ kết quả float32/float64 **giống từng bit**, nên không đổi trạng thái của các issue float32/float64 ở trên; chỉ #13699 (float16) được sửa. Nhánh `perf/fused-round-exact` (cất để làm sau) xử lý được #20514 và các lỗi làm tròn ở điểm giữa (ví dụ `np.round(2.675, 2)` ra `2.68`, Python ra `2.67`), nhưng đổi kết quả nên cần maintainer đồng ý.

## B. Làm tròn trong chuyển đổi kiểu, định dạng và in số (7)

| Issue | Tiêu đề | Cập nhật | Trạng thái | Ghi chú |
|---|---|---|---|---|
| #10645 | Inconsistent f-string output for rounded np.float32 | 2018-02-22 | ❌ Còn ✔ | `f"{round(np.float32(1337.09997559), 1)}"` ra `1337.0999755859375`, Python ra `1337.1`. |
| #25647 | BUG: Scalar `__format__` should use dragon4 printing for floats | 2024-01-21 | ❌ Còn ✔ | Cùng chủ đề #10645: `__format__` của scalar float không dùng dragon4. |
| #20687 | BUG: int to np.float32 conversion is not correctly rounded | 2022-01-06 | ❌ Còn ✔ | `np.float32(2**54 - 2**29 - 1)` ra `2**54` thay vì `2**54 - 2**30` (không làm tròn đúng). |
| #28910 | BUG: `nearest` interpolation method for quantile does not match documented behaviour | 2026-09-18 | ❌ Còn ✔ | `np.quantile(np.arange(4.0), 0.5, method="nearest")` ra `2.0`: `nearest` làm tròn half-to-even chỉ số, không khớp doc. |
| #23518 | DOC: Representing big float32 numbers as string | 2023-04-02 | 📝 Doc 📖 | Chuỗi hóa float32 lớn; chưa kiểm. |
| #30258 | BUG: `numpy.ma.testutils.assert_almost_equal` ignores documented `0.5` tolerance for | 2026-01-26 | 📖 | `ma.testutils.assert_almost_equal` dùng `np.around(...) <= 10**-decimal` nên dung sai thực tế lớn hơn doc; chưa kiểm. |
| #21903 | TST: `ftype` agnostic checks for `hex` and `fromhex` | 2022-07-02 | 📖 | Test enhancement: kiểm tra round-half-even trong `hex`/`fromhex`. |

## C. Sai số làm tròn của phép tính (không phải hàm `round`) (7)

| Issue | Tiêu đề | Cập nhật | Trạng thái | Ghi chú |
|---|---|---|---|---|
| #4260 | improve accuracy of logaddexp.reduce | 2018-10-21 | ❌ Còn (trong ISSUES_CORE.md) | `logaddexp.reduce` kém chính xác. |
| #17602 | BUG: rounding error for np.complex128 array multiplication from numpy v1.19.0 | 2023-01-16 | 📖 | Sai số làm tròn khi nhân mảng complex128 từ 1.19. |
| #27609 | BUG: np.unwrap accumulates rounding errors | 2025-09-07 | 📖 | `np.unwrap` cộng dồn sai số làm tròn. |
| #11034 | Unexpected rounding error for np.linalg.det | 2023-02-09 | 📖 | Sai số làm tròn của `np.linalg.det`. |
| #8786 | numpy.sum not stable enough sometimes (Kahan, math.fsum) | 2025-04-22 | 📖 | `np.sum` không đủ ổn định (đề xuất Kahan/`math.fsum`). |
| #10010 | Floating-point `pow` gives different results on different platforms | 2017-11-12 | 📖 | `pow` cho kết quả khác nhau giữa các nền tảng. |
| #21859 | BUG: np.log decimal precision / rounding behavior change | 2022-10-04 | ✔ Không repro | Báo `np.log(np.e)` ra `0.9999999999999999` từ 1.23.0; trên máy này (AVX2, không AVX-512) 2.5.3 ra `1.0`. Có thể phụ thuộc CPU (AVX-512). |

## D. Chia sàn và ufunc `floor`/`ceil` (4)

| Issue | Tiêu đề | Cập nhật | Trạng thái | Ghi chú |
|---|---|---|---|---|
| #32522 | BUG: incorrect floored division of negative timedeltas | 2026-09-26 | ❌ Còn ✔ | `np.timedelta64(-7, "us") // 2` ra `-3 us`, Python ra `-4 us` (làm tròn xuống). |
| #23860 | BUG: floor_divide is not normal when  calculating the input containing inf  | 2023-09-19 | 📖 ✔ | `np.floor_divide(inf, 2.0)` ra `nan`, trùng Python (`inf // 2` cũng là nan); chưa rõ issue muốn gì. |
| #32376 | BUG: Failing `test_unary_spurious_fpexception` and `test_floor_division_errors` test | 2026-08-21 | 📖 | Test `ceil`/`floor`/`floor_division` fail trên RISC-V (ngoại lệ FP giả). |
| #28365 | BUG: duplication in `ufunc.types` | 2025-12-10 | 📖 | Trùng lặp trong `ufunc.types` (liệt kê cả `ceil`/`floor`). |

## E. Chỉ nhắc gián tiếp (không tính là issue về làm tròn)

- #13105 (tắt việc dẹt mảng 0-d thành scalar) và #24897 (ngừng tự đổi scalar mảng thành scalar numpy) có dẫn #13100 vì `np.fix` bị ảnh hưởng.

## F. Khớp từ khóa nhưng không liên quan (72)

#4217, #4965, #7265, #8802, #9049, #9397, #9465, #10288, #10332, #12350, #13319, #13349, #14367, #14753, #15331, #15571, #15601, #15692, #15726, #16124, #16429, #17124, #17408, #17551, #18028, #18387, #18881, #18901, #19472, #19511, #19808, #20662, #20880, #20905, #21091, #21655, #21961, #22710, #22896, #22928, #23304, #23330, #23383, #23478, #24084, #24368, #24548, #26096, #26289, #26401, #26615, #27528, #27699, #27934, #28367, #28397, #28639, #28642, #28829, #29559, #29622, #30249, #30342, #30699, #30766, #31210, #31243, #31280, #31794, #31986, #32187, #32832

## Ghi chú về NEP

NEP 56 mô tả việc `ceil`/`floor`/`trunc` giữ dtype số nguyên, nhưng trên thực tế điều đó chỉ có từ **numpy 2.1.0** (PR #26766): bản 2.0.2 vẫn ra float64 (tôi đã cài từng bản để kiểm). Ghi chú về NEP trong nhóm Rounding của ISSUES_CORE.md đã được sửa theo.
