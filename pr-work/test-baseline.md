# Test baseline — gh-32595

Dùng để so sánh giữa các lần chạy. Nếu con số lệch đi, xem phần "Ghi chú" trước
khi kết luận là do thay đổi của mình.

## Lần chạy mới nhất (sau rebase lên main)

| | |
|---|---|
| Ngày | 2026-09-14 |
| Nhánh | perf/broadcast-buffer-fill @ de3e75be1a |
| Base | origin/main @ 3deeb20adb |
| Python | Python 3.14.4 |
| Lệnh | `spin test` |

```
= 48802 passed, 1051 skipped, 3646 deselected, 59 xfailed, 1 xpassed in 565.85s (0:09:25) =
```

**0 failed.**

## Các lần trước

| Commit | Kết quả |
|---|---|
| 04334e9f5d (trước refactor doubling) | 48802 passed, 1051 skipped, 59 xfailed, 1 xpassed |
| 0bb2bf3c2d (trước khi thêm test) | 48676 passed, 1050 skipped, 59 xfailed, 1 xpassed |

Chênh 126 test giữa hai mốc đầu là 110 test mới của PR cộng vài test bị đếm
khác do parametrize.

## File test riêng

```bash
spin test -- /mnt/c/Dev/projects/numpy-pr-work/test_pr.py -q     # 110 passed
spin test -- numpy/_core/tests/test_nditer.py numpy/_core/tests/test_ufunc.py -q
```

## Ghi chú — những thứ KHÔNG phải lỗi

- **xfailed (59)**: test đã biết là hỏng, được đánh dấu `@pytest.mark.xfail`.
  Fail đúng như dự kiến.
- **xpassed (1)**: `test_arraypad.py::TestLinearRamp::test_object_array`.
  Bug đã được sửa từ lâu nhưng nhãn xfail chưa ai gỡ. Đã kiểm chứng: xảy ra y
  hệt trên main sạch, không liên quan tới PR.
- **skipped (1051)**: thiếu thư viện tuỳ chọn, sai nền tảng, không có locale...
- **deselected (3646)**: bị lọc bởi `-m 'not slow'`, mặc định không chạy.
- 3 test `test_scripts.py::test_f2py` xfail vì môi trường dev chưa
  `pip install` nên không có script `f2py` trong PATH. Không phải lỗi code.

Chỉ cần nhìn một con số duy nhất: **failed**. Không có dòng đó nghĩa là xanh.
