# MaskedDType — quyết định ngữ nghĩa

Chốt: **theo trường phái R.** NA có kiểu, mặc định lan truyền, muốn bỏ qua thì
phải nói ra.

## Luật lõi

| Vấn đề | Quyết định | Theo ai |
|---|---|---|
| NA có kiểu | `Masked[i2]`, `Masked[f8]`... — không có singleton dùng chung | R (`NA_integer_`, `NA_real_`) |
| Elementwise | mask kết quả = hợp của mask các toán hạng | tất cả |
| Rút gọn, mặc định | **lan truyền** → NA | R (`sum(c(3,NA))` = NA) |
| Rút gọn, chọn bỏ qua | `skipna=True` | R `na.rm=TRUE` |
| Rỗng / toàn NA sau khi bỏ qua | trả identity (`sum`→0), có cờ `min_count` để đổi thành NA | pandas |
| Logic | Kleene ba trị: `NA & False = False`, `NA | True = True` | tất cả |
| So sánh | `NA == x` → NA, **không phải** False | R, SQL, pandas nullable |
| NA vs NaN | **giữ riêng biệt** — NA là thiếu dữ liệu, NaN là kết quả vô định | R (pandas hối hận vì gộp) |
| Ép sang Python | `bool()`, `int()`, `float()` đều **raise**; muốn ra thì `.filled(x)` | pandas `pd.NA` |
| Dữ liệu dưới mask | không quan sát được, ghi giá trị chuẩn hoá | không ai — rút từ bug của numpy.ma |

## Lý do chọn "lan truyền" làm mặc định

Toàn bộ bệnh của `numpy.ma` là **im lặng trả ra một con số thiếu dữ liệu**.
Mặc định lan truyền biến lỗi im lặng thành lỗi nhìn thấy được. Tiện lợi kiểu
pandas đổi lấy đúng cái rủi ro mình đang cố diệt.

## Bố cục

Xen kẽ từng phần tử `{value, valid}` — mảng numpy chỉ có một buffer.
Hệ quả tốt: phần `value` là một mảng strided (`stride = sizeof(value)+1`), mà
inner loop của numpy vốn nhận strides, nên **mượn được loop của dtype gốc mà
không cần copy**.

## Kéo từ đâu

| Nguồn | Dùng để |
|---|---|
| `numpy/_core/src/umath/_scaled_float_dtype.c` (1232 dòng) | template parametric DType hoàn chỉnh |
| `PyUFunc_AddWrappingLoop` (public từ numpy 2.0) | mượn inner loop của dtype gốc |
| `numpy/_core/src/multiarray/stringdtype/` (2631 dòng) | dtype production có NA (`na_object`) |
| `numpy/ma/tests/` (10535 dòng) | bộ test đối chiếu ngữ nghĩa |
| `numpy/ma/core.py` (9111 dòng) | đọc để biết câu hỏi nào cần trả lời, không copy code |
| NEP 25 / NEP 26 | tài liệu thiết kế và biên bản tranh luận |
| `numpy-user-dtypes` | khung build ngoài core, nơi StringDType khởi đầu |

## Phạm vi bước đầu

`Masked[float64]` và `Masked[int64]`; `add/subtract/multiply`; `sum/mean` có
`skipna`. Làm **ngoài core**, đúng khuyến nghị của NEP 26 và đúng đường
StringDType đã đi.

---

# Đọc `_scaled_float_dtype.c` — bản đồ để copy

`numpy/_core/src/umath/_scaled_float_dtype.c`, 1232 dòng, parametric DType đầy
đủ nhất có sẵn trong cây numpy.

| Khối | Dòng | Nội dung | Dùng cho `Masked[T]` |
|---|---|---|---|
| 1 | 32 | struct descr mở rộng `PyArray_Descr`, chứa tham số | đổi `scaling` → `PyArray_Descr *wrapped` |
| 2 | 130 | `NPY_DType_Slots`: discover / is_known_scalar / default_descr / common_dtype / common_instance / setitem / getitem | copy nguyên khung |
| 3 | 243 | `PyArray_DTypeMeta`, `flags = NPY_DT_PARAMETRIC` | copy |
| 4 | 263-522 | 4 cast + `resolve_descriptors` + `init_casts` | rút gọn còn `T ↔ Masked[T]` |
| 5 | 522-654 | ufunc loop qua `PyArrayMethod_Spec` + `PyUFunc_AddLoopsFromSpecs` | copy cách đăng ký |
| 6 | 654-760 | `PyUFunc_AddWrappingLoop` cho `hypot` | **KHÔNG dùng được**, xem dưới |
| 7 | 764-780 | promoter cho `sfloat × double` | cần cho `Masked[T] op T` |
| 8 | 782-1040 | sort/argsort/partition, **mượn method của `DoubleDType`** | **mẫu quan trọng nhất** |

## Đính chính: `AddWrappingLoop` không dùng được

`wrapping_array_method.c:130` chuyển thẳng `data` và `strides` cho loop gốc, chỉ
dịch descriptor. Nó chỉ đúng khi **bố cục bộ nhớ y hệt** — `sfloat` vẫn là một
`double` trong bộ nhớ. `Masked[f8]` xen kẽ `{value, valid}` nên layout khác.

## Cách đúng: tự viết loop, mượn loop của `T`

Mẫu ở dòng 927 và 990: `NPY_DT_SLOTS(&PyArray_DoubleDType)->part_meth` — lấy
method của dtype khác rồi tự gọi. Kết hợp với mẹo strides:

```c
masked_binary_loop(context, data, dims, strides, auxdata)
{
    /* 1. tro thang vao truong value, GIU NGUYEN stride xen ke -> khong copy */
    char *vals[3] = {data[0]+VOFF, data[1]+VOFF, data[2]+VOFF};
    inner_loop_cua_T(ctx_T, vals, dims, strides, aux_T);

    /* 2. hop mask */
    for (i = 0; i < n; i++)
        out[i*s2 + MOFF] = in0[i*s0 + MOFF] | in1[i*s1 + MOFF];
}
```

Một hàm này đăng ký cho toàn bộ ufunc nhị phân — thay cho 50 wrapper viết tay
của `numpy.ma`.

## Ước lượng bản chạy được đầu tiên

| Việc | Dòng C |
|---|---|
| struct descr giữ `wrapped` | ~20 |
| bảng slot | ~100 |
| DTypeMeta + `tp_new` | ~80 |
| cast `T ↔ Masked[T]` | ~150 |
| loop chung + đăng ký vòng | ~250 |
| promotion | ~60 |

**~600-700 dòng** cho `add/subtract/multiply` trên `Masked[f8]` và `Masked[i8]`.
Reduction có `skipna` làm sau.

## Bước một

Copy `_scaled_float_dtype.c` sang repo mới, đổi `scaling` thành `wrapped`, xoá
hết cast và sort, chạy cho ra được:

```python
np.array([1., 2., 3.], dtype=Masked(np.float64))
```

In được ra màn hình là xong phần khó nhất của việc khởi động.


---

# Cập nhật sau khi dựng prototype

Prototype ở `/mnt/c/Dev/projects/nulldtype` (103 test xanh) đã chứng minh phần
ngữ nghĩa chạy được: NA có kiểu, lan truyền mặc định, Kleene, `where=` khiến ô
thiếu không bị tính, byte dưới NA luôn 0.

**Nhưng phép đo hiệu năng lật lại một quyết định thiết kế.**

Layout xen kẽ `{value, valid}` làm trường giá trị bị stride `itemsize + 1`. Đo
numpy thuần, 100k phần tử `f8`:

| n | liền kề | stride 9 | hình phạt |
|---|---|---|---|
| 8.192 (trong cache) | 2.0 µs | 21 µs | **10.7x** |
| 100.000 | 21 µs | 139 µs | 6.5x |
| 5.000.000 (ngoài cache) | 5.7 ms | 10.0 ms | 1.8x |

Nguyên nhân: mất SIMD và lệch căn chỉnh. Hình phạt lớn nhất ở mảng vừa nằm trong
cache — tức đúng vùng người ta quan tâm tới tốc độ. Ở mảng khổng lồ nó giảm còn
1.8x chỉ vì cả hai đều bị chặn bởi băng thông DRAM, mà lúc đó layout xen kẽ lại
phải chuyển nhiều hơn 12.5% số byte. Đây không phải lỗi cài đặt mà là hệ quả
của ràng buộc **một buffer** của mảng numpy. Arrow và pandas tránh được vì dùng
hai buffer rời.

Lối thoát duy nhất trong khuôn khổ dtype: **NA dạng mẫu bit** — mượn một giá trị
của `T` làm dấu hiệu (NaN payload cho float, `INT_MIN` cho int). Itemsize không
đổi, mảng liền kề, SIMD nguyên vẹn. Giá phải trả là mất một giá trị hợp lệ và
với float thì NA khó tách khỏi NaN.

Đúng là hai phe của NEP 12 và NEP 24. Giờ có số đo cho phía mẫu bit.
