# Plan: fused kernel cho `np.round` (branch `perf/fused-round`)

Mục tiêu: sửa tràn số của #13699 mà **không làm chậm** `np.round`. Không viết bản Python nào; mọi thay đổi ở C và đo bằng benchmark thật.

Ràng buộc: tốc độ là điều kiện bắt buộc. Một thay đổi hành vi (kết quả khác bit) chỉ được đề xuất sau khi đã chứng minh nhanh bằng hoặc hơn.

## 0. Hiện trạng (đã đọc code)

- `np.round` / `np.around` → `_wrapfunc(a, 'round')` → `ndarray.round` ([methods.c:2753](numpy/_core/src/multiarray/methods.c#L2753)) → `PyArray_Round` ([calculation.c:563](numpy/_core/src/multiarray/calculation.c#L563)).
- Nhánh float với `decimals != 0` gọi **3 ufunc liên tiếp** qua `PyObject_Vectorcall`: `multiply(a, p, out)`, `rint(out, out)`, `true_divide(out, p, out)` (đảo vai nếu `decimals < 0`). Mỗi ufunc là một lượt đọc/ghi mảng và một lần dispatch.
- `p = power_of_ten(|decimals|)` luôn là `double` ([calculation.c:24](numpy/_core/src/multiarray/calculation.c#L24)) và được truyền như scalar Python float.
- Loop `rint` cho float32/float64 là SIMD ([loops_unary_fp.dispatch.c.src](numpy/_core/src/umath/loops_unary_fp.dispatch.c.src)). Float16: `rint` khai báo `TD('e', f='rint', astype={'e': 'f'})` ([generate_umath.py:1034](numpy/_core/code_generators/generate_umath.py#L1034)), tức cast sang float32 rồi về lại; `multiply`/`true_divide` của float16 là loop vô hướng trong `loops.c.src` (tôi chưa đọc kỹ, cần xác minh ở bước 1).
- Nguyên nhân #13699: bước nhân chạy trong dtype của `a`. float16 chỉ chứa tới 65504 nên `2.0 * 1e5` tràn; kết quả `nan`/`inf`.
- Test hiện có: `test_round`, `test_round_copies` ở [test_multiarray.py:2242](numpy/_core/tests/test_multiarray.py#L2242). Chưa có test float16 hay giá trị lớn.

**Phát hiện về benchmark:** `time_round` ở [bench_core.py:147](benchmarks/benchmarks/bench_core.py#L147) nằm trong `SmallMethodDispatch`, với `self.a = np.arange(20)`: **mảng int 20 phần tử**. Với int và `decimals>=0`, `PyArray_Round` chỉ copy, nên benchmark đó không đo đường float nào. Phải thêm benchmark mới trước khi sửa.

## 1. Đo trước khi sửa (baseline)

1. Build **stock `main`** vào thư mục build riêng. Không dùng `build/` / `build-install/` mặc định vì chúng đang chứa bản build của nhánh `refactor/ndarray-mask`; ghi đè sẽ mất. Cần kiểm tra `spin build --help` để chọn tùy chọn build-dir đúng, tôi chưa xác nhận cờ nào dùng được.
2. Build **release** (không debug), vì ghi chú trước cho thấy debug/release gây lệch ±15%.
3. Thêm class benchmark mới trong `bench_core.py` (ví dụ `Round`), param hóa:
   - dtype: `float16`, `float32`, `float64`
   - decimals: `-2`, `0`, `2`, `6`
   - kích thước: 20 (nhỏ, đo overhead dispatch), 10^4, 10^6 (đo băng thông)
   - layout: C-contiguous, strided (`a[::2]`), F-order, có `out=`
4. Chạy baseline vài lần (asv hoặc `python -m timeit` trên build riêng), lưu số liệu vào file. Chỉ tin chênh lệch lớn hơn nhiễu đo được (đo nhiễu bằng cách chạy baseline hai lần).
5. Ghi lại hành vi hiện tại làm golden: lưu mảng kết quả của `np.round` trên một tập giá trị cố định (xem bước 4) để so sánh bit-by-bit sau này.

## 2. Thiết kế (hướng 3: loop gộp)

### 2.1. Cách gọi từ `PyArray_Round`

Thêm **ufunc private** `_round_scaled` (hai đầu vào `a`, `p`; một đầu ra), khai báo trong `generate_umath.py` cạnh các ufunc private khác (`_ones_like`, `_arg`...). Lý do chọn ufunc:
- thừa hưởng sẵn xử lý `out`, overlap, buffering/casting, lỗi FP của ufunc;
- thay 3 lượt dispatch bằng 1;
- `p` truyền như operand stride 0, loop đọc một lần.

Cần quyết định (chưa kiểm): ufunc private có thể bị lộ qua `np._core.umath` hay không, và phải loại khỏi `__all__`/doc ra sao. Đối chiếu cách `_ones_like` được xử lý.

Hai chế độ:
- `decimals >= 0`: `rint(x * p) / p`
- `decimals < 0`: `rint(x / p) * p`

Hai ufunc `_round_up` / `_round_down` hoặc một ufunc và `p` mang dấu; chọn một trong hai khi viết code, cần nhất quán với `true_divide`: **không** thay `/ p` bằng `* (1/p)` vì `1/p` không chính xác và đổi kết quả.

Các nhánh còn lại giữ nguyên: complex (đệ quy), integer, `decimals == 0` (gọi `rint`), `out` sai shape.

### 2.2. Kernel

Đặt trong file dispatch mới `numpy/_core/src/umath/loops_round.dispatch.c.src` (hoặc mở rộng `loops_unary_fp.dispatch.c.src`, nhưng file đó ép `NPY_SIMD_FORCE_128` vì lý do gather/scatter, không hợp với kernel contiguous băng thông cao). Dùng universal intrinsics `npyv_mul_*`, `npyv_rint_*`, `npyv_div_*` như loop `rint` hiện có. Đăng ký trong `meson.build` cạnh mục `loops_unary_fp` (dòng 1152) với cùng danh sách ISA, và khai báo trong `loops.h.src`.

Các biến thể: `CONTIG/NCONTIG` cho src và dst như loop `rint`; xử lý đuôi bằng `npyv_load_till`/`store_till`.

### 2.3. Float16

Hiện float16 chạy 3 lượt vô hướng nên đây là chỗ lợi nhất. Hai lựa chọn, **cần chọn trước khi code**:
- **(a) Giữ bit-identical:** cast f16 → f32, nhân, làm tròn về f16 (cast ngược rồi lại), `rint`, chia, làm tròn về f16. Mô phỏng đúng kết quả cũ (double rounding và tràn như hiện nay) nhưng gộp trong một lượt.
- **(b) Sửa đúng:** tính toàn bộ trong f32 và chỉ cast về f16 một lần. Sửa tràn và double rounding, nhưng đổi kết quả. Trong thử nghiệm ngoài repo, khoảng 2–4% giá trị float16 hữu hạn cho kết quả khác (bản mới gần làm tròn thập phân chính xác hơn).

Đề xuất: làm (a) ở PR đầu để chứng minh không chậm và không đổi bit; (b) là PR sau, cần đồng thuận của maintainer.

## 3. Các giai đoạn

### Giai đoạn A: gộp loop, **bit-identical** (PR 1)
- float32/float64: `mul → rint → div` trong thanh ghi, đúng thứ tự phép toán cũ. Phải kiểm tra không có FMA contraction (không có phép cộng nên không có, nhưng cần đảm bảo bằng flag biên dịch/quan sát asm nếu nghi ngờ).
- float16: phương án (a).
- Giữ nguyên cờ FP và cảnh báo hiện tại (tràn vẫn phát RuntimeWarning như cũ).
- Chấp nhận khi: kết quả giống cũ từng bit trên golden set, và mọi ô benchmark không chậm hơn nhiễu đo; kỳ vọng nhanh hơn ở mảng lớn (1 lượt thay vì 3) và mảng nhỏ (1 dispatch thay vì 3).

### Giai đoạn B: sửa tràn (#13699) (PR 2, cần maintainer đồng ý)
Thêm guard trong chính kernel, che trong băng thông đã tiết kiệm ở giai đoạn A:
- `s = x * p`; nếu `!isfinite(s)` hoặc `|s| >= 2^mantissa` thì kết quả là `x` (đã là số nguyên ở scale đó, `rint` không đổi gì). Vector hóa bằng `abs`, `cmplt`, `select`; chi phí kỳ vọng nhỏ nhưng **phải đo**.
- float16: phương án (b).
- Xử lý `10^|d|` không biểu diễn được trong dtype: nếu `decimals > 0` thì kết quả `x`; nếu `decimals < 0` thì kết quả 0 (giữ dấu). Hiện tại `p = inf` có thể cho `0 * inf = nan`; cần chạy kiểm tra hành vi cũ rồi quyết định, tôi chưa kiểm.
- Xóa cờ FP overflow giả bằng `npy_clear_floatstatus_barrier` như loop `absolute` đã làm, để không còn RuntimeWarning với các ca đã được xử lý đúng.
- Tác động đã biết (từ thử nghiệm ngoài repo, phải xác nhận lại bằng kernel thật): với float64, khác biệt 1 ulp ở `|x| >= 2^52 / 10^d`; ví dụ trong docstring `np.round(56294995342131.5, 3)` đổi từ `…131.51` thành `…131.5`.

### Giai đoạn C (tùy chọn, tách riêng): `#3540`
Dùng ufunc sẽ kéo theo `__array_wrap__`/subclass propagation, có thể làm `round` giữ subclass nhất quán (sửa #3540). Đây là thay đổi hành vi riêng, **không gộp vào A**. Cần kiểm tra các subclass hiện có (`np.ma`, `matrix`, `memmap`) không bị vỡ.

## 4. Test

- Golden/bit-identical (giai đoạn A): so sánh với kết quả `main` trên các tập: toàn bộ float16 hữu hạn (65536 giá trị), mẫu ngẫu nhiên float32/float64 theo nhiều thang 10^k, các giá trị biên (0, -0, subnormal, ±inf, nan, `2^52`, `2^23`, `2^10`), `decimals` ∈ {-308..308, INT_MIN, INT_MAX}.
- Layout: contiguous, stride dương/âm, F-order, mảng 0-d, mảng rỗng, mảng không căn lề, overlap `out`/`a`, `out` sai dtype.
- Cờ FP: kiểm tra `np.errstate(all='raise')` cho hành vi cũ/mới ở giai đoạn tương ứng.
- Thêm test vào `test_multiarray.py` cạnh `test_round` (và test mới cho #13699 ở giai đoạn B).
- Chạy: `spin test -- numpy/_core/tests/test_multiarray.py -k round`, `test_umath.py`, và tập `numpy/_core/tests` liên quan. Lưu ý `spin test` kiểm stale chậm trên `/mnt/c`; gọi pytest trực tiếp trên bản install khi chưa đổi C.
- Mọi SIMD ISA có thể chạy trên máy: thử `NPY_DISABLE_CPU_FEATURES` để ép đường scalar/SSE/AVX2 và so sánh với nhau.

## 5. Tài liệu

- Giai đoạn B: sửa phần Notes của docstring `np.round` ([fromnumeric.py:3856](numpy/_core/fromnumeric.py#L3856)): ví dụ `56294995342131.5` và mô tả "equivalent to np.true_divide(np.rint(a * 10**decimals), 10**decimals)".
- Giai đoạn B: thêm release note vào `doc/release/upcoming_changes/` (định dạng `<PR>.change.rst`, xem các file đã có).
- Docstring `out`: nói "cast if necessary" nhưng thực tế bị `same_kind` chặn (#15438). Có thể sửa lời hoặc xử lý hành vi trong PR riêng.

## 6. Quy trình và rủi ro

- AI policy numpy: không dùng AI viết mô tả PR/comment; PR phải ghi rõ việc có dùng AI hỗ trợ. Người gửi phải tự hiểu và giải thích được mã.
- Trước khi viết code: mở hoặc comment trong #13699 để hỏi maintainer có chấp nhận thay đổi hành vi ở giai đoạn B không.
- Rủi ro: kernel thêm kích thước binary; ISA ít phổ biến (VSX, S390X, LSX) cần kiểm qua CI; hành vi `p = inf`; ufunc private lộ ra ngoài API.
- Không push `origin`. Push chỉ lên `fork`, và chỉ khi bạn bảo.

## 7. Việc đầu tiên khi bắt đầu

1. Xác minh các điểm "cần kiểm tra" nêu trên (cờ build-dir của spin, loop `HALF_multiply`, xử lý ufunc private, hành vi `p = inf`).
2. Build baseline stock `main` (release) vào thư mục riêng và lưu số đo + golden set.
3. Mới tới viết kernel giai đoạn A.

## 8. Phạm vi của `decimals` (đã kiểm trên 2.5.3 và đọc code)

- Kiểu: C `int` (32 bit) qua `PyArray_PythonPyIntFromInt` ([methods.c:2757](numpy/_core/src/multiarray/methods.c#L2757)). Nhận Python int, `np.int8/int64`, bool, mảng 0-d int; từ chối float, str, None (TypeError); `2**31` ném OverflowError. Docstring chỉ ghi `decimals : int`, không ghi khoảng.
- Phạm vi có nghĩa theo dtype (suy từ min/max/subnormal của dtype, cần xác nhận bằng test): float16 ≈ [-4, +8], float32 ≈ [-38, +45], float64 ≈ [-308, +323]. Ngoài khoảng đó kết quả toán học là `x` (decimals dương) hoặc `0` (decimals âm).
- Hành vi hiện tại ngoài khoảng là rác: float64 `decimals>=309` hoặc `<=-309` ra `nan`; `decimals=308` ra `inf`; float32 `decimals=100` ra `nan`; float16 `decimals=10` ra `nan`.
- Kernel chỉ nhận `p = 10^|d|`, không nhận `d`, nên khoảng của `d` không ảnh hưởng tốc độ kernel. Chỉ cần kẹp `d` một lần (O(1)) trong `PyArray_Round` trước khi gọi kernel.
- `power_of_ten` nhân dần `1e9 * 10 * ...` cho `n >= 9` nên cộng dồn sai số: mô phỏng lại vòng lặp đó cho thấy 240/309 giá trị `10^n` khác `float('1e%d' % n)` (bắt đầu từ n=25, lệch 1 ulp). Chỉ ảnh hưởng `decimals >= 25`; khi thay bằng bảng hằng số thì kết quả cho `decimals` lớn sẽ đổi nhẹ. Cần cân nhắc khi giữ bit-identical ở giai đoạn A (giữ nguyên hàm cũ cho đến PR 2).
