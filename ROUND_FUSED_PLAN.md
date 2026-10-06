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

## 9. Mục tiêu mới: `decimals ∈ [-8, 7]` phải chuẩn và nhanh

Thay thế/làm rõ giai đoạn B ở trên: nhắm tới kết quả **làm tròn đúng** (trùng `round(x, d)` của Python, vốn làm tròn chính xác theo giá trị nhị phân thật của `x`) cho `d ∈ [-8, 7]`. Ngoài khoảng đó giữ đường cũ (3 ufunc) để không đổi gì; ranh giới tại `d = -9, 8` sẽ có bước nhảy hành vi, phải ghi rõ trong doc. Có thể mở rộng khoảng tới ±22 sau (khi đó `10^|d|` vẫn chính xác trong double), không cần ở bước đầu.

Lý do khoảng này gọn: `p = 10^|d| ≤ 10^8` đều **biểu diễn chính xác** ở float32 và float64 (`10^8 = 2^8 · 5^8`, `5^8 = 390625` cần 19 bit) nên tra bảng 9 phần tử, hết cần vòng lặp nhân dần (và hết sai số của `power_of_ten`). float16 không biểu diễn được `p > 65504` nên phải tính trung gian ở dtype rộng hơn.

### 9.1. Vì sao hiện tại không chuẩn
`rint(x * p) / p` làm tròn tích `x*p` về double **trước** khi `rint`, nên khi giá trị thật rơi sát điểm giữa, sai số làm tròn của phép nhân đẩy sang bên sai. Đã đo (script nháp ở scratchpad, `val_round.py`, so với `round(x, d)` của Python, `d` ngẫu nhiên trong [-8, 7], float64):

| Tập mẫu (300000) | khác Python (thuật toán hiện tại) |
|---|---|
| ngẫu nhiên | 0.34% |
| đúng điểm giữa thập phân (vd 2.675, d=2) | 19.29% |
| lệch 1 ulp quanh điểm giữa | 1.51% |

### 9.2. Thuật toán đề xuất (đã kiểm chứng toán học bằng script nháp, `val_round2.py`; **không phải bản cài đặt**)
Dùng `fma` để lấy phần dư **chính xác** của phép nhân/chia, rồi hiệu chỉnh `rint` bằng phần dư đó:

```
d >= 0: p = 10^d;   s = x * p;  e = fma(x, p, -s)      # e = x*p - s, chính xác
d <  0: p = 10^-d;  s = x / p;  e = fma(-s, p, x) / p   # x/p - s,   chính xác (sai số ~1e-16)
nh = rint(s)                       # half-even
r  = (s - nh) + e                  # phần dư thật so với nh, |r| <= ~0.5
n  = nh + 1 nếu r > 0.5, hoặc r == 0.5 và nh lẻ
n  = nh - 1 nếu r < -0.5, hoặc r == -0.5 và nh lẻ
kết quả = d >= 0 ? n / p : n * p
nếu !(|s| < 2^53) thì kết quả = x   # ulp(x) >= bước thập phân: x đã là điểm gần nhất
```

Kết quả script nháp: **0 sai khác** trên 4 × 300000 mẫu (ngẫu nhiên đến 1e17, đúng điểm giữa, lệch 1 ulp quanh điểm giữa, và dải `|s| ∈ [2^52, 2^53)`). Hai lỗi tôi gặp khi thử và đã sửa trong thiết kế: (1) ngưỡng bảo vệ phải là `2^53` chứ không phải `2^52` (ở `2^52 ≤ |s| < 2^53` vẫn cần làm tròn thật; để `2^52` sai 128/300000 mẫu ngẫu nhiên); (2) với `d < 0`, phần dư `e` tính theo đơn vị của `x` nên phải chia cho `p` trước khi cộng vào `r`. Đây là kiểm chứng mẫu, **chưa phải chứng minh**; trước khi tin cần vét cạn: float16 (65536 × 16 giá trị `d`), float32 (2^32 × 16 trong C, ước tính vài phút đến vài chục phút, cần đo), float64 chỉ thử mẫu + tập đối kháng + tham chiếu chính xác bằng số học hữu tỉ (chỉ dùng trong test).

### 9.3. float32 / float16
Tính trung gian bằng **float64** rồi cast về một lần:
- `d >= 0`: tích `x * p` (24 bit × 19 bit) nằm gọn trong float64 nên chính xác, không cần `fma`; điểm giữa là điểm giữa thật, `rint` là đủ.
- `d < 0`: phép chia không chính xác nên vẫn cần phần dư (`fma`).
- Script nháp: float32 và float16 lệch 0/40000 so với tham chiếu chính xác (mẫu ngẫu nhiên, **chưa vét cạn**; double rounding `f64 → f32` ở bước chia cuối là điểm cần vét cạn bằng test, tôi chưa có chứng minh).
- float64 trung gian đồng nghĩa số lane SIMD giảm một nửa so với float32; tốc độ phải **đo**, không được giả định.

### 9.4. Chi phí và rủi ro tốc độ (chưa đo gì)
- Thêm so với `mul + rint + div`: 1 `fma`, 1 phép trừ + cộng, 2 so sánh, 2 chọn (blend). Đổi lại bỏ 2 lượt bộ nhớ và 2 lần dispatch. Phải đo.
- `fma` cần phần cứng FMA3 (x86 AVX2+FMA3, NEON/ASIMD, VSX...). Mục dispatch hiện tại của `loops_unary_fp` có `X86_V2` (SSE4.2, **không có FMA**). Hai phương án, chọn sau khi đo: (a) thêm target `AVX2`/`FMA3` và dùng đường vô hướng chính xác (hoặc tách Dekker/Veltkamp, vì `p` nhỏ nên tách rẻ) cho baseline; (b) chỉ dùng đường mới khi có FMA, còn lại giữ đường cũ (không đổi hành vi theo CPU là điều không mong muốn, vì kết quả sẽ khác nhau giữa các máy). **Nên chọn (a)**: kết quả phải giống nhau trên mọi CPU.
- `libm fma()` không có phần cứng rất chậm (mô phỏng phần mềm), tránh dùng trong đường chính.

### 9.5. Hệ quả hành vi (cần maintainer đồng ý)
- Kết quả đổi cho các giá trị gần điểm giữa; ví dụ trong docstring `np.round(16.055, 2)` (hiện `16.06`, Python `16.05`) sẽ thành `16.05`, và ví dụ `56294995342131.5` với `decimals=3` thành `…131.5`. Phải sửa docstring và thêm release note; có thể code downstream đang phụ thuộc vào kết quả cũ.
- Bù lại: hết tràn trong khoảng này (#13699), hết `nan` với float16, và `np.round` khớp `round()` của Python.

### 9.6. Kết quả kiểm chứng thuật toán (script nháp `val3.py`/`val4.py`, scratchpad; chạy CPU, Python thuần + `math.fma`)

| Kiểm tra | Kết quả |
|---|---|
| `10^k` chính xác trong float64 với k ≤ 22; trong float32 với k ≤ 10 (`10^11` thì không) | đúng |
| `np.round` thật (2.5.3) so với `round(x, d)` của Python, float64, 100000 × 16 giá trị `d` | khác 0.353% (khớp số 0.34% ước lượng trước đó, nên baseline đo trên `np.round` thật) |
| `round()` của Python so với tham chiếu hữu tỉ chính xác (`Fraction`), 50000 mẫu | khớp hoàn toàn |
| Thuật toán mục 9.2, float64, các giá trị biên (±0, ±inf, nan, cận `2^52/2^53`, subnormal nhỏ nhất, `DBL_MAX`) | khớp Python, kể cả dấu của số 0 |
| Thuật toán, float64, 400000 mẫu log-đều trên toàn dải số mũ | khớp |
| Thuật toán, **float16 vét cạn** (63488 giá trị hữu hạn × 16 giá trị `d`, 22 giây), tham chiếu chính xác | khớp hoàn toàn |
| Thuật toán, float32, 60000 mẫu (ngẫu nhiên, đúng điểm giữa, lệch 1 ulp) | khớp |

Hai lỗi nằm ở **script**, không phải thuật toán, và đã sửa trước khi có bảng trên: (1) `round()` của Python trả `int` nên mất dấu của `-0.0` (trong C, `rint(-0.0)` giữ dấu; cài đặt phải dùng `rint` thật hoặc `copysign`); (2) hàm tham chiếu chọn nhầm `65504` khi giá trị làm tròn tràn float16 (chuẩn IEEE là `inf` từ `65520`).

Chưa làm: vét cạn float32 (2^32 × 16, cần harness C với tham chiếu số nguyên 128 bit), chạy trên nhiều CPU, đo tốc độ. Các điểm phụ: máy này có `fma` và `avx2` nhưng không có `avx512`; có `gcc`.

### 9.7. Kết quả vét cạn float32, float16 và đo tốc độ proxy (CPU, đơn luồng)

Mã nháp (không nằm trong repo) lưu ở `/mnt/c/Dev/numpy-round-experiments/`: `exh32*.c` (vét cạn, tham chiếu số nguyên 64/128 bit), `bench_round.c` (benchmark), `val*.py` (kiểm chứng Python). Biên dịch: `gcc -O2 -fopenmp -mfma -mavx2 -ffp-contract=off`.

**Đơn giản hóa thuật toán (đã kiểm chứng).** `rint` chọn số chẵn khi đúng điểm giữa, nên khi `|s - nh| = 0.5` thì `nh` luôn chẵn; do đó không cần kiểm tra chẵn/lẻ và chỉ cần **dấu** của `e` (không chia `e/p`, bỏ một phép chia):
```
nh = rint(s);  diff = s - nh
nếu diff ==  0.5 và e > 0: nh += 1
nếu diff == -0.5 và e < 0: nh -= 1
```
float64: khớp `round()` của Python trên 5 × 400000 mẫu (`val5.py`).

**Vét cạn float32 (2^32 mẫu × 16 giá trị `d`, tham chiếu số nguyên chính xác, kiểm tra bằng so sánh bit gồm dấu của số 0): 0 sai khác ở cả ba biến thể:**
1. double trung gian + `fma` + hiệu chỉnh (`exh32.c`)
2. double trung gian, **không `fma`, không `e`, không guard**: `(float)(rint(x*p)/p)` (`exh32s.c`)
3. **float thuần** (`fmaf`, hiệu chỉnh theo dấu `e`, guard tại `2^24`) (`exh32n.c`)

Harness có phát hiện lỗi: một biến thể cố ý làm sai (làm tròn tích về float) báo 566231 sai khác trong một dải nhỏ.

**float16 vét cạn (63488 giá trị × 16 `d`):** 0 sai khác cho cả (a) tính trong float32 rồi cast về float16 một lần, (b) double đơn giản không `fma` (`val6.py`).

**Benchmark proxy** (C, AVX2, 1 luồng, `-O2`, mảng 24 → 8·10^6 phần tử; **không** mô hình hóa overhead gọi ufunc; chạy lặp cho thấy nhiễu khoảng ±10–20%, nên chênh lệch dưới ~15% không kết luận được). A3 = 3 lượt như hiện tại; B2 = một lượt, đường nhanh `mul → rint → sub → kiểm tra |diff|==0.5 và guard`, chỉ chạy `fma`/hiệu chỉnh khi vector có lane đúng điểm giữa hoặc lane vượt guard:

| Dữ liệu | float64 B2/A3 | float32 B2/A3 (float thuần) |
|---|---|---|
| ngẫu nhiên, mảng ≥ 1000 | 0.49–0.73× | 0.62–0.73× |
| ngẫu nhiên, 8·10^6 | ~0.41× | ~0.40× |
| ngẫu nhiên, 24 phần tử | 0.82× (d=2), 1.12× (d=-2) | 0.84× (d=2), 1.06× (d=-2) |
| **mọi lane đều là điểm giữa** (xấu nhất), mảng 1000–10^5 | 1.03–1.30× | 1.15–1.77× |
| xấu nhất, ≥ 10^6 | 0.72–0.77× | 1.23–1.45× (10^6), ~0.5× (8·10^6) |

Các biến thể nhanh hơn `exact` toàn phần (B1/Bn) vì B1/Bn luôn chạy `fma` và hiệu chỉnh: B1 (float64) 0.94–1.09× ở mảng 10^4, Bn (float32) 1.03–1.5× ở mảng nhỏ. Phương án "float32 tính trung gian bằng double" (Bf) chậm hơn A3f 1.3–1.9× khi mảng nằm trong cache, **không nên dùng cho float32** (dùng float thuần + `fmaf`).

**Số đo thật của `np.round` (2.5.3, máy này, `a ∈ [-1000, 1000]`, thời gian cả lời gọi):**

| dtype | n=24 | n=1000 | n=10^4 | n=10^6 | n=8·10^6 |
|---|---|---|---|---|---|
| float64 `round(a, 2)` | 2.46 µs | 3.35 µs | 9.0 µs | 1.31 ms | 26.6 ms |
| float32 `round(a, 2)` | 2.95 µs | 3.26 µs | 5.7 µs | 0.32 ms | 10.1 ms |
| float16 `round(a, 2)` | 5.1 µs | 50 µs | 562 µs | 53.8 ms | 436 ms |
| (một lời gọi `np.rint(a, out=)`, float64) | 0.42 µs | 0.56 µs | 1.9 µs | 0.75 ms | 7.9 ms |

Nhận xét: (1) mảng nhỏ bị overhead chi phối (~2.5–3 µs/lời gọi, mỗi ufunc ~0.4 µs), nên gộp 3 ufunc thành 1 tiết kiệm trực tiếp ~0.8–1 µs bất kể kernel; (2) float16 chậm ~50 ns/phần tử (loop vô hướng, ~50× chậm hơn float64) nên kernel gộp float16 là thắng lớn; (3) với mảng 10^4–10^6, kết quả cuối phụ thuộc kernel thật, đường nhanh B2 kỳ vọng nhanh hơn, nhưng **chưa được đo trong numpy thật**.

**Kết luận tạm thời / việc còn lại:**
- Về đúng/sai: ba biến thể float32 và float16 đã vét cạn; float64 mới kiểm bằng mẫu + tập đối kháng (5 × 400000 + 400000 log-đều), **chưa vét cạn** và không thể vét cạn.
- Về tốc độ: proxy cho thấy B2 (đường nhanh) đạt mục tiêu "không chậm" trên dữ liệu thường, nhưng **chưa đạt trong ca xấu nhất** (mọi lane là điểm giữa, mảng nằm trong cache, chậm hơn 1.2–1.8× ở phần tính toán). Cần đo trong numpy thật, bao gồm dữ liệu dạng "giá tiền có 3 chữ số thập phân" (nhiều lane là điểm giữa), trước khi hứa là không chậm. Có thể cải thiện đường chậm (dùng lại các giá trị đã tính, hoặc quyết định đường theo thống kê vector).
- float64 cần fma phần cứng; baseline x86 không có FMA chưa có phương án (xem 9.4).
