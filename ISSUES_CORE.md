# Open issues: `component: numpy._core` (numpy/numpy)

Snapshot 2026-10-01: 183 open issues (77 Bug, 57 Enhancement). Grouped by title only (not by reading each issue), so boundaries are approximate and some issues fit several groups.

| Nhóm | Số lượng | Issue tiêu biểu |
|---|---|---|
| arange / linspace (float step, overflow, endpoint) | ~9 | #630, #2457, #9059, #10332, #16159, #17155, #18881, #8909 |
| Casting, overflow, độ chính xác số học (float→int, int overflow, so sánh int64/float, mean/var) | ~18 | #1991, #6971, #4126, #8987, #9733, #8869, #8237, #13199, #4260, #21462 |
| Rounding / `__round__` | ~6 | #3540, #6248, #9791, #13699, #15438, #9068 |
| Structured dtype, record array, `np.void` | ~14 | #1076, #1737, #6743, #8969, #12207, #13683, #15470, #15849, #7552 |
| dtype nói chung (pickle, hash, endianness, scalar constructor) | ~12 | #1725, #4317, #7829, #9496, #15473, #16052, #17544, #13007, #23500 |
| Subclass, memmap, pickle, `__array_finalize__` | ~8 | #2348, #2402, #3143, #11265, #11437, #11388, #13172 |
| Shape, reshape, broadcast, indexing | ~14 | #3817, #7964, #8102, #8972, #9818, #16179, #2486, #8802, #15318, #11309, #8545 |
| Ufunc / gufunc machinery | ~18 | #7002, #8811, #8867, #8994, #9009, #11109, #11118, #11228, #12142, #15981, #17359, #30413 |
| Reduction (reduceat, argmin/argmax) | ~5 | #834, #835, #9182, #14371, #7179 |
| Sort, search, set ops | ~10 | #641, #6061, #7051, #8657, #15499, #11136, #13579, #9331, #16377 |
| File I/O (`fromfile`, `fromstring`, zip) | ~6 | #2230, #8458, #15442, #18435, #17118, #11683 |
| String, unicode, longdouble (độ chính xác, parse, format) | ~14 | #3481, #7594, #8089, #2376, #16921, #26701, #22098, #25910, #8224, #11547 |
| Platform / kiến trúc (ppc64le, SPARC, aarch64, i686, BLAS ABI) | ~10 | #12638, #15754, #15763, #25460, #25869, #19243, #13809, #26746 |
| Hiệu năng | ~12 | #619, #2657, #8480, #9311, #11232, #13229, #18512, #1858 |
| Tương quan, einsum, cross, linalg-like | ~12 | #1530, #2310, #17286, #2453, #6631, #13233, #13718, #13797, #15103 |
| C API, bộ nhớ, allocator, buffer protocol | ~10 | #6581, #21533, #3634, #13547, #13831, #8097, #6811, #27456 |
| Maintenance, docs, task | ~8 | #10296, #21457, #13654, #10801, #15567, #15726 |

## Ghi chú

- Phần lớn issue rất cũ: khoảng 25 cái migrate từ Trac; chỉ #30413 (12/2025) là gần đây.
- 8 issue ở "54 - Needs decision": #3817, #6846, #6854, #6971, #8089, #8161, #8657, #9496.
- Label `sustain-2026` = issue được NumFOCUS giữ lại cho *Sustaining Open Source Series 2026* (có chủ ý để dành cho sự kiện đó, không phải maintainer muốn dọn): #10332, #13547, #17118, #17175, #21676, #25910. Nên tránh tự nhận các issue này.
- Nhóm arange/linspace và casting/overflow có nhiều issue liên quan nhau nhất, có thể fix chung.
- Nhóm platform khó test vì cần phần cứng riêng.
