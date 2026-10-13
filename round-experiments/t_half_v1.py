import numpy as np, warnings
warnings.simplefilter("ignore")
def power_of_ten(n):                        # numpy's helper
    t = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    if n < 9: return t[n]
    r = 1e9
    while n > 9: n -= 1; r *= 10.0
    return r
h = np.arange(65536, dtype=np.uint16).view(np.float16)       # every bit pattern incl. nan/inf
x32 = h.astype(np.float32)
bad = 0; ndec = 0
for d in list(range(-12, 13)) + [30, 38, 39, 40, 100, 300, 400, -30, -38, -39, -100, -400]:
    if d == 0: continue
    neg = d < 0; p = np.float32(power_of_ten(-d if neg else d))
    with np.errstate(all="ignore"):
        r = (np.rint(x32 / p) * p) if neg else (np.rint(x32 * p) / p)
    ref = r.astype(np.float16)
    got = np.round(h, d)
    ok = (ref.view(np.uint16) == got.view(np.uint16)) | (np.isnan(ref) & np.isnan(got))
    ndec += 1
    if not ok.all():
        bad += int((~ok).sum()); i = int(np.flatnonzero(~ok)[0]); print("MISMATCH d=%d x=%r ref=%r got=%r" % (d, h[i], ref[i], got[i]))
print(f"float16 v1: {ndec} decimals x 65536 patterns, mismatches vs 'float32 math, one cast': {bad}")
# the two reported bugs
print("round(f16(2.0), 5) =", np.round(np.float16(2.0), 5), "| round([1505.1, 880.4], 2) =", np.round(np.array([1505.1, 880.4], np.float16), 2))
