"""v1 (pure fusion): f32/f64 must be BIT-IDENTICAL to the old three-pass np.round (any decimals, layouts, specials).
usage: python t_identical.py dump out.npz   (run with each build) ; python t_identical.py cmp base.npz new.npz"""
import sys, warnings, hashlib
import numpy as np
warnings.simplefilter("ignore")
def cases():
    rng = np.random.default_rng(3)
    base = np.concatenate([
        rng.random(5000) * 2000 - 1000, rng.standard_normal(3000) * 10.0**rng.integers(-12, 300, 3000),
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324, 1.7976931348623157e308, 2.5, 3.5, -2.5, 2.675, 1.005, 0.125, 2.0**52, 2.0**53, 4503599627370497.5]),
        (rng.integers(-10**6, 10**6, 3000) + 0.5) / 100.0])
    return base
x64 = cases()
out = {}
decs = [1, 2, 3, 5, 7, 8, 9, 10, 15, 22, 23, 30, 38, 39, 100, 308, 309, 400, -1, -2, -3, -8, -9, -20, -308, -309, -400, 2**31 - 1, -2**31, -2**31 + 1]
for dt in (np.float32, np.float64):
    x = x64.astype(dt)
    for d in decs:
        r = np.round(x, d); out[f"{dt.__name__}|{d}|c"] = r.view(np.uint32 if dt is np.float32 else np.uint64)
        out[f"{dt.__name__}|{d}|rev"] = np.round(x[::-1], d).view(r.dtype.str.replace('f', 'u'))
        out[f"{dt.__name__}|{d}|s3"] = np.round(x[::3], d).view(r.dtype.str.replace('f', 'u'))
        out[f"{dt.__name__}|{d}|2dF"] = np.round(np.asfortranarray(x[:10000].reshape(100, 100)), d).view(r.dtype.str.replace('f', 'u'))
        y = x.copy(); np.round(y, d, out=y); out[f"{dt.__name__}|{d}|inpl"] = y.view(r.dtype.str.replace('f', 'u'))
if sys.argv[1] == "dump":
    np.savez(sys.argv[2], **out); print("dumped", len(out))
