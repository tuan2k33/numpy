"""compare_np_round.py base.json new.json  -> ratio new/base (min), flagging > +10% as SLOWER."""
import json
import sys

a = json.load(open(sys.argv[1]))["results"]
b = json.load(open(sys.argv[2]))["results"]
worse = 0
print(f"{'config':44s} {'base us':>11s} {'new us':>11s}  new/base")
for k in a:
    if k not in b:
        continue
    r = b[k]["min"] / a[k]["min"]
    flag = "  <-- SLOWER" if r > 1.10 else ("  faster" if r < 0.90 else "")
    worse += r > 1.10
    print(f"{k:44s} {a[k]['min']*1e6:11.2f} {b[k]['min']*1e6:11.2f}  {r:6.2f}x{flag}")
print(f"\n{worse} configs slower than +10%")
