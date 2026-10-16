"""
check_lists.py  -  run this right after extracting ONLY list.tar.gz (a few MB).
It shows which driver_* archive each split lives in, so you can download just what you need.

    python check_lists.py M:\\culane
"""
import collections
import os
import sys


def find_list_dir(root, depth=3):
    for dp, dns, fns in os.walk(root):
        if "train_gt.txt" in fns:
            return dp
        if dp[len(root):].count(os.sep) >= depth:
            dns[:] = []
    return None


def per_driver(path):
    c = collections.Counter()
    with open(path) as f:
        for line in f:
            t = line.split()
            if t:
                c[t[0].lstrip("/").split("/")[0]] += 1
    return c


root = sys.argv[1]
ld = find_list_dir(root)
if not ld:
    sys.exit(f"train_gt.txt not found under {root} - extract list.tar.gz there first.")
files = [("train", "train_gt.txt"), ("val", "val_gt.txt")]
td = os.path.join(ld, "test_split")
if os.path.isdir(td):
    files += [(n[:-4], os.path.join("test_split", n)) for n in sorted(os.listdir(td)) if n.endswith(".txt")]
drivers = set()
rows = []
for name, rel in files:
    p = os.path.join(ld, rel)
    if os.path.isfile(p):
        c = per_driver(p)
        rows.append((name, c))
        drivers |= set(c)
drivers = sorted(drivers)
print(f"{'split':16s}" + "".join(f"{d[:18]:>20s}" for d in drivers))
for name, c in rows:
    print(f"{name:16s}" + "".join(f"{c.get(d, 0):>20d}" for d in drivers))
