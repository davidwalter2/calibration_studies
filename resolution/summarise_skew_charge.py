#!/usr/bin/env python3
"""One table out of the charge-sign closure arms.

Pulls the TOTAL and lowest-pT rows out of every `skew_closure_*.txt` in the
output directory and lays the arms side by side, so the charge omission and
its correction can be read without opening eight files.
"""
import glob
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/public_html/cvh/260902_skew")


def grab(path, section, rowpat):
    """Rows of `section` whose first field matches rowpat."""
    txt = open(path).read().split("\n")
    out, on = [], False
    for L in txt:
        if L.startswith("## "):
            on = L.startswith(section)
            continue
        if on and re.match(rowpat, L):
            out.append(L)
    return out


def num(L, i):
    f = L.split()
    return f[i]


ARMS = [
    ("nosign_pos", "unsigned model, mu+ only   (cached sign is RIGHT here)"),
    ("nosign_neg", "unsigned model, mu- only   (cached sign is WRONG here)"),
    ("qsign_pos",  "q-signed model, mu+ only"),
    ("qsign_neg",  "q-signed model, mu- only"),
    ("qsign",      "q-signed model, both charges"),
    ("qfold",      "charge-folded zhat = q z, cached model (exact)"),
    ("fix",        "PUBLISHED arm: unsigned model, both charges"),
]

for samp, lab in (("lowpt", "mu gun pT 2-20"), ("ul16", "mu gun pT 20-60")):
    print(f"\n{'='*112}\n{lab}\n{'='*112}")
    print(f"{'arm':<52}{'n':>8}  <z e^-uz^2> data / model / d-m at u = 0.05")
    for tag, desc in ARMS:
        f = os.path.join(OUT, f"skew_closure_mugun_{samp}_{tag}.txt")
        if not os.path.exists(f):
            continue
        r = grab(f, "## 1. ", r"^TOTAL")
        if not r:
            continue
        # the d-m column can abut the model column in the older files, so
        # pull the signed floats out with a regex instead of splitting
        v = re.findall(r"[-+]\d+\.\d+", r[0])
        nn = re.search(r"^TOTAL\s+(\d+)", r[0]).group(1)
        print(f"{desc:<52}{nn:>8}  {v[0]:>10}{v[1]:>10}{v[2]:>10}")
    print(f"\n{'arm':<52}{'k_hat':>8}{'-1s':>7}{'+1s':>7}{'chi2(k=1)':>11}"
          f"{'  <z>|z|<3 data':>18}{'model':>9}{'  mode data':>13}{'+-':>8}{'model':>9}")
    for tag, desc in ARMS:
        f = os.path.join(OUT, f"skew_closure_mugun_{samp}_{tag}.txt")
        if not os.path.exists(f):
            continue
        k = [L for L in grab(f, "## 2. ", r"^pt TOTAL")]
        t4 = grab(f, "## 4. ", r"^TOTAL")
        # section 3, h = 0.3 block is the LAST TOTAL row of section 3
        m3 = grab(f, "## 3. ", r"^TOTAL")
        kf = k[0].split() if k else ["", "", "", "", "", "", ""]
        t4f = t4[0].split() if t4 else [""] * 20
        m3f = m3[-1].split() if m3 else [""] * 20
        print(f"{desc:<52}{kf[2]:>8}{kf[3]:>7}{kf[4]:>7}{kf[6]:>11}"
              f"{t4f[8]:>18}{t4f[10]:>9}{m3f[2]:>13}{m3f[3]:>8}{m3f[4]:>9}")
print("\nz > 0: momentum LOWER for q=+1, HIGHER for q=-1.   "
      "zhat = q z > 0: momentum LOWER for both.")
