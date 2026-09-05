#!/usr/bin/env python3
"""Terse tables for the 2026-09-05 step-damping entry: the mass fits of the
three vintages side by side."""
import glob
import os
import sys

import numpy as np

RUNS = "/work/submit/david_w/ZMass/calibration_studies/resolution/runs"
VINT = [("floor 2.0  _260903x", "260903x"),
        ("floor 0.25 _260904f", "260904f"),
        ("damping    _260905d", "260905d")]
SEL = [("UNCUT, single r", "r"), ("UNCUT, families", "fam_krad1"),
       ("chi2<3, single r", "r_q3"), ("chi2<3, families", "fam_q3"),
       ("ptmin>2, single r", "r_pt2"), ("ptmin>2, families", "fam_pt2")]


def load(tag, sfx):
    for pat in (f"{RUNS}/masslikfit_jpsigun_ul16_{tag}_m0_{sfx}.npz",
                f"{RUNS}/masslikfit_jpsigun_{tag}_{sfx}.npz",
                f"{RUNS}/masslikfit_gun_{tag}_{sfx}.npz"):
        if os.path.exists(pat):
            return np.load(pat, allow_pickle=True)
    return None


def val(d, name):
    if d is None:
        return None
    import json
    meta = json.loads(str(d["meta"]))
    names = meta.get("parnames", [])
    if name not in names:
        return None
    i = names.index(name)
    return float(d["fit_x"][i]), float(d["fit_err"][i])


print("== gun mass fits, alpha [1e-3] ==")
hdr = f"{'selection':<20}" + "".join(f"{n:>26}" for n, _ in VINT)
print(hdr)
for lab, sfx in SEL:
    row = f"{lab:<20}"
    for _, tag in VINT:
        d = load(tag, sfx)
        v = val(d, "alpha[1e-3]") if d is not None else None
        row += f"{'-':>26}" if v is None else f"{v[0]:>+14.4f} +- {v[1]:.4f}"
    print(row)

print("\n== families shape parameters (UNCUT) ==")
for p in ("k_hit", "k_ms", "k_ioni"):
    row = f"{p:<20}"
    for _, tag in VINT:
        d = load(tag, "fam_krad1")
        v = val(d, p) if d is not None else None
        row += f"{'-':>26}" if v is None else f"{v[0]:>+14.4f} +- {v[1]:.4f}"
    print(row)

print("\n== files found ==")
for _, tag in VINT:
    for lab, sfx in SEL:
        d = load(tag, sfx)
        if d is not None:
            print(f"   {tag} {sfx}")
