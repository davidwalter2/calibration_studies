#!/usr/bin/env python3
"""Per-hit residual pulls, split into the BENDING (local x) and the
NON-BENDING (local y) projection, for two arms that differ by one switch.

The multiple-scattering sign defect lives in the (lambda, yt) projection, so
the non-bending pull width is where a change must show and the bending one is
the control. `hitdiag_*` needs `fillHitDiagnostics=True`.

  usage: hitpull.py <A dir-or-file> <B dir-or-file> [labelA labelB]
"""
import glob
import os
import sys

import numpy as np
import uproot


def one(d):
    fs = sorted(glob.glob(os.path.join(d, "*.root"))) if os.path.isdir(d) else [d]
    fs = [f for f in fs if os.path.getsize(f) > 0]
    if not fs:
        sys.exit(f"no ROOT under {d}")
    return fs[0]


def pulls(path):
    t = uproot.open(path)["tree"]
    need = ["hitdiag_dx", "hitdiag_dy", "hitdiag_exx", "hitdiag_eyy",
            "hitdiag_class", "run", "lumi", "event"]
    have = [n for n in need if n in t.keys()]
    if "hitdiag_dx" not in have:
        sys.exit(f"{path} has no hitdiag branches (fillHitDiagnostics was off)")
    a = t.arrays(have, library="np")
    dx = np.concatenate([np.asarray(v, dtype=float) for v in a["hitdiag_dx"]])
    dy = np.concatenate([np.asarray(v, dtype=float) for v in a["hitdiag_dy"]])
    ex = np.concatenate([np.asarray(v, dtype=float) for v in a["hitdiag_exx"]])
    ey = np.concatenate([np.asarray(v, dtype=float) for v in a["hitdiag_eyy"]])
    return dx, dy, ex, ey


def rep(name, z):
    z = z[np.isfinite(z)]
    if z.size == 0:
        return "n/a"
    q = np.percentile(z, [15.865, 50, 84.135])
    return (f"n {z.size:>8}  mean {z.mean():+.5f}  rms {z.std(ddof=1):.5f}  "
            f"median {q[1]:+.5f}  robust width {(q[2]-q[0])/2:.5f}")


def main():
    a, b = one(sys.argv[1]), one(sys.argv[2])
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    A, B = pulls(a), pulls(b)
    for lab, P in ((la, A), (lb, B)):
        dx, dy, ex, ey = P
        zx = dx / np.where(ex > 0, np.sqrt(ex), np.nan)
        zy = dy / np.where(ey > 0, np.sqrt(ey), np.nan)
        print(f"{lab}")
        print(f"  bending    (local x) pull : {rep('x', zx)}")
        print(f"  non-bending(local y) pull : {rep('y', zy)}")
        print(f"  sigma_y [um] median        : {1e4*np.nanmedian(np.sqrt(ey[ey>0])):.4f}")
        print(f"  sigma_x [um] median        : {1e4*np.nanmedian(np.sqrt(ex[ex>0])):.4f}")


if __name__ == "__main__":
    main()
