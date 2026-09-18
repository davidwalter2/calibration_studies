#!/usr/bin/env python3
"""Read the `run_subdiv.sh` outputs and report the accumulated MS block per leg
against the Geant4e step cap, for both sign conventions.

Exactly-additive scattering power => `dQMS` must not depend on the cap. The
(2,3)/(3,3) entries (phi, xt: the bending projection) are the control; the
(1,4)/(4,4) entries (lambda, yt) are the test.
"""
import glob
import os
import sys

import numpy as np
import uproot

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/work/submit/david_w/ZMass/calibration_studies/resolution/runs/msksign_260918/subdiv"
CAPS = ["10.0", "5.0", "2.5", "1.25"]


def legs(path):
    t = uproot.open(path)["legs"]
    a = t.arrays(["detid", "ok", "dQMS", "msmoliv", "msmolistride", "stepnms"], library="np") \
        if "msmolistride" in t.keys() else t.arrays(["detid", "ok", "dQMS", "stepnms"], library="np")
    q = np.stack([np.asarray(x, dtype=float).reshape(5, 5) for x in a["dQMS"]])
    nst = np.array([len(np.asarray(x)) for x in a["stepnms"]])
    return a["detid"], a["ok"], q, nst


def main():
    tabs = {}
    for sign in ("legacy", "fixed"):
        for L in CAPS:
            p = os.path.join(OUT, f"model_{sign}_L{L}.root")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                tabs[(sign, L)] = legs(p)
    if not tabs:
        sys.exit(f"no outputs under {OUT}")

    for sign in ("legacy", "fixed"):
        have = [L for L in CAPS if (sign, L) in tabs]
        if not have:
            continue
        ref = tabs[(sign, have[0])]
        ok = ref[1].astype(bool)
        print(f"\n=== res(1,4) = {'-S3 (legacy)' if sign == 'legacy' else '+S3 (fixed)'} ===")
        print(f"{'leg':>4} {'nsteps':>7} " + " ".join(f"{'Q44@'+L:>13}" for L in have)
              + "   " + " ".join(f"{'Q33@'+L:>13}" for L in have))
        nleg = min(len(tabs[(sign, L)][2]) for L in have)
        for i in range(nleg):
            if not ok[i]:
                continue
            q44 = [tabs[(sign, L)][2][i][4, 4] for L in have]
            q33 = [tabs[(sign, L)][2][i][3, 3] for L in have]
            print(f"{i:>4} {tabs[(sign, have[0])][3][i]:>7} "
                  + " ".join(f"{v:>13.6e}" for v in q44) + "   "
                  + " ".join(f"{v:>13.6e}" for v in q33))
        # the summary a gate needs: the spread across caps, relative
        r44, r33, r14, r23 = [], [], [], []
        for i in range(nleg):
            if not ok[i]:
                continue
            for key, acc, idx in (("44", r44, (4, 4)), ("33", r33, (3, 3)),
                                  ("14", r14, (1, 4)), ("23", r23, (2, 3))):
                v = np.array([tabs[(sign, L)][2][i][idx] for L in have])
                if np.abs(v[0]) > 0:
                    acc.append(v / v[0])
        for name, acc in (("Q(4,4)  lam-yt offset var", r44), ("Q(3,3)  phi-xt offset var", r33),
                          ("Q(1,4)  lam-yt correlation", r14), ("Q(2,3)  phi-xt correlation", r23)):
            if not acc:
                continue
            a = np.stack(acc)
            print(f"  {name:<28} ratio to cap {have[0]} mm, median over legs: "
                  + "  ".join(f"{L}mm {np.median(a[:, j]):.4f}" for j, L in enumerate(have)))


if __name__ == "__main__":
    main()
