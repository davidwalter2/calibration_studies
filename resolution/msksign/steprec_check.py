#!/usr/bin/env python3
"""Defect (b): with `doMassConstraint=True` the raw step records were cleared
on the CONSTRAINED pass, which never refills them, so `msmoliv` / `ioniurbanv`
/ `radstepv` / `ioniqscalev` came out EMPTY on every candidate.

The records are the propagation's, drained on the icons == 0 pass, so they must
be BIT-IDENTICAL between a mass-constrained run and an unconstrained one on the
same candidates: the constraint changes only the second pass.

  usage: steprec_check.py <mcons_on_dir> <mcons_off_dir>
"""
import glob
import os
import sys

import numpy as np
import uproot

REC = ["msmoliv", "msmoliidx", "ioniurbanv", "ioniurbanidx",
       "radstepv", "radstepidx", "radstepspecv", "ioniqscalev", "ioniqscaleidx"]
ALSO = ["Jpsi_mass", "chisqval", "ndof", "niter_cons0", "edmval_cons0",
        "Muplus_refParms", "Muminus_refParms"]


def one(d):
    fs = sorted(glob.glob(os.path.join(d, "*.root"))) if os.path.isdir(d) else [d]
    fs = [f for f in fs if os.path.getsize(f) > 0]
    if not fs:
        sys.exit(f"no ROOT under {d}")
    return fs[0]


def main():
    a, b = one(sys.argv[1]), one(sys.argv[2])
    ta, tb = uproot.open(a)["tree"], uproot.open(b)["tree"]
    print(f"mass constraint ON : {a}  ({ta.num_entries} candidates)")
    print(f"mass constraint OFF: {b}  ({tb.num_entries} candidates)")
    have = [n for n in REC + ALSO + ["run", "lumi", "event"] if n in ta.keys() and n in tb.keys()]
    A = ta.arrays(have, library="np")
    B = tb.arrays(have, library="np")

    def keys(D):
        out, seen = [], {}
        for r, l, e in zip(D["run"], D["lumi"], D["event"]):
            k = (int(r), int(l), int(e))
            o = seen.get(k, 0)
            seen[k] = o + 1
            out.append((*k, o))
        return out
    ka, kb = keys(A), keys(B)
    ia = {k: i for i, k in enumerate(ka)}
    pairs = [(ia[k], j) for j, k in enumerate(kb) if k in ia]
    print(f"matched {len(pairs)} candidates\n")

    print(f"{'branch':<18} {'empty ON':>10} {'empty OFF':>10} {'len差 ON!=OFF':>14} {'value diffs':>12}")
    for nm in REC:
        if nm not in A:
            print(f"{nm:<18} {'--- not in the tree ---':>50}")
            continue
        ea = sum(1 for i, _ in pairs if len(np.asarray(A[nm][i])) == 0)
        eb = sum(1 for _, j in pairs if len(np.asarray(B[nm][j])) == 0)
        nlen = 0
        nval = 0
        for i, j in pairs:
            x, y = np.asarray(A[nm][i]), np.asarray(B[nm][j])
            if x.shape != y.shape:
                nlen += 1
                continue
            if x.size and not np.array_equal(x, y):
                nval += 1
        print(f"{nm:<18} {ea:>10} {eb:>10} {nlen:>14} {nval:>12}")

    print()
    for nm in ALSO:
        if nm not in A:
            continue
        d = []
        for i, j in pairs:
            x, y = np.asarray(A[nm][i], dtype=float), np.asarray(B[nm][j], dtype=float)
            if x.shape == y.shape:
                d.append(np.atleast_1d(y - x))
        if d:
            d = np.concatenate(d)
            print(f"{nm:<20} max|OFF-ON| = {np.nanmax(np.abs(d)):.4e}  (nonzero {int(np.count_nonzero(d))}/{d.size})")

    st = [n for n in ("msmolistride", "ioniurbanstride", "radstepstride", "radstepnv")
          if n in ta.keys()]
    if st:
        vals = ta.arrays(st, library="np", entry_stop=1)
        print("\nstride branches:", {k: int(np.atleast_1d(v)[0]) for k, v in vals.items()})
        if "msmoliv" in A and "msmoliidx" in A:
            for i, _ in pairs[:3]:
                nv, ni = len(np.asarray(A["msmoliv"][i])), len(np.asarray(A["msmoliidx"][i]))
                if ni:
                    print(f"  candidate {i}: len(msmoliv)/len(msmoliidx) = {nv}/{ni} = {nv/ni:g}")
    else:
        print("\nno stride branches in this file")


if __name__ == "__main__":
    main()
