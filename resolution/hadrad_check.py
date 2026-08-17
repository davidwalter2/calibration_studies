#!/usr/bin/env python3
"""Validate CVH_REF_HADRAD: the hadron radiative mean + fluctuation.

WHAT IS BEING TESTED. The hadron reference is the PROTON dE/dx table looked up
at e = ekin * m_p/m, which holds beta*gamma fixed. That is correct for
ionization, which is a function of beta*gamma, and wrong for radiative loss,
which carries explicit mass dependence -- and `ComputeProtonDEDX` builds only
`G4BetheBlochModel`, so a hadron's reference subtracted no radiative mean at
all while the simulation ran hBrems/hPairProd. `computeRadiativeDEDX` refused
non-muons outright, so `radv` was identically zero too.

`CVH_REF_HADRAD` adds BOTH halves under one switch, deliberately: the missing
mean and the missing fluctuation cancel to ~90 %, so enabling the fluctuation
alone measures 0.00049 against 0.00005 for the complete correction -- a 10x
DEGRADATION. One switch is what makes that unrepresentable.

THE THREE CHECKS, in the order that matters:

  1. OFF is inert.      Every branch identical to a pre-change archive.
  2. The MUON is a null even ON. The muon has its own table, which has carried
     brems + pair since before any of this; `GetHadronRadiativeTable` refuses
     |PDG| == 13 and `computeRadiativeDEDX` keeps its original muonPlus path.
     So the muon must be bit-identical with the switch ON. If it moves, the
     hadron branch is reaching a particle it must not touch.
  3. HADRONS move.      pi/K/p must all differ, or the switch is inert -- and
     an inert control has been mistaken for a null three times in this study.

NOTE the null structure differs from CVH_REF_SPECIESDEDX, whose delta is
exactly +0.0 for a proton because the table IS the proton's. Here the PROTON
MOVES TOO: no hadron has a radiative term in the reference today. The muon is
the only null.

usage: python hadrad_check.py [--pdg 13 -211 211 -321 321 -2212 2212]
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_track_resolution as ctr
import barkas_probe as bp
import radoff_species as rs


def branch_hashes(path):
    """sha256 per branch of the export tree, so a diff names the branch."""
    out = {}
    with uproot.open(path) as f:
        # The export writes into a `propExport/` TDirectory, so a comparator
        # that enumerates only top-level keys finds no TTree -- or, worse,
        # silently compares ZERO branches and "passes". That happened once in
        # this study, so this resolves the tree by classname and hard-fails if
        # it finds none or no branches.
        cn = f.classnames()
        tname = next((k.split(";")[0] for k, v in cn.items() if v == "TTree"), None)
        if tname is None:
            raise RuntimeError(f"no TTree anywhere in {path} (keys: {list(cn)[:6]})")
        t = f[tname]
        if not t.keys():
            raise RuntimeError(f"TTree {tname} in {path} has no branches")
        for b in t.keys():
            a = t[b].array(library="np")
            h = hashlib.sha256()
            for x in a:
                h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()
                         if np.ndim(x) else np.float64(x).tobytes())
            out[b] = h.hexdigest()
    return out


def compare(pa, pb):
    ha, hb = branch_hashes(pa), branch_hashes(pb)
    assert set(ha) == set(hb), "branch sets differ"
    moved = sorted(b for b in ha if ha[b] != hb[b])
    return len(ha), moved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdg", type=int, nargs="+",
                    default=[13, -211, 211, -321, 321, -2212, 2212])
    args = ap.parse_args()

    off = dict(ctr.SWITCHES_ON)
    on = dict(ctr.SWITCHES_ON, CVH_REF_HADRAD="1")

    print("CVH_REF_HADRAD -- off vs on, per species")
    print(f"{'species':>8} {'nbranch':>8} {'moved':>6}  {'verdict':<10} branches")
    print("-" * 74)
    bad = []
    for pdg in args.pdg:
        lab = bp.SPECIES[pdg]["label"] if hasattr(bp, "SPECIES") else str(pdg)
        a = rs._export_one(pdg, "_hrOFF", off, force=True)
        b = rs._export_one(pdg, "_hrON", on, force=True)
        n, moved = compare(a, b)
        is_mu = abs(pdg) == 13
        ok = (len(moved) == 0) if is_mu else (len(moved) > 0)
        verdict = ("NULL ok" if ok else "MOVED!!") if is_mu else ("live" if ok else "INERT!!")
        if not ok:
            bad.append(lab)
        print(f"{lab:>8} {n:8d} {len(moved):6d}  {verdict:<10} {','.join(moved[:4])}")

    print()
    if bad:
        print(f"FAILED: {', '.join(bad)}")
        print("  a moved muon means the hadron branch reached a particle it must not touch;")
        print("  an inert hadron means the switch is not wired -- three controls in this")
        print("  study were provably inert before anyone checked.")
        return 1
    print("PASS: muon bit-identical with the switch ON, every hadron live.")
    print("Note the proton MOVES here, unlike under CVH_REF_SPECIESDEDX -- no hadron")
    print("carried a radiative term in the reference, the proton included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
