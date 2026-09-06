#!/usr/bin/env python3
"""Does the C++ CGF block reproduce the python offline CF, on real records?

This is the gate for deleting the python implementation. It is NOT a smoke
test: the two paths are independent transcriptions of the same formula, so the
only useful question is whether they agree to round-off on the inputs that
actually occur -- including the ones that pick the awkward branch.

WHAT IS COMPARED. `cf_track_resolution.ioni_step_exponent(steps, wstd, tau)`
against `cgfshim.ioni_step_exponent(...)`, on the `ioniurbanv` records of real
tracks, group by group, exactly as `extract` pools them.

WHY A RELATIVE TOLERANCE ON THE EXPONENT IS THE WRONG TEST. S is a log-CF: it
is ~0 at tau=0 by construction (centred) and grows to O(-1) at the far end, so
a relative comparison blows up at small tau where both are legitimately tiny.
The quantity that matters downstream is exp(S) -- what `weier` integrates -- so
the figure of merit here is |exp(S_cxx) - exp(S_py)|, an ABSOLUTE error on a
quantity bounded by 1.

THE BRANCHES THAT MUST BOTH BE HIT. `deltaTermsExact` switches on |a*w| <= 2
(series) vs beyond (closed form via Si/Cin), and the regime-1 `deltaTerm` has
its own split. A pass that only exercised one branch would prove nothing about
the other, so the branch occupancy is reported and the run FAILS if either
exact-delta branch is unvisited.

usage:
  python cgf_cxx_validate.py [--file F] [--ntracks 200] [--kok-nbin N]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgfshim                                                   # noqa: E402
import prodfiles

spec = importlib.util.spec_from_file_location("cft", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cf_track_resolution.py"))
cft = importlib.util.module_from_spec(spec)
sys.modules["cft"] = cft
spec.loader.exec_module(cft)

# a single-threaded production, so one stream file per task; single_file picks
# it up without naming stream 0 and warns if there turn out to be several.
DEFAULT_FILE = prodfiles.single_file(
    "/ceph/submit/data/user/d/david_w/ZMass/cvh/"
    "resolution_trackres_mugun_ul16_260830_m0/task_0000", "globalcor_resclosure")


def branch_census(steps):
    """Which deltaTermsExact / deltaTerm branch each step's grid lands in.

    The expansion parameter is a*w with a = gs*e0*tau and w = tmax/e0, i.e.
    gs*tmax*tau -- NOT gs*e0*tau. Guarding on the wrong one is the trap the C++
    calls out explicitly, so the census uses the same quantity the code does.
    """
    return steps


def synthetic(kok):
    """Cover the regimes the production file does not contain.

    The mugun records are ALL regime 2, so a real-data pass says nothing about
    the Gaussian branch (regime 0), the 1/E^2 collision-count branch (regime 1,
    `deltaTerm`, which has its OWN series/closed-form split) or the spin-0
    exact branch (regime 3). Deleting the python path on regime-2 evidence
    alone would leave three unvalidated transcriptions in the tree.

    Records are built in the exporter's own 13-column layout and fed to BOTH
    implementations, so this compares the same entry point as the real-data
    arm rather than a lower-level kernel.
    """
    tau = np.asarray(cft.TG, dtype=np.float64)
    cases = []
    # (label, regime, a1, e1, a2, e2, a3, e0r, tmaxr, scaling, g, beta2, etot)
    # e0/tmax chosen so a*w = gs*tmax*tau straddles the |a*w| = 2 split at both
    # ends of the tau grid.
    for lbl, reg, a3, e0, tmax, b2, et in (
            ("reg1 small a*w", 1, 3.0, 5.0e-2, 1.0e-1, 0., 0.),
            ("reg1 large a*w", 1, 3.0, 5.0e-2, 5.0e+2, 0., 0.),
            ("reg2 small a*w", 2, 2.0e-2, 5.0e-2, 1.0e-1, 0.99, 2.0e4),
            ("reg2 large a*w", 2, 2.0e-2, 5.0e-2, 5.0e+2, 0.99, 2.0e4),
            ("reg3 small a*w", 3, 2.0e-2, 5.0e-2, 1.0e-1, 0.99, 2.0e4),
            ("reg3 large a*w", 3, 2.0e-2, 5.0e-2, 5.0e+2, 0.99, 2.0e4)):
        cases.append((lbl, [reg, 0., 0.3, 1.0e-3, 0.7, 2.0e-3,
                            a3, e0, tmax, 1.0, 1.0, b2, et]))
    cases.append(("reg0 gaussian",
                  [0, 4.0e-6, 0., 0., 0., 0., 0., 0., 0., 1.0, 1.0, 0., 0.]))
    cases.append(("excitations only",
                  [1, 0., 0.9, 1.0e-3, 1.3, 3.0e-3, 0., 0., 0., 1.0, 1.0, 0., 0.]))

    print("\n--- synthetic regime coverage "
          "(the production file is all regime 2) ---")
    worst = 0.0
    for lbl, row in cases:
        steps = np.array([row], dtype=np.float64)
        bad = 0.0
        for wstd in (1.0, 37.0, 500.0):
            py = cft.ioni_step_exponent(steps, wstd, tau)
            cx = cgfshim.ioni_step_exponent(steps, wstd, tau, kok_nbin=kok)
            bad = max(bad, float(np.abs(np.exp(cx) - np.exp(py)).max()))
        worst = max(worst, bad)
        print(f"  {lbl:20} max |dexp| {bad:.3e}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--ntracks", type=int, default=200)
    ap.add_argument("--kok-nbin", type=int, default=None,
                    help="Kokoulin buckets; default = the offline module's own "
                         "IONI_KOKOULIN_NBIN when IONI_KOKOULIN is on")
    args = ap.parse_args()

    import uproot
    kok = args.kok_nbin
    if kok is None:
        kok = int(getattr(cft, "IONI_KOKOULIN_NBIN", 0)) if getattr(
            cft, "IONI_KOKOULIN", 0.0) else 0
    print(f"# file      {args.file}")
    print(f"# kok_nbin  {kok}   (offline IONI_KOKOULIN="
          f"{getattr(cft, 'IONI_KOKOULIN', None)})")
    print(f"# tau grid  {len(cft.TG)} points, {cft.TG[0]} .. {cft.TG[-1]}")

    t = uproot.open(args.file)["tree"]
    keys = ["ioniurbanv", "ioniurbanidx", "globalidxv", "resinfvarv"]
    have = set(t.keys())
    keys = [k for k in keys if k in have]
    a = t.arrays(keys, entry_stop=args.ntracks, library="np")

    tau = np.asarray(cft.TG, dtype=np.float64)
    worst = 0.0
    worst_where = None
    nblk = 0
    nser = nclosed = 0
    reg_seen = {}
    errs = []

    for ic in range(len(a["ioniurbanv"])):
        uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
        uii = np.asarray(a["ioniurbanidx"][ic])
        if not len(uii):
            continue
        stride = len(uvi) // max(len(uii), 1)
        if stride not in (11, 13):
            continue
        uvi = uvi.reshape(-1, stride)
        for g in np.unique(uii):
            steps = uvi[uii == g]
            if not len(steps):
                continue
            # a wstd of order the real one; the comparison is linear in it but
            # the BRANCH depends on it, so use a physical value not 1.
            for wstd in (1.0, 37.0):
                py = cft.ioni_step_exponent(steps, wstd, tau)
                cx = cgfshim.ioni_step_exponent(steps, wstd, tau, kok_nbin=kok)
                d = np.abs(np.exp(cx) - np.exp(py)).max()
                errs.append(d)
                if d > worst:
                    worst, worst_where = d, (ic, int(g), wstd)
                nblk += 1
                for r in np.unique(steps[:, 0].astype(int)):
                    reg_seen[r] = reg_seen.get(r, 0) + int((steps[:, 0] == r).sum())
                # branch census on the exact-delta steps
                ex = steps[:, 0] >= 2
                if ex.any():
                    gs = wstd * steps[ex, 10] * 1e-3
                    aw = np.abs(gs[:, None] * steps[ex, 8][:, None]
                                * steps[ex, 9][:, None] * tau[None, :])
                    nser += int((aw <= 2.0).sum())
                    nclosed += int((aw > 2.0).sum())

    errs = np.array(errs)
    print(f"\nblocks compared      {nblk}")
    print(f"steps by regime      {dict(sorted(reg_seen.items()))}")
    print(f"exact-delta branches series={nser}  closed-form={nclosed}")
    print(f"\nmax |exp(S_cxx) - exp(S_py)|   {worst:.3e}   at {worst_where}")
    if len(errs):
        print(f"median                        {np.median(errs):.3e}")
        print(f"99th pct                      {np.percentile(errs, 99):.3e}")

    wsyn = synthetic(kok)

    ok = True
    if wsyn > 1e-12:
        print(f"\nFAIL: synthetic regimes disagree at {wsyn:.3e}")
        ok = False
    if nblk == 0:
        print("\nFAIL: no blocks compared")
        ok = False
    if reg_seen.get(2, 0) + reg_seen.get(3, 0) > 0 and not (nser and nclosed):
        print(f"\nFAIL: exact-delta branch coverage incomplete "
              f"(series={nser}, closed={nclosed}) -- the untested branch is "
              f"exactly where a transcription error would hide")
        ok = False
    # 1e-12 on a quantity bounded by 1: far below the 1e-5 divergences that
    # motivated this, and loose enough for two different summation orders.
    if worst > 1e-12:
        print(f"\nFAIL: {worst:.3e} exceeds 1e-12")
        ok = False
    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
