"""Localize the charge-even phi-structured momentum bias in RADIUS, using
the per-module alignment gradients the fit already exports.

Why this works without a new hit-level export. The fit stores, per track,
globalidxv (which global parameters the track touched) and gradv
(d objective / d parameter). For an alignment degree of freedom of module
m, that gradient IS the weighted local residual of this track at that
module. On IDEAL geometry with unbiased hits, averaging over many tracks
must give <grad> = 0 for every module. A module with <grad> significantly
non-zero therefore has a genuine local hit-position bias -- and the
subdetector/layer it lives in is where the momentum bias is generated.

Design note: do NOT bin gradients by TRACK phi. A module sits at a fixed
phi and is only crossed by tracks of similar phi, so any per-module
quantity binned in track phi automatically inherits the module phi
granularity and manufactures high harmonics. Accumulate PER MODULE, then
ask separately whether the modules carrying the bias have phi structure.

What it reports:
  1. mean alignment gradient per subdetector and layer, with significance,
     ranked -- i.e. where the residuals are;
  2. for the worst layers, the phi profile of the per-module mean gradient
     and its harmonic content, to compare against the m = 6/8/9/10 seen in
     the momentum pull.

usage:
    python hit_residual_localize.py --files '<glob>' [--ntasks 20] [--parmtype 0]
"""

import argparse
import glob

import numpy as np
import uproot

from wums import logging

logger = logging.child_logger(__name__)

SUBDET = {0: "BPix", 1: "FPix", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--ntasks", type=int, default=20)
    ap.add_argument("--parmtype", type=int, default=0,
                    help="alignment DOF to profile (0 = local-x translation, "
                         "the precise/bending coordinate)")
    ap.add_argument("--nphi", type=int, default=72)
    a = ap.parse_args()
    logging.setup_logger(__file__, 3, False)

    fs = sorted(glob.glob(a.files))[:a.ntasks]
    logger.info(f"{len(fs)} files")

    rt = uproot.open(fs[0])["runtree"].arrays(
        ["parmtype", "subdet", "layer", "rho", "phi", "z"], library="np")
    ptype, sub, lay = rt["parmtype"], rt["subdet"], rt["layer"]
    rho, mphi = rt["rho"], rt["phi"]
    npar = len(ptype)
    sel_par = ptype == a.parmtype
    logger.info(f"{npar} global params; {sel_par.sum()} of parmtype {a.parmtype}")

    ssum = np.zeros(npar)
    ssq = np.zeros(npar)
    scnt = np.zeros(npar, dtype=np.int64)

    ntrk = 0
    for fn in fs:
        t = uproot.open(fn)["tree"]
        arr = t.arrays(["globalidxv", "gradv"], library="np")
        for gi, gv in zip(arr["globalidxv"], arr["gradv"]):
            gi = np.asarray(gi, dtype=np.int64)
            gv = np.asarray(gv, dtype=np.float64)
            m = sel_par[gi]
            if not m.any():
                continue
            idx, val = gi[m], gv[m]
            np.add.at(ssum, idx, val)
            np.add.at(ssq, idx, val * val)
            np.add.at(scnt, idx, 1)
            ntrk += 1
    logger.info(f"{ntrk} track-entries accumulated")

    ok = scnt >= 20
    mean = np.zeros(npar)
    err = np.full(npar, np.inf)
    mean[ok] = ssum[ok] / scnt[ok]
    var = ssq[ok] / scnt[ok] - mean[ok] ** 2
    err[ok] = np.sqrt(np.maximum(var, 0.) / scnt[ok])
    sig = np.zeros(npar)
    sig[ok] = np.abs(mean[ok]) / np.maximum(err[ok], 1e-300)

    print(f"\n{'subdet':>6} {'layer':>6} {'rho[cm]':>9} {'nmod':>6} "
          f"{'<grad> rms':>12} {'median sig':>11} {'frac >3sig':>11}")
    rows = []
    for s in sorted(SUBDET):
        for l in sorted(set(lay[sel_par & (sub == s)])):
            m = ok & sel_par & (sub == s) & (lay == l)
            if m.sum() < 5:
                continue
            f3 = float(np.mean(sig[m] > 3.))
            rows.append((s, l, np.median(rho[m]), int(m.sum()),
                         float(np.std(mean[m])), float(np.median(sig[m])), f3))
    for r in sorted(rows, key=lambda r: -r[6]):
        print(f"{SUBDET[r[0]]:>6} {r[1]:>6} {r[2]:9.1f} {r[3]:6d} "
              f"{r[4]:12.3e} {r[5]:11.2f} {r[6]:11.1%}")

    print("\n(frac >3sig is the fraction of modules in that layer whose mean")
    print(" alignment gradient is >3 sigma from zero. On ideal geometry with")
    print(" unbiased hits this should be ~0.3% everywhere.)")

    worst = sorted(rows, key=lambda r: -r[6])[:3]
    for s, l, _, _, _, _, _ in worst:
        m = ok & sel_par & (sub == s) & (lay == l)
        pm, vm = mphi[m], mean[m]
        if len(pm) < 12:
            continue
        e = np.std(vm) / np.sqrt(len(pm))
        hs = []
        for mm in (6, 8, 9, 10, 12):
            ca = 2 * np.mean(vm * np.cos(mm * pm))
            sa = 2 * np.mean(vm * np.sin(mm * pm))
            hs.append(f"m={mm}:{np.hypot(ca, sa)/max(e,1e-300):4.1f}s")
        print(f"  {SUBDET[s]} L{l}: module-phi harmonics of <grad>  " + "  ".join(hs))


if __name__ == "__main__":
    main()
