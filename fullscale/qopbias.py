#!/usr/bin/env python3
"""Per-leg CURVATURE bias against gen, side by side, for three samples.

THE HYPOTHESIS THIS TESTS.  The official UL16 DY sample was simulated with
CMSSW's DEFAULT Geant4 stepper tolerances; every private gun in this study used
`DeltaOneStep(Tracker) = 1e-5`, ~100x tighter, because the defaults inflate
residual widths by ~12 %.  A loose SIM stepper writes coherent sub-micron errors
into the SIMULATED trajectory, which is not a defect of the CVH fit at all:

  (i)  extra effective hit noise -> the CF model looks "too narrow";
  (ii) a SAGITTA error, which is a curvature bias proportional to pT.  At 45 GeV
       the sagitta over the tracker is 3.8 mm, so 0.5 um is 1.3e-4; at J/psi leg
       momenta the sagitta is ~34 mm and the same 0.5 um is 1.5e-5.

That is exactly the observed pattern -- the J/psi closes at +5e-5 while the Z is
off by -1.2e-4 -- and it would be eta-dependent through the step pattern.  The
test is whether the per-leg curvature bias against GEN shows an eta-dependent
~1e-4 on the official DY sample and NOT on a private tight-stepper gun at the
same momenta.

    kappa = q/pT ,  d kappa / kappa = pT_gen / pT_reco - 1
    charge-EVEN  A(eta) = 1/2 [<dk/k>_+ + <dk/k>_-]   field / scale like
    charge-ODD   M(eta) = 1/2 [<dk/k>_+ - <dk/k>_-]   misalignment like

usage:
  python qopbias.py --two-track <prod dir> --ntasks 200 --label "DY v2" ...
  python qopbias.py --track-cache <npz> --label "mu gun 20-60 (tight stepper)"
"""
import argparse
import glob
import os
import sys

import numpy as np

_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "resolution")
if _RES not in sys.path:
    sys.path.insert(0, _RES)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--two-track", default=None, help="production directory")
    p.add_argument("--track-cache", default=None, help="cf_trackres_* npz")
    p.add_argument("--ntasks", type=int, default=200)
    p.add_argument("--label", required=True)
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--max-dr", type=float, default=0.01,
                   help="gen-match cone on Mu*gen_dr")
    p.add_argument("--pt-range", type=float, nargs=2, default=[15.0, 1e9])
    p.add_argument("--eta-edges", type=float, nargs="+",
                   default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
    p.add_argument("--trim", type=float, default=0.05,
                   help="drop this fraction from each tail of dk/k before "
                        "averaging: the residual has Landau-like tails and the "
                        "MEAN is not a robust estimator of a 1e-4 shift")
    p.add_argument("-o", "--output", default=None)
    return p.parse_args(argv)


def trimmed_mean(x, w, f):
    if x.size == 0:
        return np.nan, np.nan, 0, np.nan
    if f > 0:
        lo, hi = np.quantile(x, [f, 1.0 - f])
        k = (x >= lo) & (x <= hi)
        x, w = x[k], w[k]
    if x.size < 10:
        return np.nan, np.nan, x.size, np.nan
    sw = w.sum()
    m = float(np.sum(w * x) / sw)
    v = float(np.sum(w * (x - m) ** 2) / sw)
    neff = sw ** 2 / np.sum(w ** 2)
    return m, float(np.sqrt(v / neff)), int(x.size), float(np.sqrt(v))


def load_two_track(args):
    import uproot
    from prodfiles import resolve

    files, info = resolve(args.two_track, max_tasks=args.ntasks,
                          logger=None)[:2] if False else (None, None)
    # `resolve` signature differs across versions; fall back to a glob
    tasks = sorted(glob.glob(os.path.join(args.two_track, "task_*")))
    files = []
    n = 0
    for t in tasks:
        if not os.path.exists(os.path.join(t, ".complete")):
            continue
        g = sorted(glob.glob(os.path.join(t, "globalcor_*.root")))
        if not g:
            continue
        files += g
        n += 1
        if n >= args.ntasks:
            break
    print(f"  {n} tasks / {len(files)} files")
    br = ["Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
          "Muplusgen_pt", "Muminusgen_pt", "Muplusgen_eta", "Muminusgen_eta",
          "Muplusgen_dr", "Muminusgen_dr", "chisqval", "ndof", "genweight"]
    cols = {b: [] for b in br}
    for i, f in enumerate(files):
        try:
            t = uproot.open(f)["tree"]
            a = t.arrays(br, library="np")
        except Exception as e:
            print(f"    skip {os.path.basename(f)}: {e}")
            continue
        for b in br:
            cols[b].append(np.asarray(a[b], dtype=np.float64))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(files)} files")
    d = {b: np.concatenate(v) for b, v in cols.items()}
    chi2n = d["chisqval"] / np.maximum(d["ndof"], 1.0)
    ok = chi2n < args.max_chi2_ndof
    w = d["genweight"]
    w = w / np.median(np.abs(w[np.isfinite(w) & (w != 0)]))
    legs = []
    for s, q in (("plus", +1.0), ("minus", -1.0)):
        m = ok & (d[f"Mu{s}gen_dr"] < args.max_dr) \
            & (d[f"Mu{s}gen_pt"] > 0) & (d[f"Mu{s}_pt"] > 0) \
            & (d[f"Mu{s}_pt"] >= args.pt_range[0]) \
            & (d[f"Mu{s}_pt"] < args.pt_range[1]) & np.isfinite(w)
        legs.append(dict(
            dkk=d[f"Mu{s}gen_pt"][m] / d[f"Mu{s}_pt"][m] - 1.0,
            eta=np.abs(d[f"Mu{s}_eta"][m]), q=np.full(int(m.sum()), q),
            pt=d[f"Mu{s}_pt"][m], w=np.clip(w[m], -100, 100)))
    return {k: np.concatenate([a[k] for a in legs]) for k in legs[0]}


def load_track_cache(args):
    d = np.load(args.track_cache)
    gp, tp = d["genpt"].astype(np.float64), d["trackpt"].astype(np.float64)
    m = (gp > 0) & (tp > 0) & (tp >= args.pt_range[0]) & (tp < args.pt_range[1])
    if "chisqval" in d.files and "ndof" in d.files:
        m &= (d["chisqval"].astype(np.float64)
              / np.maximum(d["ndof"].astype(np.float64), 1.0)) < args.max_chi2_ndof
    print(f"  {int(m.sum())} of {gp.size} tracks")
    return dict(dkk=gp[m] / tp[m] - 1.0, eta=np.abs(d["eta"].astype(np.float64)[m]),
                q=d["charge"].astype(np.float64)[m], pt=tp[m],
                w=np.ones(int(m.sum())))


def main(argv=None):
    args = parse_args(argv)
    print(f"=== {args.label}")
    dat = load_two_track(args) if args.two_track else load_track_cache(args)
    e = np.asarray(args.eta_edges)
    print(f"  trimmed at {100*args.trim:g} % per tail; "
          f"pT in [{args.pt_range[0]:g}, {args.pt_range[1]:g}]")
    print(f"\n  {'|eta|':>12s} {'n':>9s} {'<pT>':>7s} "
          f"{'A = charge-EVEN [1e-4]':>24s} {'M = charge-ODD [1e-4]':>23s} "
          f"{'RMS(dk/k) [1e-3]':>17s}")
    rows = []
    for i in range(len(e) - 1):
        k = (dat["eta"] >= e[i]) & (dat["eta"] < e[i + 1])
        out = {}
        for q, tag in ((+1, "p"), (-1, "m")):
            s = k & (dat["q"] > 0 if q > 0 else dat["q"] < 0)
            out[tag] = trimmed_mean(dat["dkk"][s], dat["w"][s], args.trim)
        (mp, ep, np_, sp_), (mm, em, nm, sm_) = out["p"], out["m"]
        A = 0.5 * (mp + mm)
        eA = 0.5 * np.hypot(ep, em)
        M = 0.5 * (mp - mm)
        eM = eA
        pt = float(np.median(dat["pt"][k])) if k.any() else np.nan
        rms = 0.5 * (sp_ + sm_)
        rows.append((e[i], e[i + 1], np_ + nm, pt, A, eA, M, eM, rms))
        print(f"  {e[i]:5.1f}-{e[i+1]:4.1f} {np_+nm:9d} {pt:7.1f} "
              f"{1e4*A:+12.3f} +- {1e4*eA:7.3f} "
              f"{1e4*M:+11.3f} +- {1e4*eM:7.3f} {1e3*rms:17.3f}")
    allq = {}
    for q, tag in ((+1, "p"), (-1, "m")):
        s = (dat["q"] > 0 if q > 0 else dat["q"] < 0)
        allq[tag] = trimmed_mean(dat["dkk"][s], dat["w"][s], args.trim)
    A = 0.5 * (allq["p"][0] + allq["m"][0])
    eA = 0.5 * np.hypot(allq["p"][1], allq["m"][1])
    M = 0.5 * (allq["p"][0] - allq["m"][0])
    print(f"  {'INCLUSIVE':>12s} {allq['p'][2]+allq['m'][2]:9d} "
          f"{float(np.median(dat['pt'])):7.1f} "
          f"{1e4*A:+12.3f} +- {1e4*eA:7.3f} {1e4*M:+11.3f} +- {1e4*eA:7.3f} "
          f"{1e3*0.5*(allq['p'][3]+allq['m'][3]):17.3f}")
    sp = np.array([r[4] for r in rows])
    print(f"\n  SPREAD of the charge-even bias across |eta|: "
          f"{1e4*(np.nanmax(sp)-np.nanmin(sp)):+.3f} e-4")
    if args.output:
        np.savez(args.output, edges=e, rows=np.array(rows, dtype=object),
                 label=args.label)
        print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
