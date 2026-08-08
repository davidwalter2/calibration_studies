#!/usr/bin/env python3
"""J/psi MASS closure of the two-track CVH fit against gen truth.

This is the end-to-end test of what the calibration actually delivers. Every
other closure in this workstream is a SINGLE-TRACK residual against gen; the
deliverable is a mass LINESHAPE, and mass is not a simple convolution of two
single-track residuals -- the two muons' errors are correlated through the
shared vertex and the mass constraint. Nothing else probes that.

Sample: J/psi -> mu mu flat-pT gun (step1_gensim.py is already this gun;
the muon samples were made by overriding it). Refit with useIdealGeometry=True,
useDefaultField=True and the 15_0-native GT, so misalignment is absent and the
field matches the simulation.

Mass variants in the tree, which test different things:
  Jpsi_mass        full two-track fit
  Jpsikin_mass     kinematic (no vertex/mass constraint applied to the mass)
  Jpsitrk_mass     from the track parameters alone
  Jpsicons_mass    with the mass constraint imposed
  Jpsigen_mass     GEN dimuon mass -- closure is against THIS, not an assumed
                   PDG value, so a generator-level offset cannot leak in

Estimator: MEDIAN, with a bootstrap error. Today's single-track work showed the
mean and median of a momentum residual can differ in SIGN because the tail's
lever grows with momentum; the same applies here and the mean is quoted only to
expose the tail's pull, never as the answer.

usage:
  python jpsi_mass_closure.py [--nfiles 0] [--nboot 300]
"""
import argparse
import glob

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
VARIANTS = ["Jpsi_mass", "Jpsikin_mass", "Jpsitrk_mass", "Jpsicons_mass",
            "Jpsikincons_mass"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="jpsigun_ul16")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--nboot", type=int, default=300)
    return p.parse_args()


def med_err(x, nboot, rng):
    if len(x) < 100:
        return np.nan, np.nan
    m = float(np.median(x))
    bs = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def main():
    args = parse_args()
    rng = np.random.default_rng(20260808)
    fs = sorted(glob.glob(f"{CEPH}/resolution_trackres_{args.tag}/task_*/globalcor_0.root"))
    if args.nfiles:
        fs = fs[:args.nfiles]
    if not fs:
        raise SystemExit(f"no files under resolution_trackres_{args.tag}")

    want = None
    cols = {}
    for fn in fs:
        try:
            t = uproot.open(fn)["tree"]
        except Exception:
            continue
        if want is None:
            keys = set(t.keys())
            want = [v for v in VARIANTS if v in keys] + \
                   [k for k in ("Jpsigen_mass", "Jpsi_sigmamass", "Jpsi_pt", "Jpsigen_pt")
                    if k in keys]
            cols = {k: [] for k in want}
        try:
            a = t.arrays(want, library="np")
        except Exception:
            continue
        for k in want:
            cols[k].append(np.asarray(a[k], dtype=np.float64))
    d = {k: np.concatenate(v) for k, v in cols.items() if v}
    if "Jpsigen_mass" not in d:
        raise SystemExit("no Jpsigen_mass branch -- cannot close against gen")

    mg = d["Jpsigen_mass"]
    ok = np.isfinite(mg) & (mg > 2.5) & (mg < 3.7)
    print(f"files={len(fs)}  candidates={int(ok.sum())}")
    print(f"gen mass: median={np.median(mg[ok]):.6f} GeV  "
          f"rms={np.std(mg[ok])*1e3:.3f} MeV  "
          f"(J/psi pole 3.096900, width 92.6 keV)\n")

    print(f"  {'variant':18} {'n':>7}  {'median dm/m':>22}  {'dm [MeV]':>18}  "
          f"{'sigma_m/m':>10}  {'mean dm/m':>11}")
    for v in VARIANTS:
        if v not in d:
            continue
        mr = d[v]
        g = ok & np.isfinite(mr) & (mr > 2.0) & (mr < 4.5)
        if g.sum() < 100:
            print(f"  {v:18} (too few)")
            continue
        rel = (mr[g] - mg[g]) / mg[g]
        m, e = med_err(rel, args.nboot, rng)
        lo, hi = np.percentile(rel, [15.865, 84.135])
        rob = 0.5 * (hi - lo)
        print(f"  {v:18} {int(g.sum()):7d}  {m*1e4:+9.4f} +- {e*1e4:.4f} x1e-4  "
              f"{m*np.median(mg[g])*1e3:+9.4f} MeV  {rob*100:9.3f}%  "
              f"{np.mean(rel)*1e4:+10.3f}")

    # pT dependence of the primary variant: a field-like offset is flat, an
    # energy-loss-like one falls as 1/p
    v = "Jpsi_mass" if "Jpsi_mass" in d else VARIANTS[0]
    ptk = "Jpsigen_pt" if "Jpsigen_pt" in d else ("Jpsi_pt" if "Jpsi_pt" in d else None)
    if ptk:
        pt = d[ptk]
        mr = d[v]
        g = ok & np.isfinite(mr) & (mr > 2.0) & (mr < 4.5) & np.isfinite(pt) & (pt > 0)
        rel = (mr[g] - mg[g]) / mg[g]
        p = pt[g]
        edges = np.quantile(p, np.linspace(0, 1, 9))
        print(f"\n  {v} vs {ptk}:")
        print(f"  {'<pT>':>8} {'n':>7}  {'median dm/m x1e-4':>22}")
        for i in range(len(edges) - 1):
            s = (p >= edges[i]) & (p < edges[i + 1])
            if s.sum() < 500:
                continue
            m, e = med_err(rel[s], 80, rng)
            print(f"  {np.median(p[s]):8.2f} {int(s.sum()):7d}  {m*1e4:+10.3f} +- {e*1e4:.3f}")


if __name__ == "__main__":
    main()
