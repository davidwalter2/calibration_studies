#!/usr/bin/env python3
"""The two BEAM-LINE residuals against GEN TRUTH: what the classes look like,
and what a cut on them buys.

The classification is `genbkg.classify` -- the same truth-based classes the
vertex study uses (signal / dup / otherdecay / nonres / unmatched), read off
the gen provenance the maker exports per leg.  Nothing here is refitted.

What it answers
  * the beam pulls, per class: mean, variance, and the tails
  * back-to-back COSMIC-LIKE pairs (|dphi| ~ pi AND eta+ ~ -eta-), which is
    the topology the beam line should reject and the DCA should not
  * the rejection a cut at |z_bs| < 3 / 4 / 5 achieves against the signal
    loss, alone and combined with the vertex-residual cut
  * the beam rows' own chi2 (`Jpsi_bschi2fit`, 3 rows) as a selection variable

usage:
  python3 bkg_bs.py --bsx <dy_bsx.npz> --bsy <dy_bsy.npz> --tag dy_bs
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import genbkg as GB        # noqa: E402
import vtxterm as VT       # noqa: E402


def binom(k, n):
    if n <= 0:
        return float("nan"), float("nan")
    p = k/n
    return p, float(np.sqrt(max(p*(1-p), 0.)/n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsx", required=True)
    ap.add_argument("--bsy", required=True)
    ap.add_argument("--tag", default="dy_bs")
    ap.add_argument("--max-chi2-ndof", type=float, default=0.0)
    # THE BASELINE SELECTION of the gen-background study (`STATE_bkg.md`):
    # |z_v| < 5 plus a minimum of 8 VALID hits on the WEAKER leg
    # (`minLegHits`, which the maker applies pre-fit on `nvalid`), giving
    # signal efficiency 0.9961 at background rejection 0.811.  Applied here so
    # the beam pulls are quoted as what they ADD on top of it, not instead.
    ap.add_argument("--min-leg-hits", type=int, default=8)
    ap.add_argument("--max-abs-zv", type=float, default=5.0)
    ap.add_argument("--no-baseline", action="store_true",
                    help="quote the tables on the unselected sample instead")
    a = ap.parse_args()

    dx = VT.load(a.bsx, max_chi2_ndof=(a.max_chi2_ndof or 0.0))
    dy = VT.load(a.bsy, max_chi2_ndof=(a.max_chi2_ndof or 0.0))
    n = dx["n"]
    if dy["n"] != n:
        sys.exit("the two beam npz do not hold the same candidates")
    for k in ("run", "lumi", "event"):
        if not np.array_equal(dx[k], dy[k]):
            sys.exit(f"the two beam npz differ in {k}")
    zx, zy = dx["z"], dy["z"]
    zv = np.asarray(dx["vtxz"], float)
    zmax = np.maximum(np.abs(zx), np.abs(zy))
    chi2 = np.asarray(dx.get("bschi2", np.full(n, np.nan)), float)
    chi2fit = np.asarray(dx.get("bschi2fit", np.full(n, np.nan)), float)

    cls, names, aux = GB.classify(dx)
    print(f"# {a.tag}: {n} candidates")

    # NON-FINITE EXPORTS EXIST and `Jpsi_vtxok` does not catch them: a leg of
    # <= 3-4 valid hits can give an infinite `Jpsi_sigmamass` and an absurd
    # sigma_v while every flag is true (`STATE.md` section 13).  A finiteness
    # check is therefore part of the baseline, and it is a cut on the FIT'S
    # OWN covariance, not on the residual.
    finite = np.isfinite(zx) & np.isfinite(zy) & np.isfinite(zv)
    if "sigmamass" in dx:
        finite &= np.isfinite(np.asarray(dx["sigmamass"], float))
    base = finite
    if not a.no_baseline:
        if "nvalid_plus" in dx and a.min_leg_hits > 0:
            nl = np.minimum(np.asarray(dx["nvalid_plus"], np.int64),
                            np.asarray(dx["nvalid_minus"], np.int64))
            base = base & (nl >= a.min_leg_hits)
        if a.max_abs_zv > 0:
            base = base & (np.abs(zv) < a.max_abs_zv)
    print(f"# baseline (finite, minLegHits>={a.min_leg_hits}, "
          f"|z_v|<{a.max_abs_zv:g}): {int(base.sum())} of {n} "
          f"({100*base.mean():.3f} %)"
          + ("  [DISABLED]" if a.no_baseline else ""))
    print(f"# classes BEFORE the baseline: " + ", ".join(
        f"{c} {int((cls == i).sum())}" for i, c in enumerate(names)))
    zx, zy, zv, zmax = zx[base], zy[base], zv[base], zmax[base]
    chi2, chi2fit, cls = chi2[base], chi2fit[base], cls[base]
    dx = {k: (v[base] if isinstance(v, np.ndarray) and v.ndim >= 1
              and len(v) == n else v) for k, v in dx.items()}
    n = int(base.sum())
    print(f"# classes: " + ", ".join(
        f"{c} {int((cls == i).sum())} ({100*(cls == i).mean():.3f} %)"
        for i, c in enumerate(names)))

    isig = names.index("signal")
    sig = cls == isig

    # ---------------- the pulls, per class ----------------
    print("\n=== the two beam pulls by GEN class")
    hdr = (f"{'class':<12}{'n':>7}{'<z_x>':>9}{'Var z_x':>9}{'<z_y>':>9}"
           f"{'Var z_y':>9}{'P(|z|>3)':>10}{'P(|z|>5)':>10}{'<chi2>':>9}")
    print(hdr)
    print("-"*len(hdr))
    for i, c in enumerate(names):
        m = cls == i
        if m.sum() < 5:
            continue
        p3, _ = binom(int((zmax[m] > 3).sum()), int(m.sum()))
        p5, _ = binom(int((zmax[m] > 5).sum()), int(m.sum()))
        print(f"{c:<12}{int(m.sum()):>7}{zx[m].mean():>9.3f}{zx[m].var():>9.3f}"
              f"{zy[m].mean():>9.3f}{zy[m].var():>9.3f}{p3:>10.5f}{p5:>10.5f}"
              f"{np.nanmean(chi2[m]):>9.3f}")
    # the trimmed core of the signal, for the moments that can be quoted
    for lab, m in (("signal", sig),):
        for nm, z in (("z_x", zx), ("z_y", zy)):
            v = z[m]
            lo, hi = np.percentile(v, [0.1, 99.9])
            t = v[(v > lo) & (v < hi)]
            print(f"  {lab} {nm}: n {v.size}  mean {v.mean():+.4f} "
                  f"+- {v.std()/np.sqrt(v.size):.4f}  Var {v.var():.4f}  "
                  f"trimVar {t.var():.4f}  "
                  f"skew {((t-t.mean())**3).mean()/t.std()**3:+.4f}  "
                  f"kurt {((t-t.mean())**4).mean()/t.var()**2:.4f}")

    # ---------------- cosmic-like pairs ----------------
    if "phi_plus" in dx and "eta_plus" in dx:
        pp, pm = np.asarray(dx["phi_plus"], float), np.asarray(dx["phi_minus"], float)
        ep, em = np.asarray(dx["eta_plus"], float), np.asarray(dx["eta_minus"], float)
        dphi = np.abs(np.arctan2(np.sin(pp - pm), np.cos(pp - pm)))
        cosmic = (np.abs(dphi - np.pi) < 0.05) & (np.abs(ep + em) < 0.05)
        k, nn = int(cosmic.sum()), n
        p, e = binom(k, nn)
        print(f"\n=== back-to-back COSMIC-LIKE pairs "
              f"(|dphi - pi| < 0.05 AND |eta+ + eta-| < 0.05): "
              f"{k} of {nn} = {100*p:.4f} +- {100*e:.4f} %")
        if k >= 5:
            print(f"  of those, gen class: " + ", ".join(
                f"{c} {int((cls[cosmic] == i).sum())}" for i, c in enumerate(names)))
            print(f"  their pulls: <z_x> {zx[cosmic].mean():+.3f}  "
                  f"Var {zx[cosmic].var():.3f}   <z_y> {zy[cosmic].mean():+.3f}  "
                  f"Var {zy[cosmic].var():.3f}   <|z_v|> {np.abs(zv[cosmic]).mean():.3f}")
            print(f"  the SIGNAL sample for comparison: <z_x> {zx[sig].mean():+.3f} "
                  f"Var {zx[sig].var():.3f}, <|z_v|> {np.abs(zv[sig]).mean():.3f}")
        # A Z pair is BACK TO BACK IN PHI by construction, so |dphi| ~ pi
        # alone is not a cosmic tag: the eta+ = -eta- half is what separates
        # them, and it is quoted separately.
        bb = np.abs(dphi - np.pi) < 0.05
        print(f"  (|dphi - pi| < 0.05 alone: {int(bb.sum())} = "
              f"{100*bb.mean():.3f} % -- a Z pair IS back to back, so this "
              f"alone tags nothing)")

    # ---------------- the rejection table ----------------
    print("\n=== what a cut buys: signal efficiency vs background rejection")
    bkg = ~sig
    nb0, ns0 = int(bkg.sum()), int(sig.sum())
    rows = []
    for lab, m in (("|z_bs| < 3", zmax < 3), ("|z_bs| < 4", zmax < 4),
                   ("|z_bs| < 5", zmax < 5),
                   ("|z_v| < 3", np.abs(zv) < 3), ("|z_v| < 4", np.abs(zv) < 4),
                   ("|z_v| < 5", np.abs(zv) < 5),
                   ("|z_bs|<3 AND |z_v|<3", (zmax < 3) & (np.abs(zv) < 3)),
                   ("|z_bs|<4 AND |z_v|<4", (zmax < 4) & (np.abs(zv) < 4)),
                   ("|z_bs|<5 AND |z_v|<5", (zmax < 5) & (np.abs(zv) < 5)),
                   ("chi2_bs(2dof) < 9", chi2 < 9.),
                   ("chi2_bs(2dof) < 16", chi2 < 16.),
                   ("beam-row chi2 < 9", chi2fit < 9.),
                   ("beam-row chi2 < 16", chi2fit < 16.)):
        m = np.asarray(m, bool) & np.isfinite(np.asarray(m, float) * 0 + 1)
        es, ees = binom(int((sig & m).sum()), ns0)
        eb, eeb = binom(int((bkg & m).sum()), nb0)
        rows.append((lab, es, ees, eb, eeb))
    hdr = f"{'cut':<24}{'eff(signal)':>14}{'eff(bkg)':>14}{'rejection':>12}{'loss':>10}"
    print(hdr)
    print("-"*len(hdr))
    for lab, es, ees, eb, eeb in rows:
        rej = 1 - eb
        print(f"{lab:<24}{es:>9.5f}+-{ees:<4.5f}{eb:>9.5f}+-{eeb:<4.5f}"
              f"{rej:>12.5f}{1-es:>10.5f}")
    print(f"\n  signal {ns0}, background {nb0} "
          f"({100*nb0/max(n,1):.3f} % of the sample)")

    # ---------------- the beam chi2 as a variable ----------------
    print("\n=== the beam rows' own chi2 (3 rows, at the optimum)")
    for lab, m in (("signal", sig), ("background", bkg)):
        v = chi2fit[m]
        v = v[np.isfinite(v)]
        if v.size < 5:
            continue
        print(f"  {lab:<12} n {v.size:6d}  median {np.median(v):8.4f}  "
              f"mean {v.mean():8.4f}  p90 {np.percentile(v,90):8.3f}  "
              f"p99 {np.percentile(v,99):9.3f}")


if __name__ == "__main__":
    main()
