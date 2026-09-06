#!/usr/bin/env python3
"""What the 1.5 (sigma_m/m)^2 Jensen coefficient assumes, measured on truth.

m is a nonlinear function of the fitted parameters and the CF propagates the
block fluctuations LINEARLY, so the model's location is short by
1/2 tr(H Sigma).  For m^2 = 2(E1 E2 - p1.p2) + 2 m_mu^2 with the fitted
parameters (kappa, lambda, phi) per leg,

    1/2 tr(H Sigma)/m = (3 A + B)/8 ,
    A = sigma_rel1^2 + sigma_rel2^2 ,  B = 2 rho sigma_rel1 sigma_rel2 ,

and (sigma_m/m)^2 = (A + B)/4 + (angular share), so the closed form
1.5 (sigma_m/m)^2 is exact ONLY for uncorrelated legs and a negligible angular
share.  This measures both, from the GEN leg kinematics: with

    d ln p_l = ln(p_l^reco/p_l^gen)

A, B come out directly, and swapping reco/gen angles isolates the angular share
of the mass resolution.  Everything is per bin of the EXPORTED sigma_m/m, so
the bin-by-bin correction to the closed form is the deliverable.

The two-track maker exports NO per-leg covariance (only `Jpsi_sigmamass` and
`resinfcov`), which is why this has to be measured rather than read off.
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import sigma_pull as SP                                       # noqa: E402

MMU = 0.1056583745
BR = ["Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass",
      "Muplus_pt", "Muplus_eta", "Muplus_phi",
      "Muminus_pt", "Muminus_eta", "Muminus_phi",
      "Muplusgen_pt", "Muplusgen_eta", "Muplusgen_phi",
      "Muminusgen_pt", "Muminusgen_eta", "Muminusgen_phi"]


def mass(pt1, eta1, phi1, pt2, eta2, phi2):
    px1, py1, pz1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)
    px2, py2, pz2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)
    e1 = np.sqrt(px1 ** 2 + py1 ** 2 + pz1 ** 2 + MMU ** 2)
    e2 = np.sqrt(px2 ** 2 + py2 ** 2 + pz2 ** 2 + MMU ** 2)
    return np.sqrt(np.maximum((e1 + e2) ** 2 - (px1 + px2) ** 2
                              - (py1 + py2) ** 2 - (pz1 + pz2) ** 2, 0.))


def one(fn):
    a = uproot.open(fn)["tree"].arrays(BR, library="np")
    return {k: a[k].astype(np.float64) for k in BR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--ntasks", type=int, default=60)
    ap.add_argument("--nproc", type=int, default=30)
    ap.add_argument("--nbin", type=int, default=5)
    ap.add_argument("--binon", choices=["sbar", "sigma"], default="sbar")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    files = sorted(glob.glob(a.files))[: a.ntasks]
    from multiprocessing import Pool
    with Pool(a.nproc) as pool:
        parts = pool.map(one, files)
    A = {k: np.concatenate([q[k] for q in parts]) for k in BR}
    n0 = len(A["Jpsi_mass"])

    m = A["Jpsi_mass"]; sm = A["Jpsi_sigmamass"]; mg = A["Jpsigen_mass"]
    srel = sm / mg
    ok = (np.isfinite(srel) & (srel > 0) & (np.abs(mg - 3.0969) < 0.35)
          & (A["Muplusgen_pt"] > 0) & (A["Muminusgen_pt"] > 0))
    ok &= srel < 5. * np.median(srel[ok])
    for k in A:
        A[k] = A[k][ok]
    m, sm, mg, srel = m[ok], sm[ok], mg[ok], srel[ok]
    n = len(m)

    # the three masses
    mfull = mass(A["Muplus_pt"], A["Muplus_eta"], A["Muplus_phi"],
                 A["Muminus_pt"], A["Muminus_eta"], A["Muminus_phi"])
    mgen = mass(A["Muplusgen_pt"], A["Muplusgen_eta"], A["Muplusgen_phi"],
                A["Muminusgen_pt"], A["Muminusgen_eta"], A["Muminusgen_phi"])
    mmom = mass(A["Muplus_pt"] * np.cosh(A["Muplus_eta"])
                / np.cosh(A["Muplusgen_eta"]),
                A["Muplusgen_eta"], A["Muplusgen_phi"],
                A["Muminus_pt"] * np.cosh(A["Muminus_eta"])
                / np.cosh(A["Muminusgen_eta"]),
                A["Muminusgen_eta"], A["Muminusgen_phi"])
    mang = mass(A["Muplusgen_pt"], A["Muplus_eta"], A["Muplus_phi"],
                A["Muminusgen_pt"], A["Muminus_eta"], A["Muminus_phi"])
    # per-leg relative momentum residual
    p1 = A["Muplus_pt"] * np.cosh(A["Muplus_eta"])
    p2 = A["Muminus_pt"] * np.cosh(A["Muminus_eta"])
    g1 = A["Muplusgen_pt"] * np.cosh(A["Muplusgen_eta"])
    g2 = A["Muminusgen_pt"] * np.cosh(A["Muminusgen_eta"])
    d1, d2 = np.log(p1 / g1), np.log(p2 / g2)

    L = []; P = L.append
    P(f"# {len(files)} files, {n0} tree rows, {n} after the mass/sigma window")
    P(f"# control: max|m(Muplus,Muminus) - Jpsi_mass| = "
      f"{np.abs(mfull - m).max():.2e} GeV, median "
      f"{np.median(np.abs(mfull - m)):.2e}")
    P(f"# control: max|m(gen legs) - Jpsigen_mass| = "
      f"{np.abs(mgen - mg).max():.2e} GeV")
    P("")
    P("## per bin of the EXPORTED sigma_m/m: what 1.5 (sigma_m/m)^2 misses")
    P("A = s1^2+s2^2, B = 2 rho s1 s2 measured from d ln p per leg; "
      "f_ang from swapping the angles; the Jensen coefficient is (3A+B)/8.")
    P("| bin | n | <srel> | rms(dlnm) | s1 | s2 | rho | f_ang | "
      "(3A+B)/8 | 1.5 srel^2 | ratio |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    # BINNING VARIABLE.  Binning on the EXPORTED sigma_m is binning on the mass
    # fluctuation itself (sigma_m = sigma_bar (1 + a_m x)), which displaces the
    # mass location inside each bin by exactly the artefact of the previous
    # entry -- up to 1e-3, an order above the Jensen term.  The binning must be
    # done on a sigma that cannot see the fluctuation: the leave-one-out cell
    # mean of ln sigma_m in cells built from GEN leg kinematics only.
    def qb(v, nb):
        return np.clip(np.digitize(v, np.quantile(v, np.linspace(0, 1, nb + 1))[1:-1]),
                       0, nb - 1).astype(np.int64)
    gpmin, gpmax = np.minimum(g1, g2), np.maximum(g1, g2)
    ge1 = np.abs(A["Muplusgen_eta"]); ge2 = np.abs(A["Muminusgen_eta"])
    cid = SP.cellid(qb(np.log(gpmin), 8), qb(np.log(gpmax), 8),
                    qb(np.minimum(ge1, ge2), 6), qb(np.maximum(ge1, ge2), 6))
    nc = int(cid.max()) + 1
    sbar = np.exp(SP.cellmean(np.log(sm), cid, nc, leave_one_out=True))
    binvar = sbar / mg if a.binon == "sbar" else srel
    P(f"# binning on {a.binon} ({nc} truth cells, {n/nc:.0f} candidates/cell)")
    e = np.quantile(binvar, np.linspace(0, 1, a.nbin + 1))
    b = np.clip(np.digitize(binvar, e[1:-1]), 0, a.nbin - 1)
    tot = dict(num=0., den=0.)
    for k in range(a.nbin):
        msk = b == k
        # robust (trimmed at 4 sigma of the pull) so the Landau tail does not
        # own a variance
        z = (m - mg) / sm
        msk = msk & (np.abs(z) < 4.)
        s1 = float(d1[msk].std(ddof=1)); s2 = float(d2[msk].std(ddof=1))
        rho = float(np.corrcoef(d1[msk], d2[msk])[0, 1])
        vfull = float(np.log(mfull[msk] / mgen[msk]).std(ddof=1)) ** 2
        vang = float(np.log(mang[msk] / mgen[msk]).std(ddof=1)) ** 2
        vmom = float(np.log(mmom[msk] / mgen[msk]).std(ddof=1)) ** 2
        fang = vang / vfull
        AA = s1 ** 2 + s2 ** 2
        BB = 2. * rho * s1 * s2
        jex = (3. * AA + BB) / 8.
        jcf = 1.5 * float((srel[msk] ** 2).mean())
        P(f"| {e[k]:.4f}..{e[k+1]:.4f} | {int(msk.sum())} | "
          f"{srel[msk].mean():.5f} | {np.sqrt(vfull):.5f} | {s1:.5f} | {s2:.5f} "
          f"| {rho:+.3f} | {fang:.3f} | {1e4*jex:+.3f}e-4 | {1e4*jcf:+.3f}e-4 "
          f"| {jex/jcf:.3f} |")
        # 1/sigma^2-weighted accumulation of the EXACT coefficient
        w = 1. / sm[msk] ** 2
        tot["num"] += (jex * w).sum(); tot["den"] += w.sum()
        tot.setdefault("numcf", 0.)
        tot["numcf"] = tot.get("numcf", 0.) + (jcf * w).sum()
    P("")
    P(f"1/sigma^2-weighted EXACT Jensen coefficient (bin-wise A, B): "
      f"**{1e3*tot['num']/tot['den']:+.4f} e-3**")
    P(f"1/sigma^2-weighted CLOSED FORM 1.5 (sigma_m/m)^2:            "
      f"**{1e3*tot['numcf']/tot['den']:+.4f} e-3**")
    P("")
    P("## the truth-level mass bias, same binning (what alpha should see)")
    P("The MEAN is tail-dominated and is NOT what alpha measures (alpha is a "
      "location parameter); the trimmed/median columns are the location.")
    P("| bin | mean |z|<4 [1e-4] | mean |z|<2 [1e-4] | median [1e-4] | "
      "predicted (3A+B)/8 [1e-4] |")
    P("|---|---|---|---|---|")
    zz = (m - mg) / sm
    for k in range(a.nbin):
        m4 = (b == k) & (np.abs(zz) < 4.)
        m2 = (b == k) & (np.abs(zz) < 2.)
        r4 = float((m[m4] / mg[m4] - 1.).mean())
        e4 = float((m[m4] / mg[m4]).std(ddof=1) / np.sqrt(m4.sum()))
        r2 = float((m[m2] / mg[m2] - 1.).mean())
        e2 = float((m[m2] / mg[m2]).std(ddof=1) / np.sqrt(m2.sum()))
        rm = float(np.median(m[m4] / mg[m4] - 1.))
        s1 = float(d1[m4].std(ddof=1)); s2 = float(d2[m4].std(ddof=1))
        rho = float(np.corrcoef(d1[m4], d2[m4])[0, 1])
        jex = (3. * (s1 ** 2 + s2 ** 2) + 2. * rho * s1 * s2) / 8.
        P(f"| {e[k]:.4f}..{e[k+1]:.4f} | {1e4*r4:+.3f}+-{1e4*e4:.3f} "
          f"| {1e4*r2:+.3f}+-{1e4*e2:.3f} | {1e4*rm:+.3f} | {1e4*jex:+.3f} |")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
