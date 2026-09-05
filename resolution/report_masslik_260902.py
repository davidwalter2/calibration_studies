"""Summary numbers for a mass-likelihood run: candidate count, kernel FSR
tail, sigma_m median, mass-pull width, and an unpaired distributional
comparison of (m_reco - m_gen)/sigma against a reference cache.

usage: python report_masslik_260902.py --tag btojpsix_v3_260902_m0 \
          [--ref runs/cf_masspairs_operaC.npz --ref-name "rung B (260805, single-track paired)"]
"""
import argparse
import os

import numpy as np


def robust(x, qs=(0.16, 0.5, 0.84)):
    q = np.quantile(x, qs)
    return q[1], 0.5 * (q[2] - q[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--runs", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs"))
    p.add_argument("--ref", default=None)
    p.add_argument("--ref-name", default="reference")
    a = p.parse_args()

    kf = os.path.join(a.runs, f"cf_masskernel_{a.tag}.npz")
    pf = os.path.join(a.runs, f"cf_masspairs_{a.tag}.npz")
    sf = os.path.join(a.runs, f"masslik_demo_scan_{a.tag}.npz")

    k = np.load(kf)
    dm = k["dm"]
    print(f"kernel  {kf}")
    print(f"  N gen pairs in window : {len(dm)}")
    print(f"  FSR tail (< -5 MeV)   : {100*float(k['frac_tail']):.2f} % "
          f"(+- {100*np.sqrt(float(k['frac_tail'])*(1-float(k['frac_tail']))/len(dm)):.2f})")
    print(f"  tail (< -1 MeV)       : {100*(dm < -0.001).mean():.2f} %")
    print(f"  median dm             : {1e3*np.median(dm):+.4f} MeV")
    print(f"  gen pole              : {3.0969 + np.median(dm):.6f} GeV "
          f"(cf_mass_likelihood MJPSI = 3.0969)")

    d = np.load(pf)
    z, sig = d["z"], d["sigma"]
    med, wid = robust(z)
    print(f"\npairs   {pf}")
    print(f"  N candidates          : {len(z)}")
    print(f"  sigma_m median        : {1e3*np.median(sig):.2f} MeV")
    print(f"  mass pull median      : {med:+.4f}")
    print(f"  mass pull robust width: {wid:.4f}")
    print(f"  raw pull std          : {z.std():.3f}")
    dmr = z * sig
    print(f"  m_reco - m_gen median : {1e3*np.median(dmr):+.4f} MeV "
          f"(robust width {1e3*robust(dmr)[1]:.2f} MeV)")

    if os.path.exists(sf):
        s = np.load(sf)
        nll, alphas, rs = s["nll"], s["alphas"], s["rs"]
        irb, iab = np.unravel_index(np.argmin(nll), nll.shape)
        y = nll[irb]
        ia = int(np.clip(iab, 1, len(alphas) - 2))
        h = alphas[1] - alphas[0]
        c = (y[ia+1] + y[ia-1] - 2*y[ia]) / h**2
        ah = alphas[ia] - (y[ia+1] - y[ia-1]) / (2*c*h)
        err = 1. / np.sqrt(c)
        prof = nll.min(axis=1) - nll.min()
        ir = int(np.clip(irb, 1, len(rs) - 2))
        hr = rs[1] - rs[0]
        cr = (prof[ir+1] + prof[ir-1] - 2*prof[ir]) / hr**2
        rh = rs[ir] - (prof[ir+1] - prof[ir-1]) / (2*cr*hr)
        print(f"\nscan    {sf}")
        print(f"  alpha  = ({1e3*ah:+.4f} +- {1e3*err:.4f}) e-3  "
              f"(Dm = {1e3*ah*3.0969:+.3f} MeV)")
        print(f"  r      = {rh:.4f} +- {1./np.sqrt(cr):.4f} (grid min {rs[irb]:.2f})")
        print(f"  grid edges: r {rs[0]:.2f}-{rs[-1]:.2f}, "
              f"alpha {1e3*alphas[0]:.3f}-{1e3*alphas[-1]:.3f} e-3 "
              f"(railed: r={irb in (0, len(rs)-1)}, alpha={iab in (0, len(alphas)-1)})")
        wg = 1. / sig**2
        mobs = z * sig + (d["eta"] - 3.0969)
        ag = np.sum(wg * mobs) / np.sum(wg) / 3.0969
        print(f"  naive Gaussian alpha  : {1e3*ag:+.4f} e-3  "
              f"(bias {1e3*(ag-ah):+.4f} e-3)")
    else:
        print(f"\nscan    {sf}: NOT PRESENT")

    if a.ref and os.path.exists(a.ref):
        r = np.load(a.ref)
        rz, rs_ = r["z"], r["sigma"]
        rm, rw = robust(rz)
        rdm = rz * rs_
        n1, n2 = len(z), len(rz)
        # unpaired: the older production's inputs are gone, so this compares
        # DISTRIBUTIONS, not the same candidates
        se = np.sqrt(robust(dmr)[1]**2/n1 + rw**2*np.median(rs_)**2/n2)
        print(f"\nunpaired comparison vs {a.ref_name}\n  {a.ref}")
        print(f"  {'':22s} {'this':>12s} {'ref':>12s}")
        print(f"  {'N candidates':22s} {n1:12d} {n2:12d}")
        print(f"  {'sigma_m median [MeV]':22s} {1e3*np.median(sig):12.2f} {1e3*np.median(rs_):12.2f}")
        print(f"  {'pull median':22s} {med:12.4f} {rm:12.4f}")
        print(f"  {'pull robust width':22s} {wid:12.4f} {rw:12.4f}")
        print(f"  {'m_rec-m_gen med [MeV]':22s} {1e3*np.median(dmr):12.4f} {1e3*np.median(rdm):12.4f}"
              f"   diff {1e3*(np.median(dmr)-np.median(rdm)):+.4f} +- {1e3*se:.4f}")


if __name__ == "__main__":
    main()
