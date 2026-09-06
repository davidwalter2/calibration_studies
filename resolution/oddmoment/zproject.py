#!/usr/bin/env python3
"""What the mass-pull self-consistency coefficient becomes at Z momenta.

a_m = (1 + f_hit - f_ioni) sigma_m/m, and both factors grow from J/psi to Z:
sigma_m/m because the relative q/p resolution grows with p, f_hit because the
hit term is the p-INDEPENDENT part of sigma_qop and therefore takes over.

Estimated by pairing the single-track muon-gun tracks at Z momenta (pT 20-60,
the sample the Z's muons live in) into synthetic q+q- candidates:

    sigma_m/m = 1/2 sqrt(sigma_rel1^2 + sigma_rel2^2)     [fixed angles]
    f_hit     = (v1 vgf1 + v2 vgf2)/(v1 + v2),  v_l = sigma_rel,l^2

(the opening-angle term is neglected: the angular resolution contributes at the
1e-4 level on m, an order below the momentum term).
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import decomp as D                                            # noqa: E402

MZ = 91.1876
MJPSI = 3.0969


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz")
    ap.add_argument("--F", type=float, nargs="+", default=[1.62, 2.0])
    ap.add_argument("--FJ", type=float, default=0.56,
                    help="measured response of the MLE to the second-order term")
    ap.add_argument("--harm-ratio", type=float, default=0.811,
                    help="sigma_harm/<sigma> measured on the gun candidates")
    ap.add_argument("--ratio-eff", type=float, default=0.786,
                    help="sigma_bar_eff/<sigma> measured on the gun candidates")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = D.load(a.cache, need_model=False)
    z = d["z"]; sig = d["sigma"]; vgf = d["vgf"]
    eta = d["eta"].astype(np.float64)
    q = np.sign(d["charge"])
    p = d["genpt"] * np.cosh(eta)
    srel = sig * p
    ip = np.flatnonzero(q > 0); im = np.flatnonzero(q < 0)
    n = min(len(ip), len(im))
    rng = np.random.default_rng(11)
    i1 = rng.permutation(ip)[:n]; i2 = rng.permutation(im)[:n]
    v1, v2 = srel[i1] ** 2, srel[i2] ** 2
    smrel = 0.5 * np.sqrt(v1 + v2)
    fhit = (v1 * vgf[i1] + v2 * vgf[i2]) / (v1 + v2)
    am = (1. + fhit) * smrel
    L = []; P = L.append
    P(f"# synthetic Z candidates from {os.path.basename(a.cache)}: {n} pairs")
    P("| quantity | mean | median |")
    P("|---|---|---|")
    P(f"| per-leg sigma_rel(q/p) | {srel.mean():.5f} | {np.median(srel):.5f} |")
    P(f"| per-leg vgf (hit share) | {vgf.mean():.4f} | {np.median(vgf):.4f} |")
    P(f"| sigma_m/m | {smrel.mean():.5f} | {np.median(smrel):.5f} |")
    P(f"| f_hit (mass) | {fhit.mean():.4f} | {np.median(fhit):.4f} |")
    P(f"| **a_m = (1+f_hit) sigma_m/m** | **{am.mean():.5f}** | "
      f"**{np.median(am):.5f}** |")
    P("")
    P("## the alpha bias the naive per-candidate likelihood would carry")
    P(f"sigma_bar_eff/M taken as {a.ratio_eff:.3f} x <sigma_m/m> "
      "(the ratio measured on the J/psi gun candidates)")
    P("| F | alpha bias [1e-3] | on m_Z [MeV] |")
    P("|---|---|---|")
    for F in a.F:
        for nm, (amv, smv) in (("mean", (am.mean(), smrel.mean())),
                               ("median", (np.median(am), np.median(smrel)))):
            bias = -amv * F * a.ratio_eff * smv
            P(f"| {F:.2f} ({nm}) | {1e3*bias:+.4f} | {MZ*bias*1e3:+.1f} |")
    P("")
    P("## the SECOND-ORDER (Jensen) term of the mass functional, same sample")
    P("alpha_Jensen = 1.5 <sigma_m/m>^2 with the 1/sigma^2 (harmonic) weighting "
      "the likelihood actually applies; the gun's harmonic/arithmetic ratio "
      f"({a.harm_ratio:.3f}) is carried over.")
    sh = a.harm_ratio * smrel
    aj = 1.5 * sh ** 2
    P("| quantity | mean | median |")
    P("|---|---|---|")
    P(f"| sigma_harm/m = {a.harm_ratio:.3f} x sigma_m/m | {sh.mean():.5f} | "
      f"{np.median(sh):.5f} |")
    P(f"| **alpha_Jensen = 1.5 sigma_harm^2/m^2** | {1e3*aj.mean():+.4f}e-3 "
      f"({MZ*aj.mean()*1e3:+.1f} MeV) | {1e3*np.median(aj):+.4f}e-3 "
      f"({MZ*np.median(aj)*1e3:+.1f} MeV) |")
    P("")
    P("## the NET of the two, if NEITHER correction is applied")
    P("Both scale as sigma_rel^2 -- the artefact as -(1+f_hit) F sigma_rel^2 "
      "and the Jensen term as +(1.5 - f_ang) sigma_rel^2 -- so they partially "
      "CANCEL, with a ratio fixed by (1.5 - f_ang)/[(1+f_hit) F].  The "
      "cancellation is an accident of those two numbers and must not be relied "
      "on: each term alone is ~30 MeV at the Z.")
    P("The Jensen column carries F_J, the MEASURED response of the mass MLE to "
      "a second-order (quadratic) perturbation as opposed to a location shift: "
      f"F_J = {a.FJ:.2f} (0.73 at J/psi sigma_m/m, 0.56 at Z-like 1.85 %; "
      "toy + exact-vs-shift, NOTES 2026-09-05 (IV)).  The artefact's own "
      "response is already inside F.")
    P("| F | artefact [MeV] | Jensen x F_J [MeV] | net [MeV] | ratio J/A |")
    P("|---|---|---|---|---|")
    for F in a.F:
        for nm, (amv, smv, ajv) in (("mean", (am.mean(), smrel.mean(), aj.mean())),
                                    ("median", (np.median(am), np.median(smrel),
                                                np.median(aj)))):
            art = -amv * F * a.ratio_eff * smv
            ajf = a.FJ * ajv
            P(f"| {F:.2f} ({nm}) | {MZ*art*1e3:+.1f} | {MZ*ajf*1e3:+.1f} "
              f"| {MZ*(art+ajf)*1e3:+.1f} | {-ajf/art:.2f} |")
    P("")
    P(f"For scale: the J/psi gun measured a_m = 0.0107, sigma_bar_eff/M = "
      f"0.0088, F = 1.62 -> -0.155e-3, i.e. -0.48 MeV on m(J/psi). The SAME "
      f"defect at Z momenta is {am.mean()/0.0107 * (a.ratio_eff*smrel.mean())/0.0088:.1f}x "
      f"larger in relative terms.")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
