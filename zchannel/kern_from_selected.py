#!/usr/bin/env python3
"""Rebuild the FSR kernel (and the acceptance) from the gen record of the
SELECTED reconstructed candidates.

WHY.  The kernel in use is built from GENERATOR events passing a gen fiducial
cut (`pT > 5`, `|eta| < 2.4`).  The candidates the likelihood is evaluated on
have been through the production's own `60 < Jpsitrk_mass < 120` cut, the
trigger, the reconstruction and the analysis quality cuts.  Those remove hard
radiation, and they remove it at a rate that depends on the Born mass.
Measured on this sample,

    <u> = <-ln(m_post/m_pre)>       gen, inclusive              0.027147
                                   gen, pT>5 |eta|<2.4         0.023456
                                   the kernel npz in use       0.024032
                                   RECONSTRUCTED, selected     0.014447

so the fold in the likelihood describes a sample radiating 1.66x more than the
one it is folded against.  `A(m_pre)` cannot absorb this: it is a function of
the BORN mass, and the defect is a correlation between the selection and `u` at
fixed `m_pre`.

The fix is by construction.  The model is

    p(m_obs) ~ int dm_pre BW(m_pre) A(m_pre) k(m_obs/m_pre | m_pre)

with `A(m_pre) = P(selected | m_pre)` and `k` the CONDITIONAL kernel of the
selected sample.  Both are measured here from the selected candidates' own gen
record (`mpre` = `Jpsigenpre_mass`, `eta` = `Jpsigen_mass` in the pairs cache),
`A` against the generator-level `m_pre` spectrum for its denominator.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolution"))
import selection as _SEL  # noqa: E402  (ONE value for the chi2 cut)
import fit_gen as FG                                          # noqa: E402

BANDS = [70.0, 80.0, 85.0, 88.0, 91.0, 94.0, 98.0, 105.0, 115.0, 130.0]


def load_selected(path, window, max_chi2, max_srel, verbose=True):
    """The pairs cache, cut exactly as `fullscale/make_card.py::select` cuts."""
    d = np.load(path)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mpost = d["eta"].astype(np.float64)          # `eta` IS the post-FSR gen mass
    mpre = d["mpre"].astype(np.float64)
    m = z * sig + mpost                          # the reconstructed mass
    w = (d["w"].astype(np.float64) if "w" in d.files else np.ones(len(z)))
    srel = sig / np.maximum(np.abs(m), 1e-9)

    keep = np.isfinite(m) & np.isfinite(sig) & (sig > 0.0)
    steps = [("finite m, sigma > 0", keep.sum())]
    keep &= (m >= window[0]) & (m <= window[1])
    steps.append((f"m_obs in [{window[0]:g}, {window[1]:g}]", keep.sum()))
    if max_chi2 > 0 and "chisqval" in d.files:
        keep &= (d["chisqval"].astype(np.float64)
                 / np.maximum(d["ndof"].astype(np.float64), 1.0)) < max_chi2
        steps.append((f"chi2/ndof < {max_chi2:g}", keep.sum()))
    if max_srel > 0:
        keep &= srel < max_srel
        steps.append((f"sigma_m/m < {max_srel:g}", keep.sum()))
    keep &= np.isfinite(mpre) & (mpre > 0.0) & np.isfinite(mpost) & (mpost > 0.0)
    steps.append(("finite gen masses", keep.sum()))
    if verbose:
        print(f"[sel] {len(z)} cached candidates")
        for name, n in steps:
            print(f"    {name:32s} {n:9d}")
    out = dict(mpre=mpre[keep], mpost=mpost[keep], w=w[keep], m=m[keep])
    for k in ("ptp", "ptm", "etap", "etam", "etapair"):
        if k in d.files:
            out[k] = d[k].astype(np.float64)[keep]
    return out


def census(mpre, mpost, w, label=""):
    u = -np.log(np.asarray(mpost) / np.asarray(mpre))
    sw = w.sum()
    mu = float(np.sum(w * u) / sw)
    rms = float(np.sqrt(max(np.sum(w * u * u) / sw - mu * mu, 0.0)))
    neff = float(sw * sw / np.sum(w * w))
    print(f"[u] {label:24s} <u> = {mu:.6f}  RMS = {rms:.6f}  "
          f"N_eff = {neff:.0f}  sigma(<u>) = {rms / np.sqrt(neff):.3e}")
    return mu, rms, neff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the Z pairs cache")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0])
    ap.add_argument("--max-chi2-ndof", type=float,
                    default=_SEL.MAX_CHI2_NDOF)
    ap.add_argument("--max-sigma-rel", type=float, default=0.10)
    ap.add_argument("--sigma-cap", type=float, default=3.3e-4)
    ap.add_argument("--u-fine", type=float, default=2e-5)
    ap.add_argument("--bands", type=float, nargs="*", default=BANDS)
    ap.add_argument("--eta-band", type=float, nargs=2, default=None,
                    help="restrict to this |eta| range of the LEADING muon")
    ap.add_argument("--half", choices=["a", "b"], default=None,
                    help="half the sample, for the statistical-precision test")
    ap.add_argument("--gen", default=None,
                    help="generator merge, for the acceptance denominator")
    ap.add_argument("--acc-out", default=None)
    ap.add_argument("--acc-degree", type=int, default=8)
    ap.add_argument("--acc-kind", choices=["bernstein", "grid"], default="grid",
                    help="'grid' tabulates the measured A(m_pre) and lets the "
                         "provider interpolate it. The selected sample's "
                         "acceptance is a TOP-HAT (the production's 60-120 "
                         "mass cut seen through m_post ~ m_pre) and no "
                         "low-degree Bernstein can represent a step: degree 8 "
                         "fits it at chi2/ndf = 30638/184.")
    ap.add_argument("--acc-nbins", type=int, default=300)
    ap.add_argument("--acc-lo", type=float, default=50.0)
    ap.add_argument("--acc-hi", type=float, default=200.0)
    a = ap.parse_args()

    s = load_selected(a.cache, a.window, a.max_chi2_ndof, a.max_sigma_rel)
    sel = np.ones(len(s["mpre"]), bool)
    if a.eta_band is not None and "ptp" in s:
        lead = np.where(s["ptp"] >= s["ptm"], s["etap"], s["etam"])
        sel &= (np.abs(lead) >= a.eta_band[0]) & (np.abs(lead) < a.eta_band[1])
        print(f"[sel] |eta_lead| in [{a.eta_band[0]}, {a.eta_band[1]}): "
              f"{sel.sum()}")
    if a.half:
        n = len(sel)
        sel &= (np.arange(n) < n // 2) if a.half == "a" else (np.arange(n) >= n // 2)
        print(f"[sel] half {a.half}: {sel.sum()}")
    mpre, mpost, w = s["mpre"][sel], s["mpost"][sel], s["w"][sel]

    mu, rms, neff = census(mpre, mpost, w, "selected candidates")
    print(f"[u] sigma(<u>) = {rms / np.sqrt(neff):.3e}  -> "
          f"{rms / np.sqrt(neff) * 91.1876 * 1e3:.2f} MeV on the peak position")

    bands = list(zip(a.bands[:-1], a.bands[1:]))
    bands = [(0.0, a.bands[0])] + bands + [(a.bands[-1], np.inf)]
    ker, info = FG.build_banded_kernel(mpre, mpost, w, bands,
                                       sigma_cap=a.sigma_cap, u_fine=a.u_fine)
    meta = dict(source="selected reconstructed candidates",
                cache=os.path.abspath(a.cache), n=int(len(mpre)),
                sumw=float(w.sum()), window=list(a.window),
                max_chi2_ndof=a.max_chi2_ndof, max_sigma_rel=a.max_sigma_rel,
                eta_band=a.eta_band, half=a.half, sigma_cap=a.sigma_cap,
                u_fine=a.u_fine, bands=list(a.bands),
                mean_u=mu, rms_u=rms, n_eff=neff)
    np.savez(a.output, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
             m_hi=ker["m_hi"], provenance=np.array([json.dumps(meta)]))
    print(f"[kernel] {len(mpre)} candidates, {len(info)} bands, "
          f"{len(ker['r'])} atoms -> {a.output}")
    for lo, hi, n, na, pn, md in info:
        print(f"    [{lo:6.1f}, {hi:6.1f})  n = {n:9d}  atoms = {na:4d}"
              f"  P(no rad) = {pn:.4f}  <dm/m> = {md * 1e3:+8.4f}e-3")

    if a.gen and a.acc_out:
        g = FG.load_gen(a.gen)
        wg = FG.clip_weights(g["weight"].astype(float), 100.0)
        build_acceptance(mpre, w, g["m_pre"], wg, a)


def build_acceptance(mpre_sel, w_sel, mpre_gen, w_gen, a):
    """`A(m_pre) = P(selected | m_pre)`, up to a constant, as a Bernstein fit.

    The numerator is the selected candidates' Born-mass spectrum, the
    denominator the generator's.  The two come from different task subsets of
    the same dataset, so the ratio is the efficiency SHAPE times an unknown
    constant -- which is all the model uses, `A` being renormalised inside the
    provider.
    """
    from math import comb
    lo, hi, deg = a.acc_lo, a.acc_hi, a.acc_degree
    e = np.linspace(lo, hi, a.acc_nbins + 1)
    num, _ = np.histogram(mpre_sel, e, weights=w_sel)
    num2, _ = np.histogram(mpre_sel, e, weights=w_sel ** 2)
    den, _ = np.histogram(mpre_gen, e, weights=w_gen)
    den2, _ = np.histogram(mpre_gen, e, weights=w_gen ** 2)
    ok = (num > 0) & (den > 0)
    c = 0.5 * (e[1:] + e[:-1])
    r = np.where(ok, num / np.maximum(den, 1e-12), 0.0)
    rel = np.sqrt(np.where(ok, num2 / np.maximum(num, 1e-12) ** 2
                           + den2 / np.maximum(den, 1e-12) ** 2, 0.0))
    scale = float(np.max(r[ok]))
    y, sig = r / scale, np.maximum(r * rel / scale, 1e-6)

    if a.acc_kind == "grid":
        # the measured ratio, with empty bins filled by the neighbours: the
        # provider interpolates linearly and extrapolates flat.
        yy = np.where(ok, y, np.nan)
        idx = np.arange(len(yy))
        good = ~np.isnan(yy)
        yy = np.interp(idx, idx[good], yy[good])
        cfg = {"kind": "grid", "m": [float(v) for v in c],
               "a": [float(v) for v in yy],
               "_meta": {"source": "selected reconstructed candidates",
                         "nbins": int(a.acc_nbins), "scale": scale,
                         "n_empty": int((~ok).sum())}}
        with open(a.acc_out, "w") as fh:
            json.dump(cfg, fh, indent=1)
        np.savez(a.acc_out.replace(".json", "_diag.npz"),
                 m=c, a=y, sig=sig, fit=yy, ok=ok)
        print(f"[acc] grid, {a.acc_nbins} bins over [{lo:g}, {hi:g}], "
              f"{(~ok).sum()} empty -> {a.acc_out}")
        return

    x = np.clip((c - lo) / (hi - lo), 0.0, 1.0)
    B = np.stack([comb(deg, k) * x ** k * (1 - x) ** (deg - k)
                  for k in range(deg + 1)], 1)
    A_ = B[ok] / sig[ok, None]
    coef, *_ = np.linalg.lstsq(A_, y[ok] / sig[ok], rcond=None)
    fit = B @ coef
    chi2 = float(np.sum(((y[ok] - fit[ok]) / sig[ok]) ** 2))
    cfg = {"kind": "bernstein", "lo": lo, "hi": hi,
           "coef": [float(v) for v in coef],
           "_meta": {"source": "selected reconstructed candidates",
                     "degree": deg, "chi2": chi2, "ndf": int(ok.sum() - deg - 1),
                     "scale": scale}}
    with open(a.acc_out, "w") as fh:
        json.dump(cfg, fh, indent=1)
    np.savez(a.acc_out.replace(".json", "_diag.npz"),
             m=c, a=y, sig=sig, fit=fit, ok=ok)
    print(f"[acc] degree {deg}, chi2/ndf = {chi2:.1f}/{ok.sum() - deg - 1}"
          f" -> {a.acc_out}")


if __name__ == "__main__":
    main()
