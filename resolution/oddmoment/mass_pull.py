#!/usr/bin/env python3
"""The candidate (dimuon-mass) analogue of oddmoment/sigma_pull.py.

`Jpsi_sigmamass` is the two-track fit's OWN mass error, assembled from the
block variances at the FITTED state, so it is a function of the fitted mass.
Writing sigma_m = sigma_bar (1 + a_m x) with x = (m - m_gen)/sigma_bar,

    z_m = x/(1 + a_m x) ,   <z_m> = -a_m ,   <m/m_gen - 1> unaffected.

CLOSED FORM.  sigma_m^2 = A m^4 + B m^2 + C, because at fixed angles the hit
contribution to sigma_rel grows as p (sigma_qop,hit is p-independent), the MS
one is p-independent and the ionization one falls as 1/p.  Hence

    d ln sigma_m / d ln m = 2 f_hit + f_ms  =  1 + f_hit - f_ioni ,
    a_m = (2 f_hit + f_ms) * sigma_m/m ,

with f_hit/f_ms/f_ioni the per-family shares of sigma_m^2 measured by
oddmoment/aux_gen.py from resinfvarv split by parmtype.

THE ESTIMATOR FACTOR.  The mass likelihood's score is
S(alpha) = sum_i (M/sigma_i) psi(u_i) with psi = -(ln g)' of the standardized
model g.  Expanding both the residual and the 1/sigma_i prefactor,

    E[alpha_hat] - alpha = -a_m (sigma_bar/M) * F ,
    F = (1 + E[x^2 psi'(x)]) / E[psi'(x)]     (F = 2 for a Gaussian model)

so the naive Gaussian estimate -2 a_m sigma_bar/M is an UPPER bound: for a
model with a sharp core and heavy one-sided material tails E[psi'] = I > 1 and
E[x^2 psi'] < 1, and F falls well below 2.  F is measured here from the pooled
model density itself.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)
sys.path.insert(0, _HERE)
import sigma_pull as SP                                       # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load_module("cft", os.path.join(_RES, "cf_track_resolution.py"))
TG = m.TG
MJPSI = 3.0969


class Args:
    khit = 0.0
    kms = 0.0
    kioni = 0.0
    krad = 1.0
    kdel = None
    hitmode = "gauss"


def qbin(v, nb):
    return np.clip(np.digitize(v, np.quantile(v, np.linspace(0, 1, nb + 1))[1:-1]),
                   0, nb - 1).astype(np.int64)


def pooled_phi(d, args, chunk=20000):
    n = len(d["z"])
    phi = np.zeros(len(TG), dtype=np.complex128)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sub = {k: d[k][lo:hi] for k in
               ("Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im") if k in d}
        sub["vgf"] = d["vgf"][lo:hi]
        phi += m.model_phi(sub, args).sum(axis=0)
    return phi / n


def density(phibar, zg, tg=TG):
    w = np.gradient(tg); w[0] *= 0.5; w[-1] *= 0.5
    ph = tg[None, :] * zg[:, None]
    return ((np.cos(ph) * phibar.real[None, :]
             + np.sin(ph) * phibar.imag[None, :]) @ w) / np.pi


def estimator_factor(phibar, zmax=25., nz=20001, floor=1e-12):
    """F = (1 + E[x^2 psi']) / E[psi'] with psi = -(ln g)' of the model."""
    zg = np.linspace(-zmax, zmax, nz)
    p = density(phibar, zg)
    dz = zg[1] - zg[0]
    p = np.clip(p, floor, None)
    p /= np.trapezoid(p, zg)
    dlp = np.gradient(np.log(p), dz)
    psi = -dlp
    dpsi = np.gradient(psi, dz)
    # restrict to where the density is meaningful (numerical inversion rings
    # at the 1e-6 level far out)
    ok = p > 1e-9
    Ep = float(np.trapezoid((psi[ok] ** 2) * p[ok], zg[ok]))       # = E[psi'] = I
    Ex2 = float(np.trapezoid((zg[ok] ** 2) * dpsi[ok] * p[ok], zg[ok]))
    Exp = float(np.trapezoid(zg[ok] * psi[ok] * p[ok], zg[ok]))    # must be ~1
    return Ep, Ex2, Exp, zg, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--krad", type=float, default=1.0)
    ap.add_argument("--trim", type=float, default=5.0)
    ap.add_argument("--sigwin", type=float, default=5.0,
                    help="keep sigma_m/m < sigwin x median in the SUMMARY stats")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    args = Args(); args.krad = a.krad
    d = np.load(a.cache)
    x = np.load(a.aux)
    z = d["z"].astype(np.float64); sig = d["sigma"].astype(np.float64)
    assert np.array_equal(x["z"], z) and np.array_equal(x["sigma"], sig)
    mg = x["mgen"]
    eps = z * sig                                   # m - m_gen  [GeV]
    n = len(z)
    fhit, fms, fio = x["fhit"], x["fms"], x["fioni"]
    srel = sig / mg
    # truth-only features
    gp1 = x["gpt_p"] * np.cosh(x["geta_p"]); gp2 = x["gpt_m"] * np.cosh(x["geta_m"])
    gpmin = np.minimum(gp1, gp2); gpmax = np.maximum(gp1, gp2)
    ge1, ge2 = np.abs(x["geta_p"]), np.abs(x["geta_m"])
    gemin = np.minimum(ge1, ge2); gemax = np.maximum(ge1, ge2)
    nvs = x["nv_p"] + x["nv_m"]

    L = []; P = L.append
    P(f"# {a.label or os.path.basename(a.cache)}   n = {n}   krad = {a.krad}")
    P(f"# <sigma_m/m_gen> = {srel.mean():.5f}  median {np.median(srel):.5f}")
    P(f"# <f_hit> = {fhit.mean():.4f}  <f_ms> = {fms.mean():.4f}  "
      f"<f_ioni> = {fio.mean():.5f}")
    P(f"# closed form  a_m = <(2 f_hit + f_ms) sigma_m/m> = "
      f"{((2*fhit + fms) * srel).mean():+.5f}   "
      f"(a pure-MS model would give <sigma_m/m> = {srel.mean():.5f})")

    # THE SIGMA WINDOW.  A handful of candidates carry a pathological
    # Jpsi_sigmamass (up to 1e6 x m); they are harmless in the likelihood
    # (weight 1/sigma^2) but they own every naive mean and covariance.
    smed = np.median(srel)
    good = srel < a.sigwin * smed
    msk = (np.abs(z) < a.trim) & good
    P(f"# sigma window: sigma_m/m < {a.sigwin:g} x median = "
      f"{a.sigwin*smed:.4f} keeps {100.*good.mean():.3f} % "
      f"({int((~good).sum())} candidates dropped from the SUMMARY statistics "
      f"only -- the fits below use every candidate)")
    P(f"# windowed <sigma_m/m_gen> = {srel[good].mean():.5f}   "
      f"<f_hit> = {fhit[good].mean():.4f}  <f_ms> = {fms[good].mean():.4f}  "
      f"<f_ioni> = {fio[good].mean():.5f}")
    P(f"# windowed closed form a_m = <(2 f_hit + f_ms) sigma_m/m> = "
      f"{((2*fhit + fms) * srel)[good].mean():+.5f}")
    rng = np.random.default_rng(20260905)

    def bse(v, mm):
        idx = np.flatnonzero(mm)
        return float(np.array([v[idx[rng.integers(0, len(idx), len(idx))]].mean()
                               for _ in range(200)]).std(ddof=1))

    P("")
    P("## T1  the mass pull mean vs the raw relative mass bias, SAME candidates")
    P("| statistic | value | boot err |")
    P("|---|---|---|")
    A = float(z[msk].mean()); eA = bse(z, msk)
    R = float((eps / mg)[msk].mean()); eR = bse(eps / mg, msk)
    P(f"| <z_m> (pull units) | {A:+.5f} | {eA:.5f} |")
    P(f"| <m/m_gen> - 1 (relative) | {R:+.3e} | {eR:.3e} |")
    P(f"| <z_m> x <sigma_m/m> (if uncorrelated) | {A*srel[msk].mean():+.3e} | - |")
    P(f"| Cov(z_m, sigma_rel)/<sigma_rel>  INCLUSIVE | "
      f"{np.cov(z[msk], srel[msk])[0,1]/srel[msk].mean():+.5f} | - |")

    P("")
    P("## T2 / T3  truth-only sigma_bar, and a_m split into fluctuation vs kinematic")
    P("cells from GEN quantities only (leg momenta, |eta|, n valid hits): they "
      "cannot know the mass fluctuation.")
    P("| cells | ncell | n/cell | <z_m> | <z_bar> | boot | a_incl | a_fluct (in-cell) | a_kin = incl-fluct |")
    P("|---|---|---|---|---|---|---|---|---|")
    defs = [("gp(6q)^2 x |eta|(5q)^2", [qbin(np.log(gpmin), 6), qbin(np.log(gpmax), 6),
                                        qbin(gemin, 5), qbin(gemax, 5)]),
            ("gp(8q)^2 x |eta|(6q)^2", [qbin(np.log(gpmin), 8), qbin(np.log(gpmax), 8),
                                        qbin(gemin, 6), qbin(gemax, 6)]),
            ("gp(8q)^2 x |eta|(6q)^2 x nhit(4q)",
             [qbin(np.log(gpmin), 8), qbin(np.log(gpmax), 8),
              qbin(gemin, 6), qbin(gemax, 6), qbin(nvs, 4)]),
            ("gp(10q)^2 x |eta|(8q)^2", [qbin(np.log(gpmin), 10), qbin(np.log(gpmax), 10),
                                         qbin(gemin, 8), qbin(gemax, 8)])]
    a_incl = float(np.cov(z[msk], srel[msk])[0, 1] / srel[msk].mean())
    best = None
    for nm, cols in defs:
        cid = SP.cellid(*cols); nc = int(cid.max()) + 1
        # GEOMETRIC cell mean of a clipped sigma: ln sigma = ln sigma_bar + a x,
        # so the log scale is the natural one and it is immune to the outliers
        lsc_all = np.log(np.minimum(sig, a.sigwin * smed * mg))
        sb = np.exp(SP.cellmean(lsc_all, cid, nc, leave_one_out=True))
        zb = eps / sb
        Ab = float(zb[msk].mean()); eAb = bse(zb, msk)
        ls = np.log(sig)
        lsc = ls - SP.cellmean(ls, cid, nc)
        yc = z - SP.cellmean(z, cid, nc)
        af = float((lsc[msk] * yc[msk]).sum() / (yc[msk] ** 2).sum())
        P(f"| {nm} | {nc} | {n/nc:.0f} | {A:+.5f} | {Ab:+.5f} | {eAb:.5f} "
          f"| {a_incl:+.5f} | {af:+.5f} | {a_incl-af:+.5f} |")
        if best is None:
            best = (af, sb, zb, cid, nc)
    afluct, sbar, zbar, cid, nc = best

    P("")
    P("## T3b  measured a_fluct against the closed form, in bins of sigma_m/m")
    P("| sigma_m/m bin | n | a_fluct | closed form (2f_hit+f_ms) sigma/m | ratio |")
    P("|---|---|---|---|---|")
    nb = 5
    e = np.quantile(srel, np.linspace(0, 1, nb + 1))
    b = np.clip(np.digitize(srel, e[1:-1]), 0, nb - 1)
    ls = np.log(sig)
    lsc = ls - SP.cellmean(ls, cid, nc)
    yc = z - SP.cellmean(z, cid, nc)
    for k in range(nb):
        mm = (b == k) & msk
        af = float((lsc[mm] * yc[mm]).sum() / (yc[mm] ** 2).sum())
        cf = float(((2 * fhit + fms) * srel)[mm].mean())
        P(f"| {e[k]:.4f}..{e[k+1]:.4f} | {int(mm.sum())} | {af:+.5f} | {cf:+.5f} "
          f"| {af/cf:.3f} |")

    P("")
    P("## T4  the estimator factor F of the mass likelihood")
    phib = pooled_phi(d, args)
    I, Ex2, Exp, zg, p = estimator_factor(phib)
    F = (1. + Ex2) / I
    P(f"| quantity | value |")
    P(f"|---|---|")
    P(f"| E[psi'] = Fisher information I of the standardized model | {I:.4f} |")
    P(f"| E[x psi] (must be 1; numerical control) | {Exp:.4f} |")
    P(f"| E[x^2 psi'] | {Ex2:.4f} |")
    P(f"| **F = (1 + E[x^2 psi'])/E[psi']** (Gaussian model: 2) | **{F:.4f}** |")
    w = 1. / np.maximum(sbar, 1e-12)
    sbeff = float(w[good].sum() / (w[good] ** 2).sum())
    P(f"| sigma_bar_eff = sum(1/sbar)/sum(1/sbar^2) [GeV] | {sbeff:.5f} |")
    P(f"| sigma_bar_eff / M | {sbeff/MJPSI:.5f} |")
    P(f"| **predicted alpha bias = -a_fluct F sigma_bar/M** | "
      f"**{-afluct*F*sbeff/MJPSI*1e3:+.4f} e-3** |")
    P(f"| naive Gaussian (F = 2) | {-afluct*2.*sbeff/MJPSI*1e3:+.4f} e-3 |")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write("\n".join(L) + "\n")
        np.savez_compressed(a.out.replace(".txt", ".npz"),
                            sbar=sbar, zbar=zbar, afluct=afluct,
                            a_incl=a_incl, F=F, I=I, Ex2=Ex2,
                            phib=phib, srel=srel,
                            aclosed=(2 * fhit + fms) * srel)
    print("\n".join(L))


if __name__ == "__main__":
    main()
