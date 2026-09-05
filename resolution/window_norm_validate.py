#!/usr/bin/env python3
"""Gates for the window-truncated mass likelihood (`cf_masslik_fit --window-norm`).

An INDEPENDENT numpy implementation of the same three objects -- the model
density p_i(m), its Gil-Pelaez CDF F_i(m), and the accepted-window mass
P_i = F_i(hi) - F_i(lo) -- built straight from the pairs cache, so that the
TensorFlow graph is checked against something that shares no code with it.

GATES
  1. CDF vs DIRECT INTEGRATION.  Int_lo^hi p_i(m) dm on a fine Simpson grid
     against F_i(hi) - F_i(lo) from the 1/t transform, per candidate.
     Target |diff| <= 1e-6.
  2. QUADRATURE CONVERGENCE.  The CDF integrand carries a 1/t factor, so the
     448-point TG grid has to be shown to be enough: the same P_i on TG and on
     a x4 refinement (cubic spline of the cached exponents, phi_K evaluated
     EXACTLY at the refined t from the kernel samples).
  3. THE SIZE OF THE EFFECT.  The distribution of the model probability
     OUTSIDE the window, 1 - P_i, over the sample and against sigma_i -- i.e.
     how much a window normalisation can possibly move the fit.
  4. TF AGREEMENT (optional, --tf): the same P_i from MassNLL._winnorm.

usage:
  python3 window_norm_validate.py --pairs-cache ... --kernel-cache ... \
      --n 100 --window-lo 2.7469 --window-hi 3.4469
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_masslik_fit import load_inputs, MJPSI            # noqa: E402


def model_S(inp, i, k):
    """(Sre, Sim) on the cached TG grid for candidates `i` and scales `k`."""
    TG = inp["TG"]
    khit, kms, kioni, krad = k
    Sre = khit * (-0.5 * inp["vgf"][i, None] * TG[None, :] ** 2) \
        + kms * inp["Sms"][i].astype(np.float64)
    if not inp["folded"]:
        Sre = Sre + kioni * inp["Sio_re"][i].astype(np.float64)
    Sim = kioni * inp["Sio_im"][i].astype(np.float64)
    if inp["has_rad"]:
        Sre = Sre + krad * inp["Srad_re"][i].astype(np.float64)
        Sim = Sim + krad * inp["Srad_im"][i].astype(np.float64)
    return Sre, Sim


def dens(m, sig, Sre, Sim, pKre, pKim, TG):
    """p(m) for one candidate on a vector of masses (m already m - M(1+alpha))."""
    tgi = TG / sig
    psi = Sim[None, :] - tgi[None, :] * np.asarray(m)[:, None]
    integ = np.exp(Sre)[None, :] * (pKre[None, :] * np.cos(psi)
                                    - pKim[None, :] * np.sin(psi))
    return np.trapezoid(integ, TG, axis=1) / (np.pi * sig)


def winmass(xlo, xhi, sig, Sre, Sim, pKre, pKim, TG):
    """F(hi) - F(lo) by the Gil-Pelaez DIFFERENCE (see MassNLL._winnorm)."""
    tgi = TG / sig
    A = np.exp(Sre) * pKre
    B = np.exp(Sre) * pKim
    plo = Sim - tgi * xlo
    phi_ = Sim - tgi * xhi
    num = (A * np.sin(plo) + B * np.cos(plo)) - (A * np.sin(phi_)
                                                 + B * np.cos(phi_))
    D = np.empty_like(num)
    D[0] = (xhi - xlo) / sig            # exact removable-singularity limit
    D[1:] = num[1:] / TG[1:]
    return np.trapezoid(D, TG) / np.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-cache", required=True)
    p.add_argument("--kernel-cache", required=True)
    p.add_argument("--n", type=int, default=100, help="candidates for gate 1/2")
    p.add_argument("--nbig", type=int, default=20000,
                   help="candidates for the 1 - P distribution (gate 3)")
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--window-lo", type=float, default=MJPSI - 0.35)
    p.add_argument("--window-hi", type=float, default=MJPSI + 0.35)
    p.add_argument("--alpha", type=float, default=0.25e-3)
    p.add_argument("--k", type=float, nargs=4, default=(1., 1., 1., 1.),
                   metavar=("KHIT", "KMS", "KIONI", "KRAD"))
    p.add_argument("--nfine", type=int, default=40001,
                   help="Simpson points for the direct mass integration")
    p.add_argument("--refine", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--tf", action="store_true", help="also check MassNLL")
    a = p.parse_args()

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    inp = load_inputs(a.pairs_cache, a.kernel_cache, log=log)
    TG = inp["TG"]
    n = inp["n"]
    rng = np.random.default_rng(a.seed)
    # SHUFFLE FIRST: the cache is file-ordered
    idx = np.sort(rng.choice(n, min(a.n, n), replace=False))
    xlo = a.window_lo - MJPSI * (1. + a.alpha)
    xhi = a.window_hi - MJPSI * (1. + a.alpha)
    log(f"\nwindow [{a.window_lo:.4f}, {a.window_hi:.4f}] GeV, alpha={a.alpha:.3e}"
        f" -> x in [{xlo:+.4f}, {xhi:+.4f}] GeV;  k = {tuple(a.k)}")

    # ---------------- gate 1: CDF vs direct integration ----------------
    Sre, Sim = model_S(inp, idx, a.k)
    sig = inp["sig"][idx]
    pKre, pKim = inp["phiK_re"][idx], inp["phiK_im"][idx]
    P = np.empty(len(idx))
    Pdir = np.empty(len(idx))
    nf = a.nfine if a.nfine % 2 else a.nfine + 1
    mg = np.linspace(xlo, xhi, nf)
    w = np.ones(nf)
    w[1:-1:2] = 4.
    w[2:-1:2] = 2.
    w *= (mg[1] - mg[0]) / 3.
    for j in range(len(idx)):
        P[j] = winmass(xlo, xhi, sig[j], Sre[j], Sim[j], pKre[j], pKim[j], TG)
        Pdir[j] = float(w @ dens(mg, sig[j], Sre[j], Sim[j], pKre[j], pKim[j], TG))
    d = P - Pdir
    log(f"\n== gate 1: F(hi)-F(lo) vs Simpson({nf}) integral of p(m), "
        f"{len(idx)} candidates ==")
    log(f"   max |diff| {np.abs(d).max():.3e}   rms {d.std():.3e}   "
        f"mean {d.mean():+.3e}")
    log(f"   P range [{P.min():.6f}, {P.max():.6f}]   "
        f"1-P: mean {1-P.mean():.3e}  max {1-P.min():.3e}")
    g1 = np.abs(d).max() <= 1e-6
    log(f"   GATE 1 {'PASS' if g1 else 'FAIL'} (target 1e-6)")

    # ---------------- gate 2: t-grid convergence -----------------------
    from scipy.interpolate import CubicSpline
    dm = np.load(a.kernel_cache)["dm"].astype(np.float64)
    TGf = np.linspace(TG[0], TG[-1], (len(TG) - 1) * a.refine + 1)
    Pf = np.empty(len(idx))
    for j in range(len(idx)):
        sref = CubicSpline(TG, Sre[j])(TGf)
        simf = CubicSpline(TG, Sim[j])(TGf)
        # phi_K EXACTLY at the refined absolute t (no interpolation)
        tf_ = TGf / sig[j]
        blk = 4096
        ph = np.concatenate([np.mean(np.exp(1j * np.outer(tf_[q:q + blk], dm)),
                                     axis=1) for q in range(0, len(tf_), blk)])
        Pf[j] = winmass(xlo, xhi, sig[j], sref, simf, ph.real, ph.imag, TGf)
    dq = P - Pf
    log(f"\n== gate 2: TG (448) vs x{a.refine} refinement "
        f"({len(TGf)} pts, exact phi_K) ==")
    log(f"   max |dP| {np.abs(dq).max():.3e}   rms {dq.std():.3e}   "
        f"mean {dq.mean():+.3e}")
    g2 = np.abs(dq).max() <= 1e-6
    log(f"   GATE 2 {'PASS' if g2 else 'FAIL'} (target 1e-6)")

    # ---------------- gate 3: how big is the truncation ----------------
    ib = np.sort(rng.choice(n, min(a.nbig, n), replace=False))
    SreB, SimB = model_S(inp, ib, a.k)
    sB = inp["sig"][ib]
    pRb, pIb = inp["phiK_re"][ib], inp["phiK_im"][ib]
    PB = np.empty(len(ib))
    t0 = time.time()
    for j in range(len(ib)):
        PB[j] = winmass(xlo, xhi, sB[j], SreB[j], SimB[j], pRb[j], pIb[j], TG)
    out = 1. - PB
    log(f"\n== gate 3: model probability OUTSIDE the window, {len(ib)} "
        f"candidates ({time.time()-t0:.1f} s) ==")
    log(f"   mean {out.mean():.3e}  median {np.median(out):.3e}  "
        f"q90 {np.quantile(out,.9):.3e}  q99 {np.quantile(out,.99):.3e}  "
        f"max {out.max():.3e}")
    log(f"   sum over the sample of -log P (the NLL shift at these k): "
        f"{-np.log(PB).sum():.3f}  (n={len(ib)}, scaled to n={n}: "
        f"{-np.log(PB).sum()*n/len(ib):.1f})")
    qs = np.quantile(sB, [0., .25, .5, .75, .9, 1.])
    log(f"   {'sigma bin':>18s} {'n':>6s} {'<1-P>':>12s} {'max 1-P':>12s}")
    for lo_, hi_ in zip(qs[:-1], qs[1:]):
        m_ = (sB >= lo_) & (sB < hi_ + (1e-12 if hi_ == qs[-1] else 0))
        if m_.sum():
            log(f"   [{lo_:.4f},{hi_:.4f}] {int(m_.sum()):6d} "
                f"{out[m_].mean():12.3e} {out[m_].max():12.3e}")

    # ---------------- gate 4: the TF graph -----------------------------
    if a.tf:
        import tensorflow as tf                                # noqa: F401
        from cf_masslik_fit import MassNLL
        sub = dict(inp)
        sub["n"] = len(idx)
        for kk in ("sig", "mobs", "vgf", "Sms", "Sio_re", "Sio_im",
                   "Srad_re", "Srad_im", "phiK_re", "phiK_im"):
            if sub.get(kk) is not None and np.ndim(sub[kk]) and \
                    np.shape(sub[kk])[0] == n:
                sub[kk] = sub[kk][idx]
        obj = MassNLL(sub, model="families", chunk=len(idx), log=lambda *x: None,
                      window=(a.window_lo, a.window_hi), krad0=a.k[3])
        th = obj._theta([a.alpha / 1e-3, a.k[0], a.k[1], a.k[2]])
        A = obj.tf
        i0 = A.constant(0, A.int32)
        alpha, khit, kms, kioni, krad, _ = obj._unpack(th)
        sigt = obj._slice(obj.sig, i0, False)
        Sms = A.cast(obj.Sms, obj.rdt)
        TGt = obj.TG
        tgi = TGt[None, :] / sigt[:, None]
        Sre_t = khit * (-0.5 * obj.vgf[:, None] * TGt[None, :] ** 2) + kms * Sms
        if not obj.folded:
            Sre_t = Sre_t + kioni * A.cast(obj.Sio_re, obj.rdt)
        Sim_t = kioni * A.cast(obj.Sio_im, obj.rdt)
        if obj.has_rad:
            Sre_t = Sre_t + krad * A.cast(obj.Srad_re, obj.rdt)
            Sim_t = Sim_t + krad * A.cast(obj.Srad_im, obj.rdt)
        e = A.exp(Sre_t)
        Ptf = obj._winnorm(e * obj.pKre, e * obj.pKim, Sim_t, tgi, sigt,
                           alpha).numpy()
        dt = Ptf - P
        log(f"\n== gate 4: MassNLL._winnorm vs this numpy reference ==")
        log(f"   max |dP| {np.abs(dt).max():.3e}  rms {dt.std():.3e}")
        log(f"   GATE 4 {'PASS' if np.abs(dt).max() <= 1e-12 else 'FAIL'} "
            f"(target 1e-12, same formula in two implementations)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
