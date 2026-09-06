#!/usr/bin/env python3
"""The unbinned CVH mass likelihood in numpy, with a SIGMA SOURCE switch.

Same objective as `cf_masslik_fit.py` (which is the same as
`cf_mass_likelihood.demo`), re-implemented in float64 numpy so that the
per-candidate resolution SCALE can be swapped without touching the fit code:

    L_i(alpha) = 1/(pi s_i) Int_0^T e^{Sre_i(t)}
                 [ phiK_re(t/s_i) cos psi - phiK_im(t/s_i) sin psi ] dt ,
    psi = Sim_i(t) - (t/s_i) (m_i - M(1+alpha)) ,
    NLL = - sum_i log[ (1-f) max(L_i, 0) + f/0.7 ] ,  f = 0.005 .

`s_i` is the SCALE of the model in absolute mass units.  The published fit uses
s_i = sigma_i = `Jpsi_sigmamass`, the two-track fit's own error assembled at the
FITTED state -- which is a function of the fitted mass, so the model's width is
conditioned on the very fluctuation being measured.  `--sigma-source bar` swaps
in a truth-referenced sigma_bar (from `oddmoment/mass_pull.py`), which cannot
see the fluctuation.  The residual m_i - M is PHYSICAL and is always built from
the original sigma (mobs = z sigma + (mgen - M)); only the model scale changes.

Subcommands
  fit    minimise the NLL in alpha at fixed k (Brent + parabolic error)
  scan   NLL on an alpha grid
  toy    resample z, sigma from the per-candidate model with an injected
         self-consistency coefficient a, write a synthetic pairs cache
"""
import argparse
import os
import sys
import time

import numpy as np

MJPSI = 3.0969
FBKG = 0.005
WIN = 0.7


# --------------------------------------------------------------- likelihood
class MassLik:
    def __init__(self, pairs, kernel, sbar=None, krad=1.0, maxn=0,
                 chunk=32768, log=print, corrected=False, a_scale=1.0,
                 a_vec=None, override=None):
        d = np.load(pairs)
        self.TG = np.asarray(d["tgrid"], dtype=np.float64)
        z = d["z"].astype(np.float64)
        sig = d["sigma"].astype(np.float64)
        if override is not None:
            o = np.load(override)
            z = o["z"].astype(np.float64)
            sig = o["sigma"].astype(np.float64)
            log(f"z/sigma overridden from {override}")
        mgen = d["eta"].astype(np.float64)          # `eta` IS the gen mass
        vgf = d["vgf"].astype(np.float64)
        n = len(z)
        if maxn and maxn < n:
            n = maxn
        sl = slice(0, n)
        self.n = n
        self.z, self.sig, self.mgen, self.vgf = z[sl], sig[sl], mgen[sl], vgf[sl]
        self.Sms = d["Sms"][sl]
        self.Sio_re, self.Sio_im = d["Sio_re"][sl], d["Sio_im"][sl]
        rad_model = int(d["rad_model"]) if "rad_model" in d.files else 0
        self.has_rad = ("Srad_re" in d.files) and rad_model == 1
        self.Srad_re = d["Srad_re"][sl] if self.has_rad else None
        self.Srad_im = d["Srad_im"][sl] if self.has_rad else None
        self.krad = krad
        self.chunk = chunk
        self.log = log
        # PHYSICAL residual, always from the original sigma
        self.mobs = self.z * self.sig + (self.mgen - MJPSI)
        self.s = self.sig.copy() if sbar is None else np.asarray(sbar,
                                                                 np.float64)[sl]
        # THE TRUTH-FREE CORRECTION.  sigma_i = sigma_bar_i + a_i (m_i - mu_i)
        # is the definition of the self-consistency coefficient, so
        #     sigma_bar_i(alpha) = sigma_i - a_i (m_i - M(1+alpha))
        # recovers the unconditional resolution from OBSERVED quantities alone.
        # a_i = (1 + f_hit - f_ioni) sigma_i/m, and f_hit is the cached `vgf`
        # (f_ioni is 1e-3 and is dropped unless supplied).
        self.corrected = bool(corrected)
        if a_vec is not None:
            self.a = np.asarray(a_vec, np.float64)[sl] * a_scale
        else:
            self.a = (1. + self.vgf) * self.sig / self.mgen * a_scale
        if self.corrected:
            log(f"corrected sigma: <a_i> = {np.median(self.a):.5f} (median), "
                f"a_scale = {a_scale}")
        # FSR kernel CF, tabulated on absolute t
        k = np.load(kernel)
        dm = k["dm"]
        tmax = self.TG[-1] / self.s.min()
        self.tabs = np.linspace(0., tmax, 8192)
        t0 = time.time()
        blk = 1024
        self.phiK = np.concatenate(
            [np.mean(np.exp(1j * np.outer(self.tabs[i:i + blk], dm)), axis=1)
             for i in range(0, len(self.tabs), blk)])
        log(f"phiK table ({len(dm)} kernel samples, tmax {tmax:.1f}) in "
            f"{time.time()-t0:.1f} s")

    def _scale(self, sl, alpha):
        """The model's absolute scale for this alpha."""
        if not self.corrected:
            return self.s[sl]
        delta = self.mobs[sl] - MJPSI * alpha
        s = self.s[sl] - self.a[sl] * delta
        # a_i |delta| can exceed sigma on the pathological tail; keep the scale
        # physical (this touches <0.1 % of candidates and only in the far tail)
        return np.maximum(s, 0.2 * self.s[sl])

    def _kcache(self, sl, khit, kms, kioni, krad):
        """e^{Sre} and Sim for a fixed k; independent of alpha."""
        key = (sl.start, khit, kms, kioni, krad)
        if getattr(self, "_kk", None) == key:
            return self._kv
        TG = self.TG
        Sre = khit * (-0.5 * self.vgf[sl][:, None] * TG[None, :] ** 2) \
            + kms * self.Sms[sl].astype(np.float64) \
            + kioni * self.Sio_re[sl].astype(np.float64)
        Sim = kioni * self.Sio_im[sl].astype(np.float64)
        if self.has_rad:
            Sre = Sre + krad * self.Srad_re[sl].astype(np.float64)
            Sim = Sim + krad * self.Srad_im[sl].astype(np.float64)
        self._kk, self._kv = key, (np.exp(Sre), Sim)
        return self._kv

    def _chunk_L(self, sl, alpha, khit, kms, kioni, krad):
        TG = self.TG
        s = self._scale(sl, alpha)
        eSre, Sim = self._kcache(sl, khit, kms, kioni, krad)
        if self.corrected or getattr(self, "_pk", (None,))[0] != sl.start:
            tgi = TG[None, :] / s[:, None]
            pKre = np.interp(tgi, self.tabs, self.phiK.real)
            pKim = np.interp(tgi, self.tabs, self.phiK.imag)
            if not self.corrected:
                self._pk = (sl.start, tgi, pKre, pKim)
        else:
            _, tgi, pKre, pKim = self._pk
        delta = self.mobs[sl] - MJPSI * alpha
        psi = Sim - tgi * delta[:, None]
        integ = eSre * (pKre * np.cos(psi) - pKim * np.sin(psi))
        dT = np.diff(TG)
        Li = np.sum(dT[None, :] * (integ[:, 1:] + integ[:, :-1]) * 0.5, axis=1) \
            / (np.pi * s)
        return Li

    def logL(self, alpha, khit=1., kms=1., kioni=1., krad=None, fbkg=FBKG):
        krad = self.krad if krad is None else krad
        out = np.empty(self.n)
        for lo in range(0, self.n, self.chunk):
            sl = slice(lo, min(lo + self.chunk, self.n))
            Li = self._chunk_L(sl, alpha, khit, kms, kioni, krad)
            out[sl] = np.log((1. - fbkg) * np.clip(Li, 0., None) + fbkg / WIN)
        return out

    def nll(self, alpha, **kw):
        return -float(self.logL(alpha, **kw).sum())

    def fit_alpha(self, a0=0.0, halfwidth=2e-3, **kw):
        """Parabolic minimisation on a refined grid; error from the curvature."""
        lo, hi = a0 - halfwidth, a0 + halfwidth
        for _ in range(3):
            g = np.linspace(lo, hi, 7)
            v = np.array([self.nll(x, **kw) for x in g])
            i = int(np.argmin(v))
            i = min(max(i, 1), len(g) - 2)
            c = np.polyfit(g[i - 1:i + 2] - g[i], v[i - 1:i + 2], 2)
            ahat = g[i] - 0.5 * c[1] / c[0]
            step = (g[1] - g[0])
            lo, hi = ahat - step, ahat + step
        err = float(np.sqrt(0.5 / c[0])) if c[0] > 0 else np.nan
        return float(ahat), err, float(np.polyval(c, ahat - g[i]))


# ------------------------------------------------------------------- toy
def sample_from_model(pairs, a_inject, seed=1234, nz=801, zmax=10.,
                      chunk=8192, krad=1.0, log=print):
    """Draw x_i from each candidate's OWN model density, then apply the map.

    Returns (x, z_obs, sigma_obs) with sigma_obs = sigma_bar (1 + a x) and
    z_obs = x/(1 + a x); sigma_bar is the cache's sigma (taken as the truth for
    the toy, so the injected a is exactly known)."""
    d = np.load(pairs)
    TG = np.asarray(d["tgrid"], np.float64)
    n = len(d["z"])
    sbar = d["sigma"].astype(np.float64)
    vgf = d["vgf"].astype(np.float64)
    rng = np.random.default_rng(seed)
    zg = np.linspace(-zmax, zmax, nz)
    w = np.gradient(TG); w[0] *= 0.5; w[-1] *= 0.5
    Kc = (np.cos(TG[None, :] * zg[:, None]) * w[None, :] / np.pi).T   # (nt, nz)
    Ks = (np.sin(TG[None, :] * zg[:, None]) * w[None, :] / np.pi).T
    x = np.empty(n)
    has_rad = ("Srad_re" in d.files) and int(d["rad_model"]) == 1
    for lo in range(0, n, chunk):
        sl = slice(lo, min(lo + chunk, n))
        S = (-0.5 * vgf[sl][:, None] * TG[None, :] ** 2
             + d["Sms"][sl].astype(np.float64)
             + d["Sio_re"][sl].astype(np.float64)
             + 1j * d["Sio_im"][sl].astype(np.float64))
        if has_rad:
            S = S + krad * (d["Srad_re"][sl].astype(np.float64)
                            + 1j * d["Srad_im"][sl].astype(np.float64))
        phi = np.exp(S)
        p = phi.real @ Kc + phi.imag @ Ks
        np.clip(p, 0., None, out=p)
        c = np.cumsum(p, axis=1)
        c /= c[:, -1:]
        u = rng.random(p.shape[0])
        # per-row inverse CDF by searchsorted
        idx = np.array([np.searchsorted(c[i], u[i]) for i in range(p.shape[0])])
        idx = np.clip(idx, 1, nz - 1)
        c0 = c[np.arange(len(idx)), idx - 1]
        c1 = c[np.arange(len(idx)), idx]
        frac = (u - c0) / np.maximum(c1 - c0, 1e-300)
        x[sl] = zg[idx - 1] + frac * (zg[idx] - zg[idx - 1])
        if lo % (20 * chunk) == 0:
            log(f"  sampled {lo}/{n}")
    sig_obs = sbar * (1. + a_inject * x)
    z_obs = x / (1. + a_inject * x)
    return x, z_obs, sig_obs, sbar


def write_toy(pairs, out, a_inject, seed=1234, log=print):
    """Only z, sigma and the truth are written: everything else (the
    exponents, vgf, the gen mass) is read from the parent cache at fit time
    through --override, so a toy costs 5 MB instead of 2.4 GB."""
    x, z_obs, sig_obs, sbar = sample_from_model(pairs, a_inject, seed=seed,
                                                log=log)
    np.savez_compressed(out, z=z_obs, sigma=sig_obs, sbar=sbar, x=x,
                        a_inject=a_inject)
    log(f"-> {out}  (a_inject = {a_inject:+.5f}, <x> = {x.mean():+.5f}, "
        f"<z_obs> = {z_obs.mean():+.5f}, predicted <z_obs> = {-a_inject:+.5f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "scan", "toy"])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sbar", default="", help="npz with key `sbar`")
    ap.add_argument("--sigma-source", choices=["fit", "bar", "corrected"],
                    default="fit")
    ap.add_argument("--a-scale", type=float, default=1.0)
    ap.add_argument("--a-vec", default="", help="npz with key `a` (per candidate)")
    ap.add_argument("--a-const", type=float, default=None,
                    help="use a CONSTANT a_i (the toy's injected value)")
    ap.add_argument("--override", default="",
                    help="npz with z, sigma replacing the cache's (toy)")
    ap.add_argument("--khit", type=float, default=1.0)
    ap.add_argument("--kms", type=float, default=1.0)
    ap.add_argument("--kioni", type=float, default=1.0)
    ap.add_argument("--krad", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=None,
                    help="single resolution scale: sets khit=kms=kioni=r")
    ap.add_argument("--maxn", type=int, default=0)
    ap.add_argument("--a-inject", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    if a.cmd == "toy":
        write_toy(a.pairs, a.out, a.a_inject, seed=a.seed)
        return

    sbar = None
    if a.sigma_source == "bar":
        assert a.sbar, "--sigma-source bar needs --sbar"
        sbar = np.load(a.sbar)["sbar"]
    av = np.load(a.a_vec)["a"] if a.a_vec else None
    if a.a_const is not None:
        d0 = np.load(a.pairs)
        av = np.full(len(d0["z"]), a.a_const)
    kw = dict(khit=a.khit, kms=a.kms, kioni=a.kioni, krad=a.krad)
    if a.r is not None:
        kw.update(khit=a.r, kms=a.r, kioni=a.r)
    L = MassLik(a.pairs, a.kernel, sbar=sbar, krad=a.krad, maxn=a.maxn,
                corrected=(a.sigma_source == "corrected"),
                a_scale=a.a_scale, a_vec=av,
                override=(a.override or None))
    t0 = time.time()
    if a.cmd == "scan":
        for al in np.linspace(-2e-3, 2e-3, 9):
            print(f"  alpha {al*1e3:+.4f}e-3   NLL {L.nll(al, **kw):.4f}")
    else:
        ah, eh, nll = L.fit_alpha(**kw)
        line = (f"{a.label or os.path.basename(a.pairs)} "
                f"[sigma={a.sigma_source}] alpha = {ah*1e3:+.5f} +- {eh*1e3:.5f} e-3"
                f"   NLL {nll:.4f}   n={L.n}   ({time.time()-t0:.0f} s)")
        print(line)
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
