#!/usr/bin/env python3
"""Numpy re-implementation of `cf_masslik_fit.MassNLL` (families, window off).

Identical objective, written with an ANALYTIC gradient so the 4-parameter fit
runs without the TF/singularity stack (which cannot be entered from inside the
mfs venv).  Validated against the published numbers in
`runs/rad260903x/masslikfit_summary.txt` by `nll_impact.py --validate`.

    Sre = k_hit (-1/2 vgf t^2) + k_ms Sms + k_ioni Sio_re + k_rad Srad_re
    Sim = k_ioni Sio_im + k_rad Srad_im
    psi = Sim - (TG/sigma) (mobs - M alpha)
    L_i = trapz( e^{Sre} (Re phiK cos psi - Im phiK sin psi) , TG ) / (pi sigma)
    NLL = -sum log[ (1-f) softplus_floor(L_i) + f/0.7 ]
"""
import os
import numpy as np

MJPSI = 3.0969
FBKG = 0.005
MWIN = 0.7
ASCALE = 1e-3
FLOOR_S = 1e-9
PARNAMES = ["alpha[1e-3]", "k_hit", "k_ms", "k_ioni"]

_M = None          # the Model the pool workers see (set before fork)


def _softplus_floor(x, s=FLOOR_S):
    """s*log(1+exp(x/s)) == max(x, 0) to ~s, differentiable everywhere."""
    z = x / s
    out = np.empty_like(z)
    big = z > 30.0
    out[big] = z[big]
    nb = ~big
    out[nb] = np.log1p(np.exp(z[nb]))
    return s * out, 1.0 / (1.0 + np.exp(-z))     # value, d/dx


class Model:
    """Holds the (n, 448) blocks; evaluates NLL and gradient in row chunks."""

    def __init__(self, sig, mobs, vgf, Sms, Sio_re, Sio_im, Srad_re, Srad_im,
                 pKre, pKim, TG, krad=1.0, fbkg=FBKG, chunk=8192, nproc=1):
        self.sig, self.mobs, self.vgf = sig, mobs, vgf
        self.Sms, self.Sio_re, self.Sio_im = Sms, Sio_re, Sio_im
        self.Srad_re, self.Srad_im = Srad_re, Srad_im
        self.has_rad = Srad_re is not None
        self.pKre, self.pKim = pKre, pKim
        self.TG = TG
        self.dTG = np.diff(TG)
        self.krad, self.fbkg = krad, fbkg
        self.n = len(sig)
        self.chunk = chunk
        self.nproc = nproc
        self.nfev = 0

    # ---------------- one row block ----------------
    def _block(self, lo, hi, th, want_grad=True):
        alpha, khit, kms, kioni = th[0] * ASCALE, th[1], th[2], th[3]
        sl = slice(lo, hi)
        TG = self.TG
        sig = self.sig[sl]
        tgi = TG[None, :] / sig[:, None]
        gh = -0.5 * self.vgf[sl][:, None] * TG[None, :] ** 2
        Sms = self.Sms[sl].astype(np.float64)
        Sio_re = self.Sio_re[sl].astype(np.float64)
        Sio_im = self.Sio_im[sl].astype(np.float64)
        Sre = khit * gh + kms * Sms + kioni * Sio_re
        Sim = kioni * Sio_im
        if self.has_rad:
            Sradre = self.Srad_re[sl].astype(np.float64)
            Sradim = self.Srad_im[sl].astype(np.float64)
            Sre = Sre + self.krad * Sradre
            Sim = Sim + self.krad * Sradim
        delta = self.mobs[sl] - MJPSI * alpha
        psi = Sim - tgi * delta[:, None]
        eS = np.exp(Sre)
        c, s = np.cos(psi), np.sin(psi)
        pKre, pKim = self.pKre[sl], self.pKim[sl]
        A = eS * (pKre * c - pKim * s)
        norm = np.pi * sig
        Li = self._trapz(A) / norm
        val, dsp = _softplus_floor(Li)
        Lm = (1.0 - self.fbkg) * val + self.fbkg / MWIN
        nll = -np.log(Lm).sum()
        if not want_grad:
            return nll, None
        B = eS * (-pKre * s - pKim * c)
        dL = np.empty((hi - lo, 4))
        dL[:, 0] = self._trapz(B * (tgi * (MJPSI * ASCALE))) / norm
        dL[:, 1] = self._trapz(A * gh) / norm
        dL[:, 2] = self._trapz(A * Sms) / norm
        dL[:, 3] = self._trapz(A * Sio_re + B * Sio_im) / norm
        w = -(1.0 - self.fbkg) * dsp / Lm
        return nll, (w[:, None] * dL).sum(axis=0)

    def grad_rows(self, th, lo=0, hi=None):
        """Per-candidate gradient contribution  w_i dL_i/dtheta  (the terms
        whose SUM is the gradient).  Used to split a compression-induced shift
        of the minimum into a systematic part (the mean, which survives at any
        n) and a statistical part (the spread, which dies as 1/sqrt(n))."""
        th = np.asarray(th, np.float64)
        hi = self.n if hi is None else hi
        out = []
        for a in range(lo, hi, self.chunk):
            b = min(a + self.chunk, hi)
            alpha, khit, kms, kioni = th[0] * ASCALE, th[1], th[2], th[3]
            sl = slice(a, b)
            TG = self.TG
            sig = self.sig[sl]
            tgi = TG[None, :] / sig[:, None]
            gh = -0.5 * self.vgf[sl][:, None] * TG[None, :] ** 2
            Sms = self.Sms[sl].astype(np.float64)
            Sio_re = self.Sio_re[sl].astype(np.float64)
            Sio_im = self.Sio_im[sl].astype(np.float64)
            Sre = khit * gh + kms * Sms + kioni * Sio_re
            Sim = kioni * Sio_im
            if self.has_rad:
                Sre = Sre + self.krad * self.Srad_re[sl].astype(np.float64)
                Sim = Sim + self.krad * self.Srad_im[sl].astype(np.float64)
            delta = self.mobs[sl] - MJPSI * alpha
            psi = Sim - tgi * delta[:, None]
            eS = np.exp(Sre)
            c, sn = np.cos(psi), np.sin(psi)
            pKre, pKim = self.pKre[sl], self.pKim[sl]
            A = eS * (pKre * c - pKim * sn)
            B = eS * (-pKre * sn - pKim * c)
            norm = np.pi * sig
            Li = self._trapz(A) / norm
            val, dsp = _softplus_floor(Li)
            Lm = (1.0 - self.fbkg) * val + self.fbkg / MWIN
            dL = np.empty((b - a, 4))
            dL[:, 0] = self._trapz(B * (tgi * (MJPSI * ASCALE))) / norm
            dL[:, 1] = self._trapz(A * gh) / norm
            dL[:, 2] = self._trapz(A * Sms) / norm
            dL[:, 3] = self._trapz(A * Sio_re + B * Sio_im) / norm
            w = -(1.0 - self.fbkg) * dsp / Lm
            out.append(w[:, None] * dL)
        return np.concatenate(out, axis=0)

    def _trapz(self, y):
        return (self.dTG[None, :] * (y[:, 1:] + y[:, :-1])).sum(axis=1) * 0.5

    # ---------------- driver ----------------
    def nll_grad(self, th, want_grad=True):
        th = np.asarray(th, np.float64)
        bounds = [(lo, min(lo + self.chunk, self.n))
                  for lo in range(0, self.n, self.chunk)]
        if self.nproc > 1:
            global _M
            _M = self
            import multiprocessing as mp
            with mp.get_context("fork").Pool(self.nproc) as p:
                out = p.map(_pw, [(lo, hi, th, want_grad) for lo, hi in bounds])
        else:
            out = [self._block(lo, hi, th, want_grad) for lo, hi in bounds]
        f = sum(o[0] for o in out)
        self.nfev += 1
        if not want_grad:
            return f, None
        g = np.sum([o[1] for o in out], axis=0)
        return f, g

    def fit(self, th0=(0.25, 0.94, 1.01, 0.69), tol=1e-10, log=print):
        from scipy.optimize import minimize
        r = minimize(lambda t: self.nll_grad(t), np.asarray(th0, np.float64),
                     jac=True, method="BFGS",
                     options=dict(gtol=1e-6, maxiter=200))
        # errors from a finite-difference Hessian of the ANALYTIC gradient
        H = self.hess(r.x)
        cov = np.linalg.inv(H)
        return dict(x=r.x, nll=r.fun, err=np.sqrt(np.diag(cov)), H=H, cov=cov,
                    nit=r.nit, gnorm=np.abs(r.jac).max(), success=r.success,
                    message=str(r.message))

    def hess(self, x, h=1e-4):
        H = np.zeros((4, 4))
        for j in range(4):
            e = np.zeros(4); e[j] = h
            gp = self.nll_grad(x + e)[1]
            gm = self.nll_grad(x - e)[1]
            H[:, j] = (gp - gm) / (2 * h)
        return 0.5 * (H + H.T)


def _pw(a):
    return _M._block(*a)


# ------------------------------------------------------------------ I/O ---
RUNS = "/work/submit/david_w/ZMass/calibration_studies/resolution/runs"
PHIK_TAB = {
    "jpsigun": f"{RUNS}/masslikfit_phiKtab_cf_masskernel_jpsigun_ul16_260903x"
               f"_m0_300243_1.635557262348e+03.npz",
    "btojpsix": f"{RUNS}/masslikfit_phiKtab_cf_masskernel_btojpsix_v3_260903x"
                f"_m0_128687_1.271935440271e+03.npz",
}


def phik(tag, TG, sig):
    z = np.load(PHIK_TAB[tag])
    tabs, tab = z["tabs"], z["phiK_tab"]
    tgi = TG[None, :] / sig[:, None]
    return (np.interp(tgi, tabs, tab.real), np.interp(tgi, tabs, tab.imag))
