#!/usr/bin/env python3
"""Generator-level closure of the Z/gamma* lineshape kernel.

Validates `rabbit.lineshapes.ZGammaLineshape` against the generator record of
the POWHEG-MiNNLO + Pythia8 + Photos DY sample the detector-level Z channel
will be fitted on: fit the *generated* mass spectrum -- no resolution anywhere
-- and check that the fitted (m_Z, Gamma_Z) come back at the numbers POWHEG was
run with (`data/generator_settings_*.md`).

Subcommands
-----------
``kernel``      build the multiplicative FSR kernel ``k(r)``, ``r = m_post/m_pre``
``acceptance``  fit ``A(m_pre) = P(selected | m_pre)`` as a Bernstein polynomial
``fit``         the closure fit itself

The likelihood
--------------
Fine-binned weighted maximum likelihood.  The events are histogrammed onto the
provider's *own* mass grid (``dm = 2.44 MeV`` by default, ~1/1000 of the Z
width), so

    -2 log L(theta) = -2 sum_b W_b [ log p_theta(m_b) - log Z_theta ] ,
    W_b = sum_{i in b} w_i ,   Z_theta = sum_{b in window} p_theta(m_b) dm ,

which is the unbinned weighted likelihood up to O(dm^2 (log p)'') ~ 4e-8 per
event.  ``Z_theta`` is the *truncation* term: the fit window is narrower than
the model's support, and leaving it out biases the mass (the same effect
``norm_window`` handles in the real likelihood).

MiNNLO weights are not +-1 (91.9 % at +2374.19, 7.8 % at -2374.19, a 0.3 % tail
out to 2.7e5, N_eff/N = 0.60), so the errors use the sandwich

    V = H^-1 J H^-1 ,   H = d^2(-log L) ,   J = sum_b (sum_{i in b} w_i^2) g_b g_b^T

with ``g_b`` the per-event score of bin ``b``.  With unit weights this collapses
to ``H^-1``.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

MZ_FIXED = 91.153509740726733       # POWHEG's constant-width Z mass
GZ_FIXED = 2.4932018986110700       # POWHEG's constant-width Z width
#: the seed of the acceptance toy's per-muon smearing.  One number, used by
#: every consumer, so the fit's selection and the kernel measured for it are
#: the same events.
SMEAR_SEED = 20260916

MZ_RUNNING = 91.1876                # the PDG input POWHEG converted from
GZ_RUNNING = 2.4941343245745466


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def clip_weights(w, factor=100.0):
    """Clip |w| at ``factor`` times the modal |w|.

    MiNNLO's unweighting occasionally fails and emits an event with |w| ~ 1e19;
    two such events in 12.5 M take N_eff from 8.5 M to 2. 99.7 % of the sample
    sits at exactly one value (+-2374.19 here), so the mode is a robust scale.
    """
    if not factor or factor <= 0:
        return w
    aw = np.abs(w)
    w0 = np.median(aw)
    w = np.clip(w, -factor * w0, factor * w0)
    # rescale to <w> = 1: the fit and the sandwich covariance are invariant,
    # but H^-1 then reads as the covariance the same sample would have with
    # unit weights, which is the useful reference for N_eff.
    return w * (len(w) / w.sum())


def load_gen(path, columns=None):
    d = np.load(path)
    return {k: d[k] for k in (columns or d.files) if k in d.files}


def fiducial(g, pt_cuts, eta_max, post=True, smear=None, seed=SMEAR_SEED):
    """Boolean mask: the two muons inside ``eta_max``, over the ``pT`` cuts.

    ``pt_cuts`` is ``(leading, trailing)``, or one number for the symmetric
    cut.  Which muon is the leading one is decided **after** FSR and after the
    smearing, so the condition is on ``max``/``min`` and the legs are never
    ordered before the losses are applied.

    ``post=True`` uses the post-FSR (status-1) muons, which is what a real
    selection cuts on; ``post=False`` uses the pre-FSR pair.

    ``smear`` is a `ptres.Resolution`: the cuts then act on ``pT (1 + r)`` with
    ``r`` drawn per muon at that muon's own ``(pT, eta)``, which is the toy the
    detector-level acceptance is benchmarked against.  The draw is seeded and
    the two legs are drawn in a fixed order, so every row of a fit suite and
    the MC-conditional kernel measured for it see the SAME selection.
    """
    sfx = "" if post else "_pre"
    c = np.atleast_1d(np.asarray(pt_cuts, float))
    c_lead, c_trail = ((c[0], c[0]) if c.size == 1
                       else (float(np.max(c)), float(np.min(c))))
    ok = np.ones(len(g["m_pre"]), bool)
    rng = np.random.default_rng(seed) if smear is not None else None
    pts = []
    for i in ("1", "2"):
        pt = np.asarray(g[f"pt{i}{sfx}"], float)
        eta = np.asarray(g[f"eta{i}{sfx}"], float)
        ok &= np.abs(eta) < eta_max
        if smear is not None:
            pt = pt * (1.0 + smear.sample(pt, eta, rng))
        pts.append(pt)
    return ok & (np.maximum(pts[0], pts[1]) > c_lead) \
        & (np.minimum(pts[0], pts[1]) > c_trail)


def load_resolution(path, mode="shape"):
    """A `ptres.Resolution`, or None."""
    if not path:
        return None
    import ptres
    return ptres.Resolution(path, mode=mode)


# --------------------------------------------------------------------------
# the FSR kernel
# --------------------------------------------------------------------------
def build_kernel(m_pre, m_post, w, sigma_cap=3.3e-4, u_fine=2e-5, u_max=None,
                 var_budget=None):
    """Discretise ``k(r)``, ``r = m_post / m_pre``, as ``(r_j, w_j)`` atoms.

    The fold ``p_post(m) = sum_j w_j p_born(m / r_j) / r_j`` is a quadrature of
    ``int dr k(r) p_born(m/r)/r``; placing each node at the *weighted mean* of
    ``u = -log r`` inside its group makes the first moment exact, so the
    residual error of a group is ``Var(u|group) m^2 |p''| / 2``.  Groups are
    therefore merged from a fine histogram until their in-group standard
    deviation of ``u`` would exceed ``sigma_cap`` (3.3e-3 -> 0.3 GeV at the Z,
    i.e. ~1e-4 of the peak density).  Non-radiating events (u below the MiniAOD
    float-precision floor) become a single atom at ``r = 1``.
    """
    r = np.asarray(m_post, np.float64) / np.asarray(m_pre, np.float64)
    w = np.asarray(w, np.float64)
    ok = np.isfinite(r) & (r > 0.0)
    r, w = r[ok], w[ok]
    r = np.minimum(r, 1.0)
    u = -np.log(r)

    hard = u > 1e-5                       # radiated; below this is float noise
    w_atom = w[~hard].sum()
    uh, wh = u[hard], w[hard]
    if u_max is None:
        u_max = uh.max() * (1.0 + 1e-9)

    nfine = int(np.ceil(u_max / u_fine))
    edges = np.linspace(0.0, u_max, nfine + 1)
    idx = np.clip(np.searchsorted(edges, uh, "right") - 1, 0, nfine - 1)
    s0 = np.bincount(idx, wh, nfine)
    s1 = np.bincount(idx, wh * uh, nfine)
    s2 = np.bincount(idx, wh * uh * uh, nfine)

    rj, wj = [1.0], [w_atom]
    a0 = a1 = a2 = 0.0
    for k in range(nfine):
        b0, b1, b2 = a0 + s0[k], a1 + s1[k], a2 + s2[k]
        if b0 <= 0.0:
            continue
        mu = b1 / b0
        var = max(b2 / b0 - mu * mu, 0.0)
        too_wide = (b0 * var > var_budget) if var_budget else (
            np.sqrt(var) > sigma_cap)
        if too_wide and a0 > 0.0:
            mu0 = a1 / a0
            rj.append(float(np.exp(-mu0)))
            wj.append(float(a0))
            a0, a1, a2 = s0[k], s1[k], s2[k]
        else:
            a0, a1, a2 = b0, b1, b2
    if a0 > 0.0:
        rj.append(float(np.exp(-a1 / a0)))
        wj.append(float(a0))

    rj = np.array(rj)
    wj = np.array(wj)
    o = np.argsort(-rj)
    rj, wj = rj[o], wj[o]
    tot = wj.sum()
    return {"r": rj, "w": wj / tot,
            "p_norad": float(w_atom / tot),
            "mean_dm_over_m": float(np.sum(wj * (rj - 1.0)) / tot)}


def build_banded_kernel(m_pre, m_post, w, bands, **kw):
    """One kernel per ``m_pre`` band, flattened with per-atom band limits.

    The multiplicative kernel is only *approximately* independent of
    ``m_pre``; inclusively the ``u = -ln r`` spectrum of this sample moves by a
    few per cent from 60 to 150 GeV, but once a lepton ``p_T`` cut is applied it
    moves by a factor two, because the cut removes hard emission at a rate that
    itself depends on ``m_pre``.  Banding restores the exact conditional
    kernel, piecewise constant in ``m_pre``.
    """
    m_pre = np.asarray(m_pre, np.float64)
    r_all, w_all, lo_all, hi_all = [], [], [], []
    info = []
    for lo, hi in bands:
        s_ = (m_pre >= lo) & (m_pre < hi)
        if not s_.any():
            continue
        k = build_kernel(m_pre[s_], np.asarray(m_post)[s_],
                         np.asarray(w)[s_], **kw)
        r_all.append(k["r"])
        w_all.append(k["w"])
        lo_all.append(np.full(len(k["r"]), float(lo)))
        hi_all.append(np.full(len(k["r"]), float(hi)))
        info.append((lo, hi, int(s_.sum()), len(k["r"]), k["p_norad"],
                     k["mean_dm_over_m"]))
    return (dict(r=np.concatenate(r_all), w=np.concatenate(w_all),
                 m_lo=np.concatenate(lo_all), m_hi=np.concatenate(hi_all)),
            info)


# --------------------------------------------------------------------------
# the acceptance
# --------------------------------------------------------------------------
def fit_acceptance(m, passed, w, lo, hi, degree, nbins=200):
    """Least-squares Bernstein fit of ``A(m) = P(pass | m)`` on ``[lo, hi]``."""
    from math import comb

    m = np.asarray(m, np.float64)
    w = np.asarray(w, np.float64)
    sel = (m >= lo) & (m < hi)
    m, w, passed = m[sel], w[sel], np.asarray(passed, bool)[sel]
    e = np.linspace(lo, hi, nbins + 1)
    i = np.clip(np.searchsorted(e, m, "right") - 1, 0, nbins - 1)
    den = np.bincount(i, w, nbins)
    num = np.bincount(i, w * passed, nbins)
    d2 = np.bincount(i, w * w, nbins)
    good = den > 0
    c = 0.5 * (e[1:] + e[:-1])
    a = np.where(good, num / np.where(good, den, 1.0), 0.0)
    # binomial-with-weights variance, floored so empty tails do not dominate
    var = np.where(good, np.maximum(a * (1 - a), 1e-4) * d2 / np.where(good, den, 1) ** 2, np.inf)
    u = (c - lo) / (hi - lo)
    B = np.stack([comb(degree, k) * u**k * (1 - u) ** (degree - k)
                  for k in range(degree + 1)], axis=1)
    sw = 1.0 / np.sqrt(var)
    coef, *_ = np.linalg.lstsq(B[good] * sw[good, None], a[good] * sw[good], rcond=None)
    resid = (B @ coef - a)[good]
    chi2 = float(np.sum((resid / np.sqrt(var[good])) ** 2))
    return {"kind": "bernstein", "lo": lo, "hi": hi, "coef": coef.tolist()}, dict(
        m=c, a=a, sig=np.sqrt(var), fit=B @ coef, good=good,
        chi2=chi2, ndf=int(good.sum() - degree - 1))


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
class GenFit:
    """Fine-binned weighted likelihood of one gen mass spectrum."""

    def __init__(self, provider, m, w, window, lumi_slope=False, shape=0, tf=None):
        self.tf = tf
        self.z = provider
        self.window = (float(window[0]), float(window[1]))
        grid = provider.m_grid
        dm = provider.dm
        if (self.window[0] < grid[1] - 1e-9) or (self.window[1] > grid[-2] + 1e-9):
            raise ValueError(
                f"fit window {self.window} must sit inside the model support "
                f"({grid[1]:.4f}, {grid[-2]:.4f}); the outermost Born node is "
                "zeroed by construction, so log p would be -inf there")
        lo = np.searchsorted(grid, self.window[0])
        hi = np.searchsorted(grid, self.window[1])
        self.sl = slice(lo, hi)
        m = np.asarray(m, np.float64)
        w = np.asarray(w, np.float64)
        k = np.rint((m - grid[0]) / dm).astype(np.int64)
        ok = (k >= lo) & (k < hi) & np.isfinite(m)
        k, w = k[ok] - lo, w[ok]
        n = hi - lo
        self.W = np.bincount(k, w, n)
        self.W2 = np.bincount(k, w * w, n)
        self.n_used = int(ok.sum())
        self.sumw = float(w.sum())
        self.sumw2 = float(np.sum(w * w))
        self.mgrid = grid[self.sl]
        self.lumi_slope = bool(lumi_slope)
        self.shape = int(shape)
        self.params = list(provider.param_names)
        if lumi_slope:
            self.params.append("lumi_slope")
        self.params += [f"shape{k}" for k in range(1, self.shape + 1)]
        self._tW = tf.constant(self.W, provider.dtype)
        self._logmref = tf.constant(
            np.log(provider.m_born / provider.m_ref), provider.dtype)
        # Legendre polynomials of the fit-window-scaled mass, evaluated on the
        # *Born* grid: a smooth multiplicative correction exp(sum_k c_k P_k)
        # applied to the Born spectrum, i.e. to the parton luminosity times the
        # QCD K-factor -- everything in the cross section that is not the
        # resonance.  Legendre (rather than plain powers) so the coefficients
        # are nearly uncorrelated over the window.  k starts at 1: the constant
        # is a normalisation and is already free.
        if self.shape:
            uu = (2.0 * (provider.m_born - self.window[0])
                  / (self.window[1] - self.window[0]) - 1.0)
            from numpy.polynomial import legendre
            basis = np.stack(
                [legendre.legval(uu, [0] * k + [1]) for k in range(1, self.shape + 1)])
            self._shape_basis = tf.constant(basis, provider.dtype)

    def density(self, vals):
        tf = self.tf
        z = self.z
        y = z.born_pdf(vals)
        if self.lumi_slope:
            y = y * tf.exp(vals["lumi_slope"] * self._logmref)
        if self.shape:
            c = tf.stack([vals[f"shape{k}"] for k in range(1, self.shape + 1)])
            arg = tf.tensordot(c, self._shape_basis, axes=1)
            y = y * tf.exp(tf.clip_by_value(arg, self.z.npdt(-30.0),
                                            self.z.npdt(30.0)))
        y = z.fold_fsr(y) * z._edge
        return y[self.sl]

    def nll(self, vals):
        tf = self.tf
        p = self.density(vals)
        norm = tf.reduce_sum(p) * self.z.npdt(self.z.dm)
        return -tf.reduce_sum(self._tW * (tf.math.log(p) - tf.math.log(norm)))

    def _vals(self, x):
        tf = self.tf
        return {k: tf.constant(v, self.z.dtype) for k, v in zip(self.params, x)}

    def value_grad_hess(self, x):
        tf = self.tf
        xv = [tf.Variable(float(v), dtype=self.z.dtype) for v in x]
        vals = dict(zip(self.params, xv))
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                f = self.nll(vals)
            g = t1.gradient(f, xv)
        h = [t2.gradient(gi, xv) for gi in g]
        del t2
        H = np.array([[0.0 if hij is None else float(hij) for hij in row] for row in h])
        return float(f), np.array([float(gi) for gi in g]), H

    def score_matrix(self, x):
        """``J = sum_b W2_b g_b g_b^T`` -- the sandwich's meat.

        Forward mode (one tangent per parameter) rather than reverse: the
        output is a length-``nbin`` vector and there are only 2-8 parameters,
        so ``npar`` JVPs beat a reverse-mode jacobian -- which also overflows
        ``pfor``'s int32 segment count once ``nbin`` gets large.
        """
        tf = self.tf
        npar = len(self.params)
        cols = []
        for k in range(npar):
            xv = [tf.Variable(float(v), dtype=self.z.dtype) for v in x]
            tangents = [tf.constant(1.0 if i == k else 0.0, self.z.dtype)
                        for i in range(npar)]
            with tf.autodiff.ForwardAccumulator(xv, tangents) as acc:
                vals = dict(zip(self.params, xv))
                p = self.density(vals)
                lp = tf.math.log(p) - tf.math.log(
                    tf.reduce_sum(p) * self.z.npdt(self.z.dm))
            cols.append(np.asarray(acc.jvp(lp)))
        G = np.stack(cols, axis=1)                          # (nbin, npar)
        return G.T @ (self.W2[:, None] * G)

    def fit(self, x0=None, verbose=True):
        from scipy.optimize import minimize
        if x0 is None:
            x0 = np.zeros(len(self.params))
            if self.shape or self.lumi_slope:
                # warm start: the POIs from the same fit with the nuisances held
                # at zero.  The exponential shape has enough curvature that a
                # cold trust-region start can walk into a region where the
                # density underflows and the Hessian is not finite.
                sub = GenFit.__new__(GenFit)
                sub.__dict__.update(self.__dict__)
                sub.shape = 0
                sub.lumi_slope = False
                sub.params = list(self.z.param_names)
                r0 = sub.fit(x0=np.zeros(len(sub.params)), verbose=False)
                x0[: len(r0["x"])] = r0["x"]
        x0 = np.asarray(x0, float)
        cache = {}

        def key(x):
            k = tuple(np.round(x, 12))
            if k not in cache:
                f, gr, H = self.value_grad_hess(x)
                if not (np.isfinite(f) and np.all(np.isfinite(gr))
                        and np.all(np.isfinite(H))):
                    f = np.inf
                    gr = np.zeros_like(gr)
                    H = np.eye(len(gr))
                cache[k] = (f, gr, H)
                if len(cache) > 8:
                    cache.pop(next(iter(cache)))
            return cache[k]

        t0 = time.time()
        res = minimize(lambda x: key(x)[0], x0,
                       jac=lambda x: key(x)[1], hess=lambda x: key(x)[2],
                       method="trust-exact",
                       options={"gtol": 1e-8, "maxiter": 200})
        f, g, H = self.value_grad_hess(res.x)
        Hinv = np.linalg.inv(H)
        J = self.score_matrix(res.x)
        V = Hinv @ J @ Hinv
        if verbose:
            print(f"    {res.message}  ({res.nit} it, {time.time()-t0:.1f} s, "
                  f"|g|inf = {np.max(np.abs(g)):.2e})")
        return dict(x=res.x, nll=f, grad=g, H=H, V=V, Vnaive=Hinv,
                    err=np.sqrt(np.diag(V)), errnaive=np.sqrt(np.diag(Hinv)),
                    params=self.params, nit=res.nit, ok=res.status in (0, 1, 2))


def make_provider(tf, window, nm, width_scheme, lumi, terms, fsr, acc,
                  mz_ref=None, gz_ref=None, sin2_param=None, fsr_mmax=None):
    from rabbit.lineshapes import ZGammaLineshape
    return ZGammaLineshape(
        m_ref=MZ_RUNNING, window=window, nm=nm, width_scheme=width_scheme,
        lumi=lumi, terms=terms, fsr=fsr, acceptance=acc,
        mz_ref=mz_ref, gz_ref=gz_ref, sin2_param=sin2_param,
        mz_unit=1e-3, gz_unit=1e-3,
        # the CF is not used at generator level; keep the tabulation tiny
        nfft=nm, tau_max=1.0,
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def row(name, res, refs, extra=""):
    """One line of the closure table: fitted offsets from the generator input."""
    out = [f"{name:<34s}"]
    for i, p in enumerate(res["params"]):
        v = res["x"][i]
        e = res["err"][i]
        if p in refs:
            out.append(f"{v:+9.3f} +- {e:6.3f}")
        else:
            out.append(f"{v:+9.4f} +- {e:6.4f}")
    out.append(extra)
    return "  ".join(out)


def summarise(tag, res, quiet=False):
    d = {p: (float(res["x"][i]), float(res["err"][i]), float(res["errnaive"][i]))
         for i, p in enumerate(res["params"])}
    d["_nll"] = float(res["nll"])
    d["_nit"] = int(res["nit"])
    V = res["V"]
    e = np.sqrt(np.diag(V))
    corr = V / np.outer(e, e)
    d["_corr"] = corr.tolist()
    if not quiet:
        print(f"  {tag}")
        for i, p in enumerate(res["params"]):
            extra = ""
            if p in ("m_Z", "Gamma_Z") and len(res["params"]) > 2:
                j = [k for k in range(len(res["params"]))
                     if res["params"][k] not in ("m_Z", "Gamma_Z")]
                if j:
                    extra = ("  rho(nuis) = "
                             + " ".join(f"{corr[i, k]:+.2f}" for k in j))
            print(f"      {p:<12s} {res['x'][i]:+10.4f} +- {res['err'][i]:8.4f}"
                  f"   (unit-weight +- {res['errnaive'][i]:8.4f}){extra}")
        if len(res["params"]) >= 2:
            print(f"      rho(m_Z, Gamma_Z) = {corr[0, 1]:+.3f}")
    return d


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("kernel", help="build the multiplicative FSR kernel")
    k.add_argument("--gen", required=True)
    k.add_argument("-o", "--output", required=True)
    k.add_argument("--acc-pt", type=float, default=None)
    k.add_argument("--acc-pt-trail", type=float, default=None)
    k.add_argument("--acc-eta", type=float, default=None)
    k.add_argument("--smear", default=None)
    k.add_argument("--smear-mode", default="shape", choices=("shape", "gauss"))
    k.add_argument("--smear-seed", type=int, default=SMEAR_SEED)
    k.add_argument("--sigma-cap", type=float, default=3.3e-4,
                   help="cap on the in-group sd of u = -ln(m_post/m_pre); the "
                        "fold is a midpoint quadrature, so the residual bias "
                        "scales as sigma_cap^2 (1e-2 -> +152 MeV on Gamma_Z, "
                        "3.3e-3 -> +29, 1e-3 -> +3.9)")
    k.add_argument("--var-budget", type=float, default=None,
                   help="alternative criterion: cap w_group * Var(u|group)")
    k.add_argument("--u-fine", type=float, default=2e-5)
    k.add_argument("--half", choices=["a", "b"], default=None,
                   help="use only half the sample (kernel statistical error)")
    k.add_argument("--mass-slice", nargs=2, type=float, default=None,
                   help="build the kernel from a m_pre slice (m-dependence test)")
    k.add_argument("--unweighted", action="store_true")
    k.add_argument("--wclip", type=float, default=100.0)
    k.add_argument("--bands", type=float, nargs="*", default=None,
                   help="band edges in m_pre for an m-dependent kernel")

    a = sub.add_parser("acceptance", help="fit A(m_pre) as a Bernstein polynomial")
    a.add_argument("--gen", required=True)
    a.add_argument("-o", "--output", required=True)
    a.add_argument("--acc-pt", type=float, default=25.0)
    a.add_argument("--acc-pt-trail", type=float, default=None)
    a.add_argument("--acc-eta", type=float, default=2.4)
    a.add_argument("--smear", default=None)
    a.add_argument("--smear-mode", default="shape", choices=("shape", "gauss"))
    a.add_argument("--smear-seed", type=int, default=SMEAR_SEED)
    a.add_argument("--lo", type=float, default=50.0)
    a.add_argument("--hi", type=float, default=200.0)
    a.add_argument("--degree", type=int, default=8)

    f = sub.add_parser("fit", help="the closure fit")
    f.add_argument("--gen", required=True)
    f.add_argument("--suite", default="prefsr",
                   choices=["prefsr", "postfsr", "fiducial", "quick", "perleg"])
    f.add_argument("--kernel", default=None)
    f.add_argument("--kernel-alt", nargs="*", default=[],
                   help="extra kernels to repeat the fit with (systematics); "
                        "'label=kernel.npz' renames the row and "
                        "'label=kernel.npz:acc.json' also swaps the acceptance")
    f.add_argument("--acc", default=None)
    f.add_argument("--nm", type=int, default=32768)
    f.add_argument("--born-hi", type=float, default=130.0)
    f.add_argument("--acc-pt", type=float, default=25.0,
                   help="leading muon pT cut [GeV]")
    f.add_argument("--acc-pt-trail", type=float, default=None,
                   help="trailing muon pT cut [GeV]; default = --acc-pt")
    f.add_argument("--acc-eta", type=float, default=2.4)
    f.add_argument("--smear", default=None,
                   help="a `ptres.py` resolution npz: the pT cuts then act on "
                        "a SMEARED post-FSR pT (the fitted mass is untouched)")
    f.add_argument("--smear-mode", default="shape", choices=("shape", "gauss"))
    f.add_argument("--smear-seed", type=int, default=SMEAR_SEED)
    f.add_argument("--shape", type=int, default=5,
                   help="Legendre terms of the smooth K(m) nuisance")
    f.add_argument("--fsr-mmax", type=float, default=None)
    f.add_argument("--nmax", type=int, default=0)
    f.add_argument("--wclip", type=float, default=100.0,
                   help="clip |w| at this multiple of the modal |w| (MiNNLO "
                        "produces a handful of 1e19 unweighting failures)")
    f.add_argument("-o", "--output", default=None)

    args = ap.parse_args()

    if args.cmd == "kernel":
        g = load_gen(args.gen)
        w = (np.ones(len(g["m_pre"])) if args.unweighted
             else clip_weights(g["weight"].astype(float), args.wclip))
        sel = np.ones(len(w), bool)
        if args.acc_pt is not None:
            sel &= fiducial(g, (args.acc_pt, args.acc_pt_trail or args.acc_pt),
                            args.acc_eta, post=True,
                            smear=load_resolution(args.smear, args.smear_mode),
                            seed=args.smear_seed)
        if args.mass_slice:
            sel &= (g["m_pre"] >= args.mass_slice[0]) & (g["m_pre"] < args.mass_slice[1])
        if args.half:
            n = len(w)
            sel &= (np.arange(n) < n // 2) if args.half == "a" else (np.arange(n) >= n // 2)
        meta = dict(gen=os.path.abspath(args.gen), n=int(sel.sum()),
                    sumw=float(w[sel].sum()), acc_pt=args.acc_pt,
                    acc_pt_trail=args.acc_pt_trail, smear=args.smear,
                    acc_eta=args.acc_eta, sigma_cap=args.sigma_cap,
                    half=args.half, mass_slice=args.mass_slice,
                    bands=args.bands, u_fine=args.u_fine,
                    var_budget=args.var_budget)
        if args.bands:
            bands = list(zip(args.bands[:-1], args.bands[1:]))
            bands = [(0.0, args.bands[0])] + bands + [(args.bands[-1], np.inf)]
            ker, info = build_banded_kernel(
                g["m_pre"][sel], g["m_post"][sel], w[sel], bands,
                sigma_cap=args.sigma_cap, u_fine=args.u_fine,
                var_budget=args.var_budget)
            np.savez(args.output, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
                     m_hi=ker["m_hi"], provenance=np.array([json.dumps(meta)]))
            print(f"[kernel] {sel.sum()} events, {len(info)} bands, "
                  f"{len(ker['r'])} atoms -> {args.output}")
            for lo, hi, n, na, pn, md in info:
                print(f"    [{lo:6.1f}, {hi:6.1f})  n = {n:9d}  atoms = {na:4d}"
                      f"  P(no rad) = {pn:.4f}  <dm/m> = {md*1e3:+8.4f}e-3")
            return
        ker = build_kernel(g["m_pre"][sel], g["m_post"][sel], w[sel],
                           sigma_cap=args.sigma_cap, u_fine=args.u_fine,
                           var_budget=args.var_budget)
        np.savez(args.output, r=ker["r"], w=ker["w"],
                 provenance=np.array([json.dumps(meta)]))
        print(f"[kernel] {sel.sum()} events -> {len(ker['r'])} atoms, "
              f"P(no rad) = {ker['p_norad']:.4f}, <dm/m> = "
              f"{ker['mean_dm_over_m']*1e3:.4f}e-3 -> {args.output}")
        return

    if args.cmd == "acceptance":
        g = load_gen(args.gen)
        w = clip_weights(g["weight"].astype(float), 100.0)
        p = fiducial(g, (args.acc_pt, args.acc_pt_trail or args.acc_pt),
                     args.acc_eta, post=True,
                     smear=load_resolution(args.smear, args.smear_mode),
                     seed=args.smear_seed)
        cfg, diag = fit_acceptance(g["m_pre"], p, w, args.lo, args.hi, args.degree)
        cfg["_meta"] = dict(acc_pt=args.acc_pt,
                            acc_pt_trail=args.acc_pt_trail,
                            smear=args.smear, acc_eta=args.acc_eta,
                            degree=args.degree, chi2=diag["chi2"], ndf=diag["ndf"])
        with open(args.output, "w") as fh:
            json.dump(cfg, fh, indent=1)
        np.savez(args.output.replace(".json", "_diag.npz"),
                 m=diag["m"], a=diag["a"], sig=diag["sig"], fit=diag["fit"],
                 good=diag["good"])
        print(f"[acceptance] degree {args.degree}, chi2/ndf = "
              f"{diag['chi2']:.1f}/{diag['ndf']} -> {args.output}")
        return

    run_fit(args)


def _alt_spec(spec):
    """``label=kernel.npz[:acceptance.json]`` -> ``(label, kernel, acceptance)``."""
    lab, _, rest = spec.partition("=")
    if not rest:
        lab, rest = os.path.basename(spec), spec
    kpath, _, apath = rest.partition(":")
    return lab, kpath, (apath or None)


def run_fit(args):
    import tensorflow as tf                                    # noqa: E402

    g = load_gen(args.gen)
    if args.nmax:
        g = {k: v[: args.nmax] for k, v in g.items() if v.ndim == 1}
    w = clip_weights(g["weight"].astype(np.float64), args.wclip)
    print(f"[fit] {len(w)} events, sum(w) = {w.sum():.6g}, "
          f"Neff = {w.sum()**2/np.sum(w**2):.6g} "
          f"({100*w.sum()**2/np.sum(w**2)/len(w):.1f} % of N), "
          f"wclip = {args.wclip}x modal")

    ker = None if args.kernel is None else args.kernel
    acc = None
    if args.acc:
        with open(args.acc) as fh:
            acc = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}

    born = (50.0, args.born_hi)
    refs = ("m_Z", "Gamma_Z")
    results = {}

    def one(tag, mass, sel=None, window=(60.0, 120.0), width_scheme="fixed",
            terms=("gamma", "int", "z"), lumi="nnpdf31_nnlo_13tev",
            fsr=None, acceptance=None, lumi_slope=False, shape=0,
            sin2_param=None, nm=None, born_hi=None, born_lo=None, x0=None,
            unweighted=False, fsr_mmax=None):
        m = g[mass]
        ww = np.ones_like(w) if unweighted else w
        if sel is not None:
            m, ww = m[sel], ww[sel]
        z = make_provider(
            tf, (born_lo or born[0], born_hi or born[1]),
            nm or args.nm, width_scheme, lumi,
            terms, fsr, acceptance,
            mz_ref=(MZ_FIXED if width_scheme == "fixed" else MZ_RUNNING),
            gz_ref=(GZ_FIXED if width_scheme == "fixed" else GZ_RUNNING),
            sin2_param=sin2_param,
            fsr_mmax=fsr_mmax if fsr_mmax is not None else args.fsr_mmax)
        fit = GenFit(z, m, ww, window, lumi_slope=lumi_slope, shape=shape, tf=tf)
        res = fit.fit(x0=x0, verbose=False)
        res["nused"] = fit.n_used
        results[tag] = summarise(f"{tag}  [{fit.n_used} events in {window}]", res)
        return res

    t0 = time.time()
    if args.suite in ("prefsr", "quick"):
        print("\n=== step 3: the pre-FSR (Born) spectrum, no FSR, no selection ===")
        base = one("baseline 60-120", "m_pre")
        if args.suite == "quick":
            return
        print("\n--- fit window (the LO/NNLO shape mismatch lives in the tails) ---")
        for lo, hi in ((70, 110), (80, 100), (86, 96), (89, 93), (60, 130), (51, 150)):
            one(f"window {lo}-{hi}", "m_pre", window=(lo, hi),
                born_hi=max(130.0, hi + 2.0))
        print("\n--- is the residual the parton luminosity? ---")
        for tag, path in (("NNPDF3.1 NNLO replica 1", "data/lumi/zlumi_nnpdf31_nnlo_rep1_13tev.npz"),
                          ("NNPDF3.1 LO", "data/lumi/zlumi_nnpdf31_lo_13tev.npz"),
                          ("CT18 NNLO", "data/lumi/zlumi_ct18nnlo_13tev.npz"),
                          ("mu_F = Q/2", "data/lumi/zlumi_nnpdf31_nnlo_muf05_13tev.npz"),
                          ("mu_F = 2Q", "data/lumi/zlumi_nnpdf31_nnlo_muf20_13tev.npz")):
            one(f"lumi {tag}", "m_pre", lumi=path)
        print("\n--- a smooth K(m) absorbs it: how many terms, and at what cost? ---")
        one("+ lumi slope nuisance", "m_pre", lumi_slope=True)
        for n in (1, 2, 3, 4, 5, 6):
            one(f"+ shape nuisance, {n} Legendre", "m_pre", shape=n)
        print("\n--- the same 3-term shape on top of each luminosity ---")
        for tag, path in (("NNPDF3.1 LO", "data/lumi/zlumi_nnpdf31_lo_13tev.npz"),
                          ("CT18 NNLO", "data/lumi/zlumi_ct18nnlo_13tev.npz"),
                          ("mu_F = Q/2", "data/lumi/zlumi_nnpdf31_nnlo_muf05_13tev.npz"),
                          ("mu_F = 2Q", "data/lumi/zlumi_nnpdf31_nnlo_muf20_13tev.npz")):
            one(f"shape 3 + lumi {tag}", "m_pre", lumi=path, shape=3)
        print("\n--- shape 3, other windows ---")
        for lo, hi in ((70, 110), (80, 100), (60, 130), (51, 150)):
            one(f"shape 3, window {lo}-{hi}", "m_pre", window=(lo, hi), shape=3,
                born_hi=max(130.0, hi + 2.0))
        print("\n--- matrix-element pieces, other definitions, numerics ---")
        one("m_prelep (lepton pair)", "m_prelep")
        one("m_prelep + shape 3", "m_prelep", shape=3)
        one("running-width scheme", "m_pre", width_scheme="running")
        one("running-width + shape 3", "m_pre", width_scheme="running", shape=3)
        one("no interference", "m_pre", terms=("gamma", "z"), shape=3)
        one("no photon exchange", "m_pre", terms=("int", "z"), shape=3)
        one("Z only", "m_pre", terms=("z",), shape=3)
        one("+ sin2 nuisance", "m_pre", sin2_param="sin2")
        one("+ sin2 nuisance + shape 3", "m_pre", sin2_param="sin2", shape=3)
        one("nm = 8192", "m_pre", nm=8192, shape=3)
        one("nm = 65536", "m_pre", nm=65536, shape=3)
        one("unweighted (w -> 1)", "m_pre", unweighted=True, shape=3)

    elif args.suite == "postfsr":
        S = args.shape
        print("\n=== step 4: the post-FSR spectrum with the folded kernel ===")
        one("no FSR fold, no shape", "m_post")
        one(f"no FSR fold, shape {S}", "m_post", shape=S)
        one("FSR folded, no shape", "m_post", fsr=ker)
        one(f"FSR folded, shape {S}", "m_post", fsr=ker, shape=S)
        print("\n--- window ---")
        for lo, hi in ((70, 110), (80, 100), (60, 130)):
            one(f"FSR folded, shape {S}, {lo}-{hi}", "m_post", fsr=ker, shape=S,
                window=(lo, hi), born_hi=max(130.0, hi + 2.0))
        print("\n--- kernel systematics ---")
        for kalt in args.kernel_alt:
            one(f"kernel {os.path.basename(kalt)}", "m_post", fsr=kalt, shape=S)
        print("\n--- numerics ---")
        for mm in (160.0, 200.0):
            one(f"Born grid capped at {mm:.0f} GeV", "m_post", fsr=ker, shape=S,
                fsr_mmax=mm)
        for n in (4096, 16384):
            one(f"nm = {n}", "m_post", fsr=ker, shape=S, nm=n)
        print("\n--- control: the same model fitted to the PRE-FSR mass ---")
        one(f"pre-FSR, shape {S} (no fold)", "m_pre", shape=S)
        one("pre-FSR, no shape (no fold)", "m_pre")

    elif args.suite == "perleg":
        S = args.shape
        cuts = (args.acc_pt, args.acc_pt_trail or args.acc_pt)
        rsm = load_resolution(args.smear, args.smear_mode)
        sel = fiducial(g, cuts, args.acc_eta, post=True, smear=rsm,
                       seed=args.smear_seed)
        print(f"\n=== per-leg factorised kernel, pT > {cuts[0]}/{cuts[1]}, "
              f"|eta| < {args.acc_eta}"
              + (f", smeared ({args.smear_mode})" if rsm else "")
              + f"  ({sel.sum()} / {len(sel)} events) ===")
        one(f"pre-FSR + A(m) control", "m_pre", sel=sel, acceptance=acc,
            shape=S)
        one(f"MC-conditional banded + A(m)", "m_post", sel=sel, fsr=ker,
            acceptance=acc, shape=S)
        for spec in args.kernel_alt:
            lab, kpath, apath = _alt_spec(spec)
            # a missing row is skipped, loudly: a suite is a long run and one
            # absent variant must not cost the rows that are already done
            if not os.path.exists(kpath) or (apath and not os.path.exists(apath)):
                print(f"  [skip] {lab}: missing kernel or acceptance file")
                continue
            a2 = acc
            if apath:
                with open(apath) as fh:
                    a2 = {k: v for k, v in json.load(fh).items()
                          if not k.startswith("_")}
            one(lab, "m_post", sel=sel, fsr=kpath, acceptance=a2, shape=S)

    elif args.suite == "fiducial":
        S = args.shape
        sel = fiducial(g, (args.acc_pt, args.acc_pt_trail or args.acc_pt),
                       args.acc_eta, post=True,
                       smear=load_resolution(args.smear, args.smear_mode),
                       seed=args.smear_seed)
        print(f"\n=== step 4b: fiducial selection pT > {args.acc_pt}, "
              f"|eta| < {args.acc_eta}  ({sel.sum()} / {len(sel)} events) ===")
        one(f"no FSR, no A(m), shape {S}", "m_post", sel=sel, shape=S)
        one(f"FSR folded, no A(m), shape {S}", "m_post", sel=sel, fsr=ker,
            shape=S)
        one(f"FSR folded + A(m), shape {S}", "m_post", sel=sel, fsr=ker,
            acceptance=acc, shape=S)
        one(f"pre-FSR + A(m), shape {S} (control)", "m_pre", sel=sel,
            acceptance=acc, shape=S)
        one(f"pre-FSR, no A(m), shape {S}", "m_pre", sel=sel, shape=S)
        print("\n--- kernel variants (the m-dependence of the selected kernel) ---")
        for spec in args.kernel_alt:
            lab, kpath, apath = _alt_spec(spec)
            a2 = acc
            if apath:
                with open(apath) as fh:
                    a2 = {k: v for k, v in json.load(fh).items()
                          if not k.startswith("_")}
            one(lab, "m_post", sel=sel, fsr=kpath, acceptance=a2, shape=S)
        print("\n--- acceptance parameterisation ---")
        for d in (4, 6, 10):
            f2 = f"data/acc_d{d}.json"
            if not os.path.exists(f2):
                continue
            with open(f2) as fh:
                a2 = {k: v for k, v in json.load(fh).items()
                      if not k.startswith("_")}
            one(f"A(m) Bernstein degree {d}", "m_post", sel=sel, fsr=ker,
                acceptance=a2, shape=S)
        print("\n--- window ---")
        for lo, hi in ((70, 110), (60, 130)):
            one(f"FSR + A(m), {lo}-{hi}", "m_post", sel=sel, fsr=ker,
                acceptance=acc, shape=S, window=(lo, hi),
                born_hi=max(130.0, hi + 2.0))

    print(f"\n[fit] {time.time()-t0:.0f} s")
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"[fit] -> {args.output}")


if __name__ == "__main__":
    main()
