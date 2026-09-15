#!/usr/bin/env python3
"""The reconstructed muon ``p_T`` resolution, as the acceptance needs it.

The fiducial cut of the Z channel acts on the **reconstructed** muon ``p_T``,
so the pass decision of the selection-conditional FSR kernel (`fsr_perleg.py`)
is not a step in the post-FSR ``p_T`` but a smooth probability

    P(pass | p_T, eta) = P( r > p_T^cut / p_T - 1 | p_T, eta ) ,
    r = p_T^reco / p_T^gen - 1 ,

with ``p_T`` the true (bare post-FSR) muon transverse momentum.  This module
measures the law of ``r`` on the CVH-refit Z -> mu mu MC and exposes it as

    `Resolution.sigma(pt, eta)`   the relative core width, a smooth function
    `Resolution.pass_prob(v, c)`  ``P(pass)`` at ``v = ln(p_T/c)``
    `Resolution.sample(pt, eta)`  a draw of ``r``, for the gen-level toy

so the detector side of the interface is one object and the boson-kinematics
side (`fsr_perleg.htable`) never sees it.

The measurement is the **per-muon** ratio to the charge-matched bare post-FSR
gen muon of the same candidate (``Mu{plus,minus}gen_pt``, the status-1 muon
matched within dR < 0.1 in the maker), binned in **gen** ``p_T`` and ``eta``:
binning in reconstructed ``p_T`` would bin the residual on a variable that
contains it.

Two renderings of the law are carried, because the acceptance is sensitive to
the tail and not only to the core:

* ``gauss``  ``r ~ Normal(mu(p_T, eta), sigma(p_T, eta))`` with ``sigma`` the
  4-parameter form ``(sigma/p_T)^2 = a^2 + c^2 p_T^2 + b^2/(1 + d^2/p_T^2)``
  fitted per ``eta`` band;
* ``shape``  ``r = mu + sigma * s`` with ``s`` the measured standardised shape
  of the same ``eta`` band -- same core, the real tails.

usage:
  python3 ptres.py measure --aux ../fullscale/runs/auxgen_dyv2.npz \
      --pairs ../fullscale/runs/zpairs_dyv2_full.npz \
      --seed ../fullscale/runs/auxseed_dyv2.npz -o data/ptres_dyv2.npz
  python3 ptres.py figs --res data/ptres_dyv2.npz --tag 260916_fsr_selection
"""
import argparse
import json
import os

import numpy as np

#: |eta| band edges of the resolution model.  Four bands: the resolution is a
#: function of the material and of the lever arm, both of which change at the
#: barrel/endcap transitions, and four is what the fit-level coarseness scan
#: (README) finds sufficient.
ETA_EDGES = np.array([0.0, 0.9, 1.6, 2.1, 2.4])

#: gen-p_T band edges.  Fine where the cuts are (10 and 25 GeV) and where the
#: statistics are; the model is the smooth 4-parameter fit, these are only the
#: cells it is fitted to and checked against.
PT_EDGES = np.array([8., 10., 12., 14., 17., 20., 23., 26., 30., 35., 40.,
                     45., 50., 60., 80., 120., 200.])

#: the standardised-shape grid, ``s = (r - mu)/sigma``.  +-30 core widths at
#: 0.02, so the 1e-4 tail of the pass probability is resolved.
S_EDGES = np.arange(-30.0, 30.0 + 1e-9, 0.02)

#: truncation of the core estimator, in units of the running sigma, and the
#: Gaussian correction factor of the truncated standard deviation.
CORE_K = 2.0


#: `math.erfc` over an array -- the build environment has no scipy, and the
#: Gaussian rendering of the pass probability is the only thing that needs it.
_erfc = np.vectorize(__import__("math").erfc, otypes=[float])


def _trunc_factor(k):
    """``sd(N(0,1) | |x| < k) / 1``."""
    from math import erf, exp, pi, sqrt
    phi = exp(-0.5 * k * k) / sqrt(2.0 * pi)
    P = erf(k / sqrt(2.0))
    return sqrt(1.0 - 2.0 * k * phi / P)


def core(r, w, k=CORE_K, nit=4):
    """``(mu, sigma)`` of the Gaussian core: an iterated truncated moment.

    Started from the interquartile range, which no tail can move, and corrected
    for the truncation so that a pure Gaussian returns its own sigma.
    """
    r = np.asarray(r, float)
    w = np.asarray(w, float)
    if len(r) < 20:
        return np.nan, np.nan
    q = _wquantile(r, w, [0.25, 0.5, 0.75])
    mu, sig = q[1], max(0.7413 * (q[2] - q[0]), 1e-9)
    f = _trunc_factor(k)
    for _ in range(nit):
        s = np.abs(r - mu) < k * sig
        if s.sum() < 20 or w[s].sum() <= 0:
            break
        ww = w[s]
        mu = float(np.sum(ww * r[s]) / np.sum(ww))
        v = float(np.sum(ww * (r[s] - mu) ** 2) / np.sum(ww))
        sig = np.sqrt(max(v, 1e-18)) / f
    return mu, sig


def _wquantile(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return np.full(len(qs), np.nan)
    return np.interp(np.asarray(qs) * c[-1], c - 0.5 * w, x)


# --------------------------------------------------------------------------
# the 4-parameter width model
# --------------------------------------------------------------------------
def sig_model(pt, p):
    """``sigma_pT/pT`` of ``(sigma/pT)^2 = a^2 + c^2 pT^2 + b^2/(1 + d^2/pT^2)``."""
    a, b, c, d = np.abs(np.asarray(p, float))
    pt = np.asarray(pt, float)
    return np.sqrt(a * a + c * c * pt * pt + b * b / (1.0 + d * d / (pt * pt)))


def fit_sig(pt, sig, err, p0=None, nit=400):
    """Least squares of the 4-parameter form on the measured cell widths.

    Levenberg-Marquardt with a numerical Jacobian: four parameters, one smooth
    residual, and no scipy in the build environment.
    """
    ok = np.isfinite(sig) & (sig > 0) & np.isfinite(err) & (err > 0)
    pt, sig, err = np.asarray(pt)[ok], np.asarray(sig)[ok], np.asarray(err)[ok]
    if len(pt) < 5:
        return np.full(4, np.nan), np.nan

    def res(p):
        return (sig_model(pt, p) - sig) / err

    best = None
    for p0 in ([5e-3, 1.5e-2, 1e-4, 10.0], [2e-3, 8e-3, 2e-4, 5.0],
               [1e-2, 2e-2, 5e-5, 20.0]):
        p = np.asarray(p0, float)
        lam, f = 1e-3, res(p)
        c = float(f @ f)
        for _ in range(nit):
            J = np.empty((len(f), 4))
            for k in range(4):
                q = p.copy()
                h = 1e-6 * max(abs(p[k]), 1e-8)
                q[k] += h
                J[:, k] = (res(q) - f) / h
            A = J.T @ J
            g = J.T @ f
            for _ in range(40):
                try:
                    step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-30), -g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                pn = np.abs(p + step)
                fn = res(pn)
                cn = float(fn @ fn)
                if cn < c:
                    p, f, c, lam = pn, fn, cn, max(lam * 0.3, 1e-12)
                    break
                lam *= 10.0
            else:
                break
            if np.max(np.abs(step) / np.maximum(np.abs(p), 1e-12)) < 1e-10:
                break
        if best is None or c < best[1]:
            best = (p, c)
    p = best[0]
    dev = np.max(np.abs(sig_model(pt, p) / sig - 1.0))
    return np.abs(p), float(dev)


# --------------------------------------------------------------------------
# the model object
# --------------------------------------------------------------------------
class Resolution:
    """The measured law of ``r = pT_reco/pT_gen - 1``, as the acceptance uses it.

    ``mode`` is ``gauss`` (the fitted core only) or ``shape`` (the core scaled
    by the measured standardised shape of the same ``eta`` band).
    """

    def __init__(self, path, mode="shape"):
        d = np.load(path, allow_pickle=False)
        self.path = os.path.abspath(path)
        self.mode = mode
        self.eta_edges = d["eta_edges"]
        self.pt_edges = d["pt_edges"]
        self.coef = d["coef"]                     # (n_eta, 4)
        self.mu_coef = d["mu_coef"]               # (n_eta, 2): mu = m0 + m1/pT
        self.s_edges = d["s_edges"]
        self.s_cdf = d["s_cdf"]                   # (n_eta, n_s+1)
        self.meta = json.loads(str(d["provenance"][0]))

    # -- the width and the mean -------------------------------------------
    def iband(self, eta):
        return np.clip(np.searchsorted(self.eta_edges, np.abs(eta), "right") - 1,
                       0, len(self.eta_edges) - 2)

    def sigma(self, pt, eta):
        """``sigma_pT/pT(pT, |eta|)``, the fitted 4-parameter form."""
        i = self.iband(eta)
        pt = np.asarray(pt, float)
        out = np.empty(np.broadcast(pt, i).shape)
        for b in np.unique(i):
            s = (i == b) if np.ndim(i) else slice(None)
            out[s] = sig_model(np.broadcast_to(pt, out.shape)[s], self.coef[b])
        return out

    def mean(self, pt, eta):
        """``<r>(pT, |eta|)``: a two-term fit ``m0 + m1/pT`` per band."""
        i = self.iband(eta)
        pt = np.asarray(pt, float)
        c = self.mu_coef[i]
        return c[..., 0] + c[..., 1] / pt

    # -- the standardised shape -------------------------------------------
    def cdf_s(self, s, iband):
        """``P(s' < s)`` of the standardised shape of one ``eta`` band."""
        return np.interp(s, self.s_edges, self.s_cdf[iband], left=0.0, right=1.0)

    def pass_prob(self, v, cut, iband):
        """``P(pT_reco > cut)`` at ``v = ln(pT_true/cut)``, band ``iband``.

        ``pT_reco = pT_true (1 + r)``, so the condition is
        ``r > e^{-v} - 1``.  The width is evaluated at the TRUE ``pT``,
        ``cut e^v``, which is what ``r``'s law is conditioned on.
        """
        v = np.asarray(v, float)
        pt = cut * np.exp(v)
        sg = sig_model(pt, self.coef[iband])
        mu = self.mu_coef[iband][0] + self.mu_coef[iband][1] / pt
        thr = (np.expm1(-v) - mu) / sg
        if self.mode == "gauss":
            return 0.5 * _erfc(thr / np.sqrt(2.0))
        return 1.0 - self.cdf_s(thr, iband)

    def pass_prob_ev(self, pt, cut, eta):
        """``P(pT_reco > cut)`` for muons of true ``(pT, eta)``, elementwise."""
        pt = np.asarray(pt, float)
        i = self.iband(eta)
        v = np.log(pt / cut)
        out = np.empty(pt.shape)
        for b in np.unique(i):
            s = i == b
            out[s] = self.pass_prob(v[s], cut, int(b))
        return out

    def sample(self, pt, eta, rng):
        """A draw of ``r`` at ``(pT, eta)``."""
        pt = np.asarray(pt, float)
        i = self.iband(eta)
        sg = np.empty_like(pt)
        mu = np.empty_like(pt)
        for b in np.unique(i):
            s = i == b
            sg[s] = sig_model(pt[s], self.coef[b])
            mu[s] = self.mu_coef[b][0] + self.mu_coef[b][1] / pt[s]
        if self.mode == "gauss":
            return mu + sg * rng.standard_normal(pt.shape)
        u = rng.random(pt.shape)
        out = np.empty_like(pt)
        for b in np.unique(i):
            s = i == b
            out[s] = np.interp(u[s], self.s_cdf[b], self.s_edges)
        return mu + sg * out


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def load_legs(aux, pairs=None, wclip=100.0, mass_window=None):
    """Per-muon ``(r, pt_gen, eta_gen, w, sigrel)`` after the standard selection.

    The two legs of every selected candidate are concatenated; the standard
    two-track selection (`resolution/selection.py`) is applied to the CANDIDATE,
    which is what the Z channel will fit.
    """
    d = np.load(aux)
    n = len(d["gpt_p"])
    w = np.ones(n)
    if pairs:
        with np.load(pairs) as p:
            w = np.asarray(p["w"], float)
        aw = np.abs(w)
        w = np.clip(w, -wclip * np.median(aw), wclip * np.median(aw))
        w = w * (len(w) / w.sum())
    sel = (np.isfinite(d["sigma"]) & (d["sigma"] > 0)
           & (d["nv_p"] >= 8) & (d["nv_m"] >= 8) & (d["normchi2"] < 3.0))
    sel &= (d["gpt_p"] > 0) & (d["gpt_m"] > 0)
    if mass_window:
        sel &= (d["mgen"] > mass_window[0]) & (d["mgen"] < mass_window[1])
    out = {}
    for lab in ("p", "m"):
        out[lab] = dict(r=d[f"pt_{lab}"][sel] / d[f"gpt_{lab}"][sel] - 1.0,
                        pt=d[f"gpt_{lab}"][sel], eta=d[f"geta_{lab}"][sel],
                        w=w[sel])
    cat = {k: np.concatenate([out["p"][k], out["m"][k]]) for k in out["p"]}
    cat["nsel"] = int(sel.sum())
    cat["ntot"] = n
    cat["sel"] = sel
    return cat


def measure(cat, eta_edges=ETA_EDGES, pt_edges=PT_EDGES, s_edges=S_EDGES):
    """Cell-by-cell core and tails, the 4-parameter fits, the shape per band."""
    r, pt, eta, w = cat["r"], cat["pt"], np.abs(cat["eta"]), cat["w"]
    ie = np.searchsorted(eta_edges, eta, "right") - 1
    ip = np.searchsorted(pt_edges, pt, "right") - 1
    ok = (ie >= 0) & (ie < len(eta_edges) - 1) & (ip >= 0) & (ip < len(pt_edges) - 1)
    ne, npt = len(eta_edges) - 1, len(pt_edges) - 1
    mu = np.full((ne, npt), np.nan)
    sg = np.full((ne, npt), np.nan)
    er = np.full((ne, npt), np.nan)
    nn = np.zeros((ne, npt), np.int64)
    tl = np.zeros((ne, npt, 4))                 # P(s<-3), P(s>3), P(s<-5), P(s>5)
    ptbar = np.full((ne, npt), np.nan)
    sst, wst, ist = [], [], []
    for a in range(ne):
        for b in range(npt):
            s = ok & (ie == a) & (ip == b)
            if s.sum() < 50:
                continue
            m, g = core(r[s], w[s])
            mu[a, b], sg[a, b] = m, g
            nn[a, b] = int(s.sum())
            ptbar[a, b] = float(np.sum(w[s] * pt[s]) / np.sum(w[s]))
            # the error on the core width: the Gaussian 1/sqrt(2 N_eff)
            neff = np.sum(w[s]) ** 2 / np.sum(w[s] ** 2)
            er[a, b] = g / np.sqrt(2.0 * max(neff, 1.0))
            z = (r[s] - m) / g
            ww = w[s] / np.sum(w[s])
            tl[a, b] = [np.sum(ww * (z < -3)), np.sum(ww * (z > 3)),
                        np.sum(ww * (z < -5)), np.sum(ww * (z > 5))]
            sst.append(z)
            wst.append(w[s])
            ist.append(np.full(s.sum(), a))
    # the standardised shape, per eta band, pooled over pT cells
    sst = np.concatenate(sst)
    wst = np.concatenate(wst)
    ist = np.concatenate(ist)
    cdf = np.zeros((ne, len(s_edges)))
    for a in range(ne):
        s = ist == a
        h = np.histogram(np.clip(sst[s], s_edges[0], s_edges[-1]),
                         bins=s_edges, weights=wst[s])[0]
        h = np.maximum(h, 0.0)
        c = np.concatenate([[0.0], np.cumsum(h)])
        cdf[a] = c / c[-1]
    # the width model and the mean model
    coef = np.full((ne, 4), np.nan)
    dev = np.full(ne, np.nan)
    mcoef = np.zeros((ne, 2))
    for a in range(ne):
        coef[a], dev[a] = fit_sig(ptbar[a], sg[a], er[a])
        s = np.isfinite(mu[a]) & np.isfinite(ptbar[a])
        if s.sum() >= 2:
            A = np.stack([np.ones(s.sum()), 1.0 / ptbar[a][s]], 1)
            mcoef[a] = np.linalg.lstsq(A, mu[a][s], rcond=None)[0]
    return dict(eta_edges=eta_edges, pt_edges=pt_edges, pt_bar=ptbar,
                mu=mu, sigma=sg, sigma_err=er, n=nn, tails=tl,
                s_edges=s_edges, s_cdf=cdf, coef=coef, dev=dev, mu_coef=mcoef)


def pull(aux, seed, wclip=100.0):
    """The per-leg pull ``(q/p - q/p_gen)/(sigrel |q/p|)`` and its core width.

    ``sigrel`` is the two-track fit's own reported relative momentum resolution,
    i.e. the CVH covariance.  A core width of 1 means the covariance can be used
    for the acceptance directly; anything else is the factor it is off by.
    """
    a = np.load(aux)
    d = np.load(seed)
    sel = (np.isfinite(a["sigma"]) & (a["sigma"] > 0)
           & (a["nv_p"] >= 8) & (a["nv_m"] >= 8) & (a["normchi2"] < 3.0)
           & (a["gpt_p"] > 0) & (a["gpt_m"] > 0))
    out = {}
    z = np.concatenate([d["zleg_p"][sel], d["zleg_m"][sel]])
    sr = np.concatenate([d["sigrel_p"][sel], d["sigrel_m"][sel]])
    pt = np.concatenate([a["gpt_p"][sel], a["gpt_m"][sel]])
    eta = np.abs(np.concatenate([a["geta_p"][sel], a["geta_m"][sel]]))
    ok = np.isfinite(z) & np.isfinite(sr) & (sr > 0)
    out["z"], out["sigrel"], out["pt"], out["eta"] = z[ok], sr[ok], pt[ok], eta[ok]
    return out


# --------------------------------------------------------------------------
def _measure_cmd(args):
    cat = load_legs(args.aux, args.pairs, mass_window=tuple(args.mass_window)
                    if args.mass_window else None)
    print(f"[ptres] {cat['nsel']} / {cat['ntot']} candidates "
          f"({2*cat['nsel']} muons) after the standard selection")
    m = measure(cat, eta_edges=np.asarray(args.eta_edges, float)
                if args.eta_edges else ETA_EDGES)
    meta = dict(aux=os.path.abspath(args.aux),
                pairs=os.path.abspath(args.pairs) if args.pairs else None,
                mass_window=args.mass_window,
                selection="finite sigma_m>0, nvalid>=8 both legs, chi2/ndof<3, "
                          "gen-matched both legs",
                n_cand=cat["nsel"], n_mu=2 * cat["nsel"], core_k=CORE_K)
    np.savez_compressed(args.output, provenance=np.array([json.dumps(meta)]),
                        **{k: np.asarray(v) for k, v in m.items()})
    print(f"[ptres] -> {args.output}")
    ee = m["eta_edges"]
    for a in range(len(ee) - 1):
        c = m["coef"][a]
        print(f"  |eta| in [{ee[a]:.1f}, {ee[a+1]:.1f}): "
              f"a = {c[0]*1e3:6.3f}e-3  b = {c[1]*1e3:6.3f}e-3  "
              f"c = {c[2]*1e5:6.3f}e-5  d = {c[3]:7.3f}   "
              f"max dev = {m['dev'][a]*100:5.2f} %")
        for b in range(len(m["pt_edges"]) - 1):
            if not np.isfinite(m["sigma"][a, b]):
                continue
            print(f"      pT [{m['pt_edges'][b]:5.0f}, {m['pt_edges'][b+1]:5.0f}) "
                  f"n = {m['n'][a,b]:8d}  <pT> = {m['pt_bar'][a,b]:6.1f}  "
                  f"mu = {m['mu'][a,b]*1e3:+7.3f}e-3  "
                  f"sigma = {m['sigma'][a,b]*1e3:7.3f}e-3  "
                  f"fit/meas = {sig_model(m['pt_bar'][a,b], m['coef'][a])/m['sigma'][a,b]:6.4f}  "
                  f"tails -3/+3 = {m['tails'][a,b,0]*1e2:5.2f}/{m['tails'][a,b,1]*1e2:5.2f} % "
                  f"-5/+5 = {m['tails'][a,b,2]*1e3:5.2f}/{m['tails'][a,b,3]*1e3:5.2f} e-3")
    if args.seed:
        p = pull(args.aux, args.seed)
        mu, sg = core(p["z"], np.ones(len(p["z"])))
        print(f"[ptres] covariance pull: core mean {mu:+.4f}, core width "
              f"{sg:.4f} over {len(p['z'])} legs")
        for a in range(len(ee) - 1):
            s = (p["eta"] >= ee[a]) & (p["eta"] < ee[a + 1])
            m2, s2 = core(p["z"][s], np.ones(int(s.sum())))
            print(f"      |eta| in [{ee[a]:.1f}, {ee[a+1]:.1f}): "
                  f"pull {m2:+.4f} +- {s2:.4f}  n = {int(s.sum())}")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _figs_cmd(args):
    import datetime
    import matplotlib.pyplot as plt
    import mplhep as hep

    import pubhtml
    import ratiopanel

    hep.style.use(hep.style.ROOT)
    out = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today():%y%m%d}_{args.tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)
    d = np.load(args.res, allow_pickle=False)
    ee, pe = d["eta_edges"], d["pt_edges"]
    sg, er, pb, mu = d["sigma"], d["sigma_err"], d["pt_bar"], d["mu"]
    coef, tails = d["coef"], d["tails"]

    # -- the width against the 4-parameter form
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    pg = np.geomspace(pe[0], pe[-1], 300)
    for a in range(len(ee) - 1):
        ok = np.isfinite(sg[a])
        lab = rf"${ee[a]:.1f} < |\eta| < {ee[a+1]:.1f}$"
        ax.errorbar(pb[a][ok], sg[a][ok] * 1e3, yerr=er[a][ok] * 1e3, fmt="o",
                    ms=4, lw=0, elinewidth=0.9, color=f"C{a}", label=lab)
        ax.plot(pg, sig_model(pg, coef[a]) * 1e3, color=f"C{a}", lw=1.5)
        rax.plot(pb[a][ok], sig_model(pb[a][ok], coef[a]) / sg[a][ok], "o-",
                 ms=3, lw=1.0, color=f"C{a}")
    ax.set_xscale("log")
    rax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"core $\sigma_{p_T}/p_T$  [$10^{-3}$]")
    rax.set_ylabel("fit / measured")
    rax.set_xlabel(r"gen (post-FSR) $p_T$ [GeV]")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_ylim(0.94, 1.06)
    for c in (10.0, 25.0):
        ax.axvline(c, color="grey", lw=0.8, ls=":")
        rax.axvline(c, color="grey", lw=0.8, ls=":")
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.set_title("CVH-refit muon momentum resolution, DY MC", fontsize=14,
                 loc="left")
    pubhtml.savefig(fig, os.path.join(out, "30_ptres_sigma.pdf"))
    plt.close(fig)

    # -- the standardised shape against a unit Gaussian
    se, cdf = d["s_edges"], d["s_cdf"]
    ctr = 0.5 * (se[:-1] + se[1:])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    gaus = np.exp(-0.5 * ctr ** 2) / np.sqrt(2.0 * np.pi)
    for a in range(len(ee) - 1):
        p = np.diff(cdf[a]) / np.diff(se)
        ax.step(ctr, p, where="mid", lw=1.4, color=f"C{a}",
                label=rf"${ee[a]:.1f} < |\eta| < {ee[a+1]:.1f}$")
        with np.errstate(divide="ignore", invalid="ignore"):
            rax.step(ctr, p / gaus, where="mid", lw=1.4, color=f"C{a}")
    ax.plot(ctr, gaus, color="black", lw=1.2, ls="--", label="unit Gaussian")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlim(-8, 8)
    rax.set_xlim(-8, 8)
    rax.set_yscale("log")
    rax.set_ylim(0.3, 3e3)
    ax.set_ylabel(r"density of $s = (r - \mu)/\sigma$")
    rax.set_ylabel("/ Gaussian")
    rax.set_xlabel(r"$s$")
    rax.axhline(1.0, color="grey", lw=0.8)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.set_title(r"standardised shape of $r = p_T^{\rm reco}/p_T^{\rm gen} - 1$",
                 fontsize=14, loc="left")
    pubhtml.savefig(fig, os.path.join(out, "31_ptres_shape.pdf"))
    plt.close(fig)

    # -- the tail fractions, cell by cell
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    for a in range(len(ee) - 1):
        ok = np.isfinite(sg[a])
        ax.plot(pb[a][ok], tails[a][ok, 0] * 1e2, "o-", ms=4, lw=1.2,
                color=f"C{a}",
                label=rf"${ee[a]:.1f} < |\eta| < {ee[a+1]:.1f}$, $s < -3$")
        ax.plot(pb[a][ok], tails[a][ok, 1] * 1e2, "s--", ms=4, lw=1.2,
                color=f"C{a}", mfc="none")
    ax.axhline(0.135, color="black", lw=1.0, ls=":")
    ax.text(pe[-1], 0.145, "Gaussian", ha="right", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel(r"gen (post-FSR) $p_T$ [GeV]")
    ax.set_ylabel(r"$P(|s| > 3)$ per side  [%]")
    ax.legend(fontsize=11, frameon=False, ncol=2)
    ax.set_title("non-Gaussian tails (open: $s > +3$)", fontsize=14, loc="left")
    pubhtml.savefig(fig, os.path.join(out, "32_ptres_tails.pdf"))
    plt.close(fig)

    # -- the covariance pull
    if args.aux and args.seed:
        p = pull(args.aux, args.seed)
        fig, ax = plt.subplots(figsize=(9.5, 6.4))
        b = np.linspace(-6, 6, 241)
        c = 0.5 * (b[:-1] + b[1:])
        for a in range(len(ee) - 1):
            s = (p["eta"] >= ee[a]) & (p["eta"] < ee[a + 1])
            h = np.histogram(p["z"][s], bins=b, density=True)[0]
            m2, s2 = core(p["z"][s], np.ones(int(s.sum())))
            ax.step(c, h, where="mid", lw=1.4, color=f"C{a}",
                    label=rf"${ee[a]:.1f} < |\eta| < {ee[a+1]:.1f}$: "
                          rf"${m2:+.3f} \pm {s2:.3f}$")
        ax.plot(c, np.exp(-0.5 * c ** 2) / np.sqrt(2 * np.pi), "k--", lw=1.2,
                label="unit Gaussian")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlabel(r"$(q/p - q/p_{\rm gen}) / (\sigma_{\rm rel} |q/p|)$")
        ax.set_ylabel("density")
        ax.legend(fontsize=11, frameon=False)
        ax.set_title("per-leg pull of the CVH covariance", fontsize=14,
                     loc="left")
        pubhtml.savefig(fig, os.path.join(out, "33_ptres_pull.pdf"))
        plt.close(fig)
    print(f"[ptres] figures -> {out}")


# --------------------------------------------------------------------------
# is the acceptance correlated with the per-candidate resolution?
# --------------------------------------------------------------------------
#: branches the sigma-conditioning study needs from the two-track tree
SIGBR = ["Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass", "Jpsigenpre_mass",
         "Jpsi_sigmarelplus", "Jpsi_sigmarelminus",
         "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
         "Muplusgen_pt", "Muminusgen_pt", "Muplusgen_eta", "Muminusgen_eta",
         "Muplus_nvalid", "Muminus_nvalid", "chisqval", "ndof"]


def _sig_one(fn):
    import uproot
    a = uproot.open(fn)["tree"].arrays(SIGBR, library="np")
    return {k: np.asarray(v, np.float64) for k, v in a.items()}


def sigcond(files, cuts, eta_cut=2.4, nproc=16):
    """``<u|sigma>`` of the selected sample, reco cut against true cut.

    The likelihood conditions each candidate on its own mass resolution
    ``sigma``.  If the acceptance and ``sigma`` are correlated, a
    population-level ``A(m)`` is not enough: the selected FSR content would
    depend on ``sigma`` and the conditional density would have to carry it.

    Two selections on the same events separate the two sources of that
    correlation:

    * the TRUE cut (on the bare post-FSR gen ``pT``) is correlated with
      ``sigma`` through kinematics alone -- a forward or soft muon has both a
      larger ``sigma`` and a different acceptance;
    * the RECO cut adds the promotion of candidates across the threshold by
      their own fluctuation, which is the effect that would have to be
      conditioned on.

    The difference between the two is that promotion and nothing else.
    """
    from multiprocessing import Pool
    with Pool(nproc) as p:
        out = p.map(_sig_one, files)
    d = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    c = np.atleast_1d(np.asarray(cuts, float))
    cL, cT = ((c[0], c[0]) if c.size == 1
              else (float(np.max(c)), float(np.min(c))))
    ok = (np.isfinite(d["Jpsi_sigmamass"]) & (d["Jpsi_sigmamass"] > 0)
          & (d["Muplus_nvalid"] >= 8) & (d["Muminus_nvalid"] >= 8)
          & (d["chisqval"] / np.maximum(d["ndof"], 1) < 3.0)
          & (d["Jpsigen_mass"] > 0) & (d["Jpsigenpre_mass"] > 0))
    u = -np.log(np.maximum(d["Jpsigen_mass"] / d["Jpsigenpre_mass"], 1e-300))
    k = d["Jpsi_sigmamass"] / np.maximum(d["Jpsi_mass"], 1e-9)
    sel = {}
    for lab, (p1, p2, e1, e2) in (
            ("reco", ("Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta")),
            ("true", ("Muplusgen_pt", "Muminusgen_pt",
                      "Muplusgen_eta", "Muminusgen_eta"))):
        sel[lab] = (ok & (np.maximum(d[p1], d[p2]) > cL)
                    & (np.minimum(d[p1], d[p2]) > cT)
                    & (np.abs(d[e1]) < eta_cut) & (np.abs(d[e2]) < eta_cut))
    return d, u, k, sel, ok


def _sigcond_cmd(args):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "resolution"))
    import prodfiles
    files = prodfiles.resolve(args.files, max_tasks=args.max_tasks)
    print(f"[sigcond] {len(files)} files")
    d, u, k, sel, ok = sigcond(files, args.pt_cuts, args.eta_cut, args.nproc)
    nq = args.nquantiles
    print(f"[sigcond] cuts {args.pt_cuts}, |eta| < {args.eta_cut}: "
          f"{int(ok.sum())} candidates, reco-selected {int(sel['reco'].sum())}, "
          f"true-selected {int(sel['true'].sum())}, "
          f"promoted {int((sel['reco'] & ~sel['true']).sum())}, "
          f"demoted {int((sel['true'] & ~sel['reco']).sum())}")
    print(f"  {'k = sigma_m/m quantile':>24s} {'<k>':>9s} {'n reco':>9s} "
          f"{'<u> reco':>11s} {'<u> true':>11s} {'reco/true':>10s}")
    ktot = k[ok]
    e = np.quantile(ktot, np.linspace(0.0, 1.0, nq + 1))
    e[0], e[-1] = -np.inf, np.inf
    for i in range(nq):
        m = ok & (k >= e[i]) & (k < e[i + 1])
        a = m & sel["reco"]
        b = m & sel["true"]
        if not a.any() or not b.any():
            continue
        ua, ub = float(u[a].mean()), float(u[b].mean())
        print(f"  {i*100//nq:3d}-{(i+1)*100//nq:<3d} %"
              f"  [{e[i] if np.isfinite(e[i]) else ktot.min():.5f},"
              f" {e[i+1] if np.isfinite(e[i+1]) else ktot.max():.5f})"
              f" {float(k[m].mean()):9.5f} {int(a.sum()):9d}"
              f" {ua*1e3:10.4f}e-3 {ub*1e3:10.4f}e-3 {ua/ub:10.4f}")
    a, b = sel["reco"], sel["true"]
    print(f"  {'inclusive':>24s} {float(ktot.mean()):9.5f} {int(a.sum()):9d}"
          f" {float(u[a].mean())*1e3:10.4f}e-3 {float(u[b].mean())*1e3:10.4f}e-3"
          f" {float(u[a].mean()/u[b].mean()):10.4f}")
    # Is k predicted by the two legs' (pT, eta)?  If it is, the h4 table
    # already carries it and a k-conditioned acceptance needs no new axis --
    # it is the same table restricted to the cells of a k class.
    if args.res:
        R = Resolution(args.res)
        s1 = R.sigma(d["Muplusgen_pt"], d["Muplusgen_eta"])
        s2 = R.sigma(d["Muminusgen_pt"], d["Muminusgen_eta"])
        kp = 0.5 * np.sqrt(s1 * s1 + s2 * s2)
        m = ok & sel["true"] & (k > 0) & (k < 0.2)
        r = k[m] / kp[m]
        q = np.quantile(r, [0.16, 0.5, 0.84])
        print(f"  k against the two legs' (pT, eta): k/k_pred median "
              f"{q[1]:.4f}, 68 % spread {(q[2]-q[0])/2/q[1]*100:.2f} %, "
              f"corr(ln k, ln k_pred) = "
              f"{np.corrcoef(np.log(k[m]), np.log(kp[m]))[0,1]:.4f}")

    # the promoted population, which is what a sigma-conditioned acceptance
    # would have to describe
    pm = sel["reco"] & ~sel["true"]
    if pm.any():
        print(f"  promoted: {int(pm.sum())} candidates, <k> = "
              f"{float(k[pm].mean()):.5f} against {float(ktot.mean()):.5f} "
              f"inclusive, <u> = {float(u[pm].mean())*1e3:.4f}e-3 against "
              f"{float(u[sel['true']].mean())*1e3:.4f}e-3")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--aux", required=True, help="auxgen_*.npz (gen + reco legs)")
    m.add_argument("--pairs", default=None, help="pairs cache, for the weights")
    m.add_argument("--seed", default=None, help="auxseed_*.npz, for the pull")
    m.add_argument("--mass-window", nargs=2, type=float, default=None)
    m.add_argument("--eta-edges", nargs="*", type=float, default=None)
    m.add_argument("-o", "--output", required=True)
    m.set_defaults(func=_measure_cmd)

    f = sub.add_parser("figs")
    f.add_argument("--res", required=True, help="a `measure` npz")
    f.add_argument("--aux", default=None)
    f.add_argument("--seed", default=None, help="auxseed_*.npz, for the pull")
    f.add_argument("--outpath", default=None)
    f.add_argument("--tag", default="fsr_selection")
    f.set_defaults(func=_figs_cmd)

    c = sub.add_parser("sigcond",
                       help="is the acceptance correlated with the "
                            "per-candidate mass resolution?")
    c.add_argument("--files", required=True,
                   help="a production base directory (see resolution/prodfiles)")
    c.add_argument("--max-tasks", type=int, default=40)
    c.add_argument("--pt-cuts", nargs="*", type=float, default=[25.0, 10.0])
    c.add_argument("--eta-cut", type=float, default=2.4)
    c.add_argument("--nquantiles", type=int, default=5)
    c.add_argument("--nproc", type=int, default=16)
    c.add_argument("--res", default=None,
                   help="a `measure` npz: also test whether k is predicted by "
                        "the two legs' (pT, eta), i.e. by the h4 table")
    c.set_defaults(func=_sigcond_cmd)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
