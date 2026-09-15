#!/usr/bin/env python3
"""The per-leg factorised FSR kernel against the generator record.

Validates, in order: the collinear per-leg picture (eta and phi are conserved,
``z = x_+ x_-``, the legs' independence), the single-leg radiator ``D`` against
its analytic form, ``D (x) D`` against the inclusive kernel, and the
selection-conditional ``K_sel(u|m)`` and ``A(m)`` against the MC's own.

One file per panel, ratio panel under every density-vs-model plot, mplhep ROOT
style, output under ``~/public_html/ZMass/cvh/<date>_fsr_perleg/``.
"""
import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging

import pubhtml
import ratiopanel

import fit_gen as FG
import fsr_analytic as FA
import fsr_perleg as PL

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

PEAK = (86.0, 96.0)
BANDS = [(60, 70), (70, 80), (80, 86), (86, 96), (96, 110), (110, 130)]
#: the `data` configuration of `fsr_config`
DVAR, DPAIR = "exp2nll", ("e", "mu", "tau", "had")


def clip(w, factor=100.0):
    w = np.asarray(w, np.float64)
    w = np.clip(w, -factor * np.median(np.abs(w)), factor * np.median(np.abs(w)))
    return w * (len(w) / w.sum())


def load_perleg(path):
    d = np.load(path)
    g = {k: d[k] for k in d.files}
    g["weight"] = clip(g["weight"])
    return g


def wmean(a, w):
    return float(np.sum(w * a) / np.sum(w))


def tailw(a, w, t):
    return float(np.sum(w * (a > t)) / np.sum(w))


# --------------------------------------------------------------------------
def fig_angles(out, g):
    """(a) the collinear picture: eta and phi are conserved per leg."""
    w = g["weight"]
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    t = np.geomspace(1e-6, 3.0, 60)
    for q, lab, col in (("p", r"$\mu^+$", "C0"), ("m", r"$\mu^-$", "C3")):
        de = np.abs(g[f"eta{q}"] - g[f"eta{q}_pre"]).astype(np.float64)
        dp = np.abs(g[f"phi{q}"] - g[f"phi{q}_pre"]).astype(np.float64)
        dp = np.minimum(dp, 2.0 * np.pi - dp)
        ax.plot(t, [tailw(de, w, x) for x in t], color=col, lw=1.8,
                label=lab + r"  $|\Delta\eta|$")
        ax.plot(t, [tailw(dp, w, x) for x in t], color=col, ls="--", lw=1.6,
                label=lab + r"  $|\Delta\phi|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$P(|\Delta\eta| > \delta)$,  $P(|\Delta\phi| > \delta)$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title("post-FSR minus pre-FSR direction, per leg (charge matched)",
                 fontsize=14, loc="left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, "01_angles.pdf"))
    plt.close(fig)


def fig_leg_x(out, g, band=PEAK, nbin=48):
    """(b) the per-leg radiator ``D(x)`` measured against the analytic one."""
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    mm = wmean(m[s], w[s])
    u = np.concatenate([-np.log(np.maximum(g["xp"][s], 1e-300)),
                        -np.log(np.maximum(g["xm"][s], 1e-300))])
    ww = np.concatenate([w[s], w[s]])
    e = np.geomspace(1e-4, 2.0, nbin + 1)
    c = np.sqrt(e[1:] * e[:-1])
    bw = np.diff(e)
    i = np.clip(np.searchsorted(e, u, "right") - 1, 0, nbin - 1)
    ok = (u >= e[0]) & (u < e[-1])
    W = np.bincount(i[ok], ww[ok], nbin)
    W2 = np.bincount(i[ok], ww[ok] ** 2, nbin)
    tot = ww.sum()
    dens, err = W / (tot * bw), np.sqrt(W2) / (tot * bw)

    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.errorbar(c, dens, yerr=err, fmt="o", ms=3.0, color="black", lw=0,
                elinewidth=0.9, label="Photos++, per leg (charge matched)")
    mods = {}
    for var, pair, lab, col, ls in (
            (DVAR, DPAIR, r"analytic $D$, data cfg", "C0", "-"),
            ("exp1", (), r"analytic $D$, exp. O($\alpha$)", "C3", "--")):
        D = PL.LegRadiator(mm, variant=var, pair=pair)
        y = ratiopanel.bin_average(D.pdf_u(e[:-1]), D.pdf_u(c), D.pdf_u(e[1:]))
        mods[lab] = y
        ax.plot(c, y, color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u_{\rm leg}$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title(rf"$u_{{\rm leg}} = -\ln x$, $x = E'/E$ in the Born $Z$ frame,"
                 rf" $m_{{pre}}\in[{band[0]:.0f},{band[1]:.0f})$ GeV",
                 fontsize=13, loc="left")
    for lab, y in mods.items():
        rax.plot(c, dens / y, lw=1.6,
                 color="C0" if "data" in lab else "C3",
                 ls="-" if "data" in lab else "--")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.7, 1.6)
    rax.set_xlabel(r"$u_{\rm leg}$")
    rax.set_ylabel("MC / analytic")
    pubhtml.savefig(fig, os.path.join(out, "02_leg_x.pdf"))
    plt.close(fig)


def fig_leg_corr(out, g, band=PEAK):
    """(c) the legs are NOT independent: the joint tail against the product."""
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    ww = w[s]
    a = -np.log(np.maximum(g["xp"][s], 1e-300))
    b = -np.log(np.maximum(g["xm"][s], 1e-300))
    t = np.geomspace(1e-5, 1.0, 40)
    r = []
    for x in t:
        pa = tailw(a, ww, x)
        pb = tailw(b, ww, x)
        pab = float(np.sum(ww * ((a > x) & (b > x))) / ww.sum())
        r.append(pab / (pa * pb))
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.plot(t, r, color="C0", lw=2.0)
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P(u_+>t,\,u_->t)\,/\,P(u_+>t)P(u_->t)$")
    ax.set_title("leg independence in the generator: a wide-angle photon "
                 "makes both legs lose energy", fontsize=13, loc="left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, "03_leg_corr.pdf"))
    plt.close(fig)


def one_photon(g):
    """``k^2 ~ 0``: the events whose whole FSR is a single photon.

    The photon system is ``k = Q_pre - p'_+ - p'_-`` and its invariant mass
    vanishes for one photon (and only then, up to the float32 storage of the
    record, which is what sets the 1e-8 s threshold).
    """
    def p4(pt, eta, phi, e):
        return np.stack([e, pt * np.cos(phi), pt * np.sin(phi),
                         pt * np.sinh(eta)], 0)
    Q = (p4(g["ptp_pre"], g["etap_pre"], g["phip_pre"], g["ep_pre"])
         + p4(g["ptm_pre"], g["etam_pre"], g["phim_pre"], g["em_pre"]))
    K = (Q - p4(g["ptp"], g["etap"], g["phip"], g["ep"])
         - p4(g["ptm"], g["etam"], g["phim"], g["em"]))
    sq = K[0] ** 2 - K[1] ** 2 - K[2] ** 2 - K[3] ** 2
    m = g["m_pre"].astype(np.float64)
    rad = g["m_post"].astype(np.float64) < m * (1.0 - 1e-6)
    return rad & (np.abs(sq) < 1e-8 * m * m)


def fig_share(out, g, band=PEAK, slices=((1e-2, 5e-2), (5e-2, 0.2))):
    """The exact O(alpha) sharing density against the MC's single-photon events.

    ``f = (1 - x_+)/(1 - z)`` is where on the line ``x_+ + x_- = 1 + z`` the
    event sits: ``0`` and ``1`` are the collinear end points, the middle is the
    recoil the collinear factorisation has no room for.
    """
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    z = (g["m_post"].astype(np.float64) / m) ** 2
    u = -0.5 * np.log(np.maximum(z, 1e-300))
    f = (1.0 - g["xp"]) / np.maximum(1.0 - z, 1e-300)
    one = one_photon(g)
    e = np.geomspace(1e-6, 0.5, 25)
    e = np.unique(np.concatenate([e, 1.0 - e[::-1]]))
    c = np.sqrt(e[:-1] * np.where(e[1:] < 0.5, e[1:], e[1:]))
    c = 0.5 * (e[:-1] + e[1:])
    fig, ax, rax = ratiopanel.make_ratio_fig()
    for i, (lo, hi) in enumerate(slices):
        s = (m >= band[0]) & (m < band[1]) & (u > lo) & (u < hi) & one
        ww = w[s]
        ff = np.clip(f[s], e[0] * 1.001, e[-1] * 0.999)
        j = np.clip(np.searchsorted(e, ff, "right") - 1, 0, len(e) - 2)
        h = np.bincount(j, ww, len(e) - 1)
        n = np.bincount(j, np.ones_like(ww), len(e) - 1)
        h = h / h.sum()
        zb = float(np.sum(ww * z[s]) / ww.sum())
        mb = float(np.sum(ww * m[s]) / ww.sum())
        fm, wm = FA.share_nodes(mb, np.array([zb]), npanel=256, ng=8)
        fa = np.concatenate([fm[0], 1.0 - fm[0]])
        wa = np.concatenate([wm[0], wm[0]])
        k = np.clip(np.searchsorted(e, fa, "right") - 1, 0, len(e) - 2)
        hm = np.bincount(k, wa, len(e) - 1)
        col = f"C{i}"
        # the outermost bins are the clipped overflow, not a density
        h[0] = h[-1] = np.nan
        hm[0] = hm[-1] = np.nan
        ax.step(e[:-1], h, where="post", color=col, lw=1.6,
                label=rf"MC, 1$\gamma$, $u\in[{lo:g},{hi:g})$")
        ax.step(e[:-1], hm, where="post", color=col, lw=1.6, ls="--",
                label=rf"exact O($\alpha$), $\bar z = {zb:.4f}$")
        ok = (n > 30) & (hm > 0) & np.isfinite(h) & np.isfinite(hm)
        rax.errorbar(c[ok], (h / hm)[ok], (h / np.sqrt(n))[ok] / hm[ok],
                     fmt="o", ms=3.0, color=col)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("fraction of events per bin")
    ax.legend(fontsize=11, loc="lower center", ncol=2)
    ax.set_title("the exact matrix element's energy sharing, against the "
                 "generator's own single-photon events", fontsize=13, loc="left")
    ax.grid(alpha=0.25)
    rax.axhline(1.0, color="grey", lw=1.0, ls="--")
    rax.set_ylim(0.6, 1.4)
    rax.set_xlabel(r"$f = (1-x_+)/(1-z)$")
    rax.set_ylabel("MC / exact")
    rax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, "12_share.pdf"))
    plt.close(fig)


def fig_share_u(out, g, kernels, band=PEAK):
    """How much of the loss the *other* leg takes, against the total loss.

    ``P(0.01 < f < 0.99)`` is the weight away from the two collinear end points.
    The collinear product puts it at the ``beta_D`` power law's own value, the
    exact single-photon sharing at the matrix element's, and the generator
    above both -- the excess is emissions that are not one photon.
    """
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    z = (g["m_post"].astype(np.float64) / m) ** 2
    u = -0.5 * np.log(np.maximum(z, 1e-300))
    f = (1.0 - g["xp"]) / np.maximum(1.0 - z, 1e-300)
    one = one_photon(g)
    sel = (m >= band[0]) & (m < band[1])
    e = np.geomspace(6e-4, 0.8, 15)
    cen = np.sqrt(e[:-1] * e[1:])
    mb = float(np.sum(w[sel] * m[sel]) / w[sel].sum())

    def prof(mask):
        s = sel & mask
        j = np.clip(np.searchsorted(e, u[s], "right") - 1, 0, len(e) - 2)
        tot = np.bincount(j, w[s], len(e) - 1)
        mid = np.bincount(j, w[s] * ((f[s] > 0.01) & (f[s] < 0.99)),
                          len(e) - 1)
        n = np.bincount(j, np.ones(int(s.sum())), len(e) - 1)
        # bin 0 is the clipped underflow and holds the unradiated events, for
        # which the sharing is not defined
        tot[0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            return mid / tot, n

    all_, nall = prof(np.ones(len(u), bool))
    one_, none = prof(one)
    # the exact single-photon sharing at the same z
    zc = np.exp(-2.0 * cen)
    fm, wm = FA.share_nodes(mb, zc, npanel=256, ng=8)
    mod = 2.0 * np.sum(wm * ((fm > 0.01) & (fm < 0.99)), axis=1)
    # the collinear product D (x) D at the same total u
    D = PL.LegRadiator(mb, variant=DVAR, pair=DPAIR)
    ue = PL.leg_u_edges(D, n=1500)
    ul, wl = PL.leg_atoms(PL.leg_cells(D, ue), ue)
    wl = wl / wl.sum()
    U = 0.5 * (ul[:, None] + ul[None, :])
    W = wl[:, None] * wl[None, :]
    F = np.where(U > 0, 0.5 + 0.25 * (ul[:, None] - ul[None, :]) / np.maximum(U, 1e-300), 0.5)
    j = np.clip(np.searchsorted(e, U.ravel(), "right") - 1, 0, len(e) - 2)
    tot = np.bincount(j, W.ravel(), len(e) - 1)
    mid = np.bincount(j, (W * ((F > 0.01) & (F < 0.99))).ravel(), len(e) - 1)
    tot[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        coll = mid / tot

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.errorbar(cen, all_, all_ / np.sqrt(np.maximum(nall, 1)), fmt="o",
                ms=4.0, color="k", label="MC, all")
    ax.errorbar(cen, one_, one_ / np.sqrt(np.maximum(none, 1)), fmt="s",
                ms=4.0, mfc="none", color="C3", label=r"MC, one photon")
    ax.plot(cen, mod, color="C0", lw=2.0,
            label=r"exact O($\alpha$) sharing (the model)")
    ax.plot(cen, coll, color="C2", lw=2.0, ls=":",
            label=r"collinear product $D\otimes D$")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 0.6)
    ax.set_xlabel(r"$u = -\ln(m'/m)$")
    ax.set_ylabel(r"$P(0.01 < f < 0.99 \;|\; u)$")
    ax.set_title(f"how often both legs lose, {band[0]:.0f}-{band[1]:.0f} GeV",
                 fontsize=13, loc="left")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, "13_share_u.pdf"))
    plt.close(fig)


def fig_joint(out, g, band=PEAK):
    """The joint leg tail of the correlated model against the generator's.

    The model's legs are drawn from the exact sharing at fixed ``z``, so its
    joint tail is *not* the product of its marginals; the collinear product's
    is, by construction.
    """
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    ww = w[s] / w[s].sum()
    a = -np.log(np.maximum(g["xp"][s], 1e-300))
    b = -np.log(np.maximum(g["xm"][s], 1e-300))
    mb = float(np.sum(ww * m[s]))
    K = FA.FSRKernel(mb, variant=DVAR, pair=DPAIR)
    r, wk, _ = K.atoms(var_budget=6e-11)
    z = r * r
    fm, wm = FA.share_nodes(mb, z, npanel=128, ng=8)
    xp, xm = PL.share_x(z, fm)
    UP = -np.log(np.maximum(xp, 1e-300))
    UM = -np.log(np.maximum(xm, 1e-300))
    W = wk[:, None] * wm
    t = np.geomspace(1e-5, 0.5, 30)
    rm, rd = [], []
    for x in t:
        pa = float(np.sum(ww * (a > x)))
        pb = float(np.sum(ww * (b > x)))
        rd.append(float(np.sum(ww * ((a > x) & (b > x)))) / (pa * pb))
        pM = float(np.sum(W * ((UP > x).astype(float) + (UM > x))))
        jM = 2.0 * float(np.sum(W * ((UP > x) & (UM > x))))
        rm.append(jM / pM ** 2)
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.plot(t, rd, color="k", lw=2.0, label="MC")
    ax.plot(t, rm, color="C0", lw=2.0, ls="--",
            label=r"exact O($\alpha$) sharing (the model)")
    ax.axhline(1.0, color="C2", lw=2.0, ls=":",
               label=r"collinear product $D\otimes D$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P(u_+>t,\,u_->t)\,/\,P(u_+>t)P(u_->t)$")
    ax.set_title("the leg correlation the sharing puts back, "
                 f"{band[0]:.0f}-{band[1]:.0f} GeV", fontsize=13, loc="left")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, "14_joint.pdf"))
    plt.close(fig)


def fig_z_vs_xx(out, g, band=PEAK):
    """(d) ``z = x_+ x_-``: the pair mass loss against the collinear product."""
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    ww = w[s]
    umc = -np.log(np.maximum(g["m_post"][s].astype(np.float64) / m[s], 1e-300))
    ufac = 0.5 * (-np.log(np.maximum(g["xp"][s], 1e-300))
                  - np.log(np.maximum(g["xm"][s], 1e-300)))
    t = np.geomspace(1e-4, 1.5, 44)
    p1 = np.array([tailw(umc, ww, x) for x in t])
    p2 = np.array([tailw(ufac, ww, x) for x in t])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.plot(t, p1, color="black", lw=2.0, label=r"$u = -\ln(m'/m)$ (true)")
    ax.plot(t, p2, color="C1", lw=1.8, ls="--",
            label=r"$(u_+ + u_-)/2$ (collinear)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(u > u_0)$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title(rf"collinear factorisation of the pair mass, "
                 rf"$m_{{pre}}\in[{band[0]:.0f},{band[1]:.0f})$ GeV",
                 fontsize=13, loc="left")
    rax.plot(t, p2 / p1, color="C1", lw=1.8)
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.85, 1.05)
    rax.set_xlabel(r"$u_0$")
    rax.set_ylabel("collinear / true")
    pubhtml.savefig(fig, os.path.join(out, "04_z_vs_xx.pdf"))
    plt.close(fig)


def fig_dconvd(out, mass=91.1876, n_leg=6000):
    """``D (x) D`` against the inclusive kernel it is the Mellin root of."""
    ue = np.concatenate([[0.0], np.geomspace(1e-9, 7.0, 6000)])
    t = np.geomspace(1e-4, 1.5, 50)
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    for var, pair, lab, col, ls in (
            ("exp1", (), r"exp. O($\alpha$)", "C3", "--"),
            ("exp2", (), r"+ O($\alpha^2$)LL", "C2", ":"),
            (DVAR, DPAIR, "data cfg (NLL + exact pairs)", "C0", "-")):
        K = FA.FSRKernel(mass, variant=var, pair=pair)
        D = PL.LegRadiator(mass, variant=var, pair=pair)
        e = PL.leg_u_edges(D, n=n_leg)
        u, w = PL.leg_atoms(PL.leg_cells(D, e), e)
        s = PL.conv_atoms(u, w / w.sum(), ue)
        um = np.where(s[0] > 0, s[1] / np.maximum(s[0], 1e-300), 0.0)
        pd = np.array([s[0][um > x].sum() / s[0].sum() for x in t])
        # the same ladder for K, pair factor folded in on both sides
        ek = PL.leg_u_edges(K, n=n_leg)
        kc = PL.leg_cells(K, ek)
        ku = np.where(kc[0] > 0, kc[1] / np.maximum(kc[0], 1e-300), 0.0)
        pk = np.array([kc[0][ku > x].sum() / kc[0].sum() for x in t])
        if var == DVAR:
            ax.plot(t, pk, color="black", lw=2.0, label=r"inclusive $K$")
        ax.plot(t, pd, color=col, ls=ls, lw=1.7, label=rf"$D\otimes D$, {lab}")
        rax.plot(t, pd / pk, color=col, ls=ls, lw=1.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(u > u_0)$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title(rf"$m = {mass}$ GeV; the residual is the O($\alpha^3$) "
                 r"difference of the two soft resummations", fontsize=13,
                 loc="left")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.99, 1.01)
    rax.set_xlabel(r"$u_0$")
    rax.set_ylabel(r"$D\otimes D$ / $K$")
    pubhtml.savefig(fig, os.path.join(out, "05_dconvd.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
def kernel_bands(path):
    """Per-band records of a per-leg kernel file, or ``[]`` for other kinds."""
    p = json.loads(str(np.load(path, allow_pickle=False)["provenance"][0]))
    b = p.get("bands")
    if not isinstance(b, list) or not b or not isinstance(b[0], dict):
        return []
    return [x for x in b if x.get("natoms")]


def kernel_u_of(path):
    """Per-band ``(m, <u>, A)`` of a per-leg kernel file, or ``None``."""
    b = kernel_bands(path)
    if not b:
        return None
    return (np.array([x["m_bar"] for x in b]),
            np.array([x["mean_u"] for x in b]),
            np.array([x["A"] for x in b]))


def sel_mask(d, cuts, eta_cut, smear=None):
    """The fiducial mask of a gen record: `fit_gen.fiducial`, one place.

    ``cuts`` is ``(leading, trailing)``; ``smear`` a `ptres.Resolution`, in
    which case the cuts act on a smeared post-FSR ``pT`` with the seeded draw
    every consumer of this study shares.
    """
    return FG.fiducial({k: np.asarray(d[k], np.float64) for k in
                        ("m_pre", "pt1", "eta1", "pt2", "eta2")},
                       cuts, eta_cut, post=True, smear=smear)


def cutlabel(cuts, smear=None):
    c = np.atleast_1d(np.asarray(cuts, float))
    c = (c[0], c[0]) if c.size == 1 else (np.max(c), np.min(c))
    s = (rf"$p_T > {c[0]:.0f}/{c[1]:.0f}$ GeV" if c[0] != c[1]
         else rf"$p_T > {c[0]:.0f}$ GeV")
    return s + (", smeared" if smear is not None else "")


def mc_profile(gen, cuts, eta_cut, edges, smear=None):
    """MC ``A(m)`` and ``<u>`` of the selected sample, per band."""
    d = np.load(gen)
    w = clip(d["weight"].astype(np.float64))
    m = d["m_pre"].astype(np.float64)
    sel = sel_mask(d, cuts, eta_cut, smear)
    u = -np.log(np.maximum(d["m_post"].astype(np.float64) / m, 1e-300))
    # the first and the last band mean what they say: events outside the grid
    # are dropped, not folded into the edge bins
    keep = (m >= edges[0]) & (m < edges[-1])
    w, m, sel, u = w[keep], m[keep], sel[keep], u[keep]
    i = np.clip(np.searchsorted(edges, m, "right") - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    tot = np.bincount(i, w, n)
    npass = np.bincount(i, w * sel, n)
    su = np.bincount(i, w * sel * u, n)
    mb = np.bincount(i, w * m, n) / np.maximum(tot, 1e-30)
    with np.errstate(invalid="ignore", divide="ignore"):
        return mb, npass / tot, su / npass, np.bincount(i, w * sel, n) ** 2 / \
            np.maximum(np.bincount(i, (w * sel) ** 2, n), 1e-30)


# --------------------------------------------------------------------------
def mc_leg(run, band, du=PL.KSQRT_DU, u_max=PL.KSQRT_UMAX):
    """``(u, w, D, du, m_bar, info)`` of the run band covering ``band``."""
    rows = PL._run_bands(run)
    mid = 0.5 * (band[0] + band[1])
    k = int(np.argmin([abs(0.5 * (r[0] + r[1]) - mid) for r in rows]))
    lo, hi, mb, s0, s1, t0, t1, wn = rows[k]
    u, w = PL.kernel_atoms(s0, s1, wn, t0, t1)
    g = PL.deposit(u, w, du, int(round(u_max / du)))
    D, info = PL.conv_sqrt(g)
    info.update(lo=lo, hi=hi, m_bar=mb)
    return u, w, D, du, mb, info


def fig_leg_D_mc(out, g, run, band=PEAK, nbin=48):
    """The effective leg: the numerical square root of ``K_mc`` against ``D``.

    ``D`` is *defined* by ``D (x) D = K``.  For the analytic kernel it is the
    closed form of `LegRadiator`; for the tabulated Photos kernel it is the
    numerical convolution square root.  The physical per-leg spectrum -- the
    sample's own ``x = E'/E`` -- is a third, different object.
    """
    ua, wa, D, du, mb, info = mc_leg(run, band)
    # `D` lives on a uniform grid of step 2 du in u_leg; snap the log bin edges
    # to it so every bin holds a whole number of nodes and the density carries
    # no aliasing against the grid
    e = np.unique(np.round(np.geomspace(20.0 * du, 2.0, nbin + 1)
                           / (2.0 * du)) * (2.0 * du))
    nbin = len(e) - 1
    c = np.sqrt(e[1:] * e[:-1])
    bw = np.diff(e)
    uleg = np.arange(len(D)) * (2.0 * du)
    i = np.clip(np.searchsorted(e, uleg, "right") - 1, 0, nbin - 1)
    ok = (uleg >= e[0]) & (uleg < e[-1])
    dmc = np.bincount(i[ok], D[ok], nbin) / bw

    Dan = PL.LegRadiator(mb, variant=DVAR, pair=DPAIR)
    dan = ratiopanel.bin_average(Dan.pdf_u(e[:-1]), Dan.pdf_u(c),
                                 Dan.pdf_u(e[1:]))

    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    up = np.concatenate([-np.log(np.maximum(g["xp"][s], 1e-300)),
                         -np.log(np.maximum(g["xm"][s], 1e-300))])
    ww = np.concatenate([w[s], w[s]])
    j = np.clip(np.searchsorted(e, up, "right") - 1, 0, nbin - 1)
    okp = (up >= e[0]) & (up < e[-1])
    dphys = np.bincount(j[okp], ww[okp], nbin) / (ww.sum() * bw)

    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.plot(c, dan, color="C0", lw=2.0, label=r"analytic $D$ (data cfg)")
    ax.plot(c, dmc, color="C3", ls="--", lw=1.8,
            label=r"$D_{\rm mc}$: numerical $\sqrt{K_{\rm mc}}$")
    ax.plot(c, dphys, color="grey", ls=":", lw=1.8,
            label="physical per-leg spectrum (MC)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u_{\rm leg}$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title(rf"the effective leg, $m_{{pre}}\in[{band[0]:.0f},"
                 rf"{band[1]:.0f})$ GeV", fontsize=13, loc="left")
    rax.plot(c, dmc / dan, color="C3", ls="--", lw=1.8)
    rax.plot(c, dphys / dan, color="grey", ls=":", lw=1.8)
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.7, 1.6)
    rax.set_xlabel(r"$u_{\rm leg}$")
    rax.set_ylabel(r"/ analytic $D$")
    pubhtml.savefig(fig, os.path.join(out, "10_leg_D_mc.pdf"))
    plt.close(fig)


def fig_dconvd_mc(out, run, band=PEAK, n_leg=6000):
    """``D_mc (x) D_mc`` against the tabulated ``K_mc`` it is the root of."""
    ua, wa, D, du, mb, info = mc_leg(run, band)
    n = len(D)
    DD = np.fft.irfft(np.fft.rfft(D) ** 2, n)
    ug = np.arange(n) * du
    # cuts half-way between nodes: on the grid the kernel is a point measure,
    # so a cut sitting on a node is ambiguous at the level of that node's weight
    t = (np.round(np.geomspace(1e-4, 1.5, 50) / du) + 0.5) * du
    o = np.argsort(ua)
    ck = np.concatenate([np.cumsum(wa[o][::-1])[::-1], [0.0]])
    pk = np.array([ck[np.searchsorted(ua[o], x, "right")] for x in t])
    cd = np.concatenate([np.cumsum(DD[::-1])[::-1], [0.0]])
    pd = np.array([cd[np.searchsorted(ug, x, "right")] for x in t])

    K = FA.FSRKernel(mb, variant=DVAR, pair=DPAIR)
    Da = PL.LegRadiator(mb, variant=DVAR, pair=DPAIR)
    ue = np.concatenate([[0.0], np.geomspace(1e-9, 7.0, 6000)])
    e = PL.leg_u_edges(Da, n=n_leg)
    u, w = PL.leg_atoms(PL.leg_cells(Da, e), e)
    s = PL.conv_atoms(u, w / w.sum(), ue)
    um = np.where(s[0] > 0, s[1] / np.maximum(s[0], 1e-300), 0.0)
    pda = np.array([s[0][um > x].sum() / s[0].sum() for x in t])
    kc = PL.leg_cells(K, PL.leg_u_edges(K, n=n_leg))
    ku = np.where(kc[0] > 0, kc[1] / np.maximum(kc[0], 1e-300), 0.0)
    pka = np.array([kc[0][ku > x].sum() / kc[0].sum() for x in t])

    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.plot(t, pk, color="black", lw=2.0, label=r"$K_{\rm mc}$ (Photos)")
    ax.plot(t, pd, color="C3", ls="--", lw=1.8,
            label=r"$D_{\rm mc}\otimes D_{\rm mc}$")
    ax.plot(t, pda, color="C0", ls=":", lw=1.8,
            label=r"analytic $D\otimes D$ / its own $K$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(u > u_0)$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title(rf"$\langle m_{{pre}}\rangle = {mb:.2f}$ GeV; the numerical "
                 r"root is exact, the analytic one to O($\alpha^3$)",
                 fontsize=13, loc="left")
    rax.plot(t, pd / pk, color="C3", ls="--", lw=1.8)
    rax.plot(t, pda / pka, color="C0", ls=":", lw=1.8)
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.99, 1.01)
    rax.set_xlabel(r"$u_0$")
    rax.set_ylabel(r"$D\otimes D$ / $K$")
    pubhtml.savefig(fig, os.path.join(out, "11_dconvd_mc.pdf"))
    plt.close(fig)


def fig_acceptance(out, gen, kernels, cuts, eta_cut, smear=None, sfx=""):
    edges = np.arange(56.0, 142.0, 2.0)
    mb, A, umc, neff = mc_profile(gen, cuts, eta_cut, edges, smear)
    ok = np.isfinite(A) & (mb > 0)
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.plot(mb[ok], A[ok], "o", ms=4, color="black", label="MC (post-FSR cut)")
    for path, lab, col, ls in kernels:
        got = kernel_u_of(path)
        if got is None:
            continue
        m, _, a = got
        ax.plot(m, a, color=col, ls=ls, lw=1.8, label=lab)
        rax.plot(mb[ok], np.interp(mb[ok], m, a) / A[ok], color=col, ls=ls,
                 lw=1.8)
    ax.set_ylabel(r"$A(m_{\rm pre})$")
    ax.legend(fontsize=13, frameon=False, loc="lower right")
    ax.set_title("acceptance, " + cutlabel(cuts, smear)
                 + rf", $|\eta| < {eta_cut}$", fontsize=14, loc="left")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_ylim(0.99, 1.01)
    ax.set_xlim(edges[0], edges[-1])
    rax.set_xlabel(r"$m_{\rm pre}$ [GeV]")
    rax.set_ylabel("model / MC")
    pubhtml.savefig(fig, os.path.join(out, f"07_acceptance{sfx}.pdf"))
    plt.close(fig)
    return mb, A, umc, neff, edges


def fig_meanu(out, gen, kernels, mb, umc, neff, cuts, eta_cut, smear=None,
              sfx=""):
    ok = np.isfinite(umc) & (mb > 0)
    err = umc / np.sqrt(np.maximum(neff, 1.0))
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.errorbar(mb[ok], umc[ok] * 1e3, yerr=err[ok] * 1e3, fmt="o", ms=4,
                color="black", lw=0, elinewidth=0.9, label="MC (selected)")
    for path, lab, col, ls in kernels:
        got = kernel_u_of(path)
        if got is None:
            continue
        m, u, _ = got
        ax.plot(m, u * 1e3, color=col, ls=ls, lw=1.8, label=lab)
        rax.plot(mb[ok], np.interp(mb[ok], m, u) / umc[ok], color=col, ls=ls,
                 lw=1.8)
    ax.set_ylabel(r"$\langle u\rangle$ (selected)  [$10^{-3}$]")
    ax.legend(fontsize=13, frameon=False, loc="upper left")
    ax.set_title("conditional mass loss, " + cutlabel(cuts, smear)
                 + rf", $|\eta| < {eta_cut}$", fontsize=14, loc="left")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_ylim(0.88, 1.12)
    ax.set_xlim(float(mb[ok].min()) - 2.0, float(mb[ok].max()) + 2.0)
    rax.set_xlabel(r"$m_{\rm pre}$ [GeV]")
    rax.set_ylabel("model / MC")
    pubhtml.savefig(fig, os.path.join(out, f"08_mean_u{sfx}.pdf"))
    plt.close(fig)


def corr_table(out, g, path, band=PEAK):
    """The correlated two-leg law against the generator, in one file.

    The marginal identity, the one-photon validation of the sharing density,
    the joint leg tail, and the per-leg marginals -- everything the correlated
    construction has to get right that the collinear product cannot.
    """
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    ww = w[s] / w[s].sum()
    mb = float(np.sum(ww * m[s]))
    z = (g["m_post"].astype(np.float64) / m) ** 2
    u = -0.5 * np.log(np.maximum(z, 1e-300))
    f = (1.0 - g["xp"]) / np.maximum(1.0 - z, 1e-300)
    one = one_photon(g)
    a = -np.log(np.maximum(g["xp"][s], 1e-300))
    b = -np.log(np.maximum(g["xm"][s], 1e-300))

    L = ["the correlated two-leg density against the generator record",
         f"  band {band[0]:.0f}-{band[1]:.0f} GeV, m_bar = {mb:.3f} GeV\n",
         "1. the marginal identity  int dc p_1(z, c) = R_1(z)",
         f"   {'z':>10s} {'quadrature / r1_exact - 1':>28s}"]
    for zz in (0.999, 0.99, 0.9, 0.5, 0.05):
        L.append(f"   {zz:10.4g} {FA.r1_fast(mb, np.array([zz]))[0] / FA.r1_exact(zz, mb) - 1:28.2e}")
    L += ["", "2. the sharing density against the MC's single-photon events",
          f"   {'u range':>16s} {'P(0.01<f<0.99)':>15s} {'model':>9s} "
          f"{'MC/model':>9s} {'MC, all':>9s} {'all/1gam':>9s}"]
    for lo, hi in ((1e-3, 1e-2), (1e-2, 0.05), (0.05, 0.2), (0.2, 0.6)):
        q = s & (u > lo) & (u < hi)
        for tag, sel in (("1", q & one), ("a", q)):
            wq = w[sel]
            p = float(np.sum(wq * ((f[sel] > 0.01) & (f[sel] < 0.99))) / wq.sum())
            if tag == "1":
                p1 = p
                zb = float(np.sum(wq * z[sel]) / wq.sum())
                fm, wm = FA.share_nodes(mb, np.array([zb]), npanel=256, ng=8)
                pm = 2.0 * float(np.sum(wm[0][(fm[0] > 0.01) & (fm[0] < 0.99)]))
            else:
                L.append(f"   [{lo:7.0e},{hi:6.0e}) {p1:15.4f} {pm:9.4f}"
                         f" {p1/pm:9.4f} {p:9.4f} {p/p1:9.4f}")
    K = FA.FSRKernel(mb, variant=DVAR, pair=DPAIR)
    r, wk, _ = K.atoms(var_budget=6e-11)
    zk = r * r
    fm, wm = FA.share_nodes(mb, zk, npanel=128, ng=8)
    xp, xm = PL.share_x(zk, fm)
    UP = -np.log(np.maximum(xp, 1e-300))
    UM = -np.log(np.maximum(xm, 1e-300))
    W = wk[:, None] * wm
    D = PL.LegRadiator(mb, variant=DVAR, pair=DPAIR)
    ue = PL.leg_u_edges(D, n=4000)
    ul, wl = PL.leg_atoms(PL.leg_cells(D, ue), ue)
    wl = wl / wl.sum()
    L += ["", "3. the leg law: marginal tail and the joint against the product",
          f"   {'t':>8s} {'P(u>t) MC':>11s} {'model':>10s} {'D':>10s}"
          f" {'joint/prod MC':>14s} {'model':>9s} {'D(x)D':>7s}"]
    for t in (1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2):
        pm = 0.5 * (float(np.sum(ww * (a > t))) + float(np.sum(ww * (b > t))))
        jm = float(np.sum(ww * ((a > t) & (b > t))))
        pM = float(np.sum(W * ((UP > t).astype(float) + (UM > t))))
        jM = 2.0 * float(np.sum(W * ((UP > t) & (UM > t))))
        pD = float(np.sum(wl * (ul > t)))
        L.append(f"   {t:8.0e} {pm:11.6f} {pM:10.6f} {pD:10.6f}"
                 f" {jm/pm**2:14.4f} {jM/pM**2:9.4f} {1.0:7.4f}")
    L += ["", f"   <u_leg>:  MC {0.5*float(np.sum(ww*(a+b))):.6e}"
          f"   model {float(np.sum(W*(UP+UM))):.6e}"
          f"   D {float(np.sum(wl*ul)):.6e}"]
    txt = "\n".join(L) + "\n"
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)


def band_table(out, gen, kernels, path, cuts=25.0, eta_cut=2.4, smear=None,
               edges=(60, 70, 80, 86, 90, 92, 96, 105, 120, 140)):
    """``A(m)`` and the selected ``<u|m>`` of every model against the MC's own."""
    e = np.asarray(edges, float)
    mb, A, umc, neff = mc_profile(gen, cuts, eta_cut, e, smear)
    err = umc / np.sqrt(np.maximum(neff, 1.0))
    labs, got = [], []
    for p_, lab, _c, _ls in kernels:
        u = kernel_u_of(p_)
        if u is not None:
            labs.append(lab)
            got.append(u)
    L = ["the selection-conditional model against the MC, band by band",
         f"  pT cuts {np.atleast_1d(cuts).tolist()}, |eta| < {eta_cut}, "
         f"smear = {getattr(smear, 'mode', None)};  model / MC of A(m) and of "
         f"the selected <u>\n"]
    for k, lab in enumerate(labs):
        L.append(f"  [{k}] {lab}")
    L.append("")
    L.append(f"  {'band':>10s} {'<m>':>7s} {'A (MC)':>9s} {'<u> (MC)':>11s}"
             f" {'d<u>/<u>':>9s}"
             + "".join(f" {'A [' + str(k) + ']':>9s} {'<u> [' + str(k) + ']':>9s}"
                       for k in range(len(labs))))
    for k in range(len(e) - 1):
        if not np.isfinite(A[k]) or mb[k] <= 0:
            continue
        L.append(f"  {e[k]:4.0f}-{e[k+1]:<5.0f} {mb[k]:7.2f} {A[k]:9.5f}"
                 f" {umc[k]*1e3:9.4f}e-3 {err[k]/umc[k]:9.2e}"
                 + "".join(f" {np.interp(mb[k], g[0], g[2]) / A[k]:9.4f}"
                           f" {np.interp(mb[k], g[0], g[1]) / umc[k]:9.4f}"
                           for g in got))
    tot = np.isfinite(A) & (mb > 0)
    wgt = np.nan_to_num(A[tot])
    L.append("")
    L.append(f"  {'A-weighted':>10s} {'':7s} {'':9s} {'':11s} {'':9s}"
             + "".join(
                 f" {np.average(np.interp(mb[tot], g[0], g[2]) / A[tot], weights=wgt):9.4f}"
                 f" {np.average(np.interp(mb[tot], g[0], g[1]) / umc[tot], weights=wgt):9.4f}"
                 for g in got))
    txt = "\n".join(L) + "\n"
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)


def fig_ksel(out, gen, kernels, band=PEAK, cuts=25.0, eta_cut=2.4,
             smear=None, sfx=""):
    """``K_sel(u|m)`` of the model against the MC's own selected spectrum."""
    d = np.load(gen)
    w = clip(d["weight"].astype(np.float64))
    m = d["m_pre"].astype(np.float64)
    sel = sel_mask(d, cuts, eta_cut, smear)
    s = sel & (m >= band[0]) & (m < band[1])
    u = -np.log(np.maximum(d["m_post"][s].astype(np.float64) / m[s], 1e-300))
    ww = w[s]
    t = np.geomspace(1e-4, 0.5, 40)
    p = np.array([tailw(u, ww, x) for x in t])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.plot(t, p, "o", ms=3.4, color="black", label="MC (selected)")
    for path, lab, col, ls in kernels:
        dd = np.load(path, allow_pickle=False)
        r, wk = dd["r"], dd["w"]
        if "m_lo" in dd:
            lo, hi = dd["m_lo"], dd["m_hi"]
            k = (lo >= band[0] - 1e-9) & (hi <= band[1] + 1e-9)
        else:                      # a kernel with no bands applies everywhere
            k = np.ones(len(r), bool)
        if not k.any():
            continue
        uk = -np.log(r[k])
        wk = wk[k] / wk[k].sum()
        y = np.array([np.sum(wk[uk > x]) for x in t])
        ax.plot(t, y, color=col, ls=ls, lw=1.8, label=lab)
        rax.plot(t, y / p, color=col, ls=ls, lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(u > u_0\,|\,\rm selected)$")
    ax.legend(fontsize=13, frameon=False, loc="lower left")
    ax.set_title(rf"$m_{{pre}}\in[{band[0]:.0f},{band[1]:.0f})$ GeV, "
                 + cutlabel(cuts, smear) + rf", $|\eta| < {eta_cut}$",
                 fontsize=13, loc="left")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.85, 1.15)
    rax.set_xlabel(r"$u_0$")
    rax.set_ylabel("model / MC")
    pubhtml.savefig(fig, os.path.join(out, f"06_ksel_peak{sfx}.pdf"))
    plt.close(fig)


def fig_htable(out, htable, band=91.0):
    with np.load(htable, allow_pickle=False) as d:
        h, be, mb = d["h"], d["b_edges"], d["m_bar"]
    k = int(np.argmin(np.abs(mb - band)))
    prov = json.loads(str(np.load(htable, allow_pickle=False)["provenance"][0]))
    pt_cut = float(prov.get("pt_ref", prov.get("pt_cut")))
    x = pt_cut * np.exp(be)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    # a density in (b_+, b_-) = (ln pT_+, ln pT_-): the grid spacing is not
    # uniform, so the raw per-cell weight would show the spacing, not the physics
    db = np.diff(be)
    z = np.asarray(h[k], float)[:-1, :-1] / (db[:, None] * db[None, :])
    im = ax.pcolormesh(x[:-1], x[:-1], np.where(z > 0, z, np.nan).T,
                       norm="log", cmap="viridis", shading="auto")
    ax.set_xlim(pt_cut, 70)
    ax.set_ylim(pt_cut, 70)
    ax.set_xlabel(r"$p_T^{\rm pre}(\mu^+)$ [GeV]")
    ax.set_ylabel(r"$p_T^{\rm pre}(\mu^-)$ [GeV]")
    ax.set_title(rf"$h(a_+,a_-\,|\,m)$ at $m = {mb[k]:.1f}$ GeV "
                 rf"($a = {pt_cut:.0f}\,\mathrm{{GeV}}/p_T$)",
                 fontsize=13, loc="left")
    fig.colorbar(im, ax=ax,
                 label=r"$\mathrm{d}^2P/\mathrm{d}\ln p_T^+\,\mathrm{d}\ln p_T^-$")
    pubhtml.savefig(fig, os.path.join(out, "09_htable.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_passregion(out, htable, cuts, eta_cut, h4=None, resol=None,
                   band=91.0, name="20_passregion"):
    """``G(u_+, u_-|m)``: which pre-FSR configurations survive the cuts.

    The union of two rectangles for an asymmetric pair of cuts, the same
    surface with the resolution folded in, and their difference -- which is the
    only place the two differ and is 100x smaller than either.
    """
    ht = PL.load_htable(htable)
    k = int(np.argmin([abs(0.5 * (a + b) - band) if np.isfinite(b) else 1e9
                       for a, b in ht["bands"]]))
    be = ht["b_edges"]
    s = (be >= -0.02) & (be <= 1.6)
    G = {"step": PL.PassRegion(ht, cuts=cuts).grid(k)}
    if resol is not None:
        G["resolution"] = PL.PassRegion(ht, cuts=cuts, h4=h4,
                                        resol=resol).grid(k)
        G["resolution $-$ step"] = G["resolution"] - G["step"]
    fig, axs = plt.subplots(1, len(G), figsize=(6.5 * len(G), 5.6))
    for ax, (lab, g) in zip(np.atleast_1d(axs), G.items()):
        d = lab.startswith("resolution $-$")
        v = float(np.max(np.abs(g[np.ix_(s, s)]))) if d else None
        im = ax.pcolormesh(be[s], be[s], g[np.ix_(s, s)],
                           cmap="RdBu_r" if d else "viridis",
                           vmin=-v if d else None, vmax=v if d else None,
                           shading="nearest")
        fig.colorbar(im, ax=ax).set_label(
            r"$\Delta G$" if d else r"$G(u_+, u_-)$")
        for x in PL.cut_shifts(cuts if np.size(cuts) > 1 else (cuts, cuts),
                               ht["pt_ref"]):
            ax.axvline(x, color="k" if d else "w", lw=0.8, ls=":")
            ax.axhline(x, color="k" if d else "w", lw=0.8, ls=":")
        ax.set_xlabel(r"$u_+$")
        ax.set_ylabel(r"$u_-$")
        ax.set_title(f"{lab}, {cutlabel(cuts)}, "
                     + rf"$m_{{\rm pre}} \approx {band:.0f}$ GeV",
                     fontsize=12, loc="left")
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_passprob(out, resol, cuts, name="21_passprob"):
    """The per-leg pass probability: a step against the smeared threshold.

    In the leg's own variable ``v = ln(pT/c)`` both thresholds sit at the same
    place, so the two cuts and the four ``eta`` bands are one family of curves
    and the only thing that separates them is the width.
    """
    c = np.atleast_1d(np.asarray(cuts, float))
    c = [float(np.max(c)), float(np.min(c))]
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    v = np.linspace(-0.22, 0.22, 881)
    for i, cut in enumerate(dict.fromkeys(c)):
        for a in range(len(resol.eta_edges) - 1):
            p = resol.pass_prob(v, cut, a)
            ax.plot(v, p, lw=1.5, ls=["-", "--"][i], color=f"C{a}",
                    label=rf"$c = {cut:.0f}$ GeV, "
                          rf"$|\eta| < {resol.eta_edges[a+1]:.1f}$")
            rax.plot(v, p - (v > 0), lw=1.5, ls=["-", "--"][i], color=f"C{a}")
    ax.axvline(0.0, color="grey", lw=0.8)
    rax.axvline(0.0, color="grey", lw=0.8)
    ax.set_ylabel(r"$P(p_T^{\rm reco} > c)$")
    rax.set_ylabel("smeared $-$ step")
    rax.set_xlabel(r"$v = \ln(p_T^{\rm true} / c)$")
    rax.axhline(0.0, color="grey", lw=0.8)
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper left")
    ax.set_ylim(-0.03, 1.25)
    ax.set_title("per-leg pass probability, " + cutlabel(cuts),
                 fontsize=14, loc="left")
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_fitshifts(out, specs, name="22_fitshifts"):
    """The fit rows of every suite, as offsets from the generator's own values.

    One panel per parameter; one group of points per suite, so the same-run
    differences inside a suite are read off vertically and the four selections
    are not silently compared across suites.
    """
    rows = []
    for spec in specs:
        lab, _, path = spec.partition("=")
        if not os.path.exists(path):
            logger.warning(f"missing {path}")
            continue
        with open(path) as fh:
            rows.append((lab, json.load(fh)))
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(14.0, 0.42 * sum(
        len(r[1]) for r in rows) + 2.6), sharey=True)
    ticks, labels, y = [], [], 0.0
    for j, (suite, d) in enumerate(rows):
        for tag, v in d.items():
            for ax, p in zip(axs, ("m_Z", "Gamma_Z")):
                if p not in v:
                    continue
                ax.errorbar([v[p][0]], [y], xerr=[v[p][1]], fmt="o", ms=5,
                            color=f"C{j}", elinewidth=1.2)
            ticks.append(y)
            labels.append(f"{suite}: {tag}")
            y -= 1.0
        y -= 0.6
    for ax, p, lab in zip(axs, ("m_Z", "Gamma_Z"),
                          (r"$\Delta m_Z$ [MeV]", r"$\Delta\Gamma_Z$ [MeV]")):
        ax.axvline(0.0, color="grey", lw=0.9)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=0.25)
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels(labels, fontsize=9)
    axs[0].set_ylim(y + 0.4, 1.0)
    axs[0].set_title("gen-level closure of the selection-conditional kernel",
                     fontsize=13, loc="left")
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def table(out, g, gen, path, pt_cut=25.0, eta_cut=2.4):
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    mp = g["m_post"].astype(np.float64)
    xp, xm = g["xp"], g["xm"]
    umc = -np.log(np.maximum(mp / m, 1e-300))
    ufac = -0.5 * np.log(np.maximum(xp * xm, 1e-300))
    ptp, ptm = g["ptp"].astype(np.float64), g["ptm"].astype(np.float64)
    ppre = (g["ptp_pre"].astype(np.float64), g["ptm_pre"].astype(np.float64))
    ep, em = np.abs(g["etap"]), np.abs(g["etam"])
    epP, emP = np.abs(g["etap_pre"]), np.abs(g["etam_pre"])
    sel_post = (ptp > pt_cut) & (ptm > pt_cut) & (ep < eta_cut) & (em < eta_cut)
    sel_coll = ((xp * ppre[0] > pt_cut) & (xm * ppre[1] > pt_cut)
                & (epP < eta_cut) & (emP < eta_cut))
    L = []
    L.append("the collinear per-leg picture, measured on the generator record")
    L.append(f"  {len(m)} events, charge-matched legs\n")
    for q in ("p", "m"):
        de = np.abs(g[f"eta{q}"] - g[f"eta{q}_pre"]).astype(np.float64)
        dp = np.abs(g[f"phi{q}"] - g[f"phi{q}_pre"]).astype(np.float64)
        dp = np.minimum(dp, 2.0 * np.pi - dp)
        L.append(f"  leg {q}: P(|d eta| > 1e-3) = {tailw(de, w, 1e-3):.4e}"
                 f"  > 1e-2 {tailw(de, w, 1e-2):.4e}"
                 f"  > 0.1 {tailw(de, w, 0.1):.4e}"
                 f"  | P(|d phi| > 1e-3) = {tailw(dp, w, 1e-3):.4e}")
    L.append("")
    L.append(f"  <u>_true = {wmean(umc, w):.6e}   "
             f"<u>_collinear = {wmean(ufac, w):.6e}   "
             f"ratio-1 = {wmean(ufac, w)/wmean(umc, w)-1:+.3e}   (inclusive)")
    for nm, s in (("fiducial", sel_post),):
        L.append(f"  {nm}: <u>_true = {wmean(umc[s], w[s]):.6e}   "
                 f"<u>_collinear = {wmean(ufac[s], w[s]):.6e}   "
                 f"ratio-1 = {wmean(ufac[s], w[s])/wmean(umc[s], w[s])-1:+.3e}")
    L.append("")
    L.append("  contribution to <u> by region of u_true:")
    L.append(f"  {'u range':>18s} {'P':>10s} {'<u>_true':>12s}"
             f" {'<u>_coll':>12s} {'ratio-1':>10s}")
    for lo, hi in ((0, 1e-3), (1e-3, 1e-2), (1e-2, 0.05), (0.05, 0.2),
                   (0.2, 0.5), (0.5, 10)):
        s = (umc >= lo) & (umc < hi)
        L.append(f"  [{lo:8.4g},{hi:7.4g}) {np.sum(w*s)/np.sum(w):10.5f}"
                 f" {wmean(umc[s], w[s]):12.5e} {wmean(ufac[s], w[s]):12.5e}"
                 f" {wmean(ufac[s], w[s])/wmean(umc[s], w[s])-1:+10.2e}")
    L.append("")
    L.append("  (e) the acceptance decision, collinear prediction vs the truth:")
    L.append(f"    A(post-FSR cut)     = {np.sum(w*sel_post)/np.sum(w):.6f}")
    L.append(f"    A(collinear x p_T)  = {np.sum(w*sel_coll)/np.sum(w):.6f}")
    L.append(f"    per-event mismatch  = "
             f"{np.sum(w*(sel_post != sel_coll))/np.sum(w):.5e}")
    se = (ep < eta_cut) & (em < eta_cut)
    sP = (epP < eta_cut) & (emP < eta_cut)
    L.append(f"      of which eta only = "
             f"{np.sum(w*(se != sP))/np.sum(w):.5e}")
    s1 = (ptp > pt_cut) & (ptm > pt_cut)
    s2 = (xp * ppre[0] > pt_cut) & (xm * ppre[1] > pt_cut)
    L.append(f"      of which p_T only = "
             f"{np.sum(w*(s1 != s2))/np.sum(w):.5e}")
    L.append("")
    L.append("  (c) leg independence, peak band: joint tail / product of tails")
    s = (m >= PEAK[0]) & (m < PEAK[1])
    a = -np.log(np.maximum(xp[s], 1e-300))
    b = -np.log(np.maximum(xm[s], 1e-300))
    ww = w[s]
    for t in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2):
        pa, pb = tailw(a, ww, t), tailw(b, ww, t)
        pab = float(np.sum(ww * ((a > t) & (b > t))) / ww.sum())
        L.append(f"    t = {t:8.0e}: P_joint = {pab:.5e}   product = "
                 f"{pa*pb:.5e}   ratio = {pab/(pa*pb):.4f}")
    L.append("")
    L.append("  the h table reproduces G exactly on its grid: "
             "G(u_+,u_-) from the table vs a direct event count, m = 91-92 GeV")
    L += _check_G(gen, pt_cut, eta_cut)
    txt = "\n".join(L) + "\n"
    with open(path, "w") as fh:
        fh.write(txt)
    print(txt)


def _check_G(gen, pt_cut, eta_cut, tables=("data/ht_pt25_1.0gev.npz",
                                            "data/ht_pt25_1.0gev_thin4.npz"),
             band=(91.0, 92.0)):
    """``G`` read off the ``h`` table against a direct count of the same events."""
    d = np.load(gen)
    w = clip(d["weight"].astype(np.float64))
    m = d["m_pre"].astype(np.float64)
    inb = (m >= band[0]) & (m < band[1])
    s = inb & (np.abs(d["eta1_pre"]) < eta_cut) & (np.abs(d["eta2_pre"]) < eta_cut)
    tot = w[inb].sum()
    a, b, wv = (d["pt1_pre"][s].astype(np.float64),
                d["pt2_pre"][s].astype(np.float64), w[s])
    tabs = []
    for f in tables:
        if not os.path.exists(f):
            continue
        with np.load(f, allow_pickle=False) as D:
            k = int(np.argmin(np.abs(D["m_bar"] - 0.5 * (band[0] + band[1]))))
            tabs.append((os.path.basename(f),
                         PL.survival(np.asarray(D["h"][k], float), D["b_edges"]),
                         D["b_edges"]))
    L = [f"    {'u_+':>7s} {'u_-':>7s} {'G direct':>11s}"
         + "".join(f" {t[0][:20]:>22s}" for t in tabs)]
    for u1, u2 in ((0, 0), (1e-4, 0), (1e-3, 1e-3), (0.01, 0.0), (0.05, 0.01),
                   (0.1, 0.1), (0.3, 0.05), (0.5, 0.5), (1.0, 0.2)):
        t1, t2 = pt_cut * np.exp(u1), pt_cut * np.exp(u2)
        gd = 0.5 * (np.sum(wv * ((a > t1) & (b > t2)))
                    + np.sum(wv * ((a > t2) & (b > t1)))) / tot
        row = ""
        for _, g, be in tabs:
            gv = float(PL.eval_G(g, be, np.array([u1, u2]))[0, 1])
            row += f" {gv:11.7f} {gv - gd:+10.2e}"
        L.append(f"    {u1:7.4g} {u2:7.4g} {gd:11.7f}" + row)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perleg", default="data/perleg_full.npz")
    ap.add_argument("--gen", default="data/genmerged_full.npz")
    ap.add_argument("--htable", default="data/ht_pt25_1.0gev.npz")
    ap.add_argument("--run", default="data/photos/gen_mcMix.npz",
                    help="standalone Photos run for the numerical leg D")
    ap.add_argument("--kernel", action="append", default=[],
                    help="label=path.npz (repeatable)")
    ap.add_argument("--pt-cut", type=float, default=25.0)
    ap.add_argument("--pt-cuts", nargs="*", type=float, default=None,
                    help="leading and trailing pT thresholds [GeV]")
    ap.add_argument("--eta-cut", type=float, default=2.4)
    ap.add_argument("--h4", default=None)
    ap.add_argument("--resol", default=None,
                    help="a `ptres.py` npz: the cuts act on the smeared pT")
    ap.add_argument("--resol-mode", default="shape", choices=("shape", "gauss"))
    ap.add_argument("--fit", action="append", default=[],
                    help="label=fit_selection_X.json (repeatable): the fit "
                         "shift summary panel")
    ap.add_argument("--suffix", default="",
                    help="appended to every file name of --selection-only, so "
                         "two selections can share a figure directory")
    ap.add_argument("--selection-only", action="store_true",
                    help="only the panels that depend on the selection")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--tag", default="fsr_perleg")
    ap.add_argument("--no-table", action="store_true",
                    help="skip the 20.8 M-event moment table (figures only)")
    args = ap.parse_args()

    out = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today():%y%m%d}_{args.tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out, logger=logger)
    logger.info(f"figures -> {out}")

    style = [("C0", "-"), ("C3", "--"), ("C2", ":"), ("C1", "-.")]
    kernels = []
    for i, spec in enumerate(args.kernel):
        lab, _, path = spec.partition("=")
        kernels.append((path, lab, *style[i % len(style)]))

    cuts = args.pt_cuts if args.pt_cuts else [args.pt_cut]
    resol = FG.load_resolution(args.resol, args.resol_mode)
    h4 = PL.load_h4table(args.h4) if args.h4 else None
    if args.selection_only:
        sfx = args.suffix
        if args.fit:
            fig_fitshifts(out, args.fit)
        if resol is not None:
            fig_passprob(out, resol, cuts, name=f"21_passprob{sfx}")
        fig_passregion(out, args.htable, cuts, args.eta_cut, h4, resol,
                       name=f"20_passregion{sfx}")
        if kernels:
            fig_ksel(out, args.gen, kernels, PEAK, cuts, args.eta_cut, resol,
                     sfx)
            mb, A, umc, neff, _ = fig_acceptance(out, args.gen, kernels, cuts,
                                                 args.eta_cut, resol, sfx)
            fig_meanu(out, args.gen, kernels, mb, umc, neff, cuts,
                      args.eta_cut, resol, sfx)
            band_table(out, args.gen, kernels,
                       os.path.join(out, f"00_bands{sfx}.txt"), cuts,
                       args.eta_cut, resol)
        return

    g = load_perleg(args.perleg)
    logger.info(f"{len(g['m_pre'])} per-leg events")
    if not args.no_table:
        table(out, g, args.gen, os.path.join(out, "00_perleg.txt"),
              args.pt_cut, args.eta_cut)
    fig_angles(out, g)
    fig_leg_x(out, g)
    fig_leg_corr(out, g)
    fig_z_vs_xx(out, g)
    fig_share(out, g)
    fig_share_u(out, g, kernels)
    fig_joint(out, g)
    if not args.no_table:
        corr_table(out, g, os.path.join(out, "00_corr.txt"))
    fig_dconvd(out)
    if os.path.exists(args.run):
        fig_leg_D_mc(out, g, args.run)
        fig_dconvd_mc(out, args.run)
    if kernels:
        fig_ksel(out, args.gen, kernels, PEAK, cuts, args.eta_cut)
        mb, A, umc, neff, _ = fig_acceptance(out, args.gen, kernels,
                                             cuts, args.eta_cut)
        fig_meanu(out, args.gen, kernels, mb, umc, neff, cuts, args.eta_cut)
        band_table(out, args.gen, kernels, os.path.join(out, "00_bands.txt"),
                   cuts, args.eta_cut)
    if os.path.exists(args.htable):
        fig_htable(out, args.htable)


if __name__ == "__main__":
    main()
