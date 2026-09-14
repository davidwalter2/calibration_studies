#!/usr/bin/env python3
"""Analytic FSR kernel vs the Photos++ generator record.

Weighted by the clipped MiNNLO weights (`fit_gen.clip_weights`).  Statistical
errors are half-sample splits: |a - b|/2 of the two halves of the sample.

One file per panel, ratio panel under every density-vs-model plot,
mplhep ROOT style, output under
``~/public_html/ZMass/cvh/<date>_fsr_analytic/``.
"""
import argparse
import datetime
import math
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging

import pubhtml
import ratiopanel

import fit_gen as FG
import fsr_analytic as FA

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

#: m_pre bands for the mass-dependence test
BANDS = [(50, 60), (60, 70), (70, 80), (80, 86), (86, 96), (96, 110),
         (110, 130), (130, 150), (150, 200)]
#: the variants shown against the MC
SHOW = [("exp1", "exp. O($\\alpha$)", "C3", "-"),
        ("exp2", "exp. O($\\alpha$) + O($\\alpha^2$)LL", "C0", "--"),
        ("oalpha", "O($\\alpha$), no exponentiation", "C2", ":")]


def wmean(a, w):
    return float(np.sum(w * a) / np.sum(w))


def half_split(fun, half):
    """|f(a) - f(b)|/2 over the two halves of the sample."""
    a, b = fun(half), fun(~half)
    return abs(a - b) / 2.0


def load(path, wclip=100.0):
    g = FG.load_gen(path, ["m_pre", "m_post", "weight", "npre"])
    w = FG.clip_weights(g["weight"].astype(np.float64), wclip)
    r = np.minimum(g["m_post"] / g["m_pre"], 1.0)
    # ``npre`` is the number of status-746 (pre-Photos) muon copies: Photos
    # writes them only when it modified the muons, so ``npre == 0`` is the
    # sample's own "no photon above the IR cutoff" flag.  It is cleaner than a
    # cut on u: the MiniAOD float precision puts |u| <= 6.8e-7 on those events
    # while genuinely radiated events reach down to u = 0.
    return g["m_pre"], -np.log(r), w, g["npre"] == 0


# --------------------------------------------------------------------------
def fig_density(out, u, w, m_pre, band=(86, 96), nbin=60, umin=1e-4, umax=1.0):
    sel = (m_pre >= band[0]) & (m_pre < band[1])
    uu, ww = u[sel], w[sel]
    mm = wmean(m_pre[sel], ww)
    tot = ww.sum()
    e = np.geomspace(umin, umax, nbin + 1)
    c = np.sqrt(e[1:] * e[:-1])
    bw = np.diff(e)
    i = np.clip(np.searchsorted(e, uu, "right") - 1, 0, nbin - 1)
    ok = (uu >= e[0]) & (uu < e[-1])
    W = np.bincount(i[ok], ww[ok], nbin)
    W2 = np.bincount(i[ok], ww[ok] ** 2, nbin)
    dens = W / (tot * bw)
    err = np.sqrt(W2) / (tot * bw)

    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.errorbar(c, dens, yerr=err, fmt="o", ms=3.0, color="black", lw=0,
                elinewidth=0.9, label="Photos++ (generated)")
    models = {}
    for name, lab, col, ls in SHOW:
        k = FA.FSRKernel(mm, variant=name)
        y = ratiopanel.bin_average(k.pdf_u(e[:-1]), k.pdf_u(c), k.pdf_u(e[1:]))
        models[name] = y
        ax.plot(c, y, color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u$")
    ax.legend(loc="lower left", fontsize=14, frameon=False)
    ax.set_title(f"$u=-\\ln(m_{{post}}/m_{{pre}})$, "
                 f"$m_{{pre}}\\in[{band[0]},{band[1]})$ GeV, "
                 f"$\\beta={FA.beta_fsr(mm):.5f}$", fontsize=14, loc="left")
    for name, lab, col, ls in SHOW:
        rax.plot(c, dens / models[name], color=col, ls=ls, lw=1.6)
    rax.errorbar(c, dens / models["exp1"], yerr=err / models["exp1"],
                 fmt="o", ms=2.6, color="black", lw=0, elinewidth=0.8)
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.90, 1.10)
    rax.set_xlabel("$u$")
    rax.set_ylabel("MC / model")
    pubhtml.savefig(fig, os.path.join(out, "01_u_density.pdf"))
    plt.close(fig)


def fig_tail(out, u, w, m_pre, band=(86, 96)):
    sel = (m_pre >= band[0]) & (m_pre < band[1])
    uu, ww = u[sel], w[sel]
    mm = wmean(m_pre[sel], ww)
    half = np.arange(len(uu)) < len(uu) // 2
    u0 = np.geomspace(1e-4, 0.5, 40)
    p = np.array([ww[uu > x].sum() / ww.sum() for x in u0])
    pe = np.array([half_split(lambda h, x=x: ww[h & (uu > x)].sum() / ww[h].sum(),
                              half) for x in u0])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.errorbar(u0, p, yerr=pe, fmt="o", ms=3.0, color="black", lw=0,
                elinewidth=0.9, label="Photos++ (generated)")
    mod = {}
    for name, lab, col, ls in SHOW:
        k = FA.FSRKernel(mm, variant=name)
        mod[name] = k.tail(u0)
        ax.plot(u0, mod[name], color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("$P(u > u_0)$")
    ax.legend(loc="lower left", fontsize=14, frameon=False)
    ax.set_title(f"IR-safe tail, $m_{{pre}}\\in[{band[0]},{band[1]})$ GeV",
                 fontsize=14, loc="left")
    for name, lab, col, ls in SHOW:
        rax.plot(u0, p / mod[name], color=col, ls=ls, lw=1.6)
    rax.errorbar(u0, p / mod["exp1"], yerr=pe / mod["exp1"], fmt="o", ms=2.6,
                 color="black", lw=0, elinewidth=0.8)
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.95, 1.15)
    rax.set_xlabel("$u_0$")
    rax.set_ylabel("MC / model")
    pubhtml.savefig(fig, os.path.join(out, "02_u_tail.pdf"))
    plt.close(fig)


def _band_stat(u, w, m_pre, fun):
    """Per-band (m, value, error) of ``fun(u, w)``."""
    m, v, e = [], [], []
    for lo, hi in BANDS:
        s = (m_pre >= lo) & (m_pre < hi)
        if s.sum() < 1000:
            continue
        uu, ww = u[s], w[s]
        half = np.arange(len(uu)) < len(uu) // 2
        m.append(wmean(m_pre[s], ww))
        v.append(fun(uu, ww))
        e.append(half_split(lambda h: fun(uu[h], ww[h]), half))
    return np.array(m), np.array(v), np.array(e)


def fig_mass_dep(out, u, w, m_pre):
    specs = [("03_mpre_mean_u", lambda uu, ww: wmean(uu, ww),
              lambda k: k.moments()["u"], r"$\langle u\rangle$", 1e3, "$10^{-3}$"),
             ("04_mpre_tail_0p01", lambda uu, ww: ww[uu > 0.01].sum() / ww.sum(),
              lambda k: float(k.tail([0.01])[0]), "$P(u>0.01)$", 1.0, ""),
             ("05_mpre_tail_0p05", lambda uu, ww: ww[uu > 0.05].sum() / ww.sum(),
              lambda k: float(k.tail([0.05])[0]), "$P(u>0.05)$", 1.0, ""),
             ("06_mpre_mean_x", None, None, r"$\langle 1-z\rangle$", 1e3, "$10^{-3}$")]
    for name, fmc, fan, lab, sc, unit in specs:
        if fmc is None:
            fmc = lambda uu, ww: wmean(1.0 - np.exp(-2.0 * uu), ww)
            fan = lambda k: k.moments()["x"]
        m, v, e = _band_stat(u, w, m_pre, fmc)
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
        ax.errorbar(m, v * sc, yerr=e * sc, fmt="o", ms=5, color="black", lw=0,
                    elinewidth=1.2, label="Photos++ (generated)")
        mod = {}
        mg = np.linspace(50, 200, 80)
        for nm, l, col, ls in SHOW:
            mod[nm] = np.array([fan(FA.FSRKernel(mm, variant=nm)) for mm in m])
            ax.plot(mg, [fan(FA.FSRKernel(mm, variant=nm)) * sc for mm in mg],
                    color=col, ls=ls, lw=1.8, label=l)
        kf = [fan(FA.FSRKernel(mm, variant="exp1", beta_at=91.1535)) * sc
              for mm in mg]
        ax.plot(mg, kf, color="C1", ls="-.", lw=1.5,
                label=r"exp. O($\alpha$), $\beta$ frozen at $m_Z$")
        ax.set_ylabel(lab + (f"  [{unit}]" if unit else ""))
        ax.legend(loc="upper left", fontsize=13, frameon=False)
        ax.set_title("mass dependence of the radiator", fontsize=14, loc="left")
        for nm, l, col, ls in SHOW:
            rax.plot(m, v / mod[nm], color=col, ls=ls, lw=1.4, marker="o", ms=4)
        rax.errorbar(m, v / mod["exp1"], yerr=e / mod["exp1"], fmt="o", ms=4,
                     color="black", lw=0, elinewidth=1.0)
        rax.axhline(1.0, color="grey", lw=0.8)
        rax.set_ylim(0.95, 1.08)
        rax.set_xlabel(r"$m_{pre}$  [GeV]")
        rax.set_ylabel("MC / model")
        pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
        plt.close(fig)


def fig_band_shape(out, u, w, m_pre, pair=((60, 80), (110, 150)), nbin=40):
    e = np.geomspace(1e-3, 1.0, nbin + 1)
    c = np.sqrt(e[1:] * e[:-1])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    cols = ["C3", "C0"]
    for j, (lo, hi) in enumerate(pair):
        s = (m_pre >= lo) & (m_pre < hi)
        uu, ww = u[s], w[s]
        mm = wmean(m_pre[s], ww)
        tot = ww.sum()
        i = np.clip(np.searchsorted(e, uu, "right") - 1, 0, nbin - 1)
        ok = (uu >= e[0]) & (uu < e[-1])
        W = np.bincount(i[ok], ww[ok], nbin)
        W2 = np.bincount(i[ok], ww[ok] ** 2, nbin)
        d = W / (tot * np.diff(e))
        er = np.sqrt(W2) / (tot * np.diff(e))
        k = FA.FSRKernel(mm, variant="exp1")
        mo = ratiopanel.bin_average(k.pdf_u(e[:-1]), k.pdf_u(c), k.pdf_u(e[1:]))
        ax.errorbar(c, d, yerr=er, fmt="o", ms=3, color=cols[j], lw=0,
                    elinewidth=0.9, label=f"MC $m_{{pre}}\\in[{lo},{hi})$")
        ax.plot(c, mo, color=cols[j], lw=1.6,
                label=f"exp. O($\\alpha$), $\\beta(m={mm:.1f})$")
        rax.errorbar(c, d / mo, yerr=er / mo, fmt="o", ms=3, color=cols[j],
                     lw=0, elinewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u$")
    ax.legend(loc="lower left", fontsize=13, frameon=False)
    ax.set_title("per-band shape", fontsize=14, loc="left")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.80, 1.25)
    rax.set_xlabel("$u$")
    rax.set_ylabel("MC / model")
    pubhtml.savefig(fig, os.path.join(out, "07_band_shape.pdf"))
    plt.close(fig)


def fig_variants(out, m=91.1876):
    u = np.geomspace(1e-6, 3.0, 400)
    base = FA.FSRKernel(m, variant="exp1")
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    rows = [("exp. O($\\alpha$)", dict(variant="exp1"), "C3", "-"),
            ("+ O($\\alpha^2$)LL", dict(variant="exp2"), "C0", "--"),
            ("+ $e^+e^-$ pairs", dict(variant="exp1", pair=("e",)), "C2", "-."),
            ("+ all pairs", dict(variant="exp1",
                                 pair=("e", "mu", "tau", "had")), "C4", ":"),
            (r"$L$ instead of $L-1$", dict(variant="exp1", minus_one=False),
             "C1", "-")]
    b0 = base.pdf_u(u)
    for lab, kw, col, ls in rows:
        y = FA.FSRKernel(m, **kw).pdf_u(u)
        ax.plot(u, y, color=col, ls=ls, lw=1.7, label=lab)
        rax.plot(u, y / b0, color=col, ls=ls, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_u(u)$")
    ax.legend(loc="lower left", fontsize=13, frameon=False)
    ax.set_title(f"analytic variants at $m={m}$ GeV", fontsize=14, loc="left")
    rax.axhline(1.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_ylim(0.94, 1.18)
    rax.set_xlabel("$u$")
    rax.set_ylabel("/ exp. O($\\alpha$)")
    pubhtml.savefig(fig, os.path.join(out, "08_variants.pdf"))
    plt.close(fig)


def table(out, u, w, m_pre, norad, path):
    lines = []
    A = lines.append
    A("Moments and tails, MiNNLO-weighted, half-sample errors.")
    A("")
    A(f"{'band':>12s} {'N':>10s} {'<m_pre>':>8s} {'beta':>8s} "
      f"{'<u> MC':>11s} {'+-':>8s} {'exp1':>10s} {'exp2':>10s} "
      f"{'MC/exp1':>8s} {'MC/exp2':>8s}")
    for lo, hi in BANDS:
        s = (m_pre >= lo) & (m_pre < hi)
        if s.sum() < 1000:
            continue
        uu, ww = u[s], w[s]
        half = np.arange(len(uu)) < len(uu) // 2
        mm = wmean(m_pre[s], ww)
        mc = wmean(uu, ww)
        er = half_split(lambda h: wmean(uu[h], ww[h]), half)
        a1 = FA.FSRKernel(mm, "exp1").moments()["u"]
        a2 = FA.FSRKernel(mm, "exp2").moments()["u"]
        A(f"{lo:5.0f}-{hi:<6.0f} {s.sum():10d} {mm:8.3f} "
          f"{FA.beta_fsr(mm):8.5f} {mc*1e3:11.4f} {er*1e3:8.4f} "
          f"{a1*1e3:10.4f} {a2*1e3:10.4f} {mc/a1:8.4f} {mc/a2:8.4f}")
    A("")
    A(f"{'band':>12s} {'<1-z> MC':>11s} {'+-':>8s} {'exp1':>10s} {'exp2':>10s} "
      f"{'MC/exp1':>8s} {'MC/exp2':>8s} {'<u^2> MC':>11s} {'+-':>9s}"
      f" {'exp1':>10s} {'MC/exp1':>8s}")
    for lo, hi in BANDS:
        s = (m_pre >= lo) & (m_pre < hi)
        if s.sum() < 1000:
            continue
        uu, ww = u[s], w[s]
        xx = 1.0 - np.exp(-2.0 * uu)
        half = np.arange(len(uu)) < len(uu) // 2
        mm = wmean(m_pre[s], ww)
        mc = wmean(xx, ww)
        er = half_split(lambda h: wmean(xx[h], ww[h]), half)
        a1 = FA.FSRKernel(mm, "exp1").moments()["x"]
        a2 = FA.FSRKernel(mm, "exp2").moments()["x"]
        q = wmean(uu * uu, ww)
        qe = half_split(lambda h: wmean(uu[h] * uu[h], ww[h]), half)
        q1 = FA.FSRKernel(mm, "exp1").moments()["u2"]
        A(f"{lo:5.0f}-{hi:<6.0f} {mc*1e3:11.4f} {er*1e3:8.4f} "
          f"{a1*1e3:10.4f} {a2*1e3:10.4f} {mc/a1:8.4f} {mc/a2:8.4f}"
          f" {q*1e3:11.4f} {qe*1e3:9.4f} {q1*1e3:10.4f} {q/q1:8.4f}")
    A("")
    s = (m_pre >= 86) & (m_pre < 96)
    uu, ww = u[s], w[s]
    mm = wmean(m_pre[s], ww)
    half = np.arange(len(uu)) < len(uu) // 2
    ks = {n: FA.FSRKernel(mm, variant=n) for n, *_ in SHOW}
    A(f"tails at m_pre in [86,96), <m_pre> = {mm:.4f} GeV")
    A(f"{'u0':>10s} {'P MC':>11s} {'+-':>9s} " +
      " ".join(f"{n:>11s}" for n, *_ in SHOW) +
      " " + " ".join(f"{'MC/'+n:>9s}" for n, *_ in SHOW))
    for x in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 0.1,
              0.2, 0.3, 0.5, 1.0):
        p = ww[uu > x].sum() / ww.sum()
        e = half_split(lambda h: ww[h & (uu > x)].sum() / ww[h].sum(), half)
        t = {n: float(ks[n].tail([x])[0]) for n, *_ in SHOW}
        A(f"{x:10.1e} {p:11.6f} {e:9.2e} " +
          " ".join(f"{t[n]:11.6f}" for n, *_ in SHOW) + " " +
          " ".join(f"{p/t[n]:9.4f}" for n, *_ in SHOW))
    A("")
    A("unradiated fraction.  Photos++ 3.61 runs with exponentiation on and")
    A("XPHCUT = 1e-7 on x = 2 E_gamma/m, and 1-z = 2 E_gamma/m exactly for a")
    A("single emission, so the model's P(1-z < 1e-7) is the same quantity as")
    A("the MC's fraction with no status-746 (pre-Photos) muon copy.")
    A(f"{'band':>12s} {'P0 MC':>10s} {'exp1':>10s} {'exp2':>10s} {'oalpha':>10s}"
      f" {'MC/exp1':>8s}")
    for lo, hi in BANDS:
        sb = (m_pre >= lo) & (m_pre < hi)
        if sb.sum() < 1000:
            continue
        mb = wmean(m_pre[sb], w[sb])
        p0 = w[sb & norad].sum() / w[sb].sum()
        a1 = FA.FSRKernel(mb, "exp1").p_norad(1e-7)
        a2 = FA.FSRKernel(mb, "exp2").p_norad(1e-7)
        ao = FA.FSRKernel(mb, "oalpha").p_norad(1e-7)
        A(f"{lo:5.0f}-{hi:<6.0f} {p0:10.6f} {a1:10.6f} {a2:10.6f} {ao:10.6f}"
          f" {p0/a1:8.4f}")
    p0 = w[norad].sum() / w.sum()
    A(f"   inclusive MC P0 = {p0:.6f}")
    A("")
    A("README mass-dependence ratio  <u>(110-150) / <u>(60-80):")
    r = []
    for lo, hi in ((60, 80), (110, 150)):
        sb = (m_pre >= lo) & (m_pre < hi)
        uu2, ww2 = u[sb], w[sb]
        hh = np.arange(len(uu2)) < len(uu2) // 2
        mb = wmean(m_pre[sb], ww2)
        r.append((wmean(uu2, ww2),
                  half_split(lambda h: wmean(uu2[h], ww2[h]), hh),
                  FA.FSRKernel(mb, "exp1").moments()["u"], mb))
    rr = r[1][0] / r[0][0]
    re = rr * math.hypot(r[1][1] / r[1][0], r[0][1] / r[0][0])
    A(f"   MC        {rr:.4f} +- {re:.4f}")
    A(f"   exp1      {r[1][2] / r[0][2]:.4f}")
    A(f"   (L(m2)-1)/(L(m1)-1) with m = <m_pre> of each band: "
      f"{(FA.coll_log(r[1][3])-1)/(FA.coll_log(r[0][3])-1):.4f}")
    txt = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(txt + "\n")
    print(txt)


# --------------------------------------------------------------------------
# the O(alpha^2) NLL term  (`--nll`: model only, no generator record needed)
# --------------------------------------------------------------------------
def fig_nll_ratio(out, mass=91.1053, umin=1e-5, umax=3.0, n=400):
    """exp2nll / exp2 as a density ratio in u, with the densities above it."""
    uu = np.geomspace(umin, umax, n)
    k2 = FA.FSRKernel(mass, variant="exp2")
    kn = FA.FSRKernel(mass, variant="exp2nll")
    k1 = FA.FSRKernel(mass, variant="exp1")
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    for k, lab, col, ls in ((k1, "exp. O($\\alpha$)", "C3", "-"),
                            (k2, "+ O($\\alpha^2$) LL", "C0", "--"),
                            (kn, "+ O($\\alpha^2$) NLL", "C2", "-.")):
        ax.plot(uu, k.pdf_u(uu), color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_u(u)$")
    ax.legend(loc="lower left", fontsize=14, frameon=False)
    ax.set_title(f"$m_{{pre}} = {mass:.2f}$ GeV, "
                 f"$\\beta = {FA.beta_fsr(mass):.5f}$, "
                 f"$(\\alpha/\\pi)^2 L = {kn._c_nll:.3e}$",
                 fontsize=14, loc="left")
    d2 = k2.pdf_u(uu)
    # the ratio spans +0.3 per mille in the peak region and -90 in the far
    # tail, so the ratio panel is symlog in (ratio - 1) x 1e3
    rax.plot(uu, 1e3 * (kn.pdf_u(uu) / d2 - 1.0), color="C2", ls="-.", lw=1.8)
    rax.plot(uu, 1e3 * (k1.pdf_u(uu) / d2 - 1.0), color="C3", ls="-", lw=1.4)
    rax.axhline(0.0, color="grey", lw=0.8)
    rax.set_xscale("log")
    rax.set_yscale("symlog", linthresh=1.0)
    rax.set_ylim(-200.0, 60.0)
    rax.set_xlabel("$u$")
    rax.set_ylabel(r"$10^3\,(K/K_{\rm exp2} - 1)$", fontsize=13)
    pubhtml.savefig(fig, os.path.join(out, "01_nll_ratio.pdf"))
    plt.close(fig)


def fig_nll_coeff(out, n=1200):
    """G(z) of eq. (8) and the three objects it is built from."""
    t = np.geomspace(1e-5, 0.5, n)            # panels shared by both ends
    z = np.concatenate([t, 1.0 - t[::-1]])
    x = np.concatenate([1.0 - t, t[::-1]])
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.plot(z, FA.nll_reg(z, x), color="k", lw=2.2,
            label=r"$G(z)$  (the O($\alpha^2 L$) coefficient)")
    ax.plot(z, 0.5 * FA.p1_timelike(z, x), color="C0", ls="--", lw=1.6,
            label=r"$\frac{1}{2} P^{(1),T}_{\rm NS,-}(z)$")
    ax.plot(z, FA.conv_p0_pln(z, x), color="C3", ls="-.", lw=1.6,
            label=r"$A(z) = [P_0 \otimes \hat{C}_{\rm reg}](z)$")
    ax.plot(z, -FA.KAPPA * (1.0 + z), color="C2", ls=":", lw=1.6,
            label=r"$-(\pi^2/3-5/4)(1+z)$")
    ax.plot(z, -0.5 * FA.p_minus(z) * FA.s2_cfp(z, x), color="C4", ls=(0, (3, 1, 1, 1)),
            lw=1.3, label=r"(crossed-photon $-p(-z)S_2(z)$ part of $\frac{1}{2}P^{(1)}$)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("$z = (m_{post}/m_{pre})^2$")
    ax.set_ylabel("coefficient of $(\\alpha/\\pi)^2 L$")
    ax.set_ylim(-250, 120)
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    ax.set_title(r"$\Delta K = (\alpha/\pi)^2 L\,[\,G(z) + g_\delta\,\delta(1-z)\,]$,"
                 f"  $g_\\delta = {FA.G_DELTA:.6f}$", fontsize=14, loc="left")
    pubhtml.savefig(fig, os.path.join(out, "02_nll_coefficient.pdf"))
    plt.close(fig)


def fig_nll_mellin(out, nmax=8):
    """The two Mellin-space checks of the NLL term, as residuals."""
    ns = np.arange(1, nmax + 1)
    jn = lambda n: FA._mellin(lambda z, x: (1.0 + z * z) / x * FA._lnz(z, x), n)
    r_ad, r_l2, r_l1 = [], [], []
    for n in ns:
        S1 = FA._harm(1, n) - 1.0 / n
        S2 = FA._harm(2, n) - 1.0 / n**2
        g0 = 1.5 - 2.0 * S1 - 1.0 / n - 1.0 / (n + 1.0)
        c1 = jn(n) + FA.KAPPA - g0
        g1 = FA._mellin(FA.p1_timelike, n) + FA.P1_DELTA
        gq, gn = FA._mellin(FA._q2_reg, n), FA._mellin(FA.nll_reg, n)
        r_ad.append(abs(FA._mellin(FA.p1_spacelike, n) + FA.P1_DELTA
                        + 0.25 * FA.gamma1_ns_minus_abelian(float(n))))

        def coef2(L):
            b1 = 1.5 * (L - 1.0) + FA.KAPPA
            b2 = -0.5 * (L - 1.0) ** 2 * FA.INT_Q2 + FA.G_DELTA * L
            return (b2 - 2.0 * b1 * (L - 1.0) * S1
                    + 4.0 * (L - 1.0) ** 2 * (0.5 * S2 + 0.5 * S1 * S1)
                    + 0.5 * (L - 1.0) ** 2 * gq + L * gn)
        V = np.linalg.solve(np.array([[L * L, L, 1.0] for L in (1.0, 2.0, 3.0)]),
                            np.array([coef2(L) for L in (1.0, 2.0, 3.0)]))
        r_l2.append(abs(V[0] - 0.5 * g0 * g0))
        r_l1.append(abs(V[1] - (0.5 * g1 + g0 * c1)))
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    floor = 1e-17
    ax.plot(ns, np.maximum(r_ad, floor), "o-", color="C0", ms=7,
            label=r"$M[P^{(1),S}](n) + \gamma^{(1)}_{\rm NS,-}(n)/4$")
    ax.plot(ns, np.maximum(r_l2, floor), "s--", color="C3", ms=7,
            label=r"$L^2$ coefficient $-\ \frac{1}{2}\,g_0(n)^2$")
    ax.plot(ns, np.maximum(r_l1, floor), "^-.", color="C2", ms=7,
            label=r"$L$ coefficient $-\ [\frac{1}{2}g_1(n) + g_0(n)c_1(n)]$")
    ax.axhline(1e-10, color="grey", lw=0.8, ls=":")
    ax.text(nmax, 1.3e-10, "1e-10 target", ha="right", va="bottom",
            fontsize=12, color="grey")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-8)
    ax.set_xlabel("Mellin moment $n$")
    ax.set_ylabel("|residual|")
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    ax.set_title("two-loop anomalous dimension and the O($\\alpha^2$) "
                 "expansion of the kernel", fontsize=14, loc="left")
    pubhtml.savefig(fig, os.path.join(out, "03_nll_mellin.pdf"))
    plt.close(fig)


def table_nll(path):
    """Moments and tails of exp2nll against exp2, on the `00_moments.txt` grid."""
    lines = []
    A = lambda t="": (lines.append(t), None)[1]
    A("The O(alpha^2) NLL term: exp2nll against exp2 (analytic, no MC).")
    A("")
    A(f"{'band':>12s}{'m_pre':>9s}{'beta':>9s}"
      f"{'<u> exp2':>12s}{'<u> nll':>12s}{'ratio':>9s}"
      f"{'<u2> exp2':>12s}{'<u2> nll':>12s}{'ratio':>9s}"
      f"{'<1-z> nll/exp2':>16s}")
    for lo, hi, mb in ((50, 60, 54.585), (60, 70, 64.914), (70, 80, 75.599),
                       (80, 86, 83.657), (86, 96, 91.105), (96, 110, 100.087),
                       (110, 130, 117.715), (130, 150, 138.566),
                       (150, 200, 169.240)):
        a = FA.FSRKernel(mb, variant="exp2").moments()
        b = FA.FSRKernel(mb, variant="exp2nll").moments()
        A(f"{f'{lo}-{hi}':>12s}{mb:9.3f}{FA.beta_fsr(mb):9.5f}"
          f"{a['u']*1e3:12.4f}{b['u']*1e3:12.4f}{b['u']/a['u']:9.5f}"
          f"{a['u2']*1e3:12.4f}{b['u2']*1e3:12.4f}{b['u2']/a['u2']:9.5f}"
          f"{b['x']/a['x']:16.5f}")
    A("(<u>, <u^2>, <1-z> in units of 1e-3)")
    A("")
    A("tails at the peak band, m_pre = 91.1053 GeV")
    k2 = FA.FSRKernel(91.1053, variant="exp2")
    kn = FA.FSRKernel(91.1053, variant="exp2nll")
    u0 = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2,
                   1e-1, 2e-1, 3e-1, 5e-1, 1.0])
    t2, tn = k2.tail(u0), kn.tail(u0)
    A(f"{'u0':>10s}{'exp2':>12s}{'exp2nll':>12s}{'nll/exp2':>11s}")
    for a, b, c in zip(u0, t2, tn):
        A(f"{a:10.1e}{b:12.6f}{c:12.6f}{c/b:11.5f}")
    A("")
    A("unradiated fraction P(1-z < 1e-7):"
      f"  exp2 {k2.p_norad(1e-7):.6f}   exp2nll {kn.p_norad(1e-7):.6f}"
      f"   ratio {kn.p_norad(1e-7)/k2.p_norad(1e-7):.5f}")
    A("")
    A("mass dependence <u>(110-150)/<u>(60-80):")
    for v in ("exp2", "exp2nll"):
        r = [FA.FSRKernel(m, variant=v).moments()["u"] for m in (75.6, 138.566)]
        A(f"   {v:8s} {r[1]/r[0]:.5f}")
    A("")
    A("density ratio exp2nll/exp2 at fixed u, peak band:")
    uu = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    A(f"{'u':>10s}{'ratio':>10s}")
    for a in uu:
        A(f"{a:10.3g}{float(kn.pdf_u(a)/k2.pdf_u(a)):10.5f}")
    txt = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", default="data/genmerged_full.npz")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wclip", type=float, default=100.0)
    ap.add_argument("--nll", action="store_true",
                    help="only the O(alpha^2) NLL figures (no generator record)")
    args = ap.parse_args()
    tag = args.tag or ("fsr_nll" if args.nll else "fsr_analytic")
    out = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today():%y%m%d}_{tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out, logger=logger)
    logger.info(f"figures -> {out}")

    if args.nll:
        table_nll(os.path.join(out, "00_nll.txt"))
        fig_nll_ratio(out)
        fig_nll_coeff(out)
        fig_nll_mellin(out)
        return

    m_pre, u, w, norad = load(args.gen, args.wclip)
    logger.info(f"{len(w)} gen events")
    table(out, u, w, m_pre, norad, os.path.join(out, "00_moments.txt"))
    fig_density(out, u, w, m_pre)
    fig_tail(out, u, w, m_pre)
    fig_mass_dep(out, u, w, m_pre)
    fig_band_shape(out, u, w, m_pre)
    fig_variants(out)


if __name__ == "__main__":
    main()
