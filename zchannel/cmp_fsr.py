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
      f"{'MC/exp1':>8s} {'MC/exp2':>8s}")
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
        A(f"{lo:5.0f}-{hi:<6.0f} {mc*1e3:11.4f} {er*1e3:8.4f} "
          f"{a1*1e3:10.4f} {a2*1e3:10.4f} {mc/a1:8.4f} {mc/a2:8.4f}")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", default="data/genmerged_full.npz")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--tag", default="fsr_analytic")
    ap.add_argument("--wclip", type=float, default=100.0)
    args = ap.parse_args()
    out = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today():%y%m%d}_{args.tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out, logger=logger)
    logger.info(f"figures -> {out}")

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
