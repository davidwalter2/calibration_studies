#!/usr/bin/env python3
"""The multi-emission two-leg law against the generator.

`fsr_perleg multicheck` writes the model side -- the sharing density, the leg
law and ``Gbar`` of the single-photon, the exponentiated and the collinear
constructions on one ``eps`` grid; this reads it back against the sample's own
per-leg record and, when it is there, against a standalone Photos run with the
exact-matrix-element correction forced on or off.
"""
import argparse
import datetime
import json
import math
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import logging

import pubhtml
import ratiopanel

import cmp_perleg as CP
import fsr_analytic as FA
import fsr_perleg as PL

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

PEAK = CP.PEAK
#: how the three model curves are drawn everywhere
STYLE = dict(single=("C0", "-", r"exact O($\alpha$), one photon"),
             multi=("C1", "-", r"exact O($\alpha$), every photon"),
             coll=("C2", ":", r"collinear product $D\otimes D$"))


def mc_share(g, band, f_edges, slices):
    """``(hist, n)`` per slice of the generator's own sharing, 1 photon / all."""
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    z = (g["m_post"].astype(np.float64) / m) ** 2
    u = -0.5 * np.log(np.maximum(z, 1e-300))
    f = (1.0 - g["xp"]) / np.maximum(1.0 - z, 1e-300)
    one = CP.one_photon(g)
    sel = (m >= band[0]) & (m < band[1])
    out = {}
    for tag, extra in (("all", np.ones(len(u), bool)), ("one", one)):
        H, N, W = [], [], []
        for lo, hi in slices:
            s = sel & extra & (u > lo) & (u <= hi)
            ff = np.clip(f[s], f_edges[0] * 1.001, f_edges[-1] * 0.999)
            j = np.clip(np.searchsorted(f_edges, ff, "right") - 1, 0,
                        len(f_edges) - 2)
            H.append(np.bincount(j, w[s], len(f_edges) - 1))
            N.append(np.bincount(j, np.ones(int(s.sum())), len(f_edges) - 1))
            W.append(float(np.sum(w[s] * ((f[s] > 0.01) & (f[s] < 0.99)))
                           / max(w[s].sum(), 1e-300)))
        out[tag] = (np.stack(H), np.stack(N), np.asarray(W))
    return out


def mass_convention(g, band, edges=None):
    """Which mass--leg relation the generator's multi-photon events obey.

    With `n` photons, energy conservation in the pre-FSR rest frame is exact,
    ``eps_+ + eps_- = eps``, and the mass is
    ``z = 1 - eps + (sum k)^2/m^2``, so

        q = x_+ x_- - z = (sum k)^2 / m^2 >= 0 .

    The light-cone decomposition of a photon along the two muon directions is
    ``k_i = eps_+^i p_+ + eps_-^i p_- + k_T``, so averaged over the relative
    azimuth ``q = eps_+ eps_- - sum_i eps_+^i eps_-^i``: it is **exactly**
    ``eps_+ eps_-`` for one photon (any angle) and **exactly zero** when every
    photon is collinear to one leg.  The two model conventions are those two
    limits -- ``z = 1 - eps`` (the `corr` family) and ``z = x_+ x_-`` (the
    collinear product) -- so ``R = <q>/<eps_+ eps_->`` says directly which one
    the generator follows, bin by bin in the total loss.
    """
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    z = (g["m_post"].astype(np.float64) / m) ** 2
    u = -0.5 * np.log(np.maximum(z, 1e-300))
    xp = g["xp"].astype(np.float64)
    xm = g["xm"].astype(np.float64)
    q = xp * xm - z
    ee = (1.0 - xp) * (1.0 - xm)
    sel = (m >= band[0]) & (m < band[1]) & (u > 1e-6)
    e = np.geomspace(1e-4, 1.0, 13) if edges is None else np.asarray(edges)
    j = np.clip(np.searchsorted(e, u[sel], "right") - 1, 0, len(e) - 2)
    ww = w[sel]
    tot = np.bincount(j, ww, len(e) - 1)
    Q = np.bincount(j, ww * q[sel], len(e) - 1)
    E = np.bincount(j, ww * ee[sel], len(e) - 1)
    n = np.bincount(j, np.ones(int(sel.sum())), len(e) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return e, Q / tot, E / tot, Q / E, n


def fig_share_f(out, d, mc, islice, ph=None, name="40_share_f"):
    """The sharing density at one ``u`` slice, model against generator."""
    fe = d["f_edges"]
    cen = 0.5 * (fe[:-1] + fe[1:])
    lo, hi = d["slices"][islice]
    fig, ax, rax = ratiopanel.make_ratio_fig()

    def norm(h):
        h = np.array(h, float)
        h[0] = h[-1] = np.nan                  # the clipped overflow bins
        s = np.nansum(h)
        return h / s if s > 0 else h

    ref = norm(mc["all"][0][islice])
    ax.step(fe[:-1], ref, where="post", color="k", lw=1.8, label="MC, all")
    n_all = mc["all"][1][islice]
    one = norm(mc["one"][0][islice])
    ax.step(fe[:-1], one, where="post", color="C3", lw=1.4, ls="--",
            label=r"MC, $k^2\!\approx\!0$ tag")
    if ph is not None:
        # the standalone's own f histogram: (band, class, slice, f bin) with
        # class 0 = exactly one photon, 1 = all, 2 = the k^2 tag the sample's
        # record uses.  Its slice 0 is [1e-4, 1e-3), one below the model's.
        for tag, cls, col, lab in (
                ("A", 0, "C4", r"Photos ME on, exactly 1$\gamma$"),
                ("B", 0, "C5", r"Photos ME off, exactly 1$\gamma$"),
                ("mix", 2, "C6", r"Photos mix, $k^2\!\approx\!0$ tag")):
            k = f"{tag}_hf"
            if k in ph:
                ax.step(fe[:-1], norm(ph[k][0, cls, islice + 1]), where="post",
                        color=col, lw=1.3, ls="-.", label=lab)
    for md in ("single", "multi", "coll"):
        if f"share_{md}" not in d:
            continue
        col, ls, lab = STYLE[md]
        h = norm(d[f"share_{md}"][islice])
        ax.step(fe[:-1], h, where="post", color=col, lw=1.8, ls=ls, label=lab)
        ok = np.isfinite(h) & np.isfinite(ref) & (ref > 0) & (n_all > 30)
        rax.plot(cen[ok], (h / ref)[ok], color=col, ls=ls, lw=1.6)
    ok = np.isfinite(one) & np.isfinite(ref) & (ref > 0) & (n_all > 30)
    rax.plot(cen[ok], (one / ref)[ok], color="C3", ls="--", lw=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("fraction of events per bin")
    ax.legend(fontsize=10, loc="lower center", ncol=2)
    ax.set_title(f"the energy sharing between the legs, "
                 f"$u\\in[{lo:g},{hi:g})$, {d['m']:.1f} GeV",
                 fontsize=13, loc="left")
    ax.grid(alpha=0.25)
    rax.axhline(1.0, color="grey", lw=1.0, ls="--")
    rax.set_ylim(0.5, 1.6)
    rax.set_xlabel(r"$f = (1-x_+)/(1-z)$")
    rax.set_ylabel("/ MC all")
    rax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_share_u(out, d, dfine, mc, mcfine, name="41_share_u"):
    """``P(0.01 < f < 0.99)`` against the total loss: the wide-angle weight."""
    sl = np.asarray(d["slices"], float)
    n0 = len(PL.REPORT_SLICES)
    cen = np.sqrt(sl[n0:, 0] * sl[n0:, 1])
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    for tag, col, mk, lab in (("all", "k", "o", "MC, all"),
                              ("one", "C3", "s", r"MC, $k^2\!\approx\!0$ tag")):
        v, n = mc[tag][2][n0:], mc[tag][1][n0:].sum(1)
        ax.errorbar(cen, v, v / np.sqrt(np.maximum(n, 1)), fmt=mk, ms=4.0,
                    color=col, mfc="none" if mk == "s" else col, label=lab)
    for md in ("single", "multi", "coll"):
        k = f"sharewin_{md}"
        if k not in d:
            continue
        col, ls, lab = STYLE[md]
        y = np.asarray(d[k], float)[n0:]
        ax.plot(cen, y, color=col, lw=2.0, ls=ls, label=lab)
        if dfine is not None and k in dfine:
            yf = np.asarray(dfine[k], float)[n0:]
            ok = np.isfinite(yf)
            ax.plot(cen[ok], yf[ok], color=col, lw=1.0, ls=ls, alpha=0.55)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 0.6)
    ax.set_xlabel(r"$u = -\ln(m'/m)$")
    ax.set_ylabel(r"$P(0.01 < f < 0.99 \;|\; u)$")
    ax.set_title("how often both legs lose, at the peak; faint = the fine "
                 f"$\\epsilon$ grid, {d['m']:.1f} GeV", fontsize=13, loc="left")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_joint(out, d, g, band=PEAK, name="42_joint"):
    """The joint leg tail over the product of the marginals."""
    w = g["weight"]
    m = g["m_pre"].astype(np.float64)
    s = (m >= band[0]) & (m < band[1])
    ww = w[s] / w[s].sum()
    a = -np.log(np.maximum(g["xp"][s], 1e-300))
    b = -np.log(np.maximum(g["xm"][s], 1e-300))
    t = np.asarray(d["t"], float)
    rd = []
    for x in t:
        pa = float(np.sum(ww * (a > x)))
        pb = float(np.sum(ww * (b > x)))
        rd.append(float(np.sum(ww * ((a > x) & (b > x)))) / max(pa * pb, 1e-300))
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.plot(t, rd, "o-", color="k", lw=2.0, ms=5, label="MC, all")
    for md in ("single", "multi", "coll"):
        k = f"legtail_{md}"
        if k not in d:
            continue
        col, ls, lab = STYLE[md]
        r = np.asarray(d[k], float)
        ok = r[:, 1] > 0
        ax.plot(r[ok, 0], r[ok, 3], color=col, lw=2.0, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P(u_+>t,\,u_->t)\,/\,P(u_+>t)P(u_->t)$")
    ax.set_title("the leg correlation the multi-emission sharing puts back, "
                 f"{band[0]:.0f}-{band[1]:.0f} GeV", fontsize=13, loc="left")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_gbar(out, specs, name="43_gbar"):
    """``Gbar`` of each construction over the single-photon model's."""
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    for i, (lab, d) in enumerate(specs):
        un = np.asarray(d["u_nodes"], float)
        base = np.asarray(d["gbar_single"], float)
        for md in ("multi", "coll"):
            k = f"gbar_{md}"
            if k not in d:
                continue
            col, ls, _ = STYLE[md]
            ok = (un > 0) & (base > 0)
            ax.plot(un[ok], (np.asarray(d[k], float) / base)[ok], color=col,
                    ls=ls, lw=2.0 if i == 0 else 1.0,
                    alpha=1.0 if i == 0 else 0.55,
                    label=f"{STYLE[md][2]}, {lab}" if True else None)
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 3.0)
    ax.set_xlabel(r"$u = -\ln(m'/m)$")
    ax.set_ylabel(r"$\bar G(u)\,/\,\bar G_{1\gamma}(u)$")
    ax.set_title("the selection weight of the two-leg law, against the "
                 "single-photon model", fontsize=13, loc="left")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def table(out, d, dfine, mc, mcfine, path):
    """Every number of the two figures above, as text."""
    L = ["# the multi-emission two-leg law against the generator",
         f"# band {d['band'][0]:g}-{d['band'][1]:g} GeV, m = {d['m']:.3f}, "
         f"eps grid h = {float(d['h']):.1e}"
         + ("" if dfine is None else f", fine grid h = {float(dfine['h']):.1e}"),
         "",
         "# P(0.01 < f < 0.99) per u slice",
         f"{'u slice':>18s}{'MC all':>10s}{'MC tag':>10s}"
         f"{'single':>10s}{'multi':>10s}{'coll':>10s}{'multi/MC':>10s}"]
    for i, (lo, hi) in enumerate(PL.REPORT_SLICES):
        # the f resolution of the model is h/eps, so a slice is read off the
        # FINE short grid whenever it fits inside it
        src, msrc = d, mc
        if dfine is not None and PL.eps_of_u(hi) < (int(dfine["n"]) - 1) * float(dfine["h"]):
            src, msrc = dfine, mcfine
        if not np.isfinite(src["sharewin_multi"][i]):
            src, msrc = d, mc
        if src is None:
            continue
        r = (f"  [{lo:g}, {hi:g})".rjust(18)
             + f"{msrc['all'][2][i]:10.4f}{msrc['one'][2][i]:10.4f}"
             + "".join(f"{float(src['sharewin_' + md][i]):10.4f}"
                       for md in ("single", "multi", "coll"))
             + f"{float(src['sharewin_multi'][i]) / max(msrc['all'][2][i], 1e-30):10.4f}")
        L.append(r)
    L += ["", "# the leg law: P(u_leg > t) and the joint tail over the product",
          f"{'t':>10s}" + "".join(f"{md + ' marg':>13s}{md + ' j/p':>9s}"
                                  for md in ("single", "multi", "coll"))]
    tt = np.asarray(d["t"], float)
    for i in range(len(tt)):
        L.append(f"  {tt[i]:8.1e}" + "".join(
            f"{d['legtail_' + md][i][1]:13.5f}{d['legtail_' + md][i][3]:9.3f}"
            for md in ("single", "multi", "coll")))
    if PHOTOS is not None:
        L += ["", "# the standalone Photos sharing, 2.5e9 events per setting:",
              "#   P(0.01 < f < 0.99) per class and per u slice",
              f"{'u slice':>18s}{'A 1g':>9s}{'B 1g':>9s}{'mix 1g':>9s}"
              f"{'mix all':>9s}{'mix tag':>9s}{'exact':>9s}{'multi':>9s}"]
        # the per-event counters, not the f histogram: 0.01 is not one of the
        # histogram's edges, so a binned window would carry the straddling bin
        def win(tag, cls, isl):
            n = float(np.asarray(PHOTOS[f"{tag}_sn"], float)[0, cls, isl])
            m = float(np.asarray(PHOTOS[f"{tag}_smid"], float)[0, cls, isl])
            return m / max(n, 1e-300)
        for i, (lo, hi) in enumerate(PL.REPORT_SLICES):
            j = i + 1
            src = d
            if dfine is not None and PL.eps_of_u(hi) < (int(dfine["n"]) - 1) * float(dfine["h"]):
                src = dfine
            zb = float(np.exp(-2.0 * math.sqrt(lo * hi)))
            fm, wm = FA.share_nodes(float(d["m"]), np.array([zb]),
                                    npanel=1024, ng=8)
            ex = float(2.0 * wm[0][fm[0] > 0.01].sum())
            L.append(f"  [{lo:g}, {hi:g})".rjust(18)
                     + f"{win('A', 0, j):9.4f}{win('B', 0, j):9.4f}"
                     + f"{win('mix', 0, j):9.4f}{win('mix', 1, j):9.4f}"
                     + f"{win('mix', 2, j):9.4f}{ex:9.4f}"
                     + (f"{float(src['sharewin_multi'][i]):9.4f}"
                        if src is not None else "      ---"))
    L += ["", "# the mass-leg relation of the multi-photon configurations:",
          "#   q = x_+ x_- - z = (sum k)^2/m^2 ; R = <q>/<eps_+ eps_->",
          "#   R = 1 <-> z = 1 - eps (the corr family), R = 0 <-> z = x_+ x_-",
          f"{'u slice':>20s}{'<q>':>12s}{'<e+e->':>12s}{'R':>9s}{'events':>10s}"]
    e, Q, E, R, N = MASSCONV
    for i in range(len(R)):
        L.append(f"  [{e[i]:.3g}, {e[i+1]:.3g})".rjust(20)
                 + f"{Q[i]:12.4e}{E[i]:12.4e}{R[i]:9.4f}{int(N[i]):10d}")
    L += ["", "# Gbar over the single-photon model's",
          f"{'u':>10s}{'multi':>12s}{'coll':>12s}"]
    un = np.asarray(d["u_nodes"], float)
    b0 = np.asarray(d["gbar_single"], float)
    for u0 in (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0):
        L.append(f"  {u0:8.4g}" + "".join(
            f"{np.interp(u0, un, np.asarray(d['gbar_' + md], float)) / np.interp(u0, un, b0):12.5f}"
            for md in ("multi", "coll")))
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    logger.info(f"-> {path}")
    print("\n".join(L))


def fig_massconv(out, mcv, name="45_massconv"):
    """``R = <q>/<eps_+ eps_->`` against the total loss."""
    e, Q, E, R, N = mcv
    cen = np.sqrt(e[:-1] * e[1:])
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ok = N > 30
    ax.errorbar(cen[ok], R[ok], R[ok] / np.sqrt(N[ok]), fmt="o", ms=5,
                color="k", label="generator")
    ax.axhline(1.0, color="C0", lw=2.0,
               label=r"$z = 1-\epsilon$  (one photon, any angle)")
    ax.axhline(0.0, color="C2", lw=2.0, ls=":",
               label=r"$z = x_+x_-$  (every photon collinear)")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel(r"$u = -\ln(m'/m)$")
    ax.set_ylabel(r"$\langle (\sum k)^2/m^2\rangle\,/\,"
                  r"\langle\epsilon_+\epsilon_-\rangle$")
    ax.set_title("which mass-leg relation the generator's configurations obey",
                 fontsize=13, loc="left")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(alpha=0.25)
    pubhtml.savefig(fig, os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


MASSCONV = None
PHOTOS = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", required=True,
                    help="a `fsr_perleg multicheck` npz on the full eps grid")
    ap.add_argument("--check-fine", default=None,
                    help="the same on a fine short grid, for the small-u "
                         "sharing the full grid cannot resolve")
    ap.add_argument("--check-alt", action="append", default=[],
                    help="label=path.npz: extra grids for the Gbar panel")
    ap.add_argument("--perleg", default="data/perleg_full.npz")
    ap.add_argument("--photos-share", default=None,
                    help="a standalone Photos sharing npz (ME on / off)")
    ap.add_argument("--fit", action="append", default=[],
                    help="label=fit_*.json (repeatable)")
    ap.add_argument("--outpath", default=None)
    ap.add_argument("--tag", default="fsr_multiemission")
    a = ap.parse_args()

    out = a.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today():%y%m%d}_{a.tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out, logger=logger)
    logger.info(f"figures -> {out}")

    d = dict(np.load(a.check, allow_pickle=False))
    dfine = dict(np.load(a.check_fine)) if a.check_fine else None
    ph = dict(np.load(a.photos_share)) if a.photos_share else None
    g = CP.load_perleg(a.perleg)
    band = tuple(np.asarray(d["band"], float))
    mc = mc_share(g, band, d["f_edges"], d["slices"])
    mcf = mc_share(g, band, dfine["f_edges"], dfine["slices"]) if dfine else None

    if a.fit:
        CP.fig_fitshifts(out, a.fit, name="44_fitshifts")
    fig_share_f(out, d, mc, 2, ph, name="40_share_f")          # u in [0.05,0.2)
    src = dfine if dfine is not None else d
    fig_share_f(out, src, mcf if dfine is not None else mc, 0, ph,
                name="40b_share_f_soft")                       # u in [1e-3,1e-2)
    fig_share_u(out, d, dfine, mc, mcf)
    fig_joint(out, d, g, band)
    specs = [("h = %.1e" % float(d["h"]), d)]
    for spec in a.check_alt:
        lab, _, path = spec.partition("=")
        if os.path.exists(path):
            specs.append((lab, dict(np.load(path))))
    fig_gbar(out, specs)
    global MASSCONV, PHOTOS
    PHOTOS = ph
    MASSCONV = mass_convention(g, band)
    fig_massconv(out, MASSCONV)
    table(out, d, dfine, mc, mcf, os.path.join(out, "00_multi.txt"))


if __name__ == "__main__":
    main()
