#!/usr/bin/env python3
r"""Real lepton- and hadron-pair emission: the exact O(alpha^2) term, its
eikonal limit (what Photos++ generates) and the dispersive leading log.

One file per panel, a ratio panel under every density-vs-model plot, mplhep
ROOT style, PNG twin next to every PDF.

    ./run_tf_z.sh python3 -u plot_pairs.py
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging

import pubhtml
import ratiopanel

import fsr_analytic as FA

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

EIK = "data/fsr/pairkern_eik.npz"

#: Photos++ 3.61 standalone, 1e9 events at a FIXED m_pre = 91.1876 GeV in the Z
#: rest frame, photon emission off (`photos_standalone/photos_pairdiag.cc`).
PH_M = 91.1876
PH_RATE = {"e": 2.41855e-3, "mu": 3.12600e-4}
PH_UN = {"e": 1.95999e-4, "mu": 5.68119e-5}
#: rate(q > q_cut) per event, same run
PH_QCUT = np.array([1.022060e-3, 0.01, 0.1, 1.0, 3.0, 10.0])
PH_RATE_Q = {"e": np.array([2.4186e-3, 1.2784e-3, 4.9841e-4, 1.2201e-4,
                            4.5648e-5, 9.4270e-6]),
             "mu": np.array([3.1260e-4, 3.1260e-4, 3.1260e-4, 1.2175e-4,
                             4.5556e-5, 9.3540e-6])}
#: pairs emitted in each (species, block, leg) cell of Photos's own bookkeeping:
#: PHOPAR is called once before and once after the photons, and loops over both
#: charged daughters, so there are four `trypar` calls per species per event.
PH_LEG = {"e": [605225, 603154, 606009, 604166],
          "mu": [78260, 78312, 78077, 77951]}
PH_LEG_N = 1e9
PH_COS = {"e": (0.5011, 0.4989, 0.3250, 0.3238),
          "mu": (0.5007, 0.4993, 0.2565, 0.2568)}

SPEC_LABEL = {"e": r"$e^+e^-$", "mu": r"$\mu^+\mu^-$",
              "tau": r"$\tau^+\tau^-$", "had": "hadrons"}


# --------------------------------------------------------------------------
def photos_paironly(path="data/photos/gen_paironly.npz", band=(86.0, 96.0)):
    """``(u_centres, dN/du per event, error, <m_pre>)`` of a pair-only run."""
    d = np.load(path, allow_pickle=True)
    lo, hi = d["bands_lo"], d["bands_hi"]
    sel = (lo >= band[0] - 1e-9) & (hi <= band[1] + 1e-9)
    n = float(d["n"][sel].sum())
    h = d["logu_h"][sel].sum(0)
    e = np.geomspace(1e-6, 2.0, len(h) + 1)
    c = np.sqrt(e[1:] * e[:-1])
    dens = h / (n * np.diff(e))
    err = dens / np.sqrt(np.maximum(h, 1.0))
    return c, dens, err, float(d["mpre_s1"][sel].sum() / n)


def ll_pdf_u(m, u, species):
    """The dispersive LEADING-LOG pair spectrum, ``(beta_p/2) P(z)``, in ``u``."""
    b = FA.beta_pair_ll(m, species)
    z = np.exp(-2.0 * np.asarray(u, float))
    return 2.0 * z * 0.5 * b * (1.0 + z * z) / (1.0 - z)


# --------------------------------------------------------------------------
def fig_uspec(out):
    c, dens, err, m = photos_paironly()
    sp = ("e", "mu")
    ex = FA.pair_pdf_u(m, c, sp)
    ei = FA.pair_pdf_u(m, c, sp, path=EIK)
    ll = ll_pdf_u(m, c, sp)
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    ax.plot(c, ex, color="C3", lw=2.0, label=r"exact O($\alpha^2$), $e$+$\mu$")
    ax.plot(c, ei, color="C0", lw=1.8, ls="--",
            label="eikonal limit (Photos's matrix element)")
    ax.plot(c, ll, color="C2", lw=1.8, ls=":",
            label=r"dispersive leading log, $\beta_{\rm pair}$")
    ax.errorbar(c, dens, yerr=err, fmt="o", ms=3.2, color="black", lw=0,
                elinewidth=0.9, zorder=5,
                label="Photos++ 3.61, pairs only (photons off)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e3)
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u$")
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    ax.set_title(r"real pair emission, $m_{\rm pre}\in[86,96)$ GeV, "
                 rf"$\langle m\rangle$ = {m:.3f} GeV", fontsize=13, loc="left")
    for y, col, ls in ((ei, "C0", "--"), (ll, "C2", ":")):
        rax.plot(c, y / ex, color=col, ls=ls, lw=1.6)
    rax.errorbar(c, dens / ex, yerr=err / ex, fmt="o", ms=3.0, color="black",
                 lw=0, elinewidth=0.9)
    rax.axhline(1.0, color="C3", lw=1.4)
    rax.set_xscale("log"); rax.set_ylim(0.0, 3.0)
    rax.set_xlabel("$u = -\\ln(m_{\\rm post}/m_{\\rm pre})$")
    rax.set_ylabel("/ exact")
    pubhtml.savefig(fig, os.path.join(out, "01_pair_uspec.pdf"))
    plt.close(fig)


def fig_uspec_species(out):
    m = 91.1876
    u = np.geomspace(1e-5, 5.0, 260)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    tot = np.zeros_like(u)
    for s, col in zip(FA.PAIR_SPECIES, ("C0", "C1", "C2", "C4")):
        y = FA.pair_pdf_u(m, u, (s,))
        tot += y
        n = FA.FSRKernel(m, variant="born", pair=(s,))
        ax.plot(u, y, color=col, lw=1.8,
                label=SPEC_LABEL[s] + rf",  $N$ = {n.pair_rate:.2e}")
    ax.plot(u, tot, color="black", lw=2.2, label="all species")
    ax.plot(u, ll_pdf_u(m, u, FA.PAIR_SPECIES), color="grey", lw=1.6, ls=":",
            label="dispersive leading log, all")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e2)
    ax.set_xlabel("$u = -\\ln(m_{\\rm post}/m_{\\rm pre})$")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}u$")
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    ax.set_title(rf"exact pair radiator by species, $m$ = {m} GeV",
                 fontsize=13, loc="left")
    pubhtml.savefig(fig, os.path.join(out, "02_pair_species.pdf"))
    plt.close(fig)


def fig_rate_qcut(out):
    m = PH_M
    cuts = np.array([1.022060e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0,
                     30.0])
    ex = {s: [] for s in ("e", "mu")}
    ei = {s: [] for s in ("e", "mu")}
    for qc in cuts:
        q2 = max(qc * qc, 4 * FA.M_E**2)
        kw = dict(nq=10, nth=32, npan=2, ng=16)
        a = FA.pair_moments(m, species=("e", "mu"), q2lo=q2, **kw)
        b = FA.pair_moments(m, species=("e", "mu"), q2lo=q2, eikonal=True, **kw)
        for s in ("e", "mu"):
            ex[s].append(a[s][0]); ei[s].append(b[s][0])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.6))
    for s, col, mk in (("e", "C3", "o"), ("mu", "C0", "s")):
        ax.plot(cuts, ex[s], color=col, lw=2.0,
                label="exact, " + SPEC_LABEL[s])
        ax.plot(cuts, ei[s], color=col, lw=1.6, ls="--",
                label="eikonal, " + SPEC_LABEL[s])
        ax.plot(PH_QCUT, PH_RATE_Q[s], mk, ms=6, color=col, mfc="none",
                label="Photos++, " + SPEC_LABEL[s])
        rax.plot(PH_QCUT, PH_RATE_Q[s] / np.interp(PH_QCUT, cuts, ex[s]),
                 mk, ms=6, color=col, mfc="none")
        rax.plot(cuts, np.array(ei[s]) / np.array(ex[s]), color=col, ls="--",
                 lw=1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("pairs per event with $q > q_{\\rm cut}$")
    ax.legend(fontsize=11, frameon=False, loc="lower left", ncol=2)
    ax.set_title(rf"pair rate above a pair-mass cut, $m$ = {m} GeV",
                 fontsize=13, loc="left")
    rax.axhline(1.0, color="grey", lw=0.9)
    rax.set_xscale("log"); rax.set_ylim(0.80, 1.45)
    rax.set_xlabel("$q_{\\rm cut}$ [GeV]")
    rax.set_ylabel("/ exact")
    pubhtml.savefig(fig, os.path.join(out, "03_rate_vs_qcut.pdf"))
    plt.close(fig)


def fig_collinear_log(out):
    """The effective collinear log of the massive-photon emission."""
    m = 91.1876
    s = m * m
    L = float(FA.coll_log(m))
    q2 = np.geomspace(4 * FA.M_E**2, 0.2 * s, 90)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    for z, col in ((0.9, "C0"), (0.5, "C1"), (0.1, "C2")):
        x = 1.0 - z + q2 / s
        lam = FA.spec_gammastar(m, np.full_like(q2, z), q2, ng=24) \
            / (FA.A_PI * (1 + z * z) / (1 - z))
        good = lam > 0
        ax.plot(q2[good], lam[good], color=col, lw=2.0, label=f"exact, $z$ = {z}")
        ax.axhline(L - 1 + math.log(z), color=col, lw=1.0, ls="--")
    ax.plot(q2, np.log(s / q2) - 1.0, color="grey", lw=1.8, ls=":",
            label=r"$\ln(s/q^2)-1$  (the leading-log term)")
    ax.axvline(FA.M_MU**2, color="black", lw=0.9)
    ax.text(FA.M_MU**2 * 1.2, 1.0, r"$q^2 = m_\mu^2$", fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(0, 24)
    ax.set_xlabel("$q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$R_{\gamma^*}(z;q^2)\,/\,[(\alpha/\pi)(1+z^2)/(1-z)]$")
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.set_title("effective collinear log of a massive photon"
                 "\n(dashed: $L-1+\\ln z$, the massless-photon value)",
                 fontsize=12, loc="left")
    pubhtml.savefig(fig, os.path.join(out, "04_collinear_log.pdf"))
    plt.close(fig)


def fig_per_leg(out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    cells = ["before\nleg 1", "before\nleg 2", "after\nleg 1", "after\nleg 2"]
    xs = np.arange(4)
    for i, (s, col) in enumerate((("e", "C3"), ("mu", "C0"))):
        y = np.array(PH_LEG[s]) / PH_LEG_N
        e = np.sqrt(np.array(PH_LEG[s])) / PH_LEG_N
        ax.bar(xs + 0.4 * i - 0.2, y, 0.38, yerr=e, color=col, alpha=0.85,
               label=SPEC_LABEL[s])
        ax.axhline(y.mean(), color=col, lw=1.0, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(cells, fontsize=11)
    ax.set_yscale("log")
    ax.set_ylabel("pairs / event")
    ax.legend(fontsize=11, frameon=False)
    ax.set_title("Photos: the four `trypar` calls per species\n"
                 "(before/after the photons $\\times$ two charged legs)",
                 fontsize=12, loc="left")
    lab = ["$\\cos\\theta>0$", "$\\cos\\theta<0$",
           "$\\cos\\theta>0.9$", "$\\cos\\theta<-0.9$"]
    for i, (s, col) in enumerate((("e", "C3"), ("mu", "C0"))):
        bx.bar(np.arange(4) + 0.4 * i - 0.2, PH_COS[s], 0.38, color=col,
               alpha=0.85, label=SPEC_LABEL[s])
    bx.set_xticks(np.arange(4)); bx.set_xticklabels(lab, fontsize=11)
    bx.set_ylabel(r"fraction of pairs")
    bx.set_ylim(0, 0.62)
    bx.legend(fontsize=11, frameon=False)
    bx.set_title("pair direction w.r.t. the $\\mu^-$:\n"
                 "the two legs radiate equally", fontsize=12, loc="left")
    fig.tight_layout(pad=1.4)
    pubhtml.savefig(fig, os.path.join(out, "05_per_leg.pdf"))
    plt.close(fig)


def write_summary(out):
    m = PH_M
    with open(os.path.join(out, "00_pairs.txt"), "w") as fh:
        def w(t=""):
            print(t); fh.write(t + "\n")
        w(f"Real pair emission off the muon line, m = {m} GeV")
        w()
        w("exact O(alpha^2) dispersive term vs its eikonal limit (= the matrix")
        w("element Photos++ 3.61 uses) vs the dispersive leading log")
        w("%-8s %12s %12s %12s %9s %12s %9s" % (
            "species", "rate exact", "<u>N exact", "<u>N eikonal", "eik/ex",
            "<u>N LL", "LL/ex"))
        ex = FA.pair_moments(m)
        ei = FA.pair_moments(m, eikonal=True)
        t = [0.0, 0.0, 0.0]
        for sp in FA.PAIR_SPECIES:
            ll = FA.beta_pair_ll(m, (sp,)) * 0.5099671
            w("%-8s %12.5e %12.5e %12.5e %9.4f %12.5e %9.4f" % (
                sp, ex[sp][0], ex[sp][1], ei[sp][1], ei[sp][1] / ex[sp][1],
                ll, ll / ex[sp][1]))
            t[0] += ex[sp][0]; t[1] += ex[sp][1]; t[2] += ei[sp][1]
        ll = FA.beta_pair_ll(m, FA.PAIR_SPECIES) * 0.5099671
        w("%-8s %12.5e %12.5e %12.5e %9.4f %12.5e %9.4f" % (
            "TOTAL", t[0], t[1], t[2], t[2] / t[1], ll, ll / t[1]))
        w()
        w("Photos++ 3.61 standalone, 1e9 events at this mass, photons off:")
        for sp in ("e", "mu"):
            w("   %-4s rate = %.5e (eikonal %.5e, %.4f)   <u>N = %.5e "
              "(eikonal %.5e, %.4f)" % (
                  sp, PH_RATE[sp], ei[sp][0], PH_RATE[sp] / ei[sp][0],
                  PH_UN[sp], ei[sp][1], PH_UN[sp] / ei[sp][1]))
        w()
        w("Kernel-level, exp2nll:")
        base = None
        for pr, lab, tab in (((), "no pairs", None), (("e",), "e", None),
                             (("e", "mu"), "e+mu", None),
                             (FA.PAIR_SPECIES, "all", None),
                             (("e", "mu"), "e+mu, eikonal", EIK)):
            k = FA.FSRKernel(m, variant="exp2nll", pair=pr, pair_table=tab)
            r, ww, _ = k.atoms()
            mu = -float(np.sum(ww * np.log(r)))
            base = mu if base is None else base
            w("   %-14s N_pair = %.5e   <u> = %.6e   d<u> = %+.5e "
              "(%+.3f %% of the radiator)"
              % (lab, k.pair_rate, mu, mu - base, (mu - base) / base * 100))
        w()
        pc = FA.pair_moments(m, species=("e", "mu"), q2lo=3600.0)
        w("Singlet background (pair mass inside a 60-120 GeV window): "
          "%.3e per event" % sum(v[0] for v in pc.values()))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="fsr_pairs")
    ap.add_argument("--date", default="260913")
    a = ap.parse_args()
    out = a.out or pubhtml.figdir(a.tag, date=a.date)
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)
    logger.info(f"figures -> {out}")
    fig_uspec(out)
    fig_uspec_species(out)
    fig_collinear_log(out)
    fig_per_leg(out)
    fig_rate_qcut(out)
    write_summary(out)
    logger.info("done")


if __name__ == "__main__":
    main()
