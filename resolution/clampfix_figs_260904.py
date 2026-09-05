#!/usr/bin/env python3
"""Figures for the 2026-09-04 Gauss-Newton momentum-floor clamp fix.

Compares the WITH-clamp production (`_260903x_m0`, floor 2.0 GeV) against the
WITHOUT-clamp one (`_260904f_m0`, floor 0.25 GeV = 1.25 x the propagation
limit). One file per figure, ratio panel under every density-vs-density plot,
output directory derived from today's date.
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

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

MJ = 3.0969
OLD = dict(aux="runs/censoring260904/aux_jpsigun.npz",
           cache="runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz",
           lab="floor 2.0 GeV (_260903x)", col="C3")
NEW = dict(aux="runs/clampfix260904/aux_jpsigun_260904f.npz",
           cache="runs/cf_masspairs_jpsigun_ul16_260904f_m0.npz",
           lab="floor 0.25 GeV (_260904f)", col="C0")


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"-> {os.path.join(outdir, name)}.{{pdf,png}}")


def load(spec):
    d, c = np.load(spec["aux"]), np.load(spec["cache"])
    z = d["z"].astype(np.float64)
    s = d["sigma"].astype(np.float64)
    m = z * s + c["eta"].astype(np.float64)
    return dict(lab=spec["lab"], col=spec["col"], z=z, sig=s, m=m,
                mgen=c["eta"].astype(np.float64),
                c2=d["normchi2"].astype(np.float64),
                it=d["niter"].astype(np.float64),
                fz=d["frozen"].astype(np.float64) > 0.5,
                ptmin=d["ptmin"].astype(np.float64),
                pmin=d["pmin"].astype(np.float64))


def sane(V):
    return np.isfinite(V["sig"]) & (V["sig"] > 0) & (V["sig"] < 0.15) & np.isfinite(V["z"])


# ---------------------------------------------------------------- figure 1
def fig_chi2(A, B, outdir):
    """chi2/ndof density, log x. Ratio panel = new / old."""
    ed = np.logspace(-1.3, 6., 121)
    ctr = np.sqrt(ed[1:] * ed[:-1])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.2))
    h = {}
    for V in (A, B):
        ok = sane(V)
        hh, _ = np.histogram(np.clip(V["c2"][ok], ed[0], ed[-1]), bins=ed)
        h[V["lab"]] = hh / max(int(ok.sum()), 1)
        ax.step(ed[:-1], h[V["lab"]], where="post", color=V["col"], lw=1.8,
                label=f"{V['lab']}  (>10 on {100.*np.mean(V['c2'][ok]>10):.2f} %)")
    ax.axvline(10., color="0.4", ls="--", lw=1.2)
    rax.axvline(10., color="0.4", ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel("fraction of candidates / bin")
    ax.legend(loc="upper right", fontsize="small")
    ax.set_title(r"J/$\psi$ gun ditrack: two-track fit $\chi^2$/ndof",
                 fontsize="medium")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = h[B["lab"]] / h[A["lab"]]
    rax.step(ed[:-1], r, where="post", color="k", lw=1.5)
    rax.axhline(1., color="0.6", lw=1.0)
    rax.set_yscale("log")
    rax.set_ylim(1e-3, 1e3)
    rax.set_ylabel("new / old", fontsize="small")
    rax.set_xlabel(r"$\chi^2$/ndof")
    save(fig, outdir, "gun_chi2ndof_clampfix")


# ---------------------------------------------------------------- figure 2
def fig_oddmoment_vs_pt(A, B, outdir, u=0.05):
    """<z e^{-u z^2}> in bins of the minimum daughter pT."""
    ed = np.array([0., 1., 2., 3., 4., 5., 7., 10., 15., 40.])
    ctr = 0.5 * (ed[1:] + ed[:-1])
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    for V in (A, B):
        ok = sane(V)
        z, pt = V["z"][ok], V["ptmin"][ok]
        y, e, n = [], [], []
        for i in range(len(ed) - 1):
            s = (pt >= ed[i]) & (pt < ed[i + 1])
            w = z[s] * np.exp(-u * z[s] ** 2)
            y.append(w.mean() if s.sum() else np.nan)
            e.append(w.std(ddof=1) / np.sqrt(max(s.sum(), 1)) if s.sum() > 1 else np.nan)
            n.append(int(s.sum()))
        ax.errorbar(ctr, y, yerr=e, xerr=0.5 * np.diff(ed), marker="o", ms=5,
                    ls="none", color=V["col"], label=V["lab"], capsize=2.)
        logger.info(f"{V['lab']}: n per bin {n}")
    ax.axhline(0., color="0.6", lw=1.0)
    ax.axvline(2., color="0.4", ls="--", lw=1.2)
    ax.set_xlabel(r"min daughter $p_\mathrm{T}$ [GeV]")
    ax.set_ylabel(rf"$\langle z\,e^{{-{u}z^2}}\rangle$")
    ax.set_title(r"J/$\psi$ gun: candidate-mass odd moment vs the softer daughter",
                 fontsize="medium")
    ax.legend(fontsize="small")
    save(fig, outdir, "gun_oddmoment_vs_ptmin_clampfix")


# ---------------------------------------------------------------- figure 3
def fig_softmass(A, B, outdir, ptcut=2.0):
    """Refit mass of the candidates with a daughter below `ptcut`."""
    ed = np.linspace(2.85, 3.35, 126)
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.2))
    h = {}
    for V in (A, B):
        ok = sane(V) & (V["ptmin"] < ptcut)
        hh, _ = np.histogram(np.clip(V["m"][ok], ed[0], ed[-1]), bins=ed)
        h[V["lab"]] = hh / max(int(ok.sum()), 1)
        mm = V["m"][ok] - V["mgen"][ok]
        ax.step(ed[:-1], h[V["lab"]], where="post", color=V["col"], lw=1.8,
                label=f"{V['lab']}  $\\langle m-m_\\mathrm{{gen}}\\rangle$ = "
                      f"{1e3*mm.mean():+.2f} MeV (n={int(ok.sum())})")
    ax.axvline(MJ, color="0.4", ls="--", lw=1.2)
    rax.axvline(MJ, color="0.4", ls="--", lw=1.2)
    ax.set_ylabel("fraction of candidates / bin")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_title(rf"J/$\psi$ gun: refit mass, candidates with a daughter "
                 rf"$p_\mathrm{{T}} < {ptcut}$ GeV", fontsize="medium")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = h[B["lab"]] / h[A["lab"]]
    rax.step(ed[:-1], r, where="post", color="k", lw=1.5)
    rax.axhline(1., color="0.6", lw=1.0)
    rax.set_ylim(0., 2.)
    rax.set_ylabel("new / old", fontsize="small")
    rax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    save(fig, outdir, "gun_softdaughter_mass_clampfix")


# ---------------------------------------------------------------- figure 4
def fig_alpha(outdir):
    rows = [
        (r"gun, single $r$, all", "masslikfit_jpsigun_ul16_260904f_m0_r", "C0"),
        (r"gun, single $r$, all  [floor 2.0]", "masslikfit_jpsigun_260903x_r", "C3"),
        (r"gun, single $r$, $\chi^2$/ndof<3", "masslikfit_jpsigun_ul16_260904f_m0_r_q3", "C0"),
        (r"gun, single $r$, $\chi^2$/ndof<3  [floor 2.0]", "masslikfit_gun_260904_q3_r", "C3"),
        (r"gun, single $r$, $p_\mathrm{T}^\mathrm{min}>2$", "masslikfit_jpsigun_ul16_260904f_m0_r_pt2", "C0"),
        (r"gun, single $r$, $p_\mathrm{T}^\mathrm{min}>2$  [floor 2.0]", "masslikfit_gun_260904_pt2_r", "C3"),
        (r"gun, families, all", "masslikfit_jpsigun_ul16_260904f_m0_fam_krad1", "C0"),
        (r"gun, families, all  [floor 2.0]", "masslikfit_jpsigun_260903x_fam_krad1", "C3"),
        (r"gun, families, $\chi^2$/ndof<3", "masslikfit_jpsigun_ul16_260904f_m0_fam_q3", "C0"),
        (r"gun, families, $\chi^2$/ndof<3  [floor 2.0]", "masslikfit_gun_260904_q3_fam", "C3"),
        (r"v3, single $r$, all", "masslikfit_btojpsix_v3_260904f_m0_r", "C0"),
        (r"v3, single $r$, all  [floor 2.0]", "masslikfit_btojpsix_v3_260903x_r", "C3"),
        (r"v3, families, all", "masslikfit_btojpsix_v3_260904f_m0_fam_krad1", "C0"),
        (r"v3, families, all  [floor 2.0]", "masslikfit_btojpsix_v3_260903x_fam_krad1", "C3"),
    ]
    lab, v, e, col = [], [], [], []
    for l_, nm, c in rows:
        f = f"runs/{nm}.npz"
        if not os.path.exists(f):
            logger.warning(f"missing {f}")
            continue
        d = np.load(f, allow_pickle=True)
        mt = json.loads(str(d["meta"]))
        P = dict(zip(mt["parnames"], zip(d["fit_x"], d["fit_err"])))
        a, ae = P["alpha[1e-3]"]
        lab.append(f"{l_}  (n={mt['n']/1e3:.0f}k)")
        v.append(a); e.append(ae); col.append(c)
    if not v:
        logger.warning("no fits found")
        return
    y = np.arange(len(v))[::-1]
    fig, ax = plt.subplots(figsize=(10.2, 0.62 * len(v) + 2.2))
    ax.axvline(0., color="0.7", lw=1.2)
    for i in range(len(v)):
        ax.errorbar(v[i], y[i], xerr=e[i], marker="o", ms=7, color=col[i],
                    elinewidth=2., capsize=3.)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize="small")
    ax.set_ylim(-0.8, len(v) - 0.2)
    ax.set_xlabel(r"$\alpha$ [$10^{-3}$]")
    ax.set_title("mass-scale offset: GN momentum floor 2.0 vs 0.25 GeV",
                 fontsize="medium")
    ax.grid(axis="x", alpha=0.3)
    save(fig, outdir, "alpha_clampfix")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outpath", default=None)
    p.add_argument("--figs", nargs="*", default=None)
    a = p.parse_args()
    day = datetime.date.today().strftime("%y%m%d")
    outdir = a.outpath or os.path.expanduser(f"~/public_html/cvh/{day}_clampfix")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    want = a.figs
    if want is None or "alpha" in want:
        fig_alpha(outdir)
    if want is None or set(want) & {"chi2", "odd", "soft"}:
        if not (os.path.exists(OLD["aux"]) and os.path.exists(NEW["aux"])):
            logger.warning("aux caches missing; density figures skipped")
            return
        A, B = load(OLD), load(NEW)
        if want is None or "chi2" in want:
            fig_chi2(A, B, outdir)
        if want is None or "odd" in want:
            fig_oddmoment_vs_pt(A, B, outdir)
        if want is None or "soft" in want:
            fig_softmass(A, B, outdir)


if __name__ == "__main__":
    main()
