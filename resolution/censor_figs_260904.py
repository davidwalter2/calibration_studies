#!/usr/bin/env python3
"""Figures for the 2026-09-04 selection-censoring test.

One file per figure, ratio panel under every density-vs-density plot,
mplhep ROOT style, output directory derived from today's date.
"""
import argparse, datetime, os
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging

import pubhtml
import ratiopanel

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

MJ = 3.0969
PROBES = (0.05, 0.2, 1.0)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"-> {os.path.join(outdir, name)}.{{pdf,png}}")


def load(tag, aux, cache):
    d, c = np.load(aux), np.load(cache)
    z = d["z"].astype(np.float64)
    s = d["sigma"].astype(np.float64)
    m = z * s + c["eta"].astype(np.float64)
    return dict(tag=tag, z=z, sig=s, m=m,
                mtrk=d["mtrk"].astype(np.float64),
                mkin=d["mkin"].astype(np.float64),
                c2=d["normchi2"].astype(np.float64),
                it=d["niter"].astype(np.float64),
                fz=d["frozen"].astype(np.float64) > 0.5,
                ptmin=d["ptmin"].astype(np.float64),
                pmin=d["pmin"].astype(np.float64))


# ---------------------------------------------------------------- figure 1
def fig_window(V, outdir, lo=2.95, hi=3.25):
    """v3: the ALCARECO window is sharp in the PRE-refit mass and NOT in the
    refit mass.  Ratio panel = refit / pre-refit density."""
    ed = np.linspace(2.88, 3.32, 111)
    ctr = 0.5 * (ed[1:] + ed[:-1])
    w = np.diff(ed)
    ok = V["sig"] < 0.15
    hp, _ = np.histogram(np.clip(V["mtrk"][ok], ed[0], ed[-1]), bins=ed)
    hr, _ = np.histogram(np.clip(V["m"][ok], ed[0], ed[-1]), bins=ed)
    n = int(ok.sum())
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.2))
    ax.step(ed[:-1], hp / (n * w), where="post", color="C0", lw=1.8,
            label=r"pre-refit $m_{\mu\mu}$ (the cut variable)")
    ax.step(ed[:-1], hr / (n * w), where="post", color="C3", lw=1.8,
            label=r"CVH refit $m_{\mu\mu}$")
    for x in (lo, hi):
        ax.axvline(x, color="0.3", ls="--", lw=1.4)
        rax.axvline(x, color="0.3", ls="--", lw=1.4)
    ax.set_yscale("log")
    ax.set_ylabel("candidates / GeV (normalised)")
    ax.legend(loc="lower center", fontsize="small")
    ax.set_title(r"B$\to$J/$\psi$+X v3: ALCARECO window 2.95--3.25 GeV",
                 fontsize="medium")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = hr / hp
        e = r * np.sqrt(1. / np.maximum(hr, 1) + 1. / np.maximum(hp, 1))
    keep = (hp > 0) & (hr > 0)
    rax.axhline(1., color="gray", lw=1.2)
    rax.errorbar(ctr[keep], r[keep], e[keep], marker="o", ms=3.2, ls="none",
                 color="black", elinewidth=1.)
    rax.set_ylim(0., 2.2)
    rax.set_ylabel("refit / pre-refit", fontsize="small")
    rax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    save(fig, outdir, "v3_mass_prefit_vs_refit")


def fig_acceptance(V, outdir, lo=2.95, hi=3.25):
    """The EFFECTIVE post-refit acceptance A(m) = Pr(delta in [m-hi, m-lo])
    against the sharp window the truncation term assumes."""
    dl = V["m"] - V["mtrk"]
    g = np.isfinite(dl) & (np.abs(dl) < 1.) & (V["sig"] < 0.15)
    dd = np.sort(dl[g])
    F = lambda x: np.searchsorted(dd, x) / len(dd)
    mg = np.linspace(2.85, 3.35, 501)
    A = np.array([F(x - lo) - F(x - hi) for x in mg])
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.plot(mg, A, color="C3", lw=2.2,
            label=r"measured $A(m)=\Pr(\delta\in[m-h,m-l])$")
    ax.plot(mg, ((mg > lo) & (mg < hi)).astype(float), color="C0", lw=1.8,
            ls="--", label="sharp window (what the correction assumes)")
    ax.axvline(MJ, color="0.6", lw=1.0)
    ax.set_xlabel(r"CVH refit $m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("selection acceptance")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower center", fontsize="small")
    ax.set_title(r"the refit smears the pre-refit cut: rms$(\delta)$ = "
                 f"{1e3*dl[g].std():.1f} MeV " r"$\approx 1\,\sigma_m$",
                 fontsize="medium")
    save(fig, outdir, "v3_effective_acceptance")

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ed = np.linspace(-0.12, 0.12, 121)
    h, _ = np.histogram(np.clip(dl[g], ed[0], ed[-1]), bins=ed, density=True)
    ax.step(ed[:-1], h, where="post", color="C3", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta = m_{\rm refit} - m_{\rm pre-refit}$ [GeV]")
    ax.set_ylabel("density [1/GeV]")
    ax.set_title(r"v3: mean $%+.2f$ MeV, median $%+.2f$ MeV, rms %.1f MeV"
                 % (1e3 * dl[g].mean(), 1e3 * np.median(dl[g]),
                    1e3 * dl[g].std()), fontsize="medium")
    save(fig, outdir, "v3_refit_mass_shift")


# ---------------------------------------------------------------- figure 2
def fig_oddvscut(S, outdir, models=None):
    """<z e^{-u z^2}> vs the normchi2 cut, per sample."""
    cuts = np.array([1.5, 2., 3., 5., 10., 1e9])
    xl = ["<1.5", "<2", "<3", "<5", "<10", "none"]
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    for i, V in enumerate(S):
        base = np.isfinite(V["z"]) & (V["sig"] < 0.15)
        y, f = [], []
        for c in cuts:
            s = base & (V["c2"] < c)
            y.append(float(np.mean(V["z"][s] * np.exp(-0.05 * V["z"][s] ** 2))))
            f.append(s.sum() / base.sum())
        ax.plot(np.arange(len(cuts)), y, marker="o", lw=2.0, color=f"C{i}",
                label=f"{V['tag']}  (kept {100*f[0]:.0f}...{100*f[-1]:.0f} %)")
        if models and V["tag"] in models:
            ax.axhline(models[V["tag"]], color=f"C{i}", ls=":", lw=1.8)
            ax.text(0.02, models[V["tag"]], f" model {models[V['tag']]:+.4f}",
                    color=f"C{i}", fontsize="x-small", va="bottom")
    ax.set_xticks(np.arange(len(cuts)))
    ax.set_xticklabels(xl)
    ax.set_xlabel(r"$\chi^2/{\rm ndof}$ cut on the two-track fit")
    ax.set_ylabel(r"$\langle z\,e^{-0.05 z^2}\rangle$  (mass pull)")
    ax.axhline(0., color="0.7", lw=1.0)
    ax.legend(loc="best", fontsize="small")
    ax.set_title("the candidate-mass asymmetry against fit quality",
                 fontsize="medium")
    save(fig, outdir, "mass_oddmoment_vs_chi2cut")


def fig_oddvsptmin(S, outdir):
    ed = np.array([0., 2., 3., 4., 5., 7., 10., 15., 25.])
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    for i, V in enumerate(S):
        base = np.isfinite(V["z"]) & (V["sig"] < 0.15)
        x, y, e = [], [], []
        for l_, h_ in zip(ed[:-1], ed[1:]):
            s = base & (V["ptmin"] >= l_) & (V["ptmin"] < h_)
            if s.sum() < 200:
                continue
            zz = V["z"][s]
            v = zz * np.exp(-0.05 * zz ** 2)
            x.append(0.5 * (l_ + h_)); y.append(v.mean())
            e.append(v.std() / np.sqrt(len(v)))
        ax.errorbar(x, y, e, marker="o", lw=2.0, color=f"C{i}", label=V["tag"])
    ax.axhline(0., color="0.7", lw=1.0)
    ax.set_xlabel(r"min daughter $p_{\rm T}$ [GeV]")
    ax.set_ylabel(r"$\langle z\,e^{-0.05 z^2}\rangle$  (mass pull)")
    ax.legend(loc="best", fontsize="small")
    ax.set_title("no acceptance edge: the asymmetry does not vanish away from "
                 r"the low-$p_{\rm T}$ threshold", fontsize="medium")
    save(fig, outdir, "mass_oddmoment_vs_ptmin")


def fig_pullsplit(V, outdir):
    """The mass pull for the clean and the pathological subsample, with the
    ratio panel = all / clean."""
    ed = np.linspace(-6., 6., 121)
    ctr = 0.5 * (ed[1:] + ed[:-1])
    w = np.diff(ed)
    base = np.isfinite(V["z"]) & (V["sig"] < 0.15)
    good = base & (V["c2"] < 3) & ~V["fz"] & (V["it"] < 10)
    bad = base & ~good
    hg, _ = np.histogram(np.clip(V["z"][good], ed[0], ed[-1]), bins=ed)
    hb, _ = np.histogram(np.clip(V["z"][bad], ed[0], ed[-1]), bins=ed)
    ha, _ = np.histogram(np.clip(V["z"][base], ed[0], ed[-1]), bins=ed)
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.2))
    ax.step(ed[:-1], hg / (good.sum() * w), where="post", color="C0", lw=1.8,
            label=r"clean: $\chi^2/$ndof$<3$, not clamped, niter$<10$"
                  f"  ({100*good.sum()/base.sum():.1f} %)")
    ax.step(ed[:-1], hb / (max(bad.sum(), 1) * w), where="post", color="C3",
            lw=1.8, label=f"the rest ({100*bad.sum()/base.sum():.1f} %)")
    ax.set_yscale("log")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_title(f"{V['tag']}: the pathological subsample is where the "
                 "asymmetry lives", fontsize="medium")
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = (ha / base.sum()) / (hg / good.sum())
        er = ra * np.sqrt(1. / np.maximum(ha, 1) + 1. / np.maximum(hg, 1))
    keep = (ha > 0) & (hg > 0)
    keep[0] = keep[-1] = False
    rax.axhline(1., color="gray", lw=1.2)
    rax.axhspan(0.95, 1.05, color="0.85", zorder=0)
    rax.errorbar(ctr[keep], ra[keep], er[keep], marker="o", ms=3.2, ls="none",
                 color="black", elinewidth=1.)
    rax.set_ylim(0.5, 1.9)
    rax.set_ylabel("all / clean", fontsize="small")
    rax.set_xlabel(r"$z = (m_{\rm refit}-m_{\rm gen})/\sigma_m$")
    save(fig, outdir, f"{V['tag'].split()[0].lower()}_masspull_clean_vs_patho")


# ---------------------------------------------------------------- figure 3
def fig_trackcuts(caches, labels, outdir):
    """Single track: the odd moment per charge against the chi2 cut -- flat."""
    cuts = np.array([2., 3., 5., 10., 1e9])
    xl = ["<2", "<3", "<5", "<10", "none"]
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    for i, (c, lab) in enumerate(zip(caches, labels)):
        d = np.load(c)
        z = d["z"].astype(np.float64)
        q = d["charge"].astype(np.int64)
        c2 = d["normchi2"].astype(np.float64)
        for qq, ls, mk in ((1, "-", "o"), (-1, "--", "s")):
            y, e = [], []
            for cc in cuts:
                s = (q == qq) & (c2 < cc)
                v = z[s] * np.exp(-0.05 * z[s] ** 2)
                y.append(v.mean()); e.append(v.std() / np.sqrt(len(v)))
            ax.errorbar(np.arange(len(cuts)), y, e, marker=mk, ls=ls, lw=2.0,
                        color=f"C{i}",
                        label=f"{lab}  q={'+' if qq>0 else '-'}")
    ax.set_xticks(np.arange(len(cuts)))
    ax.set_xticklabels(xl)
    ax.axhline(0., color="0.7", lw=1.0)
    ax.set_xlabel(r"$\chi^2/{\rm ndof}$ cut on the single-track fit")
    ax.set_ylabel(r"$\langle z\,e^{-0.05 z^2}\rangle$  ($q/p$ pull)")
    ax.legend(loc="center right", fontsize="x-small")
    ax.set_title(r"single track: the 1--3$\sigma$ asymmetry is cut-independent",
                 fontsize="medium")
    save(fig, outdir, "track_oddmoment_vs_chi2cut")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=None)
    a = p.parse_args()
    day = datetime.date.today().strftime("%y%m%d")
    outdir = a.outdir or os.path.expanduser(
        f"~/public_html/cvh/{day}_censoring")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    R = "runs/censoring260904"
    G = load("gun ditrack (no mass window)", f"{R}/aux_jpsigun.npz",
             "runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz")
    V = load("v3 ditrack (2.95-3.25 window)", f"{R}/aux_btojpsix.npz",
             "runs/cf_masspairs_btojpsix_v3_260903x_m0.npz")
    fig_window(V, outdir)
    fig_acceptance(V, outdir)
    fig_oddvscut([G, V], outdir,
                 models={G["tag"]: 0.00827, V["tag"]: 0.01000})
    fig_oddvsptmin([G, V], outdir)
    fig_pullsplit(G, outdir)
    fig_pullsplit(V, outdir)
    fig_trackcuts(["runs/cf_trackres_mugun_lowpt_260903x_m0_k0.npz",
                   "runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz"],
                  [r"$\mu$ $p_{\rm T}$ 2--20", r"$\mu$ $p_{\rm T}$ 20--60"],
                  outdir)
    logger.info(f"figures in {outdir}")


if __name__ == "__main__":
    logger = logging.setup_logger(__file__, 3, False)
    main()
