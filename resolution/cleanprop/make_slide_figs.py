#!/usr/bin/env python3
"""Figures for the clean-propagation-test slide deck.

Run from calibration_studies/resolution/ (imports cf_propagation_test):
    python cleanprop/make_slide_figs.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cf_propagation_test import (load_sim, load_model, model_phi, model_variance,
                                 ecf, weier_scalar, FUNCTIONALS, SIM_BRANCH,
                                 REF_BRANCH, TAU)

hep.style.use(hep.style.ROOT)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides", "assets")
os.makedirs(OUT, exist_ok=True)

SIM = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/sim_260804_pt10_eta0.30_phi0.20/simstates_*.root"
MODEL = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/model/model_pt10_eta0.30_phi0.20.root"

RED, BLUE, GREY = "#A31F34", "#1f4e9c", "0.45"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- schematic
def fig_schematic():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    planes = [1.0, 2.2, 3.4, 5.0, 6.6, 8.2]
    for xp in planes:
        ax.plot([xp, xp], [-1.05, 1.05], color="0.75", lw=6, solid_capstyle="butt", zorder=1)
    x = np.linspace(0, 9.2, 400)
    # many simulated tracks: accumulated random-walk deflection
    ends = []
    for _ in range(60):
        kicks = rng.normal(0, 0.011, len(planes))
        y = np.zeros_like(x)
        slope = 0.0
        last = 0.0
        for xp, kk in zip(planes, kicks):
            m = x > xp
            slope += kk
            y[m] = y[m] + slope * (x[m] - xp)
            last = xp
        ax.plot(x, y, color=GREY, lw=0.7, alpha=0.5, zorder=2)
        ends.append(np.interp(9.2, x, y))
    ax.plot(x, np.zeros_like(x), color=RED, lw=2.6, zorder=4,
            label="deterministic reference (one Geant4e propagation)")
    ax.plot([], [], color=GREY, lw=1.0, label="full Geant4 simulation, same initial state")
    ax.plot(0, 0, "o", color="k", ms=9, zorder=5)
    ax.annotate("fixed initial state\n(no vertex or kinematic smearing)",
                xy=(0, 0), xytext=(0.15, 0.72), fontsize=15,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    # residual distribution at the last plane
    ends = np.array(ends)
    hist, edges = np.histogram(rng.normal(0, .16, 40000) - 0.12 * rng.exponential(1., 40000),
                               bins=60, range=(-1.0, 1.0), density=True)
    cent = 0.5 * (edges[1:] + edges[:-1])
    ax.barh(cent, hist * 0.30, height=(edges[1] - edges[0]), left=9.3,
            color=RED, alpha=.35, zorder=3)
    ax.text(9.45, 0.80, "PDF of the\npropagated state", fontsize=15, color=RED)
    ax.set_xlim(-0.3, 11.4)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", fontsize=14, frameon=False)
    ax.text(4.6, 1.02, "silicon sensor planes", fontsize=14, color="0.45", ha="center")
    save(fig, "schematic")


# ------------------------------------------------------------------ the bug
def fig_bug():
    from scipy.special import exp1

    def exact(a, w):
        a = np.asarray(a, dtype=np.complex128)
        ia = 1j * a
        with np.errstate(all="ignore"):
            t1 = (-np.exp(ia * w) / w + np.exp(ia)) + ia * (exp1(-ia) - exp1(-ia * w))
            t1 -= (1.0 - 1.0 / w)
            t1 -= ia * np.log(w)
        return t1 / (1.0 - 1.0 / w)

    def series(a, w):
        return (-0.5 * a ** 2 * (w - 1.) - 1j * a ** 3 / 6. * (w * w - 1.) / 2.) / (1. - 1. / w)

    w = 1e9
    aw = np.geomspace(1e-3, 1e4, 200)
    a = aw / w
    ex, se = exact(a, w), series(a, w)
    # Importance sampling: brute-force sampling of p(E) ~ 1/E^2 on [1, w] puts
    # essentially no events in the far tail that dominates <E^2>, so it
    # undershoots badly at small a*w. Sampling uniformly in ln E and
    # reweighting covers the whole range: <f> = ln(w) * E_x[f(E)/(E(1-1/w))].
    rng = np.random.default_rng(0)
    x = rng.random(4000000) * np.log(w)
    E = np.exp(x)
    wt = np.log(w) / (E * (1. - 1. / w))
    awmc = np.geomspace(1e-2, 1e4, 16)
    mc = np.array([np.mean((np.exp(1j * (xx / w) * E) - 1. - 1j * (xx / w) * E) * wt)
                   for xx in awmc])

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.plot(aw, np.abs(ex.imag), color=RED, lw=2.4, label="closed form (correct)")
    ax.plot(aw, np.abs(se.imag), color=BLUE, lw=2.2, ls="--", label="series expansion")
    ax.plot(awmc, np.abs(mc.imag), "ko", ms=7, label="direct sampling of the spectrum")
    ax.axvspan(1e-3, 5e-2, color="0.86", alpha=.8)
    ax.text(6e-3, 2e-7, "new guard:\nseries used here", fontsize=13, ha="center", color="0.3")
    ax.axvline(5e-2, color="0.4", lw=1.2, ls=":")
    ax.annotate("the old guard used the series\nover this whole range",
                xy=(3e2, 3e-1), xytext=(3e-2, 3e1), fontsize=14, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a\,w$   ($w=t_\mathrm{max}/e_0 = 10^{9}$, a muon step)")
    ax.set_ylabel(r"$|\,\mathrm{Im}\,\langle e^{iaE}-1-iaE\rangle\,|$")
    ax.set_ylim(1e-11, 1e4)
    ax.legend(fontsize=14, loc="lower right")
    save(fig, "bug")


# ------------------------------------------------------------------- impact
def fig_impact():
    u = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    pre = np.array([0.00819, 0.01475, 0.02371, 0.03507, 0.03820, 0.03495])
    pree = np.array([0.00064, 0.00100, 0.00148, 0.00221, 0.00265, 0.00287])
    post = np.array([0.00142, 0.00224, 0.00337, 0.00497, 0.00532, 0.00432])
    poste = np.array([0.00071, 0.00110, 0.00163, 0.00242, 0.00289, 0.00314])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.errorbar(u, pre * 100, yerr=pree * 100, color=BLUE, marker="s", ms=8, lw=2.2,
                label="before fix (64.9k tracks)")
    ax.errorbar(u, post * 100, yerr=poste * 100, color=RED, marker="o", ms=8, lw=2.2,
                label="after fix (13.5k tracks)")
    ax.axhline(0, color="0.5", lw=1.2, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"probe $u$")
    ax.set_ylabel(r"$\langle e^{-uz^2}\rangle_\mathrm{data} - \langle e^{-uz^2}\rangle_\mathrm{model}$  [%]")
    ax.set_ylim(-0.6, 5.0)
    ax.legend(fontsize=14)
    ax.set_title("J/$\\psi$ MC, per-track momentum closure", fontsize=17, color="0.3")
    save(fig, "impact")


# ------------------------------------------------------- clean-test results
def _panel(name, k, sim, legs, axcf, axls, title):
    avec = FUNCTIONALS[name]
    d = sim[SIM_BRANCH[name]][:, k] - legs[k][REF_BRANCH[name]]
    var, _, _ = model_variance(legs, k, avec)
    sigma = np.sqrt(var)
    z = d / sigma
    phi = model_phi(legs, k, avec, sigma, TAU)
    e = ecf(z, TAU)
    axcf.plot(TAU[1:], e.real[1:], color=GREY, lw=4.0, alpha=.75, label="Geant4  Re")
    axcf.plot(TAU[1:], phi.real[1:], color=RED, lw=1.9, ls="--", label="model  Re")
    axcf.plot(TAU[1:], e.imag[1:], color="#8fb3e0", lw=4.0, alpha=.95, label="Geant4  Im")
    axcf.plot(TAU[1:], phi.imag[1:], color=BLUE, lw=1.9, ls="--", label="model  Im")
    axcf.set_xscale("log")
    axcf.axhline(0, color="0.75", lw=.9)
    axcf.set_xlabel("$t$")
    axcf.set_ylabel(r"$\varphi(t)$")
    axcf.set_title(title, fontsize=17, color="0.3")
    axcf.legend(fontsize=13, ncol=2)
    lim = float(np.percentile(np.abs(z), 99.5))
    zg = np.linspace(-lim, lim, 601)
    # Invert on a UNIFORM fine grid: the log grid used for the CF comparison
    # has spacing ~0.07 at large t, which cannot resolve exp(-itz) for the
    # |z| ~ 100 reached by the delta-ray tail. phi is smooth, so interpolating
    # it onto a fine uniform grid is safe. (Display only -- the test itself
    # never inverts.)
    tu = np.linspace(0.0, TAU[-1], 40000)
    phiu = np.interp(tu, TAU, phi.real) + 1j * np.interp(tu, TAU, phi.imag)
    pz = np.array([np.trapezoid((phiu * np.exp(-1j * tu * zz)).real, tu) / np.pi for zz in zg])
    pz = np.where(pz > 1e-6 * np.nanmax(pz), pz, np.nan)
    axls.hist(z, bins=np.linspace(-lim, lim, 201), density=True, histtype="step",
              color="k", lw=1.8, label="Geant4")
    axls.plot(zg, pz, color=RED, lw=2.0, ls="--", label="model")
    axls.set_yscale("log")
    axls.set_xlabel(r"$z$ = residual / (propagator $\sigma$)")
    axls.set_ylabel("density")
    axls.legend(fontsize=13)
    return z, phi, sigma


def fig_results(sim, legs):
    for name, k, tag, title in (("qop", 19, "qop", "$q/p$ at $r=110$ cm — pure ionization"),
                                ("locx", 19, "ms", "local $x$ at $r=110$ cm — multiple scattering")):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.3))
        _panel(name, k, sim, legs, a1, a2, title)
        fig.tight_layout()
        save(fig, f"result_{tag}")


def fig_ladder(sim, legs):
    ks = list(range(len(legs)))
    r, mean, med = [], [], []
    for k in ks:
        avec = FUNCTIONALS["qop"]
        d = sim["qop"][:, k] - legs[k]["refqop"]
        var, _, _ = model_variance(legs, k, avec)
        s = np.sqrt(var)
        z = d / s
        r.append(np.median(sim["globr"][:, k]))
        mean.append(z.mean())
        med.append(np.median(z))
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.plot(r, med, color=RED, marker="o", ms=7, lw=2.2, label="median (mode-like)")
    ax.plot(r, mean, color=BLUE, marker="s", ms=7, lw=2.2, label="mean")
    ax.axhline(0, color="0.5", lw=1.2, ls=":")
    ax.set_xlabel("radius of the sensor plane  [cm]")
    ax.set_ylabel(r"$\langle z \rangle$,  median $z$")
    ax.legend(fontsize=15)
    ax.set_title("$q/p$ residual vs traversed material", fontsize=17, color="0.3")
    save(fig, "ladder")
    print("ladder r:", np.round(r, 1))
    print("ladder mean:", np.round(mean, 3))
    print("ladder median:", np.round(med, 3))


if __name__ == "__main__":
    fig_schematic()
    fig_bug()
    fig_impact()
    sim = load_sim(SIM)
    legs = load_model(MODEL)
    fig_ladder(sim, legs)
    fig_results(sim, legs)
