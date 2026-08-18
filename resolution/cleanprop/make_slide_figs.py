#!/usr/bin/env python3
"""Figures for the clean-propagation-test slide deck.

Run from calibration_studies/resolution/ (imports cf_propagation_test):
    python cleanprop/make_slide_figs.py

NORMALIZATION (changed 2026-08-14). The two result figures are standardized by
the FISHER scale s_F = sigma sqrt(1/I) rather than by the propagator's own
alpha-truncated sigma. Documents/Resolution/NOTES_FISHERNORM.md: the truncated
sigma is a convention that moves with StepLengthLimit (7.1x over a 100x step
change) while s_F moves by 0.13 %. I is the Fisher information of the model's
own density by EXACT FFT inversion of the block CF (cgf_channels.exact_density
-> fisher_exact, relative floor) -- never the saddlepoint, which NOTES_XXII
showed is 5-38 % wrong away from the mode.

The change is exactly a relabelling of the axes (z -> z sqrt(I), so the CF's t
axis stretches by the same factor and the CF's extrema are invariant); it
cannot create or destroy a data-model difference, and `--check` verifies
closure_F(u) == closure_sigma(u I) on this very pair. `--legacy` reproduces the
old sigma-normalized figures.

Each result figure now also PRINTS its closure at u = 1, for this plane and
averaged over the ladder, so the quantitative residual is visible on the plot
rather than left to the caption.
"""
import argparse
import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cgf_channels as cc
import fisher_norm as fn
from cf_propagation_test import (load_sim, load_model, model_phi, model_variance,
                                 ecf, weier_scalar, FUNCTIONALS, SIM_BRANCH,
                                 REF_BRANCH, TAU)

hep.style.use(hep.style.ROOT)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides", "assets")
os.makedirs(OUT, exist_ok=True)
# Dated mirror, same convention as cf_propagation_test's own plot output.
DATED = os.path.expanduser(
    f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_cleanprop/")

# pT=3, phi=0.70 -- the SAME matched pair the deck's closure results use.
#
# NOT the old pt10/phi0.20 pair: regen_models.sh's JOBS table regenerates
# model_pt10_eta0.30_phi0.20 against targets_mu_pt3_eta0.30.txt (phi=0.70),
# i.e. a different material path, so since 2026-08-08 that model's modules do
# not correspond to the pt10 sim's at all (20 sim planes vs 19 model legs,
# detid mismatch from index 0). Any comparison on that pair is meaningless.
SIM = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/sim_260808tight_pt3_eta0.30_phi0.70/simstates_*.root"
MODEL = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/model/model_mu_pt3_eta0.30.root"

RED, BLUE, GREY = "#A31F34", "#1f4e9c", "0.45"


def save(fig, name, dated=None):
    """Write png+pdf to the assets dir and to a DATED web directory.

    `dated` defaults to this module's, so the existing callers are unchanged;
    make_dir_figs and make_radsp_figs pass their own (`_directions`,
    `_radoff_species`) because each deck keeps a separate one.  That argument
    is the only thing that differed between the three copies of this function
    that existed before 2026-08-18.
    """
    for d in (OUT, dated if dated is not None else DATED):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"{name}.{ext}"),
                        bbox_inches="tight", dpi=160)
    plt.close(fig)
    # report the dir actually written, not the module default -- printing the
    # global here made a correctly-placed figure look misfiled
    print("wrote", name, "->", OUT, "and",
          dated if dated is not None else DATED)


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
def plane_scale(legs, k, name, norm="fisher"):
    """(scale, sigma, 1/I) for one plane.

    sigma is the propagator's own alpha-truncated width; 1/I is the Fisher
    information of the MODEL's density in units of that sigma, from the exact
    FFT inversion of the block CF on a grid matched to the block, with a
    RELATIVE density floor (an absolute one was a 57x error in this study).

    Routed through `fisher_norm.plane_scales` (2026-08-15) rather than calling
    `cc.exact_density` directly. It is the SAME computation -- `plane_scales`
    calls `exact_density` with exactly these defaults (`nt=1<<17`, `npad=32`,
    `lncut=-60`, all three channels) and `fisher_exact` with the same 1e-8
    RELATIVE floor -- but it is memoized per (model, functional), in-process and
    on disk. That matters here because `fig_results` already builds every
    plane's scale in its per-plane loop and `_panel` then rebuilt the outermost
    plane's from scratch: one full FFT inversion per panel, 2.4 s at plane 13.
    """
    avec = FUNCTIONALS[name]
    sigma = float(np.sqrt(model_variance(legs, k, avec)[0]))
    if norm == "sigma":
        return sigma, sigma, 1.0
    sc = fn.plane_scales(legs, name)
    invI = float(sc["invI"][k])
    if not np.isfinite(invI):
        raise RuntimeError(f"Fisher inversion failed at plane {k} ({name})")
    return float(sc["sF"][k]), sigma, invI


def closure_at(name, k, sim, legs, scale, probes=(1.0,)):
    """<e^{-u z^2}>_data - <e^{-u z^2}>_model on one plane, z = residual/scale.

    The t grid is matched to the LARGEST probe (the Weierstrass weight is
    e^{-t^2/4u}); TAU's 40.0 ceiling already covers u <= 1 with room to spare,
    and it is the grid the CF panel is drawn on, so figure and number agree.
    """
    avec = FUNCTIONALS[name]
    d = sim[SIM_BRANCH[name]][:, k] - legs[k][REF_BRANCH[name]]
    good = sim["valid"][:, k] & np.isfinite(d)
    z = d[good] / scale
    phi = model_phi(legs, k, avec, scale, TAU)
    return np.array([float(np.mean(np.exp(-u * z ** 2))) - weier_scalar(phi, u, TAU)
                     for u in probes])


def _fit_title(fig, a1, a2, sizes=(17, 16, 15, 14, 13, 12, 11), pad=6.0):
    """Shrink the left/top-left title until it clears the right panel's note.

    THE FAILURE THIS GUARDS AGAINST IS SILENT.  At the published 17 pt the
    second title line runs UNDER the right panel's closure annotation, and
    `bbox_inches='tight'` then WIDENS the saved image instead of clipping it --
    so the overlap never shows up in the file and only shows up on the page.
    The two bounding boxes are compared in display coordinates after a draw,
    i.e. this is a measurement rather than a font-size guess, and it keeps the
    title aspect.  Returns the size actually used.

    Measured live for every set that uses it: the toy geometry strings are long
    on the left, and the radoff-species note is long on the right (it carries
    the rms and the peak u as well as the u = 1 value), so the collision is
    real for both and not assumed.

    HISTORY.  This was three character-identical copies (make_toy_figs,
    make_dir_figs, make_radsp_figs), duplicated deliberately -- the recorded
    argument was that the scripts are read side by side and a shared helper one
    of them outgrows is worse than three copies.  It was consolidated on
    2026-08-18 after the copies produced exactly the failure that argument did
    not cover: `make_dir_figs` had a two-axis `panel` while `make_radsp_figs`
    had the four-axis `panels`, so the closure figures silently lost their
    difference panel and nobody noticed until the plots were compared by eye.
    """
    r = fig.canvas.get_renderer()
    w = fig.get_window_extent(r)
    fs = sizes[-1]
    for fs in sizes:
        a1.title.set_fontsize(fs)
        fig.canvas.draw()
        b1 = a1.title.get_window_extent(r)
        b2 = a2.title.get_window_extent(r)
        if b1.x1 < b2.x0 - pad and b1.x0 > w.x0 - pad:
            break
    return fs


def _shrink(ax, xs=15, ys=13, ts=13):
    """`hep.style.ROOT` sizes labels for a full-height panel.  A 1/4-height
    ratio panel inherits them and the y-label then runs off the canvas -- which
    `bbox_inches='tight'` does not clip but does not fix either: it widens the
    image and the label ends up outside the plot box in the deck.  Measured on
    the first attempt at this set, which is why the sizes are set rather than
    left to the style."""
    ax.xaxis.label.set_size(xs)
    ax.yaxis.label.set_size(ys)
    ax.tick_params(labelsize=ts)


def panels(axcf, axcfr, axls, axlsr, z, phi, title, note):
    """Four axes: CF, CF difference, lineshape, lineshape ratio.

    Colours, line widths, scales, legend sizes and the annotation-as-title
    convention are `make_slide_figs._panel`'s, verbatim, so that these figures
    sit beside the published ones without a visible style break.
    """
    e = ecf(z, TAU)

    # ---------------------------------------------------------------- CF
    axcf.plot(TAU[1:], e.real[1:], color=GREY, lw=4.0, alpha=.75,
              label="Geant4  Re")
    axcf.plot(TAU[1:], phi.real[1:], color=RED, lw=1.9, ls="--",
              label="model  Re")
    axcf.plot(TAU[1:], e.imag[1:], color="#8fb3e0", lw=4.0, alpha=.95,
              label="Geant4  Im")
    axcf.plot(TAU[1:], phi.imag[1:], color=BLUE, lw=1.9, ls="--",
              label="model  Im")
    axcf.set_xscale("log")
    axcf.axhline(0, color="0.75", lw=.9)
    axcf.set_ylabel(r"$\varphi(t)$")
    axcf.set_title(title, fontsize=17, color="0.3")
    axcf.legend(fontsize=13, ncol=2)
    axcf.tick_params(labelbottom=False)

    # ------------------------------------------------- CF: sim MINUS model
    # A pointwise RATIO diverges: Re(phi) crosses zero near t ~ 3 for every
    # species here and Im(phi) is zero at t = 0 by construction.  The
    # DIFFERENCE is bounded, carries the CF's own units, and is exactly the
    # closure statistic's integrand -- see the module docstring.
    dre, dim = e.real - phi.real, e.imag - phi.imag
    axcfr.plot(TAU[1:], dre[1:], color=RED, lw=1.9, label="Re")
    axcfr.plot(TAU[1:], dim[1:], color=BLUE, lw=1.9, label="Im")
    axcfr.set_xscale("log")
    axcfr.axhline(0, color="0.75", lw=.9)
    axcfr.set_xlabel("$t$")
    axcfr.set_ylabel("Geant4 $-$ model")
    axcfr.legend(fontsize=11, ncol=2, loc="upper left")
    m = float(np.nanmax(np.abs(np.concatenate([dre[1:], dim[1:]]))))
    axcfr.set_ylim(-1.35 * m, 1.35 * m)
    _shrink(axcfr)

    # ---------------------------------------------------------- lineshape
    lim = float(np.percentile(np.abs(z), 99.5))
    zg = np.linspace(-lim, lim, 601)
    # Invert on a UNIFORM fine grid: the log grid used for the CF comparison
    # has spacing ~0.07 at large t, which cannot resolve exp(-itz) for the
    # |z| ~ 100 the delta-ray tail reaches.  phi is smooth, so resampling it is
    # safe.  (Display only -- the closure statistic never inverts.)
    tu = np.linspace(0.0, TAU[-1], 40000)
    phiu = np.interp(tu, TAU, phi.real) + 1j * np.interp(tu, TAU, phi.imag)
    pz = np.array([np.trapezoid((phiu * np.exp(-1j * tu * zz)).real, tu) / np.pi
                   for zz in zg])
    pz = np.where(pz > 1e-6 * np.nanmax(pz), pz, np.nan)

    bins = np.linspace(-lim, lim, 201)
    h, _ = np.histogram(z, bins=bins, density=True)
    ctr_ = 0.5 * (bins[1:] + bins[:-1])
    n, _ = np.histogram(z, bins=bins)
    # Poisson error on the density, for the ratio panel only.
    with np.errstate(divide="ignore", invalid="ignore"):
        herr = np.where(n > 0, h / np.sqrt(np.maximum(n, 1)), np.nan)

    axls.hist(z, bins=bins, density=True, histtype="step", color="k", lw=1.8,
              label="Geant4")
    axls.plot(zg, pz, color=RED, lw=2.0, ls="--", label="model")
    axls.set_yscale("log")
    axls.set_ylabel("density")
    axls.legend(fontsize=13, loc="upper right")
    axls.set_title(note, fontsize=15, color="0.2")
    axls.tick_params(labelbottom=False)

    # ------------------------------------------------- lineshape: sim/model
    pzc = np.interp(ctr_, zg, np.nan_to_num(pz, nan=0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pzc > 0, h / pzc, np.nan)
        rerr = np.where(pzc > 0, herr / pzc, np.nan)
    # Only where BOTH are populated: an empty Geant4 bin over a model density
    # of 1e-30 is a 0/0, not a measurement.
    ok = np.isfinite(ratio) & (n > 0) & (pzc > 1e-4 * np.nanmax(pzc))
    axlsr.errorbar(ctr_[ok], ratio[ok], yerr=rerr[ok], fmt="o", ms=2.6,
                   lw=0, elinewidth=0.9, color="k")
    axlsr.axhline(1.0, color=RED, lw=1.6, ls="--")
    axlsr.set_xlabel(r"$z$ = residual / $s_F=\sigma\sqrt{1/I}$")
    axlsr.set_ylabel("Geant4 / model")
    axlsr.set_ylim(0.5, 1.5)
    axlsr.set_xlim(*axls.get_xlim())
    _shrink(axlsr)
    return dict(cfmax=m, nratio=int(ok.sum()),
                ratio_med=float(np.nanmedian(ratio[ok])) if ok.any() else np.nan)




def _panel(name, k, sim, legs, axcf, axls, title, norm="fisher", note=None):
    avec = FUNCTIONALS[name]
    d = sim[SIM_BRANCH[name]][:, k] - legs[k][REF_BRANCH[name]]
    # per-plane acceptance leaves NaN where a ray never crossed this plane
    good = sim["valid"][:, k] & np.isfinite(d)
    d = d[good]
    sigma, sig_trunc, invI = plane_scale(legs, k, name, norm)
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
    axls.set_xlabel(r"$z$ = residual / $s_F$,   $s_F=\sigma\sqrt{1/I}$ (Fisher)"
                    if norm != "sigma" else
                    r"$z$ = residual / (propagator $\sigma$)")
    axls.set_ylabel("density")
    axls.legend(fontsize=13, loc="upper right")
    if note:
        # As the panel TITLE, not as an inset. Inside the axes it collides with
        # the legend in the q/p figure and with the Moliere peak in the MS one
        # (both were tried); the title band is free in both and needs no
        # rescaling of a log density axis to make room.
        axls.set_title(note, fontsize=15, color="0.2")
    return z, phi, sigma, sig_trunc, invI


_CL_CTX = None


def _closure_one_plane(kk):
    """closure(u=1) on plane kk in the requested normalization. Module level so
    `pmap` can pickle it by reference; the arrays come from the fork."""
    legs, sim, name, norm = _CL_CTX
    return closure_at(name, kk, sim, legs,
                      plane_scale(legs, kk, name, norm)[0])[0]


def fig_results(sim, legs, norm="fisher"):
    # outermost plane, not a hardcoded index: the model file was regenerated
    # 2026-08-08 with 19 legs (was 20), so k=19 is now out of range
    k = len(legs) - 1
    r = float(np.nanmedian(sim["globr"][:, k]))
    nplane = len(legs)
    out = {}
    for name, tag, what in (("qop", "qop", "pure ionization"),
                            ("locx", "ms", "multiple scattering")):
        lab = {"qop": "$q/p$", "locx": "local $x$"}[name]
        # The ladder mean is the number the deck's closure slide quotes, so it
        # is computed here in the SAME normalization rather than carried over.
        # One plane per forked worker (fisher_norm.pmap): each plane's scale and
        # model CF depend only on the legs, so this is the same serial code run
        # on 19 cores instead of one, and returns the same float64.
        global _CL_CTX
        _CL_CTX = (legs, sim, name, norm)
        cl = np.array(fn.pmap(_closure_one_plane, range(nplane)))
        print(f"  {name} closure(u=1, {norm}) per plane: "
              + " ".join(f"{v:+.4f}" for v in cl))
        print(f"  {name} closure(u=1, {norm}) this plane {cl[k]:+.4f}, "
              f"ladder mean {cl.mean():+.4f}, rms {cl.std():.4f}")
        # The closure statistic, ON the plot. Two curves can lie on top of each
        # other over 2.5 decades of a log density and still differ by several
        # percent in a bounded functional -- which is what the statistic is
        # for, so it is quoted rather than left to the caption.
        note = (r"closure $\langle e^{-uz^2}\rangle_{\rm data}-"
                r"\langle e^{-uz^2}\rangle_{\rm model}$ at $u=1$" "\n"
                f"{cl[k]:+.3f} here  ·  {nplane}-plane mean {cl.mean():+.3f}")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 5.3))
        title = (f"{lab} at $r={r:.0f}$ cm — {what}\n"
                 f"real tracker geometry, {nplane} planes · "
                 r"$\mu$, $p_T=3$ GeV, $\eta=0.30$, $\phi=0.70$")
        _, _, s, sig_trunc, invI = _panel(name, k, sim, legs, a1, a2, title,
                                          norm=norm, note=note)
        fig.tight_layout()
        save(fig, f"result_{tag}" + ("_sigma" if norm == "sigma" else ""))
        out[name] = dict(closure=cl, k=k, r=r, scale=s, sigma=sig_trunc,
                         invI=invI)
    os.makedirs(DATED, exist_ok=True)
    np.savez(os.path.join(DATED, f"result_closure_{norm}.npz"),
             **{f"{n}_{q}": v for n, d in out.items() for q, v in d.items()})
    return out


def fig_ladder(sim, legs):
    ks = list(range(len(legs)))
    r, mean, med = [], [], []
    for k in ks:
        avec = FUNCTIONALS["qop"]
        d = sim["qop"][:, k] - legs[k]["refqop"]
        good = sim["valid"][:, k] & np.isfinite(d)   # NaN under per-plane acceptance
        var, _, _ = model_variance(legs, k, avec)
        s = np.sqrt(var)
        z = d[good] / s
        r.append(np.nanmedian(sim["globr"][:, k]))
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


def check_identity(sim, legs):
    """closure_F(u) must equal closure_sigma(u I) IDENTICALLY -- the two differ
    only by z -> z sqrt(I). Computed by two independent routes (each builds its
    own scale and its own model CF), so this checks the implementation and,
    more importantly, shows the renormalization cannot move a data-model
    difference. Also confirms the cf_ms_exact j0(x)-1 guard is live and
    measures what it does HERE rather than quoting the toy number.
    """
    import cf_ms_exact
    k = len(legs) - 1
    print("IDENTITY  closure_F(u) == closure_sigma(u*I)   (plane %d)" % k)
    for name in ("qop", "locx"):
        sF, sig, invI = plane_scale(legs, k, name, "fisher")
        for u in (0.01, 0.1, 1.0):
            a = closure_at(name, k, sim, legs, sF, (u,))[0]
            b = closure_at(name, k, sim, legs, sig, (u / invI,))[0]
            print(f"  {name:5s} u={u:<5g} F={a:+.7f} sigma={b:+.7f} "
                  f"diff={a - b:+.2e}")
    print("J0M1 GUARD (cf_ms_exact.set_j0_guard)")
    g = cf_ms_exact.gshape(np.array([1e-9]), 3.16e4)[0]
    cf_ms_exact.set_j0_guard(False)
    g0 = cf_ms_exact.gshape(np.array([1e-9]), 3.16e4)[0]
    c0 = closure_at("locx", k, sim, legs,
                    plane_scale(legs, k, "locx", "fisher")[0])[0]
    cf_ms_exact.set_j0_guard(True)
    c1 = closure_at("locx", k, sim, legs,
                    plane_scale(legs, k, "locx", "fisher")[0])[0]
    print(f"  gshape(1e-9, ymax=3.16e4): on {g:.6e}  off {g0:.6e}  "
          f"rel {g0 / g - 1:+.3e}   <- knob is live")
    print(f"  locx closure(u=1): on {c1:+.8f}  off {c0:+.8f}  "
          f"diff {c1 - c0:+.2e}   <- effect on THIS block")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the old sigma-normalized result figures")
    ap.add_argument("--check", action="store_true",
                    help="identity + j0-guard checks, no figures")
    ap.add_argument("--results-only", action="store_true")
    args = ap.parse_args()

    if not (args.check or args.results_only):
        fig_schematic()
        fig_bug()
        fig_impact()
    # per-plane, NOT the default "modal": the modal-sequence cut drops whole
    # tracks that miss a plane, which is tail-first selection and was the
    # entire pT=3 non-closure (NOTES 2026-08-08). These figures were made with
    # the modal default before that was understood.
    sim = load_sim(SIM, acceptance="perplane")
    # via fisher_norm.load, not load_model directly: it is the same call, but it
    # registers the model file's content hash so the per-plane Fisher scales can
    # be served from (and written to) the on-disk cache.
    legs = fn.load(MODEL)
    # Guard: a sim/model pair built on different rays silently produces a
    # plausible-looking disagreement (this is how the pt10 mis-targeting was
    # found). Compare the module sequences before comparing any physics.
    sd = sim["detid"]
    ld = np.array([l["detid"] for l in legs])
    assert len(sd) == len(ld) and np.all(sd == ld), (
        f"sim/model module sequences differ ({len(sd)} planes vs {len(ld)} legs) "
        f"-- these are different rays, the comparison would be meaningless")
    if args.check:
        check_identity(sim, legs)
        raise SystemExit(0)
    if not args.results_only:
        fig_ladder(sim, legs)
    fig_results(sim, legs, norm="sigma" if args.legacy else "fisher")
