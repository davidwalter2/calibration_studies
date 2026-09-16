#!/usr/bin/env python3
"""Figures for the J/psi FSR kernel: the sample's own against exact QED.

One panel per file, PDF + PNG twin, ratio panel wherever two curves are
compared (`resolution/pubhtml.savefig`).

  * `jpsi_u_density`    -- ``K_u(u)``, ``u = -ln(m_gen/M)``, on log bins; MC
    against the analytic kernel, ratio panel.  The MC's core below
    ``u ~ 1e-4`` is the GENERATOR'S OWN truncated Breit-Wigner
    (`443:mWidth = 9.26e-5`, `mMin/mMax = 3.0960/3.0978`), not radiation, and
    the analytic curve is drawn with and without it.
  * `jpsi_u_tail`       -- ``P(u > u0)``, the same two, ratio panel.
  * `jpsi_dm_core`      -- ``dm = m_gen - M`` within +-1.5 MeV, linear: the
    truncated Breit-Wigner, and how far the delta at ``MJPSI`` is from it.
  * `jpsi_meanshift`    -- ``<dm | |dm| < w>`` against the window halfwidth
    ``w``, in MeV and in units of the momentum scale.  The card's window is
    marked; this is the number the delta kernel forces onto the scale.
  * `jpsi_cf`           -- ``|phi_K(t)|`` and ``arg phi_K(t)`` over the ``t``
    the term actually reads (``max(tau)/sigma``), MC against analytic.
  * `jpsi_pure_fsr`     -- the PURE FSR kernel, with the generator lineshape
    divided out candidate by candidate through the status-746 pre-FSR muons
    (`data/jpsi_fsr_from_746.npz`, the whole 21.7 M-candidate production),
    against the analytic one.  This is the clean QED comparison; the two
    truncations -- the cache's gen window and the ALCARECO's own
    ``Jpsitrk_mass`` window -- are marked.
  * `jpsi_massexact`    -- ``R_exact/R1 - 1`` against ``z`` at the J/psi, the
    Upsilon and the Z: why `FSRKernel(mass_exact=True)` is a narrow-resonance
    option.

usage:  python3 plot_jpsi_fsr.py [--pairs ...] [-o OUTDIR]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.gridspec import GridSpec

from wums import logging

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "resolution"))
sys.path.insert(0, _HERE)
import fsr_analytic as fa  # noqa: E402
import pubhtml  # noqa: E402

MJPSI = 3.0969
#: the generator's own J/psi lineshape (the GEN fragment of
#: TRK-RunIISummer20UL16GEN-00002): a Breit-Wigner truncated to [mMin, mMax]
GEN_MWIDTH = 9.26e-5
GEN_MMIN, GEN_MMAX = 3.0960, 3.0978
WINDOW = 0.35

C_MC, C_AN, C_AN2 = "#1f77b4", "#d62728", "#2ca02c"


def _ratio_fig(h=(3, 1)):
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(2, 1, height_ratios=h, hspace=0.06)
    a0 = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    a0.tick_params(labelbottom=False)
    return fig, a0, a1


def load_dm(pairs, kernel_npz):
    """The kernel's own ``dm`` samples: from the kernel npz if it saved them."""
    if kernel_npz and os.path.exists(kernel_npz):
        k = np.load(kernel_npz, allow_pickle=True)
        if "dm" in k.files:
            logger.info(f"dm from {kernel_npz}")
            return np.asarray(k["dm"], dtype=np.float64)
    logger.info(f"dm recomputed from {pairs}")
    with np.load(pairs, allow_pickle=True) as d:
        sig = d["sigma"].astype(np.float64)
        mg = d["eta"].astype(np.float64)
        z = d["z"].astype(np.float64)
        c2 = (d["chisqval"].astype(np.float64)
              / np.maximum(d["ndof"].astype(np.float64), 1.0))
    m = z * sig + mg
    sel = (np.isfinite(m) & (sig > 0) & (c2 < 3.0)
           & (sig / np.maximum(np.abs(m), 1e-9) < 0.10))
    return mg[sel] - MJPSI


def bw_truncated(n, rng):
    """Samples of the generator's own truncated Breit-Wigner, as ``dm``."""
    g = 0.5 * GEN_MWIDTH
    lo = np.arctan((GEN_MMIN - MJPSI) / g)
    hi = np.arctan((GEN_MMAX - MJPSI) / g)
    return g * np.tan(lo + (hi - lo) * rng.random(n))


def analytic_kernel(mass_exact=True):
    k = fa.FSRKernel(MJPSI, variant="exp2nll", pair=("e", "mu"),
                     mass_exact=mass_exact)
    c = k.cells(n_fine=20000)
    w, m1 = c[0], c[1]
    good = w > 0
    w, ub = w[good], m1[good] / w[good]
    o = np.argsort(ub)
    return k, ub[o], w[o] / w.sum()


# --------------------------------------------------------------------------
def fig_u_density(dm, ub, wj, out):
    """K_u(u) on log bins, MC vs analytic, with and without the generator BW."""
    u_mc = -np.log1p(dm / MJPSI)
    e = np.geomspace(1e-6, 0.13, 61)
    c = np.sqrt(e[:-1] * e[1:])
    dlnu = np.diff(np.log(e))

    def dens(u, w=None):
        h, _ = np.histogram(u, bins=e, weights=w)
        tot = len(u) if w is None else w.sum()
        return h / tot / dlnu, np.sqrt(np.maximum(h, 1.0)) / tot / dlnu

    y_mc, ey_mc = dens(u_mc)
    y_an, _ = dens(ub, wj)
    # the analytic kernel folded with the generator's own truncated BW
    rng = np.random.default_rng(1)
    n = 4_000_000
    draw = rng.choice(len(ub), n, p=wj)
    dm_an = MJPSI * np.expm1(-ub[draw]) + bw_truncated(n, rng)
    y_anbw, _ = dens(-np.log1p(dm_an / MJPSI))

    fig, a0, a1 = _ratio_fig()
    a0.step(e[:-1], y_mc, where="post", color=C_MC, lw=2,
            label=f"MC gen record (PHOTOS++ 3.61), {len(dm) / 1e6:.2f} M")
    a0.step(e[:-1], y_an, where="post", color=C_AN, lw=2,
            label=r"analytic: exp2nll + $e^+e^-\!/\mu^+\mu^-$ pairs, mass-exact")
    a0.step(e[:-1], y_anbw, where="post", color=C_AN2, lw=1.6, ls="--",
            label=r"analytic $\otimes$ generator Breit-Wigner")
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylabel(r"$u\,\mathrm{d}P/\mathrm{d}u$")
    a0.set_ylim(5e-3, 2.0)
    a0.legend(fontsize=12, loc="upper right")
    a0.text(0.02, 0.06, r"$u=-\ln(m_{\mathrm{gen}}/M_{J/\psi})$" "\n"
            r"below $u\sim3\times10^{-4}$ the MC core is the GENERATOR'S"
            "\n" r"truncated Breit-Wigner, not radiation",
            transform=a0.transAxes, fontsize=12, va="bottom")
    with np.errstate(divide="ignore", invalid="ignore"):
        a1.step(e[:-1], y_mc / y_an, where="post", color=C_MC, lw=2)
        a1.step(e[:-1], y_mc / y_anbw, where="post", color=C_AN2, lw=1.6,
                ls="--")
    a1.axhline(1.0, color="k", lw=1, ls=":")
    a1.set_xscale("log")
    a1.set_ylim(0.0, 2.6)
    a1.set_xlabel(r"$u$")
    a1.set_ylabel("MC / model")
    pubhtml.savefig(fig, os.path.join(out, "jpsi_u_density.pdf"))
    plt.close(fig)
    return dm_an


def fig_u_tail(dm, ub, wj, dm_an, out):
    u_mc = np.sort(-np.log1p(dm / MJPSI))
    u0 = np.geomspace(1e-6, 0.12, 200)
    p_mc = 1.0 - np.searchsorted(u_mc, u0) / len(u_mc)
    cw = np.cumsum(wj)
    p_an = 1.0 - np.interp(u0, ub, cw)
    u_anbw = np.sort(-np.log1p(dm_an / MJPSI))
    p_anbw = 1.0 - np.searchsorted(u_anbw, u0) / len(u_anbw)

    fig, a0, a1 = _ratio_fig()
    a0.plot(u0, p_mc, color=C_MC, lw=2, label="MC gen record")
    a0.plot(u0, p_an, color=C_AN, lw=2, label="analytic (FSR only)")
    a0.plot(u0, p_anbw, color=C_AN2, lw=1.6, ls="--",
            label=r"analytic $\otimes$ generator Breit-Wigner")
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylabel(r"$P(u>u_0)$")
    a0.set_ylim(5e-4, 1.5)
    a0.legend(fontsize=12, loc="lower left")
    with np.errstate(divide="ignore", invalid="ignore"):
        a1.plot(u0, p_mc / p_an, color=C_MC, lw=2)
        a1.plot(u0, p_mc / p_anbw, color=C_AN2, lw=1.6, ls="--")
    for ax in (a0, a1):
        ax.axvline(-np.log((MJPSI - WINDOW) / MJPSI), color="grey", lw=1.3,
                   ls="--")
    a0.text(0.62, 0.30, "the cache's gen window\n"
            r"truncates the MC here", transform=a0.transAxes, fontsize=12)
    a1.axhline(1.0, color="k", lw=1, ls=":")
    a1.set_xscale("log")
    a1.set_ylim(0.0, 2.2)
    a1.set_xlabel(r"$u_0$")
    a1.set_ylabel("MC / model")
    pubhtml.savefig(fig, os.path.join(out, "jpsi_u_tail.pdf"))
    plt.close(fig)


def fig_dm_core(dm, dm_an, out):
    e = np.linspace(-1.5, 1.5, 301)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(dm * 1e3, bins=e, histtype="step", color=C_MC, lw=2, density=True,
            label="MC gen record")
    ax.hist(dm_an * 1e3, bins=e, histtype="step", color=C_AN2, lw=1.8,
            ls="--", density=True,
            label=r"analytic $\otimes$ generator Breit-Wigner")
    ax.axvline(0.0, color="k", lw=1.4,
               label=r"the card's $\delta$ at $M_{J/\psi}=3.0969$ GeV")
    for x, lab in ((1e3 * (GEN_MMIN - MJPSI), None),
                   (1e3 * (GEN_MMAX - MJPSI), "generator mMin / mMax")):
        ax.axvline(x, color="grey", lw=1.2, ls=":", label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\mathrm{gen}} - M_{J/\psi}$  [MeV]")
    ax.set_ylabel("density  [1/MeV]")
    ax.legend(fontsize=13, loc="upper left")
    pubhtml.savefig(fig, os.path.join(out, "jpsi_dm_core.pdf"))
    plt.close(fig)


def fig_meanshift(dm, ub, wj, out):
    w = np.geomspace(2e-3, 0.5, 120)
    dms = np.sort(dm)
    mc = np.empty_like(w)
    cs = np.concatenate([[0.0], np.cumsum(dms)])
    for i, ww in enumerate(w):
        lo = np.searchsorted(dms, -ww)
        hi = np.searchsorted(dms, ww)
        mc[i] = (cs[hi] - cs[lo]) / max(hi - lo, 1)
    dm_an = MJPSI * np.expm1(-ub)
    o = np.argsort(dm_an)
    da, wa = dm_an[o], wj[o]
    ca = np.concatenate([[0.0], np.cumsum(wa)])
    cda = np.concatenate([[0.0], np.cumsum(wa * da)])
    an = np.empty_like(w)
    for i, ww in enumerate(w):
        lo = np.searchsorted(da, -ww)
        hi = np.searchsorted(da, ww)
        an[i] = (cda[hi] - cda[lo]) / max(ca[hi] - ca[lo], 1e-12)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(w, mc * 1e3, color=C_MC, lw=2, label="MC gen record")
    ax.plot(w, an * 1e3, color=C_AN, lw=2, label="analytic (exact QED)")
    ax.axvline(WINDOW, color="k", lw=1.2, ls="--",
               label=r"the card's window, $\pm 0.35$ GeV")
    ax.set_xscale("log")
    ax.set_xlabel(r"window halfwidth $w$  [GeV]")
    ax.set_ylabel(r"$\langle m_{\mathrm{gen}}-M\,|\,"
                  r"|m_{\mathrm{gen}}-M|<w\rangle$  [MeV]")
    sec = ax.secondary_yaxis(
        "right", functions=(lambda y: y * 1e-3 / MJPSI * 1e3,
                            lambda y: y * MJPSI / 1e-3 / 1e3))
    sec.set_ylabel(r"relative momentum scale  [$10^{-3}$]")
    ax.legend(fontsize=13, loc="lower left")
    pubhtml.savefig(fig, os.path.join(out, "jpsi_meanshift.pdf"))
    plt.close(fig)


def fig_cf(kmc, kan, sigmin, out):
    t = np.asarray(kmc["phik_t"], float)
    cmc = np.asarray(kmc["phik_re"], float) + 1j * np.asarray(kmc["phik_im"],
                                                              float)
    can = (np.interp(t, kan["phik_t"], kan["phik_re"])
           + 1j * np.interp(t, kan["phik_t"], kan["phik_im"]))
    tmax = 7.8926 / sigmin
    fig, a0, a1 = _ratio_fig()
    a0.plot(t, np.abs(cmc), color=C_MC, lw=2, label=r"MC gen record")
    a0.plot(t, np.abs(can), color=C_AN, lw=2,
            label=r"analytic $\otimes$ Breit-Wigner")
    a0.axvline(tmax, color="k", lw=1.2, ls="--",
               label=rf"$\max(\tau)/\sigma_{{\min}} = {tmax:.0f}$ GeV$^{{-1}}$")
    a0.set_ylabel(r"$|\varphi_K(t)|$")
    a0.set_xlim(0, 1000)
    a0.legend(fontsize=13, loc="upper right")
    a1.plot(t, np.angle(cmc) * 1e3, color=C_MC, lw=2)
    a1.plot(t, np.angle(can) * 1e3, color=C_AN, lw=2)
    a1.axvline(tmax, color="k", lw=1.2, ls="--")
    a1.set_xlabel(r"$t$  [GeV$^{-1}$]")
    a1.set_ylabel(r"$\arg\varphi_K$  [mrad]")
    pubhtml.savefig(fig, os.path.join(out, "jpsi_cf.pdf"))
    plt.close(fig)


U_CACHE = -np.log((3.0969 - 0.35) / 3.0969)     # the pairs cache's gen window
U_ALCA = np.log(3.0969 / 2.7019)                 # the ALCARECO Jpsitrk window


def fig_pure_fsr(path, out):
    """The 746-derived FSR-only kernel against the analytic one."""
    if not os.path.exists(path):
        logger.warning(f"{path} is not there; skipping jpsi_pure_fsr")
        return
    z = np.load(path, allow_pickle=True)
    e = np.asarray(z["u_edges"], float)
    c = np.asarray(z["u_centres"], float)
    y = np.asarray(z["dens"], float)
    ey = np.asarray(z["dens_err"], float)
    ya = np.asarray(z["dens_analytic"], float)
    n = int(z["n"])

    fig, a0, a1 = _ratio_fig()
    a0.errorbar(c, y, yerr=ey, fmt="o", ms=3, color=C_MC, lw=0,
                elinewidth=1.2,
                label=f"MC, lineshape divided out ({n / 1e6:.1f} M)")
    a0.step(e[:-1], ya, where="post", color=C_AN, lw=2,
            label=r"analytic: exp2nll + $e^+e^-\!/\mu^+\mu^-$ pairs, mass-exact")
    for x, lab, ls in ((U_CACHE, r"cache gen window", "--"),
                       (U_ALCA, r"ALCARECO $m_{\mu\mu}$ window", ":")):
        a0.axvline(x, color="grey", lw=1.3, ls=ls, label=lab)
        a1.axvline(x, color="grey", lw=1.3, ls=ls)
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylim(1e-3, 6e-2)
    a0.set_ylabel(r"$u\,\mathrm{d}P/\mathrm{d}u$")
    a0.legend(fontsize=12, loc="lower left")
    a0.text(0.03, 0.93, r"$u=-\ln(m_{\mathrm{post}}/m_{\mathrm{pre}})$, "
            r"$m_{\mathrm{pre}}$ from the status-746 muons",
            transform=a0.transAxes, fontsize=12, va="top")
    with np.errstate(divide="ignore", invalid="ignore"):
        a1.errorbar(c, y / ya, yerr=ey / ya, fmt="o", ms=3, color=C_MC, lw=0,
                    elinewidth=1.2)
    a1.axhline(1.0, color="k", lw=1, ls=":")
    a1.set_xscale("log")
    a1.set_ylim(0.0, 1.6)
    a1.set_xlabel(r"$u$")
    a1.set_ylabel("MC / QED")
    pubhtml.savefig(fig, os.path.join(out, "jpsi_pure_fsr.pdf"))
    plt.close(fig)


def fig_massexact(out):
    z = np.geomspace(0.01, 0.999, 240)
    fig, ax = plt.subplots(figsize=(9, 7))
    for m, lab, col in ((MJPSI, r"$J/\psi$, 3.097 GeV", C_AN),
                        (9.4603, r"$\Upsilon(1S)$, 9.460 GeV", C_AN2),
                        (91.1876, r"$Z$, 91.188 GeV", C_MC)):
        k = fa.FSRKernel(m, variant="exp2nll", pair=())
        zz = z[z * m * m > 4 * fa.M_MU**2]
        r = fa.r1_fast(m, zz) / k.r1(zz, 1.0 - zz) - 1.0
        ax.plot(zz, r * 1e3, lw=2, color=col, label=lab)
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$z=(m_{\mathrm{post}}/m_{\mathrm{pre}})^2$")
    ax.set_ylabel(r"$R_{\mathrm{exact}}/R_1 - 1$  [$10^{-3}$]")
    ax.legend(fontsize=14, loc="lower right")
    ax.text(0.03, 0.06, "muon-mass terms of the exact\n"
            r"$O(\alpha)$ matrix element", transform=ax.transAxes, fontsize=14)
    pubhtml.savefig(fig, os.path.join(out, "jpsi_massexact.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=os.path.join(
        _HERE, "..", "fullscale", "runs", "jpairs_v2_n600.npz"))
    p.add_argument("--kern-mc", default=os.path.join(_HERE, "data",
                                                     "jpsi_kern_mc.npz"))
    p.add_argument("--kern-data", default=os.path.join(
        _HERE, "data", "jpsi_kern_data.npz"))
    p.add_argument("--sigmin", type=float, default=0.0103)
    p.add_argument("-o", "--outdir", default=None)
    a = p.parse_args()
    out = a.outdir or pubhtml.figdir("jpsi_fsr")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)
    logger.info(f"figures -> {out}")

    dm = load_dm(a.pairs, a.kern_mc)
    k, ub, wj = analytic_kernel()
    logger.info(f"MC: n = {len(dm)}, <dm> = {dm.mean() * 1e3:+.4f} MeV; "
                f"analytic: beta = {k.beta:.6f}")
    dm_an = fig_u_density(dm, ub, wj, out)
    fig_u_tail(dm, ub, wj, dm_an, out)
    fig_dm_core(dm, dm_an, out)
    fig_meanshift(dm, ub, wj, out)
    fig_cf(np.load(a.kern_mc), np.load(a.kern_data), a.sigmin, out)
    fig_pure_fsr(os.path.join(_HERE, "data", "jpsi_fsr_from_746.npz"), out)
    fig_massexact(out)
    logger.info("done")


if __name__ == "__main__":
    main()
