#!/usr/bin/env python3
"""Figures for the K_S -> pi+ pi-(gamma) QED kernel.

One panel per file, PDF + PNG twin, ratio panel wherever two curves are
compared (`resolution/pubhtml.savefig`).  Everything is built from
`fsr_analytic.ScalarIBKernel`; nothing here is fitted or tuned.

  * `ks_egamma`     -- ``(1/Gamma_0) dGamma/dE*_gamma``, the exact scalar-QED
    inner bremsstrahlung against the pure eikonal ``b/E*``, ratio panel.  The
    window edges a K_S mass term would use and the two PDG photon-energy cuts
    are marked: the windows live entirely in the region where the ratio is
    within 16 % of 1, the PDG cuts where it is a factor 1.6-3.5 away.
  * `ks_rate_gate`  -- the NORMALISATION GATE: the integrated radiative
    fraction above a photon-energy cut against the two PDG entries
    (RAMBERG 93 + TAUREG 76 average at 50 MeV, RAMBERG 93 at 20 MeV).
  * `ks_kernel_u`   -- ``u dP/du``, the exponentiated kernel against the
    fixed-order O(alpha) spectrum, ratio panel.
  * `ks_meanshift`  -- ``<dm | |dm| < w>`` against the window halfwidth, in
    MeV and as a fraction of the mass, with the J/psi on the same relative
    axis and the 1e-5 target marked.
  * `ks_outwindow`  -- the fraction radiating out of the window, K_S and
    J/psi.
  * `ks_cf`         -- ``|phi_K(t)|`` and ``arg phi_K(t)`` over the ``t`` a
    `MassCFTerm` reads at a 2-6 MeV K_S mass resolution.
  * `ks_angular`    -- the fixed-z angular distribution, the closed form (S3)
    against ``sum|M|^2`` evaluated from explicit four-vectors, ratio panel.

usage:  python3 plot_ks_fsr.py [-o OUTDIR]
"""
import argparse
import datetime
import math
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
import fsr_analytic as fa          # noqa: E402
import ks_fsr_kernel as ks         # noqa: E402
import pubhtml                     # noqa: E402

MJPSI = 3.0969
C_EX, C_EIK, C_FO, C_JP = "#d62728", "#7f7f7f", "#1f77b4", "#2ca02c"


def _ratio_fig(h=(3, 1)):
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(2, 1, height_ratios=h, hspace=0.06)
    a0 = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    a0.tick_params(labelbottom=False)
    return fig, a0, a1


def _cells(k, n_fine=20000):
    c = k.cells(n_fine=n_fine)
    w, m1 = c[0], c[1]
    g = w > 0
    ub = m1[g] / w[g]
    o = np.argsort(ub)
    return ub[o], (w[g] / w[g].sum())[o]


# --------------------------------------------------------------------------
def fig_egamma(k, out):
    """dGamma/dE*_gamma: exact scalar QED against the pure eikonal."""
    egm = k.egamma(k.r)
    eg = np.geomspace(1e-4, egm * (1 - 1e-6), 800)
    ex = k.r1_egamma(eg)
    eik = k.beta / eg
    fig, a0, a1 = _ratio_fig()
    a0.plot(eg * 1e3, ex, color=C_EX, lw=2.2,
            label=r"exact scalar QED, eq. (S4)")
    a0.plot(eg * 1e3, eik, color=C_EIK, lw=1.8, ls="--",
            label=r"eikonal $b/E^*_\gamma$, $b=(\alpha/\pi)\,"
                  r"\mathcal{B}(\beta_0)$")
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylabel(r"$\Gamma_0^{-1}\,\mathrm{d}\Gamma/\mathrm{d}E^*_\gamma$"
                  r"  [GeV$^{-1}$]")
    a0.set_ylim(1e-5, 3e2)
    a0.legend(fontsize=13, loc="lower left")
    a0.text(0.97, 0.94,
            r"$K_S\to\pi^+\pi^-\gamma$ inner bremsstrahlung" "\n"
            rf"$m_\pi/M={k.mch / k.m:.3f}$, $\beta_0={k.beta0:.4f}$, "
            rf"$\ln(M^2/m_\pi^2)={k.L:.3f}$" "\n"
            rf"$E^{{*}}_{{\gamma,\max}}={egm * 1e3:.1f}$ MeV",
            transform=a0.transAxes, fontsize=12, ha="right", va="top")
    a1.plot(eg * 1e3, ex / eik, color=C_EX, lw=2.2)
    a1.axhline(1.0, color="k", lw=1, ls=":")
    for w in ks.WINDOWS:
        e = 0.5 * k.m * (1.0 - (1.0 - w / k.m) ** 2)
        for a in (a0, a1):
            a.axvline(e * 1e3, color="#1f77b4", lw=1, ls="-.", alpha=0.7)
        a1.text(e * 1e3 * 0.93, 0.06, rf"$\pm{w * 1e3:.0f}$ MeV", rotation=90,
                fontsize=10, color="#1f77b4", ha="right", va="bottom")
    for e, lab in ((0.020, "PDG 20"), (0.050, "PDG 50")):
        for a in (a0, a1):
            a.axvline(e * 1e3, color="k", lw=1, ls=":", alpha=0.7)
        a1.text(e * 1e3 * 1.07, 0.06, lab, rotation=90, fontsize=10,
                ha="left", va="bottom")
    a1.set_ylim(0.0, 1.15)
    a1.set_xlabel(r"$E^*_\gamma$  [MeV]")
    a1.set_ylabel("exact / eikonal")
    pubhtml.savefig(fig, os.path.join(out, "ks_egamma.pdf"))
    plt.close(fig)


def fig_rate_gate(k, out):
    """The integrated radiative fraction against the PDG measurements."""
    cuts = np.geomspace(2e-3, 0.16, 90)
    rate = np.array([k.rate_above(c) for c in cuts])
    fig, a0, a1 = _ratio_fig()
    a0.plot(cuts * 1e3, rate, color=C_EX, lw=2.2,
            label=r"$\int_{E^*_\gamma>\mathrm{cut}}R_1\,\mathrm{d}z$, "
                  r"exact scalar QED")
    pts = ((0.050, ks.PDG_IB_50, "PDG avg (RAMBERG 93 + TAUREG 76)"),
           (0.020, ks.PDG_IB_20, "RAMBERG 93 E731"))
    for (c, (v, e), lab), mk in zip(pts, ("o", "s")):
        a0.errorbar([c * 1e3], [v], yerr=[e], fmt=mk, ms=9, color="k",
                    capsize=5, label=lab, zorder=5)
        th = k.rate_above(c)
        a1.errorbar([c * 1e3], [th / v], yerr=[th * e / v ** 2], fmt=mk,
                    ms=9, color="k", capsize=5, zorder=5)
        a1.text(c * 1e3 * 1.10, th / v, f"{th / v:.4f}", fontsize=12,
                va="center")
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylabel(r"$\Gamma(\pi^+\pi^-\gamma)/\Gamma(\pi^+\pi^-)$")
    a0.legend(fontsize=12, loc="lower left")
    a0.text(0.97, 0.94, "NORMALISATION GATE\n"
            "no free parameter: the weak vertex cancels in the ratio",
            transform=a0.transAxes, fontsize=12, ha="right", va="top")
    a1.plot(cuts * 1e3, np.ones_like(cuts), color="k", lw=1, ls=":")
    a1.set_ylim(0.90, 1.10)
    a1.set_xlabel(r"photon-energy cut $E^*_\gamma$  [MeV]")
    a1.set_ylabel("th. / meas.")
    pubhtml.savefig(fig, os.path.join(out, "ks_rate_gate.pdf"))
    plt.close(fig)


def fig_kernel_u(k, out):
    """u dP/du: the exponentiated kernel against the fixed-order spectrum.

    Both are densities in ``u``, ``K_u = 2 z K(z)``, and NEITHER is rescaled:
    the fixed-order curve is the bare ``R_1`` (whose soft end is a
    non-integrable ``b/u`` accompanied by a ``delta(1-z)`` that carries no
    ``u``), the exponentiated one is the normalised kernel.  Their ratio is
    therefore the Sudakov form factor ``C x^b`` plus the hard remainder -- the
    entire content of the exponentiation, and a 9 % effect at ``u = 1e-6``
    falling to nothing by ``u ~ 1e-2``.
    """
    u = np.geomspace(1e-7, k._u_max() * (1 - 1e-9), 900)
    z = np.exp(-2.0 * u)
    x = -np.expm1(-2.0 * u)
    y = u * k.pdf_u(u)
    y2 = u * 2.0 * z * k.r1(z, x)
    fig, a0, a1 = _ratio_fig()
    a0.plot(u, y, color=C_EX, lw=2.2,
            label=r"exponentiated, $K(z)=Cb\,x^{b-1}+h(z)$")
    a0.plot(u, y2, color=C_FO, lw=1.8, ls="--",
            label=r"fixed order $O(\alpha)$, $R_1(z)$ (bare)")
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylim(1e-5, 3e-2)
    a0.set_ylabel(r"$u\,K_u(u)$")
    a0.legend(fontsize=13, loc="lower left")
    a0.text(0.97, 0.94,
            rf"$b={k.beta:.4e}$" "\n"
            rf"$\langle u\rangle={k.moments()['u']:.4e}$" "\n"
            r"$u=-\ln(m_{\pi\pi}/M_{K_S})$",
            transform=a0.transAxes, fontsize=12, ha="right", va="top")
    with np.errstate(divide="ignore", invalid="ignore"):
        a1.plot(u, y / y2, color=C_EX, lw=2.2)
    a1.axhline(1.0, color="k", lw=1, ls=":")
    for w_ in ks.WINDOWS:
        uu = -math.log1p(-w_ / k.m)
        for a in (a0, a1):
            a.axvline(uu, color="#1f77b4", lw=1, ls="-.", alpha=0.7)
    a1.set_ylim(0.88, 1.06)
    a1.set_xlabel(r"$u$")
    a1.set_ylabel(r"exp / $O(\alpha)$")
    a1.set_xlim(u[0], u[-1])
    pubhtml.savefig(fig, os.path.join(out, "ks_kernel_u.pdf"))
    plt.close(fig)


def _window_curves(mass, ub, w, hw):
    dm = mass * np.expm1(-ub)
    o = np.argsort(-dm)                       # dm is negative, increasing |dm|
    dms, ws = dm[o], w[o]
    cw = np.cumsum(ws)
    cm = np.cumsum(ws * dms)
    idx = np.searchsorted(-dms, hw)
    idx = np.clip(idx, 1, len(dms))
    p = cw[idx - 1]
    return p, cm[idx - 1] / p


def fig_meanshift(k, out):
    """<dm | |dm| < w> against the window halfwidth, K_S against J/psi."""
    ub, w = _cells(k)
    kj = fa.FSRKernel(MJPSI, variant="exp2nll", pair=("e", "mu"),
                      mass_exact=True)
    ubj, wj = _cells(kj)
    hw = np.geomspace(2e-4, 0.20, 240)
    _, mk = _window_curves(k.m, ub, w, hw)
    _, mj = _window_curves(MJPSI, ubj, wj, hw)

    fig, a0, a1 = _ratio_fig(h=(1, 1))
    a0.plot(hw * 1e3, -mk * 1e3, color=C_EX, lw=2.2,
            label=r"$K_S\to\pi^+\pi^-$, exact scalar QED")
    a0.plot(hw * 1e3, -mj * 1e3, color=C_JP, lw=2.0, ls="--",
            label=r"$J/\psi\to\mu^+\mu^-$, exp2nll + pairs (mass-exact)")
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.set_ylabel(r"$-\langle\delta m\rangle_w$  [MeV]")
    a0.legend(fontsize=13, loc="upper left")
    a1.plot(hw * 1e3, -mk / k.m, color=C_EX, lw=2.2)
    a1.plot(hw * 1e3, -mj / MJPSI, color=C_JP, lw=2.0, ls="--")
    a1.axhline(1e-5, color="k", lw=1.2, ls=":")
    a1.text(0.3, 1.25e-5, r"$10^{-5}$ momentum-scale target", fontsize=12)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylim(3e-6, 5e-3)
    a1.set_xlabel(r"window halfwidth $w$  [MeV]")
    a1.set_ylabel(r"$-\langle\delta m\rangle_w/M$")
    for w_ in ks.WINDOWS:
        for a in (a0, a1):
            a.axvline(w_ * 1e3, color="#1f77b4", lw=1, ls="-.", alpha=0.7)
    pubhtml.savefig(fig, os.path.join(out, "ks_meanshift.pdf"))
    plt.close(fig)
    return kj, ubj, wj


def fig_outwindow(k, kj, ubj, wj, out):
    """The fraction radiating out of the window."""
    ub, w = _cells(k)
    hw = np.geomspace(2e-4, 0.20, 240)
    pk, _ = _window_curves(k.m, ub, w, hw)
    pj, _ = _window_curves(MJPSI, ubj, wj, hw)
    fig = plt.figure(figsize=(9, 6.5))
    a = fig.add_subplot(111)
    a.plot(hw * 1e3, 1.0 - pk, color=C_EX, lw=2.2,
           label=r"$K_S\to\pi^+\pi^-$")
    a.plot(hw * 1e3, 1.0 - pj, color=C_JP, lw=2.0, ls="--",
           label=r"$J/\psi\to\mu^+\mu^-$")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"window halfwidth $w$  [MeV]")
    a.set_ylabel(r"$P(|\delta m|>w)$")
    a.set_title("radiated out of the window", fontsize=14)
    a.legend(fontsize=13, loc="upper right")
    for w_ in ks.WINDOWS:
        a.axvline(w_ * 1e3, color="#1f77b4", lw=1, ls="-.", alpha=0.7)
    pubhtml.savefig(fig, os.path.join(out, "ks_outwindow.pdf"))
    plt.close(fig)


def fig_cf(npz, out, sigmas=(0.002, 0.004, 0.006)):
    """|phi_K(t)| and arg phi_K(t) over the t a MassCFTerm reads."""
    d = np.load(npz, allow_pickle=True)
    t, re, im = d["phik_t"], d["phik_re"], d["phik_im"]
    fig, a0, a1 = _ratio_fig(h=(1, 1))
    a0.plot(t, np.hypot(re, im), color=C_EX, lw=2.2)
    a0.set_ylabel(r"$|\varphi_K(t)|$")
    a0.set_ylim(0.955, 1.001)
    a1.plot(t, np.arctan2(im, re) * 1e3, color=C_EX, lw=2.2)
    a1.set_ylabel(r"$\arg\varphi_K(t)$  [mrad]")
    a1.set_xlabel(r"$t$  [GeV$^{-1}$]")
    a1.set_xlim(0, t[-1])
    for s in sigmas:
        tm = 7.8926 / s
        if tm > t[-1]:
            continue
        for a in (a0, a1):
            a.axvline(tm, color="k", lw=1, ls=":", alpha=0.7)
        a1.text(tm * 0.99, -0.3, rf"$\sigma_m={s * 1e3:.0f}$ MeV",
                rotation=90, fontsize=11, ha="right", va="top")
    a0.text(0.97, 0.94, r"$\varphi_K(t)=\langle e^{i t\,\delta m}\rangle$, "
            r"what $\mathtt{MassCFTerm}$ multiplies into" "\n"
            r"the resolution CF; the marks are $\max(\tau)/\sigma_m$",
            transform=a0.transAxes, fontsize=12, ha="right", va="top")
    pubhtml.savefig(fig, os.path.join(out, "ks_cf.pdf"))
    plt.close(fig)


def fig_angular(k, out, zs=(0.95, 0.70, 0.40)):
    """The fixed-z angular distribution: (S3) against four-vector |M|^2."""
    fig, a0, a1 = _ratio_fig()
    cols = ("#d62728", "#1f77b4", "#2ca02c")
    M, m = k.m, k.mch
    for z, c in zip(zs, cols):
        s12 = z * M * M
        E = 0.5 * math.sqrt(s12)
        w = (M * M - s12) / (2.0 * math.sqrt(s12))
        q = E * math.sqrt(1.0 - 4.0 * m * m / s12)
        cos = np.linspace(-0.999, 0.999, 601)
        s23 = m * m + 2.0 * w * (E - q * cos)
        pp, pm, kk, _ = ks._dalitz_vectors(M, m, z, s23)
        t_vec = ks.sum_m2_over_g2(pp, pm, kk, m)
        b = q / E
        t_cf = (4.0 * b * b * (1.0 - cos ** 2)
                / (w * w * (1.0 - b * b * cos ** 2) ** 2))
        a0.plot(cos, t_cf, color=c, lw=2.2,
                label=rf"$z={z:.2f}$, $\beta={b:.3f}$ (closed form S3)")
        a0.plot(cos[::12], t_vec[::12], "o", ms=5, mfc="none", color=c,
                label=(r"$-J^2$ from explicit four-vectors"
                       if z == zs[0] else None))
        a1.plot(cos, t_vec / t_cf - 1.0, color=c, lw=1.6)
    a0.set_yscale("log")
    a0.set_ylabel(r"$\sum_{\rm pol}|\mathcal{M}|^2/(e^2|G|^2)$  [GeV$^{-2}$]")
    a0.legend(fontsize=12, ncol=2, loc="lower center")
    a0.text(0.5, 0.95, r"$K_S\to\pi^+\pi^-\gamma$, fixed $z=(m_{\pi\pi}/M)^2$"
            "\n" r"dipole pattern $\propto\sin^2\theta^*/"
            r"(1-\beta^2\cos^2\theta^*)^2$",
            transform=a0.transAxes, fontsize=12, ha="center", va="top")
    a1.axhline(0.0, color="k", lw=1, ls=":")
    a1.set_ylim(-2e-11, 2e-11)
    a1.set_xlabel(r"$\cos\theta^*$  ($\pi^-$ vs $\gamma$, $\pi\pi$ rest frame)")
    a1.set_ylabel("rel. diff.")
    pubhtml.savefig(fig, os.path.join(out, "ks_angular.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--kernel", default=os.path.join(_HERE, "data",
                                                     "ks_kern_data.npz"))
    ap.add_argument("--tag", default="ks_fsr")
    a = ap.parse_args(argv)
    day = datetime.date.today().strftime("%y%m%d")
    out = a.outdir or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{day}_{a.tag}")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out, logger=logger)
    logger.info(f"figures -> {out}")

    k = fa.ScalarIBKernel(ks.MKS)
    fig_egamma(k, out)
    fig_rate_gate(k, out)
    fig_kernel_u(k, out)
    kj, ubj, wj = fig_meanshift(k, out)
    fig_outwindow(k, kj, ubj, wj, out)
    fig_angular(k, out)
    if os.path.exists(a.kernel):
        fig_cf(a.kernel, out)
    else:
        logger.warning(f"{a.kernel} missing; skipping ks_cf "
                       "(build it with `ks_fsr_kernel.py analytic`)")
    logger.info("done")


if __name__ == "__main__":
    main()
