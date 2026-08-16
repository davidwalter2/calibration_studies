"""FSR kernel for the CF mass likelihood: truth matching + analytic LO check.

Two problems with the kernel as built by cf_mass_likelihood.build_kernel:

  1. COMBINATORICS. collect_pairs takes every opposite-charge gen-matched
     combination in an event, and the samples have up to 12 gen-matched
     muons per event. Mispaired combinations populate m_gen ABOVE the
     J/psi pole (dm up to +295 MeV), which FSR cannot produce. build_kernel
     histograms only up to +5 MeV so the plot hides this, but --demo builds
     phi_K from the raw dm array, which does not.

     FIX: require the two muons to share a gen production vertex
     (genX/Y/Z). Muons from one J/psi decay are generated at exactly the
     same point; the separation distribution is sharply bimodal (0 vs
     cm-scale), so the cut is unambiguous.

  2. Is the resulting kernel consistent with QED? Compare against the LO
     Altarelli-Parisi radiator (lineshape.functions.radiator_kernel). For a
     resonance of mass M decaying to mu+mu-, with z = m'^2/M^2 the retained
     energy fraction,

         p(m') = K(z, M) * dz/dm' = K(m'^2/M^2, M) * 2m'/M^2

     NOTE on the coupling: radiator_kernel defaults to constants.alpha_ew,
     the running alpha at the W scale (~1/128.83). Real photon emission off
     a muon is a q^2 -> 0 process, so alpha(0) = 1/137.036 is the correct
     choice here; the default overestimates beta by ~6%.

usage:
  python fsr_kernel_study.py                      # full study + plots
  python fsr_kernel_study.py --write-kernel OUT   # also save a clean kernel npz
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lineshape"))
from functions import radiator_kernel  # noqa: E402

from wums import logging, output_tools, plot_tools  # noqa: E402
from cf_mass_likelihood import MJPSI, pair_mass  # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

ALPHA0 = 1. / 137.035999
MMU = 0.1056583755


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_260802/task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--vtx-tol", type=float, default=1e-3,
                   help="max gen production-vertex separation [cm] for a true pair")
    p.add_argument("--window", type=float, default=0.35)
    p.add_argument("--write-kernel", default=None,
                   help="path to save the truth-matched kernel npz")
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def collect(args):
    """Return (dm_all, dm_true) for gen dimuon pairs in the J/psi window."""
    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    dm_all, dm_true, mult = [], [], []
    for fn in files:
        try:
            t = uproot.open(fn)["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        a = t.arrays(["run", "lumi", "event", "genParms", "genCharge",
                      "genX", "genY", "genZ"], library="np")
        gp = np.stack(a["genParms"])
        ok = np.abs(gp[:, 0]) > 0.
        ev = ((a["run"].astype(np.int64) << 40)
              ^ (a["lumi"].astype(np.int64) << 24) ^ a["event"].astype(np.int64))
        order = np.argsort(ev, kind="stable")
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and ev[order[j]] == ev[order[i]]:
                j += 1
            idx = [k for k in order[i:j] if ok[k]]
            mult.append(len(idx))
            for ia in range(len(idx)):
                for ib in range(ia + 1, len(idx)):
                    k1, k2 = idx[ia], idx[ib]
                    if a["genCharge"][k1] * a["genCharge"][k2] >= 0:
                        continue
                    m = pair_mass(gp[k1][None, :], gp[k2][None, :])[0]
                    if abs(m - MJPSI) > args.window:
                        continue
                    d = np.sqrt((a["genX"][k1] - a["genX"][k2]) ** 2
                                + (a["genY"][k1] - a["genY"][k2]) ** 2
                                + (a["genZ"][k1] - a["genZ"][k2]) ** 2)
                    dm_all.append(m - MJPSI)
                    if d < args.vtx_tol:
                        dm_true.append(m - MJPSI)
            i = j
    mult = np.array(mult)
    logger.info("gen-matched muons/event: "
                + str({int(k): int(v) for k, v in zip(*np.unique(mult, return_counts=True))}))
    return np.array(dm_all), np.array(dm_true)


def lo_density(dm, M=MJPSI, alpha=ALPHA0, nrad=1, mrad=MMU):
    """LO AP radiator as a density in dm = m' - M (support dm < 0)."""
    m = M + dm
    z = np.clip(m ** 2 / M ** 2, 1e-12, 1. - 1e-15)
    L = np.log(M ** 2 / mrad ** 2)
    beta = nrad * (2 * alpha / np.pi) * (L - 1)
    K = beta * (1 - z) ** (beta - 1) * (1 + z ** 2) / 2
    return np.where(dm < 0., K * 2 * m / M ** 2, 0.)


def lo_tail(x, M=MJPSI, alpha=ALPHA0, nrad=1):
    """P(m' < M - x) for the LO radiator, by direct integration in z."""
    out = np.empty(len(x))
    for i, xx in enumerate(x):
        zmax = (M - xx) ** 2 / M ** 2
        zz = np.linspace(1e-9, zmax, 20000)
        L = np.log(M ** 2 / MMU ** 2)
        beta = nrad * (2 * alpha / np.pi) * (L - 1)
        K = beta * (1 - zz) ** (beta - 1) * (1 + zz ** 2) / 2
        out[i] = np.trapezoid(K, zz)
    return out


def me_spectrum(xg, M=MJPSI, mrad=MMU):
    """dGamma/dx_gamma / [Gamma0 * alpha/2pi] for V -> l+ l- gamma.

    Exact tree-level massless ME (x1^2+x2^2)/((1-x1)(1-x2)) -- the QED twin of
    e+e- -> qqbar g -- integrated over the EXACT MASSIVE Dalitz boundary. With
    u = 1-x1 the integrand is symmetric under u <-> xg-u, and

        u_pm = xg(1 +- b)/2,   b = sqrt(1 - 4mu/(1-xg)),   mu = m^2/M^2

    so the integral closes analytically:

        I(xg) = (2/xg) [ (2 - 2xg + xg^2) ln((1+b)/(1-b)) - xg^2 b ]

    Soft limit: I -> 4L/xg with L = ln(M^2/m^2). The exact soft coefficient is
    (L-1), so this massless-ME/massive-PS hybrid is high by L/(L-1) = 1.17;
    callers may rescale to match the known soft limit.
    """
    mu = mrad ** 2 / M ** 2
    b = np.sqrt(np.clip(1. - 4 * mu / (1. - xg), 0., None))
    A = 2 - 2 * xg + xg ** 2
    return (2. / xg) * (A * np.log((1 + b) / (1 - b)) - xg ** 2 * b)


def me_density(dm, M=MJPSI, alpha=ALPHA0, mrad=MMU, soft_match=True):
    """LO V->llg matrix element as a density in dm = m' - M."""
    m = M + dm
    xg = 1. - m ** 2 / M ** 2
    f = (alpha / (2 * np.pi)) * me_spectrum(np.clip(xg, 1e-12, None), M, mrad)
    if soft_match:
        L = np.log(M ** 2 / mrad ** 2)
        f = f * (L - 1) / L
    return np.where(dm < 0., f * 2 * m / M ** 2, 0.)


def me_tail(x, M=MJPSI, alpha=ALPHA0, mrad=MMU, soft_match=True):
    """P(m' < M - x) at O(alpha) from the exact ME."""
    from scipy.integrate import quad
    mu = mrad ** 2 / M ** 2
    xgmax = 1. - 4 * mu
    L = np.log(M ** 2 / mrad ** 2)
    sc = (alpha / (2 * np.pi)) * ((L - 1) / L if soft_match else 1.)
    out = np.empty(len(x))
    for i, xx in enumerate(x):
        xg0 = 1. - (M - xx) ** 2 / M ** 2
        out[i] = sc * quad(me_spectrum, xg0, xgmax, args=(M, mrad), limit=200)[0]
    return out


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_fsrkernel/")
    os.makedirs(outdir, exist_ok=True)

    dm_all, dm_true = collect(args)
    nrej = len(dm_all) - len(dm_true)
    logger.info(f"pairs: {len(dm_all)} all, {len(dm_true)} vertex-matched "
                f"({nrej} rejected, {100.*nrej/max(len(dm_all),1):.2f}%)")
    for lab, d in (("all", dm_all), ("true", dm_true)):
        logger.info(f"  [{lab}] dm range [{d.min():+.4f}, {d.max():+.4f}] GeV, "
                    f"frac(dm>+1MeV)={100*(d>0.001).mean():.3f}%, "
                    f"frac(dm<-5MeV)={100*(d<-0.005).mean():.3f}%")

    L = np.log(MJPSI ** 2 / MMU ** 2)
    for lab, al in (("alpha(0)=1/137.0", ALPHA0), ("alpha_ew=1/128.8", 1/128.83)):
        b = (2 * al / np.pi) * (L - 1)
        logger.info(f"  beta [{lab}] = {b:.5f} (1 radiator), {2*b:.5f} (2 radiators)")

    # --- tail-fraction comparison: the shape test that is insensitive to the
    # integrable z->1 divergence and to gen-level numerical smearing at dm~0
    xs = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    fdata = np.array([(dm_true < -x).mean() for x in xs])
    fme = me_tail(xs)
    logger.info("\n  x [MeV]   P(dm<-x) data   LO nrad=1   LO nrad=2   exact ME   data/LO1  data/ME")
    for nr in (1, 2):
        globals()[f"_f{nr}"] = lo_tail(xs, nrad=nr)
    for i, x in enumerate(xs):
        logger.info("  %7.0f   %13.5f   %9.5f   %9.5f   %8.5f   %8.2f  %7.3f"
                    % (1e3 * x, fdata[i], _f1[i], _f2[i], fme[i],
                       fdata[i] / max(_f1[i], 1e-12), fdata[i] / max(fme[i], 1e-12)))

    if args.write_kernel:
        hist, edges = np.histogram(dm_true, bins=700, range=(-args.window, 0.005),
                                   density=True)
        np.savez_compressed(args.write_kernel, dm=dm_true, hist=hist,
                            ctr=0.5 * (edges[1:] + edges[:-1]),
                            frac_tail=float((dm_true < -0.005).mean()))
        logger.info(f"wrote {args.write_kernel}")

    # --- plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    # ABSOLUTE densities on both sides. NB: np.histogram(..., density=True)
    # would normalise to unit area over the plotted sub-range only (~10% of
    # the sample), inflating the data by ~10x relative to the analytic curves,
    # which are normalised over their full support. Normalise by the FULL
    # sample instead, and renormalise the LO to the same +-window as the data.
    ax = axs[0]
    tb = np.linspace(-0.25, -0.002, 100)
    ctr = 0.5 * (tb[1:] + tb[:-1])
    cnt, _ = np.histogram(dm_true, bins=tb)
    h = cnt / (len(dm_true) * np.diff(tb))
    ehi = np.sqrt(cnt) / (len(dm_true) * np.diff(tb))
    ax.errorbar(ctr, h, yerr=ehi, fmt="o", ms=2.5, color="black",
                label="gen-vertex matched")
    for nr, col, ls in ((1, "crimson", "-"), (2, "royalblue", "--")):
        norm = 1. - lo_tail(np.array([args.window]), nrad=nr)[0]
        ax.plot(ctr, lo_density(ctr, nrad=nr) / norm, color=col, ls=ls,
                label=rf"LO AP struct. fn., {nr} rad., $\alpha(0)$")
    ax.plot(ctr, me_density(ctr), color="seagreen", lw=2,
            label=r"exact $V\to\ell\ell\gamma$ ME, $O(\alpha)$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m^{\mathrm{gen}}_{\mu\mu} - m_{J/\psi}$ [GeV]")
    ax.set_ylabel("density [1/GeV]  (absolute, same normalisation)")
    ax.legend(fontsize="x-small")

    # cumulative tail: no binning or normalisation ambiguity at all
    ax = axs[1]
    xg = np.logspace(np.log10(0.002), np.log10(0.30), 60)
    fd = np.array([(dm_true < -x).mean() for x in xg])
    ax.plot(1e3 * xg, fd, color="black", lw=2, label="gen-vertex matched")
    for nr, col, ls in ((1, "crimson", "-"), (2, "royalblue", "--")):
        P = lo_tail(xg, nrad=nr)
        P0 = lo_tail(np.array([args.window]), nrad=nr)[0]
        ax.plot(1e3 * xg, (P - P0) / (1 - P0), color=col, ls=ls,
                label=f"LO AP struct. fn., {nr} rad.")
    ax.plot(1e3 * xg, me_tail(xg), color="seagreen", lw=2,
            label=r"exact $V\to\ell\ell\gamma$ ME")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$ [MeV]")
    ax.set_ylabel(r"$P(m^{\mathrm{gen}}_{\mu\mu} - m_{J/\psi} < -x)$")
    ax.legend(fontsize="x-small")
    name = f"fsr_kernel_study{args.postfix}"
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"wrote {outdir}/{name}")


if __name__ == "__main__":
    main()
