"""Slide figure: the RADIATIVE energy-loss distribution, and where it takes
over from the delta-ray tail.

The CF resolution deck has one distribution slide per random process --
Landau for ionization, Moliere for scattering. Bremsstrahlung + pair
production is the third such process and it has been missing, which matters
because the deck's premise ("two random processes smear every track") is only
true below ~10 GeV. At Z-muon momenta the radiative term is a percent-level
effect on the loss and carries a pT-DEPENDENT 1.1e-5 bias.

What the figure shows, all from the 1M-event pT=100 clean-prop sample:
  * the simulated total energy loss, as dN/dln(eps) so a 1/eps spectrum is
    FLAT and a 1/eps^2 spectrum falls as 1/eps -- the two components are then
    distinguishable by eye;
  * the delta-ray (ionization) component, xi/eps^2;
  * the radiative component, from Geant4's own brems + pair differential
    cross sections, each normalized to its own ComputeDEDXPerVolume;
  * their sum.

The crossover near ~5 GeV is the point of the slide: below it the difficulty
is the delta-ray tail, above it radiative straggling that the fluctuation
model did not contain at all until 2026-08-06.

usage:
    python plot_radiative_spectrum.py --sim '<simdir>/simstates_*.root' \\
        --model <model_with_radv.root>
"""

import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools

import cf_brems_exact as cb

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

RED, BLUE, GREY = "#A31F34", "#1f4e9c", "0.45"


def sim_loss(pattern):
    """Total in-tracker energy loss per event [MeV], first to last plane."""
    loss = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        a = uproot.open(f)["simstates/simstates"].arrays(["pabs"], library="np")["pabs"]
        for v in a:
            v = np.asarray(v, dtype=float)
            if len(v) > 1:
                loss.append((v[0] - v[-1]) * 1e3)
    return np.array(loss)


def model_components(legs, vg, fine):
    """(brems, pair, delta-ray) dN/deps per track on the `fine` eps grid [MeV].

    brems and pair are kept SEPARATE because their shapes differ strongly:
    brems is ~1/eps (few, hard transfers) while pair production is much
    steeper (many soft transfers). Pair carries 58% of the radiative MEAN
    here despite being soft, which is exactly why normalizing a single
    combined shape gets the mixture wrong.
    """
    brems = np.zeros_like(fine)
    pair = np.zeros_like(fine)
    delta = np.zeros_like(fine)
    for recs, spec in legs:
        for rec, sp in zip(recs, spec):
            # radiative: Geant4 shapes, each normalized to its own dE/dx
            E = rec[cb.R_ETOT] * 1e3
            L = rec[cb.R_STEPCM]
            for shape, dedx, acc in ((sp[:cb.NRADV], rec[cb.R_DEDXBREM], brems),
                                     (sp[cb.NRADV:], rec[cb.R_DEDXPAIR], pair)):
                dE = dedx * L
                norm = np.trapezoid(shape * vg * rec[cb.R_ETOT], vg)
                if dE <= 0. or norm <= 0.:
                    continue
                dNdv = shape * (dE / norm)
                eps, y = vg * E, dNdv / E
                ok = y > 0.
                if ok.sum() > 1:
                    acc += np.exp(np.interp(np.log(fine), np.log(eps[ok]),
                                            np.log(y[ok]), left=-np.inf, right=-np.inf))
            # delta rays: Rutherford tail xi/eps^2 up to the kinematic limit
            beta2 = (rec[cb.R_P] / rec[cb.R_ETOT]) ** 2
            xi = 0.1535 * (rec[cb.R_EFFZ] / rec[cb.R_EFFA]) * rec[cb.R_XG] / max(beta2, 1e-9)
            tmax = 1e3 * 2 * cb.M_E * (rec[cb.R_P] / cb.M_MU) ** 2 / (
                1. + 2. * (rec[cb.R_ETOT] / cb.M_MU) * cb.M_E / cb.M_MU + (cb.M_E / cb.M_MU) ** 2)
            delta += np.where(fine < tmax, xi / fine ** 2, 0.)
    return brems, pair, delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--outpath", default=None)
    p.add_argument("--emin", type=float, default=100.)
    p.add_argument("--emax", type=float, default=40000.)
    args = p.parse_args()
    logging.setup_logger(__file__, 3, False)

    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(
        args.outpath or os.path.expanduser(
            f"~/public_html/calibration_studies/{today}_radiative_spectrum/"))

    loss = sim_loss(args.sim)
    legs, vg = cb.load_radv(args.model)
    logger.info(f"{len(loss)} sim events, {sum(len(r) for r, _ in legs)} model steps")

    ed = np.geomspace(args.emin, args.emax, 26)
    ctr = np.sqrt(ed[1:] * ed[:-1])
    h, _ = np.histogram(loss, bins=ed)
    # dN/dln(eps) per track: flat for 1/eps, ~1/eps for 1/eps^2
    dndlne = h / np.diff(np.log(ed)) / len(loss)
    err = np.sqrt(h) / np.diff(np.log(ed)) / len(loss)

    fine = np.geomspace(args.emin, args.emax, 2000)
    brems, pair, delta = model_components(legs, vg, fine)
    rad = brems + pair

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.errorbar(ctr, dndlne, yerr=err, fmt="o", color="k", markersize=6,
                label="simulation (Geant4)", zorder=5)
    ax.plot(fine, delta * fine, color=BLUE, lw=2.2, ls="--",
            label=r"ionization $\delta$-rays  ($\propto 1/\varepsilon^2$)")
    ax.plot(fine, brems * fine, color=RED, lw=2.2, ls="-.",
            label=r"bremsstrahlung  (hard, $\sim 1/\varepsilon$)")
    ax.plot(fine, pair * fine, color="#e07b39", lw=2.2, ls=":",
            label=r"pair production  (soft, steeper)")
    ax.plot(fine, (rad + delta) * fine, color=GREY, lw=2.6, label="sum")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"energy transfer $\varepsilon$ [MeV]")
    ax.set_ylabel(r"$dN/d\ln\varepsilon$ per track")
    ax.set_ylim(max(dndlne[dndlne > 0].min() * 0.3, 1e-7), dndlne.max() * 6)
    ax.legend(fontsize=15, loc="upper right")
    ax.grid(alpha=0.25)
    fig.text(0.13, 0.955, "CMS", fontsize=24, fontweight="bold", va="top")
    fig.text(0.235, 0.95, "Simulation Work in progress", fontsize=17,
             style="italic", va="top")
    fig.text(0.90, 0.95, r"$\mu^-$, $p_T = 100$ GeV", fontsize=16, va="top", ha="right")

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"radiative_spectrum.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {outdir}/radiative_spectrum.png")


if __name__ == "__main__":
    main()
