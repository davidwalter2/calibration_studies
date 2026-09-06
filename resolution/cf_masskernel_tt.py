"""FSR kernel from a TwoTrack (ditrack) CVH production.

`cf_mass_likelihood.py --kernel` reconstructs the post-FSR gen dimuon mass
by pairing the SINGLE-track tree's `genParms`/`genCharge` branches by
(run, lumi, event).  The TwoTrack maker
(`ResidualGlobalCorrectionMakerTwoTrackG4e`) writes a per-CANDIDATE tree
that has neither branch -- it exposes the already-paired gen mass as
`Jpsigen_mass` instead -- so `--kernel` dies with

    uproot.exceptions.KeyInFileError: not found: 'genCharge'

on any ditrack production.  This script fills that gap: it builds the same
kernel cache (identical npz schema: dm/hist/ctr/frac_tail, same 700 bins
over [-0.35, +0.005] GeV, density normalized) and the same log-scale plot
directly from `Jpsigen_mass`, so the kernel and the `--pairs-tt` candidate
cache come from the SAME production, as the likelihood requires.

`Jpsigen_mass` is the gen-matched (dR < 0.1, same charge, status-1) muon
pair mass of the candidate -- i.e. exactly the quantity
`pair_mass(genParms)` computes on the single-track side, but with the
maker's own charge-ordered matching instead of an offline pairing loop, so
no vertex tolerance is needed to reject mispaired combinations.
Unmatched candidates carry the sentinel -99 and fall outside the window.

usage:
  python cf_masskernel_tt.py --files '<prod>/task_*/globalcor_*.root' --ntasks 48 \
      --kernel-cache runs/cf_masskernel_<tag>.npz --postfix _<tag>
"""

import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools
import prodfiles

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

# Same pole as cf_mass_likelihood.py (kept in sync deliberately: the demo
# scan subtracts MJPSI from this kernel's dm and from the candidate cache).
MJPSI = 3.0969


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True,
                   help="glob of TwoTrack globalcor_*.root files")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--kernel-cache", required=True)
    p.add_argument("--window", type=float, default=0.35,
                   help="|m_gen - M_Jpsi| window [GeV] (build_pairs_tt uses 0.35)")
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_masslik/")
    os.makedirs(outdir, exist_ok=True)

    files = prodfiles.resolve(args.files, args.ntasks, logger=logger.info)
    logger.info(f"{len(files)} files (TwoTrack per-candidate trees)")
    masses, ntot, nunmatched = [], 0, 0
    for fn in files:
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            mg = f["tree"]["Jpsigen_mass"].array(library="np").astype(np.float64)
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        ntot += len(mg)
        nunmatched += int(np.sum(mg < 0.))
        masses.append(mg[np.abs(mg - MJPSI) < args.window])
        logger.info(f"{fn.split('/')[-2]}: {len(mg)} candidates, "
                    f"cumulative {sum(len(x) for x in masses)} in window")
    m = np.concatenate(masses) if masses else np.zeros(0)
    logger.info(f"total {ntot} candidates, {nunmatched} with no gen match, "
                f"{len(m)} in the J/psi window")

    dm = m - MJPSI
    hist, edges = np.histogram(dm, bins=700, range=(-0.35, 0.005), density=True)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    frac_tail = float((dm < -0.005).mean())
    logger.info(f"FSR tail fraction (dm < -5 MeV): {frac_tail:.4f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.kernel_cache)), exist_ok=True)
    np.savez_compressed(args.kernel_cache, dm=dm, hist=hist, ctr=ctr,
                        frac_tail=frac_tail)
    logger.info(f"wrote {args.kernel_cache}")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(1e3 * dm, bins=280, range=(-350, 5), density=True, histtype="step",
            color="black")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m^{\mathrm{gen}}_{\mu\mu} - m_{J/\psi}$ [MeV]")
    ax.set_ylabel("density [1/GeV]")
    ax.text(0.05, 0.95, f"FSR tail (< $-5$ MeV): {100*frac_tail:.1f}%",
            transform=ax.transAxes, va="top")
    name = f"fsr_kernel_jpsi{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"wrote {outdir}/{name}")


if __name__ == "__main__":
    main()
