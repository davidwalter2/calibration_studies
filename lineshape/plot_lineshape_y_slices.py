"""Mass lineshape (10-120 GeV) compared between FO Z/gamma* (CT18Z) and MiNNLO
in slices of |Y_Z|. Each curve is normalised independently within its (mass, |Y|)
window so the comparison is purely shape.

DYTurbo is intentionally omitted here: the available DYTurbo outputs do not
have (m, |Y|)-binned data over the full 10-120 GeV mass range. Once a DYTurbo
file with matching binning exists, add it with the same resampling pattern as
in `plot_lineshape.py`.
"""

import argparse
import datetime
import os

import h5py
import hist
import lhapdf
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.integrate import quad

from wums import boostHistHelpers as hh
from wums import plot_tools

import constants
import drell_yan_xsec as dy
import ioutils


def parse_args():
    p = argparse.ArgumentParser(
        description="FO Z/gamma* (CT18Z) vs MiNNLO mass lineshape in |Y| slices"
    )
    p.add_argument(
        "-i", "--input", dest="filename_minnlo",
        default="/ceph/submit/data/user/d/david_w/WMassAnalysis/260520_z_gen/w_z_gen_dists.hdf5",
        help="MiNNLO histmaker hdf5 (default: %(default)s)",
    )
    p.add_argument(
        "-o", "--outdir", default=None,
        help="Output directory "
             "(default: ~/public_html/WMass/<YYMMDD>_lineshape_yslices/)",
    )
    p.add_argument(
        "--pdf", default="CT18ZNNLO",
        help="LHAPDF set for the FO calculation (default: %(default)s)",
    )
    p.add_argument(
        "--minnlo-hist", default="nominal_gen_pdfCT18Z",
        help="MiNNLO output key to project (default: %(default)s, "
             "CT18Z-reweighted central; use 'nominal_gen' for the bare sample)",
    )
    p.add_argument(
        "--minnlo-pdf-member", type=int, default=0,
        help="pdfVar index for nominal_gen_pdf<set>: 0 = central (default)",
    )
    p.add_argument(
        "--pt-min", type=float, default=1.0,
        help="pT(Z) cut applied to MiNNLO (GeV); default 1.0 to match DYTurbo qt_bins",
    )
    p.add_argument(
        "--y-edges", type=float, nargs="+",
        default=[0.0, 1.1, 2.0, 3.0, 5.0],
        help="|Y| slice edges (must lie on the absYVgen binning of the MiNNLO file). "
             "Default 0 1.1 2 3 5 -- closest match to 0-1, 1-2, 2-3, 3-5 "
             "given the MiNNLO bin edges (no edge at 1.0)",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# MiNNLO: project nominal_gen[_pdfXYZ] into a 1D mass hist within a |Y| slice
# with a pT(Z) > pt_min cut, summed over chargeVgen.
# -----------------------------------------------------------------------------

def _minnlo_yslice_one_sample(result, sample, histname, pdf_member, pt_min, y_lo, y_hi):
    h = result[sample]["output"][histname].get()
    axes = [a.name for a in h.axes]

    if "pdfVar" in axes:
        members = list(h.axes["pdfVar"])
        h = h[{"pdfVar": members[pdf_member]}]

    # pT(Z) > pt_min, |Y| in [y_lo, y_hi), sum over chargeVgen (single bin here)
    sel = {
        "ptVgen": slice(complex(0, pt_min), None, hist.sum),
        "absYVgen": slice(complex(0, y_lo), complex(0, y_hi), hist.sum),
        "chargeVgen": hist.sum,
    }
    h = h[sel]
    # Now h has only the massVgen axis left.

    weight_sum = result[sample]["weight_sum"]
    xsec = result[sample]["dataset"]["xsec"]
    h = hh.scaleHist(h, xsec / weight_sum)

    if h.axes[0].name != "mass":
        # Reuse ioutils' rename helper to standardise the axis name to "mass".
        h = ioutils._rename_axis_to_mass(h)
    return h


def minnlo_mass_in_yslice(result, histname, pdf_member, pt_min, y_lo, y_hi):
    z_high = _minnlo_yslice_one_sample(
        result, "Zmumu_2016PostVFP",        histname, pdf_member, pt_min, y_lo, y_hi
    )
    z_low = _minnlo_yslice_one_sample(
        result, "Zmumu10to50_2016PostVFP",  histname, pdf_member, pt_min, y_lo, y_hi
    )
    return hh.addHists(z_high, z_low)


# -----------------------------------------------------------------------------
# FO: integrate d^2 sigma / (dQ dY) over |Y| in [y_lo, y_hi] for each mass bin.
#
# Symmetric in Y for pp, so the contribution from -Y equals the contribution
# from +Y -- integrate over [y_lo, y_hi] and multiply by 2.
# Each Y range is clipped to the kinematic boundary Y_max = 0.5*log(s/Q^2).
# -----------------------------------------------------------------------------

def fo_dsigma_dQ_in_yslice(Q2_arr, y_lo, y_hi, quark_couplings, pdf, settings):
    out = np.zeros_like(Q2_arr, dtype=float)
    for i, Q2 in enumerate(Q2_arr):
        Y_max = 0.5 * np.log(settings["s"] / Q2)
        lo = min(y_lo, Y_max)
        hi = min(y_hi, Y_max)
        if hi <= lo:
            continue
        val, _ = quad(
            lambda y: dy.dsigma_dQ_dY(
                float(Q2), y, quark_couplings, pdf=pdf, **settings
            ),
            lo, hi, epsrel=1e-4,
        )
        out[i] = 2.0 * val  # combine negative-Y branch by symmetry
    return out


# -----------------------------------------------------------------------------
# Plot a single |Y| slice (one figure with a main+ratio panel)
# -----------------------------------------------------------------------------

def plot_one_yslice(h_minnlo, h_fo, y_lo, y_hi, outdir):
    # Normalise each curve within the (10-120 GeV, |Y|-slice) window.
    # h_minnlo and h_fo already share the same mass axis [10, 120].
    h_minnlo_n = hh.normalize(h_minnlo, scale=1)
    h_fo_n = hh.normalize(h_fo, scale=1)

    hists = [h_minnlo_n, h_fo_n]
    labels = ["MiNNLO (pre-FSR)", r"FO Z/$\gamma*$"]
    colors = ["black", "blue"]
    linestyles = ["-", "--"]

    m = h_minnlo.axes["mass"].centers

    # y range: pick a generous ylim from the data
    peak = max(h_minnlo_n.values().max(), h_fo_n.values().max())
    ylim = [0.0, peak * 1.15]

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h_minnlo_n,
        "mass in GeV",
        "Frequency",
        ylim,
        "FO / MiNNLO",
        [0.85, 1.15],
        xlim=(h_minnlo.axes["mass"].edges[0], h_minnlo.axes["mass"].edges[-1]),
        width_scale=1,
    )
    ax2 = ratio_axes[-1]

    hep.histplot(
        hists,
        histtype="step",
        color=colors,
        label=labels,
        linestyle=linestyles,
        yerr=False,
        ax=ax1,
        linewidth=2,
        flow="none",
    )
    ax1.axvline(constants.mass_z, color="grey", linestyle="--")
    ax1.legend(ncol=1, title=fr"$|Y_Z| \in [{y_lo:g}, {y_hi:g}]$")

    hep.histplot(
        [hh.divideHists(h_fo_n, h_minnlo_n)],
        histtype="step",
        color=[colors[1]],
        linestyle=[linestyles[1]],
        yerr=False,
        ax=ax2,
        linewidth=2,
        flow="none",
    )
    ax2.axvline(constants.mass_z, color="grey", linestyle="--")
    ax2.axhline(1, color="grey", linestyle="--")

    plot_tools.fix_axes(ax1, ax2, fig)

    label = f"yslice_{y_lo:g}to{y_hi:g}".replace(".", "p")
    outfile = "_".join(["zgamma_prefsr_lineshape", label])
    plot_tools.save_pdf_and_png(outdir, outfile)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.outdir is None:
        today = datetime.date.today().strftime("%y%m%d")
        outdir = os.path.expanduser(f"~/public_html/WMass/{today}_lineshape_yslices/")
    else:
        outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # MiNNLO -----------------------------------------------------------------
    h5file = h5py.File(args.filename_minnlo, "r")
    result = ioutils.load_results_h5py(h5file)

    # FO setup ---------------------------------------------------------------
    pdf = lhapdf.mkPDF(args.pdf, 0)
    quark_couplings = dy.get_quark_couplings(constants.sin2theta_w)
    settings = dict(
        s=constants.s,
        mass_z=constants.mass_z,
        width_z=constants.width_z,
        sin2theta_w=constants.sin2theta_w,
    )

    y_edges = args.y_edges
    for y_lo, y_hi in zip(y_edges[:-1], y_edges[1:]):
        print(f"=== |Y| slice [{y_lo}, {y_hi}] ===")

        h_minnlo = minnlo_mass_in_yslice(
            result, args.minnlo_hist, args.minnlo_pdf_member,
            args.pt_min, y_lo, y_hi,
        )
        # Slice mass to 10-120 GeV (matches existing inclusive plot range).
        h_minnlo = h_minnlo[{"mass": slice(10j, 120j)}]

        # Build the FO hist on the same mass axis as the (sliced) MiNNLO.
        m_centers = h_minnlo.axes["mass"].centers
        m_widths = np.diff(h_minnlo.axes["mass"].edges)
        # dsigma/dQ in the slice, evaluated at bin centers (pb/GeV)
        dsigma_dQ = fo_dsigma_dQ_in_yslice(
            m_centers**2, y_lo, y_hi, quark_couplings, pdf, settings
        )
        # Convert to per-bin content (~dsigma/dQ * dQ). Bin widths are uniform
        # 1 GeV here, so this is essentially the same shape as dsigma/dQ, but
        # we multiply by the bin width to give consistent "events/bin" units
        # before the per-curve normalization.
        h_fo = h_minnlo.copy()
        h_fo.values()[...] = dsigma_dQ * m_widths

        print(f"  MiNNLO integral = {h_minnlo.values().sum():.4e}, "
              f"FO integral = {h_fo.values().sum():.4e}")

        plot_one_yslice(h_minnlo, h_fo, y_lo, y_hi, outdir)


if __name__ == "__main__":
    main()
