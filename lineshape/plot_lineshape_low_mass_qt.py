"""Low-mass DYTurbo vs MiNNLO comparison at different qT(Z) cuts.

Diagnoses why DYTurbo drops below MiNNLO at low mass.

Uses the 3D-binned DYTurbo file at
``/work/submit/david_w/DYTurbo/from_valentina/Mass/CT18Z_check.txt``
(axes: m x y x qT, mass bin edges [10,20,30,40,50,60] GeV) and matches the
MiNNLO phase space sample-by-sample with slices on ``ptVgen`` / ``absYVgen``.

Outputs (in ~/public_html/WMass/<YYMMDD>_lineshape_low_mass_qt/):
  - one figure per qT cut: dsigma/dm + DYT/MiNNLO ratio
  - one summary figure: DYT/MiNNLO ratio vs mass, overlaying every qT cut
"""

import argparse
import datetime
import os

import h5py
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools

import constants
import ioutils

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="DYTurbo vs MiNNLO at m in [10, 60] GeV for different qT(Z) cuts"
    )
    p.add_argument(
        "-i", "--input", dest="filename_minnlo",
        default="/ceph/submit/data/user/d/david_w/WMassAnalysis/260520_z_gen/w_z_gen_dists.hdf5",
        help="MiNNLO histmaker hdf5 file (default: %(default)s)",
    )
    p.add_argument(
        "--dyturbo",
        default="/work/submit/david_w/DYTurbo/from_valentina/Mass/CT18Z_check.txt",
        help="DYTurbo 3D file (m,y,qT). Default: %(default)s",
    )
    p.add_argument(
        "--qt-cuts", type=float, nargs="+",
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
        help="Lower qT(Z) cuts to test; each produces one comparison plot.",
    )
    p.add_argument(
        "--y-max", type=float, default=5.0,
        help="|Y(Z)| cut applied to both MiNNLO and DYTurbo (default: %(default)s)",
    )
    p.add_argument(
        "--histname", default="nominal_gen",
        help="MiNNLO output histogram key (default: %(default)s)",
    )
    p.add_argument(
        "-o", "--outdir", default=None,
        help="Output directory (default: ~/public_html/WMass/<YYMMDD>_lineshape_low_mass_qt/)",
    )
    p.add_argument("--title", default="CMS", help="Decor title")
    p.add_argument("--subtitle", default="Work in progress", help="Decor subtitle")
    p.add_argument("--titlePos", type=int, default=0, help="Decor position (mplhep loc)")
    p.add_argument("--postfix", default="", help="Tag appended to output filenames")
    return p.parse_args()


# -----------------------------------------------------------------------------
# DYTurbo 3D reader
# -----------------------------------------------------------------------------

def read_dyturbo_3d(filename, fb_to_pb=True):
    """Read CT18Z_check.txt format (m, y, qT) into a 3D hist.

    Columns: ``q2lo q2hi ylo yhi qTlo qThi PDF0 uncertainty``. Comments start
    with ``#``. The trailing row spanning the full (m, y, qT) range (the
    DYTurbo total integral) is dropped. fb -> pb conversion by default to
    match MiNNLO's pb scale.
    """
    rows = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            mlo, mhi, ylo, yhi, qlo, qhi, val, unc = (float(x) for x in parts[:8])
            rows.append((mlo, mhi, ylo, yhi, qlo, qhi, val, unc))
    if not rows:
        raise ValueError(f"No data rows in {filename}")

    m_edges = sorted({r[0] for r in rows} | {r[1] for r in rows})
    y_edges = sorted({r[2] for r in rows} | {r[3] for r in rows})
    q_edges = sorted({r[4] for r in rows} | {r[5] for r in rows})

    # Drop the "total" row: it spans the full range on every axis.
    m_full = (m_edges[0], m_edges[-1])
    y_full = (y_edges[0], y_edges[-1])
    q_full = (q_edges[0], q_edges[-1])
    bins = [r for r in rows if not (
        (r[0], r[1]) == m_full and (r[2], r[3]) == y_full and (r[4], r[5]) == q_full
    )]

    scale = 1e-3 if fb_to_pb else 1.0
    ax_m = hist.axis.Variable(m_edges, name="mass")
    ax_y = hist.axis.Variable(y_edges, name="y")
    ax_q = hist.axis.Variable(q_edges, name="qt")
    h = hist.Hist(ax_m, ax_y, ax_q, storage=hist.storage.Weight())
    view = h.view()
    for mlo, mhi, ylo, yhi, qlo, qhi, val, unc in bins:
        im = m_edges.index(mlo)
        iy = y_edges.index(ylo)
        iq = q_edges.index(qlo)
        view["value"][im, iy, iq] = scale * val
        view["variance"][im, iy, iq] = (scale * unc) ** 2
    return h


def dyturbo_mass_for_qt_cut(dyturbo_3d, qt_lo, y_max):
    """Sum DYTurbo 3D over y in [-y_max, +y_max] and qT >= qt_lo -> 1D mass hist."""
    return dyturbo_3d[{
        "y": slice(complex(0, -y_max), complex(0, y_max), hist.sum),
        "qt": slice(complex(0, qt_lo), None, hist.sum),
    }]


# -----------------------------------------------------------------------------
# MiNNLO: project to mass on DYTurbo's coarse 10-GeV binning, with qT and Y cuts
# -----------------------------------------------------------------------------

def get_minnlo_mass_in_qt_cut(result, histname, qt_lo, y_max, m_edges_target):
    """Project MiNNLO to a 1D mass hist on ``m_edges_target`` with
    pT(Z) >= qt_lo and |Y(Z)| < y_max applied. Combines high+low DY samples.
    """
    def _one(sample):
        h = result[sample]["output"][histname].get()
        h = h[{
            "ptVgen": slice(complex(0, qt_lo), None, hist.sum),
            "absYVgen": slice(complex(0, 0.0), complex(0, y_max), hist.sum),
            "chargeVgen": hist.sum,
        }]
        h = h.project("massVgen")
        sc = result[sample]["dataset"]["xsec"] / result[sample]["weight_sum"]
        return h.values() * sc, np.asarray(h.axes[0].edges)

    vh, eh = _one("Zmumu_2016PostVFP")
    vl, el = _one("Zmumu10to50_2016PostVFP")
    assert np.allclose(eh, el), "high/low MiNNLO mass axes differ"
    v_fine = vh + vl
    edges_fine = eh

    # Rebin to the coarse DYTurbo mass binning.
    out = np.zeros(len(m_edges_target) - 1)
    for i in range(len(m_edges_target) - 1):
        lo, hi = m_edges_target[i], m_edges_target[i + 1]
        mask = (edges_fine[:-1] >= lo) & (edges_fine[1:] <= hi)
        out[i] = v_fine[mask].sum()

    h_out = hist.Hist(
        hist.axis.Variable(m_edges_target, name="mass"),
        storage=hist.storage.Double(),
    )
    h_out.view()[...] = out
    return h_out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_one_qtcut(h_minnlo, h_dyturbo, qt_lo, y_max, args, outdir, name):
    m_edges = np.asarray(h_minnlo.axes["mass"].edges)
    widths = np.diff(m_edges)

    mi_density = h_minnlo.values() / widths
    dy_density = np.asarray(h_dyturbo.values()) / widths

    h_mi_d = hist.Hist(hist.axis.Variable(m_edges, name="mass"))
    h_mi_d.view()[...] = mi_density
    h_dy_d = hist.Hist(hist.axis.Variable(m_edges, name="mass"))
    h_dy_d.view()[...] = dy_density

    peak = max(float(mi_density.max()), float(np.abs(dy_density).max()))
    ylo = min(0.0, float(dy_density.min())) * 1.2
    ylim = [ylo, peak * 1.25]

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h_mi_d,
        "mass in GeV",
        r"$d\sigma/dm$ (pb / GeV)",
        ylim,
        "DYTurbo / MiNNLO",
        [0.0, 2.0],
        xlim=(m_edges[0], m_edges[-1]),
        width_scale=1,
    )
    ax2 = ratio_axes[-1]

    hep.histplot(
        [h_mi_d, h_dy_d],
        histtype="step",
        color=["black", "green"],
        label=["MiNNLO (pre-FSR)", "DYTurbo NNLO (CT18Z)"],
        linestyle=["-", ":"],
        yerr=False,
        ax=ax1,
        linewidth=2,
        flow="none",
    )
    ax1.legend(title=fr"$q_T(Z) > {qt_lo:g}$ GeV, $|Y(Z)| < {y_max:g}$")
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    plot_tools.add_decor(
        ax1, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos
    )

    ratio = np.where(mi_density != 0, dy_density / mi_density, np.nan)
    h_ratio = hist.Hist(hist.axis.Variable(m_edges, name="mass"))
    h_ratio.view()[...] = np.nan_to_num(ratio, nan=0.0)
    hep.histplot(
        [h_ratio],
        histtype="step",
        color=["green"],
        linestyle=[":"],
        yerr=False,
        ax=ax2,
        linewidth=2,
        flow="none",
    )
    ax2.axhline(1.0, color="grey", linestyle="--")

    plot_tools.fix_axes(ax1, ax2, fig)
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_index_and_log(
        outdir, name, args=args, wd=os.path.dirname(os.path.abspath(__file__))
    )
    plt.close(fig)


def plot_ratio_summary(minnlo_by_cut, dyturbo_by_cut, qt_cuts, y_max, args, outdir, name):
    m_edges = np.asarray(list(minnlo_by_cut.values())[0].axes["mass"].edges)
    widths = np.diff(m_edges)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(qt_cuts)))

    for qt_lo, color in zip(qt_cuts, colors):
        mi = minnlo_by_cut[qt_lo].values() / widths
        dy = np.asarray(dyturbo_by_cut[qt_lo].values()) / widths
        ratio = np.where(mi != 0, dy / mi, np.nan)
        h_ratio = hist.Hist(hist.axis.Variable(m_edges, name="mass"))
        h_ratio.view()[...] = np.nan_to_num(ratio, nan=0.0)
        hep.histplot(
            h_ratio,
            histtype="step",
            color=color,
            label=fr"$q_T(Z) > {qt_lo:g}$ GeV",
            yerr=False,
            ax=ax,
            linewidth=2,
            flow="none",
        )

    ax.axhline(1.0, color="grey", linestyle="--")
    ax.set_xlabel("mass in GeV")
    ax.set_ylabel("DYTurbo / MiNNLO")
    ax.set_xlim(m_edges[0], m_edges[-1])
    ax.set_ylim(-0.5, 2.0)
    ax.legend(title=fr"qT(Z) cut, $|Y(Z)| < {y_max:g}$", loc="upper left")
    plot_tools.add_decor(
        ax, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos
    )
    fig.tight_layout()

    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_index_and_log(
        outdir, name, args=args, wd=os.path.dirname(os.path.abspath(__file__))
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.outdir is None:
        today = datetime.date.today().strftime("%y%m%d")
        outdir = os.path.expanduser(
            f"~/public_html/WMass/{today}_lineshape_low_mass_qt/"
        )
    else:
        outdir = os.path.expanduser(args.outdir)
    outdir = output_tools.make_plot_dir(outdir)
    logger.info(f"Writing plots to {outdir}")

    dyt3d = read_dyturbo_3d(args.dyturbo)
    m_edges = np.asarray(dyt3d.axes["mass"].edges)
    logger.info(
        f"DYTurbo 3D: mass {len(m_edges)-1} bins {list(m_edges)}, "
        f"y {dyt3d.axes['y'].size} bins in [{dyt3d.axes['y'].edges[0]}, {dyt3d.axes['y'].edges[-1]}], "
        f"qT {dyt3d.axes['qt'].size} bins in [{dyt3d.axes['qt'].edges[0]}, {dyt3d.axes['qt'].edges[-1]}]"
    )

    h5 = h5py.File(args.filename_minnlo, "r")
    result = ioutils.load_results_h5py(h5)

    minnlo_by_cut, dyturbo_by_cut = {}, {}
    for qt_lo in args.qt_cuts:
        logger.info(f"--- qT(Z) > {qt_lo} GeV ---")
        h_mi = get_minnlo_mass_in_qt_cut(
            result, args.histname, qt_lo, args.y_max, m_edges
        )
        h_dy = dyturbo_mass_for_qt_cut(dyt3d, qt_lo, args.y_max)
        logger.info(
            f"  MiNNLO  integral = {h_mi.values().sum():9.3f} pb  "
            f"|  DYTurbo integral = {h_dy.values().sum():9.3f} pb"
        )
        minnlo_by_cut[qt_lo] = h_mi
        dyturbo_by_cut[qt_lo] = h_dy

        tag = f"qtgt{qt_lo:g}".replace(".", "p")
        name = "_".join(filter(None, ["m10to60", tag, args.postfix]))
        plot_one_qtcut(h_mi, h_dy, qt_lo, args.y_max, args, outdir, name)

    name = "_".join(filter(None, ["m10to60_ratios_vs_qt", args.postfix]))
    plot_ratio_summary(
        minnlo_by_cut, dyturbo_by_cut, args.qt_cuts, args.y_max, args, outdir, name
    )


if __name__ == "__main__":
    main()
