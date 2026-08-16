import ioutils, constants, drell_yan_xsec as dy, functions
import h5py
import lhapdf
import matplotlib.pyplot as plt
import mplhep as hep

from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools

from scipy.integrate import quad
import hist

import numpy as np

import argparse
import os
import datetime

# CMS style + wums logging (per the cms-plots skill).
hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the Z/gamma* resonance lineshape from MiNNLO gen-level histograms"
    )
    parser.add_argument(
        "-i", "--input", dest="filename_minnlo",
        default="/ceph/submit/data/user/d/david_w/WMassAnalysis/260520_z_gen/w_z_gen_dists.hdf5",
        help="Input hdf5 file with gen-level histograms (default: %(default)s)",
    )
    parser.add_argument(
        "--prefsr", action="store_true",
        help="Use pre-FSR generator histograms (default: post-FSR)",
    )
    parser.add_argument(
        "-o", "--outdir", default=None,
        help="Output directory for plots "
             "(default: ~/public_html/WMass/<YYMMDD>_lineshape/)",
    )
    parser.add_argument(
        "--dyturbo",
        default="/work/submit/david_w/TheoryCorrections/DYTURBO/nnlo/results_z-1d-mz-nnlo-ct18z-pt1to13000.txt",
        help="DYTURBO 1D mass result file to overlay; pass '' to disable "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--pdf", default="CT18ZNNLO",
        help="LHAPDF set for the FO Z/gamma* calculation "
             "(default: %(default)s, matching the DYTURBO ct18z nnlo overlay)",
    )
    parser.add_argument(
        "--y-cut", type=float, default=5.0,
        help="Apply |Y(Z)| < Y_CUT to the FO calculation (default: %(default)s "
             "to match DYTurbo y_bins=[-5,5]). Pass a large value (e.g. 100) "
             "for the inclusive case.",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize each curve to unit integral over the plot range "
             "(default: show absolute predictions, ~pb).",
    )
    parser.add_argument(
        "--pt-min", type=float, default=1.0,
        help="Lower pT(Z) cut applied to MiNNLO in GeV (default: %(default)s, "
             "matches DYTurbo qt_bins lower edge).",
    )
    parser.add_argument(
        "--pt-max", type=float, default=100.0,
        help="Upper pT(Z) cut applied to MiNNLO in GeV (default: %(default)s, "
             "matches DYTurbo qt_bins upper edge).",
    )
    parser.add_argument(
        "--y-max", type=float, default=5.0,
        help="|Y(Z)| cut applied to MiNNLO (default: %(default)s, matches "
             "DYTurbo y_bins=[-5,5]). The cut is propagated explicitly so it "
             "works on samples whose absYVgen axis extends beyond this value "
             "(e.g. an uncut histmaker run).",
    )
    parser.add_argument(
        "--postfix", default="",
        help="Optional string appended to the output filename, e.g. 'nocut' "
             "produces zgamma_prefsr_lineshape_nocut.{pdf,png}.",
    )
    parser.add_argument(
        "--histname", default="nominal_gen",
        help="MiNNLO output histogram key to project (default: %(default)s; "
             "use 'nominal_lhe' for the LHE-level matrix-element distribution).",
    )
    # plot decoration (cms-plots skill convention)
    parser.add_argument("--title", default="CMS", help="Decor title")
    parser.add_argument("--subtitle", default="Work in progress", help="Decor subtitle")
    parser.add_argument(
        "--titlePos", type=int, default=0,
        help="Decor position (mplhep loc; 0=outside above-left, 2=inside top-left).",
    )
    return parser.parse_args()


def read_dyturbo(filename, fb_to_pb=True):
    """Read a 1D DYTURBO result file into a hist.

    The file has one row per bin with columns ``lo hi value uncertainty``
    (comment lines start with ``#``). DYTURBO appends a final row spanning
    the full range (the integral over all bins); it is identified by its
    [lo, hi] covering the global edges and dropped.

    DYTurbo's default output unit is **fb** (the .in file's ``absaccuracy``
    comment confirms this). To compare with MiNNLO / FO (both in pb), the
    values are converted to pb by default (multiplied by 1e-3); pass
    ``fb_to_pb=False`` to keep the raw file values.

    Returns a ``hist.Hist`` with a Variable ``mass`` axis and Weight storage
    (value = cross section per bin, variance = uncertainty**2).
    """
    rows = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lo, hi, val, unc = (float(x) for x in line.split()[:4])
            rows.append((lo, hi, val, unc))

    if not rows:
        raise ValueError(f"No data rows found in {filename}")

    lo0 = min(r[0] for r in rows)
    hi1 = max(r[1] for r in rows)
    # Drop the trailing total/integral row that spans the full range.
    bins = [r for r in rows if not (len(rows) > 1 and r[0] == lo0 and r[1] == hi1)]
    bins.sort(key=lambda r: r[0])

    scale = 1e-3 if fb_to_pb else 1.0
    edges = np.array([r[0] for r in bins] + [bins[-1][1]], dtype=float)
    values = scale * np.array([r[2] for r in bins], dtype=float)
    errors = scale * np.array([r[3] for r in bins], dtype=float)

    h = hist.Hist(hist.axis.Variable(edges, name="mass"),
                  storage=hist.storage.Weight())
    view = h.view()
    view["value"] = values
    view["variance"] = errors ** 2
    return h

def plot_resonance(h_minnlo, mass, width, resonance="z", sin2theta_w=constants.sin2theta_w, ylim=[0,1], postfsr=True, outdir="./", dyturbo=None, pdf_set="CT18ZNNLO", y_cut=5.0, normalize=False, postfix="", args=None):

    m = h_minnlo.axes["mass"].centers

    hists = [h_minnlo]
    if postfsr:
        labels = ["MiNNLO (post-FSR)"]
    else:
        labels = ["MiNNLO (pre-FSR)"]

    # 1. Non-relativistic Breit-Wigner x Alterelli-Parisi splitting kernel

    if resonance[0] == "z":
        settings = dict(
            s = constants.s, 
            mass_z = constants.mass_z, 
            width_z = constants.width_z, 
            sin2theta_w = sin2theta_w,
        )

        quark_couplings = dy.get_quark_couplings(sin2theta_w)

        # drell_yan_xsec no longer keeps a module-level PDF (integrals= API):
        # build the PDF here and pass the PDF-luminosity integrals explicitly,
        # computed on whatever mass grid f is evaluated at (works for both the
        # vectorized pre-FSR call and the convolution_fft sampling). This keeps
        # plot_lineshape self-contained, with no dependency on a pre-generated
        # data/<pdfset>.hdf5 grid (cf. plot_dy.py / make_pdf_grid.py).
        pdf = lhapdf.mkPDF(pdf_set, 0)

        def f(x):
            Q2 = np.atleast_1d(np.asarray(x, dtype=float)) ** 2
            integrals = [
                dy.integrate_sigma_hat_prime_sm_Ycut(
                    constants.s, fl + 1, Q2, pdf, y_cut
                )
                for fl in range(5)
            ]
            return dy.dsigma_dQ(
                Q2, quark_couplings, integrals=integrals, **settings
            )

        if postfsr:
            # # this really takes too long
            # print("Convolution through integration")
            # h_fo_int = h_minnlo.copy()
            # h_fo_int.values()[...] = [convolution(f, im) for im in m]
            # h_fo_int = hh.normalize(h_fo_int, scale=1)
            # hists.append(h_fo_int)
            # labels.append(r"FO Z/$\gamma*$")

            logger.info("Convolution through FFT")
            h_fo_fft = h_minnlo.copy()
            h_fo_fft.values()[...] = functions.convolution_fft(f, m, mass, width)
            if normalize:
                h_fo_fft = hh.scaleHist(h_fo_fft, 1/h_fo_fft[{"mass":slice(10j,None,hist.sum)}].value)

            hists.append(h_fo_fft)
            labels.append(r"FO Z/$\gamma*$ x FSR (FFT)")
        else:
            h_fo = h_minnlo.copy()
            h_fo.values()[...] = f(m)
            if normalize:
                h_fo = hh.scaleHist(h_fo, 1/h_fo[{"mass":slice(10j,None,hist.sum)}].value)

            hists.append(h_fo)
            labels.append(r"FO Z/$\gamma*$")
    else:
        if postfsr:
            def convolution(f, im):
                # print(f"Now at m={im}")
                integrand = lambda q: functions.radiator_kernel(im**2/q**2, q) * f(q) * 2*im**2/q**3

                result, error = quad(integrand, im, 13000, epsrel=1e-4)
                return result

            f = lambda x: functions.non_relativistic_breit_wigner(x, mass, width)

            logger.info("Convolution through integration")
            all = [convolution(f, im) for im in m]
            h_all = h_minnlo.copy()
            h_all.values()[...] = all
            if normalize:
                h_all = hh.normalize(h_all, scale=1)
            hists.append(h_all)
            labels.append(r"Non Rel. BW x FSR")

            logger.info("Convolution through FFT")
            h_fft = h_minnlo.copy()

            h_fft.values()[...] = functions.convolution_fft(f, m, mass, width)
            if normalize:
                h_fft = hh.normalize(h_fft, scale=1)
            hists.append(h_fft)
            labels.append(r"Non Rel. BW x FSR (FFT)")
        else:
            h_non_rel_bw = h_minnlo.copy()
            # Simple Cauchy distribution with constant width

            non_rel_bw = functions.non_relativistic_breit_wigner(m, mass, width)
            h_non_rel_bw.values()[...] = non_rel_bw
            if normalize:
                h_non_rel_bw = hh.normalize(h_non_rel_bw, scale=1)
            hists.append(h_non_rel_bw)
            labels.append("Non Rel. BW")

    # # We exclude the singularity by starting delta_E at a small cutoff
    # delta_E = np.linspace(0, max(m) - min(m), len(m))

    # L = np.log(constants.mass_z**2 / constants.mass_muon**2)
    # beta = (2 * constants.alpha_ew / np.pi) * (L - 1)
    # epsilon = 0.01
    # kernel = beta * (delta_E + epsilon)**(beta - 1)

    # non_rel_bw = functions.non_relativistic_breit_wigner(m, mass, width)

    # postfsr_radiated = fftconvolve(non_rel_bw, kernel, mode='full')[len(m)-1 : 2*len(m)-1] * (m[1] - m[0])

    # # Combine case w/o radiation with the one with radiation with weights
    # # A is the 'Sudakov' factor: prob of no emission > epsilon
    # A = epsilon**beta
    # postfsr = A * non_rel_bw + postfsr_radiated

    # h_fo_fsr = h_minnlo.copy()
    # h_fo_fsr.values()[...] = postfsr
    # h_fo_fsr = hh.scaleHist(h_fo_fsr, 1./h_fo_fsr.sum().value)

    # hists.append(h_fo_fsr)
    # labels.append(r"FO+FSR Z/$\gamma*$")


    if dyturbo is not None:
        # The DYTURBO result has its own (generally different) binning.
        # Resample it onto the MiNNLO mass axis by interpolating the per-GeV
        # density at the bin centers, so it can share the ratio panel.
        dyt_edges = dyturbo.axes["mass"].edges
        dyt_centers = dyturbo.axes["mass"].centers
        dyt_density = dyturbo.values() / np.diff(dyt_edges)

        minnlo_widths = np.diff(h_minnlo.axes["mass"].edges)
        h_dyturbo = h_minnlo.copy()
        h_dyturbo.values()[...] = (
            np.interp(m, dyt_centers, dyt_density, left=0.0, right=0.0)
            * minnlo_widths
        )
        if normalize:
            # Normalize as for the FO curve (unit integral above 10 GeV).
            h_dyturbo = hh.scaleHist(
                h_dyturbo, 1 / h_dyturbo[{"mass": slice(10j, None, hist.sum)}].value
            )

        hists.append(h_dyturbo)
        labels.append(r"DYTURBO NNLO (CT18Z)")

    if normalize:
        ylim_eff = ylim
        ylabel = "Frequency"
    else:
        # Auto-scale: peak * 1.15, so all curves are visible.
        peak = max(float(h.values().max()) for h in hists)
        ylim_eff = [0.0, peak * 1.15]
        ylabel = r"$d\sigma/dm$ (pb / GeV)"

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h_minnlo,
        "mass in GeV",
        ylabel,
        ylim_eff,
        "Ratio",
        [0.5, 1.5],
        xlim=(h_minnlo.axes["mass"].edges[0], h_minnlo.axes["mass"].edges[-1]),
        width_scale=1,
    )
    ax2 = ratio_axes[-1]

    colors=["black", "blue", "green", "orange", "red", "red", "red", "red"][:len(hists)]
    linestyles=["-", "--",":","-.", "-", "--",":","-.",][:len(hists)]

    # Plotting
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

    # ax1.plot(m, gaussian, label='Gaussian', linestyle=':', linewidth=2.5, color="green")
    # ax1.plot(m, non_rel_bw, label='Non-Rel. BW', linestyle='--', linewidth=2.5, color="orange")
    # ax1.plot(m, rel_bw, label='Rel. BW', linestyle='-', linewidth=2.5, color="red")

    ax1.axvline(mass, color='grey', linestyle='--')

    ax1.legend(ncol=1)

    # CMS-style decoration (cms-plots skill).
    if args is not None:
        plot_tools.add_decor(
            ax1, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos
        )

    hep.histplot(
        [hh.divideHists(h, hists[0]) for h in hists[1:]],
        histtype="step",
        color=colors[1:],
        linestyle=linestyles[1:],
        yerr=False,
        ax=ax2,
        linewidth=2,
        flow="none",
    )

    ax2.axvline(mass, color='grey', linestyle='--')
    ax2.axhline(1, color='grey', linestyle='--')

    plot_tools.fix_axes(ax1, ax2, fig)

    # Save the plot
    suffix = "postfsr" if postfsr else "prefsr"

    outfile = "_".join(filter(lambda x: x, [resonance, suffix, "lineshape", postfix]))

    plot_tools.save_pdf_and_png(outdir, outfile)
    # Drop a .log file (+ index.php in the web-served outdir) so each plot
    # carries its command line, parsed args, git hash and diff alongside it.
    output_tools.write_index_and_log(
        outdir, outfile, args=args, wd=os.path.dirname(os.path.abspath(__file__))
    )


# # Z boson parameters from PDG
# h5file = h5py.File("data/hists_z_80to105_nnpdf31.hdf5", "r")
# result = ioutils.load_results_h5py(h5file)
# z_hist = ioutils.get_hist(result, "Zmumu_13TeVGen")
# z_hist = hh.normalize(z_hist, scale=1)
# plot_resonance(z_hist, constants.mass_z, constants.width_z, "z", ylim=[0,0.15])


# full Z/gamma* spectrum
args = parse_args()

postfsr = not args.prefsr
# --prefsr selects the plotted curves (analytic / convolution overlays), not
# the input histogram; the input histogram is chosen by --histname.
histname = args.histname

if args.outdir is None:
    today = datetime.date.today().strftime("%y%m%d")
    outdir = os.path.expanduser(f"~/public_html/WMass/{today}_lineshape/")
else:
    outdir = os.path.expanduser(args.outdir)

# make_plot_dir creates the dir (and drops index.php for web browsing).
outdir = output_tools.make_plot_dir(outdir)
logger.info(f"Writing plots to {outdir}")

h5file = h5py.File(args.filename_minnlo, "r")
result = ioutils.load_results_h5py(h5file)

z_hist_high = ioutils.get_hist(result, "Zmumu_2016PostVFP", histname=histname,
                               pt_min=args.pt_min, pt_max=args.pt_max,
                               y_max=args.y_max)
z_hist_low = ioutils.get_hist(result, "Zmumu10to50_2016PostVFP", histname=histname,
                              pt_min=args.pt_min, pt_max=args.pt_max,
                              y_max=args.y_max)

z_hist = hh.addHists(z_hist_high, z_hist_low)
# Restrict to 10-120 GeV; every other curve is built from this hist, so the
# whole plot domain and all normalizations follow from this single slice.
z_hist = z_hist[{"mass": slice(10j, 120j)}]
if args.normalize:
    z_hist = hh.normalize(z_hist, scale=1)

dyturbo_hist = read_dyturbo(args.dyturbo) if args.dyturbo else None

plot_resonance(z_hist, constants.mass_z, constants.width_z, "zgamma", ylim=[0,0.15], postfsr=postfsr, outdir=outdir, dyturbo=dyturbo_hist, pdf_set=args.pdf, y_cut=args.y_cut, normalize=args.normalize, postfix=args.postfix, args=args)

exit()

# J/Psi
j_m_min = 3.096
j_m_max = 3.098

j_hist = hist.Hist(hist.axis.Regular(100, j_m_min, j_m_max, name="mass",flow=False))

plot_resonance(j_hist, constants.mass_j, constants.width_j, "jpsi", ylim=[0,0.3])


# Upsilon
u1_m_min = 9.459
u1_m_max = 9.461
u1_hist = hist.Hist(hist.axis.Regular(100, u1_m_min, u1_m_max, name="mass",flow=False))

plot_resonance(u1_hist, constants.mass_u1, constants.width_u1, "upsilon_1s", ylim=[0,0.6])


u2_m_min = 10.022
u2_m_max = 10.024
u2_hist = hist.Hist(hist.axis.Regular(100, u2_m_min, u2_m_max, name="mass",flow=False))

plot_resonance(u2_hist, constants.mass_u2, constants.width_u2, "upsilon_2s", ylim=[0,0.6])


u3_m_min = 10.354
u3_m_max = 10.357
u3_hist = hist.Hist(hist.axis.Regular(100, u3_m_min, u3_m_max, name="mass",flow=False))

plot_resonance(u3_hist, constants.mass_u3, constants.width_u3, "upsilon_3s", ylim=[0,0.6])
