import ioutils, constants, drell_yan_xsec as dy, functions
import h5py
import matplotlib.pyplot as plt
import mplhep as hep

from wums import boostHistHelpers as hh
from wums import plot_tools

from scipy.integrate import quad
import hist

import numpy as np

def plot_resonance(h_minnlo, mass, width, resonance="z", sin2theta_w=constants.sin2theta_w, ylim=[0,1]):
    print(f"Now at {resonance}")

    m = h_minnlo.axes["mass"].centers

    hists = [h_minnlo]
    labels = ["MiNNLO (post-FSR)"]
    colors = ["black"]
    linestyles=["-"]

    h_non_rel_bw = h_minnlo.copy()

    def convolution(f, im):
        integrand = lambda q: functions.radiator_kernel(im**2/q**2, q) * f(q) * 2*im**2/q**3

        result, error = quad(integrand, im, 13000, epsrel=1e-4)
        return result

    f = lambda x: functions.non_relativistic_breit_wigner(x, mass, width)

    if resonance[0] == "z":
        settings = dict(
            s = constants.s, 
            mass_z = constants.mass_z, 
            width_z = constants.width_z, 
            sin2theta_w = sin2theta_w,
        )

        quark_couplings = dy.get_quark_couplings(sin2theta_w)

        f = lambda x: dy.dsigma_dQ(x**2, quark_couplings, **settings)

        # this really takes too long
        # print("Convolution through integration")
        # h_fo_int = h_minnlo.copy()
        # h_fo_int.values()[...] = [convolution(f, im) for im in m]
        # h_fo_int = hh.normalize(h_fo_int, scale=1)
        # hists.append(h_fo_int)
        # labels.append(r"FO Z/$\gamma*$")  

        print("Convolution through FFT")
        h_fo_fft = h_minnlo.copy()
        h_fo_fft.values()[...] = functions.convolution_fft(f, m, mass, width)
        h_fo_fft = hh.scaleHist(h_fo_fft, 1/h_fo_fft[{"mass":slice(10j,None,hist.sum)}].value)

        hists.append(h_fo_fft)
        labels.append(r"FO Z/$\gamma*$ x FSR (FFT)")
        colors.append("red")
        linestyles.append("--")

        # normalize resonance models on Z peak region
        norm = h_fo_fft[{"mass":slice(80j,105j,hist.sum)}].value
    else:
        norm = 1

    # 1. Non-relativistic Breit-Wigner x Alterelli-Parisi splitting kernel
    non_rel_bw = functions.non_relativistic_breit_wigner(m, mass, width)
    h_non_rel_bw.values()[...] = non_rel_bw
    h_non_rel_bw = hh.normalize(h_non_rel_bw, scale=norm)
    hists.append(h_non_rel_bw)
    labels.append("Non Rel. BW")

    print("Convolution through integration")
    all = [convolution(f, im) for im in m]
    h_all = h_minnlo.copy()
    h_all.values()[...] = all
    h_all = hh.normalize(h_all, scale=norm)
    hists.append(h_all)
    labels.append(r"Non Rel. BW x FSR")
    colors.append("orange")
    linestyles.append(":")

    print("Convolution through FFT")
    h_fft = h_minnlo.copy()

    h_fft.values()[...] = functions.convolution_fft(f, m, mass, width)
    h_fft = hh.normalize(h_fft, scale=norm)
    hists.append(h_fft)
    labels.append(r"Non Rel. BW x FSR (FFT)")
    colors.append("orange")
    linestyles.append("-.")

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h_minnlo,
        "mass in GeV",
        "Frequency",
        ylim,
        "Ratio",
        [0.5, 1.5],
        xlim=(min(m), max(m)),
        width_scale=1,
    )
    ax2 = ratio_axes[-1]

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

    ax1.axvline(mass, color='grey', linestyle='--')

    ax1.legend(ncol=1)

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
    suffix = "postfsr"

    outfile = "_".join(filter(lambda x: x, [resonance, suffix, "lineshape"]))

    plot_tools.save_pdf_and_png("results/", outfile)


# Z boson parameters from PDG
h5file = h5py.File("data/hists_z_80to105_nnpdf31.hdf5", "r")
result = ioutils.load_results_h5py(h5file)
z_hist = ioutils.get_hist(result, "Zmumu_13TeVGen")

z_hist = hh.normalize(z_hist, scale=1)
plot_resonance(z_hist, constants.mass_z, constants.width_z, "z", ylim=[0,0.15])


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


# full Z/gamma* spectrum

h5file = h5py.File("data/hists_z_0to120_nnpdf31.hdf5", "r")
result = ioutils.load_results_h5py(h5file)

z_hist_high = ioutils.get_hist(result, "Zmumu_13TeVGen")
z_hist_low = ioutils.get_hist(result, "DYJetsToMuMuMass10to50_13TeVGen")

z_hist = hh.addHists(z_hist_high, z_hist_low)
z_hist = z_hist[{"mass":slice(10j,None)}]
z_hist = hh.normalize(z_hist, scale=1)

plot_resonance(z_hist, constants.mass_z, constants.width_z, "zgamma", ylim=[0,0.15])