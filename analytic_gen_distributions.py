"""
This script compares generator level distributions of resonances: J/Psi, Upsilon, Z
- from MC events, pre-FSR & post FSR
- from Gaussian assumption
- from non-relativistic BW
- from relativistic BW
- convolutions?
- ...
"""
import constants, drell_yan_xsec as dy, functions, ioutils

import h5py

import numpy as np
import matplotlib.pyplot as plt
from wums import plot_tools 
import hist
import mplhep as hep
from wums import boostHistHelpers as hh



def plot_resonance(h_minnlo, mass, width, resonance="z", sin2theta_w=constants.sin2theta_w, ylim=[0,1]):

    m = h_minnlo.axes["mass"].centers

    hists = [h_minnlo]
    labels = ["MiNNLO (pre-FSR)"]
    
    # 1. Gaussian
    h_gaussian = h_minnlo.copy()
    # We match the Gaussian's Full Width at Half Maximum (FWHM) to the Z boson's width
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((m - mass) / sigma)**2)
    h_gaussian.values()[...] = gaussian
    h_gaussian = hh.normalize(h_gaussian, scale=1)
    hists.append(h_gaussian)
    labels.append("Gaussian")

    # 2. Non-relativistic Breit-Wigner
    h_non_rel_bw = h_minnlo.copy()
    # Simple Cauchy distribution with constant width

    non_rel_bw = functions.non_relativistic_breit_wigner(m, mass, width)
    h_non_rel_bw.values()[...] = non_rel_bw
    h_non_rel_bw = hh.normalize(h_non_rel_bw, scale=1)
    hists.append(h_non_rel_bw)
    labels.append("Non Rel. BW")

    # 3. Relativistic Breit-Wigner
    h_rel_bw = h_minnlo.copy()
    # With energy-dependent width: Gamma(m) = width * (m^2 / mass^2)
    gamma_m = width * (m**2 / mass**2)
    rel_bw = (m**2 * gamma_m) / ((m**2 - mass**2)**2 + (m * gamma_m)**2)
    h_rel_bw.values()[...] = rel_bw
    h_rel_bw = hh.normalize(h_rel_bw, scale=1)
    hists.append(h_rel_bw)
    labels.append("Rel. BW")
    
    if resonance[0] == "z":
        # 4. Fixed order calculation

        settings = dict(
            s = constants.s, 
            mass_z = constants.mass_z, 
            width_z = constants.width_z, 
            sin2theta_w = sin2theta_w,
        )

        quark_couplings = dy.get_quark_couplings(sin2theta_w)

        Q_val = m
        # gamma = [functions.dsigma_dQ_1(Q**2, quark_couplings) for Q in Q_val]
        # gamma_z = [functions.dsigma_dQ_2(Q**2, quark_couplings, **settings) for Q in Q_val]
        # z = [functions.dsigma_dQ_3(Q**2, quark_couplings, **settings) for Q in Q_val]

        prefsr = dy.dsigma_dQ(Q_val**2, quark_couplings, **settings)

        h_fo = h_minnlo.copy()
        h_fo.values()[...] = prefsr
        h_fo = hh.scaleHist(h_fo, 1/h_fo[{"mass":slice(10j,None,hist.sum)}].value)

        hists.append(h_fo)
        labels.append(r"FO Z/$\gamma*$")

    else:
        #FIXME: replace by MC
        hists[0].values()[...] = rel_bw
        hists[0] = hh.normalize(hists[0], scale=1)

    # Normalize the shapes over the plotted range to compare their profiles fairly
    # gaussian_normalized = gaussian / np.trapz(gaussian, m)
    # non_rel_bw_normalized = non_rel_bw / np.trapz(non_rel_bw, m)
    # rel_bw_normalized = rel_bw / np.trapz(rel_bw, m)

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h_minnlo,
        "mass in GeV",
        "Frequency",
        ylim,
        # (0,0.001),
        "Ratio",
        [0.5, 1.5],
        xlim=(min(m), max(m)),
        # xlim=(30,80),
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

    ax1.legend(ncol=2)

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
    suffix = "prefsr"

    outfile = "_".join(filter(lambda x: x, [resonance, suffix, "lineshape"]))

    plot_tools.save_pdf_and_png("results/", outfile)


# Z boson
h5file = h5py.File("data/hists_z_80to105_nnpdf31.hdf5", "r")
result = ioutils.load_results_h5py(h5file)
z_hist = ioutils.get_hist(result, "Zmumu_13TeVGen", histname="nominal_gen")
z_hist = hh.normalize(z_hist, scale=1)
plot_resonance(z_hist, constants.mass_z, constants.width_z, "z", ylim=[0,0.15])


# full Z/gamma* spectrum
h5file = h5py.File("data/hists_z_0to120_nnpdf31.hdf5", "r")
result = ioutils.load_results_h5py(h5file)

z_hist_high = ioutils.get_hist(result, "Zmumu_13TeVGen", histname="nominal_gen")
z_hist_low = ioutils.get_hist(result, "DYJetsToMuMuMass10to50_13TeVGen", histname="nominal_gen")

z_hist = hh.addHists(z_hist_high, z_hist_low)
z_hist = hh.normalize(z_hist, scale=1)

plot_resonance(z_hist, constants.mass_z, constants.width_z, "zgamma", ylim=[0,0.15])


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
