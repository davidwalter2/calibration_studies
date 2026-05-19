import constants, ioutils, drell_yan_xsec as dy
import h5py
from wums import boostHistHelpers as hh
import hist
from wums import plot_tools 
import mplhep as hep

hists = []
labels = []
colors = []
linestyles = []

h5file = h5py.File("data/hists_z_0to120_nnpdf31.hdf5", "r")
result = ioutils.load_results_h5py(h5file)

z_hist_high = ioutils.get_hist(result, "Zmumu_13TeVGen", histname="nominal_gen")
z_hist_low = ioutils.get_hist(result, "DYJetsToMuMuMass10to50_13TeVGen", histname="nominal_gen")

z_hist = hh.addHists(z_hist_high, z_hist_low)
z_hist = z_hist[{"mass":slice(10j,None)}]
z_hist = hh.normalize(z_hist, scale=1)

# hists.append(z_hist)
# labels.append("MiNNLO (pre-FSR)")
# colors.append("black")
# linestyles.append("-")

Q_val = z_hist.axes["mass"].centers
Q2 = Q_val**2

print("Load integrals")
h5file = h5py.File("data/NNPDF31_nnlo_as_0118.hdf5", "r")
integrals = ioutils.load_results_h5py(h5file)
# integrals = {}
# for i, r in results.items():
#     integrals[i] = [a[40:] for a in r]

def get_pred(
    s = constants.s, 
    mass_z = constants.mass_z, 
    width_z = constants.width_z, 
    sin2theta_w = constants.sin2theta_w,
    pdf_member=0,
    norm=None
):
    quark_couplings = dy.get_quark_couplings(sin2theta_w)

    h = z_hist.copy()
    h.values()[...] = dy.dsigma_dQ(
        Q2, 
        quark_couplings,
        s = s, 
        mass_z = mass_z, 
        width_z = width_z,
        sin2theta_w = sin2theta_w,
        integrals=integrals[pdf_member],
        )
    if norm is not None:
        h = hh.scaleHist(h, 1./norm)
        
    return h

print("Central prediction")
h_nom = get_pred()
norm = h_nom.sum().value
h_nom = hh.normalize(h_nom, scale=1)
hists.append(h_nom)
labels.append(r"FO Z/$\gamma*$")
colors.append("black")
linestyles.append("-")

hists_pdfs = [get_pred(pdf_member=i, norm=norm) for i in range(1,101)]

print("Variation")
hists.append(get_pred(sin2theta_w=constants.sin2theta_w+0.0003, norm=norm))
labels.append(r"$sin^2(\theta_\mathrm{W}*) \pm 0.0003$")
colors.append("red")
linestyles.append("--")

print("Variation")
hists.append(get_pred(mass_z=constants.mass_z+0.002, norm=norm))
labels.append(r"$m_\mathrm{Z} \pm 2MeV$")
colors.append("green")
linestyles.append(":")

print("Variation")
hists.append(get_pred(width_z=constants.width_z+0.002, norm=norm))
labels.append(r"$\Gamma_\mathrm{Z} \pm 2MeV$")
colors.append("blue")
linestyles.append("-.")


fig, ax1, ratio_axes = plot_tools.figureWithRatio(
    hists[0],
    "mass in GeV",
    "Frequency",
    [0,0.05],
    # (0,0.001),
    "Ratio",
    [0.91, 1.09],
    xlim=(10, 120),
    # xlim=(30,80),
    width_scale=1,
)
ax2 = ratio_axes[-1]

# Plotting
hep.histplot(
    hists[0],
    histtype="step",
    color=colors[0],
    label=labels[0],
    linestyle=linestyles[0],
    yerr=False,
    ax=ax1,
    linewidth=2,
    flow="none",
)

ax1.legend(ncol=2)

# PDF variations
hep.histplot(
    [hh.divideHists(h, hists[0]) for h in hists_pdfs],
    histtype="step",
    color="grey",
    linestyle="-",
    yerr=False,
    ax=ax2,
    linewidth=2,
    flow="none",
)

hep.histplot(
    [hh.divideHists(h, hists[0]) for h in hists[1:]],
    histtype="step",
    color=colors[1:],
    linestyle=linestyles[1:],
    label=labels[1:],
    yerr=False,
    ax=ax2,
    linewidth=2,
    flow="none",
)



ax2.legend(ncol=3)


ax2.axhline(1, color='grey', linestyle='--')

plot_tools.fix_axes(ax1, ax2, fig)

# Save the plot
suffix = "prefsr_var"

outfile = "_".join(filter(lambda x: x, ["zgamma", suffix, "lineshape"]))

plot_tools.save_pdf_and_png("results/", outfile)

