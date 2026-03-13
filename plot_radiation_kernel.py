import functions, constants
import hist
import mplhep as hep

from wums import boostHistHelpers as hh
from wums import plot_tools

def get_kernel_hist(h, mass):

    k = h.axes["k"].centers
    kernel_z = functions.radiator_kernel(k, mass)

    h_kernel = h.copy()
    h_kernel.values()[...] = kernel_z
    h_kernel = hh.normalize(h_kernel, 1.)

    return h_kernel

def plot(h_kernel):

    hists = [
        get_kernel_hist(h_kernel, constants.mass_z),
        get_kernel_hist(h_kernel, constants.mass_u3),
        get_kernel_hist(h_kernel, constants.mass_u2),
        get_kernel_hist(h_kernel, constants.mass_u1),
        get_kernel_hist(h_kernel, constants.mass_j)
        ]
    labels = [
        r"p($\mathit{k};\mathit{m}_\mathrm{Z}$)",
        r"p($\mathit{k};\mathit{m}_{\mathrm{Y}(3S)}$)",
        r"p($\mathit{k};\mathit{m}_{\mathrm{Y}(2S)}$)",
        r"p($\mathit{k};\mathit{m}_{\mathrm{Y}(1S)}$)",
        r"p($\mathit{k};\mathit{m}_{J/\psi}$)",
        ]

    fig, ax1, ratio_axes = plot_tools.figureWithRatio(
        h_kernel,
        r"$\mathit{k}=\mathit{E}_\gamma / \mathit{E}_\mu$",
        "Frequency",
        [0,0.4],
        "Ratio",
        [0.88, 1.09],
        xlim=[0, 1],
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

    # ax1.axvline(mass, color='grey', linestyle='--')

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

    # ax2.axvline(mass, color='grey', linestyle='--')
    ax2.axhline(1, color='grey', linestyle='--')

    plot_tools.fix_axes(ax1, ax2, fig)

    # Save the plot
    suffix = "radiation_kernel"

    outfile = "_".join(filter(lambda x: x, [suffix, "lineshape"]))

    plot_tools.save_pdf_and_png("results/", outfile)

kmin = 0
kmax = 1

k_hist = hist.Hist(hist.axis.Regular(100, kmin, kmax, name="k",flow=False))

plot(k_hist)