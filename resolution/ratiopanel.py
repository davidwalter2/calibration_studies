"""A data/model ratio panel underneath a binned density plot.

Two conventions are baked in here so every figure in this directory
gets them identically:

  * the MODEL is integrated over the bin, not sampled at the bin centre.
    `bin_average` does the Simpson average from the model density
    evaluated at the two edges and the centre of each bin -- for a
    lineshape with real curvature (a J/psi peak in 4 MeV bins, the
    cusp of a pull distribution at z = 0) the centre value can be off
    by a few per mille, which is exactly the size of the effects these
    closure plots are made to show.

  * the ERROR BAR is the Poisson error of the data counts only
    (ratio/sqrt(N)); the model is a prediction, not a measurement with
    its own uncertainty, and the subsample noise of the model curve is
    negligible next to sqrt(N) in every bin that is plotted.

The outermost bins of these histograms are CLIPPED overflow bins (the
scripts `np.clip` the sample into the plotted range so that nothing is
silently dropped from the density normalisation), so their content is
not the density of that bin and `drop_edges=True` leaves them out of
the ratio.
"""

import matplotlib.pyplot as plt
import numpy as np


def make_ratio_fig(figsize=(10., 7.5), height_ratios=(3., 1.), hspace=0.06):
    """Main axis + a shorter ratio axis sharing its x."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=list(height_ratios),
                          hspace=hspace)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    return fig, ax, rax


def bin_average(f_lo, f_mid, f_hi):
    """Simpson bin average of a density from edge/centre/edge samples."""
    return (np.asarray(f_lo) + 4. * np.asarray(f_mid) + np.asarray(f_hi)) / 6.


def _ylim(ratio, err, clamp, pad=0.35, errmax=0.10):
    """Auto range from the bins that actually measure something.

    A far-tail bin with three entries has a 60% Poisson error; letting it
    set the range would push every plot to the clamp and squash the
    per-mille structure in the core, so the range is taken from the bins
    with err <= errmax (>= ~100 entries) and only then clamped.
    """
    good = np.isfinite(ratio) & np.isfinite(err)
    fine = good & (err <= errmax)
    use = fine if fine.sum() >= 3 else good
    if not use.any():
        return clamp
    lo = np.min((ratio - err)[use])
    hi = np.max((ratio + err)[use])
    half = max(1. - lo, hi - 1., 1e-3) * (1. + pad)
    return (max(clamp[0], 1. - half), min(clamp[1], 1. + half))


def draw_ratio(rax, edges, counts, model_density, ntot,
               ylabel="data / model", clamp=(0.5, 1.5), drop_edges=True,
               color="black", band=0.05, xlabel=None):
    """Draw counts/(model integrated over the bin) with Poisson errors.

    `model_density` is the per-bin AVERAGE model density (i.e. the bin
    integral divided by the bin width) -- see `bin_average`. Returns
    (centres, ratio, err, n_outside) where n_outside counts the plotted
    bins whose central value falls outside the clamped y range.
    """
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    w = np.diff(edges)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    dens = counts / (ntot * w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dens / np.asarray(model_density, dtype=float)
        err = ratio / np.sqrt(np.where(counts > 0, counts, np.nan))
    keep = counts > 0
    if drop_edges and len(keep) > 2:
        keep[0] = keep[-1] = False
    ratio = np.where(keep, ratio, np.nan)
    err = np.where(keep, err, np.nan)

    if band:
        rax.axhspan(1. - band, 1. + band, color="0.85", zorder=0)
    rax.axhline(1., color="gray", lw=1.2, zorder=1)
    rax.errorbar(ctr, ratio, err, marker="o", markersize=3.2, linestyle="none",
                 color=color, elinewidth=1.0, capsize=0., zorder=2)
    lo, hi = _ylim(ratio, err, clamp)
    rax.set_ylim(lo, hi)
    rax.set_ylabel(ylabel, fontsize="small")
    rax.tick_params(labelsize="small")
    if xlabel is not None:
        rax.set_xlabel(xlabel)
    n_out = int(np.sum(np.isfinite(ratio) & ((ratio < lo) | (ratio > hi))))
    return ctr, ratio, err, n_out
