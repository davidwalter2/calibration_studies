"""Shared style for the Z-mass working-group deck of 2026-09-09.

One place for the palette and the figure geometry so the four figures of
`slides/260909_unbinned_likelihood_walter/assets/` look like one set.

Rules baked in here (from the user's plotting preferences):
  * `hep.style.ROOT` via mplhep;
  * `constrained_layout=True` -- no manual `subplots_adjust`;
  * 13 x 8 in at 160 dpi, so nothing is smaller than ~14 pt at slide scale;
  * SHORT axis labels (long ones get clipped by `bbox_inches="tight"` when a
    ratio panel shares the left margin);
  * MIT red / gray / dark as the palette, black points for the MC candidates
    and red for the model.

`TeX Gyre Heros` (the mplhep ROOT font) has Greek, arrows, superscript two,
plus/minus and one-half, but NOT SUPERSCRIPT ZERO (U+2070) or CIRCLED TIMES
(U+2297) -- use mathtext (`$D^{0}$`) or a plain `x` for those two.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep

# ---------------------------------------------------------------- palette ---
MIT_RED = "#A31F34"
MIT_GRAY = "#8A8B8C"
MIT_DARK = "#222222"

# one colour family per data-channel row of `vision.png`.  Chosen to sit
# around the MIT red rather than to compete with it.
CH_COLORS = [
    "#A31F34",  # J/psi        -- MIT red
    "#C1663A",  # Upsilon      -- warm brown
    "#7A2E5B",  # Z/gamma*     -- plum
    "#2E6E75",  # V0           -- teal
    "#39557A",  # heavy flavour-- slate blue
    "#5A5C5E",  # cosmics      -- dark gray (no mass term)
]

FIGSIZE = (13.0, 8.0)
DPI = 160


def use_style():
    hep.style.use(hep.style.ROOT)
    # mplhep's ROOT style leaves the top/right spines on, which is what we
    # want for the spectra; the schematics switch the axes off entirely.
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI


def blank_canvas(xlim=(0, 100), ylim=(0, 100)):
    """A figure whose data coordinates are per-cent of the canvas."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    return fig, ax


def box(ax, x0, y0, x1, y1, fc="white", ec=MIT_DARK, lw=1.6, r=1.6,
        alpha=1.0, zorder=2, ls="-"):
    """A rounded rectangle in canvas coordinates; returns the patch."""
    p = mpl.patches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=mpl.patches.BoxStyle("Round", pad=0, rounding_size=r),
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
        zorder=zorder, linestyle=ls, mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=MIT_DARK, lw=2.2, rad=0.0, zorder=3,
          ls="-", ms=18):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=ms,
                                linewidth=lw, color=color, linestyle=ls,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)


def lines(ax, xs, ys, color=MIT_GRAY, lw=2.0, zorder=1, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, zorder=zorder, ls=ls,
            solid_capstyle="round")


def label(ax, x, y, s, size=16, color=MIT_DARK, ha="left", va="center",
          weight="normal", zorder=4, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, zorder=zorder, **kw)
