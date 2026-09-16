#!/usr/bin/env python3
"""What the J/psi FSR kernel does to the fitted calibration parameters.

Two panels, one file each, from `native_dump.py` JSONs:

  * `fsrfit_shift`  -- the change in every calibration parameter between the
    kernel and the no-kernel fit, in units of its own postfit error, with the
    50 field modes and the 42 material amounts separated.  Truth is zero for
    all of them on this MC, so the no-kernel row's own pulls are drawn behind
    the difference.
  * `fsrfit_dmean`  -- the ONE number the comparison is about: the mean
    predicted J/psi mass shift the fitted calibration vector produces,
    `<D_card> . theta`, per row, against the kernel's own `<dm>`.

usage::

    python3 plot_jpsi_fsr_fit.py --ref J0 \\
        J0=runs/engaging_260916/rabbit_J0.json \\
        JK=runs/engaging_260916/rabbit_JK.json \\
        JD=runs/engaging_260916/rabbit_JD.json
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.gridspec import GridSpec

from wums import logging

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "resolution"))
import jpsi_fsr_table as T  # noqa: E402
import pubhtml  # noqa: E402

MJPSI = 3.0969
#: the kernel means, MeV (`zchannel/jpsi_fsr_kernel.py --report`)
DM_KERNEL = {"J0": 0.0, "P2N": 0.0, "P2XP": 0.0,
             "JK": -7.1969, "P2K": -7.1969, "JD": -8.0089}
COL = {"J0": "#7f7f7f", "JK": "#1f77b4", "JD": "#d62728",
       "P2N": "#9467bd", "P2K": "#2ca02c", "P2XP": "#8c564b"}


def fig_shift(rows, ref, out):
    R = rows[ref]
    names = [n for n in R["p"] if n.startswith(("bfield_mode", "material_"))]
    nb = sum(n.startswith("bfield_mode") for n in names)
    x = np.arange(len(names))
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=(1, 1), hspace=0.08)
    a0 = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    a0.tick_params(labelbottom=False)

    e = np.array([T.get(R, n)[1] for n in names])
    v0 = np.array([T.get(R, n)[0] for n in names])
    a0.bar(x, np.where(e > 0, v0 / e, 0.0), color="#cccccc", width=0.85,
           label=f"{ref} (no kernel), pull against truth 0")
    a0.axhline(0.0, color="k", lw=1)
    a0.set_ylabel("pull")
    a0.set_yscale("symlog", linthresh=10)
    a0.legend(fontsize=13, loc="upper left")

    for tag, r in rows.items():
        if tag == ref:
            continue
        v = np.array([T.get(r, n)[0] for n in names])
        a1.step(x, np.where(e > 0, (v - v0) / e, 0.0), where="mid", lw=1.8,
                color=COL.get(tag, None), label=f"{tag} - {ref}")
    a1.axhline(0.0, color="k", lw=1)
    for ax in (a0, a1):
        ax.axvline(nb - 0.5, color="grey", lw=1.4, ls="--")
    a1.set_ylabel(r"$(\theta - \theta_{\rm ref})/\sigma_\theta$")
    a1.set_xlabel("calibration parameter")
    a1.set_xticks([nb / 2, nb + (len(names) - nb) / 2])
    a1.set_xticklabels(["50 B-field modes", "42 material amounts"])
    a1.legend(fontsize=13, loc="upper left")
    pubhtml.savefig(fig, os.path.join(out, "fsrfit_shift.pdf"))
    plt.close(fig)


def fig_dmean(rows, dbar, dnames, out):
    dm = T.dmean(rows, dbar, dnames)
    tags = list(rows)
    y = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(10, 1.1 * len(tags) + 3))
    ax.barh(y, [dm[t] for t in tags],
            color=[COL.get(t, "#1f77b4") for t in tags], height=0.6)
    for i, t in enumerate(tags):
        k = DM_KERNEL.get(t)
        if k:
            ax.plot([-k], [i], marker="|", ms=26, mew=2.5, color="k",
                    zorder=5,
                    label=(r"$-\langle dm\rangle$ of the kernel"
                           if i == 0 or "lab" not in dir() else None))
        ax.text(dm[t] + (0.06 if dm[t] >= 0 else -0.06), i,
                f"{dm[t]:+.3f} MeV", va="center",
                ha="left" if dm[t] >= 0 else "right", fontsize=13)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\langle D_{\rm card}\rangle\cdot\theta$  [MeV]"
                  "\n(the mean predicted J/$\\psi$ mass shift the fit produces)")
    lo = min(list(dm.values()) + [-v for v in DM_KERNEL.values()])
    hi = max(list(dm.values()) + [0.0])
    ax.set_xlim(lo - 1.2, hi + 1.2)
    h, lab = ax.get_legend_handles_labels()
    if h:
        ax.legend(h[:1], lab[:1], fontsize=13, loc="lower right")
    pubhtml.savefig(fig, os.path.join(out, "fsrfit_dmean.pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rows", nargs="+", metavar="TAG=JSON")
    p.add_argument("--ref", default="J0")
    p.add_argument("--dbar", default=os.path.join(HERE, "runs",
                                                  "jpsi_Dbar.npz"))
    p.add_argument("-o", "--outdir", default=None)
    a = p.parse_args()
    out = a.outdir or pubhtml.figdir("jpsi_fsr")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)
    rows = {}
    for spec in a.rows:
        tag, path = spec.split("=", 1)
        if os.path.exists(path):
            rows[tag] = T.load(path)
        else:
            logger.warning(f"{tag}: {path} is not there")
    if a.ref not in rows:
        raise SystemExit(f"the reference row {a.ref} is not among {list(rows)}")
    fig_shift(rows, a.ref, out)
    z = np.load(a.dbar, allow_pickle=True)
    fig_dmean(rows, np.asarray(z["dbar"], float),
              [str(s) for s in z["names"]], out)
    logger.info(f"figures -> {out}")


if __name__ == "__main__":
    main()
