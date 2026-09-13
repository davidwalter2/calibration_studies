#!/usr/bin/env python3
"""Figures for the STANDARD-SELECTION study (`STATE.md` section 14).

One file per panel, a PNG twin beside every PDF (`resolution/pubhtml.savefig`),
`hep.style.ROOT` + `wums`.

Panels
  `zv_spectrum_<tag>`      the `|z_v|` spectrum with the cut marked, and the
                           fraction beyond it -- what the truncation removes
  `cutflow_dy`             the gen-class composition through the cut flow,
                           `chi2/ndof < 3` in its place
  `shift_<comparison>`     every parameter's shift in units of its own error,
                           for (a) vs (b), (c) vs (b) and the mass term

usage:
  python3 sel_plots.py --npz <gun vtx.npz> --dy <dy_vtxon_gen_vtx_all.npz> \
      --fits <R>/fits --groups <materialGroups50.txt>
"""
import argparse
import datetime
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres"),
           os.path.join(_RES, "hitlik")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selection  # noqa: E402
import pubhtml  # noqa: E402
import sel_report as SR  # noqa: E402
import recovery as RC  # noqa: E402  (read_fit)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
from wums import logging  # noqa: E402

logger = logging.child_logger(__name__)
plt.style.use(hep.style.ROOT)


def zv_spectrum(z, tag, outdir, cut=selection.MAX_ABS_VTXZ):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = np.linspace(0, 12, 121)
    h, _ = np.histogram(np.abs(z), bins=edges)
    # empty bins are MASKED, not floored: a log axis with a 1e-9 floor draws
    # spikes to the bottom of the frame and hides the shape
    y = np.where(h > 0, h.astype(float), np.nan)
    ax.step(edges[:-1], y, where="post", lw=1.6,
            label=f"{tag} ({len(z)} candidates)")
    ax.set_ylim(0.5, max(h.max() * 2.0, 2.0))
    ax.axvline(cut, color="C3", ls="--", lw=2,
               label=f"$|z_v| < {cut:g}$ (standard selection)")
    frac = float((np.abs(z) >= cut).mean())
    ax.set_yscale("log")
    ax.set_xlabel(r"$|z_v| = |r_v| / \sigma_v$")
    ax.set_ylabel("candidates / 0.1")
    ax.set_title(f"beyond the cut: {100*frac:.3f} %", fontsize=14)
    ax.legend(fontsize=13)
    pubhtml.savefig(fig, os.path.join(outdir, f"zv_spectrum_{tag}.pdf"))
    plt.close(fig)
    return frac


def cutflow_dy(npz, outdir, args):
    import genbkg
    d = genbkg.load_npz(npz, max_chi2_ndof=0.0)
    n0 = len(np.asarray(d["vtxz"]))
    cls, names, _ = genbkg.classify(d)
    steps, keep = [], np.ones(n0, bool)
    steps.append(("all", keep.copy()))
    for label, kw in (("vtxok +\nfinite",
                       dict(max_abs_vtxz=0, min_leg_hits=0, max_chi2_ndof=0)),
                      ("leg hits\n$\\geq 8$",
                       dict(max_abs_vtxz=0, max_chi2_ndof=0)),
                      ("$\\chi^2/\\mathrm{ndof} < 3$", dict(max_abs_vtxz=0)),
                      ("$|z_v| < 5$", {})):
        m, _s = selection.standard(d, args, n=n0, **kw)
        keep = keep & m
        steps.append((label, keep.copy()))

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(steps))
    bot = np.zeros(len(steps))
    for i, c in enumerate(names):
        if c == "signal":
            continue
        y = np.array([int((m & (cls == i)).sum()) for _, m in steps], float)
        if y.max() == 0:
            continue
        ax.bar(x, y, bottom=bot, label=c, width=0.6)
        bot += y
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=12)
    ax.set_ylabel("background candidates")
    ax.set_title("DY, gen-classified: what the standard selection removes\n"
                 "(the number above each bar is the SIGNAL count)",
                 fontsize=13)
    ax.set_ylim(0, bot.max() * 1.30)
    ax.legend(fontsize=13, loc="upper right")
    for xi, (_, m) in zip(x, steps):
        ax.text(xi, bot[xi] + 0.02 * bot.max(),
                f"{int((m & (cls == names.index('signal'))).sum())}",
                ha="center", fontsize=11, color="C0")
    pubhtml.savefig(fig, os.path.join(outdir, "cutflow_dy.pdf"))
    plt.close(fig)


def shift_panel(rows, name, title, outdir):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[0])
    lbl = [r[0].replace("material_", "mat ").replace("hitres_", "hit ")
           for r in rows]
    ns = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, ns, color=np.where(np.abs(ns) > 0.2, "C3", "C0"))
    for t in (-0.2, 0.2):
        ax.axvline(t, color="0.4", ls=":", lw=1.2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=8)
    ax.set_xlabel(r"shift / $\sigma$ of the parameter")
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-1, len(rows))
    pubhtml.savefig(fig, os.path.join(outdir, f"shift_{name}.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="the gun vertex npz")
    ap.add_argument("--dy", default=None, help="the gen-classified DY npz")
    ap.add_argument("--fits", default=None)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--outdir", default=None)
    selection.add_args(ap)
    a = ap.parse_args()

    outdir = a.outdir or os.path.expanduser(
        "~/public_html/ZMass/cvh/"
        + datetime.date.today().strftime("%y%m%d") + "_selection")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir)
    logger.info(f"figures -> {outdir}")

    if a.npz:
        d = np.load(a.npz, allow_pickle=True)
        f = zv_spectrum(d["vtxz"], "gun", outdir)
        logger.info(f"gun: P(|z_v| > 5) = {f:.5f}")
    if a.dy:
        cutflow_dy(a.dy, outdir, a)
        import genbkg
        dd = genbkg.load_npz(a.dy, max_chi2_ndof=0.0)
        f = zv_spectrum(dd["vtxz"], "dy", outdir)
        logger.info(f"DY: P(|z_v| > 5) = {f:.5f}")

    if a.fits:
        import groups as G
        gnames, _ = G.group_param_names(42, a.groups)
        unit_of = dict(zip(gnames, G.card_group_units(len(gnames), a.groups)))
        P = {}
        for nm in ("vtx_a_nocut", "vtx_b_cut_norm", "vtx_c_cut_nonorm",
                   "mass_a_nocut", "mass_b_cut_norm"):
            path = os.path.join(a.fits, nm, "fitresults.hdf5")
            if os.path.exists(path):
                P[nm] = SR.physical(RC.read_fit(path), unit_of)
        hits = sorted({k for v in P.values() for k in v
                       if k.startswith("hitres_")})
        names = gnames + hits
        if "vtx_a_nocut" in P and "vtx_b_cut_norm" in P:
            shift_panel(SR.compare(P["vtx_a_nocut"], P["vtx_b_cut_norm"],
                                   names, "(a) no cut -> (b) cut + norm",
                                   unit_of, top=0),
                        "vtx_a_vs_b",
                        "vertex term, (cut + truncated norm) $-$ (no cut)",
                        outdir)
        if "vtx_b_cut_norm" in P and "vtx_c_cut_nonorm" in P:
            shift_panel(SR.compare(P["vtx_b_cut_norm"], P["vtx_c_cut_nonorm"],
                                   names, "(b) -> (c) normalisation left out",
                                   unit_of, sigma_mode="same", top=0),
                        "vtx_b_vs_c",
                        "vertex term, (cut, NO norm) $-$ (cut + norm): "
                        "the bias", outdir)
        if "mass_a_nocut" in P and "mass_b_cut_norm" in P:
            shift_panel(SR.compare(P["mass_a_nocut"], P["mass_b_cut_norm"],
                                   ["alpha"] + names, "mass: (a) -> (b)",
                                   unit_of, top=0),
                        "mass_a_vs_b",
                        r"mass term, (with $|z_v|<5$) $-$ (without)", outdir)


if __name__ == "__main__":
    main()
