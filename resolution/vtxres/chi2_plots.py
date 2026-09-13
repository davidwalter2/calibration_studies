#!/usr/bin/env python3
"""Figures for the CHI2/NDOF cut study (`STATE.md` section 15.8).

One file per panel, a PNG twin beside every PDF (`resolution/pubhtml.savefig`),
`hep.style.ROOT` + `wums`.

Panels
  `chi2_spectrum_<tag>`    the reduced-chi2 spectrum with the three cut values
                           marked and the fraction beyond each, with the
                           MODEL's own `chi2_ndof` density under it -- the
                           picture of why the cut needs no truncation factor
                           and why the sample does not match the model
  `chi2_cutflow_dy`        the gen-class composition through the cut flow, the
                           chi2 cut in its place
  `chi2_shift_<channel>`   every parameter's shift between the no-cut arm and
                           the cut arm, in units of its own error
  `chi2_alpha`             the fitted mass scale against the cut value, in MeV

usage:
  python3 chi2_plots.py --gun <gun vtx.npz> --dy <dy gen npz> \
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
import recovery as RC  # noqa: E402
from chi2_report import MREF, ARMS, ARMLAB  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
from wums import logging  # noqa: E402

logger = logging.child_logger(__name__)
plt.style.use(hep.style.ROOT)

CUTS = (3.0, 5.0, 10.0)


def chi2_spectrum(q, ndof, tag, outdir):
    """The measured reduced chi2 against the chi2_k/k density the FIT implies.

    The model curve is not a fit: it is the density the covariance the terms
    condition on would produce if it were right, at the sample's own median
    `ndof`.  The gap between the two IS the unmodelled per-candidate noise
    scale of section 16, and it is what the cut removes.
    """
    q = np.asarray(q, float)
    q = q[np.isfinite(q) & (q > 0)]
    k = float(np.median(np.asarray(ndof, float)))
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = np.linspace(0, 6, 121)
    h, _ = np.histogram(q, bins=edges)
    y = np.where(h > 0, h.astype(float), np.nan)
    ax.step(edges[:-1], y, where="post", lw=1.6, color="C0",
            label=f"{tag} ({len(q)} candidates, median ndof {k:.0f})")
    # chi2_k / k, normalised to the same area
    from scipy import stats
    ctr = 0.5 * (edges[1:] + edges[:-1])
    pdf = stats.chi2.pdf(ctr * k, df=k) * k
    ax.plot(ctr, pdf * len(q) * (edges[1] - edges[0]), color="k", lw=1.6,
            ls="-", label=r"$\chi^2_k/k$ at the model covariance")
    for c, col in zip(CUTS, ("C3", "C1", "C2")):
        f = float((q >= c).mean())
        ax.axvline(c, color=col, ls="--", lw=1.8,
                   label=rf"$\chi^2/\mathrm{{ndof}} < {c:g}$"
                         f"  (removes {100*f:.2f} %)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(h.max() * 3.0, 2.0))
    ax.set_xlabel(r"$\chi^2/\mathrm{ndof}$")
    ax.set_ylabel("candidates / 0.05")
    ax.legend(fontsize=11)
    pubhtml.savefig(fig, os.path.join(outdir, f"chi2_spectrum_{tag}.pdf"))
    plt.close(fig)
    return {c: float((q >= c).mean()) for c in CUTS}


def cutflow_dy(npz, outdir, args):
    import genbkg
    d = genbkg.load_npz(npz, max_chi2_ndof=0.0)
    n0 = len(np.asarray(d["vtxz"]))
    cls, names, _ = genbkg.classify(d)
    names = list(names)
    steps, keep = [("all", np.ones(n0, bool))], np.ones(n0, bool)
    for label, kw in (("vtxok +\nfinite",
                       dict(max_abs_vtxz=0, min_leg_hits=0, max_chi2_ndof=0)),
                      ("leg hits\n$\\geq 8$",
                       dict(max_abs_vtxz=0, max_chi2_ndof=0)),
                      ("$\\chi^2/\\mathrm{ndof}$\n$< 3$", dict(max_abs_vtxz=0)),
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
    ax.set_xticklabels([s[0] for s in steps], fontsize=11)
    ax.set_ylabel("background candidates")
    ax.set_title("DY, gen-classified: the standard selection with the "
                 "$\\chi^2$ cut in its place\n"
                 "(the number above each bar is the SIGNAL count)",
                 fontsize=12)
    ax.set_ylim(0, bot.max() * 1.30)
    ax.legend(fontsize=13, loc="upper right")
    si = names.index("signal")
    for xi, (_, m) in zip(x, steps):
        ax.text(xi, bot[xi] + 0.02 * bot.max(),
                f"{int((m & (cls == si)).sum())}",
                ha="center", fontsize=11, color="C0")
    pubhtml.savefig(fig, os.path.join(outdir, "chi2_cutflow_dy.pdf"))
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
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-1, len(rows))
    pubhtml.savefig(fig, os.path.join(outdir, f"chi2_shift_{name}.pdf"))
    plt.close(fig)


def alpha_panel(P, outdir):
    """The fitted mass scale against the cut value, in MeV -- the number the
    mass measurement would see."""
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = {"off": 0.0, "c3": 3.0, "c5": 5.0, "c10": 10.0}
    any_row = False
    for samp, col in (("gun", "C0"), ("dy", "C3")):
        x, y, e = [], [], []
        for arm in ARMS:
            nm = f"{samp}_mass_{arm}"
            if nm in P and "alpha" in P[nm]:
                v, s = P[nm]["alpha"]
                x.append(xs[arm] if arm != "off" else 0.0)
                y.append(MREF[samp] * v)
                e.append(MREF[samp] * s)
        if not x:
            continue
        any_row = True
        ax.errorbar(x, y, yerr=e, marker="o", ls="-", color=col, capsize=4,
                    label=f"{samp}  ($m_\\mathrm{{ref}}$ = "
                          f"{MREF[samp]:.4f} GeV)")
    if not any_row:
        plt.close(fig)
        return
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks([0, 3, 5, 10])
    ax.set_xticklabels(["no cut", "< 3", "< 5", "< 10"])
    ax.set_xlabel(r"$\chi^2/\mathrm{ndof}$ cut")
    ax.set_ylabel(r"fitted mass scale $m_\mathrm{ref}\,\alpha$  [MeV]")
    ax.legend(fontsize=12)
    pubhtml.savefig(fig, os.path.join(outdir, "chi2_alpha.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gun", default=None, help="the gun vertex/mass npz")
    ap.add_argument("--dy", default=None, help="the gen-classified DY npz")
    ap.add_argument("--fits", default=None)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--outdir", default=None)
    selection.add_args(ap)
    a = ap.parse_args()

    outdir = a.outdir or os.path.expanduser(
        "~/public_html/ZMass/cvh/"
        + datetime.date.today().strftime("%y%m%d") + "_chi2cut")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir)
    logger.info(f"figures -> {outdir}")

    for tag, fn in (("gun", a.gun), ("dy", a.dy)):
        if not fn:
            continue
        d = np.load(fn, allow_pickle=True)
        f = chi2_spectrum(d["chi2ndof"], d["ndof"], tag, outdir)
        logger.info(f"{tag}: P(chi2/ndof >= cut) = "
                    + ", ".join(f"{c:g}: {100*v:.3f} %" for c, v in f.items()))
    if a.dy:
        cutflow_dy(a.dy, outdir, a)

    if a.fits:
        import groups as G
        gnames, _ = G.group_param_names(42, a.groups)
        unit_of = dict(zip(gnames, G.card_group_units(len(gnames), a.groups)))
        P = {}
        for nm in sorted(os.listdir(a.fits)):
            path = os.path.join(a.fits, nm, "fitresults.hdf5")
            if os.path.exists(path):
                P[nm] = SR.physical(RC.read_fit(path), unit_of)
        hits = sorted({k for v in P.values() for k in v
                       if k.startswith("hitres_")})
        names = list(gnames) + hits
        for samp in ("gun", "dy"):
            for chan in ("mass", "vtx"):
                base = f"{samp}_{chan}_off"
                if base not in P:
                    continue
                for arm in ("c3", "c5", "c10"):
                    nm = f"{samp}_{chan}_{arm}"
                    if nm not in P:
                        continue
                    rows = SR.compare(P[base], P[nm], names, f"{nm} vs {base}",
                                      unit_of, top=1)
                    shift_panel(rows, f"{samp}_{chan}_{arm}",
                                f"{samp} {chan}: {ARMLAB['off']} "
                                f"$\\rightarrow$ {ARMLAB[arm]}", outdir)
        alpha_panel(P, outdir)


if __name__ == "__main__":
    main()
