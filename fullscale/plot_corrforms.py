#!/usr/bin/env python3
"""The two correction forms, side by side, on the real Z candidates.

Panel per file, into `~/public_html/cvh/<YYMMDD>_fullscale/`:

 20  what each form DISPLACES the model by, per candidate.  The residual form
     is fed `delta_i = m_i - M(theta)` and its exact Jensen map moves the
     residual by `|u_i m_i - delta_i|`; the fluctuation form's deterministic
     part is `|E[Delta_i]| = |Var(x)(c_i - a_i sigma_i) + d_i|`.  The line at
     1.5 s^2 m = 20.6 MeV is the mean shift the Jensen term exists to apply.
 21  the pieces of the fluctuation form: `c_i`, `d_i`, `-a_i sigma_i` and their
     sum, showing the cancellation `c_i = -vgf_i sigma_i^2/m_i`.
 22  the clip scan: `m_Z` and `Gamma_Z` against `corr_clip`, i.e. the
     measurement that the stopgap is not stable in its own knob.

usage:
  ./run_tf.sh python3 plot_corrforms.py --card cards/z_full380_fl.hdf5 \
      --clip-results 'results/fit_n300k_*.json'
"""
import argparse
import datetime
import glob
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--clip-results", default=None)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="work in progress")
    p.add_argument("--titlePos", type=int, default=2)
    return p.parse_args()


def outpath(args):
    if args.outpath:
        return os.path.expanduser(args.outpath)
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/cvh/{today}_fullscale/")


def save(outdir, name, args):
    name = "_".join(filter(None, [name, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    plt.close("all")


def main():
    import h5py
    from rabbit import unbinned

    args = parse_args()
    outdir = outpath(args)
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(args.card, "r") as f:
        t = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
    sig = t.sigma.numpy()
    delta = t.mobs.numpy()
    m = delta + t.m_ref
    w = np.ones(t.n) if t.weights is None else t.weights.numpy()
    s2 = t.jensen_s2.numpy()
    c = t._fl_g.numpy() * sig
    d = np.zeros(t.n) if t._fl_d is None else t._fl_d.numpy()
    asig = t._fl_a.numpy() * sig
    shift = c - asig + d                       # E[Delta_i], Var(x) = 1

    # what the residual form's exact Jensen map does to the same candidates
    r = delta / m
    disc = np.maximum(1.0 + 4.0 * (r - 0.5 * s2), 0.1)
    u = 0.5 * (np.sqrt(disc) - 1.0)
    moved = np.abs(u * m - delta)

    # ---- 20. the displacement each form applies --------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    b = np.logspace(-1, 4.2, 180)
    ax.hist(1e3 * moved, bins=b, weights=w, histtype="step", color="tab:red",
            lw=1.8, label=r"residual form: $|u_i m_i - \delta_i|$")
    ax.hist(1e3 * np.abs(shift), bins=b, weights=w, histtype="step",
            color="tab:blue", lw=1.8,
            label=r"fluctuation form: $|\mathrm{E}[\Delta_i]|$")
    ax.axvline(1e3 * 1.5 * np.median(s2) * t.m_ref, color="k", ls="--", lw=1.2,
               label=r"$1.5\,s^2 m$ (the shift it must apply)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("displacement of the model [MeV]")
    ax.set_ylabel("weighted candidates")
    ax.set_ylim(0.5, 1e7)
    ax.legend(fontsize=12, loc="upper right")
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "20_correction_displacement", args)
    logger.info(
        f"residual form: median {1e3*np.median(moved):.1f} MeV, "
        f"q99 {1e3*np.quantile(moved, 0.99):.0f} MeV, max {1e3*moved.max():.0f} MeV; "
        f"fluctuation form: median {1e3*np.median(np.abs(shift)):.1f} MeV, "
        f"q99 {1e3*np.quantile(np.abs(shift), 0.99):.0f} MeV")

    # ---- 21. the pieces of the fluctuation map ---------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    b = np.linspace(-60, 40, 201)
    for arr, lab, col in (
            (c, r"$c_i = -\mathrm{vgf}_i\,\sigma_i^2/m_i$", "tab:green"),
            (d, r"$d_i = m_i s_i^2/2$", "tab:orange"),
            (-asig, r"$-a_i\sigma_i$", "tab:purple"),
            (shift, r"$\mathrm{E}[\Delta_i]$ (the sum)", "k")):
        ax.hist(1e3 * arr, bins=b, weights=w, histtype="step", lw=1.8,
                color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("[MeV]")
    ax.set_ylabel("weighted candidates / 0.5 MeV")
    ax.set_ylim(top=ax.get_ylim()[1] * 30)
    ax.legend(fontsize=12)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "21_fluctuation_pieces", args)

    # ---- 22. the clip scan ------------------------------------------------
    if args.clip_results:
        pts = []
        for f in sorted(glob.glob(args.clip_results)):
            with open(f) as fh:
                res = json.load(fh)
            lab = res.get("label", "")
            cl = res.get("corr_clip")
            if "clip" in lab:
                cl = float(lab.split("clip")[-1])
            elif cl in (None, 0.0) and lab.endswith("base"):
                cl = np.inf
            else:
                continue
            i, j = res["params"].index("m_Z"), res["params"].index("Gamma_Z")
            e = res.get("sandwich_err") or res.get("err")
            pts.append((cl, res["fitted"][i], e[i], res["fitted"][j], e[j]))
        if pts:
            pts.sort()
            x = np.array([p[0] for p in pts])
            xp = np.where(np.isfinite(x), x, 20.0)     # inf drawn at 20
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.errorbar(xp, [1e0 * p[1] for p in pts],
                        yerr=[p[2] for p in pts], marker="o", color="tab:red",
                        label=r"$m_Z$ fitted $-$ generator")
            ax.errorbar(xp, [1e0 * p[3] for p in pts],
                        yerr=[p[4] for p in pts], marker="s", color="tab:blue",
                        label=r"$\Gamma_Z$ fitted $-$ generator")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xlabel(r"$\mathtt{corr\_clip}$  [$\sigma_i$]  "
                          r"($20 = $ no clip)")
            ax.set_ylabel("fitted $-$ generator [MeV]")
            ax.set_yscale("symlog", linthresh=100)
            ax.legend(fontsize=12)
            plot_tools.add_decor(ax, args.title, args.subtitle, data=False,
                                 lumi=None, loc=args.titlePos)
            save(outdir, "22_clip_scan", args)
            logger.info(f"clip scan points: {[(p[0], round(p[1],1)) for p in pts]}")


if __name__ == "__main__":
    main()
