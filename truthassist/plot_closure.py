#!/usr/bin/env python3
"""Single-track closure of the truth-assisted corrections.

Reads runCvhTruthAssisted.py mode=closure outputs (free single-track CVH fits of
gen-matched muons on realistic-alignment MC) and the solve_truth.py result, and
plots the curvature response

    (q/p)_reco / (q/p)_gen   vs   q*p_gen

without corrections and with the truth-assisted corrections applied linearly
through the stored reference-point jacobian:

    (q/p)_cor = (q/p)_ref + sum_j jacrefv[0, j] * x[globalidxv[j]]

Per bin the response is a 2.5-sigma clipped mean (the curvature ratio is
unbiased under Gaussian curvature resolution; clipping removes the
non-Gaussian tails identically for both variants). One file per panel: each
sample (J/psi, Z) and each |eta| region, plus both samples together.

usage:
  ~/.claude/skills/cms-plots/.venv/bin/python plot_closure.py \
      --cor truthcor.npz --jpsi '/.../jpsi_closure/task_*/globalcor_truth_*.root' \
      --z '/.../z_closure/task_*/globalcor_truth_*.root'
"""
import argparse
import datetime
import glob
import os

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

BRANCHES = ["refParms", "genParms", "genEta", "nValidHits", "nValidPixelHits",
            "edmvalref", "nParms", "jacrefv", "globalidxv"]

QP_EDGES = np.array([-400, -150, -80, -50, -35, -25, -18, -13, -10, -8, -6, -4.5, -3,
                     3, 4.5, 6, 8, 10, 13, 18, 25, 35, 50, 80, 150, 400], dtype=float)
ETA_REGIONS = [("all", 0., 2.4), ("barrel", 0., 0.9), ("overlap", 0.9, 1.5), ("endcap", 1.5, 2.4)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cor", required=True, help="solve_truth.py output npz")
    p.add_argument("--jpsi", default=None, help="glob of J/psi closure files")
    p.add_argument("--z", default=None, help="glob of Z closure files")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-edm", type=float, default=1e-2, help="convergence gate on edmvalref")
    p.add_argument("--truth-ideal", default=None,
                   help="runtree file of an IDEAL-geometry run with the same catalogue: "
                        "adds the fitted-alignment vs true-misalignment plots")
    p.add_argument("--truth-real", default=None, help="runtree file of the matching REALISTIC run")
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="Work in progress")
    return p.parse_args()


def default_outpath(script_path):
    stem = os.path.splitext(os.path.basename(script_path))[0]
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/calibration_studies/{today}_truthassist_{stem}/")


def load(pattern, x, max_files, max_edm):
    files = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    out = {k: [] for k in ("qop", "qopgen", "cor", "eta")}
    ntot = 0
    for fn in files:
        t = uproot.open(fn)["tree"]
        a = t.arrays(BRANCHES)
        ntot += len(a)
        sel = ((a.nValidHits >= 9) & (a.nValidPixelHits > 0) & (a.edmvalref < max_edm)
               & (a.edmvalref >= 0))
        a = a[sel]
        jac = a.jacrefv
        jq = jac[ak.local_index(jac, axis=1) < a.nParms]            # row 0 = d(q/p)/d(a)
        idx = a.globalidxv
        xv = ak.unflatten(x[ak.to_numpy(ak.flatten(idx)).astype(np.int64)], ak.num(idx))
        out["cor"].append(ak.to_numpy(ak.sum(jq * xv, axis=1)))
        out["qop"].append(ak.to_numpy(a.refParms[:, 0]))
        out["qopgen"].append(ak.to_numpy(a.genParms[:, 0]))
        out["eta"].append(ak.to_numpy(a.genEta))
    d = {k: np.concatenate(v) for k, v in out.items()}
    logger.info(f"{pattern}: {len(files)} files, {ntot} tracks, {len(d['qop'])} selected")
    return d


def clipped_mean(r, nsig=2.5, niter=10):
    if len(r) < 20:
        return np.nan, np.nan
    m, s = np.median(r), 1.4826 * np.median(np.abs(r - np.median(r)))
    for _ in range(niter):
        w = r[np.abs(r - m) < nsig * s]
        if len(w) < 20:
            return np.nan, np.nan
        m_new, s = w.mean(), w.std()
        if abs(m_new - m) < 1e-3 * s / np.sqrt(len(w)):
            m = m_new
            break
        m = m_new
    return m, s / np.sqrt(len(w))


def response(d, emin, emax):
    qp = 1. / d["qopgen"]
    sel = (np.abs(d["eta"]) >= emin) & (np.abs(d["eta"]) < emax)
    res = {}
    for key, qop in (("nocor", d["qop"]), ("cor", d["qop"] + d["cor"])):
        r = qop / d["qopgen"]
        vals = []
        for lo, hi in zip(QP_EDGES[:-1], QP_EDGES[1:]):
            m = sel & (qp >= lo) & (qp < hi)
            vals.append(clipped_mean(r[m]))
        res[key] = np.array(vals)
    return res


def plot_response(outdir, name, args, series, label_extra):
    fig, ax = plt.subplots(figsize=(10, 7))
    centers = 0.5 * (QP_EDGES[:-1] + QP_EDGES[1:])
    xerr = 0.5 * (QP_EDGES[1:] - QP_EDGES[:-1])
    styles = {"nocor": dict(marker="o", color="#1f77b4", mfc="none", label="no corrections"),
              "cor": dict(marker="s", color="#d62728", label="truth-assisted corrections")}
    for sample, res, off in series:
        for key in ("nocor", "cor"):
            m, e = res[key][:, 0], res[key][:, 1]
            st = dict(styles[key])
            pretty = sample.replace("J/psi", r"J/$\psi$")
            st["label"] = f"{pretty}, {st['label']}"
            if sample.startswith("Z"):
                st["marker"] = "^" if key == "nocor" else "D"
            ax.errorbar(centers * (1 + off), (m - 1.) * 1e3, xerr=xerr, yerr=e * 1e3,
                        linestyle="none", markersize=6, **st)
    ax.axhline(0., color="black", lw=1)
    ax.set_xscale("symlog", linthresh=3., linscale=0.25)
    ax.set_xlim(QP_EDGES[0], QP_EDGES[-1])
    ticks = [-300, -100, -30, -10, -3, 3, 10, 30, 100, 300]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.7 * (hi - lo))
    ax.set_xlabel(r"$q \cdot p_{\mathrm{gen}}$ (GeV)")
    ax.set_ylabel(r"$(q/p)_{\mathrm{reco}} / (q/p)_{\mathrm{gen}} - 1$ ($\times 10^{-3}$)")
    ax.legend(loc="upper right", fontsize=13, title=label_extra, title_fontsize=13)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None, loc=2)
    full = "_".join(filter(None, [name, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, full)
    output_tools.write_logfile(outdir, full, args=args, wd=os.path.dirname(os.path.abspath(__file__)))
    plt.close(fig)


def truth_plots(outdir, args, x):
    """Fitted alignment translations vs the true misalignment (realistic - ideal
    module position, projected on the local axes of the ideal module). With the
    residual convention r + J a (J = +1 on the measured coordinate) a module
    displaced by +d needs a = -d."""
    def cat(fn):
        rt = uproot.open(fn)["runtree"]
        return rt.arrays(["parmtype", "subdet", "x", "y", "z", "lxx", "lxy", "lxz",
                          "lyx", "lyy", "lyz"], library="np")
    a, b = cat(args.truth_ideal), cat(args.truth_real)
    names = {0: "BPix", 1: "FPix", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}
    for pt, (ax_, lbl) in {0: (("lxx", "lxy", "lxz"), "local x"), 1: (("lyx", "lyy", "lyz"), "local y")}.items():
        sel = a["parmtype"] == pt
        d = np.stack([(b[k] - a[k])[sel] for k in ("x", "y", "z")], 1)
        u = np.stack([a[k][sel] for k in ax_], 1)
        true_shift = -np.sum(d * u, 1) * 1e4           # expected correction (um)
        fit = x[np.flatnonzero(sel)] * 1e4
        sub = a["subdet"][sel]
        fig, ax = plt.subplots(figsize=(8, 8))
        for s, n in names.items():
            m = sub == s
            if m.any() and np.any(fit[m] != 0):
                ax.plot(true_shift[m], fit[m], ".", ms=2, alpha=0.5, label=n)
        lim = max(5., np.quantile(np.abs(true_shift), 0.99) * 1.2)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"expected correction, {lbl} ($\\mu$m)")
        ax.set_ylabel(f"fitted correction, {lbl} ($\\mu$m)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=13, markerscale=5)
        plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None, loc=2)
        full = "_".join(filter(None, [f"alignment_vs_truth_parmtype{pt}", args.postfix]))
        plot_tools.save_pdf_and_png(outdir, full)
        output_tools.write_logfile(outdir, full, args=args, wd=os.path.dirname(os.path.abspath(__file__)))
        plt.close(fig)
        m = fit != 0
        if m.any():
            cc = np.corrcoef(true_shift[m], fit[m])[0, 1]
            slope = np.sum(true_shift[m] * fit[m]) / np.sum(true_shift[m] ** 2)
            logger.info(f"parmtype {pt}: n={m.sum()} corr={cc:.3f} slope={slope:.3f} "
                        f"rms(fit-true)={np.sqrt(np.mean((fit[m]-true_shift[m])**2)):.2f} um "
                        f"rms(true)={np.sqrt(np.mean(true_shift[m]**2)):.2f} um")


def main():
    args = parse_args()
    outdir = output_tools.make_plot_dir(args.outpath or default_outpath(__file__))
    logger.info(f"Writing plots to {outdir}")
    x = np.load(args.cor)["x"]
    samples = []
    if args.jpsi:
        samples.append(("J/psi", load(args.jpsi, x, args.max_files, args.max_edm), -0.02))
    if args.z:
        samples.append(("Z", load(args.z, x, args.max_files, args.max_edm), 0.02))
    summary = []
    for rname, emin, emax in ETA_REGIONS:
        lab = f"{emin:g} < |$\\eta$| < {emax:g}"
        allres = []
        for sname, d, off in samples:
            res = response(d, emin, emax)
            allres.append((sname, res, off))
            plot_response(outdir, f"closure_{sname.replace('/', '')}_{rname}", args,
                          [(sname, res, 0.)], lab)
            for key in ("nocor", "cor"):
                m = res[key][:, 0]
                ok = np.isfinite(m)
                summary.append(f"{sname:6s} {rname:8s} {key:6s} rms(response-1) over bins "
                               f"= {1e3*np.sqrt(np.nanmean((m[ok]-1)**2)):.3f}e-3")
        plot_response(outdir, f"closure_both_{rname}", args, allres, lab)
    for l in summary:
        logger.info(l)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    if args.truth_ideal and args.truth_real:
        truth_plots(outdir, args, x)


if __name__ == "__main__":
    main()
