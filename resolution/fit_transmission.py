"""Fit-transmission study: how strongly does the FITTED q/p respond to the
TRUE per-track energy-loss fluctuation?

Motivation: the +0.2e-3 peak-vs-mean split of the mass/qop is real (clean
propagation test: median z grows to +5 sigma_trunc through the tracker) but
the linear-response model transmits almost none of it (correct transmitted
skew is tiny). Either the fit genuinely averages loss fluctuations away, or
the true response is much larger than the influence-weight prediction.

Method (rung-E sample, sim-hit-position refits, simPabsFirst/Last export):
  Delta-E_true = E(first sim hit) - E(last sim hit)   [in-tracker loss]
  dqop = qop_fit - qop_gen
  In bins of (gen pT, |eta|): robust slope  s = d(dqop)/d(Delta-E_true)
  Transmission T = s / (q * E/p^3)  -- T = 1 would mean the fitted momentum
  fully tracks the actual post-loss trajectory (mean-corrected); T = 0 means
  the fit rejects loss fluctuations entirely. The lever-arm-weighted
  expectation for a distributed loss lies between.

usage: python fit_transmission.py [--files GLOB] [--ntasks N]
"""

import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

MMU = 0.1056584


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_simhitp02b/task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def robust_slope(x, y):
    """Least-squares slope after 3-sigma clipping on the residual (two
    passes); returns slope, error, n."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    for _ in range(2):
        if len(x) < 50:
            return np.nan, np.nan, len(x)
        A = np.vstack([x - x.mean(), np.ones_like(x)]).T
        c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c
        keep = np.abs(r) < 3. * r.std()
        x, y = x[keep], y[keep]
    n = len(x)
    xc = x - x.mean()
    s = np.sum(xc * (y - y.mean())) / np.sum(xc ** 2)
    serr = (y - y.mean() - s * xc).std() / np.sqrt(np.sum(xc ** 2))
    return s, serr, n


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_transmission/")
    os.makedirs(outdir, exist_ok=True)

    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    cols = {k: [] for k in ("dqop", "dE", "pgen", "eta", "q")}
    for fn in files:
        f = uproot.open(fn)
        t = f["tree"]
        a = t.arrays(["refParms", "genParms", "genPt", "genEta", "genCharge",
                      "simPabsFirst", "simPabsLast"], library="np")
        for ic in range(len(a["genPt"])):
            qg = a["genParms"][ic][0]
            pF, pL = a["simPabsFirst"][ic], a["simPabsLast"][ic]
            if qg == 0. or pF <= 0. or pL <= 0. or pL > pF:
                continue
            cols["dqop"].append(a["refParms"][ic][0] - qg)
            cols["dE"].append(np.sqrt(pF**2 + MMU**2) - np.sqrt(pL**2 + MMU**2))
            cols["pgen"].append(1. / abs(qg))
            cols["eta"].append(abs(a["genEta"][ic]))
            cols["q"].append(np.sign(qg))
        logger.info(f"{fn.split('/')[-2]}: {len(cols['dqop'])} tracks")
    d = {k: np.array(v) for k, v in cols.items()}
    n = len(d["dqop"])
    logger.info(f"total {n} tracks; median dE = {1e3*np.median(d['dE']):.1f} MeV")

    # transmission per (pT, |eta|) bin: slope of q*dqop vs dE, normalized
    # by E/p^3 at the bin's median momentum
    ptedges = [2, 4, 6, 8, 12, 30]
    etaedges = [0, 0.9, 1.5, 2.0, 2.4]
    ptgen = d["pgen"] / np.cosh(d["eta"])
    results = []
    for ie in range(len(etaedges) - 1):
        for ip in range(len(ptedges) - 1):
            m = ((d["eta"] >= etaedges[ie]) & (d["eta"] < etaedges[ie + 1])
                 & (ptgen >= ptedges[ip]) & (ptgen < ptedges[ip + 1]))
            if m.sum() < 300:
                continue
            s, serr, nn = robust_slope(d["dE"][m], d["q"][m] * d["dqop"][m])
            pmed = np.median(d["pgen"][m])
            Emed = np.sqrt(pmed**2 + MMU**2)
            full = Emed / pmed**3
            results.append((etaedges[ie], etaedges[ie + 1], ptedges[ip],
                            ptedges[ip + 1], s / full, serr / full, nn))
            logger.info(f"|eta| {etaedges[ie]}-{etaedges[ie+1]} pT {ptedges[ip]}-{ptedges[ip+1]}: "
                        f"T = {s/full:+.3f} +- {serr/full:.3f}  (n={nn}, p_med={pmed:.1f})")
    res = np.array([r[:6] for r in results])
    Tmean = np.average(res[:, 4], weights=1. / res[:, 5] ** 2)
    Terr = 1. / np.sqrt(np.sum(1. / res[:, 5] ** 2))
    logger.info(f"WEIGHTED-MEAN TRANSMISSION T = {Tmean:+.4f} +- {Terr:.4f}")
    logger.info("(T=1: fitted momentum fully tracks the actual loss; T=0: fit "
                "rejects loss fluctuations; the linear-response model predicts "
                "a specific small value -- the measured T tells the truth)")

    fig, ax = plt.subplots(figsize=(10, 7))
    for ie in range(len(etaedges) - 1):
        sel = [r for r in results if r[0] == etaedges[ie]]
        if not sel:
            continue
        xs = [0.5 * (r[2] + r[3]) for r in sel]
        ys = [r[4] for r in sel]
        es = [r[5] for r in sel]
        ax.errorbar(xs, ys, es, marker="o", linestyle="-",
                    label=rf"$|\eta| \in [{etaedges[ie]}, {etaedges[ie+1]}]$")
    ax.axhline(1., color="gray", ls="--", lw=1)
    ax.axhline(0., color="gray", lw=1)
    ax.set_xlabel(r"gen $p_T$ [GeV]")
    ax.set_ylabel(r"transmission $T = \frac{d(q\,\delta q/p)}{d(\Delta E)} \,/\, (E/p^3)$")
    ax.legend(fontsize="small")
    name = f"fit_transmission{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"wrote {outdir}/{name}")


if __name__ == "__main__":
    main()
