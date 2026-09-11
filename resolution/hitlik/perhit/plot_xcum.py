#!/usr/bin/env python3
"""Figures for the WG question: how the per-hit innovations depend on the
truth-referenced q/p pull -- the variable the MASS term is built from.

Two statements, two panels each, one file per panel:

* SECOND order they are EXACTLY uncorrelated (`F^T R = 0`), so
  `corr(z_k, z_qp)` must be zero in every hit-position bin -- plotted with its
  `1/sqrt(N)` band, as a null test.
* FOURTH order they are not: the cross-cumulant correlation
  `kappa(z_qp,z_qp,z_k,z_k)/sqrt(kappa4(qp) kappa4(k))` is 0.05-0.65 depending
  on where along the track the hit sits.  That is the size of what a
  PRODUCT-of-marginals (composite) likelihood over the two terms drops, and it
  is why the same-track joint must be checked with a sandwich.

usage: plot_xcum.py --npz runs/perhit/perhit20k.npz --outpath ~/public_html/...
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import mplhep as hep              # noqa: E402
import numpy as np                # noqa: E402

from wums import logging          # noqa: E402

# `pubhtml.savefig` writes the .png twin every .pdf needs to show up in the
# plot browser; it lives two directories up, in resolution/.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import pubhtml                    # noqa: E402

hep.style.use(hep.style.ROOT)


def qband(x, m, rel, edges):
    """median / p90 / N of x in each relpos bin."""
    out = []
    for b in range(len(edges) - 1):
        s = m & (rel >= edges[b]) & (rel < edges[b + 1] +
                                     (1e-9 if b == len(edges) - 2 else 0.0))
        s = s & np.isfinite(x)
        if s.sum() < 100:
            out.append((np.nan,) * 4)
            continue
        v = x[s]
        out.append((np.median(v), np.percentile(v, 90),
                    np.percentile(v, 10), int(s.sum())))
    return np.array(out, float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True)
    p.add_argument("--nbin", type=int, default=8)
    p.add_argument("--max-inflat", type=float, default=1e4)
    p.add_argument("--outpath", required=True)
    a = p.parse_args()
    log = logging.setup_logger(__file__, 3)
    os.makedirs(a.outpath, exist_ok=True)

    d = np.load(a.npz, allow_pickle=True)
    ck = np.asarray(d["ckind"])
    rel = np.asarray(d["relpos"])
    infl = np.asarray(d["inflat"])
    trk = np.asarray(d["trk"])
    comp = np.asarray(d["comp"])
    z = np.asarray(d["z"])
    xc0 = np.asarray(d["xc_ref0"])
    xcp = np.asarray(d["xc_prev"])
    hit = (ck == 0) & (infl < a.max_inflat)
    ref = ck == 1
    ntrk = int(len(d["eta"]))
    log.info(f"{ntrk} tracks, {hit.sum()} per-hit rows")

    # the q/p reference component = the FIRST reference slot of each track
    zref = np.full(ntrk, np.nan)
    ir = np.where(ref)[0]
    order = np.lexsort((comp[ir], trk[ir]))
    ir = ir[order]
    first = np.ones(len(ir), bool)
    first[1:] = trk[ir][1:] != trk[ir][:-1]
    zref[trk[ir][first]] = z[ir][first]

    edges = np.linspace(0.0, 1.0, a.nbin + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])

    # --- panel 1: the Gaussian-level null ----------------------------------
    cc, ee, nn = [], [], []
    for b in range(a.nbin):
        m = hit & (rel >= edges[b]) & (rel < edges[b + 1] +
                                       (1e-9 if b == a.nbin - 1 else 0.0))
        m &= np.isfinite(zref[trk])
        if m.sum() < 100:
            cc.append(np.nan); ee.append(np.nan); nn.append(0); continue
        cc.append(np.corrcoef(z[m], zref[trk][m])[0, 1])
        ee.append(1.0 / np.sqrt(m.sum()))
        nn.append(int(m.sum()))
    cc, ee = np.array(cc), np.array(ee)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axhline(0.0, color="k", lw=1.0)
    ax.fill_between(mid, -ee, ee, color="0.85", label=r"$\pm 1/\sqrt{N}$")
    ax.errorbar(mid, cc, yerr=ee, fmt="o", color="C0", ms=6,
                label=r"$\mathrm{corr}(z_k, z_{q/p})$, data")
    ax.set_xlabel("hit position along the track  (row / ($n_\\mathrm{meas}$-1))")
    ax.set_ylabel(r"$\mathrm{corr}(z_k,\, z_{q/p})$")
    ax.set_ylim(-0.06, 0.06)
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title("second order: the per-hit innovations are uncorrelated with "
                 "the q/p pull", fontsize=13)
    fn = os.path.join(a.outpath, "xcum_corr_zqp_vs_relpos.pdf")
    pubhtml.savefig(fig, fn)
    plt.close(fig)
    log.info(f"wrote {fn}")

    # --- panel 2: the fourth cross cumulant with q/p -----------------------
    q0 = qband(xc0, hit, rel, edges)
    qp = qband(xcp, hit, rel, edges)
    for tag, q, lab, ttl in (
        ("xcum_ref0_vs_relpos", q0,
         r"$\kappa(z_{q/p},z_{q/p},z_k,z_k)/\sqrt{\kappa_4\kappa_4}$",
         "fourth order: shared non-Gaussianity with the truth-referenced q/p"),
        ("xcum_adjacent_vs_relpos", qp,
         r"$\kappa(z_{k-1},z_{k-1},z_k,z_k)/\sqrt{\kappa_4\kappa_4}$",
         "fourth order: adjacent per-hit innovations")):
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        ax.fill_between(mid, q[:, 2], q[:, 1], color="C0", alpha=0.20,
                        label="10-90 %")
        ax.plot(mid, q[:, 0], "o-", color="C0", ms=6, label="median")
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_xlabel("hit position along the track  "
                      "(row / ($n_\\mathrm{meas}$-1))")
        ax.set_ylabel(lab)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=13)
        ax.set_title(ttl, fontsize=13)
        fn = os.path.join(a.outpath, tag + ".pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        log.info(f"wrote {fn}")

    # --- panel 3: by hit class --------------------------------------------
    cls = np.asarray(d["cls"])
    names = [str(s) for s in d["hit_classes"]]
    md, p9, lbl = [], [], []
    for c in range(len(names)):
        m = hit & (cls == c) & np.isfinite(xc0)
        if m.sum() < 100:
            continue
        md.append(np.median(xc0[m])); p9.append(np.percentile(xc0[m], 90))
        lbl.append(names[c])
    y = np.arange(len(lbl))
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(lbl) + 2.2))
    ax.barh(y, md, color="C0", height=0.62, label="median")
    ax.plot(p9, y, "k|", ms=14, label="p90")
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\kappa(z_{q/p},z_{q/p},z_k,z_k)/\sqrt{\kappa_4\kappa_4}$")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_title("shared non-Gaussianity with the q/p pull, by hit class",
                 fontsize=13)
    fn = os.path.join(a.outpath, "xcum_ref0_by_class.pdf")
    pubhtml.savefig(fig, fn)
    plt.close(fig)
    log.info(f"wrote {fn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
