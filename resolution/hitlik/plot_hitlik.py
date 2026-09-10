#!/usr/bin/env python3
"""Figures for the residual-vector CF likelihood study.

One file per panel, into ``~/public_html/cvh/<YYMMDD>_hitlik/``.

``--densities``
    per whitened component: the data pull histogram against the AVERAGE
    predicted density of each arm, with a data/model ratio panel.  The model
    density is evaluated directly from the stored exponents,

        p_i(z) = (1/pi) Int_0^tmax dtau Re[ exp(S_i(tau)) ] cos(tau z)
                 + (1/pi) Int Im[exp S] sin(tau z)

    (the same inverse Fourier transform ``MassCFTerm`` does in graph), on a
    tau grid cubic-spline-upsampled by ``--upsample``, then averaged over the
    rows of that component.  This is the picture the whole study is about:
    where the Gaussian and the CF differ, and by how much.

``--ratios``
    the per-parameter information ratios from ``report.py``'s npz.

``--recovery``
    the injection-recovery summary from a set of rabbit fitresults.
"""

import argparse
import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wums import logging  # noqa: E402
import pubhtml  # noqa: E402
import ratiopanel  # noqa: E402
import hitlik_term as HT  # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

CNAME = ("q/p", r"$\lambda$", r"$\phi$", r"$d_0$", r"$z_0$")
ARMLAB = {"cf": "CF (full non-Gaussian)",
          "gauss": "Gaussian, variance-matched",
          "gaussq": r"Gaussian, fit's $Q$ (the $\chi^2$)"}
ARMCOL = {"cf": "#d62728", "gauss": "#2ca02c", "gaussq": "#1f77b4"}


def upsample_matrix(tg, up):
    from scipy.interpolate import CubicSpline
    tf_ = np.linspace(tg[0], tg[-1], (len(tg) - 1) * up + 1)
    return tf_, CubicSpline(tg, np.eye(len(tg)), axis=0)(tf_)


def mean_density(sel, arm, comp, zgrid, upsample=8, chunk=4000):
    """Row-averaged predicted density of one component, one arm."""
    fam, ptr, gid, ndrop, ng, amp, drop = HT.arm_families(sel, arm)
    tg = sel["tgrid"]
    tfine, U = upsample_matrix(tg, upsample)
    nt = len(tfine)
    rows = np.where(sel["comp"] == comp)[0]
    seg = np.repeat(np.arange(len(sel["z"])), np.diff(ptr))
    # per-row total exponent
    Sre = np.zeros((len(sel["z"]), len(tg)))
    Sim = np.zeros((len(sel["z"]), len(tg)))
    for m in fam:
        if "re" in m:
            np.add.at(Sre, seg, m["re"].astype(np.float64))
        if "im" in m:
            np.add.at(Sim, seg, m["im"].astype(np.float64))
        if "fix_re" in m:
            Sre += m["fix_re"].astype(np.float64)
        if "fix_im" in m:
            Sim += m["fix_im"].astype(np.float64)
    Sre = Sre[rows] @ U.T
    Sim = Sim[rows] @ U.T
    Sre += -0.5 * np.outer(sel["vgf"][rows], tfine ** 2)
    E = np.exp(Sre)
    cre = E * np.cos(Sim)
    cim = E * np.sin(Sim)
    w = np.gradient(tfine)
    w[0] *= 0.5
    w[-1] *= 0.5
    C = np.cos(np.outer(tfine, zgrid))
    Ssin = np.sin(np.outer(tfine, zgrid))
    dens = np.zeros(len(zgrid))
    for i0 in range(0, len(rows), chunk):
        sl = slice(i0, i0 + chunk)
        dens += ((cre[sl] * w) @ C + (cim[sl] * w) @ Ssin).sum(axis=0)
    return dens / (np.pi * len(rows))


def densities(args, sel, outdir):
    zlo, zhi, nb = args.zrange[0], args.zrange[1], args.nbins
    edges = np.linspace(zlo, zhi, nb + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fine = np.unique(np.concatenate([edges, ctr]))
    for comp in sorted(set(sel["comp"].tolist())):
        rows = np.where(sel["comp"] == comp)[0]
        z = np.clip(sel["z"][rows], zlo, zhi)
        h, _ = np.histogram(z, bins=edges)
        n = len(z)
        dens_data = h / (n * np.diff(edges))
        err = np.where(h > 0, dens_data / np.sqrt(np.maximum(h, 1)), np.nan)
        models = {}
        for arm in args.arms:
            f = mean_density(sel, arm, comp, fine, upsample=args.upsample)
            g = np.interp(fine, fine, f)
            lo = np.interp(edges[:-1], fine, f)
            hi = np.interp(edges[1:], fine, f)
            mid = np.interp(ctr, fine, f)
            models[arm] = (ratiopanel.bin_average(lo, mid, hi), fine, f)
        for logy in (False, True):
            fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.0, 7.0))
            ax.errorbar(ctr, dens_data, yerr=err, fmt="ko", ms=3.2, lw=1.0,
                        label=f"data ({n} rows)", zorder=5)
            for arm in args.arms:
                _, ff, fv = models[arm]
                ax.plot(ff, fv, "-", color=ARMCOL[arm], lw=1.8,
                        label=ARMLAB[arm])
            ax.set_ylabel("probability density")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(max(1e-6, np.nanmin(dens_data[dens_data > 0]) * 0.3),
                            np.nanmax(dens_data) * 2.0)
            ax.legend(loc="upper right", fontsize=13, frameon=False)
            ax.set_title(f"whitened residual component {comp}  "
                         f"({CNAME[comp]}),  20-60 GeV $\\mu$ gun",
                         fontsize=15)
            for arm in args.arms:
                mb = models[arm][0]
                r = np.where(mb > 0, dens_data / mb, np.nan)
                re_ = np.where(mb > 0, err / mb, np.nan)
                rax.errorbar(ctr, r, yerr=re_, fmt="o", ms=3.0, lw=1.0,
                             color=ARMCOL[arm])
            rax.axhline(1.0, color="k", lw=0.8)
            rax.set_ylim(0.55, 1.45)
            rax.set_ylabel("data / model", fontsize=13)
            rax.set_xlabel(r"whitened residual $z$")
            fn = os.path.join(
                outdir, f"density_c{comp}{'_log' if logy else ''}.pdf")
            fig.savefig(fn, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"wrote {fn}")


def ratios(args, outdir):
    d = np.load(args.ratios, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    keys = [k for k in d.files if k.startswith("ratio_")]
    for kind, pref, lab in (("material", "material_", r"material $k_g$"),
                            ("hitres", "hitres_", r"hit class $\epsilon_c$")):
        idx = [i for i, p in enumerate(params) if p.startswith(pref)]
        if not idx:
            continue
        base = None
        for k in keys:
            if "CF_over_gaussq" in k.replace("/", "_over_") or "gaussq" in k:
                base = k
                break
        if base is None:
            base = keys[0]
        r = np.asarray(d[base], float)[idx]
        nm = [params[i][len(pref):] for i in idx]
        ok = np.isfinite(r)
        o = np.argsort(-r[ok])
        rr = r[ok][o]
        nn = [nm[i] for i in np.where(ok)[0][o]]
        fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.34 * len(rr))))
        y = np.arange(len(rr))
        ax.barh(y, rr, color="#d62728", alpha=0.85)
        ax.axvline(1.0, color="k", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(nn, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(r"information ratio  $I_{\rm CF}\,/\,I_{\rm Gauss}$")
        ax.set_title(f"what the non-Gaussian densities buy: {lab}", fontsize=14)
        fn = os.path.join(outdir, f"inforatio_{kind}.pdf")
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"wrote {fn}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=None)
    p.add_argument("--max-tracks", type=int, default=4000)
    p.add_argument("--comps", default="0123")
    p.add_argument("--arms", nargs="+", default=["cf", "gaussq"])
    p.add_argument("--densities", action="store_true")
    p.add_argument("--ratios", default=None)
    p.add_argument("--zrange", type=float, nargs=2, default=[-6.0, 6.0])
    p.add_argument("--nbins", type=int, default=80)
    p.add_argument("--upsample", type=int, default=8)
    p.add_argument("--outpath", default=None)
    args = p.parse_args()
    logging.setup_logger(__file__, 3, False)
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_hitlik/")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    if args.densities:
        sel = HT.load(args.npz, max_tracks=args.max_tracks,
                      comps=[int(c) for c in args.comps])
        densities(args, sel, outdir)
    if args.ratios:
        ratios(args, outdir)


if __name__ == "__main__":
    main()
