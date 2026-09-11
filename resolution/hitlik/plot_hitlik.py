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
import re
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


def mean_density(sel, arm, rows, zgrid, upsample=8, chunk=4000):
    """Row-averaged predicted density over an explicit row set, one arm.

    `rows` used to be a component INDEX; it is now the row index array, so the
    same function serves the fixed 5-component truth-referenced study and the
    per-hit one, where the natural grouping is by hit class or by position
    along the track rather than by component slot.
    """
    fam, ptr, gid, ndrop, ng, amp, drop = HT.arm_families(sel, arm)
    tg = sel["tgrid"]
    tfine, U = upsample_matrix(tg, upsample)
    nt = len(tfine)
    rows = np.asarray(rows)
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


def _groups(args, sel):
    """(label, row indices) pairs, by whichever axis the figure wants.

    `comp`   the whitened component slot (the prototype's picture: hit order
             along the track for the complement components, then the
             truth-referenced ones);
    `cls`    the hit-resolution class the component's row belongs to -- the
             axis the 18 hit parameters live on;
    `relpos` the row's position along the track in five bins;
    `ckind`  per-hit complement vs truth-referenced, two panels.
    """
    ax = getattr(args, "group_by", "comp")
    if ax == "comp":
        return [(f"c{c}", np.where(sel["comp"] == c)[0])
                for c in sorted(set(np.asarray(sel["comp"]).tolist()))]
    if ax == "cls":
        names = sel["hit_classes"]
        out = []
        cl = np.asarray(sel["cls"])
        for c in sorted(set(cl.tolist())):
            if c < 0:
                continue
            r = np.where(cl == c)[0]
            if len(r) >= args.min_rows:
                out.append((names[c] if c < len(names) else f"cls{c}", r))
        return out
    if ax == "relpos":
        rel = np.asarray(sel["relpos"])
        e = np.linspace(0, 1, 6)
        out = []
        for b in range(5):
            r = np.where((rel >= e[b]) & (rel < e[b + 1] + (1e-9 if b == 4 else 0))
                         & (rel >= 0))[0]
            if len(r) >= args.min_rows:
                out.append((f"pos{e[b]:.1f}-{e[b+1]:.1f}", r))
        return out
    if ax == "ckind":
        ck = np.asarray(sel["ckind"])
        return [(lab, np.where(ck == k)[0]) for k, lab in
                ((0, "perhit"), (1, "reference"))
                if (ck == k).sum() >= args.min_rows]
    raise ValueError(ax)


def densities(args, sel, outdir):
    zlo, zhi, nb = args.zrange[0], args.zrange[1], args.nbins
    edges = np.linspace(zlo, zhi, nb + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fine = np.unique(np.concatenate([edges, ctr]))
    for comp, rows in _groups(args, sel):
        z = np.clip(sel["z"][rows], zlo, zhi)
        h, _ = np.histogram(z, bins=edges)
        n = len(z)
        dens_data = h / (n * np.diff(edges))
        err = np.where(h > 0, dens_data / np.sqrt(np.maximum(h, 1)), np.nan)
        zfull = sel["z"][rows]
        models = {}
        for arm in args.arms:
            f = mean_density(sel, arm, rows, fine, upsample=args.upsample)
            g = np.interp(fine, fine, f)
            lo = np.interp(edges[:-1], fine, f)
            hi = np.interp(edges[1:], fine, f)
            mid = np.interp(ctr, fine, f)
            models[arm] = (ratiopanel.bin_average(lo, mid, hi), fine, f)
        zg = np.linspace(-30, 30, 1201)
        dens_arm = {arm: mean_density(sel, arm, rows, zg,
                                      upsample=args.upsample)
                    for arm in args.arms}
        msg = [f"component {comp}: N {n}, Var(z) {np.var(zfull):.4f}, "
               f"mean {np.mean(zfull):+.4f}, "
               + ", ".join(f"norm({a}) {np.trapezoid(dens_arm[a], zg):.5f}"
                           for a in args.arms)]
        for thr in (2.0, 3.0, 4.0):
            fd = float(np.mean(np.abs(zfull) > thr))
            parts = []
            for arm in args.arms:
                # the two tails are DISJOINT: one trapezoid over the union
                # would add a spurious slab across the core.
                lo = zg <= -thr
                hi = zg >= thr
                fm = float(np.trapezoid(dens_arm[arm][lo], zg[lo])
                           + np.trapezoid(dens_arm[arm][hi], zg[hi]))
                parts.append(f"{arm} {fm:.5f} (data/model "
                             f"{fd/max(fm,1e-12):.3f})")
            msg.append(f"   P(|z| > {thr:g}): data {fd:.5f}   "
                       + "   ".join(parts))
        logger.info("\n".join(msg))
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
            lab = (f"component {comp} ({CNAME[int(str(comp)[1:])]})"
                   if getattr(args, "group_by", "comp") == "comp"
                   and str(comp).startswith("c")
                   and str(comp)[1:].isdigit()
                   and int(str(comp)[1:]) < len(CNAME)
                   else str(comp))
            ax.set_title(f"whitened residual, {lab},  20-60 GeV $\\mu$ gun",
                         fontsize=15)
            # the reference of the ratio panel is the LAST arm (the Gaussian
            # the CF is being compared against): the data points then say which
            # arm they prefer, and the smooth curve is the MODEL/MODEL ratio,
            # i.e. exactly what the non-Gaussian densities change.
            refarm = args.arms[-1]
            mref = models[refarm][0]
            for arm in args.arms:
                mb = models[arm][0]
                if arm == refarm:
                    r = np.where(mref > 0, dens_data / mref, np.nan)
                    re_ = np.where(mref > 0, err / mref, np.nan)
                    rax.errorbar(ctr, r, yerr=re_, fmt="o", ms=3.0, lw=1.0,
                                 color="k", label="data")
                else:
                    _, ff, fv = models[arm]
                    fr = np.interp(ff, ff, fv) / np.maximum(
                        np.interp(ff, models[refarm][1], models[refarm][2]),
                        1e-300)
                    rax.plot(ff, fr, "-", color=ARMCOL[arm], lw=1.8)
            rax.axhline(1.0, color="k", lw=0.8)
            if logy:
                rax.set_yscale("log")
                rax.set_ylim(0.4, 40.0)
            else:
                rax.set_ylim(0.55, 1.45)
            rax.set_ylabel(f"/ {ARMLAB[refarm].split(',')[0]}", fontsize=11)
            rax.set_xlabel(r"whitened residual $z$")
            fn = os.path.join(
                outdir,
                "density_" + re.sub(r"[^0-9A-Za-z_.+-]", "_", str(comp))
                + ("_log" if logy else "") + ".pdf")
            pubhtml.savefig(fig, fn)
            plt.close(fig)
            logger.info(f"wrote {fn}")


def ratios(args, outdir):
    """Horizontal-bar information ratios, one file per (key, family)."""
    d = np.load(args.ratios, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    keys = args.ratio_keys or [k for k in d.files if k.startswith("ratio_")
                               and not k.startswith("ratio_alone")]
    LAB = {
        "ratio_CF_over_gaussq[0123]":
            (r"$I_{\rm CF}\,/\,I_{\chi^2}$  (4 components)",
             "what the non-Gaussian densities cost", "#d62728", (0.0, 1.35)),
        "ratio_CF_over_gaussq[0]":
            (r"$I_{\rm CF}\,/\,I_{\chi^2}$  (q/p only)",
             "what the non-Gaussian densities cost", "#d62728", (0.0, 1.35)),
        "ratio_CF0123_over_CF0":
            (r"$I_{\rm 4\ comp}\,/\,I_{q/p}$",
             "what the residual VECTOR buys over the q/p scalar",
             "#1f77b4", None),
    }
    for key in keys:
        if key not in d.files:
            continue
        lab, ttl, col, xl = LAB.get(
            key, (key, key, "#7f7f7f", None))
        for kind, pref, fam in (("material", "material_", "material $k_g$"),
                                ("hitres", "hitres_",
                                 r"hit class $\epsilon_c$")):
            idx = [i for i, p_ in enumerate(params) if p_.startswith(pref)]
            r = np.asarray(d[key], float)[idx]
            nm = [params[i][len(pref):] for i in idx]
            ok = np.isfinite(r) & (np.abs(r - 1.0) > 1e-6)
            if ok.sum() < 2:
                continue
            o = np.argsort(-r[ok])
            rr = r[ok][o]
            nn = [nm[i] for i in np.where(ok)[0][o]]
            fig, ax = plt.subplots(figsize=(9.0, max(3.5, 0.32 * len(rr) + 1.4)))
            y = np.arange(len(rr))
            ax.barh(y, rr, color=col, alpha=0.85)
            ax.axvline(1.0, color="k", lw=1.2)
            ax.set_yticks(y)
            ax.set_yticklabels(nn, fontsize=10)
            ax.invert_yaxis()
            if xl:
                ax.set_xlim(*xl)
            elif rr.max() / max(rr.min(), 1e-9) > 30:
                ax.set_xscale("log")
            ax.set_xlabel(lab)
            ax.set_title(f"{ttl}: {fam}", fontsize=14)
            ax.text(0.98, 0.03, f"median {np.median(rr):.2f}",
                    transform=ax.transAxes, ha="right", fontsize=12)
            tag = key.replace("ratio_", "").replace("[", "_").replace("]", "")
            fn = os.path.join(outdir, f"inforatio_{tag}_{kind}.pdf")
            pubhtml.savefig(fig, fn)
            plt.close(fig)
            logger.info(f"wrote {fn}")


def efficiency_fig(args, outdir):
    """The number that answers the question: what the Gaussian CLAIMS against
    what it actually COSTS."""
    d = np.load(args.efficiency, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    for kind, pref, fam in (("material", "material_", r"material $k_g$"),
                            ("hitres", "hitres_",
                             r"hit class $\epsilon_c$")):
        idx = [i for i, p_ in enumerate(params) if p_.startswith(pref)]
        cfq = np.asarray(d["sigma_quoted_cf"], float)[idx]
        cfa = np.asarray(d["sigma_actual_cf"], float)[idx]
        gq = np.asarray(d["sigma_quoted_gaussq"], float)[idx]
        ga = np.asarray(d["sigma_actual_gaussq"], float)[idx]
        pri = np.asarray(d["prior"], float)[idx]
        nm = [params[i][len(pref):] for i in idx]
        ok = cfq < 0.98 * pri
        if ok.sum() < 2:
            continue
        claim = (gq[ok] / cfq[ok]) ** 2
        real = (ga[ok] / cfa[ok]) ** 2
        o = np.argsort(-real)
        y = np.arange(int(ok.sum()))
        nn = [nm[i] for i in np.where(ok)[0][o]]
        fig, ax = plt.subplots(figsize=(9.0, max(3.5, 0.36 * len(y) + 1.5)))
        ax.barh(y - 0.19, real[o], height=0.36, color="#d62728", alpha=0.9,
                label=r"ACTUAL (sandwich)")
        ax.barh(y + 0.19, claim[o], height=0.36, color="#8c8c8c", alpha=0.9,
                label=r"what the $\chi^2$ CLAIMS")
        ax.axvline(1.0, color="k", lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(nn, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\sigma^2_{\chi^2}\,/\,\sigma^2_{\rm CF}$"
                      "     (>1: the full PDF constrains better)")
        ax.set_title(f"efficiency of the full PDF over the $\chi^2$: {fam}",
                     fontsize=14)
        ax.legend(loc="center right", fontsize=12, frameon=False)
        ax.set_xlim(0.0, max(1.35, real.max() * 1.10))
        ax.text(0.99, 1.012, f"median: actual {np.median(real):.2f}, "
                             f"claimed {np.median(claim):.2f}",
                transform=ax.transAxes, ha="right", fontsize=11)
        fn = os.path.join(outdir, f"efficiency_{kind}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=None)
    p.add_argument("--max-tracks", type=int, default=4000)
    p.add_argument("--max-inflat", type=float, default=1e4)
    p.add_argument("--comps", default="0123")
    p.add_argument("--arms", nargs="+", default=["cf", "gaussq"])
    p.add_argument("--densities", action="store_true")
    p.add_argument("--ratios", default=None)
    p.add_argument("--efficiency", default=None)
    p.add_argument("--ratio-keys", nargs="*", default=None)
    p.add_argument("--zrange", type=float, nargs=2, default=[-7.0, 7.0])
    p.add_argument("--nbins", type=int, default=112)
    p.add_argument("--upsample", type=int, default=8)
    p.add_argument("--outpath", default=None)
    p.add_argument("--group-by", choices=["comp", "cls", "relpos", "ckind"],
                   default="comp",
                   help="the axis the density panels are grouped on")
    p.add_argument("--min-rows", type=int, default=200)
    args = p.parse_args()
    logging.setup_logger(__file__, 3, False)
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_hitlik/")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    if args.densities:
        sel = HT.load(args.npz, max_tracks=args.max_tracks,
                      comps=args.comps,
                      max_inflat=args.max_inflat)
        densities(args, sel, outdir)
    if args.ratios:
        ratios(args, outdir)
    if args.efficiency:
        efficiency_fig(args, outdir)


if __name__ == "__main__":
    main()
