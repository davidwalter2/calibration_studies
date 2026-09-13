#!/usr/bin/env python3
"""Figures for the vertex-constraint-residual CF term.

One file per panel, into ``~/public_html/ZMass/cvh/<YYMMDD>_vtxres/``.

--densities    z_v against the three arms' average predicted density, with a
               data/model ratio panel (linear and log), and the tails table.
--composition  the variance shares of sigma_v^2 by FAMILY (hit / MS /
               ionization) and by MATERIAL GROUP, and vs the GEN muon pT --
               i.e. the answer to "is the vertex residual MS- or hit-dominated".
--sigma        sigma_v against z_v: the self-consistent-sigma check.
"""
import argparse, datetime, os, sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import selection as _SEL  # noqa: E402  (ONE value for the chi2 cut)

from wums import logging  # noqa: E402
import pubhtml  # noqa: E402
import ratiopanel  # noqa: E402
import vtxterm as VT  # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

ARMLAB = {"cf": "CF (full non-Gaussian)",
          "gauss": "Gaussian, variance-matched",
          "gaussq": r"Gaussian, fit's $Q$ (the $\chi^2$)"}
ARMCOL = {"cf": "#d62728", "gauss": "#2ca02c", "gaussq": "#1f77b4"}


def densities(a, sel, outdir, tag):
    zlo, zhi, nb = a.zrange[0], a.zrange[1], a.nbins
    edges = np.linspace(zlo, zhi, nb + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fine = np.unique(np.concatenate([edges, ctr]))
    rows = np.arange(sel["n"])
    z = sel["z"]
    zc = np.clip(z, zlo, zhi)
    h, _ = np.histogram(zc, bins=edges)
    dens_data = h / (len(z) * np.diff(edges))
    err = np.where(h > 0, dens_data / np.sqrt(np.maximum(h, 1)), np.nan)
    models, dens_arm = {}, {}
    zg = np.linspace(-40, 40, 3201)
    for arm in a.arms:
        f = VT.mean_density(sel, arm, rows, fine, upsample=a.upsample)
        models[arm] = (fine, f)
        dens_arm[arm] = VT.mean_density(sel, arm, rows, zg, upsample=a.upsample)
    msg = [f"{tag}: N {len(z)}, Var(z) {np.var(z):.5f}, mean {np.mean(z):+.5f} "
           f"+- {np.std(z)/np.sqrt(len(z)):.5f}, "
           f"skew {((z-z.mean())**3).mean()/z.std()**3:+.4f}, "
           f"kurt {((z-z.mean())**4).mean()/z.var()**2:.4f}",
           "   " + ", ".join(
               f"norm({m}) {np.trapezoid(dens_arm[m], zg):.6f}  "
               f"var(model) {np.trapezoid(dens_arm[m]*zg**2, zg):.5f}  "
               f"skew(model) {np.trapezoid(dens_arm[m]*zg**3, zg):+.4f}  "
               f"kurt(model) {np.trapezoid(dens_arm[m]*zg**4, zg)/max(np.trapezoid(dens_arm[m]*zg**2, zg),1e-12)**2:.3f}"
               for m in a.arms),
           f"   DATA trimmed at |z| < 5: Var {np.var(z[np.abs(z)<5]):.5f}  "
           f"skew {((lambda y: ((y-y.mean())**3).mean()/y.std()**3)(z[np.abs(z)<5])):+.4f}  "
           f"kurt {((lambda y: ((y-y.mean())**4).mean()/y.var()**2)(z[np.abs(z)<5])):.4f}  "
           f"(drops {100*np.mean(np.abs(z)>=5):.3f} %)"]
    tails = {}
    for thr in (1.0, 2.0, 3.0, 4.0, 5.0):
        fd = float(np.mean(np.abs(z) > thr))
        parts, row = [], {"data": fd}
        for arm in a.arms:
            lo, hi = zg <= -thr, zg >= thr
            fm = float(np.trapezoid(dens_arm[arm][lo], zg[lo])
                       + np.trapezoid(dens_arm[arm][hi], zg[hi]))
            row[arm] = fm
            parts.append(f"{arm} {fm:.6f} (data/model {fd/max(fm,1e-12):.3f})")
        tails[thr] = row
        msg.append(f"   P(|z| > {thr:g}): data {fd:.6f}   " + "   ".join(parts))
    logger.info("\n".join(msg))

    for logy in (False, True):
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.0, 7.0))
        ax.errorbar(ctr, dens_data, yerr=err, fmt="ko", ms=3.2, lw=1.0,
                    label=f"data ({len(z)} candidates)", zorder=5)
        for arm in a.arms:
            ff, fv = models[arm]
            ax.plot(ff, fv, "-", color=ARMCOL[arm], lw=1.8, label=ARMLAB[arm])
        ax.set_ylabel("probability density")
        if logy:
            ax.set_yscale("log")
            pos = dens_data[dens_data > 0]
            ax.set_ylim(max(1e-7, pos.min() * 0.3), dens_data.max() * 2.0)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_title(f"{tag}: the vertex-constraint residual pull "
                     r"$z_v = r_v/\sigma_v$", fontsize=13)
        ref = a.arms[-1]
        _, fref = models[ref]
        mref = ratiopanel.bin_average(np.interp(edges[:-1], fine, fref),
                                      np.interp(ctr, fine, fref),
                                      np.interp(edges[1:], fine, fref))
        r = dens_data / np.where(mref > 0, mref, np.nan)
        re_ = err / np.where(mref > 0, mref, np.nan)
        rax.errorbar(ctr[1:-1], r[1:-1], yerr=re_[1:-1], fmt="ko", ms=3.0, lw=1.0)
        for arm in a.arms:
            ff, fv = models[arm]
            rax.plot(ff, fv / np.interp(ff, fine, fref), "-",
                     color=ARMCOL[arm], lw=1.6)
        rax.axhline(1.0, color="k", lw=0.8)
        rax.set_ylim((0.3, 6.0) if logy else (0.7, 1.3))
        if logy:
            rax.set_yscale("log")
        rax.set_ylabel(f"/ {ARMLAB[ref].split(',')[0]}", fontsize=10)
        rax.set_xlabel(r"$z_v = r_v/\sigma_v$")
        fn = os.path.join(outdir, f"density_{tag}{'_log' if logy else ''}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")
    return tails


def composition(a, sel, outdir, tag):
    n = sel["n"]
    # ---- by family --------------------------------------------------------
    # `vbs` is the LUMINOUS REGION's own share, present only for the two
    # beam-line functionals (it is a registered Gaussian block, family 16).
    fams = [("hit (Gaussian)", sel.get("vhit")), ("multiple scattering", sel.get("vms")),
            ("ionization", sel.get("vioni")), ("beam line (Gaussian)", sel.get("vbs"))]
    fams = [(k, v) for k, v in fams if v is not None]
    if fams:
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        for k, v in fams:
            ax.hist(v, bins=np.linspace(0, 1, 101), histtype="step", lw=1.8, label=k)
        ax.set_xlabel(r"share of $\sigma_v^2$")
        ax.set_ylabel("candidates")
        ax.legend(fontsize=11)
        ax.set_title(f"{tag}: what the residual is made of", fontsize=13)
        fn = os.path.join(outdir, f"family_shares_{tag}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")
        logger.info(f"{tag} family shares (median): " + ", ".join(
            f"{k} {np.median(v):.4f}" for k, v in fams))

    # ---- vs GEN pT --------------------------------------------------------
    if "genpt_plus" in sel and "vhit" in sel:
        pt = np.minimum(sel["genpt_plus"], sel["genpt_minus"])
        ed = np.percentile(pt, np.linspace(0, 100, 9))
        ed = np.unique(ed)
        ic = np.clip(np.digitize(pt, ed) - 1, 0, len(ed) - 2)
        xc = 0.5 * (ed[1:] + ed[:-1])
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        for k, v in fams:
            m = np.array([np.median(v[ic == i]) for i in range(len(ed) - 1)])
            ax.plot(xc, m, "o-", lw=1.8, label=k)
        ax.set_xlabel(r"softer muon GEN $p_T$ [GeV]")
        ax.set_ylabel(r"median share of $\sigma_v^2$")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.set_title(f"{tag}: MS- or hit-dominated?", fontsize=13)
        fn = os.path.join(outdir, f"shares_vs_genpt_{tag}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")
        for k, v in fams:
            logger.info(f"  {k:22s} " + " ".join(
                f"{np.median(v[ic==i]):.3f}" for i in range(len(ed) - 1))
                + f"   (pT bins {np.round(ed,2)})")

    # ---- by MATERIAL GROUP -----------------------------------------------
    gid = sel["grp_id"]
    if len(gid):
        gnames = sel["group_names"]
        ng = len(gnames)
        # a group's share of the candidate's MS+ioni variance, from the
        # exponents' second moment (the CF's own kappa2 per group row)
        k2 = VT.kappa2_from_grid(sel["Sms"] + sel["Sio_re"], sel["tgrid"])
        tot = np.zeros(ng)
        np.add.at(tot, gid, k2)
        tot = tot / max(n, 1)
        o = np.argsort(-tot)[:18]
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        ax.barh(range(len(o)), tot[o][::-1], color="#4c72b0")
        ax.set_yticks(range(len(o)))
        ax.set_yticklabels([gnames[i] for i in o][::-1], fontsize=9)
        ax.set_xlabel(r"mean share of $\sigma_v^2$ (MS + ionization)")
        ax.set_title(f"{tag}: which material the term sees", fontsize=13)
        fn = os.path.join(outdir, f"group_shares_{tag}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")
        logger.info(f"{tag} top material groups: " + ", ".join(
            f"{gnames[i]} {tot[i]:.4f}" for i in o[:10]))

    # ---- by HIT CLASS -----------------------------------------------------
    if len(sel["hit_cls"]):
        cn = sel["hit_classes"]
        nc = len(cn)
        tc = np.zeros(nc)
        np.add.at(tc, sel["hit_cls"], sel["hit_v"])
        tc /= max(n, 1)
        o = np.argsort(-tc)[:18]
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        ax.barh(range(len(o)), tc[o][::-1], color="#c44e52")
        ax.set_yticks(range(len(o)))
        ax.set_yticklabels([cn[i] for i in o][::-1], fontsize=9)
        ax.set_xlabel(r"mean share of $\sigma_v^2$")
        ax.set_title(f"{tag}: which hit classes the term sees", fontsize=13)
        fn = os.path.join(outdir, f"hitclass_shares_{tag}.pdf")
        pubhtml.savefig(fig, fn)
        plt.close(fig)
        logger.info(f"wrote {fn}")
        logger.info(f"{tag} top hit classes: " + ", ".join(
            f"{cn[i]} {tc[i]:.4f}" for i in o[:8]))


def sigma_check(a, sel, outdir, tag):
    s, z = sel["sigma"], sel["z"]
    c1 = np.corrcoef(s, z)[0, 1]
    c2 = np.corrcoef(s, np.abs(z))[0, 1]
    e = 1.0 / np.sqrt(len(z))
    logger.info(f"{tag}: corr(sigma, z) = {c1:+.5f} +- {e:.5f}; "
                f"corr(sigma, |z|) = {c2:+.5f}")
    # the first-order Jacobian term of a self-consistent sigma:
    # d ln p / d ln sigma ~ (z d/dz + 1), so a correlation c between sigma and
    # z shifts the mean of z by ~ c * (dsigma/sigma).  Quote the size.
    ed = np.percentile(s, np.linspace(0, 100, 11))
    ic = np.clip(np.digitize(s, ed) - 1, 0, len(ed) - 2)
    xc = 0.5 * (ed[1:] + ed[:-1])
    m = np.array([z[ic == i].mean() for i in range(len(ed) - 1)])
    me = np.array([z[ic == i].std() / np.sqrt(max((ic == i).sum(), 1))
                   for i in range(len(ed) - 1)])
    v = np.array([z[ic == i].var() for i in range(len(ed) - 1)])
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.errorbar(xc * 1e4, m, yerr=me, fmt="o-", label=r"$\langle z_v\rangle$")
    ax.plot(xc * 1e4, v - 1.0, "s--", label=r"${\rm Var}(z_v) - 1$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_v$ [$\mu$m]")
    ax.set_ylabel("")
    ax.legend(fontsize=11)
    ax.set_title(f"{tag}: is the pull flat in the fit's own error?", fontsize=13)
    fn = os.path.join(outdir, f"sigma_check_{tag}.pdf")
    pubhtml.savefig(fig, fn)
    plt.close(fig)
    logger.info(f"wrote {fn}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, nargs="+")
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--maxn", type=int, default=0)
    p.add_argument("--max-chi2-ndof", type=float,
                   default=_SEL.MAX_CHI2_NDOF)
    p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
    p.add_argument("--zrange", type=float, nargs=2, default=[-6.0, 6.0])
    p.add_argument("--nbins", type=int, default=60)
    p.add_argument("--upsample", type=int, default=8)
    p.add_argument("--outpath", default=None)
    p.add_argument("--outtag", default="vtxres",
                   help="the study tag in ~/public_html/ZMass/cvh/<YYMMDD>_<tag>")
    p.add_argument("--densities", action="store_true")
    p.add_argument("--composition", action="store_true")
    p.add_argument("--sigma", action="store_true")
    a = p.parse_args()
    if not (a.densities or a.composition or a.sigma):
        a.densities = a.composition = a.sigma = True
    logging.setup_logger(__file__, 3, False)
    day = datetime.date.today().strftime("%y%m%d")
    outdir = a.outpath or pubhtml.figdir(a.outtag, day)
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    tags = a.tags or [os.path.basename(x).replace(".npz", "") for x in a.npz]
    for npz, tag in zip(a.npz, tags):
        sel = VT.load(npz, maxn=(a.maxn or None), max_chi2_ndof=a.max_chi2_ndof)
        logger.info(f"=== {tag}: {sel['n']} candidates, functional "
                    f"{sel['functional']}")
        if a.densities:
            densities(a, sel, outdir, tag)
        if a.composition:
            composition(a, sel, outdir, tag)
        if a.sigma:
            sigma_check(a, sel, outdir, tag)


if __name__ == "__main__":
    main()
