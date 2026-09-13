#!/usr/bin/env python3
"""Figures for the TAIL study (STATE.md section 16).

One file per panel into ``~/public_html/ZMass/cvh/<YYMMDD>_tail/``, a ratio
panel under every density-against-model plot.

  --densities   the three pulls against a Gaussian and against the SCALE
                MIXTURE built from the measured chi2/ndof (no free parameter)
  --chi2        the chi2 probability (flat if the model is right), the
                Var(z | chi2/ndof) ladder, and where the excess chi2 lives
  --lumi        hypothesis A: the gen vertex against the beam-spot record
  --align       hypothesis B: the misalignment in the measurement direction
  --ideal       the ideal-geometry re-production against the nominal one
"""
import argparse, datetime, os, sys
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from wums import logging   # noqa: E402
import pubhtml            # noqa: E402
import ratiopanel         # noqa: E402
import genbkg as GB       # noqa: E402
import tail_common as TC  # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
SUBDET = {0: "BPix", 1: "FPix", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}


def load(fn, need_bs=True, chi2=0.0, signal=True):
    d = dict(np.load(fn, allow_pickle=False))
    base, chi2n, nl = TC.baseline(d, need_bs=need_bs, max_abs_vtxz=0.0,
                                  chi2=chi2, verbose=False)
    if signal and "genidx_plus" in d:
        cls, names, _ = GB.classify(d)
        base = base & (cls == names.index("signal"))
    zv = np.asarray(d["vtxz"], float)
    nd = np.asarray(d["ndof"], float)
    sub = np.asarray(d.get("vtxdchi2", zv ** 2), float)
    nsub = 1.0
    if need_bs and "bschi2fit" in d:
        sub = sub + np.asarray(d["bschi2fit"], float)
        nsub += 3.0
    ndr = np.maximum(nd - nsub, 1.0)
    chir = np.asarray(d["chisq"], float) - sub
    d["_chired"] = chir / ndr
    d["_ndr"] = ndr
    d["_u"] = stats.chi2.sf(chir, ndr)
    d["_sel"] = base
    d["_zv"] = zv
    return d


def mixture_density(zgrid, s):
    """The scale mixture of normals with scales `s` (one per candidate)."""
    zg = np.asarray(zgrid)[:, None]
    ss = np.asarray(s)[None, :]
    return np.mean(np.exp(-0.5 * (zg / ss) ** 2) / (np.sqrt(2 * np.pi) * ss),
                   axis=1)


def density_panel(z, s, outdir, tag, label, zmax=8.0, nb=80):
    edges = np.linspace(-zmax, zmax, nb + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    zc = np.clip(z, -zmax, zmax)
    h, _ = np.histogram(zc, bins=edges)
    n = len(z)
    dens = h / (n * np.diff(edges))
    err = np.where(h > 0, dens / np.sqrt(np.maximum(h, 1)), np.nan)
    fine = np.linspace(-zmax, zmax, 1601)
    gaus = stats.norm.pdf(fine)
    mix = mixture_density(fine, s)
    # per-bin averages (Simpson) for the ratio
    mlo = mixture_density(edges[:-1], s)
    mhi = mixture_density(edges[1:], s)
    mmid = mixture_density(ctr, s)
    mavg = ratiopanel.bin_average(mlo, mmid, mhi)

    fig, ax, rax = ratiopanel.make_ratio_fig()
    ax.errorbar(ctr, dens, err, marker="o", ms=3.2, ls="none", color="black",
                label=f"{label}  (N = {n})")
    ax.plot(fine, gaus, color="#2ca02c", lw=1.8, ls="--",
            label=r"Gaussian, unit variance")
    ax.plot(fine, mix, color="#d62728", lw=2.0,
            label=r"scale mixture from the measured $\chi^2/\mathrm{ndof}$")
    ax.set_yscale("log")
    ax.set_ylim(max(1e-6, 0.3 / (n * (edges[1] - edges[0]))), 1.0)
    ax.set_ylabel("probability density")
    ax.legend(fontsize="small", loc="upper right")
    ratiopanel.draw_ratio(rax, edges, h, mavg, n,
                          ylabel="data / mixture", clamp=(0.0, 3.0),
                          xlabel=label)
    pubhtml.savefig(fig, os.path.join(outdir, f"density_{tag}.pdf"))
    plt.close(fig)
    logger.info(f"  density_{tag}: P(|z|>5) data {(np.abs(z)>5).mean():.5f}, "
                f"mixture {np.mean(2*stats.norm.sf(5.0/s)):.5f}, "
                f"Gaussian {2*stats.norm.sf(5.0):.2e}")


def ladder_panel(sets, outdir, tag):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (lab, c2, z) in enumerate(sets):
        edges = np.percentile(c2, [0, 10, 20, 30, 40, 50, 60, 70, 80,
                                   88, 94, 97.5, 99.2, 100])
        cs, vs, es = [], [], []
        for k in range(len(edges) - 1):
            b = (c2 >= edges[k]) & (c2 < edges[k + 1])
            if b.sum() < 40:
                continue
            cs.append(c2[b].mean())
            vs.append(np.var(z[b]))
            es.append(np.var(z[b]) * np.sqrt(2 / b.sum()))
        ax.errorbar(cs, vs, es, marker="o", ms=5, ls="-", lw=1.2,
                    color=cols[i % len(cols)], label=lab)
    lim = ax.get_xlim()
    xs = np.geomspace(max(lim[0], 0.3), max(lim[1], 3.2), 50)
    ax.plot(xs, xs, color="black", lw=1.5, ls=":",
            label=r"$\mathrm{Var}(z\,|\,\chi^2) = \chi^2/\mathrm{ndof}$")
    ax.set_xlabel(r"$\chi^2/\mathrm{ndof}$  (the constraint rows removed)")
    ax.set_ylabel(r"$\mathrm{Var}(z)$ in the bin")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize="small", loc="upper left")
    pubhtml.savefig(fig, os.path.join(outdir, f"ladder_{tag}.pdf"))
    plt.close(fig)


def chi2prob_panel(sets, outdir, tag):
    fig, ax, rax = ratiopanel.make_ratio_fig()
    edges = np.linspace(0, 1, 51)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for i, (lab, u) in enumerate(sets):
        h, _ = np.histogram(u, bins=edges)
        dens = h / (len(u) * np.diff(edges))
        err = np.where(h > 0, dens / np.sqrt(np.maximum(h, 1)), np.nan)
        ax.errorbar(ctr, dens, err, marker="o", ms=3.2, ls="none",
                    color=cols[i % len(cols)], label=f"{lab} (N = {len(u)})")
        rax.errorbar(ctr, dens, err, marker="o", ms=3.2, ls="none",
                     color=cols[i % len(cols)])
    ax.axhline(1.0, color="black", lw=1.5, ls=":",
               label="a correct model: flat")
    ax.set_yscale("log")
    ax.set_ylabel("probability density")
    ax.legend(fontsize="small", loc="upper center")
    rax.axhline(1.0, color="black", lw=1.2, ls=":")
    rax.set_ylim(0.6, 1.8)
    rax.set_ylabel("density", fontsize="small")
    rax.set_xlabel(r"$\chi^2$ probability of the fit")
    pubhtml.savefig(fig, os.path.join(outdir, f"chi2prob_{tag}.pdf"))
    plt.close(fig)


def rate_panel(sets, outdir, tag, xlabel, edges, ylabel=None):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for i, (lab, x, u) in enumerate(sets):
        cs, ps, es = [], [], []
        for k in range(len(edges) - 1):
            b = (x >= edges[k]) & (x < edges[k + 1])
            if b.sum() < 50:
                continue
            p, e = TC.binom(int((u[b] < 1e-3).sum()), int(b.sum()))
            cs.append(0.5 * (edges[k] + edges[k + 1])); ps.append(p); es.append(e)
        ax.errorbar(cs, ps, es, marker="o", ms=5, ls="-", lw=1.2,
                    color=cols[i % len(cols)], label=lab)
    ax.axhline(1e-3, color="black", lw=1.5, ls=":", label="a correct model")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or r"$P(\chi^2\ \mathrm{prob} < 10^{-3})$")
    ax.legend(fontsize="small")
    pubhtml.savefig(fig, os.path.join(outdir, f"chi2excess_{tag}.pdf"))
    plt.close(fig)


def mass_panel(d, sel, outdir):
    """What the excess chi2 does to the MASS -- the consequence that matters."""
    m = np.asarray(d["mass"], float)
    mg = np.asarray(d["genmass"], float)
    ok = sel & (mg > 0) & np.isfinite(m)
    c2 = d["_chired"]
    edges = np.array([0.6, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0, 2.6, 4.0])
    cs, ys, es = [], [], []
    for i in range(len(edges) - 1):
        b = ok & (c2 >= edges[i]) & (c2 < edges[i + 1])
        if b.sum() < 60:
            continue
        v = np.sort((m[b] - mg[b]) * 1e3)
        k = max(int(0.01 * len(v)), 1)
        v = v[k:-k]
        cs.append(np.median(c2[b])); ys.append(v.mean())
        es.append(v.std() / np.sqrt(len(v)))
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.errorbar(cs, ys, es, marker="o", ms=6, ls="-", lw=1.4, color="#d62728",
                label="DY MC, gen signal (1 %-trimmed)")
    ax.axhline(0.0, color="black", lw=1.2, ls=":")
    ax.set_xlabel(r"$\chi^2/\mathrm{ndof}$  (the constraint rows removed)")
    ax.set_ylabel(r"$\langle m - m_\mathrm{gen}\rangle$  [MeV]")
    ax.legend(fontsize="small")
    pubhtml.savefig(fig, os.path.join(outdir, "mass_vs_chi2.pdf"))
    plt.close(fig)


def scalespread_panels(d, sel, g, gsel, outdir):
    """The chi2/ndof density against the chi2_ndof one, and the tail
    enhancement that its UNMODELLED part alone predicts."""
    # (a) chi2/ndof against its own expectation
    fig, ax, rax = ratiopanel.make_ratio_fig()
    edges = np.linspace(0.2, 2.6, 61)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    cols = {"DY MC, gen signal": ("#d62728", d, sel),
            r"J/$\psi$ gun": ("#1f77b4", g, gsel)}
    for lab, (c, dd, ss) in cols.items():
        if dd is None:
            continue
        x = dd["_chired"][ss]
        nd = np.median(dd["_ndr"][ss])
        h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
        dens = h / (len(x) * np.diff(edges))
        err = np.where(h > 0, dens / np.sqrt(np.maximum(h, 1)), np.nan)
        mod = stats.chi2.pdf(ctr * nd, nd) * nd
        ax.errorbar(ctr, dens, err, marker="o", ms=3.0, ls="none", color=c,
                    label=f"{lab}  (N = {len(x)})")
        ax.plot(ctr, mod, color=c, lw=1.6, ls="--",
                label=rf"$\chi^2_{{{nd:.0f}}}/{nd:.0f}$")
        rax.errorbar(ctr, dens / mod, err / mod, marker="o", ms=3.0,
                     ls="none", color=c)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 4.0)
    ax.set_ylabel("probability density")
    ax.legend(fontsize="x-small", ncol=2)
    rax.axhline(1.0, color="black", lw=1.2, ls=":")
    rax.set_ylim(0.0, 3.0)
    rax.set_ylabel("data / model", fontsize="small")
    rax.set_xlabel(r"$\chi^2/\mathrm{ndof}$  (the constraint rows removed)")
    pubhtml.savefig(fig, os.path.join(outdir, "chi2ndof_density.pdf"))
    plt.close(fig)

    # (b) the enhancement the unmodelled spread alone predicts
    rng = np.random.default_rng(11)

    def enh(extra, t):
        sg = np.sqrt(np.log1p(extra ** 2))
        u = rng.lognormal(-0.5 * sg ** 2, sg, 400000)
        return (np.mean(2 * stats.norm.sf(t / np.sqrt(u)))
                / (2 * stats.norm.sf(t)))
    ts = np.linspace(2., 5.5, 15)
    rows = [(r"DY $z_1$ ($\lambda$ = 1.06)", 0.2554 * 1.06, "#d62728", 4.90, 15.98),
            (r"DY $z_2$ ($\lambda$ = 0.89)", 0.2554 * 0.89, "#1f77b4", 4.97, 6.84),
            (r"DY $z_v$ ($\lambda$ = 0.49)", 0.2554 * 0.49, "#2ca02c", 4.10, 1.08),
            (r"gun $z_v$ ($\lambda$ = 0.14)", 0.0796 * 0.14, "#9467bd", 5.17, 1.35)]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for lab, e, c, tp, pub in rows:
        ax.plot(ts, [enh(e, t) for t in ts], color=c, lw=2.0, label=lab)
        ax.plot([tp], [pub], marker="*", ms=16, color=c, ls="none")
    ax.plot([], [], marker="*", ms=14, color="black", ls="none",
            label=r"published data / CF at 5 $\sigma$ (offset in $t$ to separate)")
    ax.set_yscale("log")
    ax.set_xlabel(r"threshold $t$  [$\sigma$]")
    ax.set_ylabel("tail enhancement over the model")
    ax.legend(fontsize="small", loc="upper left")
    pubhtml.savefig(fig, os.path.join(outdir, "scalespread_enhancement.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gun", default="")
    ap.add_argument("--ideal", default="")
    ap.add_argument("--realref", default="",
                    help="the SAME-EVENTS aligned-geometry leg the --ideal one "
                         "is compared against (default: the main --npz)")
    ap.add_argument("--align", default="")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    outdir = a.outdir or os.path.expanduser(
        "~/public_html/ZMass/cvh/" +
        datetime.date.today().strftime("%y%m%d") + "_tail")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    logger.info(f"figures -> {outdir}")

    d = load(a.npz)
    sel = d["_sel"]
    s = np.sqrt(np.maximum(d["_chired"][sel], 1e-12))
    density_panel(d["bsz"][sel, 0], s, outdir, "z1_dy",
                  r"$z_1$  (beam line, $x$ pull) -- DY, gen signal")
    density_panel(d["bsz"][sel, 1], s, outdir, "z2_dy",
                  r"$z_2$  (beam line, $y\,|\,x$ pull) -- DY, gen signal")
    density_panel(d["_zv"][sel], s, outdir, "zv_dy",
                  r"$z_v$  (vertex constraint) -- DY, gen signal")

    g = gs = None
    if a.gun and os.path.exists(a.gun):
        g = load(a.gun, need_bs=False)
        gs = g["_sel"]
    sets = [(r"DY $z_1$", d["_chired"][sel], d["bsz"][sel, 0]),
            (r"DY $z_2$", d["_chired"][sel], d["bsz"][sel, 1]),
            (r"DY $z_v$", d["_chired"][sel], d["_zv"][sel])]
    probs = [("DY MC, gen signal", d["_u"][sel])]
    rates = [("DY MC, gen signal",
              np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"]))[sel],
              d["_u"][sel])]
    if g is not None:
        density_panel(g["_zv"][gs], np.sqrt(np.maximum(g["_chired"][gs], 1e-12)),
                      outdir, "zv_gun", r"$z_v$ -- J/$\psi$ gun, ideal geometry")
        sets.append((r"J/$\psi$ gun $z_v$", g["_chired"][gs], g["_zv"][gs]))
        probs.append((r"J/$\psi$ gun (ideal geometry, no pileup)", g["_u"][gs]))
        rates.append((r"J/$\psi$ gun",
                      np.maximum(np.abs(g["eta_plus"]), np.abs(g["eta_minus"]))[gs],
                      g["_u"][gs]))
    mass_panel(d, sel, outdir)
    scalespread_panels(d, sel, g, gs, outdir)
    ladder_panel(sets, outdir, "all")
    chi2prob_panel(probs, outdir, "dy_gun")
    rate_panel(rates, outdir, "vs_eta", r"$\max |\eta|$ of the two legs",
               np.array([0, .9, 1.4, 1.8, 2.05, 2.2, 2.4]))
    rate_panel([("DY MC, gen signal", d["ntrueint"][sel], d["_u"][sel])],
               outdir, "vs_pileup", r"$n_\mathrm{true\ int}$ (pileup)",
               np.array([0, 12, 17, 22, 27, 32, 60]))

    # -------- hypothesis A: the luminous region --------
    spot, slope, wid = d["bsspot"], d["bsslope"], d["bswidth"]
    gx, gy, gz = d["genvx"], d["genvy"], d["genvz"]
    hasgen = (gx > -90) & (gy > -90) & (gz > -90)
    m = sel & hasgen
    gl_x = (gx - (spot[:, 0] + slope[:, 0] * (gz - spot[:, 2]))) / wid[:, 0]
    gl_y = (gy - (spot[:, 1] + slope[:, 1] * (gz - spot[:, 2]))) / wid[:, 1]
    for nm, q, lab in (("x", gl_x[m], r"gen vertex $-$ beam line, $x$  [record $\sigma_x$]"),
                       ("y", gl_y[m], r"gen vertex $-$ beam line, $y$  [record $\sigma_y$]")):
        edges = np.linspace(-6, 6, 61)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        h, _ = np.histogram(np.clip(q, -6, 6), bins=edges)
        dens = h / (len(q) * np.diff(edges))
        err = np.where(h > 0, dens / np.sqrt(np.maximum(h, 1)), np.nan)
        sd = 1.4826 * np.median(np.abs(q - np.median(q)))
        fine = np.linspace(-6, 6, 1201)
        mod = stats.norm.pdf(fine, np.median(q), sd)
        mavg = ratiopanel.bin_average(stats.norm.pdf(edges[:-1], np.median(q), sd),
                                      stats.norm.pdf(ctr, np.median(q), sd),
                                      stats.norm.pdf(edges[1:], np.median(q), sd))
        fig, ax, rax = ratiopanel.make_ratio_fig()
        ax.errorbar(ctr, dens, err, marker="o", ms=3.2, ls="none", color="black",
                    label=f"simulated luminous region (N = {len(q)})")
        ax.plot(fine, mod, color="#d62728", lw=2.0,
                label=f"Gaussian, MAD width {sd:.3f}")
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1.0)
        ax.set_ylabel("probability density")
        ax.legend(fontsize="small")
        ratiopanel.draw_ratio(rax, edges, h, mavg, len(q),
                              ylabel="data / Gaussian", clamp=(0.0, 2.5),
                              xlabel=lab)
        pubhtml.savefig(fig, os.path.join(outdir, f"genvtx_pull_{nm}.pdf"))
        plt.close(fig)

    # -------- hypothesis B: the misalignment --------
    if a.align and os.path.exists(a.align):
        A = np.load(a.align)
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        edges = np.linspace(0, 40, 81)
        for s_ in sorted(SUBDET):
            mm = A["sub_mod"] == s_
            if not mm.any():
                continue
            sv = A["dlocx_mod"][mm] * 1e4
            v = np.abs(sv)
            ax.hist(np.clip(v, 0, 40), bins=edges, histtype="step", lw=1.8,
                    density=True, label=f"{SUBDET[s_]}  (rms {sv.std():.1f} "
                                        rf"$\mu$m, max {v.max():.0f})")
        ax.axvline(0.0, color="black", lw=2.5,
                   label=r"J/$\psi$ gun: identically zero")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\Delta_\mathrm{local\ x}|$, aligned $-$ ideal  [$\mu$m]")
        ax.set_ylabel("modules / bin (normalised)")
        ax.legend(fontsize="small")
        pubhtml.savefig(fig, os.path.join(outdir, "misalignment_locx.pdf"))
        plt.close(fig)

    # -------- the ideal-geometry re-production --------
    if a.ideal and os.path.exists(a.ideal):
        di = load(a.ideal)
        si = di["_sel"]
        dr = load(a.realref) if a.realref else d
        sr = dr["_sel"] if a.realref else sel
        chi2prob_panel([("DY, UL16 MC alignment", dr["_u"][sr]),
                        ("DY, IDEAL geometry (same events)", di["_u"][si]),
                        (r"J/$\psi$ gun (ideal, no pileup)", g["_u"][gs])],
                       outdir, "dy_ideal")
        density_panel(di["bsz"][si, 0],
                      np.sqrt(np.maximum(di["_chired"][si], 1e-12)),
                      outdir, "z1_dy_ideal",
                      r"$z_1$ -- DY refitted with the IDEAL geometry")
        ladder_panel([(r"DY aligned $z_1$", dr["_chired"][sr], dr["bsz"][sr, 0]),
                      (r"DY ideal $z_1$", di["_chired"][si], di["bsz"][si, 0])],
                     outdir, "ideal")
        # the chi2/ndof density, the quantity the whole tail is
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        edges = np.linspace(0.2, 3.0, 71)
        for lab, x, c in ((f"DY, UL16 MC alignment (N = {int(sr.sum())})",
                           dr["_chired"][sr], "#d62728"),
                          ("DY, IDEAL geometry, same events",
                           di["_chired"][si], "#1f77b4"),
                          (r"J/$\psi$ gun (ideal, no pileup)",
                           g["_chired"][gs], "#2ca02c")):
            ax.hist(np.clip(x, edges[0], edges[-1]), bins=edges, density=True,
                    histtype="step", lw=2.0, color=c, label=lab)
        ax.set_yscale("log"); ax.set_ylim(1e-3, 4.0)
        ax.set_xlabel(r"$\chi^2/\mathrm{ndof}$  (the constraint rows removed)")
        ax.set_ylabel("probability density")
        ax.legend(fontsize="small")
        pubhtml.savefig(fig, os.path.join(outdir, "chi2ndof_ideal.pdf"))
        plt.close(fig)
    logger.info("done")


if __name__ == "__main__":
    main()
