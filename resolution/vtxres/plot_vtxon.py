#!/usr/bin/env python3
"""Figures of what the VERTEX CONSTRAINT does, ON against OFF.

Reads the matched-candidate dump of `cmp_vtxon.py --dump`, so every panel is
the SAME candidate in the two regimes.

--gain      sigma_m(ON)/sigma_m(OFF) against the prediction sqrt(1-rho^2),
            with rho = cov(m, theta_6)/(sigma_m(unc) sigma_v) read off the
            ON export alone -- and the same gain against the GEN pT of the
            softer muon.
--rho       the per-candidate mass-vertex correlation: the free regime's
            influence-vector correlation, the constrained regime's (which is
            zero by construction), and the exported `rho`.
--dca       the one-Newton-step identity: r_v(ON) against r_v(OFF).
--dy        the DY outlier population: what the constraint does to the ~2 %
            of candidates the CF does not model, and how far the constrained
            mass moves for them -- the argument for exporting both masses.

One file per panel, into ``~/public_html/ZMass/cvh/<YYMMDD>_vtxon/``.
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

from wums import logging  # noqa: E402
import pubhtml  # noqa: E402
import ratiopanel  # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def _hist(x, edges):
    h, _ = np.histogram(x, bins=edges)
    w = np.diff(edges)
    return h / max(h.sum(), 1) / w, np.sqrt(h) / max(h.sum(), 1) / w


def gain(d, outdir):
    ratio = d["sigma_m_on"] / d["sigma_m_off"]
    smu = np.sqrt(d["sigma_m_on"] ** 2 + (d["covmassvtx_on"] / d["sigma_v_on"]) ** 2)
    rho = d["covmassvtx_on"] / (smu * d["sigma_v_on"])
    pred = np.sqrt(1. - rho ** 2)

    # --- the distribution of the gain, measured against predicted
    edges = np.linspace(0.90, 1.0005, 71)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    dm, em = _hist(np.clip(ratio, edges[0], edges[-1]), edges)
    dp, _ = _hist(np.clip(pred, edges[0], edges[-1]), edges)
    fig, ax, rax = ratiopanel.make_ratio_fig()
    ax.errorbar(ctr, dm, yerr=em, fmt="ko", ms=3.0, lw=1.0,
                label=r"measured $\sigma_m(\mathrm{ON})/\sigma_m(\mathrm{OFF})$")
    ax.step(edges[:-1], dp, where="post", color="#d62728", lw=1.8,
            label=r"predicted $\sqrt{1-\rho^2}$ from the ON export alone")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("the mass resolution gained by constraining the vertex "
                 f"(mean {ratio.mean():.5f} vs {pred.mean():.5f})", fontsize=13)
    r = dm / np.where(dp > 0, dp, np.nan)
    rax.errorbar(ctr, r, yerr=em / np.where(dp > 0, dp, np.nan),
                 fmt="ko", ms=3.0, lw=1.0)
    rax.axhline(1.0, color="#d62728", lw=1.2)
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("meas / pred", fontsize=10)
    rax.set_xlabel(r"$\sigma_m(\mathrm{ON})\,/\,\sigma_m(\mathrm{OFF})$")
    fn = os.path.join(outdir, "sigma_m_gain.pdf")
    pubhtml.savefig(fig, fn); plt.close(fig); logger.info(f"wrote {fn}")

    # --- the gain vs GEN pT of the softer muon
    gpt = d["genpt_soft"]
    ed = np.percentile(gpt, np.linspace(0, 100, 13))
    xc, gm, gp, ge = [], [], [], []
    for b in range(len(ed) - 1):
        m = (gpt >= ed[b]) & (gpt < ed[b + 1])
        if m.sum() < 20:
            continue
        xc.append(np.median(gpt[m]))
        gm.append(100 * (1 - ratio[m].mean()))
        gp.append(100 * (1 - pred[m].mean()))
        ge.append(100 * ratio[m].std() / np.sqrt(m.sum()))
    xc, gm, gp, ge = map(np.asarray, (xc, gm, gp, ge))
    fig, ax, rax = ratiopanel.make_ratio_fig()
    ax.errorbar(xc, gm, yerr=ge, fmt="ko", ms=5.0, lw=1.2, label="measured")
    ax.plot(xc, gp, "-", color="#d62728", lw=1.8,
            label=r"predicted, $1-\sqrt{1-\rho^2}$")
    ax.set_ylabel(r"gain in $\sigma_m$  [%]")
    ax.set_xscale("log")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("the vertex constraint buys more at higher momentum", fontsize=13)
    rax.errorbar(xc, gm / gp, yerr=ge / gp, fmt="ko", ms=5.0, lw=1.2)
    rax.axhline(1.0, color="#d62728", lw=1.2)
    rax.set_ylim(0.9, 1.1)
    rax.set_ylabel("meas / pred", fontsize=10)
    rax.set_xlabel(r"GEN $p_T$ of the softer muon  [GeV]")
    fn = os.path.join(outdir, "sigma_m_gain_vs_genpt.pdf")
    pubhtml.savefig(fig, fn); plt.close(fig); logger.info(f"wrote {fn}")


def rho_panel(d, c, outdir):
    smu = np.sqrt(d["sigma_m_on"] ** 2 + (d["covmassvtx_on"] / d["sigma_v_on"]) ** 2)
    rho = d["covmassvtx_on"] / (smu * d["sigma_v_on"])
    edges = np.linspace(-0.6, 0.6, 97)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig = plt.figure(figsize=(10., 7.))
    ax = fig.add_subplot(111)
    for nm, x, col, lab in (
            ("off", c["corr_OFF"], "#1f77b4",
             r"free fit, $\mathrm{corr}(w_m, w_v)$ per candidate"),
            ("rho", rho, "#d62728",
             r"$\rho = \mathrm{cov}(m,\theta_6)/(\sigma_m^{\rm unc}\sigma_v)$, ON export"),
            ("on", c["corr_ON"], "#2ca02c",
             r"constrained fit, $\mathrm{corr}(w_m, w_v)$ per candidate")):
        h, e = _hist(np.clip(x, edges[0], edges[-1]), edges)
        ax.step(edges[:-1], h, where="post", color=col, lw=1.8, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("per-candidate mass-vertex correlation")
    ax.set_ylabel("density")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("the constraint removes the correlation the free fit carries "
                 f"(rms {np.sqrt((c['corr_OFF']**2).mean()):.3f} "
                 rf"$\to$ {np.sqrt((c['corr_ON']**2).mean()):.1e})", fontsize=13)
    fn = os.path.join(outdir, "mass_vertex_correlation.pdf")
    pubhtml.savefig(fig, fn); plt.close(fig); logger.info(f"wrote {fn}")


def dca_panel(d, outdir):
    dz = np.abs(d["r_v_on"] - d["r_v_off"]) / d["sigma_v_off"]
    edges = np.logspace(-6, 1, 71)
    h, _ = np.histogram(np.clip(dz, edges[0], edges[-1]), bins=edges)
    fig = plt.figure(figsize=(10., 7.))
    ax = fig.add_subplot(111)
    ax.step(edges[:-1], h / h.sum(), where="post", color="#d62728", lw=1.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|r_v(\mathrm{ON}) - r_v(\mathrm{OFF})|\,/\,\sigma_v$")
    ax.set_ylabel("fraction of candidates")
    ax.axvline(np.median(dz), color="k", ls="--", lw=1.2,
               label=f"median {np.median(dz):.2e}")
    ax.axvline(np.percentile(dz, 90), color="k", ls=":", lw=1.2,
               label=f"p90 {np.percentile(dz, 90):.2e}")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("the one-Newton-step identity for the DCA", fontsize=13)
    fn = os.path.join(outdir, "dca_identity.pdf")
    pubhtml.savefig(fig, fn); plt.close(fig); logger.info(f"wrote {fn}")


def dy_panel(on, off, outdir):
    """The DY tail, and what the vertex constraint does to the mass there."""
    z_on, z_off = on["vtxz"], off["vtxz"]
    out = {}
    for tag, z in (("ON", z_on), ("OFF", z_off)):
        pass
    # MATCHED, because the two productions do not select the same candidates:
    # the constrained fit converges on events the free one produced nothing
    # for, so the inclusive tails are over different samples.
    # Only events with exactly ONE selected candidate in EACH regime are
    # paired: a (run, lumi, event) with two candidates would otherwise pair
    # the mu+mu- combination of one regime with a different one of the other.
    key = lambda d: [tuple(int(x) for x in t) for t in
                     np.stack([d[k] for k in ("run", "lumi", "event")], 1)]
    kon, koff = key(on), key(off)
    from collections import Counter
    con, coff = Counter(kon), Counter(koff)
    mo = {k: j for j, k in enumerate(koff) if coff[k] == 1}
    pair = [(i, mo[k]) for i, k in enumerate(kon)
            if con[k] == 1 and k in mo]
    ion = np.array([p_[0] for p_ in pair]); iof = np.array([p_[1] for p_ in pair])
    logger.info(f"  DY matched {ion.size} of ON {z_on.size} / OFF {z_off.size}")
    for tag, z in (("ON  (matched)", z_on[ion]), ("OFF (matched)", z_off[iof]),
                   ("ON  (all)", z_on), ("OFF (all)", z_off)):
        out[tag] = {t: float(np.mean(np.abs(z) > t)) for t in (1, 2, 3, 4, 5)}
        logger.info(f"  DY {tag}: n {z.size}  " + "  ".join(
            f"P(|z|>{t}) {out[tag][t]*100:.3f} %" for t in (1, 2, 3, 4, 5)))
    # WHERE the inclusive tail lives: events with more than one selected
    # dimuon candidate are combinatorial pairings, not a resolution statement.
    for tag, d_, c_, k_ in (("ON", on, con, kon), ("OFF", off, coff, koff)):
        mult = np.array([c_[k] > 1 for k in k_])
        z = d_["vtxz"]
        for nm, m in (("1 cand/event", ~mult), (">1 cand/event", mult)):
            if m.sum():
                logger.info(f"  DY {tag} {nm}: n {int(m.sum())}  "
                            f"P(|z|>5) {np.mean(np.abs(z[m])>5)*100:.3f} %  "
                            f"P(|z|>3) {np.mean(np.abs(z[m])>3)*100:.3f} %")
    # the ON-only candidates -- the events the free fit produced nothing for
    onlyon = np.array([i for i, k in enumerate(kon) if k not in set(koff)])
    if onlyon.size:  # noqa
        z = z_on[onlyon]
        logger.info(f"  DY ON-ONLY (the free fit produced nothing): n {z.size}  "
                    + "  ".join(f"P(|z|>{t}) {np.mean(np.abs(z)>t)*100:.3f} %"
                                for t in (1, 2, 3, 4, 5)))
    dm = (on["mass"] - on["mass_unc"]) * 1e3            # MeV
    sm = on["sigmamass"] * 1e3
    tail = np.abs(z_on) > 5
    core = np.abs(z_on) < 3
    for nm, m in (("core |z_v| < 3", core), ("TAIL |z_v| > 5", tail)):
        logger.info(f"  DY {nm}: n {int(m.sum())}  "
                    f"m_c - m_unc mean {dm[m].mean():+.3f} MeV  "
                    f"rms {dm[m].std():.3f} MeV  "
                    f"median |.| {np.median(np.abs(dm[m])):.3f} MeV  "
                    f"in units of sigma_m: median "
                    f"{np.median(np.abs(dm[m]) / sm[m]):.4f}  "
                    f"p90 {np.percentile(np.abs(dm[m]) / sm[m], 90):.4f}")
    # in units of sigma_m and on a log axis: the shift spans four decades,
    # so a linear MeV axis is all clipping.
    edges = np.logspace(-4, 2, 61)
    fig = plt.figure(figsize=(10., 7.))
    ax = fig.add_subplot(111)
    for nm, m, col in (("core, $|z_v| < 3$", core, "#1f77b4"),
                       ("tail, $|z_v| > 5$", tail, "#d62728")):
        x = np.clip(np.abs(dm[m]) / sm[m], edges[0], edges[-1])
        h, _ = np.histogram(x, bins=edges)
        ax.step(edges[:-1], h / max(h.sum(), 1), where="post", color=col, lw=1.8,
                label=f"{nm}  ({int(m.sum())} cand, median "
                      f"{np.median(np.abs(dm[m]) / sm[m]):.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|m_{\rm constrained} - m_{\rm unconstrained}|\,/\,\sigma_m$")
    ax.set_ylabel("fraction of candidates")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("DY: the vertex constraint moves the mass of the outliers, "
                 "not of the core", fontsize=13)
    fn = os.path.join(outdir, "dy_mass_shift.pdf")
    pubhtml.savefig(fig, fn); plt.close(fig); logger.info(f"wrote {fn}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default=None)
    p.add_argument("--corr", default=None)
    p.add_argument("--outpath", default=None)
    p.add_argument("--gain", action="store_true")
    p.add_argument("--rho", action="store_true")
    p.add_argument("--dca", action="store_true")
    p.add_argument("--dy", nargs=2, default=None,
                   metavar=("ON_NPZ", "OFF_NPZ"),
                   help="the DY vertex npz of the two regimes")
    a = p.parse_args()
    logging.setup_logger(__file__, 3, False)
    if not (a.gain or a.rho or a.dca or a.dy):
        a.gain = a.rho = a.dca = True
    day = datetime.date.today().strftime("%y%m%d")
    outdir = a.outpath or pubhtml.figdir("vtxon", day)
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir)
    if a.dy:
        dy_panel(np.load(a.dy[0]), np.load(a.dy[1]), outdir)
        if not (a.gain or a.rho or a.dca):
            return
    d = np.load(a.dump)
    if a.gain:
        gain(d, outdir)
    if a.dca:
        dca_panel(d, outdir)
    if a.rho:
        c = np.load(a.corr or a.dump.replace(".npz", "") + "_corr.npz")
        rho_panel(d, c, outdir)


if __name__ == "__main__":
    main()
