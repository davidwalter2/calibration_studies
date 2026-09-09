#!/usr/bin/env python3
"""Figures for the CONVERGENCE variants. One file per panel, ratio panel under
every comparison, index.php written by `pubhtml.ensure_index`."""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conv_common as cc                                          # noqa: E402
import pubhtml                                                    # noqa: E402
from wums import logging                                          # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
BANDLAB = [r"$|\eta|<0.9$", r"$0.9<|\eta|<1.6$", r"$1.6<|\eta|<2.4$"]
COL = {"base": "k", "tight": "#d62728", "damp": "#1f77b4"}
MRK = {"base": "o", "tight": "s", "damp": "^"}


def save(fig, outdir, name):
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["base", "tight", "damp"])
    a = ap.parse_args()
    day = datetime.date.today().strftime("%y%m%d")
    outdir = os.path.expanduser(f"~/public_html/cvh/{day}_hitclassbias")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    rng = np.random.default_rng(909)
    V = {v: cc.Var(f"data/conv_{v}.npz", v) for v in a.variants}
    ref = V[a.variants[0]]

    # ---- 1. charge-even <x> per eta band, with the PAIRED difference below --
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.07})
    xs = np.arange(3)
    for i, (nm, Vv) in enumerate(V.items()):
        vs, es = [], []
        for ib in range(3):
            v, e, _ = cc.even_mean(Vv.x, Vv.q, Vv.good & (Vv.band == ib), rng)
            vs.append(v * 1e3)
            es.append(e * 1e3)
        ax.errorbar(xs + 0.12 * (i - 1), vs, yerr=es, fmt=MRK[nm], color=COL[nm],
                    ms=8, capsize=4, label=nm)
        if nm == a.variants[0]:
            continue
        ia, ib_ = cc.pair(ref, Vv)
        q = ref.q[ia]
        good = ref.good[ia] & Vv.good[ib_]
        dx = Vv.x[ib_] - ref.x[ia]
        ds, de = [], []
        for jb in range(3):
            m = good & (ref.band[ia] == jb)
            idx = np.where(m)[0]
            val = 0.5 * (dx[idx][q[idx] > 0].mean() + dx[idx][q[idx] < 0].mean())
            bs = np.empty(300)
            for k in range(300):
                s = idx[rng.integers(0, len(idx), len(idx))]
                bs[k] = 0.5 * (dx[s][q[s] > 0].mean() + dx[s][q[s] < 0].mean())
            ds.append(val * 1e3)
            de.append(bs.std() * 1e3)
        rx.errorbar(xs + 0.12 * (i - 1), ds, yerr=de, fmt=MRK[nm], color=COL[nm],
                    ms=8, capsize=4, label=f"{nm} $-$ base")
    ax.axhline(0, color="0.6", lw=1, ls=":")
    rx.axhline(0, color="0.6", lw=1, ls=":")
    ax.set_ylabel(r"charge-even $\langle x\rangle$  [$10^{-3}$]")
    rx.set_ylabel(r"paired $\Delta$  [$10^{-3}$]")
    rx.set_xticks(xs)
    rx.set_xticklabels(BANDLAB)
    ax.legend(fontsize=15)
    rx.legend(fontsize=13)
    ax.set_title("truth-referenced pull, 20-60 GeV muon gun", fontsize=15)
    save(fig, outdir, "conv_band")

    # ---- 2. the mixture, per variant ---------------------------------------
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.07})
    for i, (nm, Vv) in enumerate(V.items()):
        t = np.percentile(Vv.dq_seed[Vv.good], 90.)
        for lab, cut, mk, off in (("IN", Vv.dq_seed > t, "*", 0.0),
                                  ("OUT", Vv.dq_seed <= t, MRK[nm], 0.0)):
            vs, es = [], []
            for ib in range(3):
                v, e, _ = cc.even_mean(Vv.x, Vv.q,
                                       Vv.good & (Vv.band == ib) & cut, rng)
                vs.append(v * 1e3)
                es.append(e * 1e3)
            tgt = ax if lab == "IN" else rx
            tgt.errorbar(xs + 0.12 * (i - 1), vs, yerr=es, fmt=mk, color=COL[nm],
                         ms=10 if lab == "IN" else 8, capsize=4,
                         label=f"{nm} {lab}")
    for t_, y in ((ax, 20.59), (rx, -6.33)):
        t_.axhline(0, color="0.6", lw=1, ls=":")
        t_.axhline(y, color="0.3", lw=1.2, ls="--")
    ax.set_ylabel(r"IN (top 10%) $\langle x\rangle_{even}$  [$10^{-3}$]")
    rx.set_ylabel(r"OUT (bulk) $\langle x\rangle_{even}$  [$10^{-3}$]")
    rx.set_xticks(xs)
    rx.set_xticklabels(BANDLAB)
    ax.legend(fontsize=12, ncol=3)
    rx.legend(fontsize=12, ncol=3)
    ax.set_title(r"mixture split at $p_{90}$ of $|$seed$\to$final $\delta(q/p)|$",
                 fontsize=15)
    save(fig, outdir, "conv_mixture")

    # ---- 3. iteration census ----------------------------------------------
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.07})
    nmax = int(max(Vv.d["niter"].max() for Vv in V.values()))
    edges = np.arange(0.5, nmax + 1.6)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    h0 = None
    for nm, Vv in V.items():
        h, _ = np.histogram(Vv.d["niter"], bins=edges)
        h = h / h.sum()
        ax.step(ctr, h, where="mid", color=COL[nm], lw=2, label=nm)
        if h0 is None:
            h0 = h
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rx.step(ctr, np.where(h0 > 0, h / h0, np.nan), where="mid",
                        color=COL[nm], lw=2, label=f"{nm} / base")
    rx.axhline(1, color="0.6", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("fraction of tracks")
    rx.set_ylabel("ratio to base")
    rx.set_xlabel("Gauss-Newton iterations")
    ax.legend(fontsize=15)
    rx.legend(fontsize=13)
    save(fig, outdir, "conv_niter")

    # ---- 4. the second-order scaling test ----------------------------------
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.07})
    for nm, Vv in V.items():
        gp = Vv.d["genpt"].astype(float)
        qs = np.percentile(gp[Vv.good], [0, 20, 40, 60, 80, 100])
        sr, ev, ee = [], [], []
        for i in range(5):
            m = Vv.good & (gp >= qs[i]) & ((gp < qs[i + 1]) | (i == 4))
            v, e, _ = cc.even_mean(Vv.x, Vv.q, m, rng)
            sr.append(Vv.sigrel[m].mean())
            ev.append(v * 1e3)
            ee.append(e * 1e3)
        sr, ev, ee = map(np.asarray, (sr, ev, ee))
        ax.errorbar(sr, ev, yerr=ee, fmt=MRK[nm], color=COL[nm], ms=8,
                    capsize=4, label=nm)
        if nm == a.variants[0]:
            w = 1 / ee ** 2
            k0 = (w * sr * ev).sum() / (w * sr * sr).sum()
            kA = (w * (1 / sr) * ev).sum() / (w / sr ** 2).sum()
            g = np.linspace(sr.min() * 0.85, sr.max() * 1.05, 100)
            ax.plot(g, k0 * g, "-", color="0.3", lw=1.6,
                    label=fr"2nd order $\propto\sigma_{{rel}}$, $c={k0:+.2f}$")
            ax.plot(g, kA / g, "--", color="0.55", lw=1.6,
                    label=r"location $\propto 1/\sigma_{rel}$")
            rx.errorbar(sr, ev - k0 * sr, yerr=ee, fmt="o", color="0.3", ms=7,
                        capsize=4, label="data $-$ 2nd-order fit")
            rx.errorbar(sr, ev - kA / sr, yerr=ee, fmt="s", color="0.55", ms=7,
                        capsize=4, label="data $-$ location fit")
    ax.axhline(0, color="0.6", lw=1, ls=":")
    rx.axhline(0, color="0.6", lw=1, ls=":")
    ax.set_ylabel(r"charge-even $\langle x\rangle$  [$10^{-3}$]")
    rx.set_ylabel(r"residual  [$10^{-3}$]")
    rx.set_xlabel(r"$\langle\sigma_{rel}\rangle$ in bins of GEN $p_T$")
    ax.legend(fontsize=13)
    rx.legend(fontsize=12)
    save(fig, outdir, "conv_scaling")

    # ---- 5. paired per-track change in q/p ---------------------------------
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.07})
    bins = np.linspace(-8, 8, 121)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for nm, Vv in V.items():
        if nm == a.variants[0]:
            continue
        ia, ib_ = cc.pair(ref, Vv)
        d = (Vv.d["qop_ref"][ib_] - ref.d["qop_ref"][ia]) / ref.sigma[ia]
        h, _ = np.histogram(np.clip(d, bins[0], bins[-1]), bins=bins)
        ax.step(ctr, h / h.sum(), where="mid", color=COL[nm], lw=2,
                label=f"{nm} $-$ base  (rms {d.std():.3f}$\\sigma$)")
        q = ref.q[ia]
        m = ref.good[ia] & Vv.good[ib_]
        prof, err, xc = [], [], []
        eb = np.linspace(-2.4, 2.4, 13)
        for i in range(12):
            s = m & (ref.eta[ia] >= eb[i]) & (ref.eta[ia] < eb[i + 1])
            if s.sum() < 50:
                continue
            v = 0.5 * (d[s & (q > 0)].mean() + d[s & (q < 0)].mean())
            e = d[s].std() / np.sqrt(s.sum())
            prof.append(v * 1e3)
            err.append(e * 1e3)
            xc.append(0.5 * (eb[i] + eb[i + 1]))
        rx.errorbar(xc, prof, yerr=err, fmt=MRK[nm], color=COL[nm], ms=7,
                    capsize=3, label=f"{nm} $-$ base")
    ax.set_yscale("log")
    ax.set_ylabel("fraction of tracks")
    ax.set_xlabel(r"$\Delta(q/p)\,/\,\sigma$")
    rx.axhline(0, color="0.6", lw=1, ls=":")
    rx.set_ylabel(r"$\langle\Delta(q/p)/\sigma\rangle_{even}$ [$10^{-3}$]")
    rx.set_xlabel(r"$\eta$")
    ax.legend(fontsize=13)
    rx.legend(fontsize=12)
    save(fig, outdir, "conv_dqop")

    # ---- 6. the Box prediction against the measurement, both channels ------
    import c4_secondorder as c4                                   # noqa: E402
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1],
                                              "hspace": 0.07})
    ref_full = cc.Var("data/conv_ref903x_full.npz", "baseline 160 tasks")
    d1 = (ref_full.d["qop_it0"] - ref_full.d["qop_seed"]).astype(float)
    d2 = (ref_full.d["qop_ref"] - ref_full.d["qop_it0"]).astype(float)
    g = ref_full.good
    g &= (np.abs(d1) < np.percentile(np.abs(d1[g]), 99)) &          (np.abs(d2) < np.percentile(np.abs(d2[g]), 99))
    xs2 = np.arange(3)
    for ib in range(3):
        pr = {}
        for qq in (+1, -1):
            m = g & (ref_full.band == ib) & (ref_full.q == qq)
            r = c4.fit_quad(d1[m], d2[m])
            pr[qq] = -0.5 * r[2] * ref_full.sigma[m].mean()
        for tgt, val, c, lb in ((ax, 0.5 * (pr[1] + pr[-1]), "#2ca02c",
                                 "Box prediction"),
                                (rx, 0.5 * (pr[1] - pr[-1]), "#2ca02c",
                                 "Box prediction")):
            tgt.plot([ib - 0.3, ib + 0.3], [val * 1e3] * 2, "-", color=c, lw=3,
                     label=lb if ib == 0 else None)
        m = g & (ref_full.band == ib)
        ve, ee, _ = cc.even_mean(ref_full.x, ref_full.q, m, rng)
        vo, eo, _ = cc.odd_mean(ref_full.x, ref_full.q, m, rng)
        ax.errorbar([ib], [ve * 1e3], yerr=[ee * 1e3], fmt="o", color="k", ms=8,
                    capsize=4, label="measured" if ib == 0 else None)
        rx.errorbar([ib], [vo * 1e3], yerr=[eo * 1e3], fmt="o", color="k", ms=8,
                    capsize=4, label="measured" if ib == 0 else None)
    for t_ in (ax, rx):
        t_.axhline(0, color="0.6", lw=1, ls=":")
        t_.legend(fontsize=13)
    ax.set_ylabel(r"charge-EVEN $\langle x\rangle$ (sagitta) [$10^{-3}$]")
    rx.set_ylabel(r"charge-ODD $\langle x\rangle$ (scale) [$10^{-3}$]")
    rx.set_xticks(xs2)
    rx.set_xticklabels(BANDLAB)
    ax.set_title("second-order Gauss-Newton bias vs measurement", fontsize=15)
    save(fig, outdir, "conv_boxbias")

    # ---- 7. phi profile with the n=8 + n=10 harmonic model ---------------
    if "genphi" in ref_full.d:
        import c8_phiquad as c8                                   # noqa: E402
        gg = ref_full.good & (np.abs(ref_full.x) < 10)
        ph = ref_full.d["genphi"].astype(float)
        NB = 24
        eb = np.linspace(-np.pi, np.pi, NB + 1)
        cn = 0.5 * (eb[1:] + eb[:-1])
        vs, es = [], []
        for i in range(NB):
            m = gg & (ph >= eb[i]) & (ph < eb[i + 1])
            xp, xm = ref_full.x[m & (ref_full.q > 0)], ref_full.x[m & (ref_full.q < 0)]
            vs.append(0.5 * (xp.mean() + xm.mean()) * 1e3)
            es.append(0.5 * np.hypot(xp.std() / np.sqrt(len(xp)),
                                     xm.std() / np.sqrt(len(xm))) * 1e3)
        vs, es = np.asarray(vs), np.asarray(es)
        w = 1 / es ** 2
        mu = (vs * w).sum() / w.sum()
        # the UNBINNED n = 8 and n = 10 amplitudes, evaluated on a fine grid
        gfine = np.linspace(-np.pi, np.pi, 600)
        model = np.full_like(gfine, mu)
        for n in (8, 10):
            o = c8.amps(ref_full, gg, ph, n)
            model += o["cos"][0] * np.cos(n * gfine) + o["sin"][0] * np.sin(n * gfine)
        mbin = np.full(NB, mu)
        for n in (8, 10):
            o = c8.amps(ref_full, gg, ph, n)
            mbin += o["cos"][0] * np.cos(n * cn) + o["sin"][0] * np.sin(n * cn)
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1],
                                                  "hspace": 0.07})
        ax.errorbar(cn, vs, yerr=es, fmt="o", color="k", ms=6, capsize=3,
                    label=r"charge-even $\langle x\rangle$, all $\eta$")
        ax.axhline(mu, color="0.45", lw=1.6, ls="--",
                   label=fr"flat, $\mu={mu:+.2f}\times10^{{-3}}$")
        ax.plot(gfine, model, "-", color="#d62728", lw=2,
                label=r"flat $+\ n=8\ +\ n=10$ (unbinned amplitudes)")
        ax.axhline(0, color="0.75", lw=1, ls=":")
        rx.errorbar(cn, (vs - mu) / es, yerr=np.ones(NB), fmt="s", color="0.45",
                    ms=5, capsize=2, label="residual to flat")
        rx.errorbar(cn, (vs - mbin) / es, yerr=np.ones(NB), fmt="o",
                    color="#d62728", ms=5, capsize=2, label="residual to model")
        rx.axhline(0, color="0.75", lw=1, ls=":")
        c2f = float((w * (vs - mu) ** 2).sum())
        c2m = float((w * (vs - mbin) ** 2).sum())
        ax.set_ylabel(r"charge-even $\langle x\rangle$  [$10^{-3}$]")
        rx.set_ylabel("pull")
        rx.set_xlabel(r"generated $\phi$")
        ax.set_title("ideal geometry:  "
                     r"$\chi^2$(flat) $= $" + f"{c2f:.0f}/23,  "
                     + r"$\chi^2$(flat $+\,n=8,10$) $= $" + f"{c2m:.0f}/19",
                     fontsize=14)
        ax.legend(fontsize=12)
        rx.legend(fontsize=11, ncol=2)
        save(fig, outdir, "conv_phi")

        # ---- 8. the harmonic power spectrum --------------------------------
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [1.6, 1],
                                                  "hspace": 0.07})
        ns = np.arange(1, 26)
        sa, se, pw = [], [], []
        for n in ns:
            o = c8.amps(ref_full, gg, ph, n)
            sa.append(o["sin"][0])
            se.append(o["sin"][2])
            pw.append((o["cos"][0] / o["cos"][2]) ** 2
                      + (o["sin"][0] / o["sin"][2]) ** 2)
        ax.errorbar(ns, sa, yerr=se, fmt="o", color="k", ms=6, capsize=3,
                    label=r"$2\langle x\sin(n\phi)\rangle_{even}$")
        ax.axhline(0, color="0.6", lw=1, ls=":")
        for n in (8, 10):
            ax.axvline(n, color="#d62728", lw=1, ls="--", alpha=0.5)
        ax.set_ylabel(r"sine amplitude  [$10^{-3}$]")
        ax.legend(fontsize=13)
        ax.set_title("unbinned charge-even harmonics of the truth-referenced "
                     "pull", fontsize=14)
        rx.bar(ns, pw, color="0.35", width=0.7)
        rx.axhline(2., color="#d62728", lw=1.6, ls="--",
                   label="expectation with no signal (2 dof)")
        rx.set_yscale("log")
        rx.set_ylabel("power  [2 dof]")
        rx.set_xlabel("harmonic $n$")
        rx.legend(fontsize=12)
        save(fig, outdir, "conv_phiharm")
    logger.info(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
