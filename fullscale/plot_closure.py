#!/usr/bin/env python3
"""The closure summary of the certified table, one panel per file.

Reads `certtable.py`'s ledger (so a point is only drawn when it PASSES the
four-part acceptance test of STATE sec. 0f.16 -- a fit that stopped early is
drawn hollow and labelled, never silently) and makes:

  * `mz_eta_<form>.png`  -- `m_Z` closure against the leading-muon |eta| band,
    m form and v form on the same axes, with the fit-free prediction of
    sec. 0f.1 overlaid and a PULL panel underneath;
  * `mz_eta_safe.png`    -- the SAME `eta` bands with the two band DEFINITIONS
    against each other: the `|eta|` of the RECO-leading leg (a cut on the
    residual, `corr = +0.0203`) and `max(|eta_p|,|eta_m|)` (`corr = +0.0025`),
    with the pre-registered prediction of STATE 0f.66 overlaid. The two
    definitions select different candidates, so the comparison is the SPREAD;
  * `gz_kladder.png`     -- `Gamma_Z` against the number of `K(m)` terms;
  * `mz_variants.png`    -- the inclusive `m_Z` ladder over the model variants.

usage:  python3 plot_closure.py [-o OUTDIR]
"""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.gridspec import GridSpec

from wums import logging

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certtable as CT  # noqa: E402

# the fit-free prediction of sec. 0f.1, MeV: conditioning on sigma vs on k
PREDICT = {"z_full380_fl": -15.33, "z_M_etaB": -12.04,
           "z_M_etaT": -16.50, "z_M_etaE": -25.61}
PREDICT_V = {"z_V_full": +0.30, "z_V_etaB": -0.23,
             "z_V_etaT": +0.44, "z_V_etaE": +0.20}
# STATE 0f.66, RECORDED BEFORE THE FITS LANDED: `dm/m = -A`, with the per-leg
# charge-even momentum bias `A` measured in the SAME cells the card selects
# (`legscale.py`, 5 % trim, bootstrap over candidates).
PREDICT_LEAD = {"z_V_etaB": -23.4, "z_V_etaT": +0.6, "z_V_etaE": +44.4}
PREDICT_SAFE = {"z_VX_etaB": +4.3, "z_VX_etaT": -4.6, "z_VX_etaE": -3.9}


def ledger(edm_tol=1e-3, nll_tol=0.01):
    """{card: (row, quotable)} -- the best fit of each card and its verdict."""
    R = CT.rows()
    default = {}
    for r in R:
        if r["model"] is not None:
            default.setdefault(r["card"], {}).setdefault(r["model"], 0)
            default[r["card"]][r["model"]] += 1
    default = {c: max(v, key=v.get) for c, v in default.items()}
    for r in R:
        if r["model"] is None:
            r["model"] = default.get(r["card"])
        r["is_default"] = r["model"] == default.get(r["card"])
    by = {}
    for r in R:
        by.setdefault(r["card"], []).append(r)
    best = {c: min((x["nll"] for x in v
                    if x["nll"] is not None and x["is_default"]), default=None)
            for c, v in by.items()}
    out = {}
    for c, v in by.items():
        cand = [r for r in v if r["nll"] is not None and r["is_default"]]
        if not cand:
            continue
        def ok(r):
            if r["edm"] is None or not np.isfinite(r["edm"]) or r["edm"] > edm_tol:
                return False
            if best[c] is not None and r["nll"] > best[c] + nll_tol:
                return False
            if r.get("step") is not None and abs(r["step"]) > 0.045:
                return False
            return True
        good = [r for r in cand if ok(r)]
        pick = min(good or cand, key=lambda x: x["nll"])
        out[c] = (pick, bool(good))
    return out


def band(ax, cards, L, q, colour, label, marker, keep=None):
    """`keep` selects WHICH cards this series draws, at their index in `cards`.

    Without it both the `m` and the `v` call drew EVERY card and the second
    overplotted the first, so every panel showed one series in the other's
    colour (the legend still claimed two).
    """
    x, y, e, hollow = [], [], [], []
    for i, c in enumerate(cards):
        if c not in L or (keep is not None and c not in keep):
            continue
        r, good = L[c]
        if q not in r:
            continue
        x.append(i)
        y.append(r[q][0])
        e.append(r[q][1])
        hollow.append(not good)
    x, y, e, hollow = map(np.asarray, (x, y, e, hollow))
    if not len(x):
        return x, y, e, hollow
    m = ~hollow
    if m.any():
        ax.errorbar(x[m], y[m], yerr=e[m], fmt=marker, color=colour,
                    label=label, markersize=9, capsize=4, lw=2)
    if (~m).any():
        ax.errorbar(x[~m], y[~m], yerr=e[~m], fmt=marker, color=colour,
                    markerfacecolor="none", markersize=9, capsize=4, lw=2,
                    alpha=0.45, label=f"{label} (NOT converged)")
    return x, y, e, hollow


def panel_bandpair(outdir, name):
    """The two band DEFINITIONS on one axis (STATE 0f.63/0f.66/0f.72.2).

    Both card families are v-form, so `panel`'s m/v split cannot separate them
    -- the series are keyed on the card list instead, and each carries its own
    pre-registered prediction. The x positions are the bands; the two
    definitions select DIFFERENT candidates, so the readable quantity is the
    SPREAD of each series, not any single point.
    """
    L = ledger()
    lead = ["z_V_etaB", "z_V_etaT", "z_V_etaE"]
    safe = ["z_VX_etaB", "z_VX_etaT", "z_VX_etaE"]
    ticks = [r"$|\eta|<0.9$", r"$0.9-1.6$", r"$1.6-3.0$"]
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    for cards, colour, mk, lab, pred, dx in (
            (lead, "#1f77b4", "o",
             r"band $=|\eta|$ of the RECO-leading leg   "
             r"($\rho=+0.020$)", PREDICT_LEAD, -0.06),
            (safe, "#d62728", "s",
             r"band $=\max(|\eta_+|,|\eta_-|)$   ($\rho=+0.0025$)",
             PREDICT_SAFE, +0.06)):
        x, y, e, ho = [], [], [], []
        for i, c in enumerate(cards):
            if c not in L:
                continue
            r, good = L[c]
            if "m_Z" not in r:
                continue
            x.append(i + dx); y.append(r["m_Z"][0]); e.append(r["m_Z"][1])
            ho.append(not good)
        x, y, e, ho = map(np.asarray, (x, y, e, ho))
        if len(x):
            m = ~ho
            if m.any():
                ax.errorbar(x[m], y[m], yerr=e[m], fmt=mk, color=colour,
                            label=lab, markersize=9, capsize=4, lw=2)
            if (~m).any():
                ax.errorbar(x[~m], y[~m], yerr=e[~m], fmt=mk, color=colour,
                            markerfacecolor="none", markersize=9, capsize=4,
                            lw=2, alpha=0.45, label=f"{lab} (NOT converged)")
            sp = float(np.nanmax(y) - np.nanmin(y)) if len(y) > 1 else np.nan
            logger.info(f"{name}: {cards[0][:6]} spread {sp:+.1f} MeV")
        ax.plot([i + dx for i in range(len(cards))],
                [pred.get(c, np.nan) for c in cards], "v", color=colour,
                alpha=0.45, markersize=8, ls=":",
                label=r"  $\hookrightarrow$ its prediction $-A$"
                      " (pre-registered)")
    ax.axhline(0.0, color="k", lw=1.2, ls="--")
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, fontsize=14)
    ax.set_xlim(-0.5, len(ticks) - 0.5)
    ax.set_ylabel(r"$m_Z^{\rm fit}-m_Z^{\rm gen}$  [MeV]")
    ax.set_title(r"the $\eta$ band spread is the band SELECTOR", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    os.makedirs(outdir, exist_ok=True)
    o = os.path.join(outdir, name)
    fig.savefig(o, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.info(f"wrote {o}")


def panel(outdir, name, cards, ticks, q, title, ylabel, predict=None,
          predict_v=None):
    L = ledger()
    fig = plt.figure(figsize=(9.0, 7.5))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax, rx = fig.add_subplot(gs[0]), None
    mcards = [c for c in cards if not c.startswith(("z_V", "z_VK"))]
    vcards = [c for c in cards if c.startswith(("z_V", "z_VK"))]
    xm = band(ax, cards, L, q, "#1f77b4", "m formulation", "o", keep=mcards)
    xv = band(ax, cards, L, q, "#d62728", "v formulation  (p = 1.264)", "s",
              keep=vcards)
    ax.axhline(0.0, color="k", lw=1.2, ls="--")
    if predict:
        yy = [predict.get(c, np.nan) for c in cards]
        ax.plot(range(len(cards)), yy, "v", color="#1f77b4", alpha=0.5,
                markersize=8, ls=":", label="fit-free prediction, m form")
    if predict_v:
        yy = [predict_v.get(c, np.nan) for c in cards]
        ax.plot(range(len(cards)), yy, "^", color="#d62728", alpha=0.5,
                markersize=8, ls=":", label="fit-free prediction, v form")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(ticks)))
    ax.tick_params(labelbottom=False)
    ax.set_xlim(-0.5, len(ticks) - 0.5)
    ax.legend(fontsize=13, ncol=1, loc="best")
    ax.set_title(title, fontsize=16)

    rx = fig.add_subplot(gs[1], sharex=ax)
    for (x, y, e, h), col, mk in ((xm, "#1f77b4", "o"), (xv, "#d62728", "s")):
        if not len(x):
            continue
        p = np.where(e > 0, y / e, np.nan)
        rx.errorbar(x[~h], p[~h], fmt=mk, color=col, markersize=8)
        rx.errorbar(x[h], p[h], fmt=mk, color=col, markersize=8,
                    markerfacecolor="none", alpha=0.45)
    rx.axhline(0.0, color="k", lw=1.2, ls="--")
    for s in (-1, 1):
        rx.axhline(s, color="grey", lw=0.8, ls=":")
    rx.set_ylabel("pull")
    rx.set_xticks(range(len(ticks)))
    rx.set_xticklabels(ticks, fontsize=13)
    rx.set_xlim(-0.5, len(ticks) - 0.5)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, name)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.info(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=None)
    a = ap.parse_args()
    out = a.outdir or os.path.expanduser(
        "~/public_html/cvh/"
        + datetime.date.today().strftime("%y%m%d") + "_fullscale")

    panel(out, "mz_eta.png",
          ["z_M_etaB", "z_V_etaB", "z_M_etaT", "z_V_etaT", "z_M_etaE", "z_V_etaE"],
          [r"$|\eta|<0.9$", "", r"$0.9-1.6$", "", r"$1.6-3.0$", ""],
          "m_Z", r"$m_Z$ closure per $|\eta_{\rm lead}|$ band",
          r"$m_Z^{\rm fit}-m_Z^{\rm gen}$  [MeV]",
          predict=PREDICT, predict_v=PREDICT_V)
    panel_bandpair(out, "mz_eta_safe.png")
    panel(out, "mz_sigmasplit_safe.png",
          ["z_V_etaB_slo", "z_V_etaB_shi", "z_VX_etaB_slo", "z_VX_etaB_shi"],
          [r"lead band, low $\sigma/m$", r"lead band, high",
           r"safe band, low", r"safe band, high"],
          "m_Z", r"the barrel $\sigma/m$ split, both band definitions",
          r"$m_Z^{\rm fit}-m_Z^{\rm gen}$  [MeV]")
    panel(out, "gz_kladder.png",
          ["z_full380_fl", "z_V_full", "z_full380_fl_s6", "z_V_s6",
           "z_full380_fl_s7", "z_V_s7", "z_full380_fl_s9", "z_V_s9",
           "z_full380_fl_s12", "z_V_s12"],
          ["5 terms", "", "6 terms", "", "7 terms", "", "9 terms", "",
           "12 terms", ""],
          "Gamma_Z", r"$\Gamma_Z$ against the $K(m)$ truncation",
          r"$\Gamma_Z^{\rm fit}-\Gamma_Z^{\rm gen}$  [MeV]")
    panel(out, "mz_kladder.png",
          ["z_full380_fl", "z_V_full", "z_full380_fl_s6", "z_V_s6",
           "z_full380_fl_s7", "z_V_s7", "z_full380_fl_s9", "z_V_s9",
           "z_full380_fl_s12", "z_V_s12"],
          ["5 terms", "", "6 terms", "", "7 terms", "", "9 terms", "",
           "12 terms", ""],
          "m_Z", r"$m_Z$ against the $K(m)$ truncation",
          r"$m_Z^{\rm fit}-m_Z^{\rm gen}$  [MeV]")
    panel(out, "mz_sigmasplit.png",
          ["z_V_etaB_slo", "z_V_etaB_shi", "z_V_etaE_slo", "z_V_etaE_shi"],
          [r"barrel low $\sigma/m$", r"barrel high", r"endcap low",
           r"endcap high"],
          "m_Z", r"$m_Z$ against $\sigma/m$ AT FIXED $\eta$",
          r"$m_Z^{\rm fit}-m_Z^{\rm gen}$  [MeV]")
    panel(out, "mz_variants.png",
          ["z_full380_fl", "z_V_full", "z_F_dc8", "z_F_toy", "z_V_toy",
           "z_F_toydc", "z_F_w70110"],
          ["baseline", "", r"$\sigma$-reweighted", "toy", "", "toy+dc",
           "70-110 GeV"],
          "m_Z", r"$m_Z$ closure over the model variants",
          r"$m_Z^{\rm fit}-m_Z^{\rm gen}$  [MeV]")


if __name__ == "__main__":
    main()
