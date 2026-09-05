#!/usr/bin/env python3
"""alpha vs selection / truncation treatment -- the summary figure of the
2026-09-04 censoring test."""
import datetime, glob, json, os
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging
import pubhtml

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def get(name):
    f = f"runs/masslikfit_{name}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    mt = json.loads(str(d["meta"]))
    P = dict(zip(mt["parnames"], zip(d["fit_x"], d["fit_err"])))
    return P, int(mt["n"])


def panel(rows, title, fname, outdir, xlab=r"$\alpha$ [$10^{-3}$]"):
    lab, v, e, col = [], [], [], []
    for l_, nm, c in rows:
        g = get(nm)
        if g is None:
            logger.warning(f"missing {nm}")
            continue
        P, n = g
        a, ae = P["alpha[1e-3]"]
        lab.append(f"{l_}  (n={n/1e3:.0f}k)")
        v.append(a); e.append(ae); col.append(c)
    if not v:
        return
    y = np.arange(len(v))[::-1]
    fig, ax = plt.subplots(figsize=(10.2, 0.62 * len(v) + 2.4))
    ax.axvline(0., color="0.7", lw=1.2)
    for i in range(len(v)):
        ax.errorbar(v[i], y[i], xerr=e[i], marker="o", ms=7, color=col[i],
                    elinewidth=2., capsize=3.)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize="small")
    ax.set_ylim(-0.8, len(v) - 0.2)
    ax.set_xlabel(xlab)
    ax.set_title(title, fontsize="medium")
    ax.grid(axis="x", alpha=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"-> {outdir}/{fname}.{{pdf,png}}")


def main():
    day = datetime.date.today().strftime("%y%m%d")
    outdir = os.path.expanduser(f"~/public_html/cvh/{day}_censoring")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)

    panel([
        ("gun: full sample, no window", "jpsigun_260903x_fam_krad1", "C0"),
        (r"gun: $\sigma_m<0.15$ (`sane`), no window", "gun_260904_sane_fam", "C0"),
        (r"gun: sane + hand cut $\pm$0.15 GeV, NO correction", "gun_260904_cut015_fam", "C3"),
        (r"gun: sane + hand cut $\pm$0.15 GeV, window term", "gun_260904_cut015_fam_win", "C2"),
        (r"gun: sane + hand cut $\pm$0.05 GeV, NO correction", "gun_260904_cut005_fam", "C3"),
        (r"gun: sane + hand cut $\pm$0.05 GeV, window term", "gun_260904_cut005_fam_win", "C2"),
    ], "CONTROLLED TRUNCATION on the J/$\\psi$ gun (no window in its chain):\n"
       "a 4.6$\\sigma$ cut costs +0.02e-3, a 1.5$\\sigma$ cut costs what the "
       "figure shows; the window term recovers it",
       "alpha_controlled_truncation_gun", outdir)

    panel([
        ("v3: full sample, no window", "btojpsix_v3_260903x_fam_krad1", "C0"),
        ("v3: sane+window subset, no window term", "btojpsix_v3_260904_fam_sane", "C0"),
        ("v3: same subset, window term 2.95-3.25", "btojpsix_v3_260904_fam_sane_win", "C2"),
        ("v3: window term, 2.90-3.30", "v3_260904_win290_330_fam", "C2"),
        ("v3: window term, 2.85-3.35", "v3_260904_win285_335_fam", "C2"),
        ("v3: wide window 2.0-4.2 (null)", "btojpsix_v3_260904_fam_winwide", "C1"),
    ], "B$\\to$J/$\\psi$+X v3: the real 2.95--3.25 GeV cut is on the PRE-refit\n"
       "mass, so the assumed post-refit edge is an ambiguity of this size",
       "alpha_v3_window_width", outdir)

    panel([
        ("gun: sane (all)", "gun_260904_sane_fam", "C0"),
        (r"gun: $\chi^2/$ndof$<3$", "gun_260904_q3_fam", "C2"),
        (r"gun: $\chi^2<3$, not clamped, niter$<10$", "gun_260904_q3nf_fam", "C2"),
        ("gun: not clamped", "gun_260904_nofrozen_fam", "C4"),
        (r"gun: min daughter $p_{\rm T}>2$ GeV", "gun_260904_pt2_fam", "C1"),
        (r"gun: min daughter $p_{\rm T}>3$ GeV", "gun_260904_pt3_fam", "C1"),
        (r"gun: $p_{\rm T}>3$ and $\chi^2<3$", "gun_260904_pt3q3_fam", "C1"),
        ("v3: sane (all)", "v3_260904_sane_fam", "C0"),
        (r"v3: $\chi^2/$ndof$<3$", "v3_260904_q3_fam", "C2"),
        (r"v3: $p_{\rm T}>3$ and $\chi^2<3$", "v3_260904_pt3q3_fam", "C1"),
    ], "the FIT-PATHOLOGY population: 12 % of gun candidates have "
       "$\\chi^2/$ndof$>10$,\nall with a daughter below the 2 GeV "
       "Gauss-Newton momentum floor",
       "alpha_vs_fit_quality", outdir)

    # the same ladder in the SINGLE-r model, where alpha is a pure location
    # parameter and does not trade against the family scales
    panel([
        ("gun: sane (all)", "gun_260904_sane_r", "C0"),
        (r"gun: $\chi^2/$ndof$<3$", "gun_260904_q3_r", "C2"),
        (r"gun: $\chi^2<3$, not clamped, niter$<10$", "gun_260904_q3nf_r", "C2"),
        ("gun: not clamped", "gun_260904_nofrozen_r", "C4"),
        (r"gun: min daughter $p_{\rm T}>2$ GeV", "gun_260904_pt2_r", "C1"),
        (r"gun: min daughter $p_{\rm T}>3$ GeV", "gun_260904_pt3_r", "C1"),
        (r"gun: $p_{\rm T}>3$ and $\chi^2<3$", "gun_260904_pt3q3_r", "C1"),
    ], r"single-$r$ model ($\alpha$ is a pure location parameter):"
       "\n"
       r"the whole $+0.21\times10^{-3}$ offset is the clamped / high-$\chi^2$ "
       r"low-$p_{\rm T}$ population",
       "alpha_vs_fit_quality_rmodel", outdir)
    logger.info(f"figures in {outdir}")


if __name__ == "__main__":
    logger = logging.setup_logger(__file__, 3, False)
    main()
