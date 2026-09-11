#!/usr/bin/env python3
"""Two panels for the 2026-09-08 findings, one file each.

  * `am_closed.png`      -- the MEASURED `a/(sigma/m)` per `|eta|` band against
    `1 + vgf` (the spec) and `1 + f_hit - f_ioni` (the Q-matrix closed form of
    `aux_gen.py`), with the residual `measured - form` underneath.  The two
    forms are indistinguishable because the Q matrix's `f_ioni` is 4.5e-06.
  * `mixture_legs.png`   -- the leg-level charge-even odd moment per `|eta|`
    band, split IN / OUT on `|seed -> final dq/p|` at its 90th percentile, with
    the mixing fraction on a second axis and the mixture identity underneath.

Numbers come from `measure_a.py --aux` and `mixture_legs.py`; this script
re-derives them from the caches rather than parsing logs, so the figure and the
table cannot drift apart.

    python3 plot_mixture.py [-o OUTDIR]
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "resolution"))
import pubhtml  # noqa: E402      (savefig: .png twin for every .pdf)
from model_odd_mass import PROBES  # noqa: E402
from mixture_legs import odd, boot_odd, flatness, BANDS  # noqa: E402
from measure_a import wls_slope  # noqa: E402  (ONE implementation)

DEF_OUT = os.path.expanduser(
    "~/public_html/cvh/" + datetime.date.today().strftime("%y%m%d") + "_fullscale")




def panel_am(out):
    P = np.load(os.path.join(HERE, "runs/zpairs_dyv2_full.npz"))
    G = np.load(os.path.join(HERE, "runs/auxgen_dyv2.npz"))
    z = P["z"].astype(np.float64)
    sig = P["sigma"].astype(np.float64)
    m = z * sig + P["eta"].astype(np.float64)
    vgf = P["vgf"].astype(np.float64)
    w = P["w"].astype(np.float64)
    etal = np.maximum(np.abs(P["etap"]), np.abs(P["etam"]))
    fh, fi = G["fhit"].astype(np.float64), G["fioni"].astype(np.float64)
    ok = (np.isfinite(z) & (sig > 0) & (m > 0) & (np.abs(z) < 2.0)
          & (sig / m < 0.10) & np.isfinite(w))
    lns = np.log(sig)
    xs, meas, err, spec, clo = [], [], [], [], []
    # the SAME bands `measure_a.py` quotes -- the top band runs to 3.0, not to
    # 2.4: cut at 2.4 the deficit in it vanishes, so the band edge matters and
    # the two must not drift apart.
    ABANDS = [(0.0, 0.9, r"$|\eta|$ 0.0-0.9"), (0.9, 1.6, "0.9-1.6"),
              (1.6, 3.0, "1.6-3.0")]
    for i, (lo, hi, lab) in enumerate(ABANDS):
        b = ok & (etal >= lo) & (etal < hi)
        sr = np.average((sig / m)[b], weights=w[b])
        s, se = wls_slope(z[b], lns[b], w[b])
        xs.append(i)
        meas.append(s / sr)
        err.append(se / sr)
        spec.append(1.0 + np.average(vgf[b], weights=w[b]))
        clo.append(1.0 + np.average(fh[b], weights=w[b])
                   - np.average(fi[b], weights=w[b]))
    xs = np.array(xs, float)
    meas, err = np.array(meas), np.array(err)
    spec, clo = np.array(spec), np.array(clo)

    fig = plt.figure(figsize=(8.5, 8.0))
    gs = GridSpec(2, 1, height_ratios=[2.4, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax.errorbar(xs, meas, yerr=err, fmt="o", ms=9, color="k",
                label=r"measured  $d\ln\sigma/dz \,/\, (\sigma_m/m)$")
    ax.plot(xs, spec, "s--", ms=9, color="#c1272d", label=r"spec  $1+v_{gf}$")
    ax.plot(xs, clo, "^:", ms=10, color="#0072b2", mfc="none",
            label=r"closed form  $1+f_{hit}-f_{ioni}$  (Q matrix)")
    ax.set_ylabel(r"$a\,/\,(\sigma_m/m)$")
    ax.set_xticks(xs)
    ax.tick_params(labelbottom=False)
    ax.set_xlim(-0.4, 2.4)
    ax.legend(loc="upper center", fontsize=15, frameon=False)
    ax.text(0.02, 0.05, "Z$\\to\\mu\\mu$ MC, DY v2, 3.48 M candidates\n"
            "$f_{ioni}$ (Q matrix) $=4.5\\times10^{-6}$: the two forms coincide",
            transform=ax.transAxes, fontsize=13, va="bottom")
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.axhline(0.0, color="k", lw=1)
    ax2.errorbar(xs, meas - spec, yerr=err, fmt="s", ms=8, color="#c1272d")
    ax2.errorbar(xs + 0.06, meas - clo, yerr=err, fmt="^", ms=9,
                 color="#0072b2", mfc="none")
    ax2.set_ylabel("measured $-$ form", fontsize=15)
    ax2.set_ylim(1.35 * min(meas - spec), max(0.004, -0.15 * min(meas - spec)))
    ax2.set_xticks(xs)
    ax2.set_xticklabels([b[2] for b in ABANDS])
    ax2.set_xlabel(r"$|\eta|$ of the leading muon")
    fn = os.path.join(out, "am_closed.png")
    pubhtml.savefig(fig, fn, dpi=140)
    plt.close(fig)
    logger.info(f"-> {fn}")


def panel_mixture(out, pct=90.0, nboot=150):
    P = np.load(os.path.join(HERE, "runs/zpairs_dyv2_full.npz"))
    S = np.load(os.path.join(HERE, "runs/auxseed_dyv2.npz"))
    rng = np.random.default_rng(20260908)
    vgf = P["vgf"].astype(np.float64)
    w1 = P["w"].astype(np.float64)
    zl = np.concatenate([S["zleg_p"], S["zleg_m"]])
    q = np.concatenate([S["q_p"], S["q_m"]])
    sr = np.concatenate([S["sigrel_p"], S["sigrel_m"]])
    dq = np.concatenate([S["dq_p"], S["dq_m"]])
    eta = np.abs(np.concatenate([S["geta_p"], S["geta_m"]]))
    vg = np.concatenate([vgf, vgf])
    w = np.concatenate([w1, w1])
    den = 1.0 - sr * (1.0 - vg) * q * zl
    x = np.where(np.abs(den) > 1e-3, zl / den, np.nan)
    ok = np.isfinite(x) & (np.abs(x) < 30) & np.isfinite(dq) & (dq >= 0)
    thr = np.percentile(dq[ok], pct)
    u = PROBES[0]

    xs = np.arange(3, dtype=float)
    res = {"IN": ([], []), "OUT": ([], []), "ALL": ([], [])}
    frac = []
    for lo, hi, lab in BANDS[:3]:
        b = ok & (eta >= lo) & (eta < hi)
        frac.append(float(np.average((dq > thr)[b], weights=w[b])))
        for nm, sel in (("IN", dq > thr), ("OUT", dq <= thr),
                        ("ALL", np.ones_like(ok))):
            mk = b & sel
            v = 0.5 * (odd(x[mk & (q > 0)], u, w[mk & (q > 0)])
                       + odd(x[mk & (q < 0)], u, w[mk & (q < 0)]))
            e = boot_odd(x[mk], u, w[mk], nboot, rng)
            res[nm][0].append(1e3 * v)
            res[nm][1].append(1e3 * e)
    frac = np.array(frac)

    fig = plt.figure(figsize=(8.5, 8.4))
    gs = GridSpec(2, 1, height_ratios=[2.4, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax.axhline(0.0, color="0.6", lw=1)
    for nm, c, mk, lab in (("IN", "#c1272d", "o", f"IN  (top {100-pct:.0f} %)"),
                           ("OUT", "#0072b2", "s", "OUT (the bulk)"),
                           ("ALL", "k", "D", "all legs")):
        ax.errorbar(xs + (0.06 if nm == "IN" else -0.06 if nm == "OUT" else 0),
                    res[nm][0], yerr=res[nm][1], fmt=mk, ms=9, color=c,
                    label=lab, mfc="none" if nm == "ALL" else c)
        mu, c2, nd = flatness(*res[nm])
        ax.axhline(mu, color=c, ls=":", lw=1)
    ax.set_ylabel(r"charge-even $\langle x\,e^{-u x^2}\rangle$  [$10^{-3}$]")
    ax.set_xticks(xs)
    ax.tick_params(labelbottom=False)
    ax.set_xlim(-0.4, 2.4)
    ax.legend(loc="lower left", fontsize=15, frameon=False)
    axf = ax.twinx()
    axf.plot(xs, 100 * frac, "v--", color="#2ca02c", ms=10)
    axf.set_ylabel(r"$f_{IN}$ [%]", color="#2ca02c")
    axf.tick_params(axis="y", colors="#2ca02c")
    axf.set_ylim(0, 1.5 * 100 * frac.max())
    ax.text(0.03, 0.94, "Z$\\to\\mu\\mu$ MC legs, truth-referenced pull\n"
            f"split on |seed$\\to$final d$q/p$| at p{pct:.0f}",
            transform=ax.transAxes, fontsize=13, va="top")
    # the LOWER panel is the flatness test, not the mixture identity: the
    # identity `f IN + (1-f) OUT == all` holds to <0.06e-3 by construction and
    # says nothing. What the hypothesis requires is that EACH component be
    # eta-INDEPENDENT, and neither is.
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.axhline(0.0, color="k", lw=1)
    for nm, c, mk, dx in (("IN", "#c1272d", "o", 0.06),
                          ("OUT", "#0072b2", "s", -0.06)):
        mu, c2, nd = flatness(*res[nm])
        ax2.errorbar(xs + dx, np.array(res[nm][0]) - mu, yerr=res[nm][1],
                     fmt=mk, ms=8, color=c,
                     label=rf"{nm}: $\chi^2$ {c2:.0f}/{nd}")
    ax2.legend(fontsize=12, frameon=False, ncol=2, loc="lower left")
    ax2.set_ylabel("component $-$ its\nown mean", fontsize=13)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([b[2] for b in BANDS[:3]])
    ax2.set_xlabel(r"$|\eta|$ of the leg (generator)")
    fn = os.path.join(out, "mixture_legs.png")
    pubhtml.savefig(fig, fn, dpi=140)
    plt.close(fig)
    logger.info(f"-> {fn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=DEF_OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    idx = os.path.join(a.outdir, "index.php")
    if not os.path.exists(idx):
        src = os.path.expanduser("~/public_html/cvh/260814_cleanprop/index.php")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, idx)
    panel_am(a.outdir)
    panel_mixture(a.outdir)


if __name__ == "__main__":
    main()
