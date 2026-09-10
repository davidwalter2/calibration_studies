#!/usr/bin/env python3
"""P(|z| > t) of the data against each arm's model density, per component.

The two tails are integrated SEPARATELY -- one trapezoid over the union would
add a spurious slab across the core, which is the bug the first version of
this table had.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hitlik_term as HT  # noqa: E402
import plot_hitlik as P  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--npz", required=True)
p.add_argument("--max-tracks", type=int, default=20000)
p.add_argument("--comps", default="0123")
p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
p.add_argument("--upsample", type=int, default=8)
p.add_argument("--plot", action="store_true")
p.add_argument("--outpath", default=None)
a = p.parse_args()

sel = HT.load(a.npz, max_tracks=a.max_tracks, comps=a.comps)
zg = np.linspace(-30, 30, 2401)
CN = ("q/p", "lambda", "phi", "d0", "z0")
print(f"{'comp':<8}{'N':>7}{'Var(z)':>9}{'mean':>9}   "
      + "".join(f"{'norm ' + m:>13}" for m in a.arms))
dens = {}
for c in sorted(set(sel["comp"].tolist())):
    rows = sel["comp"] == c
    z = sel["z"][rows]
    dens[c] = {m: P.mean_density(sel, m, c, zg, upsample=a.upsample)
               for m in a.arms}
    print(f"{CN[c]:<8}{len(z):>7}{np.var(z):>9.4f}{np.mean(z):>9.4f}   "
          + "".join(f"{np.trapezoid(dens[c][m], zg):>13.6f}" for m in a.arms))
print()
for thr in (1.0, 2.0, 3.0, 4.0, 5.0):
    print(f"P(|z| > {thr:g})")
    print(f"{'  comp':<8}{'data':>11}" + "".join(f"{m:>11}{'d/m':>8}"
                                                 for m in a.arms))
    for c in sorted(dens):
        rows = sel["comp"] == c
        fd = float(np.mean(np.abs(sel["z"][rows]) > thr))
        row = f"  {CN[c]:<6}{fd:>11.5f}"
        for m in a.arms:
            lo, hi = zg <= -thr, zg >= thr
            fm = float(np.trapezoid(dens[c][m][lo], zg[lo])
                       + np.trapezoid(dens[c][m][hi], zg[hi]))
            row += f"{fm:>11.5f}{fd/max(fm,1e-12):>8.2f}"
        print(row)


if a.plot:
    import datetime
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.ROOT)
    import pubhtml
    from plot_hitlik import ARMCOL, ARMLAB
    out = a.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_hitlik/")
    os.makedirs(out, exist_ok=True)
    pubhtml.ensure_index(out)
    ths = np.linspace(0.5, 5.5, 26)
    for c in sorted(dens):
        rows = sel["comp"] == c
        zz = sel["z"][rows]
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        for m in a.arms:
            rat, err = [], []
            for t in ths:
                fd = float(np.mean(np.abs(zz) > t))
                nd = int(np.sum(np.abs(zz) > t))
                lo, hi = zg <= -t, zg >= t
                fm = float(np.trapezoid(dens[c][m][lo], zg[lo])
                           + np.trapezoid(dens[c][m][hi], zg[hi]))
                rat.append(fd / max(fm, 1e-300))
                err.append(rat[-1] / max(np.sqrt(max(nd, 1)), 1.0))
            ax.errorbar(ths, rat, yerr=err, fmt="o-", ms=3.5, lw=1.4,
                        color=ARMCOL[m], label=ARMLAB[m])
        ax.axhline(1.0, color="k", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlabel(r"threshold $t$")
        ax.set_ylabel(r"data / model  of  $P(|z| > t)$")
        ax.set_title(f"tail closure, component {c} ({CN[c]}), "
                     f"{int(rows.sum())} rows", fontsize=14)
        ax.legend(loc="upper left", fontsize=12, frameon=False)
        fn = os.path.join(out, f"tailclosure_c{c}.pdf")
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fn)
