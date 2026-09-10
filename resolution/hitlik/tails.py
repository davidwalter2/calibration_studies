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
p.add_argument("--group-by", default="comp",
               choices=["comp", "pooled", "relpos", "cls", "refparm"],
               help="how to group rows.  `comp` (the component slot) is the\n"
                    "natural axis for the 5 truth-referenced components; for\n"
                    "the per-hit complement there are up to 25 slots and the\n"
                    "physics axes are `relpos` (position along the track) and\n"
                    "`cls` (hit class), with `pooled` for one table.")
a = p.parse_args()

sel = HT.load(a.npz, max_tracks=a.max_tracks, comps=a.comps)
zg = np.linspace(-30, 30, 2401)
CN = ("q/p", "lambda", "phi", "d0", "z0")


def _groups(sel, how):
    """(label, row-index array) pairs.  `mean_density` takes ROW INDICES --
    it used to take a component index, and passing the index straight through
    is what broke this script on the per-hit file (2026-09-10)."""
    cs = sorted(set(sel["comp"].tolist()))
    if how == "pooled":
        return [("all", np.arange(len(sel["z"])))]
    if how == "comp":
        named = len(cs) <= 5 and max(cs) < 5
        return [(CN[c] if named else f"k={c}", np.where(sel["comp"] == c)[0])
                for c in cs]
    if how == "refparm":
        # For the per-hit file the truth-referenced components sit at slot
        # `d + j` of their track, so their `comp` value is NOT j -- it varies
        # with the track's own `d` and grouping by it mixes q/p with z0.  Rank
        # within the track instead: with `--comps ref` there are exactly five
        # rows per track and they are stored in order.
        trk, cp = sel["trk"], sel["comp"]
        o = np.lexsort((cp, trk))
        j = np.zeros(len(cp), np.int64)
        run = 0
        for a_, i in enumerate(o):
            if a_ and trk[i] != trk[o[a_ - 1]]:
                run = 0
            j[i] = run
            run += 1
        return [(CN[k] if k < 5 else f"j={k}", np.where(j == k)[0])
                for k in sorted(set(j.tolist()))]
    if how == "relpos":
        r = sel["relpos"]
        ed = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
        return [(f"rel {ed[i]:.1f}-{min(ed[i+1],1.0):.1f}",
                 np.where((r >= ed[i]) & (r < ed[i + 1]))[0])
                for i in range(len(ed) - 1)]
    names = [str(x) for x in sel["hit_classes"]]
    return [(names[c], np.where(sel["cls"] == c)[0])
            for c in sorted(set(sel["cls"].tolist())) if c >= 0]


grp = [(lab, ix) for lab, ix in _groups(sel, a.group_by) if len(ix) >= 200]
print(f"{'group':<14}{'N':>8}{'Var(z)':>9}{'mean':>9}   "
      + "".join(f"{'norm ' + m:>13}" for m in a.arms))
dens = {}
for c, (lab, rows) in enumerate(grp):
    z = sel["z"][rows]
    dens[c] = {m: P.mean_density(sel, m, rows, zg, upsample=a.upsample)
               for m in a.arms}
    print(f"{lab:<14}{len(z):>8}{np.var(z):>9.4f}{np.mean(z):>9.4f}   "
          + "".join(f"{np.trapezoid(dens[c][m], zg):>13.6f}" for m in a.arms))
print()
for thr in (1.0, 2.0, 3.0, 4.0, 5.0):
    print(f"P(|z| > {thr:g})")
    print(f"{'  group':<14}{'data':>11}" + "".join(f"{m:>11}{'d/m':>8}"
                                                 for m in a.arms))
    for c in sorted(dens):
        lab, rows = grp[c]
        fd = float(np.mean(np.abs(sel["z"][rows]) > thr))
        row = f"  {lab:<12}{fd:>11.5f}"
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
        lab, rows = grp[c]
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
        ax.set_title(f"tail closure, {lab}, {len(rows)} rows", fontsize=14)
        ax.legend(loc="upper left", fontsize=12, frameon=False)
        tag = lab.replace(" ", "").replace("/", "").replace("-", "_")
        fn = os.path.join(out, f"tailclosure_{tag}.pdf")
        fig.savefig(fn, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fn)
