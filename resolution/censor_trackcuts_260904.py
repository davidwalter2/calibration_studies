#!/usr/bin/env python3
"""TASK 2 (track side): does the data's odd asymmetry depend on the track
quality cut, i.e. is the 1-3 sigma asymmetry a censoring artefact?

DATA ONLY (no CF): the odd moment <z e^{-u z^2}> and the trimmed means <z>
for |z| < T, per charge, under a scan of

    normchi2 < c        c = 2, 3, 5, 10, inf
    nvalidhits >= h     h = 8, 10, 12, inf(=0)

If the observed asymmetry were produced by a selection that removes the
large-loss tail, TIGHTENING the cut must make it LARGER (more censoring) and
loosening it must make it smaller; a cut-independent asymmetry is not a
censoring of that variable.  Bootstrap errors on the difference to the
uncut sample are computed on the SAME rows (paired), so the comparison is
not limited by the sample size.
"""
import argparse, os, sys
import numpy as np

PROBES = (0.05, 0.2, 1.0, 2.0)
TRIMS = (1., 2., 3., 5.)


def stats(z, probes=PROBES, trims=TRIMS):
    od = [float(np.mean(z * np.exp(-u * z ** 2))) for u in probes]
    tm = [float(z[np.abs(z) < T].mean()) for T in trims]
    return od, tm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caches", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    labels = a.labels or [os.path.basename(c) for c in a.caches]
    L = []

    def log(s=""):
        print(s, flush=True)
        L.append(s)

    rng = np.random.default_rng(a.seed)
    for cache, lab in zip(a.caches, labels):
        d = np.load(cache)
        z = d["z"].astype(np.float64)
        q = d["charge"].astype(np.int64)
        c2 = d["normchi2"].astype(np.float64)
        nv = d["nvalidhits"].astype(np.float64)
        gp = d["genpt"].astype(np.float64) if "genpt" in d.files else None
        log(f"\n{'='*100}")
        log(f"# {lab}   n={len(z)}   cache {cache}")
        log(f"#   normchi2  : med {np.median(c2):.3f}  q99 {np.quantile(c2,.99):.3f} "
            f" max {c2.max():.4g}   frac>2 {np.mean(c2>2):.4f}  >3 {np.mean(c2>3):.4f} "
            f" >5 {np.mean(c2>5):.4f}  >10 {np.mean(c2>10):.5f}")
        log(f"#   nvalidhits: min {nv.min():.0f}  q1 {np.quantile(nv,.01):.0f} "
            f" med {np.median(nv):.0f}  max {nv.max():.0f}   frac<10 {np.mean(nv<10):.4f} "
            f" <12 {np.mean(nv<12):.4f}")
        if gp is not None:
            log(f"#   genpt     : min {gp.min():.3f}  med {np.median(gp):.3f} "
                f" max {gp.max():.3f}")
        for qq, qn in ((1, "q+"), (-1, "q-"), (0, "all")):
            sq = np.ones(len(z), bool) if qq == 0 else (q == qq)
            zq, c2q, nvq = z[sq], c2[sq], nv[sq]
            od0, tm0 = stats(zq)
            log(f"\n  -- {lab}  {qn}  (n={sq.sum()}) --")
            hdr = (f"  {'cut':<20}{'n':>9}{'frac':>8}"
                   + "".join(f"{'<ze-'+format(u,'g')+'z2>':>13}" for u in PROBES)
                   + "".join(f"{'<z>|z|<'+format(T,'g'):>13}" for T in TRIMS))
            log(hdr)
            rows = [("(no cut)", np.ones(len(zq), bool))]
            for cc in (10., 5., 3., 2.):
                rows.append((f"normchi2 < {cc:g}", c2q < cc))
            for hh in (8., 10., 12.):
                rows.append((f"nvalidhits >= {hh:g}", nvq >= hh))
            rows.append(("chi2<3 & nv>=10", (c2q < 3) & (nvq >= 10)))
            for nm, sel in rows:
                if sel.sum() < 100:
                    continue
                od, tm = stats(zq[sel])
                log(f"  {nm:<20}{int(sel.sum()):>9}{sel.mean():>8.5f}"
                    + "".join(f"{v:>+13.5f}" for v in od)
                    + "".join(f"{v:>+13.5f}" for v in tm))
            # bootstrap on the DIFFERENCE (paired) for the tightest cut
            sel = (c2q < 3)
            if sel.sum() > 100:
                dif = np.empty((a.nboot, len(PROBES)))
                n = len(zq)
                for b in range(a.nboot):
                    i = rng.integers(0, n, n)
                    zb, sb = zq[i], sel[i]
                    o1, _ = stats(zb)
                    o2, _ = stats(zb[sb])
                    dif[b] = np.array(o2) - np.array(o1)
                od3, _ = stats(zq[sel])
                log(f"  {'d(chi2<3 - nocut)':<20}{'':>9}{'':>8}"
                    + "".join(f"{od3[i]-od0[i]:>+13.5f}"
                              for i in range(len(PROBES))))
                log(f"  {'  bootstrap sigma':<20}{'':>9}{'':>8}"
                    + "".join(f"{dif[:,i].std():>13.5f}"
                              for i in range(len(PROBES))))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
