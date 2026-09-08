#!/usr/bin/env python3
"""The class-share attribution on the ACTUAL Z legs (the DY groups cache).

The gun said the charge-even odd moment runs -4.9 / -3.6 / +0.1 across the
three |eta| bands and that tracks whose curvature is carried by single-strip
(N1) clusters are more negatively skewed in all three -- at ~1 sigma each. This
runs the same split on `gzpairs_dyv2_n50.npz`, 487 742 Z candidates with their
per-hit class labels and per-hit influence weights, which is the sample the
mass fit actually uses.

Two differences from the track-level version, both forced by the data:
  * it is a two-track MASS cache, so there is no per-candidate charge and the
    charge split stays a track-level measurement;
  * the pull-normalisation artefact at mass level is `a` from sec. 0f.33,
    MEASURED at `a/(sigma/m) = 1.211` inclusively rather than the spec's
    `1 + vgf`. `x = z/(1 - a z)` uses the measured coefficient.

    python3 attribute_skew_mass.py
"""
import argparse

import numpy as np

PROBES = (0.05, 0.2)


def odd(z, u, w=None):
    return np.average(z * np.exp(-u * z * z), weights=w)


def odd_err(z, u, nboot, rng, w=None):
    n = len(z)
    if n < 50:
        return np.nan
    b = np.empty(nboot)
    for i in range(nboot):
        k = rng.integers(0, n, n)
        b[i] = odd(z[k], u, None if w is None else w[k])
    return b.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/work/submit/david_w/ZMass/"
                    "calibration_studies/fullscale/runs/gzpairs_dyv2_n50.npz")
    ap.add_argument("--nboot", type=int, default=150)
    ap.add_argument("--zmax", type=float, default=8.0)
    ap.add_argument("--acoef", type=float, default=1.211,
                    help="a/(sigma/m), MEASURED in sec. 0f.33 (the spec's "
                         "1+vgf is 1.2625 and is high)")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(20260908)

    d = np.load(a.cache, allow_pickle=True)
    z = d["z"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    mgen = d["mgen"].astype(np.float64) if "mgen" in d.files else None
    m = z * sig + mgen
    etal = np.maximum(np.abs(d["etap"].astype(np.float64)),
                      np.abs(d["etam"].astype(np.float64)))
    w = d["w"].astype(np.float64) if "w" in d.files else np.ones(len(z))
    cls = d["hit_cls"].astype(np.int64)
    hv = d["hit_v"].astype(np.float64)
    ptr = d["hit_ptr"].astype(np.int64)
    names = [str(s) for s in d["hit_classes"]]

    a_i = a.acoef * sig / m
    den = 1.0 - a_i * z
    x = z if a.raw else np.where(np.abs(den) > 1e-3, z / den, np.nan)
    ok = np.isfinite(x) & (np.abs(x) < a.zmax) & np.isfinite(w) & (sig / m < 0.10)

    ntr = len(z)
    cnt = np.diff(ptr)
    trk = np.repeat(np.arange(ntr), cnt)
    tot = np.bincount(trk, weights=hv, minlength=ntr)
    ncl = len(names)
    share = np.zeros((ntr, ncl))
    for c in range(ncl):
        mm = cls == c
        share[:, c] = np.bincount(trk[mm], weights=hv[mm], minlength=ntr)
    share /= np.maximum(tot, 1e-300)[:, None]
    idx = {n: i for i, n in enumerate(names)}
    def grp(pref):
        cs = [i for n, i in idx.items() if n.startswith(pref)]
        return share[:, cs].sum(1) if cs else np.zeros(ntr)
    n1 = grp("str_N1"); pix = grp("pix"); strp = grp("str")

    print(f"{a.cache.split('/')[-1]}: {ntr} candidates, {len(cls)} hits, "
          f"{int(ok.sum())} used")
    print(f"classes: {names}")
    print(f"influence-weighted shares: pixel {np.median(pix):.3f}, strip "
          f"{np.median(strp):.3f}, single-strip {np.median(n1):.3f}")
    print(f"a = {a.acoef} sigma/m: median {np.median(a_i):.5f}; using "
          f"{'RAW z' if a.raw else 'x = z/(1-a z)'}\n")

    def table(title, groups):
        print(title)
        hdr = (f"  {'cut':34s} {'n':>8s}" +
               "".join(f"{'u='+str(u):>18s}" for u in PROBES))
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for lab, mk in groups:
            mk = mk & ok
            if mk.sum() < 500:
                print(f"  {lab:34s} {int(mk.sum()):8d}   (too few)"); continue
            row = f"  {lab:34s} {int(mk.sum()):8d}"
            for u in PROBES:
                row += (f"  {1e3*odd(x[mk], u, w[mk]):+8.2f} "
                        f"+-{1e3*odd_err(x[mk], u, a.nboot, rng, w[mk]):5.2f}")
            print(row)
        print()

    BANDS = [(0.0, 0.9, "|eta| 0.0-0.9"), (0.9, 1.6, "0.9-1.6"),
             (1.6, 3.0, "1.6-3.0")]
    table("A. per |eta| band", [(l, (etal >= lo) & (etal < hi))
                               for lo, hi, l in BANDS]
          + [("inclusive", np.ones(ntr, bool))])
    for lo, hi, lab in BANDS:
        b = (etal >= lo) & (etal < hi)
        if (b & ok).sum() < 5000:
            continue
        q = np.percentile(n1[b & ok], [50, 90])
        qp = np.percentile(pix[b & ok], [33, 67])
        table(f"B. hit composition, {lab}",
              [(f"{lab}, single-strip < {q[0]:.3f}", b & (n1 < q[0])),
               (f"{lab}, single-strip > {q[1]:.3f}", b & (n1 >= q[1])),
               (f"{lab}, pixel share < {qp[0]:.3f}", b & (pix < qp[0])),
               (f"{lab}, pixel share > {qp[1]:.3f}", b & (pix >= qp[1]))])


if __name__ == "__main__":
    main()
