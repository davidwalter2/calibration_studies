#!/usr/bin/env python3
"""Size the composite-likelihood approximation, and answer the WG question.

The whitened components are UNCORRELATED by construction -- exactly, and per
track: `sum_b A_b[j] . A_b[k] = 0`, which for a per-hit component against the
truth-referenced q/p one is `F^T R = 0`.  They are not INDEPENDENT: they are
different linear functionals of the same non-Gaussian block noises, so the
product-of-marginals likelihood drops their joint cumulants.  The leading one
is the fourth, and for the multiple-scattering blocks (which carry the
non-Gaussianity) its correlation

    kappa(z_j,z_j,z_k,z_k) / sqrt(kappa_4(j) kappa_4(k))

is what `extract_perhit.py` stores per track.  Three tables:

  (1) the truth-referenced q/p component (the functional the MASS term uses)
      against every per-hit innovation, BY HIT POSITION along the track --
      the answer to "are there correlations between the mass term and the hit
      residuals";
  (2) adjacent per-hit innovations (component k against k-1), which the task
      expects to be larger than for the reference parameters;
  (3) the 6 pairs among the first 4 truth-referenced components, directly
      comparable with the prototype's table (q/p-lam 0.138 ... phi-d0 0.712).

and the GAUSSIAN-LEVEL check, both per track (algebraic, `dot_ref0`) and as an
ensemble correlation of the `z` themselves.
"""

import argparse
import numpy as np


def qtab(x, w=None):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan,) * 4
    return (np.median(x), np.percentile(x, 90), x.mean(), len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--nbin", type=int, default=8,
                   help="hit-position bins along the track")
    p.add_argument("--max-inflat", type=float, default=1e4)
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    ck = np.asarray(d["ckind"])
    trk = np.asarray(d["trk"])
    comp = np.asarray(d["comp"])
    rel = np.asarray(d["relpos"])
    infl = np.asarray(d["inflat"])
    z = np.asarray(d["z"])
    dot = np.asarray(d["dot_ref0"])
    xc0 = np.asarray(d["xc_ref0"])
    xcp = np.asarray(d["xc_prev"])
    hit = (ck == 0) & (infl < a.max_inflat)
    ref = ck == 1

    ntrk = len(d["eta"])
    print(f"# {ntrk} tracks, {hit.sum()} per-hit rows "
          f"(inflat < {a.max_inflat:g}), {ref.sum()} reference rows\n")

    # ---------------- the exact, per-track uncorrelatedness ----------------
    print("=== Gaussian level: the components are EXACTLY uncorrelated")
    m = hit & np.isfinite(dot)
    print(f"  per track, algebraic  |sum_b A_b[k].A_b[qp]| : "
          f"median {np.median(np.abs(dot[m])):.3e}  "
          f"p99 {np.percentile(np.abs(dot[m]), 99):.3e}  "
          f"max {np.abs(dot[m]).max():.3e}     (this is F^T R = 0)")
    # ensemble correlation of the z themselves, per component index
    zref = np.full(ntrk, np.nan)
    ir = np.where(ref & (comp == 0 + (comp[ref].min() if False else 0)))[0]
    # the q/p reference component is the FIRST reference slot of its track
    firstref = {}
    for i in np.where(ref)[0]:
        t = trk[i]
        if t not in firstref or comp[i] < comp[firstref[t]]:
            firstref[t] = i
    for t, i in firstref.items():
        zref[t] = z[i]
    print("\n=== ensemble corr(z_k, z_qp) by hit position along the track")
    print("  relpos bin      N     corr        +-")
    edges = np.linspace(0, 1, a.nbin + 1)
    rows_pos = []
    for b in range(a.nbin):
        m = hit & (rel >= edges[b]) & (rel < edges[b + 1] + (1e-9 if b == a.nbin - 1 else 0))
        m &= np.isfinite(zref[trk])
        if m.sum() < 100:
            continue
        c = np.corrcoef(z[m], zref[trk][m])[0, 1]
        e = 1.0 / np.sqrt(m.sum())
        rows_pos.append((0.5 * (edges[b] + edges[b + 1]), int(m.sum()), c, e))
        print(f"  {edges[b]:.2f}-{edges[b+1]:.2f}  {m.sum():7d}  {c:+.4f}  {e:.4f}")

    # ---------------- (1) fourth cross cumulant with q/p -------------------
    print("\n=== (1) 4th cross-cumulant correlation with the truth-referenced"
          " q/p, by hit position")
    print("  relpos bin      N    median     p90     mean")
    for b in range(a.nbin):
        m = hit & (rel >= edges[b]) & (rel < edges[b + 1] + (1e-9 if b == a.nbin - 1 else 0))
        m &= np.isfinite(xc0)
        if m.sum() < 100:
            continue
        md, p9, mn, n = qtab(xc0[m])
        print(f"  {edges[b]:.2f}-{edges[b+1]:.2f}  {n:7d}   {md:.4f}  {p9:.4f}  {mn:.4f}")
    md, p9, mn, n = qtab(xc0[hit])
    print(f"  ALL                 {n:7d}   {md:.4f}  {p9:.4f}  {mn:.4f}")

    # also by hit class, since that is what the hit parameters are
    cls = np.asarray(d["cls"])
    names = [str(s) for s in d["hit_classes"]]
    print("\n  by hit class:  class                 N    median     p90")
    for c in range(len(names)):
        m = hit & (cls == c) & np.isfinite(xc0)
        if m.sum() < 100:
            continue
        md, p9, mn, n = qtab(xc0[m])
        print(f"    {names[c]:<22s} {n:7d}   {md:.4f}  {p9:.4f}")

    # ---------------- (2) adjacent innovations -----------------------------
    print("\n=== (2) 4th cross-cumulant correlation between ADJACENT per-hit"
          " innovations (k, k-1)")
    md, p9, mn, n = qtab(xcp[hit])
    print(f"  all adjacent pairs  {n:7d}   median {md:.4f}  p90 {p9:.4f}  "
          f"mean {mn:.4f}")
    print("  relpos bin      N    median     p90")
    for b in range(a.nbin):
        m = hit & (rel >= edges[b]) & (rel < edges[b + 1] + (1e-9 if b == a.nbin - 1 else 0))
        m &= np.isfinite(xcp)
        if m.sum() < 100:
            continue
        md, p9, mn, n = qtab(xcp[m])
        print(f"  {edges[b]:.2f}-{edges[b+1]:.2f}  {n:7d}   {md:.4f}  {p9:.4f}")

    # ---------------- (3) the reference-parameter pairs --------------------
    print("\n=== (3) the 6 pairs among the first 4 truth-referenced components"
          " (the prototype's table)")
    lbl = ["q/p-lam", "q/p-phi", "q/p-d0", "lam-phi", "lam-d0", "phi-d0"]
    xr = np.asarray(d["xc_refpairs"])
    print("  " + "  ".join(f"{s:>9s}" for s in lbl))
    print("  " + "  ".join(f"{np.nanmedian(xr[:, i]):9.3f}" for i in range(6)))

    if a.output:
        np.savez(a.output, pos=np.array(rows_pos), xc0=xc0[hit],
                 xcp=xcp[hit], xr=xr)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
