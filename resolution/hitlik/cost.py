#!/usr/bin/env python3
"""Cost of the residual-vector CF likelihood: time, memory and export bytes.

Times the three things a fit actually does -- NLL, NLL+gradient, and one
Hessian-vector product -- on a real card's worth of rows, and converts to a
per-track and per-component number.  Then prints the EXPORT bill: what a maker
would have to write per track for (a) the 5-component reference-state version
this prototype uses and (b) a per-HIT version, at the measured group and hit
multiplicities.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hitlik_term as HT  # noqa: E402

NTAU = 64
NFAM = 6
F32 = 4


def timeit(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True)
    p.add_argument("--max-tracks", type=int, default=2000)
    p.add_argument("--max-inflat", type=float, default=1e4)
    p.add_argument("--comps", default="0123")
    p.add_argument("--arm", default="cf")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--prune-frac", type=float, default=0.001)
    p.add_argument("--nz-legs", type=float, default=7.0e6,
                   help="Z legs at full scale")
    p.add_argument("--njpsi-legs", type=float, default=34.0e6)
    a = p.parse_args()
    import tensorflow as tf

    comps = [int(c) for c in a.comps]
    t0 = time.perf_counter()
    sel = HT.load(a.npz, max_tracks=a.max_tracks, comps=comps, max_inflat=a.max_inflat)
    t_load = time.perf_counter() - t0
    ntrk, ncomp = sel["ntrk"], len(comps)
    nrow = len(sel["z"])
    t0 = time.perf_counter()
    term, data, meta = HT.build(sel, arm=a.arm, prune_frac=a.prune_frac,
                                groups_file=sel["groups_file"],
                                chunk=a.chunk)
    t_build = time.perf_counter() - t0
    npar = len(term.param_names)
    nnz = meta["nrows"]
    x = tf.Variable(np.zeros(npar), dtype=tf.float64)
    v = tf.constant(np.ones(npar) / np.sqrt(npar))

    def f_nll():
        return float(term.nll(x).numpy())

    def f_grad():
        with tf.GradientTape() as t1:
            y = term.nll(x)
        return t1.gradient(y, x).numpy()

    def f_hvp():
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                y = term.nll(x)
            g = t1.gradient(y, x)
        return t2.gradient(g, x, output_gradients=v).numpy()

    t_nll = timeit(f_nll)
    t_grad = timeit(f_grad)
    t_hvp = timeit(f_hvp)

    prov = json.loads(sel["provenance"]) if sel["provenance"] else {}
    print("=" * 74)
    print(f"COST  ({ntrk} tracks x {ncomp} components = {nrow} rows, "
          f"{nnz} group rows, {npar} parameters, arm '{a.arm}')")
    print("=" * 74)
    print(f"  npz load                 {t_load:8.2f} s")
    print(f"  term build (numpy + tf)  {t_build:8.2f} s")
    print(f"  NLL                      {t_nll*1e3:8.1f} ms   "
          f"{t_nll/nrow*1e6:7.2f} us/row   {t_nll/ntrk*1e6:7.2f} us/track")
    print(f"  NLL + gradient           {t_grad*1e3:8.1f} ms   "
          f"{t_grad/nrow*1e6:7.2f} us/row   {t_grad/ntrk*1e6:7.2f} us/track")
    print(f"  one Hessian-vector prod  {t_hvp*1e3:8.1f} ms   "
          f"{t_hvp/nrow*1e6:7.2f} us/row")
    print(f"  full Hessian ({npar} HVPs)  {t_hvp*npar:8.1f} s")
    print(f"  CF evaluations per row: nt {NTAU} x upsample {term.upsample} "
          f"= {NTAU*term.upsample} quadrature points, "
          f"{nnz/max(nrow,1):.1f} group rows/row")
    print()
    print(f"  extrapolated to 320 000 tracks (this production):")
    sc = 320000.0 / ntrk
    print(f"     one NLL+grad  {t_grad*sc:8.1f} s      "
          f"one HVP {t_hvp*sc:8.1f} s")
    print(f"  extrapolated to {a.nz_legs/1e6:.0f} M Z legs + "
          f"{a.njpsi_legs/1e6:.0f} M J/psi legs = "
          f"{(a.nz_legs+a.njpsi_legs)/1e6:.0f} M tracks:")
    sc = (a.nz_legs + a.njpsi_legs) / ntrk
    print(f"     one NLL+grad  {t_grad*sc/3600:8.2f} h CPU   "
          f"one HVP {t_hvp*sc/3600:8.2f} h CPU")
    print(f"     (a GPU is 20-60x this hardware on the same graph; the term is "
          f"a dense (chunk, nt) matmul per family)")

    # ---- memory -----------------------------------------------------------
    mem = 0
    for k, arr in data.items():
        if isinstance(arr, np.ndarray):
            mem += arr.nbytes
    print()
    print(f"  card datasets in memory  {mem/1e9:8.3f} GB  "
          f"({mem/ntrk/1e3:.1f} kB/track)")

    # ---- export bill ------------------------------------------------------
    gm = nnz / max(nrow, 1)
    hm = len(sel["hit_cls"]) / max(nrow, 1)
    print()
    print("=" * 74)
    print("EXPORT BILL per track (float32), at the measured multiplicities")
    print(f"  {gm:.1f} material groups and {hm:.1f} hit classes per "
          f"(track, component)")
    print("=" * 74)
    rows = []
    for nres, lab in ((1, "q/p only (the status quo)"),
                      (ncomp, f"{ncomp} reference-state components (this study)"),
                      (5, "5 reference-state components"),
                      (18, "18 per-HIT residuals")):
        flat = nres * NFAM * NTAU * F32
        grp = nres * gm * NFAM * NTAU * F32
        grp16 = nres * gm * NFAM * 16 * F32
        hits = nres * hm * (2 + 4)
        binf = 68 * nres * 5 * F32          # B_b, n_res x 5 per block
        rows.append((lab, flat, grp, grp16, hits, binf))
    print(f"{'variant':<38}{'flat':>9}{'per-grp':>10}{'rank16':>9}"
          f"{'hitcls':>8}{'B_b':>9}{'total(r16)':>12}")
    for lab, flat, grp, grp16, hits, binf in rows:
        tot = grp16 + hits + binf
        print(f"{lab:<38}{flat/1e3:>8.1f}k{grp/1e3:>9.1f}k{grp16/1e3:>8.1f}k"
              f"{hits/1e3:>7.2f}k{binf/1e3:>8.1f}k{tot/1e3:>11.1f}k")
    print()
    n_all = (a.nz_legs + a.njpsi_legs)
    for lab, flat, grp, grp16, hits, binf in rows:
        tot = grp16 + hits + binf
        print(f"  {lab:<38} {tot*n_all/1e12:6.2f} TB at "
              f"{n_all/1e6:.0f} M tracks")
    print()
    print("  (`B_b` assumes the 68 resolution blocks/track measured on this "
          "production;\n   `rank16` is the tau-axis PCA of NOTES 2026-09-05 "
          "(II) s8, exact to 1e-5)")


if __name__ == "__main__":
    main()
