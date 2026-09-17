#!/usr/bin/env python3
"""THE NOMINAL-POINT GATE of `--beam3`, on real cards.

The 3x3 luminous-region block must be a REPARAMETERISATION of what was there
before, not a different model: at the nominal parameters it has to reproduce
the term the two linear width classes built, and its derivatives with respect
to the two width scales have to be the ones the maker exports.  The anchored
form makes the first exact by construction (`v(0) = v_nominal` identically);
what is left to measure is whether the surrounding assembly -- `vg_other`
reduced by the block's WHOLE share instead of by its two transverse
directions, one fewer pair of rows in the per-candidate class list -- leaves
the likelihood where it was.

Reports, per term:
  * NLL(0), baseline vs beam3, absolute and relative
  * the gradient at 0 on every SHARED parameter (the 42 material amounts and
    the 18 hit classes), max |difference|
  * d NLL/d beamwidth_{x,y} at 0 in the two constructions -- the beam3 form
    rebuilds `covBS` from the scaled width (the physical change), the class
    form scales the whole row and column (`covBS -> D covBS D`), and the two
    differ only through `dC_xz/dk_x`; the ratio is the number to quote

usage:
  ./run_tf.sh python3 gate_beam3_card.py --base <card> --beam3 <card>
"""
import argparse
import sys

import h5py
import numpy as np
import tensorflow as tf

from rabbit import unbinned


def load(path):
    with h5py.File(path, "r") as f:
        return {t.name: t for t in unbinned.read_unbinned_terms_from_h5(
            f["unbinned_terms"])}


def at_zero(t):
    x = tf.Variable(np.zeros(len(t.param_names)), dtype=tf.float64)
    with tf.GradientTape() as tp:
        v = t.nll(x)
    g = np.asarray(tp.gradient(v, x).numpy())
    return float(v.numpy()), dict(zip(t.param_names, g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--beam3", required=True)
    a = ap.parse_args()
    A, B = load(a.base), load(a.beam3)
    ok = True
    for nm in sorted(set(A) & set(B)):
        na, ga = at_zero(A[nm])
        nb, gb = at_zero(B[nm])
        rel = abs(na - nb) / max(abs(na), 1e-300)
        # the two WIDTH scales are shared by name but not by convention: the
        # class form scales the beam row and column (`covBS -> D covBS D`),
        # beam3 rebuilds `covBS` from the scaled width, which is what a
        # physical change of sigma_x does.  They are compared separately,
        # below, and are the ONLY thing that may move.
        shared = sorted((set(ga) & set(gb)) - set(unbinned.BEAM3_ROLES)
                        - {"beamwidth_x", "beamwidth_y"})
        dg = np.array([ga[p] - gb[p] for p in shared])
        scale = max(np.abs([ga[p] for p in shared]).max(), 1e-300)
        print(f"\n--- {nm}")
        print(f"  NLL(0)  base {na!r}")
        print(f"          beam3 {nb!r}")
        print(f"          |diff| {abs(na-nb):.3e}   rel {rel:.3e}   "
              f"identical: {na == nb}")
        print(f"  gradient on {len(shared)} shared parameters: "
              f"max |diff| {np.abs(dg).max():.3e}  "
              f"(max |grad| {scale:.4g}, rel {np.abs(dg).max()/scale:.3e})")
        for p in ("beamwidth_x", "beamwidth_y"):
            if p in ga and p in gb:
                r = gb[p] / ga[p] if ga[p] != 0 else np.nan
                print(f"  d NLL/d {p}: class form {ga[p]:+.9e}  "
                      f"beam3 {gb[p]:+.9e}  ratio {r:.9f}")
        for p in sorted(set(gb) - set(ga)):
            print(f"  beam3-only: d NLL/d {p} = {gb[p]:+.6e}")
        ok &= rel < 1e-12 and np.abs(dg).max() / scale < 1e-9
    print("\nGATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
