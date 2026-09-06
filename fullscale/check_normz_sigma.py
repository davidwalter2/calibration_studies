#!/usr/bin/env python3
"""How much does the truncation normalisation Z care about sigma?

`_norm_z` is a class-level Gil-Pelaez edge integral evaluated at the STORED
class resolution; it does not see the per-candidate self-consistent
`s_i(theta) = sigma_i - a_i delta_i(theta)`. That is an approximation, and this
measures it: rebuild the same term with every sigma scaled by `1 + eps` and
compare `Z` and, more to the point, the NLL's dependence on `m_Z`.

The relevant eps is `a_i |delta_i| / sigma_i`, whose typical value on the DY
production is `a_i ~ 0.0147` times `|delta| ~ sigma ~ 1.1 GeV` over
`sigma = 1.1 GeV`, i.e. **~1.5-2.5 %**.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

DT = tf.float64


def main():
    import h5py
    from rabbit import unbinned

    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--eps", type=float, nargs="*", default=[0.005, 0.025, 0.05])
    p.add_argument("--dmz", type=float, default=5.0,
                   help="m_Z displacement [MeV] the sensitivity is quoted over")
    a = p.parse_args()

    with h5py.File(a.card, "r") as f:
        g = f["unbinned_terms"]
        name = list(g.keys())[0]
        term = unbinned.read_unbinned_terms_from_h5(g)[0]
        sub = g[name]
        data = {k: np.asarray(sub[k]) for k in sub.keys()
                if k not in ("config", "params")}
    names = list(term.param_names)
    x0 = np.asarray(term.param_defaults, np.float64)
    i_mz = names.index("m_Z")

    # `_norm_z` sees sigma ONLY through `self._norm_sigma` (the class
    # exponents are functions of the STANDARDIZED t and the kernel CF is
    # evaluated at t_abs = t/sigma inside it), so scaling that one tensor is
    # exactly "the same term with every class resolution scaled".
    sig0 = term._norm_sigma.numpy().copy()

    def zvals(scale):
        term._norm_sigma = tf.constant(sig0 * scale, DT)
        out = {}
        for dmz in (0.0, a.dmz):
            xv = x0.copy()
            xv[i_mz] += dmz
            out[dmz] = term._norm_z(
                term._values(tf.constant(xv, DT))).numpy()
        return out

    ref = zvals(1.0)
    print(f"{'eps':>8s} {'max |dZ/Z|':>12s} {'max |d(dlnZ/dm_Z)|':>20s}  "
          f"(dm_Z = {a.dmz} MeV)")
    d0 = np.log(ref[a.dmz]) - np.log(ref[0.0])
    for e in a.eps:
        v = zvals(1.0 + e)
        dz = np.max(np.abs(v[0.0] / ref[0.0] - 1.0))
        d1 = np.log(v[a.dmz]) - np.log(v[0.0])
        print(f"{e:8.4f} {dz:12.3e} {np.max(np.abs(d1 - d0)):20.3e}")
    print(f"\n  reference: Z in [{ref[0.0].min():.6f}, {ref[0.0].max():.6f}]")
    print(f"  d lnZ / d m_Z over {a.dmz} MeV: max |{np.max(np.abs(d0)):.4e}|")
    print("\n  The second column is what matters: an error in Z that does not")
    print("  MOVE with m_Z is absorbed by the normalisation and biases nothing.")


if __name__ == "__main__":
    sys.exit(main())
