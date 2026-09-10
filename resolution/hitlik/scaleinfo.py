#!/usr/bin/env python3
"""WHY the CF carries LESS information than the Gaussian: the scale-information
of a heavy-tailed density.

For a pure scale family p(z; s) = p_1(z/s)/s the Fisher information about
ln s is

    I(ln s) = Int dz p(z) [ 1 + z d ln p / dz ]^2  -  1     ... = 2 for a
                                                                Gaussian,

and for a Student-t with nu degrees of freedom it is 2 nu / (nu + 3) < 2 --
i.e. a density with heavier tails carries LESS information about its own
width, because the tail is where a width change shows up least in the
log-likelihood.

This evaluates it on the ACTUAL row-averaged residual densities of each arm.
The Gaussian arms are the gate: they must come out at 2.000.
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
p.add_argument("--max-tracks", type=int, default=4000)
p.add_argument("--comps", default="0123")
p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
p.add_argument("--zmax", type=float, default=25.0)
p.add_argument("--nz", type=int, default=20001)
a = p.parse_args()

sel = HT.load(a.npz, max_tracks=a.max_tracks, comps=[int(c) for c in a.comps])
zg = np.linspace(-a.zmax, a.zmax, a.nz)
CN = ("q/p", "lambda", "phi", "d0", "z0")


def scale_info(z, f):
    """I(ln s) of a density tabulated on a uniform grid."""
    f = np.maximum(f, 0.0)
    n = np.trapezoid(f, z)
    f = f / n
    dl = np.gradient(f, z) / np.maximum(f, 1e-300)
    g = 1.0 + z * dl
    ok = f > 1e-14                       # the score is noise where p underflows
    # I(ln s) = E[(1 + z dlnp/dz)^2].  There is NO "-1": for N(0,1) the
    # integrand is (1 - z^2)^2 and E = 1 - 2 + 3 = 2.
    return float(np.trapezoid((g[ok] ** 2) * f[ok], z[ok]))


# gate: an exact unit Gaussian on the same grid must give 2.000
gauss = np.exp(-0.5 * zg ** 2) / np.sqrt(2 * np.pi)
print(f"GATE  exact N(0,1) on this grid: I(ln s) = {scale_info(zg, gauss):.4f} "
      f"(must be 2.000)")
t5 = (1 + zg ** 2 / 5.0) ** (-3.0)   # Student-t, nu = 5
print(f"GATE  Student-t5 (analytic 2*5/8 = 1.250): "
      f"{scale_info(zg, t5 / np.trapezoid(t5, zg)):.4f}")
print()
print(f"{'component':<10}" + "".join(f"{'I(ln s) ' + m:>18}" for m in a.arms)
      + f"{'CF/gaussq':>12}")
for c in sorted(set(sel["comp"].tolist())):
    vals = {}
    for m in a.arms:
        f = P.mean_density(sel, m, c, zg, upsample=8)
        vals[m] = scale_info(zg, f)
    row = f"{CN[c]:<10}" + "".join(f"{vals[m]:>18.4f}" for m in a.arms)
    if "gaussq" in vals:
        row += f"{vals['cf']/vals['gaussq']:>12.3f}"
    print(row)
