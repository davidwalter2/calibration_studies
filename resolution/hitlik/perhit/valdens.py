#!/usr/bin/env python3
"""GATE 3: are the model densities densities, and is `gaussq` the fit's own?

The prototype did this in a throwaway `/tmp/valdens.py`; it belongs in the
repo.  For each arm and each component group it evaluates the row-averaged
predicted density by the same inverse Fourier transform the term uses,

    p(z) = (1/pi) Int_0^tmax dtau [ Re e^S cos(tau z) + Im e^S sin(tau z) ]

on a wide z grid, and reports its NORM, MEAN and VARIANCE, plus the MODEL
variance computed analytically from the exponents (`sum_g kappa2 + vgf`).  The
two must agree, the norm must be 1 and the mean 0; and for `gaussq` the
variance must be 1.0000 EXACTLY, because that arm IS the fit's own Q-matrix
decomposition of the component's unit variance -- which is the statement that
makes it the Gaussian chi2 the whole study is measured against.

The DATA variance is printed alongside: it is NOT 1, and the gap is the
physics the parameters absorb.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HL = os.path.dirname(_HERE)
_RES = os.path.dirname(_HL)
for _p in (_HERE, _HL, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hitlik_term as HT  # noqa: E402
import plot_hitlik as PH  # noqa: E402


def model_var(sel, arm):
    """`sum_g kappa2_g + vgf` per row, analytically from the exponents."""
    fam, ptr, gid, ndrop, ng, amp, drop = HT.arm_families(sel, arm)
    tg = sel["tgrid"]
    n = len(sel["z"])
    seg = np.repeat(np.arange(n), np.diff(ptr))
    tot = np.zeros(n)
    for m in fam:
        if "re" not in m:
            continue
        k2 = HT.kappa2_from_grid(m["re"].astype(np.float64), tg)
        np.add.at(tot, seg, k2)
        if "fix_re" in m:
            tot += HT.kappa2_from_grid(m["fix_re"].astype(np.float64), tg)
    return tot + sel["vgf"]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--comps", default="all")
    p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
    p.add_argument("--group-by", choices=["comp", "cls", "relpos", "ckind"],
                   default="ckind")
    p.add_argument("--max-tracks", type=int, default=400)
    p.add_argument("--max-inflat", type=float, default=1e4)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--zmax", type=float, default=40.0)
    p.add_argument("--nz", type=int, default=3201)
    p.add_argument("--upsample", type=int, default=8)
    a = p.parse_args()

    sel = HT.load(a.npz, max_tracks=a.max_tracks, comps=a.comps,
                  max_inflat=a.max_inflat)
    zg = np.linspace(-a.zmax, a.zmax, a.nz)
    print(f"# {sel['ntrk']} tracks, {len(sel['z'])} rows, comps '{a.comps}', "
          f"grouped by {a.group_by}")
    print(f"{'group':<22}{'N':>8}{'arm':>8}{'norm':>11}{'mean':>11}"
          f"{'var(dens)':>12}{'var(model)':>12}{'var(DATA)':>12}")
    for lab, rows in PH._groups(a, sel):
        if len(rows) < a.min_rows:
            continue
        z = np.asarray(sel["z"])[rows]
        for arm in a.arms:
            d = PH.mean_density(sel, arm, rows, zg, upsample=a.upsample)
            nrm = np.trapezoid(d, zg)
            mean = np.trapezoid(d * zg, zg)
            var = np.trapezoid(d * zg ** 2, zg) - mean ** 2
            mv = model_var(sel, arm)[rows].mean()
            print(f"{str(lab):<22}{len(rows):>8}{arm:>8}{nrm:>11.6f}"
                  f"{mean:>+11.6f}{var:>12.5f}{mv:>12.5f}{np.var(z):>12.5f}")


if __name__ == "__main__":
    main()
