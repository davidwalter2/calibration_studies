#!/usr/bin/env python3
"""How many tracks before the per-hit term stops needing its prior?

The decision this answers: does the DATA fit run the CF term on everything, or
on a subsample with the accumulated quadratic term on the disjoint remainder?

The PRIOR-FREE (standalone) sandwich error of a parameter scales exactly as
`1/sqrt(N)`: `H` and `J` are both sums over rows, so `H_N = (N/n0) H`,
`J_N = (N/n0) J`, and `sigma_free(N) = sigma_free(n0) sqrt(n0/N)`.  The
saturation point is where that crosses the parameter's own prior,

    N_sat = n0 * (sigma_free(n0) / sigma_prior)^2 ,

i.e. the number of tracks at which the term alone knows the parameter as well
as the tier prior does.  Beyond it the prior stops mattering; below it the fit
is prior-dominated and extra tracks buy little (the `S/Q` numbers of the
2500-track subsample fits are the prototype's example of that regime).

MARGINAL errors are quoted too, at the same N, but they do NOT scale as
1/sqrt(N) -- the prior curvature does not grow with the data -- so the sweep
is done properly, by re-inverting `(H_N + P)` at each N.
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

import groups as G  # noqa: E402


def psd_inv(A, rtol=1e-12):
    w, V = np.linalg.eigh(A)
    keep = w > max(rtol * w.max(), 0.0)
    return (V[:, keep] / w[keep]) @ V[:, keep].T


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fisher", required=True)
    p.add_argument("--cset", default="hit")
    p.add_argument("--arms", nargs="+", default=["cf", "gaussq"])
    p.add_argument("--ntrk", type=int, required=True,
                   help="tracks the fisher npz was built on")
    p.add_argument("--hit-prior", type=float, default=1.0)
    p.add_argument("--groups", default="/work/submit/david_w/ZMass/"
                   "CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/"
                   "materialGroups50.txt")
    p.add_argument("--sweep", type=float, nargs="*",
                   default=[2e4, 1e5, 3e5, 1e6, 7e6, 4.1e7])
    p.add_argument("--top", type=int, default=14)
    a = p.parse_args()

    d = np.load(a.fisher, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    npar = len(params)
    mats = [i for i, s in enumerate(params) if s.startswith("material_")]
    hits = [i for i, s in enumerate(params) if s.startswith("hitres_")]
    ng = len(mats)
    _, gpri = G.group_param_names(ng, a.groups)
    # `H` and `J` are PHYSICAL (`fisher_cmp.py` converts with the term's own
    # units), so the prior a saturation point is measured against is the
    # group's parmtype-15 tier prior itself, and `--hit-prior` for a class.
    pri = np.ones(npar)
    for k, i in enumerate(mats):
        pri[i] = gpri[k]
    for i in hits:
        pri[i] = a.hit_prior
    P = np.diag(1.0 / pri ** 2)

    print(f"# {a.ntrk} tracks, cset '{a.cset}', priors: material "
          f"{gpri.min():g}-{gpri.max():g} (parmtype-15 tier), hit "
          f"{a.hit_prior}")
    for arm in a.arms:
        H = np.asarray(d[f"H_{arm}_{a.cset}"], float)
        J = np.asarray(d[f"J_{arm}_{a.cset}"], float)
        # STANDALONE prior-free sandwich sigma, `sqrt(J_pp)/H_pp` -- the same
        # quantity `efficiency.py` calls `aa`.  The MARGINAL prior-free
        # version (`psd_inv(H) J psd_inv(H)`) is NOT usable here: with 32 of
        # the 42 groups carrying essentially no information of their own, the
        # pseudo-inverse drops their directions and returns sigma = 0, which
        # then reports `N_sat = 1e-27 tracks`.  The standalone form is well
        # defined for every parameter and scales exactly as 1/sqrt(N).
        hd = np.diag(H)
        live = hd > 1e-8 * np.max(hd)
        free = np.where(live, np.sqrt(np.clip(np.diag(J), 0, None))
                        / np.maximum(hd, 1e-300), np.inf)
        nsat = a.ntrk * (free / pri) ** 2
        print()
        print("=" * 96)
        print(f"ARM {arm}: prior-free sandwich sigma at {a.ntrk} tracks, and "
              f"the track count where it reaches the prior")
        print("=" * 96)
        print(f"{'parameter':<26}{'sigma_free':>12}{'/prior':>9}"
              f"{'N_sat':>13}   " + "".join(f"{n:>11.0f}" for n in a.sweep))
        print(f"{'':<26}{'':>12}{'':>9}{'':>13}   "
              + "".join(f"{'sig@N':>11}" for _ in a.sweep))
        for lab, idxs in (("MATERIAL", mats), ("HIT CLASSES", hits)):
            print(f"-- {lab}")
            keep = [i for i in idxs if live[i]]
            order = sorted(keep, key=lambda i: free[i])[: a.top]
            marg = {}
            for N in a.sweep:
                sc = N / a.ntrk
                C = psd_inv(H * sc + P)
                marg[N] = np.sqrt(np.clip(np.diag(C @ (J * sc) @ C), 0, None))
            for i in order:
                row = (f"{params[i]:<26}{free[i]:>12.4f}{free[i]/pri[i]:>9.3f}"
                       f"{nsat[i]:>13.3g}   ")
                for N in a.sweep:
                    row += f"{marg[N][i]:>11.4f}"
                print(row)
            if keep:
                v = nsat[keep]
                print(f"   {len(keep)}/{len(idxs)} {lab.lower()} with "
                      f"information; median N_sat {np.median(v):.3g} tracks, "
                      f"16-84 % {np.percentile(v,16):.3g}-"
                      f"{np.percentile(v,84):.3g}, max {np.max(v):.3g}")


if __name__ == "__main__":
    main()
