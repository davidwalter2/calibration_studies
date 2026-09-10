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
                   "CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/"
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
    # the card is WHITENED: a material parameter is in units of its own tier
    # prior, so every material prior sigma is 1 in card units.  The hit
    # classes carry `--hit-prior` (1.0 by default) in the same way.
    pri = np.ones(npar)
    for i in hits:
        pri[i] = a.hit_prior
    P = np.diag(1.0 / pri ** 2)

    print(f"# {a.ntrk} tracks, cset '{a.cset}', priors: material 1.0 "
          f"(whitened tier), hit {a.hit_prior}")
    for arm in a.arms:
        H = d[f"H_{arm}_{a.cset}"]
        J = d[f"J_{arm}_{a.cset}"]
        Hi = psd_inv(H)
        S = Hi @ J @ Hi
        free = np.sqrt(np.clip(np.diag(S), 0, None))
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
            order = sorted(idxs, key=lambda i: free[i])[: a.top]
            for i in order:
                row = (f"{params[i]:<26}{free[i]:>12.4f}{free[i]/pri[i]:>9.3f}"
                       f"{nsat[i]:>13.3g}   ")
                for N in a.sweep:
                    s = N / a.ntrk
                    Sn = psd_inv(H * s + P) @ (J * s) @ psd_inv(H * s + P)
                    row += f"{np.sqrt(max(Sn[i, i], 0)):>11.4f}"
                print(row)
            med = np.median(nsat[idxs])
            print(f"   median N_sat over {lab.lower()}: {med:.3g} tracks "
                  f"(range {np.min(nsat[idxs]):.3g} - {np.max(nsat[idxs]):.3g})")


if __name__ == "__main__":
    main()
