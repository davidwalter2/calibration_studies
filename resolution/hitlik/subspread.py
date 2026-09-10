#!/usr/bin/env python3
"""The EMPIRICAL estimator spread: K real fits on K DISJOINT subsamples.

Assumes nothing -- in particular not `H = J`, which is exactly what the
sandwich of `efficiency.py` is testing.  For each arm and each parameter,

    sigma_sub  = std over the K subsample fits (ddof = 1)
    sigma_full = sigma_sub / sqrt(K)          <- the full-sample error

and the ratio `sigma_full(Gauss) / sigma_full(CF)` squared is the same
EFFICIENCY the sandwich predicts.  Parameters whose subsample errors are at
their prior carry no information and are skipped.

usage:
    subspread.py --fits runs/hitlik/fits --k 8 --arms cf gaussq \\
        --compare runs/hitlik/efficiency.npz
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def read_fit(path):
    from rabbit import io_tools
    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in np.array(h.axes["parms"])]
    edm = np.nan
    if "edmval" in fr:
        v = fr["edmval"]
        edm = float(np.asarray(v.get() if hasattr(v, "get") else v))
    return names, np.asarray(h.values(), np.float64), \
        np.sqrt(np.asarray(h.variances(), np.float64)), edm


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--arms", nargs="+", default=["cf", "gaussq"])
    p.add_argument("--ref", default="cf")
    p.add_argument("--hit-prior", type=float, default=1.0)
    p.add_argument("--compare", default=None,
                   help="an efficiency.py npz, to put the sandwich next to it")
    p.add_argument("--groups", default="/work/submit/david_w/ZMass/"
                   "CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/"
                   "materialGroups50.txt")
    p.add_argument("--top", type=int, default=14)
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()

    import groups as G
    gnames, gpri = G.group_param_names(42, a.groups)
    prior_of = dict(zip(gnames, gpri))

    V, E, names = {}, {}, None
    for arm in a.arms:
        vals, errs, edms = [], [], []
        for k in range(a.k):
            f = os.path.join(a.fits, f"sub_{arm}_{k}", "fitresults.hdf5")
            if not os.path.exists(f):
                continue
            nm, v, e, edm = read_fit(f)
            names = names or nm
            vals.append(v)
            errs.append(e)
            edms.append(edm)
        if not vals:
            sys.exit(f"no subsample fits for arm '{arm}' under {a.fits}")
        V[arm] = np.stack(vals)
        E[arm] = np.stack(errs)
        print(f"   {arm}: {len(vals)}/{a.k} subsample fits, EDM "
              f"{np.nanmin(edms):.2e} .. {np.nanmax(edms):.2e}")

    npar = len(names)
    pv = np.array([prior_of.get(q, a.hit_prior if q.startswith("hitres_")
                                else np.inf) for q in names])
    K = {arm: V[arm].shape[0] for arm in a.arms}
    sub = {arm: V[arm].std(axis=0, ddof=1) for arm in a.arms}
    full = {arm: sub[arm] / np.sqrt(K[arm]) for arm in a.arms}
    quoted_sub = {arm: E[arm].mean(axis=0) for arm in a.arms}
    quoted_full = {arm: quoted_sub[arm] / np.sqrt(K[arm]) for arm in a.arms}

    cmp_ = np.load(a.compare, allow_pickle=True) if a.compare else None
    cnames = [str(s) for s in cmp_["params"]] if cmp_ is not None else []

    ref = a.ref
    live = quoted_sub[ref] < 0.98 * pv
    order = np.argsort(quoted_sub[ref] / np.where(np.isfinite(pv), pv, 1.0))
    for lab, pref in (("MATERIAL GROUPS", "material_"),
                      ("HIT CLASSES", "hitres_")):
        idxs = [i for i in order if names[i].startswith(pref) and live[i]]
        if not idxs:
            continue
        print()
        print("-" * 104)
        print(f"{lab}: {K[ref]} disjoint subsample fits, sigma scaled to the "
              f"full sample")
        print("-" * 104)
        hdr = f"{'parameter':<26}"
        for arm in a.arms:
            hdr += f"{arm + ' quoted':>13}{arm + ' SPREAD':>13}{'S/Q':>6}"
        hdr += f"{'EFF emp':>9}{'EFF sand':>9}"
        print(hdr)
        eff_e, eff_s = [], []
        for i in idxs[: a.top]:
            row = f"{names[i]:<26}"
            for arm in a.arms:
                row += (f"{quoted_full[arm][i]:>13.4f}{full[arm][i]:>13.4f}"
                        f"{full[arm][i]/max(quoted_full[arm][i],1e-300):>6.2f}")
            e_ = (full["gaussq"][i] / full[ref][i]) ** 2 \
                if "gaussq" in full else np.nan
            s_ = np.nan
            if cmp_ is not None and names[i] in cnames:
                j = cnames.index(names[i])
                s_ = (cmp_["sigma_actual_gaussq"][j]
                      / cmp_[f"sigma_actual_{ref}"][j]) ** 2
            eff_e.append(e_)
            eff_s.append(s_)
            row += f"{e_:>9.3f}{s_:>9.3f}"
            print(row)
        eff_e = np.array(eff_e)
        eff_s = np.array(eff_s)
        ok = np.isfinite(eff_e)
        print(f"   EMPIRICAL efficiency (spread ratio squared): median "
              f"{np.median(eff_e[ok]):.3f}, 16-84 % "
              f"{np.percentile(eff_e[ok],16):.3f}-"
              f"{np.percentile(eff_e[ok],84):.3f}")
        if np.isfinite(eff_s).any():
            print(f"   SANDWICH efficiency on the same parameters:   median "
                  f"{np.nanmedian(eff_s):.3f}")
        print(f"   NOTE: with K = {K[ref]} the spread itself carries a "
              f"{100/np.sqrt(2*(K[ref]-1)):.0f} % statistical error per "
              f"parameter, so read the MEDIAN, not a single row.")
        if a.output:
            np.savez_compressed(
                a.output, params=np.array(names, dtype=object),
                **{f"spread_{arm}": full[arm] for arm in a.arms},
                **{f"quoted_{arm}": quoted_full[arm] for arm in a.arms})


if __name__ == "__main__":
    main()
