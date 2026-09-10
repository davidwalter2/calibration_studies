#!/usr/bin/env python3
"""The one table: errors, fitted values at truth, and injection recovery.

Per parameter, for the CF arm and the Gaussian (fit's Q) arm:

    sigma CF (Fisher)        the CF term's own error -- and, since the CF
                             describes the data, its actual one
    sigma Gauss QUOTED       what a chi2 fit would report
    sigma Gauss ACTUAL       what it would actually scatter by (sandwich; the
                             bootstrap agrees to 0.1 %, and `--subspread` adds
                             the assumption-free version)
    EFFICIENCY               (sigma Gauss ACTUAL / sigma CF)^2
    fitted value at truth    both arms, EDM-certified, physical k / eps
    injection recovery       both arms, prior-corrected

Errors from `efficiency.py` are at `--ntrk-fisher` tracks and the fits at
`--ntrk-fit`; the Fisher column is rescaled to the fit's statistics so the two
can be read side by side.
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

from recovery import read_fit  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--efficiency", required=True)
    p.add_argument("--fits", required=True)
    p.add_argument("--cf", default="cf")
    p.add_argument("--gauss", default="gaussq")
    p.add_argument("--inj-cf", default="inj_cf")
    p.add_argument("--inj-gauss", default="inj_gaussq")
    p.add_argument("--inj-param", default="material_tib_support")
    p.add_argument("--inj-truth", type=float, default=0.00243951)
    p.add_argument("--inj-prior", type=float, default=0.0025)
    p.add_argument("--hit-inj-cf", default="inj_hit")
    p.add_argument("--hit-inj-gauss", default="inj_hit_gaussq")
    p.add_argument("--hit-param", default="hitres_str_N3_lo")
    p.add_argument("--hit-inj", type=float, default=0.10)
    p.add_argument("--ntrk-fisher", type=int, default=20000)
    p.add_argument("--ntrk-fit", type=int, default=6000)
    p.add_argument("--groups", default="/work/submit/david_w/ZMass/"
                   "CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/"
                   "materialGroups50.txt")
    p.add_argument("--top", type=int, default=13)
    a = p.parse_args()

    import groups as G
    gnames, gpri = G.group_param_names(42, a.groups)
    pri_of = dict(zip(gnames, gpri))
    units = {n: 1.0 / p_ for n, p_ in zip(gnames, gpri)}

    d = np.load(a.efficiency, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    pv = np.asarray(d["prior"], float)
    sc = np.sqrt(a.ntrk_fisher / a.ntrk_fit)      # Fisher -> the fits' stats

    F = {}
    for lab, sub in (("cf", a.cf), ("gauss", a.gauss)):
        F[lab] = read_fit(os.path.join(a.fits, sub, "fitresults.hdf5"))
    I = {}
    for lab, sub in (("cf", a.inj_cf), ("gauss", a.inj_gauss),
                     ("hcf", a.hit_inj_cf), ("hgauss", a.hit_inj_gauss)):
        f = os.path.join(a.fits, sub, "fitresults.hdf5")
        I[lab] = read_fit(f) if os.path.exists(f) else None

    def val(t, nm, u=1.0):
        if t is None or nm not in t["names"]:
            return np.nan, np.nan
        j = t["names"].index(nm)
        return t["val"][j] * u, t["err"][j] * u

    print("=" * 132)
    print(f"THE FINAL TABLE.  sigma columns at {a.ntrk_fit} tracks x 4 "
          f"components (the Fisher ones rescaled from {a.ntrk_fisher} by "
          f"sqrt({a.ntrk_fisher}/{a.ntrk_fit}) = {sc:.3f}).")
    print("   Fitted values are physical (k = ln material amount, eps = "
          "linear hit-variance scale), EDM-certified.")
    print("=" * 132)
    print(f"   EDM: cf {F['cf'].get('edmval', np.nan):.2e}   "
          f"gaussq {F['gauss'].get('edmval', np.nan):.2e}")

    live = np.asarray(d["sigma_quoted_cf"], float) < 0.98 * pv
    order = np.argsort(np.asarray(d["sigma_quoted_cf"], float)
                       / np.where(np.isfinite(pv), pv, 1.0))
    for lab, pref in (("MATERIAL GROUPS", "material_"),
                      ("HIT CLASSES", "hitres_")):
        idxs = [i for i in order if params[i].startswith(pref)
                and (live[i] or pref == "hitres_")]
        if not idxs:
            continue
        print()
        print("-" * 132)
        print(lab)
        print("-" * 132)
        print(f"{'parameter':<24}{'sig CF':>9}{'sig G quot':>11}"
              f"{'sig G ACT':>10}{'EFF':>7}   "
              f"{'CF value at truth':>22}{'Gauss value at truth':>24}")
        for i in idxs[: a.top]:
            nm = params[i]
            u = units.get(nm, 1.0)
            scf = float(d["sigma_actual_cf"][i]) * sc
            sgq = float(d["sigma_quoted_gaussq"][i]) * sc
            sga = float(d["sigma_actual_gaussq"][i]) * sc
            eff = (sga / scf) ** 2 if scf > 0 else np.nan
            v1, e1 = val(F["cf"], nm, u if nm.startswith("material_") else 1.0)
            v2, e2 = val(F["gauss"], nm,
                         u if nm.startswith("material_") else 1.0)
            print(f"{nm:<24}{scf:>9.4f}{sgq:>11.4f}{sga:>10.4f}{eff:>7.2f}   "
                  f"{v1:>13.4f} +-{e1:>7.4f}"
                  f"{v2:>15.4f} +-{e2:>7.4f}")

    print()
    print("-" * 132)
    print("INJECTION RECOVERY (prior-corrected: the posterior moves by "
          "f = 1 - sigma_post^2/sigma_pri^2 of a likelihood shift)")
    print("-" * 132)
    print(f"{'injection':<34}{'arm':<8}{'baseline':>11}{'injected':>11}"
          f"{'shift':>11}{'f_pri':>7}{'corrected/truth':>17}{'pull':>7}")
    for nm, tru, spri, kcf, kg, lab in (
            (a.inj_param, a.inj_truth, a.inj_prior, "cf", "gauss",
             f"{a.inj_param} x1.05 material"),
            (a.hit_param, None, 1.0, "hcf", "hgauss",
             f"{a.hit_param} x1.10 variance")):
        for arm, key in (("CF", kcf), ("Gauss", kg)):
            b = F["cf"] if arm == "CF" else F["gauss"]
            q = I[key]
            if q is None or nm not in b["names"]:
                print(f"{lab:<34}{arm:<8}   (fit not present)")
                continue
            ib, iq = b["names"].index(nm), q["names"].index(nm)
            v0, e0 = b["val"][ib], b["err"][ib]
            v1, e1 = q["val"][iq], q["err"][iq]
            sh = v1 - v0
            t = tru if tru is not None else \
                -(a.hit_inj / (1.0 + a.hit_inj)) * (1.0 + v0)
            f_pri = 1.0 - (e1 ** 2) / (spri ** 2) if e1 < spri else np.nan
            corr = sh / f_pri if np.isfinite(f_pri) and f_pri else np.nan
            slik = e1 / np.sqrt(f_pri) if np.isfinite(f_pri) else e1
            r = abs(corr) / abs(t) if t else np.nan
            print(f"{lab:<34}{arm:<8}{v0:>11.5f}{v1:>11.5f}{sh:>11.5f}"
                  f"{f_pri:>7.3f}{r:>17.3f}"
                  f"{(abs(corr)-abs(t))/slik:>7.2f}")
            lab = ""


if __name__ == "__main__":
    main()
