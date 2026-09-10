#!/usr/bin/env python3
"""Tables for the residual-vector CF likelihood study.

Reads the ``fisher_cmp.py`` output (observed information at MC truth for each
arm and component subset) and the ``globalfit/extract.py`` quadratic term over
the same production, and prints:

1. the per-component data pull variance and each arm's model variance;
2. sigma(k_g) and sigma(eps_c) per arm, marginal and standalone;
3. the INFORMATION RATIOS  (sigma_B / sigma_A)^2  between arms and between
   component subsets;
4. the same against the quadratic hit-chi2 term, scaled to the same number of
   tracks.

Everything is in PHYSICAL units: ``k_g`` is the log material amount of group
``g`` and ``eps_c`` the linear scale of hit class ``c``'s variance.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "globalfit"),
           os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sigmas(H, ridge=0.0):
    """(sigma_marginal, sigma_standalone, npos) from an information matrix.

    A direction with non-positive information has no error; it is returned as
    inf and counted.
    """
    d = np.diag(H).copy()
    alone = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), np.inf)
    w, V = np.linalg.eigh(H)
    keep = w > max(1e-12 * w.max(), 0.0)
    Hinv = (V[:, keep] / w[keep]) @ V[:, keep].T
    dm = np.diag(Hinv)
    marg = np.where(dm > 0, np.sqrt(np.maximum(dm, 0.0)), np.inf)
    return marg, alone, int(keep.sum()), w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fisher", required=True)
    p.add_argument("--quad", default=None)
    p.add_argument("--ntrk-res", type=int, default=20000,
                   help="tracks in the residual term (for the quad scaling)")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()

    d = np.load(a.fisher, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    params = [str(s) for s in d["params"]]
    keyset = sorted(meta)
    arms = sorted({k.rsplit("_", 1)[0] for k in keyset})
    csets = sorted({k.rsplit("_", 1)[1] for k in keyset})
    out = {}

    print("=" * 78)
    print("1. THE DATA, AND WHAT EACH ARM SAYS ITS VARIANCE IS")
    print("=" * 78)
    print(f"{'component':<12}{'N':>9}{'mean z':>10}{'Var(z)':>10}"
          + "".join(f"{'modelvar ' + m:>18}" for m in arms))
    ncomp = max(int(c) for cs in csets for c in cs) + 1
    for k in range(ncomp):
        if f"varz_c{k}" not in d.files:
            continue
        cs = [c for c in csets if str(k) in c]
        row = (f"{['q/p','lambda','phi','d0','z0'][k]:<12}"
               f"{int(d[f'n_c{k}']):>9}{float(d[f'meanz_c{k}']):>10.4f}"
               f"{float(d[f'varz_c{k}']):>10.4f}")
        for m in arms:
            v = [float(d[f"modelvar_{m}_{c}"]) for c in cs
                 if f"modelvar_{m}_{c}" in d.files]
            row += f"{(np.mean(v) if v else np.nan):>18.4f}"
        print(row)
    print("\n(model variance is the row mean over the components of the subset,"
          "\n so it is only exactly per-component for the 1-component subset)")

    print()
    print("=" * 78)
    print("2. NLL AT MC TRUTH -- how well each arm describes the residuals")
    print("=" * 78)
    print(f"{'subset':<10}{'rows':>9}" + "".join(f"{'NLL ' + m:>16}" for m in arms)
          + f"{'dNLL/row (gaussq-cf)':>24}")
    for cs in csets:
        n = meta[f"{arms[0]}_{cs}"]["n"]
        row = f"{cs:<10}{n:>9}"
        for m in arms:
            row += f"{float(d[f'nll0_{m}_{cs}']):>16.2f}"
        if "cf" in arms and "gaussq" in arms:
            dd = float(d[f"nll0_gaussq_{cs}"]) - float(d[f"nll0_cf_{cs}"])
            row += f"{dd / n:>24.4f}"
        print(row)

    print()
    print("=" * 78)
    print("3. sigma PER PARAMETER (physical units: k = ln material amount,")
    print("   eps = linear hit-variance scale), marginal over all 60")
    print("=" * 78)
    S = {}
    for cs in csets:
        for m in arms:
            H = d[f"H_{m}_{cs}"]
            marg, alone, npos, w = sigmas(H)
            S[(m, cs)] = (marg, alone, npos, w)
            out[f"sigma_marg_{m}_{cs}"] = marg
            out[f"sigma_alone_{m}_{cs}"] = alone
            print(f"   {m:<8} {cs:<6} rank {npos}/{len(params)}   "
                  f"eig min {w.min():+.3e} max {w.max():.3e}")

    # order the parameters by the CF full-vector standalone information
    ref = ("cf", csets[-1])
    order = np.argsort(S[ref][1])
    mats = [i for i in order if params[i].startswith("material_")]
    hits = [i for i in order if params[i].startswith("hitres_")]

    def table(idxs, title):
        print()
        print("-" * 78)
        print(title)
        print("-" * 78)
        hdr = f"{'parameter':<28}"
        for m in arms:
            for cs in csets:
                hdr += f"{m + '/' + cs:>14}"
        print(hdr)
        for i in idxs[: a.top]:
            row = f"{params[i]:<28}"
            for m in arms:
                for cs in csets:
                    v = S[(m, cs)][0][i]
                    row += (f"{v:>14.4f}" if np.isfinite(v) and v < 1e3
                            else f"{'--':>14}")
            print(row)

    table(mats, "3a. MATERIAL GROUPS -- sigma(k), marginal "
                "(the 12 best-measured)")
    table(hits, "3b. HIT CLASSES -- sigma(eps), marginal (the 12 best)")

    print()
    print("=" * 78)
    print("4. INFORMATION RATIOS")
    print("=" * 78)
    pairs = []
    for cs in csets:
        for base in ("gaussq", "gauss"):
            if base in arms and "cf" in arms:
                pairs.append((("cf", cs), (base, cs), f"CF / {base}  [{cs}]"))
    if len(csets) > 1 and "cf" in arms:
        pairs.append((("cf", csets[-1]), ("cf", csets[0]),
                      f"CF {csets[-1]} / CF {csets[0]}"))
        if "gaussq" in arms:
            pairs.append((("gaussq", csets[-1]), ("gaussq", csets[0]),
                          f"gaussq {csets[-1]} / gaussq {csets[0]}"))
    print(f"{'parameter':<28}" + "".join(f"{t:>22}" for _, _, t in pairs))
    for i in list(mats[: a.top]) + list(hits[: a.top]):
        row = f"{params[i]:<28}"
        for A, B, _ in pairs:
            sa, sb = S[A][0][i], S[B][0][i]
            r = (sb / sa) ** 2 if np.isfinite(sa) and np.isfinite(sb) and sa > 0 \
                else np.nan
            row += f"{r:>22.3f}" if np.isfinite(r) else f"{'--':>22}"
        print(row)
    for A, B, t in pairs:
        sa, sb = S[A][0], S[B][0]
        ok = np.isfinite(sa) & np.isfinite(sb) & (sa > 0) & (sa < 1e3) & (sb < 1e3)
        okm = ok & np.array([p.startswith("material_") for p in params])
        okh = ok & np.array([p.startswith("hitres_") for p in params])
        r = (sb / sa) ** 2
        print(f"   {t}: median over material {np.median(r[okm]):.3f} "
              f"({okm.sum()} live), over hit classes {np.median(r[okh]):.3f} "
              f"({okh.sum()} live)")
        out["ratio_" + t.replace(" ", "").replace("/", "_over_")] = r

    # ---- 5. against the quadratic term -------------------------------------
    if a.quad:
        q = np.load(a.quad, allow_pickle=False)
        pt, si = q["parmtype"], q["subidx"]
        idx = np.where(np.isin(pt, (14, 15)))[0]
        Hq = 0.5 * q["hess"][np.ix_(idx, idx)].astype(np.float64)
        ncand = int(q["ncand_quadratic"])
        margq, aloneq, nposq, wq = sigmas(Hq)
        # material column of group g -> row in Hq
        col = {int(si[i]): j for j, i in enumerate(idx) if pt[i] == 15}
        matnames = [p for p in params if p.startswith("material_")]
        gidx = {nm: g for g, nm in enumerate(matnames)}
        sc = np.sqrt(ncand / max(a.ntrk_res, 1))
        print()
        print("=" * 78)
        print("5. THE EXISTING QUADRATIC (hit-chi2) TERM over the SAME production")
        print(f"   {ncand} tracks, {len(idx)} parameters (50 field + 42 material),"
              f" rank {nposq}")
        print(f"   scaled to {a.ntrk_res} tracks: sigma x {sc:.3f}")
        print("=" * 78)
        hdr = (f"{'parameter':<28}{'quad all':>11}{'quad@' + str(a.ntrk_res):>12}")
        for m in arms:
            hdr += f"{'res ' + m:>12}"
        hdr += f"{'I_res/I_quad':>14}"
        print(hdr)
        out["sigma_quad"] = margq
        out["sigma_quad_scaled"] = margq * sc
        out["quad_ncand"] = ncand
        rat = {}
        for i in mats:
            nm = params[i]
            g = gidx.get(nm)
            if g is None or g not in col:
                continue
            k = col[g]
            row = f"{nm:<28}{margq[k]:>11.4f}{margq[k]*sc:>12.4f}"
            for m in arms:
                row += f"{S[(m, csets[-1])][0][i]:>12.4f}"
            r = (margq[k] * sc / S[("cf", csets[-1])][0][i]) ** 2
            rat[nm] = r
            row += f"{r:>14.3f}"
            if len(rat) <= a.top:
                print(row)
        rr = np.array([v for v in rat.values() if np.isfinite(v)])
        if len(rr):
            print(f"   median I_res(CF, {csets[-1]}) / I_quad @ same tracks: "
                  f"{np.median(rr):.3f}  over {len(rr)} groups")
        out["ratio_res_over_quad"] = np.array(
            [rat.get(p, np.nan) for p in params])

    if a.output:
        out["params"] = np.array(params, dtype=object)
        np.savez_compressed(a.output, **out)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
