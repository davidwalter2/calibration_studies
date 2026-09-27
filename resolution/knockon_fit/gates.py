#!/usr/bin/env python3
"""Validation gates of the fit-level knock-on families (cf_knockon.map_rows,
map_block, map_rad, joint_rows, pair_weights) on real maker rows.

  gate 1a  decomposition: map + joint = step_correction on the same nodes
           (+ the closed-form part below the map's lower limit)
  gate 1b  brute force: dense trapezoid per row (map from e0), <= 3e-3 relative
  gate 1c  mean: Im dS_x(t)/t at small t = map_mean
  gate 1d  map lower limit FIT_MAP_TLO 1e-4 against 1e-6
  gate 4   (--pool) pooled against per-row, per track

usage: gates.py FILE [--entries N]
"""
import argparse
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cf_knockon as ck                                   # noqa: E402
import cf_track_resolution as ctr                         # noqa: E402
import cf_delta_ray as cdr                                # noqa: E402
import prodfiles                                          # noqa: E402

TG = ctr.TG


def load(fn, entries):
    f = uproot.open(fn)
    pt = f["runtree"]["parmtype"].array(library="np")
    t = f["tree"]
    br = ["refCov", "reseigidx", "resinfvarv", "genParms", "msmoliidx",
          "msmoliv", "ioniurbanidx", "ioniurbanv", "radstepidx", "radstepv",
          "radstepspecv", "radvgrid", "ioniqscaleidx", "ioniqscalev"]
    br += prodfiles.stride_keys(t, ("msmoliv", "ioniurbanv", "radstepv",
                                    "radstepspecv"))
    a = t.arrays(br, library="np", entry_stop=entries)
    out = []
    for ic in range(len(a["refCov"])):
        c00 = float(a["refCov"][ic][0])
        if not c00 > 0 or a["genParms"][ic][0] == 0:
            continue
        sig = np.sqrt(c00)
        chg = np.sign(a["genParms"][ic][0])
        gi = np.asarray(a["reseigidx"][ic])
        vb = np.asarray(a["resinfvarv"][ic], np.float64)
        fam = pt[gi]
        rec = {}
        for nm, ib, vbn, mc in (("M", "msmoliidx", "msmoliv", 8),
                                ("I", "ioniurbanidx", "ioniurbanv", 11),
                                ("R", "radstepidx", "radstepv", 11),
                                ("P", "radstepidx", "radstepspecv", 1)):
            idx = np.asarray(a[ib][ic])
            rec[nm + "idx"] = idx
            rec[nm] = prodfiles.reshape_records(
                a[vbn][ic], nrec=len(idx),
                stride=prodfiles.entry_stride(a, vbn, ic), branch=vbn,
                min_cols=mc).astype(np.float64)
        qsi = np.asarray(a["ioniqscaleidx"][ic])
        qsv = np.asarray(a["ioniqscalev"][ic], np.float64).reshape(-1, 2)
        wms, wio = {}, {}
        for g in np.unique(gi[fam == 10]):
            vp = vb[(fam == 10) & (gi == g)].sum()
            st = rec["M"][rec["Midx"] == g]
            if vp > 0 and len(st) and st[:, 5].sum() > 0:
                wms[int(g)] = np.sqrt(vp / st[:, 5].sum()) / sig
        for g in np.unique(gi[fam == 11]):
            vp = vb[(fam == 11) & (gi == g)].sum()
            st = rec["I"][rec["Iidx"] == g]
            if vp > 0 and len(st):
                sq2 = ctr.ioni_sq2(st, qsv[qsi == g])
                if sq2 > 0:
                    wio[int(g)] = chg * np.sqrt(vp / sq2) / sig
        rec.update(wms=wms, wio=wio, sig=sig, chg=chg,
                   vg=np.asarray(a["radvgrid"][ic], np.float64))
        out.append(rec)
    return out


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def brute_T(tlo, thi, n=4_000_000):
    return np.geomspace(tlo, thi, n)


def trap(x, y):
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--entries", type=int, default=3)
    ap.add_argument("--nbrute", type=int, default=6)
    ap.add_argument("--pool", action="store_true", help="gate 4 only")
    args = ap.parse_args()
    if args.pool:
        return pool_gate(args.file, args.entries)
    recs = load(args.file, args.entries)
    taus = TG[[1, 64, 200, -1]]
    tcut, cap = ctr.DELTA_TCUT, ctr.DELTA_TMAXCAP
    worst = dict(dec_map=0., dec_joint=0., brute_map=0., brute_joint=0.,
                 mean=0., tlo=0.)
    for r in recs:
        beta, alpha = ck.pair_weights(r["Midx"], r["Ridx"], r["M"], r["R"],
                                      r["wms"], r["wio"])
        I = r["I"]
        # ---- map piece, per ionisation row
        rows = [s for s in range(len(I)) if I[s, 0] in (2, 3)
                and int(r["Iidx"][s]) in r["wio"]]
        for s in rows[:args.nbrute]:
            row = I[s:s + 1]
            w = r["wio"][int(r["Iidx"][s])]
            got = ck.map_block(row, w, taus)
            gam = row[0, 9]
            xi, e0 = row[0, 6] * gam, row[0, 7] * gam
            tmax, b2, E = row[0, 8] * gam, row[0, 11], row[0, 12]
            p = E * np.sqrt(b2)
            al = w * row[0, 10] * 1e-3
            # 1a: step_correction (joint off, exact on) on the same nodes
            T = ck.nodes(e0, tmax, ck.FIT_NPERDEC, tlo=max(e0, ck.FIT_MAP_TLO * tmax))
            dN = ck.rate(T, xi, tmax, b2, E, row[0, 0] == 2,
                         ctr.IONI_KOKOULIN != 0 and bool(ctr._is_muon_record(b2, E)
                                                          and E - ctr._MMU > ctr._KOK_MUMIN))
            ref = ck.step_correction(taus * al, taus * 0, T, dN, ck.t_eff(T, E, p),
                                     ck.dteff_dt(T, E, p), None, False, True)
            ref = ref + ck._map_below(e0, T[0], xi, ck._teff_c2(E, p), taus * al)
            worst["dec_map"] = max(worst["dec_map"], rel(got, ref))
            # 1b: brute force (linear in T is fine at 4e6 log nodes)
            # from e0 (the closed-form part below the lower limit included),
            # in the cancellation-free form 2i sin(a d/2) e^{i a (2T + d)/2}
            Tb = brute_T(e0, tmax * (1 - 1e-12), 16_000_000)
            dNb = ck.rate(Tb, xi, tmax, b2, E, row[0, 0] == 2, False) \
                if ctr.IONI_KOKOULIN == 0 else None
            if dNb is not None:
                d = ck.t_eff(Tb, E, p) - Tb
                bf = np.array([trap(Tb, dNb * 2j * np.sin(0.5 * t * al * d)
                                    * np.exp(0.5j * t * al * (2 * Tb + d)))
                               for t in taus])
                worst["brute_map"] = max(worst["brute_map"], rel(got, bf))
            # 1c: the mean
            ts = 1e-4
            num = ck.map_block(row, w, np.array([ts])).imag[0] / ts
            ana = ck.map_mean(row, w)
            worst["mean"] = max(worst["mean"], abs(num / ana - 1) if ana else 0)
            # 1d: lower limit
            old = ck.FIT_MAP_TLO
            ck.FIT_MAP_TLO = 1e-6
            lo6 = ck.map_block(row, w, taus)
            ck.FIT_MAP_TLO = old
            worst["tlo"] = max(worst["tlo"], rel(got, lo6))
            print(f"map row: tmax={tmax:.3g} MeV a*tmax*t_max={al*tmax*TG[-1]:.3g} "
                  f"|dS|={np.abs(got).max():.3e} mean={ana:.3e}/{num:.3e}")
        # ---- joint piece, per MS row
        M = r["M"]
        ok = np.flatnonzero((alpha != 0) & (beta != 0) & (M[:, 2] > 0))
        # the heaviest rows (largest xg) first
        ok = ok[np.argsort(-M[ok, 2])][:args.nbrute]
        for s in ok:
            got = ck.joint_rows(M[s:s + 1], alpha[s], beta[s], taus, tcut, cap, True)
            bt = min(M[s, 4], 1 - 1e-15)
            xi = cdr.xi_mev(M[s, 0], M[s, 1], M[s, 2], bt)
            tkin = cdr._tmx(M[s:s + 1], tcut, 0)[0] * 1e3
            thi = min(tkin, cap * 1e3)
            tlo = tcut * 1e3
            fsp = 1 - 0.5 * bt ** 2 / np.log(max(thi / tlo, 1.0001))
            p = M[s, 3] * 1e3
            E = p / bt
            T = ck._joint_nodes(tlo, thi, tkin, ck.FIT_NPERDEC)
            dN = fsp * xi / T ** 2
            th = np.sqrt(2 * ck.ME_MEV * T) / p
            X, dX = ck.t_eff(T, E, p), ck.dteff_dt(T, E, p)
            full = ck.step_correction(taus * alpha[s], taus * beta[s], T, dN, X, dX,
                                      th, True, True)
            mp = ck.step_correction(taus * alpha[s], taus * 0, T, dN, X, dX,
                                    None, False, True)
            worst["dec_joint"] = max(worst["dec_joint"], rel(got, full - mp))
            Tb = brute_T(tlo, thi * (1 - 1e-12))
            dNb = fsp * xi / Tb ** 2
            thb = np.sqrt(2 * ck.ME_MEV * Tb) / p
            Xb = ck.t_eff(Tb, E, p)
            from scipy.special import j0
            bf = np.array([trap(Tb, dNb * (np.exp(1j * t * alpha[s] * Xb) - 1)
                                * (j0(t * beta[s] * thb) - 1)) for t in taus])
            worst["brute_joint"] = max(worst["brute_joint"], rel(got, bf))
            print(f"joint row: p={p:.4g} MeV thi={thi:.3g} a*thi*t={alpha[s]*thi*TG[-1]:.3g} "
                  f"b*th*t={beta[s]*th[-1]*TG[-1]:.3g} |dS|={np.abs(got).max():.3e}")
    print({k: f"{v:.2e}" for k, v in worst.items()})



def pool_gate(fn, entries):
    """gate 4: pooled against per-row, per track, absolute on S over TG."""
    recs = load(fn, entries)
    tcut, cap = ctr.DELTA_TCUT, ctr.DELTA_TMAXCAP
    import time
    worst_x = worst_j = 0.0
    tt = dict(xr=0., xp=0., jr=0., jp=0.)
    for r in recs:
        I, Ii = r["I"], r["Iidx"]
        t0 = time.time()
        sx = sum((ck.map_block(I[Ii == g], w, TG) for g, w in r["wio"].items()),
                 np.zeros(len(TG), complex))
        t1 = time.time()
        px = sum((ck.fit_map(I[Ii == g], w, TG) for g, w in r["wio"].items()),
                 np.zeros(len(TG), complex))
        t2 = time.time()
        beta, alpha = ck.pair_weights(r["Midx"], r["Ridx"], r["M"], r["R"],
                                      r["wms"], r["wio"])
        sj = ck.joint_rows(r["M"], alpha, beta, TG, tcut, cap, True)
        t3 = time.time()
        keys = np.stack([r["Midx"], r["Ridx"], r["M"][:, ck.MS_GROUP_COL]], axis=1)
        pj = ck.fit_joint(r["M"], alpha, beta, r["Midx"], r["Ridx"], TG, tcut, cap, True)
        t4 = time.time()
        tt["xr"] += t1 - t0; tt["xp"] += t2 - t1; tt["jr"] += t3 - t2; tt["jp"] += t4 - t3
        dx, dj = np.abs(sx - px).max(), np.abs(sj - pj).max()
        worst_x, worst_j = max(worst_x, dx), max(worst_j, dj)
        print(f"map |S|max={np.abs(sx).max():.2e} |d|={dx:.2e}   joint |S|max="
              f"{np.abs(sj).max():.2e} |d|={dj:.2e}  nrows={len(r['M'])} "
              f"npool={len(np.unique(keys, axis=0))}")
    print(f"worst |dS| map {worst_x:.2e} joint {worst_j:.2e}; time {tt}")


if __name__ == "__main__":
    main()
