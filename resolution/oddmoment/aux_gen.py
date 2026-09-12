#!/usr/bin/env python3
"""GEN leg kinematics + per-family mass-variance shares, aligned to a pairs cache.

`cf_mass_likelihood.build_pairs_tt` keeps only what the CF model needs (z,
sigma, mgen, vgf and the exponents), so the candidate caches carry no truth
kinematics -- and truth kinematics are exactly what is needed to define a
sigma_bar that CANNOT see the mass fluctuation (the candidate analogue of
`oddmoment/sigma_pull.py` T2).

It also rebuilds the per-family split of the mass variance from the fit's own
**Q matrix**, `resinfvarv` grouped by the runtree `parmtype` of `reseigidx`:

    f_hit  = (sigma_m^2 - resinfcov)/sigma_m^2      (= the cached `vgf`)
    f_hitx = sum_{parmtype==8}  resinfvarv / sigma_m^2   (local-x hit resolution)
    f_hity = sum_{parmtype==9}  resinfvarv / sigma_m^2   (local-y, pixel only)
    f_ms   = sum_{parmtype==10} resinfvarv / sigma_m^2
    f_ioni = sum_{parmtype==11} resinfvarv / sigma_m^2
    f_mat  = sum_{parmtype==15} resinfvarv / sigma_m^2

**parmtype 15 is a RE-PARTITION of the parmtype-10/11 noise into material
groups, not an addition** (`sum_g dQ_g == dQMS + dQI`, in
`ResidualGlobalCorrectionMakerTwoTrackG4e.cc`), so it must NOT enter the
closure sum -- it is carried as an independent CHECK, `f_mat ~ f_ms + f_ioni`.
Likewise `f_hitx + f_hity ~ f_hit`. The closure that must hold is

    f_hit + f_ms + f_ioni = 1

and on both v2 productions it holds to a mean 7e-8 / max 8e-7 -- unlike the
CF-exponent "shares", which sum to 1.074 at one finite-difference step and to
0.980 at another, because Moliere and Landau have no finite second moment.
`f_other` is therefore restricted to parmtypes outside 8, 9, 10, 11, 15: summing
every non-10/11 family into it would fold 8, 9 and 15 back in and report a
closure of 2.0.

These give the CLOSED-FORM self-consistency coefficient of the mass pull.
sigma_m^2 = A m^4 + B m^2 + C, because the hit contribution to sigma_rel grows
as p (sigma_qop,hit is p-independent), the MS one is p-independent and the
ionization one falls as 1/p, so

    a_m = d ln sigma_m / d ln m * sigma_m/m = (2 f_hit + f_ms) * sigma_m/m
        = (1 + f_hit - f_ioni) * sigma_m/m .

Alignment against the cache is PROVEN, not assumed: the per-entry pull is
recomputed in float64 and matched bit-for-bit, the same contract
`censoring_aux.py` uses.

usage:
  python3 aux_gen.py --files GLOB --cache runs/cf_masspairs_X.npz \
      --out oddmoment/out/auxgen_X.npz [--nproc 32]
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import prodfiles  # noqa: E402  (needs resolution/ on sys.path)
import selection  # noqa: E402  (the standard two-track selection)

MJPSI = 3.0969
BR = ["run", "lumi", "event",
      "Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass",
      "Jpsigen_pt", "Jpsigen_eta", "Jpsigen_phi",
      "Muplusgen_pt", "Muplusgen_eta", "Muplusgen_phi",
      "Muminusgen_pt", "Muminusgen_eta", "Muminusgen_phi",
      "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
      "Muplus_nvalid", "Muminus_nvalid",
      "chisqval", "ndof", "niter",
      "resinfvarv", "reseigidx", "resinfcov"]
# THE STANDARD TWO-TRACK SELECTION's columns, carried through so a card built
# from the joined cache can apply it.  This script does NOT cut: its rows are
# joined ROW BY ROW to a pairs cache, so dropping any here would silently
# misalign the two.  Optional -- a production without `exportVtxResidual` has
# no `Jpsi_vtxz`.
BR_OPT = ["Jpsi_vtxz", "Jpsi_vtxsig", "Jpsi_vtxok"]


def one(fn):
    f = uproot.open(fn)
    pt = f["runtree"]["parmtype"].array(library="np")
    t = f["tree"]
    keys = set(k.split(";")[0] for k in t.keys())
    got_opt = [b for b in BR_OPT if b in keys]
    a = t.arrays(BR + got_opt, library="np")
    n = len(a["Jpsi_mass"])
    m = a["Jpsi_mass"].astype(np.float64)
    mg = a["Jpsigen_mass"].astype(np.float64)
    s = a["Jpsi_sigmamass"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (m - mg) / s
    fms = np.zeros(n); fio = np.zeros(n); fmat = np.zeros(n)
    fhx = np.zeros(n); fhy = np.zeros(n); fother = np.zeros(n)
    KNOWN = (8, 9, 10, 11, 15)
    for ic in range(n):
        gi = np.asarray(a["reseigidx"][ic])
        if not len(gi):
            continue
        vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
        fam = pt[gi]
        s2 = s[ic] * s[ic]
        if not (s2 > 0.):
            continue
        fhx[ic] = vb[fam == 8].sum() / s2
        fhy[ic] = vb[fam == 9].sum() / s2
        fms[ic] = vb[fam == 10].sum() / s2
        fio[ic] = vb[fam == 11].sum() / s2
        fmat[ic] = vb[fam == 15].sum() / s2
        fother[ic] = vb[~np.isin(fam, KNOWN)].sum() / s2
    with np.errstate(divide="ignore", invalid="ignore"):
        fhit = (s * s - a["resinfcov"].astype(np.float64)) / (s * s)
    return dict(
        # the join keys: `run`/`lumi`/`event` make the alignment
        # order-independent, which a cache built by `append_pairs.py` requires
        run=a["run"].astype(np.int64), lumi=a["lumi"].astype(np.int64),
        event=a["event"].astype(np.int64),
        z=z, sigma=s, mgen=mg, fhit=fhit, fms=fms, fioni=fio, fother=fother,
        fhitx=fhx, fhity=fhy, fmat=fmat,
        gpt_p=a["Muplusgen_pt"].astype(np.float64),
        gpt_m=a["Muminusgen_pt"].astype(np.float64),
        geta_p=a["Muplusgen_eta"].astype(np.float64),
        geta_m=a["Muminusgen_eta"].astype(np.float64),
        gphi_p=a["Muplusgen_phi"].astype(np.float64),
        gphi_m=a["Muminusgen_phi"].astype(np.float64),
        gjpt=a["Jpsigen_pt"].astype(np.float64),
        gjeta=a["Jpsigen_eta"].astype(np.float64),
        pt_p=a["Muplus_pt"].astype(np.float64),
        pt_m=a["Muminus_pt"].astype(np.float64),
        nv_p=a["Muplus_nvalid"].astype(np.float64),
        nv_m=a["Muminus_nvalid"].astype(np.float64),
        normchi2=(a["chisqval"] / np.maximum(a["ndof"], 1)).astype(np.float64),
        niter=a["niter"].astype(np.float64),
        sigmam=s,
        **{b.split("_", 1)[1]: np.asarray(a[b]).astype(np.float64)
           for b in got_opt},
    )


def join_to_cache(A, cache_path):
    """Return the index of each cache row in the concatenated tree arrays `A`.

    ORDER-INDEPENDENT JOIN on (run, lumi, event, z). Shared so that every aux
    extractor aligned to the same caches uses ONE implementation; `aux_seed.py`
    is the other consumer.
    """
    d = np.load(cache_path)
    zc = d["z"].astype(np.float64)
    n = len(zc)
    zt = A["z"]

    # ORDER-INDEPENDENT JOIN. A sequential scan would assume the cache is an
    # in-order subsequence of the tree. That is false for any cache built by
    # `fullscale/append_pairs.py` (base + a later tail) or from a different file
    # ordering, and it fails hundreds of thousands of rows in:
    # `zpairs_dyv2_full.npz` breaks at row 986 433 of 3 733 323.
    # `run`/`lumi`/`event` are cached precisely so this join can be done on a
    # key (that is what `append_pairs.py`'s disjointness check uses), and `z`
    # disambiguates the several candidates an event can carry.
    have_key = all(k in d.files for k in ("run", "lumi", "event")) and \
        all(k in A for k in ("run", "lumi", "event"))
    if have_key:
        def mkkey(src):
            return (np.asarray(src["run"], dtype=np.int64).astype(np.uint64) << np.uint64(44)) \
                 ^ (np.asarray(src["lumi"], dtype=np.int64).astype(np.uint64) << np.uint64(24)) \
                 ^ np.asarray(src["event"], dtype=np.int64).astype(np.uint64)
        kt, kc = mkkey(A), mkkey(d)
        table = {}
        for j, (k, z) in enumerate(zip(kt.tolist(), zt.tolist())):
            table.setdefault((k, z), j)
        idx = np.empty(n, dtype=np.int64)
        miss = 0
        for i, (k, z) in enumerate(zip(kc.tolist(), zc.tolist())):
            j = table.get((k, z))
            if j is None:
                if miss < 5:
                    print(f"  unmatched cache row {i}: run/lumi/event key "
                          f"{k}, z={z!r}", flush=True)
                miss += 1
                idx[i] = -1
            else:
                idx[i] = j
        if miss:
            sys.exit(f"alignment failed: {miss} of {n} cache rows have no "
                     "(run, lumi, event, z) match in the tree")
        print(f"joined on (run, lumi, event, z): {n} rows, order-independent",
              flush=True)
    else:
        print("cache or tree lacks run/lumi/event; falling back to the "
              "in-order scan (valid only if the cache is an in-order "
              "subsequence of the tree)", flush=True)
        idx = np.empty(n, dtype=np.int64)
        j = 0
        for i in range(n):
            while j < len(zt) and not (zt[j] == zc[i]):
                j += 1
            if j >= len(zt):
                sys.exit(f"alignment failed at cache row {i} (z={zc[i]!r})")
            idx[i] = j
            j += 1
    return idx, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nproc", type=int, default=32)
    p.add_argument("--ntasks", type=int, default=100000)
    a = p.parse_args()
    files = prodfiles.resolve(a.files, a.ntasks, logger=lambda m: print(m, flush=True))
    print(f"{len(files)} files", flush=True)
    from multiprocessing import Pool
    with Pool(a.nproc) as pool:
        parts = pool.map(one, files)
    A = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    print(f"{len(A['z'])} tree entries", flush=True)

    idx, d = join_to_cache(A, a.cache)
    out = {k: v[idx] for k, v in A.items()}
    # WHAT THE STANDARD SELECTION WOULD REMOVE.  Reported, not applied: these
    # rows are joined to the pairs cache row by row, so the cut belongs to the
    # card builder that consumes both (`resolution/selection.py`).
    _m, _s = selection.standard(out, None, n=len(idx))
    _s.log(lambda l: print(l, flush=True))
    assert np.array_equal(out["sigma"], d["sigma"]), "sigma mismatch"
    dv = np.abs(out["fhit"] - d["vgf"].astype(np.float64))
    print(f"ALIGNED: {len(idx)} rows, sigma bit-identical; "
          f"{len(A['z']) - len(idx)} tree rows dropped by the cache; "
          f"max|fhit - cache vgf| = {dv.max():.3e}", flush=True)
    tot = out["fhit"] + out["fms"] + out["fioni"]
    print(f"Q-matrix variance closure  <f_hit+f_ms+f_ioni> = {tot.mean():.9f} "
          f"(rms {tot.std():.2e}, max|.-1| {np.abs(tot-1).max():.2e})",
          flush=True)
    print(f"  <f_hit> = {out['fhit'].mean():.5f}  <f_ms> = {out['fms'].mean():.5f}"
          f"  <f_ioni> = {out['fioni'].mean():.4e}", flush=True)
    print(f"  CHECKS (not part of the sum): "
          f"<f_hitx+f_hity - f_hit> = "
          f"{(out['fhitx']+out['fhity']-out['fhit']).mean():.2e}; "
          f"<f_mat - (f_ms+f_ioni)> = "
          f"{(out['fmat']-out['fms']-out['fioni']).mean():.2e}; "
          f"<f_other> (unrecognised parmtypes, must be 0) = "
          f"{out['fother'].mean():.2e}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
