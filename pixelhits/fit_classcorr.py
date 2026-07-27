#!/usr/bin/env python
"""MC closure fit of the pixel pathological-hit class-correction
parameters (parmtypes 16-21) from a runCvhJpsiGenMC.py run with
fitFromGenParms=True keepPixelEdgeHits=True pixelMinSizeX=1
pixelHitClassCorrections=True fillGrads=True.

Aggregates the per-candidate gradient (gradv) and packed Hessian
(hesspackedv, row-major upper triangle over globalidxv) restricted to
the class parameters, grouped per (parmtype, subdet, layer), and solves
theta = -H^{-1} g with all other global parameters fixed at their MC
truth (zero) -- valid for closure on unmodified MC.

Parameters are local translations in cm; results reported in um.
"""

import argparse
import glob
from collections import defaultdict

import numpy as np
import uproot

PARMNAMES = {16: "edge-x-mean", 17: "edge-x-diff", 18: "edge-y-mean",
             19: "edge-y-diff", 20: "sizeX1", 21: "sizeY1",
             22: "dtanLA[e-3]"}
CM2UM = 1e4
# parmtype 22 is dimensionless (delta-tan thetaL); report in units of 1e-3
PARMSCALE = {22: 1e3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    p.add_argument("--per-module", action="store_true",
                   help="fit per module instead of per (parmtype, subdet, layer) group")
    p.add_argument("--with-alignment", action="store_true",
                   help="simultaneous fit: additionally free the per-module "
                        "pixel local-x/y alignment translations (parmtypes 0/1 "
                        "on pixel modules) with a Gaussian prior, so the class "
                        "parameters are marginalised over pixel misalignment")
    p.add_argument("--align-prior-um", type=float, default=20.,
                   help="Gaussian prior width on the alignment deviations (um); "
                        "regularises the weak modes of the free fit")
    p.add_argument("--align-6dof", action="store_true",
                   help="free all six pixel alignment dofs per module "
                        "(translations x/y/z + rotations tx/ty/tz) instead "
                        "of local-x/y only; rotations get a prior of "
                        "align-prior-um * 1e-4 rad (i.e. the same transverse "
                        "scale over a 1 cm lever arm)")
    p.add_argument("--run-min", type=int, default=0,
                   help="only accumulate candidates with run >= this")
    p.add_argument("--run-max", type=int, default=10**9,
                   help="only accumulate candidates with run <= this")
    p.add_argument("--lumi-min", type=int, default=0,
                   help="only accumulate candidates with lumisection >= this")
    p.add_argument("--lumi-max", type=int, default=10**9,
                   help="only accumulate candidates with lumisection <= this")
    p.add_argument("--max-edm", type=float, default=1e-2,
                   help="skip candidates whose fit did not converge "
                        "(edmval above this); guards the unprotected "
                        "gradient sum against diverged fits (the fake "
                        "run-283453 anomaly was ONE such candidate)")
    p.add_argument("--exclude-from-permodule", type=float, default=0.,
                   help="blacklist modules whose per-module fit "
                        "(classcorr_permodule.pkl in the rundir) has any "
                        "class value with |theta| above this (um); their "
                        "class-param slots are dropped from the group fit")
    p.add_argument("--lorentz-x", action="store_true",
                   help="physics parameterization of the x-drift effects: "
                        "replace {sizeX1 (20), edge-x-mean (16)} by a single "
                        "Lorentz-drift parameter per group with fixed relative "
                        "response weights (w_size1 = 1, w_edge = lorentz-edge-"
                        "weight, clean hits = 0 by the differential-to-"
                        "trajectory convention). Reported in size-1-equivalent "
                        "um; the absolute delta-tan(thetaL) normalisation needs "
                        "a digitizer-level study. edge-x-diff (17) stays free "
                        "for the truncation asymmetry.")
    p.add_argument("--lorentz-edge-weight", type=float, default=-0.6,
                   help="edge-x response relative to size-1 (default -0.6, "
                        "measured from the run-283453 condition shift)")
    args = p.parse_args()

    files = sorted(glob.glob(args.rundir + "/globalcor_*.root"))
    assert files

    # --- catalog: global index -> (parmtype, subdet, layer, detid) --------
    rt = uproot.open(files[0])["runtree"]
    cat = rt.arrays(["parmtype", "rawdetid", "subdet", "layer"], library="np")
    ptypes = cat["parmtype"]
    isclass = (ptypes >= 16) & (ptypes <= 22)
    print(f"catalog: {len(ptypes)} params, {isclass.sum()} class params")

    # optional module blacklist from a previous per-module fit
    badmods = set()
    if args.exclude_from_permodule > 0:
        import pickle as _pkl
        with open(args.rundir.rstrip("/") + "/classcorr_permodule.pkl", "rb") as fpk:
            pm = _pkl.load(fpk)
        for k, t in zip(pm["keys"], pm["theta"]):
            if isinstance(k[0], int) and np.isfinite(t) and \
                    abs(t) * CM2UM > args.exclude_from_permodule:
                badmods.add(k[1])
        print(f"excluding {len(badmods)} modules (|theta| > "
              f"{args.exclude_from_permodule} um in per-module fit): {sorted(badmods)}")
        isclass = isclass & ~np.isin(cat["rawdetid"], list(badmods))

    # group id per global index (-1 = not fitted)
    if args.per_module:
        keys = [(int(t), int(d)) if c else None
                for t, d, c in zip(ptypes, cat["rawdetid"], isclass)]
    else:
        keys = [(int(t), int(s), int(l)) if c else None
                for t, s, l, c in zip(ptypes, cat["subdet"], cat["layer"], isclass)]

    # Response weight of each catalog entry within its group (1 except for
    # the composite Lorentz-drift parameter below): the group parameter g
    # acts on the underlying translation as x_p = w_p * g, so gradients
    # accumulate as w*grad and Hessians as w_i*w_j*H_ij.
    wglobal = np.ones(len(ptypes))
    if args.lorentz_x:
        for i in np.where(isclass)[0]:
            if ptypes[i] == 20:      # sizeX1: reference response
                pass
            elif ptypes[i] == 16:    # edge-x-mean: fixed relative response
                wglobal[i] = args.lorentz_edge_weight
            else:
                continue
            keys[i] = (("LX", int(cat["rawdetid"][i])) if args.per_module
                       else ("LX", int(cat["subdet"][i]), int(cat["layer"][i])))
    if args.with_alignment:
        # pixel alignment, one group per module per dof: translations 0/1
        # (+2 = local z and rotations 3/4/5 with --align-6dof). Subdet
        # codes 0 = BPix, 1 = FPix in the runtree.
        maxpt = 5 if args.align_6dof else 1
        isalign = (ptypes <= maxpt) & \
                  ((cat["subdet"] == 0) | (cat["subdet"] == 1))
        for i in np.where(isalign)[0]:
            keys[i] = ("A", int(ptypes[i]), int(cat["rawdetid"][i]))
    keyset = sorted(set(k for k in keys if k is not None), key=str)
    keyid = {k: i for i, k in enumerate(keyset)}
    ngrp = len(keyset)
    grp = np.array([keyid[k] if k is not None else -1 for k in keys], dtype=np.int64)
    nalign = sum(1 for k in keyset if k[0] == "A")
    print(f"{ngrp} groups ({nalign} pixel-alignment)")

    G = np.zeros(ngrp)
    H = np.zeros((ngrp, ngrp))
    ntouch = np.zeros(ngrp, dtype=np.int64)

    ncand = 0
    nskip_edm = 0
    for fn in files:
        t = uproot.open(fn)["tree"]
        if t.num_entries == 0:
            continue
        arrs = t.arrays(["globalidxv", "gradv", "hesspackedv", "run", "lumi",
                         "edmvalref"], library="np")
        for gi, gr, hp, rn, ls, edm in zip(arrs["globalidxv"], arrs["gradv"],
                                           arrs["hesspackedv"], arrs["run"],
                                           arrs["lumi"], arrs["edmvalref"]):
            if rn < args.run_min or rn > args.run_max:
                continue
            if ls < args.lumi_min or ls > args.lumi_max:
                continue
            if edm > args.max_edm:
                nskip_edm += 1
                continue
            ncand += 1
            gi = np.asarray(gi)
            g = grp[gi]                      # group id per local param (-1 = other)
            pos = np.where(g >= 0)[0]
            if len(pos) == 0:
                continue
            n = len(gi)
            gids = g[pos]
            wv = wglobal[gi[pos]]
            np.add.at(G, gids, wv * gr[pos])
            np.add.at(ntouch, gids, 1)
            # Vectorised symmetric accumulation over the ordered pair mesh:
            # each unordered off-diagonal pair appears in both orientations
            # (adding to H[ga,gb] and H[gb,ga], or twice to H[g,g] when the
            # two locals share a group -- the required factor), diagonal
            # elements once. Packed upper-triangle lookup:
            # idx(i, j>=i) = i*n - i(i-1)/2 + (j-i).
            ii = np.broadcast_to(pos[:, None], (len(pos), len(pos)))
            jj = ii.T
            imin, imax = np.minimum(ii, jj), np.maximum(ii, jj)
            v = hp[imin * n - imin * (imin - 1) // 2 + (imax - imin)] \
                * np.outer(wv, wv)
            gmi = np.broadcast_to(gids[:, None], v.shape)
            np.add.at(H, (gmi, gmi.T), v)

    print(f"candidates: {ncand}  (skipped {nskip_edm} non-converged, edm > {args.max_edm})")

    # Gaussian prior on the alignment deviations (centred on the current
    # alignment, i.e. theta = 0): H += 1/sigma^2 on their diagonal.
    if args.with_alignment:
        sig = args.align_prior_um / CM2UM
        sigrot = args.align_prior_um * 1e-4   # rad; ~same transverse scale @1cm
        for k, i in keyid.items():
            if k[0] == "A":
                H[i, i] += 1. / (sigrot**2 if k[1] >= 3 else sig**2)
        print(f"alignment prior: {args.align_prior_um} um (rot {sigrot:.1e} rad) "
              f"on {nalign} params")

    ok = np.diag(H) > 0
    theta = np.full(ngrp, np.nan)
    err = np.full(ngrp, np.nan)
    Hok = H[np.ix_(ok, ok)]
    if args.per_module:
        # Per-module blocks are frequently rank-deficient (a module whose
        # edge hits all touch ONE side has 100% mean-diff correlation), so
        # solve with the pseudo-inverse and let the error/rank filtering
        # below pick the constrained combinations.
        Hpinv = np.linalg.pinv(Hok, rcond=1e-8)
        theta[ok] = Hpinv @ (-G[ok])
        err[ok] = np.sqrt(np.abs(np.diag(Hpinv)))
    else:
        # pinv: low-stats groups can be rank-deficient (mean/diff degenerate
        # when only one edge side is populated); unconstrained combinations
        # surface as large errors instead of crashing the solve.
        Hpinv = np.linalg.pinv(Hok, rcond=1e-10)
        theta[ok] = Hpinv @ (-G[ok])
        err[ok] = np.sqrt(np.abs(np.diag(Hpinv)))

    if args.per_module:
        # Summarise per (parmtype, subdet-from-detid, BPix-layer): weighted
        # mean of the per-module values, consistency chi2, and spread.
        print(f"\n{'group':>24s} {'nmod':>6s} {'wmean (um)':>11s} {'werr':>6s}"
              f" {'chi2/nmod':>10s} {'RMS (um)':>9s}")
        bysummary = defaultdict(list)
        for k, i in keyid.items():
            if not ok[i] or not np.isfinite(theta[i]) or err[i] * CM2UM > 100:
                continue
            det = k[1]
            sub = (det >> 25) & 0x7   # 1 = PXB, 2 = PXF (rawdetid)
            lay = (det >> 16) & 0xF if sub == 1 else 0
            bysummary[(k[0], sub, lay)].append((theta[i], err[i]))
        for key in sorted(bysummary):
            vals = np.array(bysummary[key])
            th, er = vals[:, 0], vals[:, 1]
            w = 1. / er**2
            wm = np.sum(w * th) / np.sum(w)
            we = 1. / np.sqrt(np.sum(w))
            chi2 = np.sum(w * (th - wm)**2) / max(len(th) - 1, 1)
            sdname = {1: "BPix", 2: "FPix"}.get(key[1], str(key[1]))
            pname = "x-drift[size1-eq]" if key[0] == "LX" else PARMNAMES[key[0]]
            label = f"{pname} {sdname} L{key[2]}"
            print(f"{label:>24s} {len(th):6d} {wm*CM2UM:+11.2f} {we*CM2UM:6.2f}"
                  f" {chi2:10.2f} {np.std(th)*CM2UM:9.1f}")
    else:
        print(f"\n{'group':>32s} {'ntouch':>8s} {'theta (um)':>12s} {'err (um)':>10s}")
        for k, i in keyid.items():
            if not ok[i] or k[0] == "A":
                continue
            pname = "x-drift[size1-eq]" if k[0] == "LX" else PARMNAMES[k[0]]
            sdname = {0: "BPix", 1: "FPix"}.get(k[1], str(k[1]))
            label = f"{pname} {sdname} L{k[2]:+d}"
            sc = PARMSCALE.get(k[0], CM2UM)
            print(f"{label:>32s} {ntouch[i]:8d} {theta[i]*sc:+12.2f} {err[i]*sc:10.2f}")
        if args.with_alignment:
            print("\nfitted pixel-alignment summary (per parmtype, um):")
            for pt, name in ((0, "local-x"), (1, "local-y")):
                sel = [i for k, i in keyid.items()
                       if k[0] == "A" and k[1] == pt and ok[i] and np.isfinite(theta[i])]
                th = theta[sel] * CM2UM
                print(f"  {name}: n={len(th)}  mean={th.mean():+6.2f}  RMS={th.std():6.2f}"
                      f"  max|.|={np.abs(th).max():6.1f}")

    # machine-readable dump for downstream use (apply step / corFile writer)
    import pickle
    tag = ("_withalign" if args.with_alignment else "") + \
          ("_lorentzx" if args.lorentz_x else "")
    outpkl = args.rundir.rstrip("/") + (f"/classcorr_permodule{tag}.pkl" if args.per_module
                                        else f"/classcorr_groups{tag}.pkl")
    with open(outpkl, "wb") as fpk:
        pickle.dump({"keys": keyset, "theta": theta, "err": err,
                     "ntouch": ntouch, "per_module": args.per_module}, fpk)
    print("\nsaved", outpkl)


if __name__ == "__main__":
    main()
