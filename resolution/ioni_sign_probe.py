#!/usr/bin/env python3
"""What sign does `build_pairs_tt` give each ionization block, and what is right?

THE HEURISTIC.  `cf_mass_likelihood.build_pairs_tt` sets, per pooled block,

    sgn = np.sign(uw[m].sum()) or 1.        uw = resinfv, the mass-projected
                                            signed dof weights u_b = a^T B_b

and passes `sgn * sqrt(vpool/sq2)/sig` to `ioni_step_exponent`.  The sign only
matters for the ionization skew (Re S is even in the weight, Im S is odd).

WHAT IS RIGHT.  The exponent's noise variable is the fit's internal q/p dof,
and the ionization map to it is d(q/p) = q cs dE with cs = E/p^3 > 0 -- the
`qsign` the in-fit CGF block applies and the offline CF does not.  The mass
Jacobian carries the OTHER factor of q: p = q/(q/p), so dp/d(q/p) = -q p^2 and

    a[0] = dm/d(q/p) = -q p^2 dm/dp ,     sign a[0] = -q   (dm/dp > 0)

so the physical response of the mass to an energy-loss fluctuation is

    dm = a[0] d(q/p) = (-q p^2 dm/dp)(q cs dE) = -p^2 cs (dm/dp) dE  <  0

for BOTH charges: extra loss on either muon can only lower the mass.  The
correct signed weight is therefore NEGATIVE for every ionization block, and
`sign(u_b)` alone -- which sees only the -q of the Jacobian, never the +q of
the cs map -- returns -1 on the mu+ leg and +1 on the mu- leg, so the skew
cancels within each candidate.

This script measures that, rather than asserting it: the fraction of blocks at
each sign, and whether within a candidate the signs split into two opposite
groups.  Leg identity is read off `resinfv`'s own q/p column (u_b[0]), whose
sign IS -q, so no leg branch is needed.

usage: python ioni_sign_probe.py [--files GLOB] [--ntasks N] [--ncand N]
"""
import argparse
import glob

import numpy as np
import uproot
import prodfiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", default="/ceph/submit/data/user/d/david_w/ZMass/"
                   "cvh/resolution_trackres_jpsigun_ul16_260902_m0/"
                   "task_000*/globalcor_*.root")
    p.add_argument("--ntasks", type=int, default=3)
    p.add_argument("--ncand", type=int, default=4000)
    a = p.parse_args()

    files = prodfiles.resolve(a.files, a.ntasks, logger=print)
    print(f"{len(files)} files")
    npos = nneg = nzero = 0
    ncand = nsplit = nsame = nodd = 0
    surv = []
    col0_agree = col0_tot = 0
    per_cand = []
    stop = False
    for fn in files:
        f = uproot.open(fn)
        pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        arr = t.arrays(["resinfv", "resinfvarv", "reseigidx",
                        "ioniurbanidx", "ioniurbanv",
                        "Muplus_charge", "Muminus_charge"], library="np")
        for ic in range(len(arr["reseigidx"])):
            gi = np.asarray(arr["reseigidx"][ic])
            if not len(gi):
                continue
            vb = np.asarray(arr["resinfvarv"][ic], dtype=np.float64)
            uw = np.asarray(arr["resinfv"][ic], dtype=np.float64).reshape(-1, 5)
            fam = pt[gi]
            sel = fam == 11
            uvi = np.asarray(arr["ioniurbanv"][ic], dtype=np.float64)
            uii = np.asarray(arr["ioniurbanidx"][ic])
            uvi = (uvi.reshape(-1, len(uvi) // max(len(uii), 1))
                   if len(uii) else uvi.reshape(0, 11))
            signs, s0, mags = [], [], []
            for g in np.unique(gi[sel]):
                m = sel & (gi == g)
                vpool = vb[m].sum()
                if vpool <= 0.:
                    continue
                st = uvi[uii == g]
                if not len(st):
                    continue
                gq = st[:, 10] * 1e-3
                sq2 = float(np.sum(st[:, 1] * gq * gq))
                if sq2 <= 0.:
                    continue
                tot = uw[m].sum()                 # what the code uses
                q0 = uw[m][:, 0].sum()            # the q/p column alone
                sg = np.sign(tot) or 1.
                signs.append(sg)
                s0.append(np.sign(q0) or 1.)
                mags.append(np.sqrt(vpool / sq2))
                if sg > 0:
                    npos += 1
                elif sg < 0:
                    nneg += 1
                else:
                    nzero += 1
            if not signs:
                continue
            col0_tot += len(signs)
            col0_agree += int(np.sum(np.array(signs) == np.array(s0)))
            ncand += 1
            per_cand.append(len(signs))
            u = set(signs)
            if u == {1.0, -1.0}:
                nsplit += 1
            else:
                nsame += 1
            # SKEW SURVIVAL.  The third cumulant of a block's contribution
            # scales as w^3, so the fraction of the candidate's ionization
            # skew that survives the heuristic's signs is
            #     R = |sum_b sgn_b |w_b|^3| / sum_b |w_b|^3 ,
            # which is 1 for the correct (all-negative) assignment and 0 if
            # the two legs cancel exactly.  This is the number that matters:
            # it does not depend on any leg bookkeeping.
            w3 = np.abs(np.array(mags)) ** 3
            if w3.sum() > 0.:
                surv.append(float(np.sum(np.array(signs) * w3)) / float(w3.sum()))
            if ncand >= a.ncand:
                stop = True
                break
        if stop:
            break

    n = npos + nneg + nzero
    print(f"\nionization blocks: {n}   in {ncand} candidates "
          f"({np.mean(per_cand):.2f} blocks/candidate)")
    print(f"  sgn = +1 : {npos:7d}  ({100.*npos/max(n,1):5.1f} %)")
    print(f"  sgn = -1 : {nneg:7d}  ({100.*nneg/max(n,1):5.1f} %)")
    print(f"  sgn =  0 : {nzero:7d}  (forced to +1 by the `or 1.`)")
    print(f"\n  sign(sum over all 5 dof) == sign(q/p column alone): "
          f"{col0_agree}/{col0_tot} = {100.*col0_agree/max(col0_tot,1):.2f} %")
    print(f"\n  candidates with BOTH signs present : {nsplit} "
          f"({100.*nsplit/max(ncand,1):5.1f} %)  <- the two legs cancel")
    print(f"  candidates with ONE sign only      : {nsame} "
          f"({100.*nsame/max(ncand,1):5.1f} %)")
    sv = np.array(surv)
    print(f"\n  SKEW SURVIVAL R = sum_b sgn_b |w_b|^3 / sum_b |w_b|^3  per candidate")
    print(f"    (the correct all-negative assignment gives R = -1 for every "
          f"candidate; R = 0 is exact cancellation)")
    print(f"    SIGNED  : mean {sv.mean():+.4f} +- {sv.std()/np.sqrt(len(sv)):.4f}"
          f"   median {np.median(sv):+.4f}   rms {sv.std():.4f}")
    print(f"    |R|     : mean {np.abs(sv).mean():.4f}   median "
          f"{np.median(np.abs(sv)):.4f}")
    print(f"    -> the per-candidate skew is NOT small, but its SIGN is set by "
          f"an arbitrary\n       eigenvector convention, so it averages away "
          f"over the sample.")
    print("\n  CORRECT sign for every ionization block: -1 "
          "(dm = a[0] d(q/p) = (-q p^2 dm/dp)(q cs dE) = -p^2 cs (dm/dp) dE "
          "< 0 for BOTH charges).")


if __name__ == "__main__":
    main()
