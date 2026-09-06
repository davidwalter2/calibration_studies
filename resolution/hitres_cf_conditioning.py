#!/usr/bin/env python3
"""Can the per-hit resolution classes be SEPARATED from data by the CF?

The offline CF writes the fitted momentum error as an exact linear form,
delta(q/p) = sum_b w_b n_b, so the hit contribution to Var(q/p) is
sum_b v_b with v_b = w_b^T V_b w_b (`resinfvarv`, exported per block and
summing to refCov(0,0) exactly). Grouping the hit blocks by CLASS c and
letting theta_c be the true-to-assigned variance ratio of that class,

    sigma^2(track; theta) = sum_c theta_c V_c(track) + V_other(track)

with V_c(track) the class's share of that track's hit variance. Every
ingredient is a per-track number the refit already exports, so the question
"is theta identifiable from data" is a question about the DESIGN MATRIX of
the f_c = V_c / sigma^2, and it can be answered before writing any fit.

What matters is not that the classes exist but that their per-track shares
VARY and are not collinear: a class whose share is a fixed multiple of
another's contributes one direction, not two. The eigen-spectrum of

    M_cc' = sum_tracks f_c f_c'          (per-track weight 1/Var(z^2) = 1/2)

says which combinations are measurable and which are not, and its inverse
gives the reachable sigma(theta_c) for the statistics in hand.

This is a WIDTH-level statement. The shape parameters carry more information
than the variance does, so these numbers are a conservative bound on what a
full CF likelihood could reach -- but a direction that is degenerate here is
degenerate there too, because it is degenerate in the CLASS COMPOSITION.

usage:
  python hitres_cf_conditioning.py --tag mugun_lowpt_cf [--nfiles 0]
"""
import argparse
import glob
import os

import numpy as np
import uproot
import prodfiles

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
BR = ["reshitidx", "reseigidx", "resinfvarv", "refCov", "nValidHits",
      "clusterSizeX", "clusterChargeBin", "hitUProj", "hitDetId",
      "genPt", "genEta", "trackPt"]


def class_of(subdet, N, uproj, qbin, isy):
    """The class labels, chosen from what the pull study found to move:
    strips split on the cluster width and on uProj (the CPE's own variable,
    across which the core runs +12 % to -13 %); pixels split on the template
    charge bin (core 0.52 -> 1.17 across qbin 0-3)."""
    if subdet <= 2:
        return f"pix_{'y' if isy else 'x'}_q{min(int(qbin), 3)}"
    n = min(int(N), 5)
    u = "lo" if uproj < 0.25 else "hi"
    return f"str_N{n}{'' if n < 5 else 'p'}_{u}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", default=["mugun_lowpt_cf"],
                   help="one or more productions, pooled into ONE design -- the "
                        "realistic case is a joint J/psi + Z fit, where the "
                        "hit/material balance differs strongly between samples")
    p.add_argument("--subdir", default="resolution_trackres")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--minshare", type=float, default=0.002,
                   help="drop classes below this mean share (they carry no information)")
    args = p.parse_args()

    fs = []
    for tg in args.tags:
        fs += prodfiles.resolve(
            f"{CEPH}/{args.subdir}_{tg}/task_*/globalcor_resclosure_*.root",
            args.nfiles)
    if not fs:
        raise SystemExit("no complete files")

    rt = uproot.open(fs[0])["runtree"].arrays(["iidx", "parmtype"], library="np")
    pm = {int(i): int(t) for i, t in zip(rt["iidx"], rt["parmtype"])}

    rows, others, sigs, pts, etas = [], [], [], [], []
    names = {}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(BR, library="np")
        for i in range(len(a["reshitidx"])):
            c00 = a["refCov"][i][0]
            if c00 <= 0:
                continue
            hi = np.asarray(a["reshitidx"][i])
            gi = np.asarray(a["reseigidx"][i])
            vb = np.asarray(a["resinfvarv"][i], float)
            if hi.size == 0:
                continue
            fam = np.array([pm.get(int(g), -1) for g in gi])
            det = np.asarray(a["hitDetId"][i]).astype(np.uint32)
            sd = (det >> 25) & 0x7
            N = np.asarray(a["clusterSizeX"][i])
            up = np.asarray(a["hitUProj"][i])
            qb = np.asarray(a["clusterChargeBin"][i])
            acc, oth = {}, 0.0
            bad = False
            for j in range(len(fam)):
                if fam[j] in (8, 9):
                    h = hi[j]
                    if h < 0 or h >= len(sd):
                        bad = True
                        break
                    c = class_of(sd[h], N[h], up[h], qb[h], fam[j] == 9)
                    acc[c] = acc.get(c, 0.0) + vb[j]
                    names.setdefault(c, 0)
                    names[c] += 1
                elif fam[j] == 10:
                    acc["MS"] = acc.get("MS", 0.0) + vb[j]
                    names.setdefault("MS", 0); names["MS"] += 1
                elif fam[j] == 11:
                    acc["ioni"] = acc.get("ioni", 0.0) + vb[j]
                    names.setdefault("ioni", 0); names["ioni"] += 1
                else:
                    oth += vb[j]
            if bad:
                continue
            rows.append(acc)
            others.append(oth)
            sigs.append(c00)
            pts.append(float(a["genPt"][i]))
            etas.append(float(a["genEta"][i]))

    cls = sorted(names)
    F = np.zeros((len(rows), len(cls)))
    for r, acc in enumerate(rows):
        for c, v in acc.items():
            F[r, cls.index(c)] = v
    sig2 = np.asarray(sigs)
    f = F / sig2[:, None]                      # per-track fractional share
    keep = f.mean(axis=0) > args.minshare
    dropped = [c for c, k in zip(cls, keep) if not k]
    cls = [c for c, k in zip(cls, keep) if k]
    f = f[:, keep]
    ntrk = f.shape[0]

    print(f"{'+'.join(args.tags)}: {len(fs)} files, {ntrk} tracks, "
          f"{len(cls)} parameters kept (hit classes + MS + ioni)")
    if dropped:
        print(f"  dropped (mean share < {args.minshare}): {', '.join(dropped)}")
    ihit = [j for j, c in enumerate(cls) if c not in ("MS", "ioni")]
    print(f"  share of Var(q/p):  hit classes {f[:, ihit].sum(axis=1).mean():.4f}"
          f"   MS+ioni {f.sum(axis=1).mean() - f[:, ihit].sum(axis=1).mean():.4f}"
          f"   unmodelled {1 - f.sum(axis=1).mean():.4f}\n")
    print(f"  {'class':<16} {'mean share':>11} {'rms/mean':>9}  "
          f"{'corr with the largest':>22}")
    order = np.argsort(-f.mean(axis=0))
    big = order[0]
    for j in order:
        m = f[:, j].mean()
        r = f[:, j].std() / m if m > 0 else np.nan
        cc = np.corrcoef(f[:, j], f[:, big])[0, 1]
        print(f"  {cls[j]:<16} {m:11.4f} {r:9.3f}  {cc:22.3f}")

    # design matrix, per-track weight 1/Var(z^2) = 1/2 for a width measurement
    M = f.T @ f / 2.0
    ev, evec = np.linalg.eigh(M)
    cov = np.linalg.pinv(M)
    err = np.sqrt(np.diag(cov))
    print(f"\n  eigenvalues of the design matrix (largest -> smallest):")
    print("   ", "  ".join(f"{e:.3e}" for e in ev[::-1]))
    print(f"  condition number  {ev[-1]/max(ev[0], 1e-300):.3e}")
    mat = [j for j, c in enumerate(cls) if c in ("MS", "ioni")]
    hit = [j for j, c in enumerate(cls) if c not in ("MS", "ioni")]
    if mat and hit:
        Mh = M[np.ix_(hit, hit)]
        errfix = np.sqrt(np.diag(np.linalg.pinv(Mh)))
    else:
        errfix = None
    print(f"\n  reachable sigma(theta) with these {ntrk} tracks "
          f"(theta = true/assigned VARIANCE ratio):")
    print(f"    {'parameter':<16} {'MS+ioni floated':>16} {'MS+ioni fixed':>15}"
          f" {'inflation':>10}   {'-> on the WIDTH ratio':>22}")
    for j in order:
        if errfix is not None and j in hit:
            ef = errfix[hit.index(j)]
            infl = f"{err[j]/ef:10.2f}"
        else:
            ef, infl = float("nan"), " " * 10
        print(f"    {cls[j]:<16} {err[j]:16.4f} {ef:15.4f} {infl}"
              f"   {0.5*err[j]:22.4f}")
    print("    'inflation' is the price of separating the hit classes from the")
    print("    material scales; >> 1 means the two are degenerate in this sample")
    print("    and need a second sample at a different momentum to break.")
    print(f"\n  weakest direction (smallest eigenvalue {ev[0]:.3e}):")
    v = evec[:, 0]
    for j in np.argsort(-np.abs(v)):
        if abs(v[j]) > 0.15:
            print(f"    {v[j]:+.3f} * {cls[j]}")
    print("\n  A direction with a small eigenvalue is one the CLASS COMPOSITION")
    print("  cannot resolve: those classes always appear on a track in a fixed")
    print("  ratio, so only their sum is measurable and they must be merged.")


if __name__ == "__main__":
    main()
