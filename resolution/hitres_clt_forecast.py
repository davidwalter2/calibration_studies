#!/usr/bin/env python3
"""Does the per-hit SHAPE survive the sum into the momentum?

The offline CF writes delta(q/p) = sum_b w_b n_b over ~15-20 hits. A sum of
that many independent bounded variables is far more Gaussian than any of its
parts, so before replacing the CF's Gaussian hit term with the measured
per-class densities it is worth asking how much of the difference survives.

The forecast uses only measured inputs, no model:
  * per-track hit WEIGHTS v_b and their classes, from the nominal-fit
    production (resinfvarv + reshitidx + the class variables);
  * per-class standardised RESIDUALS n_b, resampled from the gen-anchored
    production's actual pulls -- so the true shape, tails included.
Two arms per track: n_b drawn from the measured class density, or from a
Gaussian of the same variance. Everything else is identical, so the
difference IS the effect of the shape.

Reported at two levels: the hit-only part of z, and the full z once the
material blocks are added (they carry ~82 % of Var(q/p), which dilutes any
hit-shape effect further).

usage: python hitres_clt_forecast.py
"""
import argparse
import glob
import os

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"


def class_of(subdet, N, uproj, qbin, isy):
    if subdet <= 2:
        return f"pix_{'y' if isy else 'x'}_q{min(int(qbin), 3)}"
    n = min(int(N), 5)
    return f"str_N{n}_{'lo' if uproj < 0.25 else 'hi'}"


def shape_bank(nfiles):
    """Measured standardised residual per class, from the gen-anchored arm."""
    br = ["dxrecsim", "dxerr", "dyrecsim", "dyerr", "hitDetId", "hitUProj",
          "clusterSizeX", "clusterChargeBin"]
    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/hitres3_mugun_lowpt/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))][:nfiles]
    cols = {b: [] for b in br}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(br, library="np")
        for b in br:
            cols[b].append(np.concatenate(a[b]))
    d = {b: np.concatenate(v) for b, v in cols.items()}
    sd = (d["hitDetId"].astype(np.uint32) >> 25) & 0x7
    bank = {}
    for isy, (rk, ek) in enumerate((("dxrecsim", "dxerr"), ("dyrecsim", "dyerr"))):
        m = (d[rk] > -98) & (d[ek] > 0)
        if isy:
            m &= sd <= 2
        pull = np.where(m, d[rk] / np.where(d[ek] > 0, d[ek], 1), np.nan)
        for i in np.where(m)[0]:
            c = class_of(sd[i], d["clusterSizeX"][i], d["hitUProj"][i],
                         d["clusterChargeBin"][i], bool(isy))
            bank.setdefault(c, []).append(pull[i])
    out = {}
    for c, v in bank.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size < 500:
            continue
        v = v - np.median(v)
        out[c] = v / v.std()            # standardised to unit VARIANCE
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nfiles-shape", type=int, default=25)
    p.add_argument("--nfiles-weight", type=int, default=25)
    p.add_argument("--ntoy", type=int, default=40)
    args = p.parse_args()
    rng = np.random.default_rng(20260830)

    bank = shape_bank(args.nfiles_shape)
    print("measured class shapes used (standardised to unit variance):")
    for c in sorted(bank):
        v = bank[c]
        def w(q):
            a, b = np.percentile(v, [50 - 50 * q, 50 + 50 * q]); return 0.5 * (b - a)
        print(f"  {c:<14} n={v.size:7d}  R90 {w(.90)/w(.6827):5.3f}  R99 {w(.99)/w(.6827):5.3f}")
    print("  (Gaussian 1.645 / 2.576)\n")

    br = ["reshitidx", "reseigidx", "resinfvarv", "refCov", "hitDetId",
          "hitUProj", "clusterSizeX", "clusterChargeBin"]
    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/resolution_trackres_mugun_lowpt_cf/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))][:args.nfiles_weight]
    rt = uproot.open(fs[0])["runtree"].arrays(["iidx", "parmtype"], library="np")
    pm = {int(i): int(t) for i, t in zip(rt["iidx"], rt["parmtype"])}

    zh_m, zh_g, zf_m, zf_g = [], [], [], []
    nhit = []
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(br, library="np")
        for i in range(len(a["reshitidx"])):
            c00 = a["refCov"][i][0]
            if c00 <= 0:
                continue
            hi = np.asarray(a["reshitidx"][i]); gi = np.asarray(a["reseigidx"][i])
            vb = np.asarray(a["resinfvarv"][i], float)
            fam = np.array([pm.get(int(g), -1) for g in gi])
            det = np.asarray(a["hitDetId"][i]).astype(np.uint32); sd = (det >> 25) & 0x7
            N = np.asarray(a["clusterSizeX"][i]); up = np.asarray(a["hitUProj"][i])
            qb = np.asarray(a["clusterChargeBin"][i])
            amp, cls, vother = [], [], 0.0
            bad = False
            for j in range(len(fam)):
                if fam[j] in (8, 9):
                    h = hi[j]
                    if h < 0 or h >= len(sd) or vb[j] <= 0:
                        bad = True; break
                    c = class_of(sd[h], N[h], up[h], qb[h], fam[j] == 9)
                    if c not in bank:
                        vother += vb[j]; continue
                    amp.append(np.sqrt(vb[j] / c00)); cls.append(c)
                else:
                    vother += vb[j]
            if bad or not amp:
                continue
            amp = np.asarray(amp)
            nhit.append(len(amp))
            for _ in range(args.ntoy):
                nm = np.array([bank[c][rng.integers(0, bank[c].size)] for c in cls])
                ng = rng.standard_normal(len(amp))
                hm = float(amp @ nm); hg = float(amp @ ng)
                zh_m.append(hm); zh_g.append(hg)
                rest = np.sqrt(max(vother, 0.0) / c00) * rng.standard_normal()
                zf_m.append(hm + rest); zf_g.append(hg + rest)

    def rep(name, x):
        x = np.asarray(x)
        def w(q):
            a, b = np.percentile(x, [50 - 50 * q, 50 + 50 * q]); return 0.5 * (b - a)
        w68 = w(.6827)
        return (f"  {name:<28} w68 {w68:.4f}  R90 {w(.90)/w68:6.4f}  "
                f"R99 {w(.99)/w68:6.4f}  >3core {100*np.mean(np.abs(x)>3*w68):6.3f}%")

    print(f"tracks {len(nhit)}, mean hit blocks per track {np.mean(nhit):.1f}, "
          f"{args.ntoy} toys each")
    print("\nHIT-ONLY part of z:")
    print(rep("measured class shapes", zh_m))
    print(rep("Gaussian hits", zh_g))
    print("\nFULL z (material blocks added as Gaussian, identical in both arms):")
    print(rep("measured class shapes", zf_m))
    print(rep("Gaussian hits", zf_g))
    zm, zg = np.asarray(zf_m), np.asarray(zf_g)
    def w68(x):
        a, b = np.percentile(x, [15.865, 84.135]); return 0.5 * (b - a)
    print(f"\n  shape effect on the FULL z:  d(w68)/w68 = "
          f"{w68(zm)/w68(zg)-1:+.5f}   d(R99) = "
          f"{(np.percentile(zm,99.5)-np.percentile(zm,0.5))/(np.percentile(zg,99.5)-np.percentile(zg,0.5))-1:+.5f}")


if __name__ == "__main__":
    main()
