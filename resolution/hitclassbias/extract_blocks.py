#!/usr/bin/env python3
"""Per-TRACK and per-BLOCK table from a `resolution_trackres_*` production.

Per track: the standardized momentum error `z = (qop_ref - qop_gen)/sigma`
(the same definition and the same three cuts as `cf_track_resolution.py`, so
the sample is the one whose per-eta odd moments are the target), plus
`sigma`, `eta`, `charge`, `vgf`, `pt`, `nvalid`.

Per parmtype-8/9 (HIT) resolution block: the BENDING SENSE `s_b` and the
amplitude `a_b = sqrt(v_b)/sigma`, together with the hit's own observables so
the block can be given the class location measured on the `hitres` arm.

    s_b comes from `resinfbv` row 0 -- B_b = M_b dV_b^{1/2}, so for a rank-1
    hit block its single nonzero entry is exactly s_b sqrt(v_b). `resinfv`
    would be WRONG for the pixel local-y entries, which share the 2-D block
    range with the local-x ones and therefore share `resinfv[b][0]`.

usage: extract_blocks.py --prod resolution_trackres_mugun_ul16_260903x_m0 \
                        --out data/blocks_mugun_ul16_260903x.npz
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hcb_common as H  # noqa: E402

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
COVTOL = 5e-3

BR = ["reseigidx", "reshitidx", "resinfbv", "resinfvarv", "resinfcov",
      "refCov", "refParms", "genParms", "hitDetId", "hitUProj",
      "clusterSizeX", "clusterSizeY", "clusterChargeBin", "clusterOnEdge",
      "hitPitch", "hitThickness", "localdxdz", "localdydz", "dxerr", "dyerr",
      "trackPt", "genPt", "nValidHits", "normalizedChi2"]


def one(fn):
    import uproot
    try:
        f = uproot.open(fn)
        pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        a = t.arrays(BR, library="np")
    except Exception as e:
        return None, f"{fn}: {type(e).__name__} {e}"

    T = {k: [] for k in ("z", "sigma", "eta", "q", "vgf", "pt", "genpt",
                         "nvalid", "chi2n")}
    Bk = {k: [] for k in ("s", "a2", "isy", "sd", "lay", "side", "ring",
                          "uproj", "N", "NY", "qbin", "onedge", "pitch",
                          "thick", "dxdz", "dydz", "err", "og", "detid")}
    nblk = []
    ndrop = 0
    for ic in range(len(a["resinfcov"])):
        qg = a["genParms"][ic][0]
        c00 = float(a["refCov"][ic][0])
        cov = float(a["resinfcov"][ic])
        if qg == 0. or not (c00 > 0.) or abs(cov / c00 - 1.) > COVTOL:
            ndrop += 1
            continue
        sig = np.sqrt(c00)
        gi = np.asarray(a["reseigidx"][ic])
        fam = pt[gi]
        vb = np.asarray(a["resinfvarv"][ic], float)
        Bb = np.asarray(a["resinfbv"][ic], float).reshape(-1, 5, 5)
        hidx = np.asarray(a["reshitidx"][ic])
        hit = (fam == 8) | (fam == 9)
        sel = np.where(hit & (vb > 0.) & (hidx >= 0))[0]
        hd = np.asarray(a["hitDetId"][ic]).astype(np.uint32)
        sel = sel[hidx[sel] < len(hd)]
        if len(sel) == 0:
            ndrop += 1
            continue
        row0 = Bb[sel, 0, :]
        # the block's own dof: the largest |B_0j|. For a rank-1 hit block the
        # others are exactly zero, so this is the single entry and its sign is
        # the bending sense.
        j = np.abs(row0).argmax(1)
        s = np.sign(row0[np.arange(len(sel)), j])
        s[s == 0] = 1.
        hh = hidx[sel]
        isy = (fam[sel] == 9).astype(np.int8)
        T["z"].append((a["refParms"][ic][0] - qg) / sig)
        T["sigma"].append(sig)
        T["eta"].append(-np.log(np.tan((np.pi / 2. - a["refParms"][ic][1]) / 2.)))
        T["q"].append(np.sign(qg))
        T["vgf"].append(vb[hit].sum() / c00)
        T["pt"].append(float(a["trackPt"][ic]))
        T["genpt"].append(float(a["genPt"][ic]))
        T["nvalid"].append(int(a["nValidHits"][ic]))
        T["chi2n"].append(float(a["normalizedChi2"][ic]))
        nblk.append(len(sel))
        Bk["s"].append(s)
        Bk["a2"].append(vb[sel] / c00)
        Bk["isy"].append(isy)
        for k, src in (("uproj", "hitUProj"), ("N", "clusterSizeX"),
                       ("NY", "clusterSizeY"), ("qbin", "clusterChargeBin"),
                       ("onedge", "clusterOnEdge"), ("pitch", "hitPitch"),
                       ("thick", "hitThickness"), ("dxdz", "localdxdz"),
                       ("dydz", "localdydz")):
            v = np.asarray(a[src][ic])
            Bk[k].append(v[hh] if len(v) > hh.max() else np.full(len(hh), -99.))
        ex = np.asarray(a["dxerr"][ic])
        ey = np.asarray(a["dyerr"][ic])
        Bk["err"].append(np.where(isy == 1, ey[hh] if len(ey) > hh.max() else -99.,
                                  ex[hh] if len(ex) > hh.max() else -99.))
        sd, lay, side = H.decode(hd[hh])
        Bk["sd"].append(sd)
        Bk["lay"].append(lay)
        Bk["side"].append(side)
        Bk["ring"].append(H.ring(hd[hh]))
        Bk["og"].append(H.orient_group(hd[hh]))
        Bk["detid"].append(hd[hh].astype(np.int64))
    if not nblk:
        return None, f"{fn}: no tracks"
    out = {("t_" + k): np.asarray(v, float) for k, v in T.items()}
    out["nblk"] = np.asarray(nblk, np.int32)
    for k, v in Bk.items():
        out["b_" + k] = np.concatenate(v)
    out["_ndrop"] = np.array([ndrop])
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--ntasks", type=int, default=0)
    args = ap.parse_args()
    import glob as _g

    import prodfiles
    stem = ("globalcor_resclosure"
            if _g.glob(f"{CEPH}/{args.prod}/task_*/globalcor_resclosure_*.root")
            else "globalcor")
    fs = prodfiles.resolve(f"{CEPH}/{args.prod}/task_*/{stem}_0.root",
                           args.ntasks)
    print(f"{args.prod}: {len(fs)} files", flush=True)
    with Pool(args.nproc) as p:
        res = p.map(one, fs)
    for _, e in [r for r in res if r[1]][:5]:
        print("SKIP", e, flush=True)
    parts = [d for d, e in res if d is not None]
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    ndrop = int(d.pop("_ndrop").sum())
    d["b_cls18"] = H.class18(d["b_sd"], d["b_N"], d["b_uproj"], d["b_qbin"],
                             d["b_isy"] == 1)
    for k in ("b_sd", "b_lay", "b_side", "b_ring", "b_isy"):
        d[k] = d[k].astype(np.int8)
    for k in ("b_N", "b_NY", "b_qbin", "b_onedge"):
        d[k] = d[k].astype(np.int16)
    for k in ("b_s", "b_a2", "b_uproj", "b_pitch", "b_thick", "b_dxdz",
              "b_dydz", "b_err"):
        d[k] = d[k].astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **d)
    print(f"wrote {args.out}: {len(d['t_z'])} tracks (drop {ndrop}), "
          f"{len(d['b_s'])} hit blocks", flush=True)


if __name__ == "__main__":
    main()
