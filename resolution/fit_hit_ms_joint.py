"""Leakage test: hit-resolution scales (parmtype 8/9) and MS group scales
(parmtype 10) fitted in one system with the same tail-robust ECF-convention
statistic.

Hypothesis under test (NOTES 2026-07-29): the large MS deviations of the
ACTIVE groups are not multiple scattering but absorption of a hit-resolution
mismatch, because those MS parameters sit on exactly the surfaces that carry
the hits, where MS has almost no lever arm on the residual.

Two observables:
1. **Per-subdetector hit-resolution scales** (PXB/PXF/TIB/TID/TOB/TEC x and
   y, from the DetId in the runtree). Never measured at this granularity
   before -- the cf0 ECF fit tied all modules into one hit-x / hit-y scale.
2. **Correlation** of that pattern with the MS active-group deviations
   aggregated per subdetector. If the subdetectors whose hit resolution is
   most mis-modelled are the ones whose active-layer MS parameters deviate
   most, leakage is established.

Note on the model: the block-diagonal shrinkage response
s_b = 1 + h_b (sum_p f_bp e^{k_p} - 1) means hit blocks and MS blocks do not
couple THROUGH THE MODEL -- so this is deliberately a comparison of two
independently determined patterns, not a fit in which one absorbs the other.
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools
from fit_ms_material import group_names, model_F

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NEIG = 5
NLAM = 2 * NEIG
# runtree carries subdet/layer directly (iidx is NOT the raw DetId).
# Mapping verified from the module counts per subdet code:
# 768 PXB / 672 PXF / 2724 TIB / 5208 TOB / 816 TID / 6400 TEC, and
# subdet 3 shows 6 layers (TOB) not 4 (TIB).
SUBDET = {0: "PXB", 1: "PXF", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}

# Hit classes at LAYER granularity where the layers map 1:1 onto the MS
# active groups (PXB L1-3, TIB L1-4, TOB L1-6); PXF/TID/TEC lumped (their
# MS groups are by disk/ring, which `layer` does not reproduce). This turns
# the leakage correlation from 3 points (subdetector) into 13 (layer).
HITCLASS = ([f"PXB_L{i}" for i in (1, 2, 3)] + ["PXF"] +
            [f"TIB_L{i}" for i in (1, 2, 3, 4)] +
            [f"TOB_L{i}" for i in (1, 2, 3, 4, 5, 6)] + ["TID", "TEC"])
NHC = len(HITCLASS)
# MS active group <-> hit class pairing for the correlation test
MSPAIR = {f"PXB_L{i}": f"bpix_active_L{i}" for i in (1, 2, 3)}
MSPAIR.update({f"TIB_L{i}": f"tib_active_L{i}" for i in (1, 2, 3, 4)})
MSPAIR.update({f"TOB_L{i}": f"tob_active_L{i}" for i in (1, 2, 3, 4, 5, 6)})


def hitclass(sd, layer):
    if sd == 0 and 1 <= layer <= 3:
        return layer - 1
    if sd == 1:
        return 3
    if sd == 2 and 1 <= layer <= 4:
        return 3 + layer
    if sd == 3 and 1 <= layer <= 6:
        return 7 + layer
    if sd == 4:
        return 14
    if sd == 5:
        return 15
    return -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_eig_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=12)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/hit_ms_joint.npz"))
    p.add_argument("--groups-file",
                   default="/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
                           "Analysis/HitAnalyzer/data/materialGroups50.txt")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probe", type=float, default=1.0)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


def extract(files, args):
    """Blocks of parmtype 8/9 (hit, one-hot on subdet x/y) and 10 (MS, group
    fractions), in a common parameter basis."""
    pt = iidx = None
    z2s, hs, lams, attrs = [], [], [], []
    kinds = []
    npar = None
    nhit = nms = 0
    for fn in files[:args.ntasks]:
        f = uproot.open(fn)
        if "tree" not in f:
            continue
        if pt is None:
            rt = f["runtree"]
            pt = rt["parmtype"].array(library="np")
            sdet = rt["subdet"].array(library="np").astype(np.int64)
            slay = rt["layer"].array(library="np").astype(np.int64)
            ngroups = int(np.sum(pt == 15))
            # parameter basis: 12 hit classes (6 subdet x {x,y}) + ngroups MS
            npar = 2 * NHC + ngroups
            logger.info(f"{ngroups} MS groups + {2*NHC} hit slots = {npar} params; "
                        f"subdet codes: {sorted(set(sdet[np.isin(pt,[8,9])].tolist()))}")
        t = f["tree"]
        a = t.arrays(["globalidxv", "gradchisqv", "gradllv",
                      "reseigidx", "reseigv", "msmoliidx", "msmoliv"], library="np")
        for ic in range(len(a["globalidxv"])):
            gi = np.asarray(a["globalidxv"][ic])
            tp = pt[gi]
            sel = np.where(np.isin(tp, [8, 9, 10]))[0]
            if not len(sel):
                continue
            nu = np.asarray(a["gradllv"][ic], dtype=np.float64)
            q = -np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            eidx = np.asarray(a["reseigidx"][ic])
            ev = np.asarray(a["reseigv"][ic], dtype=np.float64).reshape(-1, NEIG)
            uidx = np.asarray(a["msmoliidx"][ic])
            uv = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            stride = len(uv) // max(len(uidx), 1)
            uv = uv.reshape(-1, stride)
            for j in sel:
                h = nu[j]
                if not (0. < h < 1.) or q[j] < 0.:
                    continue
                lam = ev[eidx == gi[j]].ravel()
                lam = lam[lam > 0.][:NLAM]
                if not len(lam):
                    continue
                att = np.zeros(npar)
                if tp[j] == 10:
                    steps = uv[uidx == gi[j]]
                    if not len(steps):
                        continue
                    thp2 = steps[:, 5]
                    grp = steps[:, 7].astype(int)
                    tot = thp2.sum()
                    if tot <= 0.:
                        continue
                    ok = (grp >= 0) & (grp < npar - 2 * NHC)
                    np.add.at(att, 2 * NHC + grp[ok], thp2[ok])
                    att /= tot
                    nms += 1
                    kinds.append(1)
                else:
                    hc = hitclass(int(sdet[gi[j]]), int(slay[gi[j]]))
                    if hc < 0:
                        continue
                    att[hc * 2 + (1 if tp[j] == 9 else 0)] = 1.
                    nhit += 1
                    kinds.append(0)
                lamn = np.zeros(NLAM)
                lamn[:len(lam)] = lam / lam.sum()
                z2s.append(q[j] / h)
                hs.append(h)
                lams.append(lamn)
                attrs.append(att)
        logger.info(f"{fn.split('/')[-2]}: {nhit} hit blocks, {nms} MS blocks")
    gn = group_names(args.groups_file)
    labels = [f"hit_{c}_{xy}" for c in HITCLASS for xy in ("x", "y")]
    labels += [f"MS_{gn.get(g, f'grp{g}')}" for g in range(npar - 2 * NHC)]
    np.savez_compressed(args.cache, z2=np.array(z2s), h=np.array(hs),
                        lam=np.array(lams), frac=np.array(attrs),
                        kind=np.array(kinds), labels=np.array(labels))
    logger.info(f"wrote {args.cache}: {nhit} hit + {nms} MS blocks, {npar} params")


def fit(args, outdir):
    d = np.load(args.cache)
    z2, h, lam, W = d["z2"], d["h"], d["lam"], d["frac"]
    labels = [str(x) for x in d["labels"]]
    npar = W.shape[1]
    u = args.probe
    F = np.exp(-u * z2)
    fmean = W.mean(axis=0)
    active = fmean > 1e-4
    logger.info(f"{len(z2)} blocks; {int(active.sum())}/{npar} params active")

    k = np.zeros(npar)
    for it in range(40):
        r = F - model_F(u, k, lam, h, W)
        E = W.T @ r
        J = np.zeros((npar, npar))
        eps = 1e-3
        base = model_F(u, k, lam, h, W)
        for g in np.where(active)[0]:
            kp = k.copy(); kp[g] += eps
            J[:, g] = W.T @ (-(model_F(u, kp, lam, h, W) - base) / eps)
        Ja = J[np.ix_(active, active)]
        ridge = 1e-6 * np.max(np.abs(np.diag(Ja)))
        dk = np.zeros(npar)
        dk[active] = np.clip(np.linalg.solve(Ja + ridge * np.eye(int(active.sum())),
                                             -E[active]), -0.5, 0.5)
        k += dk
        k = np.clip(k, -5., 5.)
        if np.max(np.abs(dk)) < 1e-5:
            break
    resid = F - model_F(u, k, lam, h, W)
    Sm = (W[:, active] * (resid ** 2)[:, None]).T @ W[:, active]
    Jinv = np.linalg.inv(Ja + ridge * np.eye(int(active.sum())))
    cov = Jinv @ Sm @ Jinv.T
    err = np.zeros(npar); err[active] = np.sqrt(np.diag(cov))

    print(f"\n=== hit-resolution scales, layer granularity (u = {u}) ===")
    print(f"{'parameter':16s} {'k':>8s} {'err':>7s} {'pull':>6s}")
    for i in range(2 * NHC):
        if active[i] and err[i] > 0:
            print(f"{labels[i]:16s} {k[i]:8.3f} {err[i]:7.3f} {k[i]/max(err[i],1e-9):6.1f}")
    print(f"\n=== leakage test: per-layer hit scale vs SAME-layer MS active scale ===")
    print(f"{'layer':10s} {'hit-x k':>9s} {'MS k':>9s} {'MS err':>8s}")
    idx = {l: i for i, l in enumerate(labels)}
    xs, ys = [], []
    for hc, ms in MSPAIR.items():
        ih, im = idx.get(f"hit_{hc}_x"), idx.get(f"MS_{ms}")
        if ih is None or im is None or not (active[ih] and active[im]):
            continue
        if not (0 < err[im] < 3.):
            continue
        print(f"{hc:10s} {k[ih]:9.3f} {k[im]:9.3f} {err[im]:8.3f}")
        xs.append(k[ih]); ys.append(k[im])
    if len(xs) > 3:
        r = np.corrcoef(xs, ys)[0, 1]
        print(f"\ncorrelation(hit-x, same-layer MS) = {r:+.2f}  (n={len(xs)} layers)")
        print("  -> strongly positive/negative correlation would establish leakage;"
              "\n     ~0 means the two effects are independent.")
    np.savez(os.path.join(outdir, "hit_ms_joint_results.npz"),
             k=k, err=err, active=active, labels=np.array(labels))
    output_tools.write_logfile(outdir, "hit_ms_joint", args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))


def main():
    args = parse_args()
    if args.extract:
        files = sorted(glob.glob(args.input))
        if not files:
            sys.exit(f"no files match {args.input}")
        extract(files, args)
    if args.fit:
        today = datetime.date.today().strftime("%y%m%d")
        outdir = output_tools.make_plot_dir(
            args.outpath or os.path.expanduser(
                f"~/public_html/calibration_studies/{today}_hit_ms_joint/"))
        fit(args, outdir)


if __name__ == "__main__":
    main()
