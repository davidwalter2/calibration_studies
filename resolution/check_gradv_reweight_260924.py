#!/usr/bin/env python3
"""Does the momentum dependence of the process noise explain the field
columns of gradv?

The exported field column of gradv is the derivative of the objective at
FIXED covariance.  A real field change moves the fitted momentum, and every
process-noise block moves with it: multiple scattering Q_MS ~ 1/(beta p)^2 and,
in q/p units, the ionization straggling Q_I ~ 1/p^2 for muons in this range.
The derivative that the full refit sees therefore carries the extra term

    dObj/dc |_V  ->  dObj/dc |_V + sum_b dObj/dtheta_b * dln Q_b/dc,
    dln Q_b/dc ~= 2 dln|q/p|/dc,

where theta_b is the log-scale of block b's MS (parmtype 10) or ionization
(parmtype 11) noise, whose exported variance-gradient columns ARE
dObj/dtheta_b.  This script forms that prediction per candidate from the
NOMINAL run (varianceGradFamilies must include 10 and 11) and compares

    FD(c)            central difference of objchisq (objval) under +-delta
    an(c)            gradchisqv (gradv) field column
    an(c) + extra    the prediction above

and also fits the per-family coefficients freely (2 = the scaling above).

    python3 check_gradv_reweight_260924.py [--fdroot DIR] [--mode luFalse]
"""
import argparse
import datetime
import glob
import os
import re

import numpy as np
import uproot

FDROOT = "/work/submit/david_w/ZMass/scratch_lu_260924/fd2"
OUTDIR = ("/work/submit/david_w/ZMass/calibration_studies/resolution/runs/"
          f"{datetime.date.today().strftime('%y%m%d')}_localupdate_grads")

_LINES = []


def log(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


BR = ["run", "lumi", "event", "globalidxv", "gradv", "gradchisqv", "objval",
      "objchisq", "objnullv", "Muplus_refParms", "Muminus_refParms",
      "Muplus_jacRef", "Muminus_jacRef"]


def load(d):
    f = glob.glob(os.path.join(d, "globalcor_*.root"))[0]
    uf = uproot.open(f)
    a = uf["tree"].arrays(BR, library="np")
    a["_key"] = {(int(r), int(l), int(e)): i for i, (r, l, e) in
                 enumerate(zip(a["run"], a["lumi"], a["event"]))}
    a["_pt"] = uf["runtree"]["parmtype"].array(library="np")
    return a


def injected_global(d):
    with open(os.path.join(d, "cmsrun.log"), errors="replace") as fh:
        for line in fh:
            m = re.match(r"injectFieldModes: mode (\d+) \(global (\d+)\)", line)
            if m:
                return int(m.group(2))
    return None


def rel(fd, x):
    return np.sqrt(np.mean((fd - x) ** 2)) / np.sqrt(np.mean(fd ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdroot", default=FDROOT)
    ap.add_argument("--mode", default="luFalse")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(args.fdroot, args.mode)
    nom = load(os.path.join(base, "nom"))
    log(f"fd root {args.fdroot}  mode {args.mode}  nominal candidates {len(nom['run'])}")
    log("rel = rms(FD - model)/rms(FD); coefficient 2 = noise ~ (q/p)^2")

    for pdir in sorted(glob.glob(os.path.join(base, "field_m*_p*"))):
        stem, dtxt = os.path.basename(pdir).rsplit("_p", 1)
        mdir = os.path.join(base, f"{stem}_m{dtxt}")
        if not os.path.isdir(mdir):
            continue
        ap_, am_ = load(pdir), load(mdir)
        g, delta = injected_global(pdir), float(dtxt)
        rows = {k: [] for k in ("fdc", "anc", "s10c", "s11c", "fdo", "ano", "s10o", "s11o")}
        for k, ic in nom["_key"].items():
            if k not in ap_["_key"] or k not in am_["_key"]:
                continue
            ip, im = ap_["_key"][k], am_["_key"][k]
            if not (nom["objnullv"][ic] == ap_["objnullv"][ip] == am_["objnullv"][im]):
                continue
            gi = np.asarray(nom["globalidxv"][ic])
            w = np.where(gi == g)[0]
            if not len(w):
                continue
            c = int(w[0])
            # mean relative q/p response of the two legs (row 0 of the 3 x n jacRef)
            dlq = 0.5 * (nom["Muplus_jacRef"][ic][c] / nom["Muplus_refParms"][ic][0]
                         + nom["Muminus_jacRef"][ic][c] / nom["Muminus_refParms"][ic][0])
            pt = nom["_pt"][gi]
            gc = np.asarray(nom["gradchisqv"][ic], dtype=float)
            go = np.asarray(nom["gradv"][ic], dtype=float)
            rows["fdc"].append((ap_["objchisq"][ip] - am_["objchisq"][im]) / (2 * delta))
            rows["fdo"].append((ap_["objval"][ip] - am_["objval"][im]) / (2 * delta))
            rows["anc"].append(gc[c])
            rows["ano"].append(go[c])
            rows["s10c"].append(dlq * gc[pt == 10].sum())
            rows["s11c"].append(dlq * gc[pt == 11].sum())
            rows["s10o"].append(dlq * go[pt == 10].sum())
            rows["s11o"].append(dlq * go[pt == 11].sum())
        r = {k: np.asarray(v) for k, v in rows.items()}
        log()
        log(f"  {stem}  delta {dtxt}  (global {g})  N={len(r['fdc'])}")
        for lab, fd, an, s10, s11 in (("chi2 term (objchisq vs gradchisqv)", r["fdc"], r["anc"], r["s10c"], r["s11c"]),
                                      ("full objective (objval vs gradv)", r["fdo"], r["ano"], r["s10o"], r["s11o"])):
            if len(fd) == 0 or not np.any(s10):
                log(f"    {lab}: no family-10/11 columns (rerun nominal with varianceGradFamilies=10,11,...)")
                continue
            pred = an + 2. * (s10 + s11)
            A = np.stack([s10, s11], axis=1)
            coef, *_ = np.linalg.lstsq(A, fd - an, rcond=None)
            log(f"    {lab}")
            log(f"      rms FD {np.sqrt(np.mean(fd ** 2)):.3e}   rms an {np.sqrt(np.mean(an ** 2)):.3e}")
            log(f"      rel(FD, an)                  {rel(fd, an):.3f}")
            log(f"      rel(FD, an + 2*(MS+ioni))    {rel(fd, pred):.3f}")
            log(f"      free fit: coef MS {coef[0]:+.3f}  ioni {coef[1]:+.3f}   rel {rel(fd, an + A @ coef):.3f}")
    with open(os.path.join(OUTDIR, f"check_gradv_reweight_{args.mode}.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    log(f"\nwritten {OUTDIR}/check_gradv_reweight_{args.mode}.txt")


if __name__ == "__main__":
    main()
