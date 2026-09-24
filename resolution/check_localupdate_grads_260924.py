#!/usr/bin/env python3
"""Finite-difference test of the exported global derivatives of the two-track
maker, for both Gauss-Newton linearisation points (``localUpdate``).

Input: the directory tree written by ``fd_localupdate_260924.sh``::

    <fdroot>/lu{False,True}/nom
    <fdroot>/lu{False,True}/mat_g<G>_{p,m}<delta>      material group k_g
    <fdroot>/lu{False,True}/field_m<M>_{p,m}<delta>    scalar-potential mode

Every +-delta run is a full refit with the parameter moved in the
propagator.  Per candidate (present in all three runs, same null-space
dimension) the central differences are compared with what the NOMINAL run
exports for that global column:

  objective   (objval+ - objval-)/2d            vs gradv
              (objchisq+ - objchisq-)/2d        vs gradchisqv (chi2 term alone)
              (objval+ + objval- - 2 objval0)/d^2 vs the Hessian diagonal
  mass        (Jpsi_mass+ - Jpsi_mass-)/2d        vs Jpsi_jacMass
  q/p         (Mu*_refParms[0]+ - ...-)/2d         vs Mu*_jacRef row 0

Figures of merit per (parameter, delta): the least-squares slope FD = s * an
(1 = consistent), and the relative residual rms(FD - an)/rms(an).  The
residual must be flat between the two deltas; if it scales like delta^2 it
is finite-difference truncation, if like 1/delta it is the refit's stopping
noise (or float32 storage for mass and q/p), and if it is flat and large it
is a derivative that is not the derivative of what the fit returns.

    python3 check_localupdate_grads_260924.py [--fdroot DIR]
"""
import argparse
import datetime
import glob
import os
import re

import numpy as np
import uproot

FDROOT = "/work/submit/david_w/ZMass/scratch_lu_260924/fd"
OUTDIR = ("/work/submit/david_w/ZMass/calibration_studies/resolution/runs/"
          f"{datetime.date.today().strftime('%y%m%d')}_localupdate_grads")

_LINES = []


def log(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


BRANCHES = ["run", "lumi", "event", "globalidxv", "gradv", "hesspackedv",
            "objval", "objchisq", "gradchisqv", "objnullv", "Jpsi_mass", "Jpsi_jacMass",
            "Muplus_refParms", "Muminus_refParms", "Muplus_jacRef",
            "Muminus_jacRef", "Muplus_nvalid", "Muminus_nvalid"]


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "globalcor_*.root")))
    if len(fs) != 1:
        raise RuntimeError(f"{d}: expected one globalcor file, found {len(fs)}")
    t = uproot.open(fs[0])["tree"]
    keys = set(t.keys())
    a = t.arrays([b for b in BRANCHES if b in keys], library="np")
    a["_key"] = {(int(r), int(l), int(e)): i
                 for i, (r, l, e) in enumerate(zip(a["run"], a["lumi"], a["event"]))}
    return a


def injected_global(d):
    """global index of the injected field mode, from the run's log"""
    with open(os.path.join(d, "cmsrun.log"), errors="replace") as fh:
        for line in fh:
            m = re.match(r"injectFieldModes: mode (\d+) \(global (\d+)\)", line)
            if m:
                return int(m.group(2))
    return None


def material_global(d, group):
    """global index of material group `group` (parmtype 15, rawdetid = group)"""
    fs = glob.glob(os.path.join(d, "globalcor_*.root"))
    rt = uproot.open(fs[0])["runtree"]
    pt = rt["parmtype"].array(library="np")
    rd = rt["rawdetid"].array(library="np")
    w = np.where((pt == 15) & (rd == group))[0]
    return int(w[0]) if len(w) else None


def hess_diag(h, n, c):
    # packed upper triangle, row major: row c starts at c*n - c*(c-1)/2
    return h[c * n - c * (c - 1) // 2]


def fom(fd, an):
    fd, an = np.asarray(fd), np.asarray(an)
    ok = np.isfinite(fd) & np.isfinite(an)
    fd, an = fd[ok], an[ok]
    if len(an) == 0 or not np.any(an):
        return len(an), np.nan, np.nan, np.nan
    s = float(fd @ an / (an @ an))
    rel = float(np.sqrt(np.mean((fd - an) ** 2)) / np.sqrt(np.mean(an ** 2)))
    med = float(np.median(np.abs(fd - an)) / np.median(np.abs(an)))
    return len(an), s, rel, med


def compare(nom, ap, am, gidx, delta, min_nvalid):
    rows = {k: ([], []) for k in ("grad", "gradchi2", "hdiag", "mass", "qop+", "qop-")}
    for k, ic in nom["_key"].items():
        if k not in ap["_key"] or k not in am["_key"]:
            continue
        ip, im = ap["_key"][k], am["_key"][k]
        if min(nom["Muplus_nvalid"][ic], nom["Muminus_nvalid"][ic]) < min_nvalid:
            continue
        if not (nom["objnullv"][ic] == ap["objnullv"][ip] == am["objnullv"][im]):
            continue
        gi = np.asarray(nom["globalidxv"][ic])
        w = np.where(gi == gidx)[0]
        if not len(w):
            continue
        c, n = int(w[0]), len(gi)
        o0, op, om = nom["objval"][ic], ap["objval"][ip], am["objval"][im]
        rows["grad"][0].append((op - om) / (2 * delta))
        rows["grad"][1].append(nom["gradv"][ic][c])
        rows["gradchi2"][0].append((ap["objchisq"][ip] - am["objchisq"][im]) / (2 * delta))
        rows["gradchi2"][1].append(nom["gradchisqv"][ic][c])
        rows["hdiag"][0].append((op + om - 2 * o0) / delta ** 2)
        rows["hdiag"][1].append(hess_diag(nom["hesspackedv"][ic], n, c))
        rows["mass"][0].append((float(ap["Jpsi_mass"][ip]) - float(am["Jpsi_mass"][im])) / (2 * delta))
        rows["mass"][1].append(nom["Jpsi_jacMass"][ic][c])
        for lab, pre in (("qop+", "Muplus"), ("qop-", "Muminus")):
            rows[lab][0].append((float(ap[pre + "_refParms"][ip][0])
                                 - float(am[pre + "_refParms"][im][0])) / (2 * delta))
            rows[lab][1].append(nom[pre + "_jacRef"][ic][c])  # row 0 of 3 x n
    return rows


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--fdroot", default=FDROOT)
    ap_.add_argument("--min-nvalid", type=int, default=8)
    args = ap_.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    log(f"fd root: {args.fdroot}")
    log("columns: N  slope(FD on analytic)  rms(FD-an)/rms(an)  median|FD-an|/median|an|")
    for mode in ("luFalse", "luTrue"):
        base = os.path.join(args.fdroot, mode)
        nom = load(os.path.join(base, "nom"))
        log()
        log("=" * 96)
        log(f"{mode}   nominal candidates: {len(nom['run'])}")
        log("=" * 96)
        for pdir in sorted(glob.glob(os.path.join(base, "*_p*"))):
            name = os.path.basename(pdir)
            stem, dtxt = name.rsplit("_p", 1)
            mdir = os.path.join(base, f"{stem}_m{dtxt}")
            if not os.path.isdir(mdir):
                continue
            try:
                a_p, a_m = load(pdir), load(mdir)
            except RuntimeError as e:
                log(f"  {name}: {e}")
                continue
            if stem.startswith("field"):
                gidx = injected_global(pdir)
            else:
                gidx = material_global(pdir, int(stem.split("_g")[1]))
            rows = compare(nom, a_p, a_m, gidx, float(dtxt), args.min_nvalid)
            log(f"  {stem}  delta {dtxt}  (global {gidx})")
            for k, (fd, an) in rows.items():
                n, s, rel, med = fom(fd, an)
                log(f"    {k:6s} N={n:4d}  slope={s:+.5f}  rel.rms={rel:.3e}  rel.med={med:.3e}")
    with open(os.path.join(OUTDIR, "check_localupdate_grads.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    log(f"\nwritten {OUTDIR}/check_localupdate_grads.txt")


if __name__ == "__main__":
    main()
