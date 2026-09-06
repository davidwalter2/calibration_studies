#!/usr/bin/env python3
"""Per-step comparison of the CVH fit's Gaussian MS variance (Q) with the
Moliere model the offline CF uses, split by parmtype-15 material group.

WHY
---
The joint (quadratic hit-chi2 + unbinned mass CF) fit on the J/psi gun MC finds
`material_tec_services` at k = -0.465 +- 0.068 from the hit chi2 and
k = -0.009 +- 0.095 from the mass term, on MC whose truth is k = 0 for every
group.  Both functionals scale the SAME per-step material amount, but they do
it through two different statistics of the same scattering distribution:

* the fit's `Q` uses `Geant4ePropagator::PropagateErrorMSC`,

      theta_p^2(step) = (15 MeV / p beta)^2 * L_cm / Xs_cm,
      Xs = X0 (Z+1)/Z ln(287/sqrt(Z)) / ln(159 Z^{-1/3}),   Z = effZ

  i.e. ROSSI's scattering power evaluated at the MASS-AVERAGED effective Z of
  the compound, with no dependence on the step's own thickness;

* the CF uses the Moliere/screened-Rutherford log-CF

      S_step(u) = (chi_c^2/chi_a^2) G(u sqrt(chi_a^2); ymax),

  with chi_c^2 built from the PER-ELEMENT sum sum_i w_i Z_i(Z_i+1)/A_i and
  chi_a^2 from the scattering-power-weighted geometric mean of the G4 screening
  radius -- both exported per step in `msmoliv` columns 7 and 8 precisely
  because "averaging Z first is wrong for compounds"
  (`Geant4ePropagator::CalculateMoliereSums`).

The Gaussian variance the CF's core implies for a step is

      sigma^2_eff(u) = -2 S_step(u) / u^2 = chi_c^2 F(y),   F(y) = -2 G(y)/y^2,
      y = u sqrt(chi_a^2),      F(y -> 0) = Lm/2  (the second moment).

so the ratio the two functionals disagree by is

      R = sigma^2_eff / theta_p^2(Q)
        = (0.157e-6 / 2.25e-4) * ZZA * Xs_g * F(y),   ZZA = sum_i w_i Z_i(Z_i+1)/A_i,

which is INDEPENDENT of the step's thickness and of p except through F's slowly
varying log.  Everything in it is in the step record.

This script reads `msmoliv` (stride 10: effZ effA xg pGeV beta thp2 dOverX0
zzp1OverA lnScreenW stepGroup) from a CVH production and writes per-group and
per-step summaries to an npz.  No fitting, no corrections.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "matres"))

import cf_ms_exact as MSX          # noqa: E402  (gshape, moliere_params constants)
import cf_track_resolution as CTR  # noqa: E402  (the PRODUCTION ms_step_exponent)
import groups as G                 # noqa: E402  (column layout, read_groups)
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import prodfiles  # noqa: E402  (needs resolution/ on sys.path)

# msmoliv columns (groups.py)
C_EFFZ, C_EFFA, C_XG, C_P, C_BETA, C_THP2, C_DOX0, C_ZZP1, C_LNSW, C_GRP = range(10)

ALPHA_EM = MSX.ALPHA_EM
CHI0 = MSX._CHI0_G4 if MSX.MS_CHI0_G4 else MSX._CHI0_OURS
FF_CONSTN = MSX._FF_CONSTN_MEV2

ROSSI_E2 = 2.25e-4      # (15 MeV)^2 in GeV^2, PropagateErrorMSC
CHIC_K = 0.157e-6       # GeV^2 cm^2/g, Moliere


def moliere_cols(rec):
    """chi_c^2, chi_a^2, theta_FF^2 for an (n,10) msmoliv block, vectorized and
    numerically identical to cf_ms_exact.moliere_params on stride-10 rows."""
    effZ, effA = rec[:, C_EFFZ], rec[:, C_EFFA]
    xg, p, beta = rec[:, C_XG], rec[:, C_P], rec[:, C_BETA]
    zza, lnsw = rec[:, C_ZZP1], rec[:, C_LNSW]
    p2b2 = np.maximum(p * p * beta * beta, 1e-30)
    chic2 = CHIC_K * zza * xg / p2b2
    chia2 = (CHI0 / np.maximum(p, 1e-30)) ** 2 * np.exp(lnsw)
    pmev = p * 1.0e3
    thff2 = 2.0 / (FF_CONSTN * np.maximum(effA, 1.0) ** 0.54 * pmev * pmev)
    return chic2, chia2, thff2


def rossi_xs_over_x0(Z):
    """Xs/X0 as PropagateErrorMSC computes it, at the mass-averaged effZ."""
    Z = np.maximum(Z, 1e-6)
    return (Z + 1.0) / Z * np.log(287.0 / np.sqrt(Z)) / np.log(159.0 * Z ** (-1.0 / 3.0))


def core_u(rec, ulo, uhi, target=-0.5, nit=40):
    """u* with S_total(u*) = target (default -1/2, i.e. the CF has fallen to
    exp(-1/2): the Gaussian that matches the core at 1 sigma has
    sigma_core = 1/u*).  Bisection on the exact production exponent."""
    for _ in range(nit):
        um = 0.5 * (ulo + uhi)
        s = float(CTR.ms_step_exponent(rec, 1.0, np.array([um]))[0])
        if s > target:          # |S| too small -> need larger u
            ulo = um
        else:
            uhi = um
    return 0.5 * (ulo + uhi)


def process_file(fn):
    args = _ARGS
    try:
        f = uproot.open(fn)
        if "tree" not in f:
            return None
        t = f["tree"]
    except Exception as e:                                   # noqa: BLE001
        print(f"WARNING: cannot open {fn} ({type(e).__name__})")
        return None
    if t.num_entries == 0:
        return None
    want = ["msmoliidx", "msmoliv", "chisqval", "ndof"]
    keys = set(t.keys())
    for b in ("gradmax", "hessmax"):
        if b in keys:
            want.append(b)
    a = t.arrays(want, entry_stop=(args.entries or None), library="np")
    n = len(a["msmoliv"])
    rchi2 = np.asarray(a["chisqval"], np.float64) / np.maximum(
        np.asarray(a["ndof"], np.float64), 1.0)
    ok = rchi2 < args.max_chi2_ndof if args.max_chi2_ndof > 0 else np.ones(n, bool)
    for b, c in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if c > 0 and b in a:
            ok &= np.abs(np.asarray(a[b], np.float64)) < c

    NG = args.ngroups
    UF = np.asarray(args.ufac, float)
    NU = len(UF)
    acc = dict(
        n=np.zeros(NG, np.int64),
        thp2=np.zeros(NG),                 # sum of Q's projected variance
        chic2=np.zeros(NG),                # sum of chi_c^2
        seff=np.zeros((NU, NG)),           # sum of -2 S_g(u)/u^2 at u = ufac*u*
        s2nd=np.zeros(NG),                 # sum of chi_c^2 Lm/2  (analytic u->0)
        s2ndk=np.zeros(NG),                # same from the kernel at u -> 0
        xg=np.zeros(NG),
        dox0=np.zeros(NG),
        w_effZ=np.zeros(NG), w_effA=np.zeros(NG), w_zza=np.zeros(NG),
        w_x0g=np.zeros(NG), w_xsg=np.zeros(NG), w_lnsw=np.zeros(NG),
        w_om0=np.zeros(NG), w_lm=np.zeros(NG), w_R2=np.zeros(NG),
        w_len=np.zeros(NG), w_p=np.zeros(NG),
        w=np.zeros(NG),
        ncand=0, nstep=0,
        sig_core=np.zeros(2),              # sum, sum^2 of 1/u* over candidates
    )
    lb = np.linspace(-9.0, 0.0, 91)
    acc["hist_dox0"] = np.zeros((NG, len(lb) - 1), np.int64)
    acc["hist_bins"] = lb
    samp = []

    for ic in range(n):
        if not ok[ic]:
            continue
        idx = np.asarray(a["msmoliidx"][ic])
        v = np.asarray(a["msmoliv"][ic], np.float64)
        if not len(idx):
            continue
        stride = len(v) // len(idx)
        if stride < 10:
            sys.exit(f"{fn}: msmoliv stride {stride} < 10")
        rec = v.reshape(-1, stride)
        rec = rec[rec[:, C_THP2] > 0.0]
        if not len(rec):
            continue
        chic2, chia2, thff2 = moliere_cols(rec)
        act = (chic2 > 0.0) & (chia2 > 0.0)
        rec, chic2, chia2, thff2 = rec[act], chic2[act], chia2[act], thff2[act]
        if not len(rec):
            continue
        gid = rec[:, C_GRP].astype(np.int64)
        gid[(gid < 0) | (gid >= NG)] = 0
        thp2 = rec[:, C_THP2]
        sig_tot = np.sqrt(thp2.sum())
        if not (sig_tot > 0):
            continue

        # core scale of the WHOLE candidate: S_total(u*) = -1/2
        ustar = core_u(rec, 0.02 / sig_tot, 50.0 / sig_tot)
        uu = UF * ustar
        usmall = 1e-3 * ustar               # u -> 0 probe for the second moment

        gu = np.unique(gid)
        for g in gu:
            m = gid == g
            rg = rec[m]
            Sg = CTR.ms_step_exponent(rg, 1.0, uu)
            acc["seff"][:, g] += -2.0 * Sg / (uu * uu)
            S0 = float(CTR.ms_step_exponent(rg, 1.0, np.array([usmall]))[0])
            acc["s2ndk"][g] += -2.0 * S0 / (usmall * usmall)

        ym = np.sqrt(np.clip(thff2 / chia2, 1e2, 1e14))
        Lm = np.log(ym * ym) - 1.0
        np.add.at(acc["n"], gid, 1)
        np.add.at(acc["thp2"], gid, thp2)
        np.add.at(acc["chic2"], gid, chic2)
        np.add.at(acc["s2nd"], gid, chic2 * Lm / 2.0)
        np.add.at(acc["xg"], gid, rec[:, C_XG])
        np.add.at(acc["dox0"], gid, rec[:, C_DOX0])
        w = rec[:, C_XG]
        x0g = rec[:, C_XG] / np.maximum(rec[:, C_DOX0], 1e-300)
        xsg = x0g * rossi_xs_over_x0(rec[:, C_EFFZ])
        R2 = (CHIC_K / ROSSI_E2) * rec[:, C_ZZP1] * xsg * (Lm / 2.0)
        np.add.at(acc["w"], gid, w)
        np.add.at(acc["w_effZ"], gid, w * rec[:, C_EFFZ])
        np.add.at(acc["w_effA"], gid, w * rec[:, C_EFFA])
        np.add.at(acc["w_zza"], gid, w * rec[:, C_ZZP1])
        np.add.at(acc["w_x0g"], gid, w * x0g)
        np.add.at(acc["w_xsg"], gid, w * xsg)
        np.add.at(acc["w_lnsw"], gid, w * rec[:, C_LNSW])
        np.add.at(acc["w_om0"], gid, w * (chic2 / chia2))
        np.add.at(acc["w_lm"], gid, w * Lm)
        np.add.at(acc["w_R2"], gid, w * R2)
        np.add.at(acc["w_len"], gid, w * rec[:, C_DOX0] * x0g)   # x_g again (bookkeeping)
        np.add.at(acc["w_p"], gid, w * rec[:, C_P])
        ib = np.clip(np.digitize(np.log10(np.maximum(rec[:, C_DOX0], 1e-12)), lb) - 1,
                     0, len(lb) - 2)
        np.add.at(acc["hist_dox0"], (gid, ib), 1)
        acc["ncand"] += 1
        acc["nstep"] += len(rec)
        acc["sig_core"] += np.array([1.0 / ustar, 1.0 / ustar ** 2])

        if args.sample and len(samp) * 64 < args.sample:
            k = min(64, len(rec))
            sel = np.random.default_rng(ic).choice(len(rec), size=k, replace=False)
            samp.append(np.column_stack([
                rec[sel, C_EFFZ], rec[sel, C_EFFA], rec[sel, C_XG], rec[sel, C_P],
                rec[sel, C_BETA], rec[sel, C_THP2], rec[sel, C_DOX0],
                rec[sel, C_ZZP1], rec[sel, C_LNSW], gid[sel].astype(float),
                chic2[sel], chia2[sel], thff2[sel],
                np.full(k, ustar)]))
    acc["sample"] = np.concatenate(samp) if samp else np.zeros((0, 14))
    return acc


def _init(args):
    global _ARGS
    _ARGS = args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--entries", type=int, default=0)
    ap.add_argument("--max-chi2-ndof", type=float, default=3.0)
    ap.add_argument("--max-grad", type=float, default=1e6)
    ap.add_argument("--max-hess", type=float, default=1e8)
    ap.add_argument("--ngroups", type=int, default=64)
    ap.add_argument("--ufac", type=float, nargs="+", default=[1.0, 0.3333, 3.0])
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args()

    # --max-files caps TASKS (a multi-stream task is N files)
    files = prodfiles.resolve(args.files, args.max_files,
                              logger=lambda m: print(m, flush=True))
    print(f"{len(files)} files", flush=True)
    t0 = time.time()
    tot = None
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(args,)) as ex:
        for i, r in enumerate(ex.map(process_file, files)):
            if r is None:
                continue
            if tot is None:
                tot = r
            else:
                for k, v in r.items():
                    if k == "hist_bins":
                        continue
                    if k == "sample":
                        tot[k] = np.concatenate([tot[k], v])
                    else:
                        tot[k] = tot[k] + v
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f}s "
                      f"ncand={tot['ncand']}", flush=True)
    tot["ufac"] = np.asarray(args.ufac)
    np.savez_compressed(args.output, **tot)
    print(f"wrote {args.output}  ncand={tot['ncand']}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
