#!/usr/bin/env python3
"""The same covariance + truth extraction for a J/psi -> mu mu production.

The K_S needs an offline truth join because the maker's gen matching is
muon-specific; a J/psi production has it in the file (`Jpsigen_mass`,
`Mu*gen_pt/eta/phi`), so this reads the candidates straight out and applies
`cf_inmaker`'s own mass-window acceptance, producing the two caches
`ares_angles.py` consumes -- the `--cov` one, plus a minimal `--pairs` one
carrying `z`, `sigma`, `eta` (the gen mass), `vgf` and `fang`.

usage:
  python3 ares_jpsi_extract.py --prod <dir> --ntasks 60 \\
      --cov runs/jpsicov_ideal.npz --pairs runs/jpsimin_ideal.npz -j 32
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import uproot

_F32 = ("Jpsi_covrefmom", "Jpsi_jacrefmom")
_SCAL = ("Jpsi_qoprefplus", "Jpsi_qoprefminus", "Jpsi_mass", "Jpsi_sigmamass",
         "Muplus_pt", "Muplus_eta", "Muplus_phi",
         "Muminus_pt", "Muminus_eta", "Muminus_phi",
         "Jpsi_x", "Jpsi_y", "Jpsi_z", "cfmass_vgf", "chisqval", "ndof",
         "Jpsi_fang", "Jpsigen_mass",
         "Muplusgen_pt", "Muplusgen_eta", "Muplusgen_phi",
         "Muminusgen_pt", "Muminusgen_eta", "Muminusgen_phi")
_INT = ("run", "lumi", "event", "Muplus_nvalid", "Muminus_nvalid",
        "Muplus_nvalidpixel", "Muminus_nvalidpixel")


def read_one(job):
    fn, mref, mhw = job
    try:
        f = uproot.open(fn)
        if "tree" not in f:
            return None
        t = f["tree"]
    except Exception:
        return None
    need = sorted(set(_F32 + _SCAL + _INT + ("cfmass_ok",)))
    have = set(t.keys())
    miss = [b for b in need if b not in have]
    if miss:
        raise SystemExit(f"{fn} is missing {miss}")
    a = t.arrays(need, library="np")
    if not len(a["Jpsi_mass"]):
        return None
    ok = np.asarray(a["cfmass_ok"]).astype(bool)
    sig = np.asarray(a["Jpsi_sigmamass"], dtype=np.float64)
    mg = np.asarray(a["Jpsigen_mass"], dtype=np.float64)
    keep = ok & np.isfinite(sig) & (sig > 0.0) & (np.abs(mg - mref) <= mhw)
    idx = np.where(keep)[0]
    if not len(idx):
        return None
    out = {}
    for b, ncol in zip(_F32, (21, 6)):
        v = a[b]
        arr = np.empty((len(idx), ncol))
        for k, i in enumerate(idx):
            row = np.asarray(v[i], dtype=np.float64)
            arr[k] = row if len(row) == ncol else np.nan
        out[b] = arr
    for b in _SCAL:
        out[b] = np.asarray(a[b], dtype=np.float64)[idx]
    for b in _INT:
        out[b] = np.asarray(a[b], dtype=np.int64)[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--ntasks", type=int, default=60)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=32)
    ap.add_argument("--mass-window", type=float, nargs=2, default=(3.0969, 0.35))
    args = ap.parse_args()

    tds = sorted(glob.glob(os.path.join(args.prod, "task_[0-9]*")))[:args.ntasks]
    fs = []
    for td in tds:
        fs += sorted(glob.glob(os.path.join(td, "globalcor_*.root")))
    print(f"{len(tds)} tasks, {len(fs)} files")
    jobs = [(fn, args.mass_window[0], args.mass_window[1]) for fn in fs]
    parts = []
    with Pool(args.jobs) as pool:
        for n, r in enumerate(pool.imap(read_one, jobs)):
            if r is not None:
                parts.append(r)
            if (n + 1) % 500 == 0:
                print(f"  {n+1}/{len(jobs)}", flush=True)
    keys = parts[0].keys()
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    n = len(out["Jpsi_mass"])
    print(f"{n} candidates")
    np.savez_compressed(args.cov, **out)
    # the minimal pairs cache `ares_angles.py` needs
    mini = dict(z=(out["Jpsi_mass"] - out["Jpsigen_mass"]) / out["Jpsi_sigmamass"],
                sigma=out["Jpsi_sigmamass"], eta=out["Jpsigen_mass"],
                vgf=out["cfmass_vgf"], fang=out["Jpsi_fang"],
                nvp=out["Muplus_nvalid"], nvm=out["Muminus_nvalid"],
                run=out["run"], lumi=out["lumi"], event=out["event"])
    np.savez_compressed(args.pairs, **mini)
    print("wrote", args.cov, "and", args.pairs)


if __name__ == "__main__":
    main()
