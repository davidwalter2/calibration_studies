#!/usr/bin/env python3
"""Materialise SHUFFLED subsamples of the CF exponent caches (read-only).

The production caches are compressed npz: every ``d["Sms"]`` re-inflates a
~0.5 GB array.  This script pays that cost ONCE per cache, draws a fixed
random subsample of candidates (seeded, so every downstream script sees the
same rows) and writes the result as an UNCOMPRESSED npz in the scratchpad, so
the spectral / reconstruction / NLL studies load in ~1 s.

Usage
  python3 make_subsets.py --cache <name> --n 60000 --seed 1234
"""
import argparse, os, time, zipfile
import numpy as np
import numpy.lib.format as fmt

RUNS = "/work/submit/david_w/ZMass/calibration_studies/resolution/runs"
# The shuffled subsamples are multi-GB, so they live beside the full CF
# caches under resolution/runs, never in the repository.
SCRATCH = ("/work/submit/david_w/ZMass/calibration_studies/resolution/"
           "runs/cfcompress")

CACHES = {
    "jpsigun": dict(
        path=f"{RUNS}/cf_masspairs_jpsigun_ul16_260903x_m0.npz",
        kernel=f"{RUNS}/cf_masskernel_jpsigun_ul16_260903x_m0.npz",
        mats=["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"],
        scal=["z", "sigma", "eta", "vgf"]),
    "btojpsix": dict(
        path=f"{RUNS}/cf_masspairs_btojpsix_v3_260903x_m0.npz",
        kernel=f"{RUNS}/cf_masskernel_btojpsix_v3_260903x_m0.npz",
        mats=["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"],
        scal=["z", "sigma", "eta", "vgf"]),
    "trk_lowpt": dict(
        path=f"{RUNS}/cf_trackres_mugun_lowpt_260903x_m0_k0.npz",
        kernel=None,
        mats=["Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im"],
        scal=["z", "sigma", "eta", "vgf", "charge", "genpt", "trackpt",
              "normchi2", "nvalidhits"]),
    "trk_ul16": dict(
        path=f"{RUNS}/cf_trackres_mugun_ul16_260903x_m0_k0.npz",
        kernel=None,
        mats=["Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im"],
        scal=["z", "sigma", "eta", "vgf", "charge", "genpt", "trackpt",
              "normchi2", "nvalidhits"]),
}


def nrows(path, key):
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as fh:
        shape, _, _ = fmt._read_array_header(fh, fmt.read_magic(fh))
    return shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, choices=sorted(CACHES))
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c = CACHES[a.cache]
    n = nrows(c["path"], "z")
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.permutation(n)[: min(a.n, n)])   # shuffle, then subsample
    print(f"{a.cache}: {n} candidates -> {len(idx)} (seed {a.seed})", flush=True)
    out = a.out or f"{SCRATCH}/sub_{a.cache}_n{len(idx)}_s{a.seed}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d = np.load(c["path"])
    res = dict(idx=idx, tgrid=np.asarray(d["tgrid"], np.float64),
               ntot=np.int64(n), src=np.str_(c["path"]))
    for k in c["scal"]:
        res[k] = np.asarray(d[k])[idx]
    for k in c["mats"]:
        t0 = time.time()
        res[k] = np.ascontiguousarray(d[k][idx])
        print(f"  {k:8s} {res[k].shape} {res[k].dtype}  {time.time()-t0:.1f} s",
              flush=True)
    for k in ("rad_model", "ioni_sign_fixed", "ioni_charge_signed"):
        if k in d.files:
            res[k] = d[k]
    if c["kernel"]:
        kd = np.load(c["kernel"])
        res["dm"] = np.asarray(kd["dm"], np.float64)
        print(f"  kernel dm {res['dm'].shape}", flush=True)
    np.savez(out, **res)
    print(f"wrote {out}  ({os.path.getsize(out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
