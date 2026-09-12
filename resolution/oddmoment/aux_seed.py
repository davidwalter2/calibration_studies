#!/usr/bin/env python3
"""Per-leg SEED -> FINAL q/p step, aligned to a two-track pairs cache.

The track-level `eta` dependence of the charge-even skew is a MIXTURE:
splitting the 20-60 GeV gun on `|seed -> final dq/p|` at its 90th percentile
gives two `eta`-INDEPENDENT components (`+20.59 +- 6.45` and `-6.33 +- 1.84`
e-3) whose mixing fraction runs 2.4 -> 7.0 -> 21.2 % with `|eta|`. The mass
caches do not carry that step, and `chi2/ndof` is NOT a proxy for it: the
population it selects is `eta`-flat but so is its fraction, where theirs grows
9x. This extracts the real thing from the two-track trees:

    q/p seed  = q / (Mu{plus,minus}trk_pt * cosh(Mu{plus,minus}trk_eta))
    q/p final = Jpsi_qopref{plus,minus}        (== Mu*_refParms[0], checked)
    dq/p      = |q/p final - q/p seed| / |q/p seed|

**ABSOLUTE VALUE ONLY.** The SIGNED step has `corr(., x) = +0.1132` on the gun
-- a worse conditioning trap than reco `pT`'s `+0.042`, because the seed is the
generalTracks KF, which shares its hits with CVH, so their difference is partly
the residual itself. The absolute step measured `+0.0061` there. Every
consumer must state `corr(|delta|, |x|)` on ITS OWN sample before binning on
it.

Also carried, so a leg-level pull can be built without a second pass:

    sigrel_{p,m}   `Jpsi_sigmarel{plus,minus}`, the per-leg relative momentum
                   resolution the two-track fit reports
    qopgen_{p,m}   q / (Mu*gen_pt cosh(Mu*gen_eta)) -- the same truth the
                   `aux_gen.py` columns are built from
    z_{p,m}        (q/p final - q/p gen) / (sigrel |q/p final|), the per-leg
                   analogue of the gun's `t_z`

The alignment to the cache is `aux_gen.join_to_cache` -- ONE implementation,
order-independent on (run, lumi, event, z), and the post-checks assert the
mass-level `z` and `sigma` come back bit-identical.

usage:
  python3 aux_seed.py --files DIR --cache runs/zpairs_dyv2_full.npz \\
      --out runs/auxseed_dyv2.npz [--nproc 32]
"""
import argparse
import os
import sys

import numpy as np
import uproot

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import prodfiles  # noqa: E402
from oddmoment.aux_gen import join_to_cache  # noqa: E402

BR = ["run", "lumi", "event", "Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass",
      "Jpsi_qoprefplus", "Jpsi_qoprefminus",
      "Jpsi_sigmarelplus", "Jpsi_sigmarelminus",
      "Muplustrk_pt", "Muplustrk_eta", "Muminustrk_pt", "Muminustrk_eta",
      "Muplus_pt", "Muplus_eta", "Muminus_pt", "Muminus_eta",
      "Muplusgen_pt", "Muplusgen_eta", "Muminusgen_pt", "Muminusgen_eta",
      "Muplus_charge", "Muminus_charge",
      # hit content. `nvalidpixel` is the pixel count the mass-level endcap
      # miss is indexed on.
      "Muplus_nvalid", "Muminus_nvalid",
      "Muplus_nvalidpixel", "Muminus_nvalidpixel"]


def one(fn):
    t = uproot.open(fn)["tree"]
    a = t.arrays(BR, library="np")
    f8 = lambda k: a[k].astype(np.float64)  # noqa: E731
    m = f8("Jpsi_mass")
    mg = f8("Jpsigen_mass")
    s = f8("Jpsi_sigmamass")
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (m - mg) / s
    out = dict(run=a["run"].astype(np.int64), lumi=a["lumi"].astype(np.int64),
               event=a["event"].astype(np.int64), z=z, sigma=s)
    for lab, sfx in (("p", "plus"), ("m", "minus")):
        q = f8(f"Mu{sfx}_charge")
        pseed = f8(f"Mu{sfx}trk_pt") * np.cosh(f8(f"Mu{sfx}trk_eta"))
        pgen = f8(f"Mu{sfx}gen_pt") * np.cosh(f8(f"Mu{sfx}gen_eta"))
        with np.errstate(divide="ignore", invalid="ignore"):
            qopseed = q / pseed
            qopgen = q / pgen
        qopref = f8(f"Jpsi_qopref{sfx}")
        sr = f8(f"Jpsi_sigmarel{sfx}")
        with np.errstate(divide="ignore", invalid="ignore"):
            dq = np.abs(qopref - qopseed) / np.abs(qopseed)
            zl = (qopref - qopgen) / (sr * np.abs(qopref))
        out[f"dq_{lab}"] = dq
        out[f"qopref_{lab}"] = qopref
        out[f"qopseed_{lab}"] = qopseed
        out[f"qopgen_{lab}"] = qopgen
        out[f"sigrel_{lab}"] = sr
        out[f"zleg_{lab}"] = zl
        out[f"q_{lab}"] = q
        out[f"eta_{lab}"] = f8(f"Mu{sfx}_eta")
        out[f"geta_{lab}"] = f8(f"Mu{sfx}gen_eta")
        out[f"pt_{lab}"] = f8(f"Mu{sfx}_pt")
        out[f"nvalid_{lab}"] = f8(f"Mu{sfx}_nvalid")
        out[f"npix_{lab}"] = f8(f"Mu{sfx}_nvalidpixel")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nproc", type=int, default=32)
    p.add_argument("--ntasks", type=int, default=100000)
    a = p.parse_args()
    files = prodfiles.resolve(a.files, a.ntasks,
                              logger=lambda m: print(m, flush=True))
    print(f"{len(files)} files", flush=True)
    from multiprocessing import Pool
    with Pool(a.nproc) as pool:
        parts = pool.map(one, files)
    A = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    print(f"{len(A['z'])} tree entries", flush=True)

    idx, d = join_to_cache(A, a.cache)
    out = {k: v[idx] for k, v in A.items()}
    assert np.array_equal(out["z"], d["z"].astype(np.float64)), "z post-check"
    assert np.array_equal(out["sigma"], d["sigma"]), "sigma mismatch"
    for lab, cn in (("p", "sigrelp"), ("m", "sigrelm")):
        if cn in d.files:
            dv = np.abs(out[f"sigrel_{lab}"] - d[cn].astype(np.float64)).max()
            print(f"  max|sigrel_{lab} - cache {cn}| = {dv:.3e}", flush=True)
    n = len(out["z"])
    for lab in ("p", "m"):
        v = out[f"dq_{lab}"]
        ok = np.isfinite(v)
        print(f"  |seed->final dq/p| leg {lab}: median {np.median(v[ok]):.4e}  "
              f"p90 {np.percentile(v[ok], 90):.4e}  "
              f"p99 {np.percentile(v[ok], 99):.4e}", flush=True)
    print(f"ALIGNED: {n} rows, z and sigma bit-identical", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
