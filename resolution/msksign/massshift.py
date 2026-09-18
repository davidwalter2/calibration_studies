#!/usr/bin/env python3
"""The momentum-scale-relevant number: the shift of the reconstructed mass
between two arms, as a mean, a trimmed mean and a median with bootstrap errors,
plus the translation to a uniform momentum scale via the measured
d ln m / d ln p of the decay.

A plain mean is reported alongside because it is what a scale fit uses, but on
a sample with a heavy per-candidate tail the trimmed and median forms say where
the bulk sits.
"""
import glob
import os
import sys

import numpy as np
import uproot


def files(pat):
    out = []
    for part in pat.split(","):
        part = part.strip()
        if not part:
            continue
        fs = sorted(glob.glob(os.path.join(part, "*.root"))) if os.path.isdir(part) else [part]
        out += [f for f in fs if os.path.getsize(f) > 0]
    return out


BR = ["Jpsi_mass", "Jpsi_sigmamass", "run", "lumi", "event",
      "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
      "Muplus_phi", "Muminus_phi"]


def load(pat):
    parts = []
    for f in files(pat):
        t = uproot.open(f)["tree"]
        parts.append(t.arrays([b for b in BR if b in t.keys()], library="np"))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def keys(D):
    out, seen = [], {}
    for r, l, e in zip(D["run"], D["lumi"], D["event"]):
        k = (int(r), int(l), int(e))
        o = seen.get(k, 0)
        seen[k] = o + 1
        out.append((*k, o))
    return out


def boot(x, fn, n=2000, rng=None):
    rng = rng or np.random.default_rng(3)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return float(np.std([fn(x[i]) for i in idx], ddof=1))


def dlnm_dlnp(D, md):
    def vec(pt, eta, phi):
        return np.stack([pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)])
    p1 = vec(D["Muplus_pt"], D["Muplus_eta"], D["Muplus_phi"])
    p2 = vec(D["Muminus_pt"], D["Muminus_eta"], D["Muminus_phi"])
    n1, n2 = np.linalg.norm(p1, axis=0), np.linalg.norm(p2, axis=0)
    E1, E2 = np.hypot(n1, md), np.hypot(n2, md)
    dot = (p1*p2).sum(axis=0)
    m2 = 2*md**2 + 2*(E1*E2 - dot)
    return 0.5*2*(E1*n2**2/E2 + E2*n1**2/E1 - 2*dot)/m2


def main():
    for label, a, b, md in (
        ("J/psi -> mu mu (gun, 1300 ev)", sys.argv[1], sys.argv[2], 0.1056583745),
        ("K_S -> pi pi (33000 ev)", sys.argv[3], sys.argv[4], 0.13957039),
        ("Z -> mu mu (DY MiniAOD, 700 ev)", sys.argv[5], sys.argv[6], 0.1056583745),
    ):
        A, B = load(a), load(b)
        ia = {k: i for i, k in enumerate(keys(A))}
        pr = [(ia[k], j) for j, k in enumerate(keys(B)) if k in ia]
        si = np.array([p[0] for p in pr])
        sj = np.array([p[1] for p in pr])
        r = (B["Jpsi_mass"][sj] - A["Jpsi_mass"][si]) / A["Jpsi_mass"][si]
        r = r[np.isfinite(r)].astype(float)
        f = dlnm_dlnp({k: A[k][si] for k in A}, md)
        fm = float(np.median(f))
        tr = float(np.mean(r[(r > np.percentile(r, 1)) & (r < np.percentile(r, 99))]))
        print(f"\n{label}   n = {len(r)}   d ln m / d ln p = {fm:.4f}")
        print(f"  mean   Delta m/m = {r.mean():+.3e} +- {r.std(ddof=1)/np.sqrt(len(r)):.3e}"
              f"   -> momentum scale {r.mean()/fm:+.3e} +- {r.std(ddof=1)/np.sqrt(len(r))/fm:.3e}")
        print(f"  1-99 % trimmed     = {tr:+.3e} +- {boot(r, lambda x: np.mean(x[(x>np.percentile(x,1))&(x<np.percentile(x,99))])):.3e}")
        print(f"  median             = {np.median(r):+.3e} +- {boot(r, np.median):.3e}")
        print(f"  per-candidate |Delta m/m|: median {np.median(np.abs(r)):.3e}  p95 {np.percentile(np.abs(r),95):.3e}  max {np.abs(r).max():.3e}")
        sg = A["Jpsi_sigmamass"][si] / A["Jpsi_mass"][si]
        z = (B["Jpsi_mass"][sj] - A["Jpsi_mass"][si]) / A["Jpsi_sigmamass"][si]
        z = z[np.isfinite(z)]
        print(f"  in sigma_m units: mean {z.mean():+.3e} +- {z.std(ddof=1)/np.sqrt(len(z)):.3e}"
              f"   (sigma_m/m median {np.median(sg):.4f})")
        ds = (B["Jpsi_sigmamass"][sj] - A["Jpsi_sigmamass"][si]) / A["Jpsi_sigmamass"][si]
        ds = ds[np.isfinite(ds)]
        print(f"  sigma_m relative change: mean {ds.mean():+.3e} +- {ds.std(ddof=1)/np.sqrt(len(ds)):.3e}"
              f"   median {np.median(ds):+.3e}")


if __name__ == "__main__":
    main()
