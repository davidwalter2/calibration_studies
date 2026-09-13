#!/usr/bin/env python3
"""Is the VERTEX residual correlated with the MASS residual, and by how much?

Three levels, all from the maker's own per-block influence vectors
`a_b = dV_b^{1/2} w_b` (`resinfvtxv` for the vertex functional, `resinfv` for
the mass one):

(1) THE ALGEBRA.  Both are linear functionals of the same fit, so
        Cov(r_v, dm) = sum_b a_b^v . a_b^m = e_6^T C a_m
    exactly (every residual row is covered by a registered block when the
    beamspot rows are off, which is the `vtxvchk` gate).  It is NOT zero by
    construction the way the per-hit complement was (`F^T R = 0`); what makes
    it small is a MIRROR SYMMETRY: reflecting the event in the plane spanned
    by the two momenta sends `n_hat = p_a x p_b` to `-n_hat`, hence
    `theta_6 -> -theta_6`, while the momenta -- and so the mass -- are
    unchanged.  Only the magnetic field breaks that symmetry.  So the
    prediction is "small, and set by the field", and this script measures it.

(2) THE ENSEMBLE.  corr(z_v, z_m) with `z_m = (m - m_gen)/sigma_m`, which is
    the mass pull with the FSR kernel divided out.

(3) THE FOURTH CROSS CUMULANT, the thing a product-of-marginals (composite)
    likelihood drops.  Per material block, for the iso-Gaussian-plus-radial-
    tail structure of Moliere,
        kappa(z_v, z_v, z_m, z_m) ~ (A_v.A_m)^2 + |A_v|^2 |A_m|^2 / 2
    against a diagonal (3/2)|A_v|^4, exactly as `hitlik/xcum_perhit.py` uses.
"""
import argparse, glob, sys
import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), "."))
import selection as _SEL  # noqa: E402  (ONE value for the chi2 cut)
import uproot

BR = ["resinfvtxv", "resinfv", "resinfvarv", "vtxvarv", "reseigidx",
      "Jpsi_vtxsig", "Jpsi_sigmamass", "Jpsi_vtxres", "Jpsi_mass",
      "Jpsigen_mass", "Jpsi_vtxz", "Jpsi_vtxok", "cfmass_ok",
      "chisqval", "ndof", "reshitcls"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", required=True)
    p.add_argument("--max", type=int, default=20000)
    p.add_argument("--max-chi2-ndof", type=float,
                   default=_SEL.MAX_CHI2_NDOF)
    a = p.parse_args()

    files = sorted(glob.glob(a.files))
    out, got = {}, 0
    for f in files:
        try:
            fh = uproot.open(f)
        except Exception as e:
            print(f"# skip {f}: {e}", file=sys.stderr)
            continue
        with fh:
            t = fh["tree"]
            keys = set(k.split(";")[0] for k in t.keys())
            want = [b for b in BR if b in keys]
            d = t.arrays(want, library="np")
        for k, v in d.items():
            out.setdefault(k, []).append(v)
        got += len(d[want[0]])
        if got >= a.max:
            break
    d = {k: np.concatenate(v)[: a.max] for k, v in out.items()}
    n = len(d["Jpsi_vtxsig"])
    ok = (np.asarray(d["Jpsi_vtxok"], bool) & np.asarray(d["cfmass_ok"], bool)
          & (np.asarray(d["Jpsi_vtxsig"]) > 0) & (np.asarray(d["Jpsi_sigmamass"]) > 0))
    ndof = np.asarray(d["ndof"], float)
    ok &= (np.asarray(d["chisqval"], float) / np.maximum(ndof, 1)) < a.max_chi2_ndof
    idx = np.flatnonzero(ok)
    print(f"# {len(files)} files, {n} candidates, {len(idx)} good")

    sv = np.asarray(d["Jpsi_vtxsig"], float)
    sm = np.asarray(d["Jpsi_sigmamass"], float)
    rho_alg, k4c = [], []
    for i in idx:
        av = np.asarray(d["resinfvtxv"][i], float).reshape(-1, 5)
        am = np.asarray(d["resinfv"][i], float).reshape(-1, 5)
        if av.shape != am.shape:
            continue
        cov = float((av * am).sum())
        rho_alg.append(cov / (sv[i] * sm[i]))
        # the fourth cross cumulant, material blocks only (hit blocks are
        # Gaussian and contribute none)
        cls = np.asarray(d["reshitcls"][i], int) if "reshitcls" in d else \
            np.full(av.shape[0], -1)
        m = cls[: av.shape[0]] < 0
        Av = (av[m] ** 2).sum(axis=1)
        Am = (am[m] ** 2).sum(axis=1)
        Avm = (av[m] * am[m]).sum(axis=1)
        num = (Avm ** 2 + 0.5 * Av * Am).sum()
        dv = (1.5 * Av ** 2).sum()
        dm = (1.5 * Am ** 2).sum()
        k4c.append(num / max(np.sqrt(dv * dm), 1e-300))
    rho_alg = np.array(rho_alg)
    k4c = np.array(k4c)

    zm = (np.asarray(d["Jpsi_mass"], float) - np.asarray(d["Jpsigen_mass"], float)) / sm
    zv = np.asarray(d["Jpsi_vtxz"], float)
    good = idx[np.isfinite(zm[idx]) & np.isfinite(zv[idx])]
    print("\n=== (1) the ALGEBRA: per-candidate Cov(r_v, dm)/(sigma_v sigma_m)")
    print(f"  median {np.median(rho_alg):+.6f}  mean {rho_alg.mean():+.6f}  "
          f"rms {rho_alg.std():.6f}  p1 {np.percentile(rho_alg,1):+.5f}  "
          f"p99 {np.percentile(rho_alg,99):+.5f}  max|.| {np.abs(rho_alg).max():.5f}")
    print("\n=== (2) the ENSEMBLE: corr(z_v, z_m),  z_m = (m - m_gen)/sigma_m")
    c = np.corrcoef(zv[good], zm[good])[0, 1]
    print(f"  corr = {c:+.5f} +- {1/np.sqrt(len(good)):.5f}   (N = {len(good)})")
    print(f"  corr(|z_v|, |z_m|) = {np.corrcoef(np.abs(zv[good]), np.abs(zm[good]))[0,1]:+.5f}")
    print(f"  corr(z_v^2, z_m^2) = {np.corrcoef(zv[good]**2, zm[good]**2)[0,1]:+.5f}")
    print("\n=== (3) the FOURTH CROSS CUMULANT (material blocks)")
    print(f"  kappa(v,v,m,m)/sqrt(kappa4_v kappa4_m): median {np.median(k4c):.4f}  "
          f"p16 {np.percentile(k4c,16):.4f}  p84 {np.percentile(k4c,84):.4f}")
    print("\n  (for scale: the per-hit study found 0.09-0.14 between q/p and the "
          "other reference components and 0.71 between phi and d0)")


if __name__ == "__main__":
    main()
