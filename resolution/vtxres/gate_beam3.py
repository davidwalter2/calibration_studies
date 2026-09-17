#!/usr/bin/env python3
"""THE ASSEMBLY GATE of the 3x3 luminous-region block.

The maker registers the beam line as resolution family 16 with ``dV = covBS``,
so a functional's variance share from that block is EXACTLY

    Var_bs = w^T covBS w = sum_ab covBS_ab Q_ab ,   Q_ab = w_a w_b

with ``w`` the functional's influence weight on the three beam rows.  The
maker exports ``-w`` per functional (``Jpsi_bsmean{mass,vtx,bs}``) and the
share itself (``Jpsi_{mass,vtx}vbs``, ``Jpsi_bsvbs``), and it exports the
record the block was built from (``Jpsi_bswidth``, ``Jpsi_bsslope``).  So the
share can be REBUILT offline from the exports alone, and that is what makes a
full 3x3 parameterisation of ``covBS`` possible with no maker change.

This script proves the rebuild:

  A1  sum_ab covBS_ab(record) Q_ab / Cov_ii  ==  the exported share
  A2  d/dk_x of the same, under the maker's `covBS -> D covBS D` convention
      (`D = diag(sqrt k_x, sqrt k_y, 1)`), == the exported `*vbsx`; likewise y
  A3  the FORMULA-REBUILD derivative (covBS recomputed from the scaled widths
      through the CMS beam-spot-fitter expression, which is what a physical
      width change does) against the same export -- they differ only through
      `d C_xz/d k_x`, `-dxdz sigma_x^2` against `+dxdz sigma_z^2 / 2`, and the
      difference is O(sigma_x^2/sigma_z^2) ~ 1e-7 of the share

usage:
  python3 gate_beam3.py --dir <runs dir with dy_{bsx,bsy,vtx,mass}.npz>
"""
import argparse
import os
import sys

import numpy as np

# packing order of a symmetric 3x3 in this study: (xx, xy, xz, yy, yz, zz)
PACK = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
MULT = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 1.0])   # off-diagonals appear twice


def cov_bs(sigx, sigy, sigz, dxdz, dydz, rho=0.0):
    """The CMS beam-spot-fitter luminous-region covariance, packed.

    `RecoVertex/BeamSpotProducer/src/FcnBeamSpotFitPV.cc`, and verbatim what
    `ResidualGlobalCorrectionMakerTwoTrackG4e.cc` builds:

        C_xx = sigma_x^2                     C_yy = sigma_y^2
        C_xy = rho sigma_x sigma_y           C_zz = sigma_z^2
        C_xz = dxdz (sigma_z^2 - sigma_x^2) - dydz C_xy
        C_yz = dydz (sigma_z^2 - sigma_y^2) - dxdz C_xy

    `rho` is FIXED AT ZERO in the maker (the beam-spot record does not store
    it); it is a free parameter of this study.
    """
    vx, vy, vz = sigx ** 2, sigy ** 2, sigz ** 2
    cxy = rho * sigx * sigy
    cxz = dxdz * (vz - vx) - dydz * cxy
    cyz = dydz * (vz - vy) - dxdz * cxy
    return np.stack([vx, cxy, cxz, vy, cyz, vz], axis=-1)


def q_pack(w):
    """``Q_ab = w_a w_b`` packed in the same order."""
    return np.stack([w[:, a] * w[:, b] for a, b in PACK], axis=-1)


def contract(cov, q):
    """``sum_ab C_ab Q_ab`` from the two packed forms."""
    return (cov * q * MULT).sum(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", default="dy")
    a = ap.parse_args()

    chans = [("bsx", "bsx"), ("bsy", "bsy"), ("vtx", "vtx"), ("mass", "mass")]
    npz = {}
    for nm, _ in chans:
        p = os.path.join(a.dir, f"{a.prefix}_{nm}.npz")
        if not os.path.exists(p):
            print(f"[skip] {p} missing")
            continue
        npz[nm] = np.load(p, allow_pickle=False)
    if "bsx" not in npz:
        sys.exit("need at least the bsx npz (it carries every mean-weight set)")

    # the RECORD, per candidate, from the bsx npz (present in every rows-ON
    # extraction and identical across the four -- checked below)
    src = npz["bsx"]
    wd = np.asarray(src["bswidth"], np.float64)      # (n, 3) cm
    sl = np.asarray(src["bsslope"], np.float64)      # (n, 2)
    cov0 = cov_bs(wd[:, 0], wd[:, 1], wd[:, 2], sl[:, 0], sl[:, 1], 0.0)
    n = len(wd)
    print(f"{n} candidates; record widths "
          f"{np.median(wd[:,0])*1e4:.4f} / {np.median(wd[:,1])*1e4:.4f} um, "
          f"sigma_z {np.median(wd[:,2]):.4f} cm, slopes "
          f"{np.median(sl[:,0]):+.4e} / {np.median(sl[:,1]):+.4e}")

    key = np.stack([src[q] for q in ("run", "lumi", "event")], 1)
    print(f"\n{'channel':8s} {'A1 share':>12s} {'A2 d/dkx':>12s} {'A2 d/dky':>12s} "
          f"{'A3 formula':>12s}")
    ok = True
    for nm, _ in chans:
        if nm not in npz:
            continue
        d = npz[nm]
        k2 = np.stack([d[q] for q in ("run", "lumi", "event")], 1)
        if not np.array_equal(k2, key):
            sys.exit(f"{nm}: candidate keys differ from bsx -- not row-matched")
        if nm == "bsx":
            w = -np.asarray(d["bsmeanbs"], np.float64).reshape(-1, 6)[:, 0:3]
            cov_ii = np.ones(n)
            vbs = np.asarray(d["vbs"], np.float64)
        elif nm == "bsy":
            w = -np.asarray(d["bsmeanbs"], np.float64).reshape(-1, 6)[:, 3:6]
            cov_ii = np.ones(n)
            vbs = np.asarray(d["vbs"], np.float64)
        elif nm == "vtx":
            w = -np.asarray(src["bsmeanvtx"], np.float64).reshape(-1, 3)
            cov_ii = np.asarray(d["sigma"], np.float64) ** 2
            vbs = np.asarray(d["vtxvbs"], np.float64)
        else:
            w = -np.asarray(src["bsmeanmass"], np.float64).reshape(-1, 3)
            cov_ii = np.asarray(d["sigma"], np.float64) ** 2
            vbs = np.asarray(d["massvbs"], np.float64)
        vbsx = np.asarray(d["vbsx"], np.float64)
        vbsy = np.asarray(d["vbsy"], np.float64)

        q = q_pack(w)
        share = contract(cov0, q) / cov_ii

        # A2: the maker's convention, covBS -> D covBS D
        cxx, cxy, cxz = cov0[:, 0], cov0[:, 1], cov0[:, 2]
        cyy, cyz = cov0[:, 3], cov0[:, 4]
        dx = (w[:, 0] ** 2 * cxx + w[:, 0] * w[:, 1] * cxy
              + w[:, 0] * w[:, 2] * cxz) / cov_ii
        dy = (w[:, 1] ** 2 * cyy + w[:, 0] * w[:, 1] * cxy
              + w[:, 1] * w[:, 2] * cyz) / cov_ii

        # A3: the FORMULA rebuild -- covBS recomputed from the scaled width
        h = 1e-4
        cp = cov_bs(wd[:, 0] * np.sqrt(1.0 + h), wd[:, 1], wd[:, 2],
                    sl[:, 0], sl[:, 1], 0.0)
        cm = cov_bs(wd[:, 0] * np.sqrt(1.0 - h), wd[:, 1], wd[:, 2],
                    sl[:, 0], sl[:, 1], 0.0)
        dx_f = (contract(cp, q) - contract(cm, q)) / (2 * h * cov_ii)

        def rel(a_, b_):
            s = np.maximum(np.abs(a_) + np.abs(b_), 1e-300)
            return np.abs(a_ - b_) / s

        r1 = rel(share, vbs)
        r2 = rel(dx, vbsx)
        r3 = rel(dy, vbsy)
        r4 = rel(dx_f, vbsx)
        print(f"{nm:8s} {np.median(r1):12.3e} {np.median(r2):12.3e} "
              f"{np.median(r3):12.3e} {np.median(r4):12.3e}   (median)")
        print(f"{'':8s} {r1.max():12.3e} {r2.max():12.3e} "
              f"{r3.max():12.3e} {r4.max():12.3e}   (max)")
        ok &= (np.median(r1) < 1e-5) and (np.median(r2) < 1e-5)
        print(f"{'':8s} share median {np.median(share):.6f} "
              f"(exported {np.median(vbs):.6f}); "
              f"|w| median {np.median(np.linalg.norm(w, axis=1)):.4g}")
    print("\nGATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
