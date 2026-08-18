#!/usr/bin/env python3
"""Finite-difference closure on H = curv2localJacobianAltelossD.

WHY
---
Switching the closure to the H basis (`hbasis.py`) flattens locx, locy, dxdz
and dydz on the pT = 3 toy but BREAKS qop: -0.0110 (mu-), -0.0082 (mu+),
+0.0412 (p), +0.0403 (pbar) at the outermost plane, against ~0.000 in the
legacy basis.  That is charge-EVEN (so it is not a q sign bug) and strongly
species-dependent.  It is isolated to H's energy-loss term: with `dedx = 0` the
width ratio sigma(H^T e_0)/sigma(e_0) is exactly 1.00000 on every plane.

So the question is narrow: is H's alteloss column right?  This answers it
without the closure and without Geant4, by differencing the exact nonlinear map
that H is the linearization OF.

THE MAP
-------
H relates a curvilinear deviation at the reference point to the local deviation
at the module plane.  The reference point lies ON that plane, so the required
arc length is O(delta) and the map is closed-form to the order that matters:

    start   qop = qop0 + dqop,  W = W(lam0+dlam, phi0+dphi),
            r   = r0 + dxT u_T + dyT v_T
    reach   s   = -((r - r0).I1) / (W.I1)          [first order in delta]
    bend    W_s = W + s (q/p) (W x B)              [O(delta), KEPT]
    lose    qop_s = qop + s dqop/ds,
            dqop/ds = -q E |qop|^3 dE/ds           [O(delta), KEPT -- THE TERM]
    read    x = (r + W s - r0).J1,  y = (...).K1,
            dxdz = (W_s.J1)/(W_s.I1),  dydz = (W_s.K1)/(W_s.I1)

Everything dropped is O(delta^2) and cancels in a central difference.

Curvilinear axes, CMS convention, in the phi0 = 0 gauge `frames()` returns:
    u_T = (-sin phi, cos phi, 0)                        -> (0, 1, 0)
    v_T = (-cos phi sin lam, -sin phi sin lam, cos lam) -> (-sin lam, 0, cos lam)
    u_T x v_T = W0, i.e. (u_T, v_T, W0) is right handed.

UNITS.  `curv2local` is handed Bvec = B[T] * K_TESLA_TO_INVGEV, which makes
|qop| |B| = 1/R_3D with lengths in cm; checked here against pT/(0.3 B).
`dEdxlast` is GeV/cm and NEGATIVE (energy decreasing) -- on the layered toy it
is -0.01734, which is the dE/dx of the DENSE layer material (rho = 9.0), not a
leg average, and it is the right quantity because the extra path to the module
plane is traversed in the material AT the plane.

HOW TO READ THE OUTPUT
----------------------
The whole 5x5 is compared, not just the alteloss column.  Agreement on the
geometric entries is what VALIDATES the finite difference; a disagreement
confined to one column then localizes the defect.  A sign convention I guessed
wrong would show as a clean sign flip on a whole column or row, which is
diagnostic rather than ambiguous.

usage:
    python curv2local_fd.py --pdg 13 --planes 0 6 13
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import curv2local as c2l                                        # noqa: E402
import cf_propagation_test as cpt                               # noqa: E402

CURV = ("qop", "lambda", "phi", "xT", "yT")
LOC = ("qop", "dxdz", "dydz", "x", "y")


def curv_axes(lam, phi):
    u = np.array([-np.sin(phi), np.cos(phi), 0.0])
    v = np.array([-np.cos(phi) * np.sin(lam), -np.sin(phi) * np.sin(lam),
                  np.cos(lam)])
    w = np.array([np.cos(lam) * np.cos(phi), np.cos(lam) * np.sin(phi),
                  np.sin(lam)])
    return u, v, w


def local_state(d, fr, qop0, q, dEdx, mass, Bvec):
    """The exact map, at curvilinear deviation d = (dqop, dlam, dphi, dxT, dyT).

    Returns the LOCAL 5-vector at the module-plane crossing, relative to the
    module origin (which is the reference point).
    """
    dqop, dlam, dphi, dxT, dyT = d
    I1, J1, K1 = fr["I1"], fr["J1"], fr["K1"]
    lam0, phi0 = fr["lam0"], fr["phi0"]

    u_T, v_T, _ = curv_axes(lam0, phi0)
    _, _, W = curv_axes(lam0 + dlam, phi0 + dphi)
    qop = qop0 + dqop
    dr = dxT * u_T + dyT * v_T

    # arc length to the module plane
    s = -float(dr @ I1) / float(W @ I1)

    # the two O(s) physics terms
    W_s = W + s * qop * np.cross(W, Bvec)
    p = 1.0 / abs(qop)
    E = np.sqrt(mass ** 2 + p ** 2)
    dqopds = -q * E * abs(qop) ** 3 * dEdx
    qop_s = qop + s * dqopds

    r_c = dr + W * s
    return np.array([qop_s - qop0,
                     float(W_s @ J1) / float(W_s @ I1),
                     float(W_s @ K1) / float(W_s @ I1),
                     float(r_c @ J1), float(r_c @ K1)])


def H_fd(fr, qop0, q, dEdx, mass, Bvec, steps=None):
    """Central-difference Jacobian of `local_state`, d(local)/d(curvilinear)."""
    if steps is None:
        # scaled to each parameter's own size; the map is linear to O(delta^2)
        # so the answer is insensitive over orders of magnitude -- `--scan`
        # shows that explicitly.
        steps = np.array([1e-6 * abs(qop0), 1e-7, 1e-7, 1e-5, 1e-5])
    H = np.zeros((5, 5))
    for j in range(5):
        e = np.zeros(5)
        e[j] = steps[j]
        hi = local_state(e, fr, qop0, q, dEdx, mass, Bvec)
        lo = local_state(-e, fr, qop0, q, dEdx, mass, Bvec)
        H[:, j] = (hi - lo) / (2 * steps[j])
    return H


def compare(legs, k, mass, bfield=3.8):
    leg = legs[k]
    fr = c2l.frames(leg, sinlam_sign=c2l._sinlam_sign(legs, k),
                    uz_sign=c2l.uz_sign_from_zoff(leg))
    q = np.sign(leg["refqop"]) or 1.0
    dEdx = leg.get("dEdxlast", 0.0)
    Bvec = np.array([0.0, 0.0, bfield * c2l.K_TESLA_TO_INVGEV])
    Hp = c2l.curv2local(fr, leg["refqop"], q, dEdx, mass, Bvec)
    Hf = H_fd(fr, leg["refqop"], q, dEdx, mass, Bvec)
    # FREE VALIDATION of the frame and the map: at zero deviation the local
    # state must reproduce the export's own reference direction, and the
    # position and q/p deviations must be identically zero (the reference point
    # IS the module origin, and it needs no arc length to reach the plane).
    ref = local_state(np.zeros(5), fr, leg["refqop"], q, dEdx, mass, Bvec)
    chk = dict(dxdz=(ref[1], leg["refdxdz"]), dydz=(ref[2], leg["refdydz"]),
               qop=(ref[0], 0.0), x=(ref[3], 0.0), y=(ref[4], 0.0))
    return Hp, Hf, fr, dEdx, chk


def _fmt(H, ref):
    sc = max(np.abs(ref).max(), 1e-30)
    out = []
    for i in range(5):
        out.append("  " + f"{LOC[i]:>6} " + "".join(
            f"{H[i, j]:14.6g}" for j in range(5)))
    return "\n".join(out), sc


def cmd_main(args):
    import radoff_species as rs
    import hadron_probe as hp

    mass = hp.SPECIES[args.pdg]["mass"] * 1e-3
    path = rs.mp(args.pdg, True)
    legs = cpt.load_model(path)
    c2l.attach_extras(legs, path)
    print(f"{hp.SPECIES[args.pdg]['label']}  {os.path.basename(path)}  "
          f"mass {mass:.6f} GeV  {len(legs)} planes")

    # unit check, stated rather than assumed
    qop0 = legs[0]["refqop"]
    B = 3.8 * c2l.K_TESLA_TO_INVGEV
    print(f"units: 1/(|qop| B) = {1/(abs(qop0)*B):8.3f} cm   vs   "
          f"p/(0.3 B) = {(1/abs(qop0))/(0.003*3.8)*1:8.3f} cm")

    worst = {}
    for k in args.planes:
        Hp, Hf, fr, dEdx, chk = compare(legs, k, mass)
        sc = np.abs(Hp).max()
        rel = np.abs(Hf - Hp) / sc
        print(f"\n--- plane {k}   dEdxlast {dEdx:+.5f} GeV/cm   "
              f"incidence cos = {float(fr['I1'] @ fr['W0']):.5f}")
        print("  map at zero deviation vs the export: " + "  ".join(
            f"{n} {a:+.6f}/{b:+.6f}" for n, (a, b) in chk.items()))
        print("      " + "".join(f"{c:>14}" for c in CURV))
        for i in range(5):
            print(f"  {LOC[i]:>6} " + "".join(
                f"{Hp[i, j]:14.6g}" for j in range(5)) + "   production")
            print(f"  {'':>6} " + "".join(
                f"{Hf[i, j]:14.6g}" for j in range(5)) + "   finite diff")
            print(f"  {'':>6} " + "".join(
                f"{rel[i, j]:14.2e}" for j in range(5)) + "   |diff|/max|H|")
        worst[k] = float(rel.max())
        # the alteloss column specifically: row 0, columns xT and yT
        al = np.array([Hp[0, 3], Hp[0, 4]])
        alf = np.array([Hf[0, 3], Hf[0, 4]])
        print(f"  ALTELOSS entries H[qop, xT/yT]: production {al}, "
              f"finite diff {alf}")
        if np.abs(al).max() > 0:
            print(f"      ratio fd/production = "
                  f"{np.where(np.abs(al) > 0, alf / np.where(al == 0, 1, al), np.nan)}")
    print("\nworst relative entry mismatch per plane:")
    for k, v in worst.items():
        print(f"  plane {k:3d}   {v:.3e}")


def cmd_scan(args):
    """Step-size independence: a correct central difference is flat in delta."""
    import radoff_species as rs
    import hadron_probe as hp
    mass = hp.SPECIES[args.pdg]["mass"] * 1e-3
    path = rs.mp(args.pdg, True)
    legs = cpt.load_model(path)
    c2l.attach_extras(legs, path)
    leg = legs[args.planes[0]]
    fr = c2l.frames(leg, sinlam_sign=c2l._sinlam_sign(legs, args.planes[0]),
                    uz_sign=c2l.uz_sign_from_zoff(leg))
    q = np.sign(leg["refqop"]) or 1.0
    Bvec = np.array([0.0, 0.0, 3.8 * c2l.K_TESLA_TO_INVGEV])
    Hp = c2l.curv2local(fr, leg["refqop"], q, leg["dEdxlast"], mass, Bvec)
    print(f"plane {args.planes[0]}: H[qop,xT] production = {Hp[0, 3]:.8g}")
    for e in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        st = np.array([1e-6 * abs(leg["refqop"]), 1e-7, 1e-7, e, e])
        Hf = H_fd(fr, leg["refqop"], q, leg["dEdxlast"], mass, Bvec, steps=st)
        print(f"  delta_xT = {e:8.1e}   fd = {Hf[0, 3]:.8g}")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdg", type=int, default=13)
    p.add_argument("--planes", type=int, nargs="+", default=[0, 6, 13])
    p.add_argument("--scan", action="store_true", help="step-size scan only")
    a = p.parse_args()
    (cmd_scan if a.scan else cmd_main)(a)


if __name__ == "__main__":
    main()
