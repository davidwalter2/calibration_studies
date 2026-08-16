#!/usr/bin/env python3
"""Clean-propagation closure in DIRECTIONS of the 5D track state, not just two axes.

WHY
---
`qop` and `locx` are two directions in ONE 5D space, and the model CF
(`cf_propagation_test.model_phi`) is already 5D-aware -- per-step noise is
transported and THEN projected.  By Cramer-Wold a multivariate law is fixed by
its 1D projections, so closure in enough directions tests the 5D object using
only the existing 1D machinery.  At present the model is validated in two of
five axis directions and zero off-axis, and a 5x5 block is about to be built on
it.

WHAT IS ADDED
-------------
1. `locy` -- the NON-BENDING position.  MS is isotropic, so locy and locx can
   differ only through transport; a discrepancy localises something
   bending-plane-specific rather than the MS model.
2. `dxdz`, `dydz` -- the ANGLES.  Position is the transported integral of the
   angular kicks, so testing angles directly separates "is the kick
   distribution right" from "is the transport right", which are entangled in
   the locx result.
3. MIXED directions -- the only probe of the CORRELATIONS, which is what a 5x5
   needs and what no axis-aligned test can see.  Eigen-directions of the
   model's own predicted covariance (best conditioned choice), a few
   interpretable 2D mixes, and random directions as a cross-check.

THE BASIS FIX THIS NEEDED
-------------------------
The sim residuals are LOCAL, the exported Q/F/jacc are CURVILINEAR, and
`cf_propagation_test.py` has carried that mismatch as a known defect since
2026-08-06.  `curv2local.py` rebuilds the production Jacobian
`H = curv2localJacobianAltelossD` offline, so a direction is specified in the
local basis and pushed to the curvilinear one as `H^T a`.  `--no-h` restores
the legacy raw a-vectors and reproduces every published number exactly; that
is the control.

The mismatch is NOT a harmless overall scale.  The data side is a fixed
physical residual; only the MODEL's predicted width moves, so the ratio
`sigma(H^T e_i)/sigma(e_i)` is a direct width error.  Measured here:

    qop   1.000 - 1.029 (real), 1.000 - 1.142 (dense layered toy)
    locx  sec(alpha)      = 1.000 - 1.029
    locy  sec(lambda)     = 1.045
    dxdz, dydz            the legacy a-vector is not even the same VARIABLE
                          (|corr| <= 0.14): local dx/dz is essentially the
                          curvilinear PHI and local dy/dz the curvilinear
                          LAMBDA, because local x is the r-phi direction in a
                          barrel.  The two were swapped.

HOW A DIRECTION IS DEFINED
--------------------------
The five local components have incommensurable units (q/p ~ 1e-4 GeV^-1,
angles ~ 1e-3, positions ~ 0.1 cm), so a direction only means something in the
STANDARDIZED local state

    y = D (x_local - x_ref),   D = diag(1/sqrt(C_local_ii)),
    C_local = H C_curv H^T,    C_curv = sum_j A_j Q_j A_j^T

A unit 5-vector `ahat` in y-space gives `a_local = D ahat` and
`a_curv = H^T a_local`; the model's predicted variance of the projection is
then `ahat^T R ahat` with `R = D C_local D` the predicted CORRELATION matrix.
That number is also the CONDITIONING of the direction: `ahat^T R ahat << 1`
means the model predicts a near-cancellation, so the closure there is a
sensitive function of the correlations and a poor one of the marginals.

STATISTICS
----------
Fisher normalization throughout (`s_F = sigma sqrt(1/I)`, `1/I` by exact FFT
inversion of the model CF -- never the saddlepoint).  Errors are the
per-EVENT correlated estimator of NOTES_GEOMCLOSURE section 1
(`err_plane/sqrt(nplanes)` is 2.8x too small).  DIRECTIONS SHARE EVENTS TOO,
so the same per-event accumulator is kept for every direction and the
direction-to-direction covariance is computed from it; differences between
directions are quoted with THAT error, never with the quadrature sum.

subcommands
    basis    frame reconstruction + H, and its validation.  RUN FIRST.
    cond     the conditioning report
    closure  the closure table across directions and geometries
    figs     the two-panel CF/lineshape figures
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_propagation_test as cpt                                # noqa: E402
import cgf_channels as cc                                        # noqa: E402
import curv2local as c2l                                         # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
from cf_propagation_test import model_phi, weier_scalar          # noqa: E402

SCRATCH = fn.SCRATCH
UCURVE = fn.UCURVE
UHEAD = fn.UHEAD
IHEAD = fn.IHEAD

LOCAL = ("qop", "dxdz", "dydz", "locx", "locy")
REF = ("refqop", "refdxdz", "refdydz", "reflocx", "reflocy")

# The two configurations that matter (NOTES_GEOMCLOSURE): the layered toy at
# NSUB = 1 -- the toy that lands ON the detector for the position channel --
# and the real tracker.  realgeo_*/rg_* are NOT sim-matched and are not used.
GEOMS = {
    "lay1": dict(key="H_layK1", label="layered toy, 1 mm shells, NSUB=1",
                 short="lay1", nev="100k"),
    "real": dict(key="real", label="real tracker, 19 barrel modules",
                 short="real", nev="200k"),
}

# Published Fisher/u=1 ladder means, to be reproduced before anything new is
# read off the machinery (NOTES_GEOMCLOSURE sections 2 and 3).
REFERENCE = {
    "lay1": dict(locx=(+0.00002, 0.00083), qop=(+0.00081, 0.00034)),
    "real": dict(locx=(+0.00138, 0.00058), qop=(+0.01148, 0.00027)),
}

_MODEL = {}


# ==========================================================================
# model / sim, with the extra branches H needs
# ==========================================================================

def model(g):
    if g not in _MODEL:
        d = gc.GEOMS[GEOMS[g]["key"]]
        p = d["model"] if os.path.sep in d["model"] else os.path.join(
            SCRATCH, d["model"])
        legs = cpt.load_model(p)
        c2l.attach_extras(legs, p)
        _MODEL[g] = legs
    return _MODEL[g]


def sim(g):
    return gc.geom_sim(GEOMS[g]["key"])


def cov_curv(legs, k):
    """C_curv = sum_j A_j Q_j A_j^T, A_j = F_k ... F_{j+1}.

    Its diagonal reproduces `cf_propagation_test.model_variance` exactly for
    every a-vector -- checked in `cmd_basis`, not assumed.
    """
    suf = [np.eye(5)]
    for m in range(k, 0, -1):
        suf.append(suf[-1] @ legs[m]["F"])
    suf = suf[::-1]
    C = np.zeros((5, 5))
    for j in range(k + 1):
        C += suf[j] @ legs[j]["Q"] @ suf[j].T
    return C


def plane_geom(legs, k, useH=True, orient=True, **hkw):
    """H, C_curv, C_local, D (standardize+ORIENT) and R for one plane.

    THE ORIENTATION MATTERS AND IT IS NOT COSMETIC.  A module's local frame
    carries a sign convention that has nothing to do with physics: whether
    local x runs along +r-phi or -r-phi, and which side of the sensor the track
    enters from, both flip freely down the ladder (on the real pT = 3 ray the
    local-z momentum sign changes 8 times in 19 modules).  The sign of a SINGLE
    component is irrelevant -- the closure statistic is even in z -- but the
    RELATIVE sign inside a MIXED direction is not: "locx + dxdz" is the
    near-cancelling combination on one module and the reinforcing one on the
    next, and averaging it down the ladder averages two different tests.

    The fix is to orient each standardized local component onto its curvilinear
    partner, using the sign of the dominant entry of H:

        qop <- qop,  dxdz <- phi,  dydz <- lambda,  locx <- xT,  locy <- yT

    (dx/dz pairs with PHI and dy/dz with LAMBDA, not the other way round: local
    x is the r-phi direction in a barrel.)  After this a mixed direction means
    the same physical combination on every plane.  Axis directions and the
    per-plane eigen-directions are unaffected.
    """
    C = cov_curv(legs, k)
    if useH:
        H, fr = c2l.leg_H(legs, k, **hkw)
    else:
        H, fr = np.eye(5), None
    Cl = H @ C @ H.T
    d = np.sqrt(np.diag(Cl))
    s = np.ones(5)
    if orient:
        for i, j in ((0, 0), (1, 2), (2, 1), (3, 3), (4, 4)):
            s[i] = np.sign(H[i, j]) or 1.0
    D = np.diag(s / d)
    return dict(H=H, fr=fr, C=C, Cl=Cl, D=D, R=D @ Cl @ D, sd=d, orient=s)


# ==========================================================================
# directions
# ==========================================================================

AXIS = {n: np.eye(5)[i] for i, n in enumerate(LOCAL)}

# Interpretable 2D mixes.  The (locx, dxdz) pair is the sharp one: the model
# predicts rho ~ -0.9 there, so the SUM direction has a predicted standardized
# variance of ~0.1 -- a near-cancellation that no axis test can see and that
# only survives if the correlation is right.  Both signs are kept: the
# well-conditioned partner is the control that says the sample is fine.
def _mix(i, j, s):
    v = np.zeros(5)
    v[i], v[j] = 1.0, float(s)
    return v / np.linalg.norm(v)


MIX = {
    "locx+dxdz": _mix(3, 1, +1), "locx-dxdz": _mix(3, 1, -1),
    "locy+dydz": _mix(4, 2, +1), "locy-dydz": _mix(4, 2, -1),
    "qop+locx":  _mix(0, 3, +1), "qop-locx":  _mix(0, 3, -1),
    "locx+locy": _mix(3, 4, +1),
}

_RNG = np.random.default_rng(20260815)
RAND = {}
for _i in range(3):
    _v = _RNG.normal(size=5)
    RAND[f"rnd{_i + 1}"] = _v / np.linalg.norm(_v)

EIGN = [f"eig{i + 1}" for i in range(5)]
ALLDIRS = list(AXIS) + list(MIX) + EIGN + list(RAND)


def direction(name, pg):
    """The standardized-space unit vector `ahat` for a direction name.

    Eigen-directions are PER PLANE (they are eigenvectors of that plane's own
    predicted correlation matrix R), ordered by eigenvalue descending; every
    other direction is the same 5-vector on every plane.  The sign of `ahat` is
    irrelevant -- the statistic depends on z^2.
    """
    if name in AXIS:
        return AXIS[name]
    if name in MIX:
        return MIX[name]
    if name in RAND:
        return RAND[name]
    if name.startswith("eig"):
        w, V = np.linalg.eigh(pg["R"])
        return V[:, ::-1][:, int(name[3:]) - 1]
    raise KeyError(name)


def avecs(name, pg):
    """(ahat, a_local, a_curv, sigma, cond) for one direction on one plane."""
    ahat = direction(name, pg)
    a_local = pg["D"] @ ahat
    a_curv = pg["H"].T @ a_local
    v = float(ahat @ pg["R"] @ ahat)          # predicted standardized variance
    return ahat, a_local, a_curv, np.sqrt(max(v, 0.0)), v


# ==========================================================================
# scales (parallel: the FFT inversion is 15 s per plane on the real geometry)
# ==========================================================================

_W = {}


def _init(g, useH, hkw):
    _W["legs"] = model(g)
    _W["useH"] = useH
    _W["hkw"] = hkw


def _scale_task(arg):
    k, name = arg
    legs = _W["legs"]
    pg = plane_geom(legs, k, useH=_W["useH"], **_W["hkw"])
    ahat, a_local, a_curv, sigma, v = avecs(name, pg)
    # cross-check: the covariance route and the propagator's own leg-Q route
    sig2 = np.sqrt(cpt.model_variance(legs, k, a_curv)[0])
    z, p, dp = cc.exact_density(legs, k, a_curv, sigma)
    I, info = cc.fisher_exact(z, p, dp, floor=1e-8)
    sF = sigma * np.sqrt(1.0 / I)
    tau = fn.closure_tau(float(np.max(UCURVE)))
    phi = model_phi(legs, k, a_curv, sF, tau)
    mvals = np.array([weier_scalar(phi, u, tau) for u in UCURVE])
    return dict(k=k, name=name, ahat=ahat, a_local=a_local, a_curv=a_curv,
                sigma=sigma, sig_check=sig2, invI=1.0 / I, sF=sF, cond=v,
                mass=info.get("mass_frac", np.nan), mvals=mvals)


def model_side(g, names, useH=True, nproc=48, hkw=None):
    """The model half of the closure for every (plane, direction)."""
    import multiprocessing as mp
    legs = model(g)
    tasks = [(k, n) for k in range(len(legs)) for n in names]
    hkw = hkw or {}
    if nproc <= 1:
        _init(g, useH, hkw)
        out = [_scale_task(t) for t in tasks]
    else:
        with mp.Pool(nproc, initializer=_init, initargs=(g, useH, hkw)) as pool:
            out = pool.map(_scale_task, tasks, chunksize=1)
    res = {}
    for r in out:
        res[(r["k"], r["name"])] = r
    return res


# ==========================================================================
# data side + closure, with the per-event accumulator
# ==========================================================================

def closure(g, names, ms, nplane=None):
    """Per-plane closure and the per-EVENT accumulator for every direction.

    Returns rows[name] = (nplane, nprobe), and cev[name] = (nprobe, nevent),
    the per-event ladder average of e^{-u z^2}.  The accumulator is what makes
    a DIFFERENCE between two directions quotable: they are evaluated on the
    same events and their statistics are strongly correlated.
    """
    legs, s = model(g), sim(g)
    nl = min(s["valid"].shape[1], len(legs)) if nplane is None else nplane
    nev = s["valid"].shape[0]
    resid = np.stack([s[b] - np.array([l[r] for l in legs])[None, :nl]
                      for b, r in zip(LOCAL, REF)], axis=-1)      # (nev,nl,5)
    ok = s["valid"][:, :nl] & np.isfinite(resid).all(axis=-1)

    rows, cev, ks, ns = {}, {}, [], {}
    for name in names:
        R = np.zeros((nl, len(UCURVE)))
        acc = np.zeros((len(UCURVE), nev))
        kept, nk = [], []
        for k in range(nl):
            good = ok[:, k]
            if good.sum() < 100:
                continue
            r = ms[(k, name)]
            z = (resid[good, k, :] @ r["a_local"]) / r["sF"]
            e = np.exp(-UCURVE[:, None] * z[None, :] ** 2)
            R[k] = e.mean(axis=1) - r["mvals"]
            acc[:, good] += e * (nev / good.sum())
            kept.append(k)
            nk.append(int(good.sum()))
        acc /= max(len(kept), 1)
        rows[name] = R[kept]
        cev[name] = acc
        ks, ns[name] = np.array(kept), np.array(nk)
    return rows, cev, ks, ns


def errs_from_cev(cev, names):
    """(err[name], corr matrix between directions) at every probe."""
    nev = cev[names[0]].shape[1]
    err = {n: cev[n].std(axis=1) / np.sqrt(nev) for n in names}
    C = np.zeros((len(UCURVE), len(names), len(names)))
    for iu in range(len(UCURVE)):
        M = np.stack([cev[n][iu] for n in names])
        C[iu] = np.corrcoef(M)
    return err, C


def diff_err(cev, n1, n2, iu):
    """Error on (closure_{n1} - closure_{n2}), accounting for shared events."""
    d = cev[n1][iu] - cev[n2][iu]
    return float(d.std() / np.sqrt(len(d)))


# ==========================================================================
# subcommand: basis
# ==========================================================================

def cmd_basis(args):
    print("=" * 100)
    print("PREMISE CHECKS ON THE CURVILINEAR -> LOCAL JACOBIAN H")
    print("=" * 100)
    print(__doc__.split("HOW A DIRECTION IS DEFINED")[0].split(
        "THE BASIS FIX THIS NEEDED")[1].strip())
    print()

    from toy_loader import plane_frames
    for g in args.geoms:
        legs = model(g)
        print("-" * 100)
        print(f"### {g}: {GEOMS[g]['label']}, {len(legs)} planes")

        print("\n(1) frame reconstruction: psi = angle of the global z axis in "
              "the module plane.")
        print("    psi = 0 or pi  <=> local y || +-global z  (an rphi barrel "
              "module)")
        print("    a stereo module sits 100 mrad away from either -- NOTHING "
              "here puts that number in.")
        print("    uz = sign of the track's local-z momentum, from the "
              "entry-face offset zoff.")
        print(f"    {'k':>3} {'detid':>10} {'zoff':>9} {'uz':>4} {'psi':>10} "
              f"{'other root':>11} {'|psi-{0,pi}|':>13} {'H33':>9} {'H34':>9} "
              f"{'H43':>9} {'H44':>9}")
        for k in range(len(legs)):
            H, fr = c2l.leg_H(legs, k)
            p = np.mod(fr["psi"], 2 * np.pi)
            dev = min(abs(p), abs(p - np.pi), abs(p - 2 * np.pi))
            other = [r for r in fr["roots"] if abs(r - fr["psi"]) > 1e-9]
            print(f"    {k:3d} {legs[k]['detid']:10d} {legs[k]['zoff']:+9.5f} "
                  f"{c2l.uz_sign_from_zoff(legs[k]):+4.0f} {fr['psi']:+10.5f} "
                  f"{(other[0] if other else np.nan):+11.5f} {dev:13.5f} "
                  f"{H[3,3]:+9.5f} {H[3,4]:+9.5f} {H[4,3]:+9.5f} "
                  f"{H[4,4]:+9.5f}")
        if g == "lay1":
            ns = {}
            exec(open(fn.PLANES).read(), ns)
            org = np.asarray(ns["origin"]).reshape(-1, 3)
            nrm = np.asarray(ns["normal"]).reshape(-1, 3)
            ch = np.r_[[np.dot(org[1] - org[0], nrm[0])],
                       [np.dot(org[j] - org[j - 1], nrm[j])
                        for j in range(1, len(org))]]
            print("    toy uz sign, INDEPENDENTLY from the plane geometry "
                  "(chord . normal, no zoff and no dxdz):")
            print("      " + " ".join(f"{np.sign(v):+.0f}" for v in ch)
                  + f"   -> all +1: {bool(np.all(ch > 0))}")

        print("\n(2) the position block against the closed form for a barrel "
              "module")
        print("    locx = sec(alpha) a ,  locy = tan(alpha) tan(lambda) a + "
              "sec(lambda) b   (alpha = incidence in the bending plane)")
        wr, ws, nst = 0.0, 0.0, 0
        for k in range(len(legs)):
            H, fr = c2l.leg_H(legs, k)
            dx = legs[k]["refdxdz"]
            cl = legs[k]["refpt"] / legs[k]["refp"]
            tl = np.sqrt(1 - cl ** 2) / cl
            e = max(abs(abs(H[3, 3]) - np.sqrt(1 + dx ** 2)),
                    abs(abs(H[4, 4]) - 1 / cl),
                    abs(abs(H[4, 3]) - abs(dx) * tl))
            if abs(H[3, 4]) < 1e-9:                     # rphi module
                wr = max(wr, e)
            else:                                       # stereo module
                ws, nst = max(ws, e), nst + 1
        print(f"    rphi modules   : max deviation from the closed form "
              f"{wr:.2e}   (exact)")
        print(f"    stereo modules : {nst} of {len(legs)}, max deviation "
              f"{ws:.2e}   (the closed form does NOT hold there -- psi != 0, "
              f"and H34 != 0)")

        if g == "lay1":
            print("\n(3) TOY ONLY: H rebuilt from the export alone, against H "
                  "built from the")
            print("    TRUE plane axes in toyPlanes_pt3.py.  The real geometry "
                  "has no such input,")
            print("    so this is the only place the reconstruction can be "
                  "checked against truth.")
            ns = {}
            exec(open(fn.PLANES).read(), ns)
            o, Rm = plane_frames(ns["origin"], ns["normal"], ns["uaxis"])
            worst = 0.0
            for k in range(len(legs)):
                Hr, _ = c2l.leg_H(legs, k)
                N = np.sqrt(1 + legs[k]["refdxdz"] ** 2
                            + legs[k]["refdydz"] ** 2)
                pl = np.array([legs[k]["refdxdz"], legs[k]["refdydz"], 1.0]) / N
                frt = c2l.frames_from_axes(Rm[k][0], Rm[k][1], Rm[k][2],
                                           Rm[k].T @ pl)
                Ht = c2l.curv2local(frt, legs[k]["refqop"],
                                    np.sign(legs[k]["refqop"]),
                                    legs[k]["dEdxlast"], c2l.MU_MASS,
                                    np.array([0., 0., 3.8 * c2l.K_TESLA_TO_INVGEV]))
                worst = max(worst, np.abs(Hr - Ht).max() / np.abs(Ht).max())
            print(f"    worst relative |H_reconstructed - H_true| over "
                  f"{len(legs)} planes: {worst:.2e}")

        print("\n(4) internal consistency: cov_curv() vs the propagator's own "
              "model_variance()")
        w = 0.0
        rng = np.random.default_rng(1)
        for k in range(len(legs)):
            C = cov_curv(legs, k)
            for _ in range(4):
                a = rng.normal(size=5)
                w = max(w, abs(np.sqrt(a @ C @ a)
                               / np.sqrt(cpt.model_variance(legs, k, a)[0]) - 1))
        print(f"    worst relative difference over all planes x 4 random "
              f"a-vectors: {w:.2e}")

        print("\n(5) how much H matters, per direction, and in what way.")
        print("    The legacy test modelled the LOCAL residual v_i = e_i.(x - "
              "x_ref) by the")
        print("    CURVILINEAR variable u_i = e_i.delta_curv.  The correct "
              "model is")
        print("    v_i = (H^T e_i).delta_curv.  Two things can therefore be "
              "wrong:")
        print("      rho   = corr(u_i, v_i) in the model's own metric -- is it "
              "even the same")
        print("              variable?  rho = 1 means yes.")
        print("      ratio = sigma(v_i)/sigma(u_i) -- by how much the model's "
              "predicted WIDTH")
        print("              of the SAME data variable changes.  This does NOT "
              "cancel: the")
        print("              data side is a fixed physical residual and only "
              "the model's")
        print("              predicted width moves, so a 3 % ratio is a 3 % "
              "width error.")
        print(f"    {'dir':<10} {'|rho| min':>10} {'|rho| med':>10} "
              f"{'ratio min':>10} {'ratio max':>10}   {'worst plane':>11}")
        for name in ("qop", "dxdz", "dydz", "locx", "locy"):
            rho, wr = [], []
            for k in range(len(legs)):
                pg = plane_geom(legs, k, useH=True)
                e = AXIS[name]
                C, H = pg["C"], pg["H"]
                sL = np.sqrt(e @ C @ e)
                sH = np.sqrt(e @ pg["Cl"] @ e)
                rho.append(abs(float(e @ H @ C @ e)) / (sL * sH))
                wr.append(sH / sL)
            rho, wr = np.array(rho), np.array(wr)
            print(f"    {name:<10} {rho.min():10.6f} {np.median(rho):10.6f} "
                  f"{wr.min():10.5f} {wr.max():10.5f}   "
                  f"{int(np.argmin(rho)):11d}")

        print("\n(6) the field and dE/dx entering H are NOT exported: B is "
              "taken as the")
        print("    nominal (0,0,3.8 T) and dE/dx as the propagator's "
              "`dEdxlast`.  Sensitivity")
        print("    of the direction (mrad) to a +-2 % field change and to "
              "switching the")
        print("    dE/dx term off entirely:")
        print(f"    {'dir':<10} {'dB=+2%':>10} {'dEdx=0':>10}   "
              f"(outermost plane)")
        k = len(legs) - 1
        pg0 = plane_geom(legs, k, useH=True)
        pgB = plane_geom(legs, k, useH=True, bscale=1.02)
        pgE = plane_geom(legs, k, useH=True, dedx=0.0)
        for name in ("qop", "dxdz", "dydz", "locx", "locy"):
            r = []
            for pg in (pgB, pgE):
                a0 = avecs(name, pg0)[2]
                a1 = avecs(name, pg)[2]
                c = abs(a0 @ a1) / np.linalg.norm(a0) / np.linalg.norm(a1)
                r.append(1e3 * np.arccos(min(c, 1.0)))
            print(f"    {name:<10} {r[0]:10.3f} {r[1]:10.3f}")
        print()


# ==========================================================================
# subcommand: cond
# ==========================================================================

def cmd_cond(args):
    print("=" * 100)
    print("CONDITIONING REPORT")
    print("=" * 100)
    print("For a standardized direction ahat, the model predicts variance "
          "v = ahat^T R ahat")
    print("with R the predicted CORRELATION matrix of the local 5D residual. "
          "v << 1 means")
    print("the model predicts a near-cancellation: the closure there is a "
          "sensitive function")
    print("of the correlations (which is the point) and an ill-conditioned "
          "one (which is the")
    print("risk).  `rot` = |d ln v / d(direction)| in units of 1/rad -- how "
          "fast the predicted")
    print("width moves if the direction is misspecified, e.g. by an error in "
          "H.\n")

    for g in args.geoms:
        legs = model(g)
        nl = len(legs)
        print("-" * 100)
        print(f"### {g}: {GEOMS[g]['label']}")
        pgs = [plane_geom(legs, k, useH=not args.no_h) for k in range(nl)]

        print("\neigenvalues of R (predicted correlation matrix), per plane")
        print(f"  {'k':>3} " + "".join(f"{f'lam{i+1}':>10}" for i in range(5))
              + f"{'cond(R)':>12}")
        for k in (0, nl // 3, nl // 2, 2 * nl // 3, nl - 1):
            w = np.linalg.eigvalsh(pgs[k]["R"])[::-1]
            print(f"  {k:3d} " + "".join(f"{v:10.5f}" for v in w)
                  + f"{w[0] / max(w[-1], 1e-30):12.1f}")

        print("\nmodel-predicted correlation matrix R at the outermost plane "
              "(qop, dxdz, dydz, locx, locy)")
        for r in pgs[-1]["R"]:
            print("   " + "  ".join(f"{v:+7.4f}" for v in r))

        print(f"\nper-direction conditioning "
              f"(v = predicted standardized variance)")
        print(f"  {'dir':<11} {'v(min)':>9} {'v(med)':>9} {'v(max)':>9} "
              f"{'rot(max)':>10}  verdict")
        for name in args.dirs:
            vs, rots = [], []
            for pg in pgs:
                ahat = direction(name, pg)
                v = float(ahat @ pg["R"] @ ahat)
                gvec = 2 * (pg["R"] @ ahat - v * ahat)
                vs.append(v)
                rots.append(np.linalg.norm(gvec) / max(v, 1e-30))
            vs = np.array(vs)
            verdict = ("OK" if vs.min() > 0.3 else
                       ("MARGINAL" if vs.min() > 0.05 else
                        "ILL-CONDITIONED -- flag"))
            print(f"  {name:<11} {vs.min():9.4f} {np.median(vs):9.4f} "
                  f"{vs.max():9.4f} {max(rots):10.2f}  {verdict}")
        print()


# ==========================================================================
# subcommand: closure
# ==========================================================================

def _fmt(v, e=None):
    return f"{v:+9.5f}" + (f" +-{e:8.5f}" if e is not None else "")


def cmd_closure(args):
    names = args.dirs
    out = {}
    print("=" * 100)
    print("CLOSURE ACROSS DIRECTIONS, FISHER NORMALIZATION")
    print("=" * 100)
    print(f"s_F = sigma sqrt(1/I), 1/I by exact FFT inversion of the model CF. "
          f"H {'OFF (legacy)' if args.no_h else 'ON'}.")
    print(f"u probes: {UCURVE}\n")

    for g in args.geoms:
        legs = model(g)
        print("-" * 100)
        print(f"### {g}: {GEOMS[g]['label']}, {len(legs)} planes, "
              f"{GEOMS[g]['nev']} events")
        ms = model_side(g, names, useH=not args.no_h, nproc=args.nproc)
        rows, cev, ks, ns = closure(g, names, ms)
        err, corr = errs_from_cev(cev, names)

        print(f"\n{'dir':<11} {'cond':>7} " +
              "".join(f"{u:>10.3g}" for u in UCURVE))
        for name in names:
            m = rows[name].mean(axis=0)
            v = np.median([ms[(k, name)]["cond"] for k in ks])
            print(f"{name:<11} {v:7.4f} " + "".join(f"{x:+10.5f}" for x in m))
            print(f"{'  +- (corr)':<11} {'':>7} "
                  + "".join(f"{x:10.5f}" for x in err[name]))
            print(f"{'  rms/plane':<11} {'':>7} "
                  + "".join(f"{x:10.5f}" for x in rows[name].std(axis=0)))

        print(f"\nheadline probes  (mean over {len(ks)} planes)")
        print(f"  {'dir':<11} {'cond':>7} " +
              "".join(f"{f'u={u}':>22}" for u in UHEAD) +
              f"{'|.|/err at u=1':>16}")
        for name in names:
            m = rows[name].mean(axis=0)
            v = np.median([ms[(k, name)]["cond"] for k in ks])
            s = "".join(f"{_fmt(m[i], err[name][i]):>22}" for i in IHEAD)
            sig = abs(m[IHEAD[-1]]) / max(err[name][IHEAD[-1]], 1e-12)
            print(f"  {name:<11} {v:7.4f} {s}{sig:16.1f}")

        # controls
        if not args.no_h:
            print("\n  (the qop/locx entries above use H; the published "
                  "reference values are the")
            print("   legacy raw a-vectors -- `--no-h` reproduces them and is "
                  "run separately)")
        else:
            print(f"\n  CONTROL vs the published table (u = 1, Fisher):")
            for name in ("locx", "qop"):
                if name not in names:
                    continue
                m = float(rows[name].mean(axis=0)[IHEAD[-1]])
                e = float(err[name][IHEAD[-1]])
                rm, re = REFERENCE[g][name]
                print(f"    {name:<5} {m:+.5f} +- {e:.5f}   published "
                      f"{rm:+.5f} +- {re:.5f}   pull "
                      f"{(m - rm) / max(e, 1e-12):+.2f}")

        iu = IHEAD[-1]
        print(f"\n  direction-to-direction CORRELATION of the ladder-mean "
              f"statistic at u=1")
        print("  (the same events feed every direction; a naive quadrature "
              "sum of two")
        print("   directions' errors is wrong by this much)")
        print("    " + " " * 11 + "".join(f"{n[:9]:>10}" for n in names))
        for i, n in enumerate(names):
            print(f"    {n:<11}" + "".join(f"{corr[iu][i, j]:10.3f}"
                                           for j in range(len(names))))

        out[g] = dict(rows={n: rows[n] for n in names},
                      err={n: err[n] for n in names}, ks=ks, corr=corr,
                      cond={n: np.array([ms[(k, n)]["cond"] for k in ks])
                            for n in names},
                      sF={n: np.array([ms[(k, n)]["sF"] for k in ks])
                          for n in names},
                      invI={n: np.array([ms[(k, n)]["invI"] for k in ks])
                            for n in names},
                      mass={n: np.array([ms[(k, n)]["mass"] for k in ks])
                            for n in names})
        print()

        if args.perplane:
            print(f"  per-plane closure at u = 1")
            print(f"    {'k':>3} " + "".join(f"{n[:9]:>10}" for n in names))
            for i, k in enumerate(ks):
                print(f"    {k:3d} " + "".join(f"{rows[n][i, iu]:10.5f}"
                                               for n in names))
            print()

    if args.npz:
        d = {}
        for g, r in out.items():
            for n in r["rows"]:
                d[f"{g}|{n}|rows"] = r["rows"][n]
                d[f"{g}|{n}|err"] = r["err"][n]
                d[f"{g}|{n}|cond"] = r["cond"][n]
                d[f"{g}|{n}|sF"] = r["sF"][n]
                d[f"{g}|{n}|invI"] = r["invI"][n]
            d[f"{g}|ks"] = r["ks"]
            d[f"{g}|corr"] = r["corr"]
        d["u"] = UCURVE
        d["names"] = np.array(names)
        os.makedirs(os.path.dirname(args.npz), exist_ok=True)
        np.savez(args.npz, **d)
        print(f"wrote {args.npz}")
    return out


# ==========================================================================
# subcommand: deltas -- H on/off, and direction-to-direction differences
# ==========================================================================

def _q(z, p, q):
    c = np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(z))
    c = np.r_[0.0, c] / c[-1]
    return float(np.interp(q, c, z))


def _winit(g):
    _W["legs"], _W["sim"] = model(g), sim(g)


def _wtask(arg):
    k, name, useH = arg
    legs, s = _W["legs"], _W["sim"]
    pg = plane_geom(legs, k, useH=useH)
    _, a_local, a_curv, sigma, _ = avecs(name, pg)
    z, p, dp = cc.exact_density(legs, k, a_curv, sigma)
    wm = 0.5 * (_q(z, p, 0.84135) - _q(z, p, 0.15865))
    good = s["valid"][:, k] & np.isfinite(s["locx"][:, k])
    d = np.stack([s[b][good, k] - legs[k][r] for b, r in zip(LOCAL, REF)],
                 axis=-1)
    zz = (d @ a_local) / sigma
    wd = 0.5 * (np.percentile(zz, 84.135) - np.percentile(zz, 15.865))
    return float(wd / wm)


def cmd_width(args):
    """Direct, closure-independent check of the H scale.

    The closure statistic is only ONE functional of the standardized residual,
    so a width error and a shape error enter it together.  This compares the
    WIDTH alone: the central 68.27 % half-width of the model's own exact
    density against the same quantity measured on the Geant4 sample.  It uses
    no probe, no Weierstrass transform and no Fisher information -- just the
    inverted model density and the data's percentiles -- so it is an
    independent statement about the basis, and the prediction is sharp:
    without H the ratio must track sec(alpha) for locx and sec(lambda) for
    locy, with H it must be 1.
    """
    for g in args.geoms:
        legs, s = model(g), sim(g)
        print("=" * 100)
        print(f"### {g}: {GEOMS[g]['label']}")
        print("  central 68.27 % half-width, data / model, in the model's own "
              "units")
        print(f"  {'k':>3} {'sec(a)':>8} {'sec(l)':>8} | "
              + " | ".join(f"{n:>8} {'noH':>7} {'withH':>7}"
                           for n in args.dirs))
        import multiprocessing as mp
        tasks = [(k, n, u) for k in range(len(legs)) for n in args.dirs
                 for u in (False, True)]
        with mp.Pool(args.nproc, initializer=_winit,
                     initargs=(g,)) as pool:
            res = dict(zip(tasks, pool.map(_wtask, tasks, chunksize=1)))
        agg = {n: ([], []) for n in args.dirs}
        for k in range(len(legs)):
            H, _ = c2l.leg_H(legs, k)
            row = ""
            for name in args.dirs:
                v0, v1 = res[(k, name, False)], res[(k, name, True)]
                agg[name][0].append(v0)
                agg[name][1].append(v1)
                row += f" | {'':>8} {v0:7.4f} {v1:7.4f}"
            print(f"  {k:3d} {abs(H[3,3]):8.5f} {abs(H[4,4]):8.5f}" + row)
        print("\n  RMS of |data/model - 1| over planes")
        print(f"  {'dir':<10} {'no H':>10} {'with H':>10}")
        for name in args.dirs:
            a = np.array(agg[name][0]) - 1
            b = np.array(agg[name][1]) - 1
            print(f"  {name:<10} {np.sqrt((a**2).mean()):10.5f} "
                  f"{np.sqrt((b**2).mean()):10.5f}")
        print()


def cmd_rho(args):
    """Localise a mixed-direction residual: the model's predicted correlation
    against the data's, WITHOUT using the closure statistic.

    For an oriented standardized pair (A, B) with model correlation rho_m, the
    four widths (A, B, (A+B)/sqrt2, (A-B)/sqrt2) over-determine the data's
    correlation.  Writing the data/model width ratios as a, b, r+, r-,

        rho_d = [ r+^2 (1 + rho_m) - r-^2 (1 - rho_m) ] / (2 a b)

    This uses only central 68.27 % half-widths -- no probe, no Weierstrass
    transform, no Fisher information -- so it is independent of the closure
    table, and it is far less tail-sensitive than a raw Pearson correlation of
    a Moliere-tailed residual (which is quoted alongside as a sanity check, not
    as the measurement).  The half-width is a variance proxy, exact only for a
    Gaussian; for the MS-dominated components used here that is good to the
    percent level and it is stated rather than assumed.
    """
    import multiprocessing as mp
    for g in args.geoms:
        legs, s = model(g), sim(g)
        print("=" * 100)
        print(f"### {g}: {GEOMS[g]['label']}")
        for A, B, P, M in (("locx", "dxdz", "locx+dxdz", "locx-dxdz"),
                           ("locy", "dydz", "locy+dydz", "locy-dydz")):
            names = [A, B, P, M]
            tasks = [(k, n, True) for k in range(len(legs)) for n in names]
            with mp.Pool(args.nproc, initializer=_winit,
                         initargs=(g,)) as pool:
                res = dict(zip(tasks, pool.map(_wtask, tasks, chunksize=1)))
            print(f"\n  ({A}, {B})")
            print(f"  {'k':>3} {'rho model':>10} {'rho data':>10} "
                  f"{'diff':>9} {'rho Pearson':>12} {'a':>7} {'b':>7} "
                  f"{'r+':>7} {'r-':>7}")
            dif = []
            for k in range(len(legs)):
                pg = plane_geom(legs, k)
                i, j = LOCAL.index(A), LOCAL.index(B)
                rm = float(pg["R"][i, j])
                a, b = res[(k, A, True)], res[(k, B, True)]
                rp, rmm = res[(k, P, True)], res[(k, M, True)]
                rd = (rp ** 2 * (1 + rm) - rmm ** 2 * (1 - rm)) / (2 * a * b)
                good = s["valid"][:, k] & np.isfinite(s["locx"][:, k])
                d = np.stack([(s[A][good, k] - legs[k][REF[i]]) * pg["D"][i, i],
                              (s[B][good, k] - legs[k][REF[j]]) * pg["D"][j, j]])
                pe = float(np.corrcoef(d)[0, 1])
                dif.append(rd - rm)
                print(f"  {k:3d} {rm:10.4f} {rd:10.4f} {rd - rm:+9.4f} "
                      f"{pe:12.4f} {a:7.4f} {b:7.4f} {rp:7.4f} {rmm:7.4f}")
            dif = np.array(dif)
            print(f"  mean (rho_data - rho_model) = {dif.mean():+.4f}, "
                  f"outer half of the ladder {dif[len(dif)//2:].mean():+.4f}")
        print()


PAIRS = [("locx", "locy"), ("dxdz", "dydz"), ("locx", "dxdz"),
         ("locy", "dydz"), ("locx+dxdz", "locx-dxdz"),
         ("locy+dydz", "locy-dydz"), ("locx", "eig5"), ("locx", "eig4"),
         ("locx", "locx+dxdz"), ("qop", "eig3")]


def cmd_deltas(args):
    axis = list(AXIS)
    iu = IHEAD[-1]
    for g in args.geoms:
        legs = model(g)
        print("=" * 100)
        print(f"### {g}: {GEOMS[g]['label']}")

        print("\nA. THE BASIS FIX: legacy raw a-vector vs H^T a, same events")
        msH = model_side(g, axis, useH=True, nproc=args.nproc)
        msL = model_side(g, axis, useH=False, nproc=args.nproc)
        rH, cH, ks, _ = closure(g, axis, msH)
        rL, cL, _, _ = closure(g, axis, msL)
        eH, _ = errs_from_cev(cH, axis)
        eL, _ = errs_from_cev(cL, axis)
        print(f"  {'dir':<7} {'legacy (published basis)':>26} "
              f"{'with H':>26} {'difference':>22}")
        for n in axis:
            a, b = float(rL[n][:, iu].mean()), float(rH[n][:, iu].mean())
            de = float((cH[n][iu] - cL[n][iu]).std() / np.sqrt(len(cH[n][iu])))
            print(f"  {n:<7} {a:+13.5f} +- {eL[n][iu]:.5f} "
                  f"{b:+13.5f} +- {eH[n][iu]:.5f} "
                  f"{b - a:+11.5f} +- {de:.5f}")
        print("  per-plane difference (with H) - (legacy), u = 1")
        print(f"    {'k':>3} {'stereo':>7} " + "".join(f"{n:>10}" for n in axis))
        for i, k in enumerate(ks):
            H, _ = c2l.leg_H(legs, k)
            st = "yes" if abs(H[3, 4]) > 1e-9 else ""
            print(f"    {k:3d} {st:>7} "
                  + "".join(f"{rH[n][i, iu] - rL[n][i, iu]:10.5f}"
                            for n in axis))

        print("\nB. DIRECTION-TO-DIRECTION DIFFERENCES, u = 1, with the "
              "SHARED-EVENT error")
        print("   (the naive quadrature sum of the two errors is wrong: the "
              "directions are")
        print("    evaluated on the same events.  Both are printed.)")
        ms = model_side(g, args.dirs, useH=True, nproc=args.nproc)
        rows, cev, ks, _ = closure(g, args.dirs, ms)
        err, _ = errs_from_cev(cev, args.dirs)
        print(f"  {'pair':<24} {'diff':>10} {'err(shared)':>12} "
              f"{'err(naive)':>11} {'sigma(shared)':>14} {'sigma(naive)':>13}")
        for n1, n2 in PAIRS:
            if n1 not in rows or n2 not in rows:
                continue
            d = float(rows[n1][:, iu].mean() - rows[n2][:, iu].mean())
            es = diff_err(cev, n1, n2, iu)
            en = float(np.hypot(err[n1][iu], err[n2][iu]))
            print(f"  {n1 + ' - ' + n2:<24} {d:+10.5f} {es:12.5f} "
                  f"{en:11.5f} {d / max(es, 1e-12):14.1f} "
                  f"{d / max(en, 1e-12):13.1f}")
        print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("basis", "cond", "closure", "deltas", "width", "rho"))
    ap.add_argument("--geoms", nargs="+", default=["lay1", "real"])
    ap.add_argument("--dirs", nargs="+", default=ALLDIRS)
    ap.add_argument("--no-h", action="store_true",
                    help="legacy raw a-vectors (reproduces published numbers)")
    ap.add_argument("--perplane", action="store_true")
    ap.add_argument("--nproc", type=int, default=48)
    ap.add_argument("--npz", default="")
    args = ap.parse_args()
    {"basis": cmd_basis, "cond": cmd_cond, "closure": cmd_closure,
     "deltas": cmd_deltas, "width": cmd_width, "rho": cmd_rho}[args.cmd](args)


if __name__ == "__main__":
    main()
