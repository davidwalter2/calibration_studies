#!/usr/bin/env python3
"""The curvilinear -> local Jacobian H, rebuilt offline for the clean-propagation test.

WHY THIS EXISTS
---------------
`cf_propagation_test.py` carries a standing warning at the top of the file:

    !! BASIS MISMATCH -- KNOWN, NOT YET FIXED (found 2026-08-06) !!
    The residuals are LOCAL (DetUnit frame: q/p, dx/dz, dy/dz, x, y) but the
    exported Q/F/dQMS/dQI are CURVILINEAR (q/p, lambda, phi, xT, yT) and
    load_model reads them raw.

For the two functionals tested so far the mismatch happens to be harmless:

  * `qop`  is the same variable in both frames (up to a tiny path-length term);
  * `locx` maps onto the curvilinear xT direction times a PURE SCALE
    (sec(alpha)) whenever the module's local y axis is along the global z
    axis -- and the closure statistic standardizes by the model's own sigma,
    so a pure scale on the a-vector CANCELS EXACTLY.

Neither is true for `locy`, `dydz` or any mixed direction: there H rotates the
direction as well as scaling it, and a few-percent rotation moves the closure
at u = 1 by ~1e-3, i.e. by the size of the residual being measured.  So the
extra directions cannot be tested without H.

WHAT IS REBUILT, AND FROM WHAT
------------------------------
`H = curv2localJacobianAltelossD` (ResidualGlobalCorrectionMakerBase.cc:2831)
is transcribed here verbatim -- same SymPy-generated expressions, same
symbols -- so this is a port, not a re-derivation.  It needs, per plane:

    I1, J1, K1  the module's local z (normal), x, y axes in GLOBAL coordinates
    W0          the reference momentum direction, global
    lam0, phi0  its dip and azimuth
    qop0, q     the reference q/p and charge
    B, H        the field magnitude and direction at the reference point
    dEdx        the propagator's dE/dx at the end of the leg
    mass        the particle mass

The model export (`G4ePropagationExport.cc`) gives `refdxdz`, `refdydz`,
`refp`, `refpt`, `refqop`, `refglobr`, `refglobz` and `dEdxlast` but NOT the
module rotation.  It is reconstructed here from the local direction plus ONE
geometric assumption:

    THE MODULE PLANE CONTAINS THE GLOBAL Z AXIS  (n . zhat = 0)

which is what "barrel module" means and holds for every plane in both
geometries used (the toy planes are built that way by `gen_toy_config.py`;
the real ladder is 19 barrel modules, BPIX + TIB + TOB).  Under it the global
z axis lies IN the module plane,

    zhat_local = (sin psi, cos psi, 0)

with a single unknown psi, fixed by the exactly-known projection

    zhat . phat = sin(lambda)   ->   sin psi dxdz + cos psi dydz = N sin lambda

(N = |(dxdz, dydz, 1)|).  That is a one-parameter equation with two roots; the
root nearest {0, pi} is taken, and `frames()` reports both so the choice is
auditable.  On the real ladder the chosen root comes out at EXACTLY 0 or pi on
the 15 rphi modules and at 0.100 rad away from it on the 4 stereo modules --
the CMS stereo angle, which nothing here put in.  That is the premise check.

The field is taken as the nominal solenoid (0, 0, 3.8 T).  It enters only the
angle<-position corner of H (a displaced track reaches the plane after a
different path length, over which the field has rotated it); `bscale` lets the
sensitivity be measured rather than assumed.
"""

import numpy as np

# CMSSW MagneticField::kTeslaToInvGeV
K_TESLA_TO_INVGEV = 2.99792458e-3
MU_MASS = 0.1056583745


# --------------------------------------------------------------------------
# frame reconstruction
# --------------------------------------------------------------------------

def solve_psi(dxdz, dydz, sinlam):
    """Both roots of  sin(psi) dxdz + cos(psi) dydz = N sin(lambda).

    psi is the angle of the GLOBAL z axis in the module's own (x, y) plane, so
    psi = 0 means local y || global z (an rphi barrel module) and psi = pi means
    local y || -global z.  A stereo module sits 100 mrad away from either.
    """
    N = np.sqrt(1.0 + dxdz ** 2 + dydz ** 2)
    A = np.hypot(dxdz, dydz)
    rhs = N * sinlam / A
    if abs(rhs) > 1.0:
        if abs(rhs) > 1.0 + 1e-9:
            raise ValueError(
                f"no barrel-frame solution: |N sin(lambda)/|d|| = {rhs:.6f} > 1. "
                "The module plane does not contain the global z axis "
                "(endcap?), and this reconstruction does not apply.")
        rhs = np.sign(rhs)
    psi0 = np.arctan2(dxdz, dydz)
    d = np.arccos(rhs)
    return psi0 + d, psi0 - d


def _pick_psi(roots):
    """The root nearest 0 or pi (mod 2 pi)."""
    def dist(p):
        p = np.mod(p, 2 * np.pi)
        return min(abs(p), abs(p - np.pi), abs(p - 2 * np.pi))
    return min(roots, key=dist)


def _rot_local_to_global(phat_loc, zhat_loc, phat_glob, zhat_glob):
    """R with R phat_loc = phat_glob and R zhat_loc = zhat_glob (exact: the two
    pairs subtend the same angle by construction)."""
    def triad(a, b):
        f1 = a / np.linalg.norm(a)
        t = b - f1 * (f1 @ b)
        f2 = t / np.linalg.norm(t)
        return np.stack([f1, f2, np.cross(f1, f2)], axis=1)   # columns
    return triad(phat_glob, zhat_glob) @ triad(phat_loc, zhat_loc).T


def frames(leg, sinlam_sign=+1.0, psi_root=None, uz_sign=+1.0):
    """Per-leg frame reconstruction.

    `uz_sign` is the sign of the track's LOCAL z momentum component, i.e. which
    side of the module the track enters from.  It is NOT recoverable from
    (dxdz, dydz) -- those are ratios and flip with it -- and it MATTERS: it
    flips the curvilinear xT axis relative to the module, hence the relative
    sign of the two position (and the two angle) columns of H.  Derived from
    the export's `zoff` = the entry-face local z: the propagation target is
    built AT the entry face, so a track moving in +local z enters at
    zoff = -half-thickness.  See `uz_sign_from_zoff`.

    Returns a dict with the local axes in global coordinates (I1 = normal,
    J1 = local x, K1 = local y), the momentum direction W0, lam0, phi0 (= 0,
    which is WLOG: everything downstream is covariant under rotation about the
    global z axis) and both psi roots.
    """
    dxdz, dydz = leg["refdxdz"], leg["refdydz"]
    coslam = leg["refpt"] / leg["refp"]
    coslam = min(max(coslam, -1.0), 1.0)
    sinlam = sinlam_sign * np.sqrt(max(0.0, 1.0 - coslam ** 2))
    lam0 = np.arctan2(sinlam, coslam)

    roots = solve_psi(uz_sign * dxdz, uz_sign * dydz, sinlam)
    psi = _pick_psi(roots) if psi_root is None else roots[psi_root]

    N = np.sqrt(1.0 + dxdz ** 2 + dydz ** 2)
    phat_loc = uz_sign * np.array([dxdz, dydz, 1.0]) / N
    zhat_loc = np.array([np.sin(psi), np.cos(psi), 0.0])
    phat_g = np.array([coslam, 0.0, sinlam])
    zhat_g = np.array([0.0, 0.0, 1.0])

    R = _rot_local_to_global(phat_loc, zhat_loc, phat_g, zhat_g)
    return dict(J1=R @ np.array([1.0, 0.0, 0.0]),
                K1=R @ np.array([0.0, 1.0, 0.0]),
                I1=R @ np.array([0.0, 0.0, 1.0]),
                W0=phat_g, lam0=lam0, phi0=0.0, psi=psi, roots=roots, R=R)


def frames_from_axes(ux, uy, uz, W0):
    """Frames when the module axes ARE known in global coordinates (the toy).

    Used only to validate `frames()`; the real geometry has no such input.
    """
    W0 = np.asarray(W0, dtype=float)
    W0 = W0 / np.linalg.norm(W0)
    lam0 = np.arcsin(W0[2])
    phi0 = np.arctan2(W0[1], W0[0])
    return dict(J1=np.asarray(ux, float), K1=np.asarray(uy, float),
                I1=np.asarray(uz, float), W0=W0, lam0=lam0, phi0=phi0,
                psi=np.nan, roots=(np.nan, np.nan), R=None)


# --------------------------------------------------------------------------
# the Jacobian itself -- verbatim transcription of
# ResidualGlobalCorrectionMakerBase::curv2localJacobianAltelossD
# --------------------------------------------------------------------------

def curv2local(fr, qop0, q, dEdx, mass, Bvec):
    """H = d(local)/d(curvilinear), local = (q/p, dx/dz, dy/dz, x, y),
    curvilinear = (q/p, lambda, phi, xT, yT)."""
    Ix1, Iy1, Iz1 = fr["I1"]
    Jx1, Jy1, Jz1 = fr["J1"]
    Kx1, Ky1, Kz1 = fr["K1"]
    W0x, W0y, W0z = fr["W0"]
    lam0, phi0 = fr["lam0"], fr["phi0"]

    Bvec = np.asarray(Bvec, dtype=float)
    B = np.linalg.norm(Bvec)
    hx, hy, hz = (Bvec / B) if B > 0 else (0.0, 0.0, 1.0)

    x0 = q ** 2
    x1 = x0 ** 1.5
    x2 = Ix1 * W0y - Iy1 * W0x
    x3 = qop0 ** 2
    x4 = W0x ** 2 + W0y ** 2
    x5 = x4 ** -0.5
    x6 = np.sin(lam0)
    x7 = Iz1 * x6
    x8 = np.cos(lam0)
    x9 = np.cos(phi0)
    x10 = Ix1 * x9
    x11 = np.sin(phi0)
    x12 = Iy1 * x11
    x13 = x10 * x8 + x12 * x8 + x7
    x14 = x5 / x13
    x15 = dEdx * q * x14 * x3 * np.sqrt(mass ** 2 * x3 + x0) / x1
    x16 = Ix1 * W0x * W0z + Iy1 * W0y * W0z - Iz1 * x4
    x17 = x13 ** -2
    x18 = Iz1 * Jx1
    x19 = Iz1 * Jy1
    x20 = lam0 + phi0
    x21 = -phi0
    x22 = lam0 + x21
    x23 = (Ix1 * np.cos(x20) + Ix1 * np.cos(x22) + Iy1 * np.sin(x20)
           - Iy1 * np.sin(x22) + 2 * x7) ** -2
    x24 = 2 * Ix1
    x25 = Jy1 * x24
    x26 = 2 * Iy1
    x27 = Jx1 * x26
    x28 = 2 * lam0
    x29 = np.cos(x28)
    x30 = phi0 + x28
    x31 = np.cos(x30)
    x32 = np.sin(x30)
    x33 = Ix1 * Jz1
    x34 = Iy1 * Jz1
    x35 = x21 + x28
    x36 = np.cos(x35)
    x37 = np.sin(x35)
    x38 = x8 * x9
    x39 = x11 * x8
    x40 = Jx1 * x38 + Jy1 * x39 + Jz1 * x6
    x41 = hz * x8
    x42 = hy * x6 - x11 * x41
    x43 = x8 * (hx * x11 - hy * x9)
    x44 = hx * x6 - x41 * x9
    x45 = -Ix1 * x42 + Iy1 * x44 - Iz1 * x43
    x46 = B * qop0 * x5 / x13 ** 3
    x47 = x46 * (x13 * (-Jx1 * x42 + Jy1 * x44 - Jz1 * x43) - x40 * x45)
    x48 = Iz1 * Kx1
    x49 = Iz1 * Ky1
    x50 = Ky1 * x24
    x51 = Kx1 * x26
    x52 = Ix1 * Kz1
    x53 = Iy1 * Kz1
    x54 = Kx1 * x38 + Ky1 * x39 + Kz1 * x6
    x55 = x46 * (x13 * (-Kx1 * x42 + Ky1 * x44 - Kz1 * x43) - x45 * x54)
    x56 = x13 * x4
    x57 = W0z * x13

    H = np.zeros((5, 5))
    H[0, 0] = x1 * abs(qop0) / (q ** 3 * qop0)
    H[0, 3] = -x15 * x2
    H[0, 4] = -x15 * x16
    H[1, 1] = x17 * (Jz1 * x10 + Jz1 * x12 - x11 * x19 - x18 * x9)
    H[1, 2] = x23 * (x18 * x31 - x18 * x36 + x19 * x32 + x19 * x37 + x25 * x29
                     + x25 - x27 * x29 - x27 - x31 * x33 - x32 * x34
                     + x33 * x36 - x34 * x37)
    H[1, 3] = x2 * x47
    H[1, 4] = x16 * x47
    H[2, 1] = x17 * (Kz1 * x10 + Kz1 * x12 - x11 * x49 - x48 * x9)
    H[2, 2] = x23 * (x29 * x50 - x29 * x51 + x31 * x48 - x31 * x52 + x32 * x49
                     - x32 * x53 - x36 * x48 + x36 * x52 + x37 * x49
                     - x37 * x53 + x50 - x51)
    H[2, 3] = x2 * x55
    H[2, 4] = x16 * x55
    H[3, 3] = x14 * (x13 * (-Jx1 * W0y + Jy1 * W0x) + x2 * x40)
    H[3, 4] = x14 * (Jz1 * x56 + x16 * x40 - x57 * (Jx1 * W0x + Jy1 * W0y))
    H[4, 3] = x14 * (x13 * (-Kx1 * W0y + Ky1 * W0x) + x2 * x54)
    H[4, 4] = x14 * (Kz1 * x56 + x16 * x54 - x57 * (Kx1 * W0x + Ky1 * W0y))
    return H


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def uz_sign_from_zoff(leg, fallback=+1.0):
    """Sign of the track's local-z momentum component, from the entry face.

    `zoff` (== `reflocz`) is the local z of the surface the reference was
    propagated to, and that surface is the module's ENTRY face -- it is written
    by `cf_propagation_test.write_targets` as the sim's median PSimHit entry
    local z.  A track moving in +local z enters at -half-thickness, so

        zoff < 0  ->  uz > 0 ,   zoff > 0  ->  uz < 0

    On the real ladder zoff = +-0.01425 / +-0.0145 / +-0.0235 cm, i.e. exactly
    the half-thicknesses of BPIX (285 um), TIB (290 um) and TOB (470 um), and
    it changes sign 8 times down the ladder.  The toy planes have zoff = 0
    (they pass through the reference), and there `fallback` applies -- with the
    outward radial normals and outward motion `gen_toy_config.py` builds, uz is
    +1 on every plane.
    """
    z = leg.get("zoff", 0.0)
    if abs(z) < 1e-9:
        return fallback
    return -1.0 if z > 0 else +1.0


def leg_H(legs, k, sinlam_sign=None, bfield=3.8, bscale=1.0, mass=MU_MASS,
          psi_root=None, dedx=None, uz_sign=None):
    """H for plane k of a loaded model, from the export alone."""
    leg = legs[k]
    if sinlam_sign is None:
        sinlam_sign = _sinlam_sign(legs, k)
    if uz_sign is None:
        uz_sign = uz_sign_from_zoff(leg)
    fr = frames(leg, sinlam_sign=sinlam_sign, psi_root=psi_root,
                uz_sign=uz_sign)
    q = np.sign(leg["refqop"]) or 1.0
    de = leg.get("dEdxlast", 0.0) if dedx is None else dedx
    Bv = np.array([0.0, 0.0, bscale * bfield * K_TESLA_TO_INVGEV])
    return curv2local(fr, leg["refqop"], q, de, mass, Bv), fr


def attach_extras(legs, path):
    """`dEdxlast` and `refglobz` are exported but `load_model` does not read
    them (it is left untouched here on purpose).  Read them separately."""
    import uproot
    f = uproot.open(path)
    tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
    br = ["dEdxlast", "refglobz", "zoff", "reflocz", "refglobr"]
    a = f[tk].arrays(br, library="np")
    for k, leg in enumerate(legs):
        for b in br:
            leg[b] = float(a[b][k])
    return legs


def _sinlam_sign(legs, k):
    """Sign of sin(lambda) from the reference trajectory's own z ordering."""
    zs = [l.get("refglobz", np.nan) for l in legs]
    zs = [z for z in zs if np.isfinite(z)]
    if len(zs) >= 2 and zs[-1] != zs[0]:
        return 1.0 if zs[-1] > zs[0] else -1.0
    return 1.0


def all_H(legs, **kw):
    return [leg_H(legs, k, **kw) for k in range(len(legs))]
