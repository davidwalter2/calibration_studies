#!/usr/bin/env python3
"""The curvilinear -> local basis for the CLOSURE's a-vectors.

WHY
---
`cf_propagation_test.FUNCTIONALS` hands the closure a FIXED curvilinear
a-vector per functional:

    qop -> qop,  dxdz -> PHI,  dydz -> LAMBDA,  locx -> xT

The closure's numerator is the SIM's residual, which is LOCAL (DetUnit frame),
while sigma comes from that curvilinear a-vector.  The two frames differ by
`H = curv2localJacobianAltelossD`, so sigma is the width of the wrong
variable.  `curv2local.py` already rebuilds H offline and `dir_closure.py`
already uses it (`useH=True`); `fisher_norm` / `geom_closure` -- i.e. every
number `allcorr`, `radoff_species` and `geom_closure` print -- never did.

This module is the one place that decides which basis those two use.

WHY IT IS NOT A HARMLESS SCALE
------------------------------
`curv2local.py`'s header says a pure scale on the a-vector cancels, because
the closure standardizes by the model's own sigma.  That is true when the
SIM's residual and the MODEL's sigma describe the SAME variable -- but they do
not: the numerator is a fixed physical local residual and only the denominator
moves.  So the ratio sigma(H^T e_i)/sigma(e_i) is a direct width error, and it
is NOT flat in plane: it carries sec(alpha), alpha = the bending-plane
incidence, which runs 0.008 -> 0.204 rad down the pT = 3 layered ladder.

Predicted closure shift from the frame alone (gauge 0.1677 = the measured
locy offset 0.0156 per 0.093 of variance), plane 0 -> plane 13:

    locy   sec(lam)                 -0.0156  flat          (already MEASURED)
    locx   sec(alpha)                0.0000 -> -0.0072
    dxdz   sec^2(alpha)              0.0000 -> -0.0147
    dydz   sec^2(lam) sec(alpha)    -0.0326 -> -0.0412

against the measured seven-correction rows (HANDOFF_BENDING_PLANE s1):

    dydz   mu-  -0.0336 -> -0.0420   p  -0.0305 -> -0.0407
    dxdz   mu-  -0.0015 -> -0.0303   p  +0.0025 -> -0.0331
    locx   mu-  -0.0001 -> -0.0039   p  +0.0039 -> -0.0069

so `dydz` is entirely frame (offset AND slope, both species, all planes), and
about half of the `dxdz` radial slope is too.  The handoff's premise that
"dydz and locy are flat, so the flat-vs-arch comparison survives them" does
not hold: the sec(alpha) factor is the same geometry that moves dxdz.

PRE-REGISTERED VALIDATION
-------------------------
`locy` is the control, because its non-closure is already measured and already
attributed: a flat -0.0156 at every plane, = sec(lambda) = 1.0453 at
eta = 0.30.  Under `USE_H` it must go to ZERO, flat.  If it does not, H is
wrong and nothing else here should be believed.  `python hbasis.py check`
runs exactly that.

USAGE
-----
    import hbasis
    hbasis.bind(legs, model_path, pdg=13)   # once, in the PARENT
    hbasis.set_use_h(True)                  # NOT `hbasis.USE_H = True` from
                                            # inside this file -- see _canonical
    ...                                     # fisher_norm / geom_closure follow

`USE_H` is a physics global: it is in `PHYSICS_GLOBALS`, in
`fisher_norm.scale_identity` and in `fisher_norm._CACHE_MODULES`, so both the
in-process and the on-disk scale caches key on it.  With `USE_H = False`
`avecs` returns the very same array object `FUNCTIONALS` holds, so the legacy
path is bit-identical and not merely equivalent.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import curv2local as c2l                                        # noqa: E402
from cf_propagation_test import FUNCTIONALS                     # noqa: E402

# H's row order.  Local state is (q/p, dx/dz, dy/dz, x, y) and curvilinear is
# (q/p, lambda, phi, xT, yT) -- see curv2local.curv2local's docstring.  Note
# dx/dz pairs with PHI and dy/dz with LAMBDA, not the other way round: local x
# is the r-phi direction in a barrel.  Getting this backwards is the swap
# commit e636024 fixed in FUNCTIONALS.
LOCAL = ("qop", "dxdz", "dydz", "locx", "locy")

# THE KNOB.  Default OFF: turning it on changes every closure number in the
# tree, so it must be an explicit act by a caller, never a default that a
# rerun silently picks up.
USE_H = False

# The field H is built at, in Tesla.  It is a property of the MODEL being read,
# not a free choice: H's curvature term is proportional to it.  A field scan
# must move this together with the field the model was exported at, or H
# describes a trajectory the export does not have.
BFIELD = 3.8

PHYSICS_GLOBALS = ("USE_H", "BFIELD")
_NOT_PHYSICS = ()


def physics_state():
    g = vars(_canonical())
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


def _canonical():
    """The module object `fisher_norm` and `geom_closure` actually see.

    THIS IS NOT PEDANTRY.  `python hbasis.py ...` executes this file as
    `__main__` while those modules `import hbasis`, so there are TWO module
    objects with INDEPENDENT `USE_H`.  Setting it on the wrong one is silent in
    the worst way: the banner says H while `avecs` hands back the legacy
    vector.  (It first showed up as a KeyError on locy, which was luck --
    every other functional would have quietly printed legacy numbers under an
    "H" heading.)  Everything that flips the switch goes through `set_use_h`,
    and every reader goes through `_canonical`, so the two views cannot drift.
    """
    import hbasis
    return hbasis


def set_use_h(v):
    """Set `USE_H` on BOTH views of this module.  See `_canonical`."""
    v = bool(v)
    _canonical().USE_H = v
    globals()["USE_H"] = v
    return v


def _state():
    """The binding and a-vector caches, on the canonical view only.

    Same reason as `_canonical`: two module objects means two dicts, and a
    `bind` on the `__main__` copy would leave `avecs` on the imported copy
    convinced the legs were never bound."""
    m = _canonical()
    return m._BOUND, m._AVEC


# id(legs) -> (path, mass, bfield).  `legs` is held by the caller (and by
# fisher_norm._MODEL_CACHE) for as long as it is used, so the id cannot be
# recycled onto a different model while an entry is live -- the same argument
# fisher_norm._PROVENANCE relies on.
_BOUND = {}
_AVEC = {}

MU_MASS = c2l.MU_MASS


def bind(legs, path, pdg=13, mass=None, bfield=None):
    """Attach what H needs to `legs` and record the species.

    `curv2local.attach_extras` reads `dEdxlast`, `refglobz`, `zoff`, `reflocz`
    and `refglobr`, which `cf_propagation_test.load_model` deliberately does
    not.  H depends on the MASS through the energy-loss term, so a hadron cell
    bound at the muon mass is wrong -- pass `pdg` and the mass is looked up.

    `mass` is in GeV, as `curv2local` wants it.  `hadron_probe.SPECIES` stores
    MeV (105.6583745 for the muon, against curv2local.MU_MASS = 0.1056583745),
    so the lookup converts.  Getting this wrong is silent: H stays finite and
    only its energy-loss term is off by 1000.

    Call this in the PARENT, before any pool is forked: the workers inherit
    the attached legs and the memoized a-vectors through the fork and never
    touch the ROOT file.
    """
    if mass is None:
        import hadron_probe as hp
        mass = hp.SPECIES[pdg]["mass"] * 1e-3
    if bfield is None:
        bfield = _canonical().BFIELD
    if "dEdxlast" not in legs[0]:
        c2l.attach_extras(legs, path)
    B, A = _state()
    B[id(legs)] = (path, float(mass), float(bfield))
    A.pop(id(legs), None)
    return legs


def bound(legs):
    return _state()[0].get(id(legs))


def avecs(legs, func):
    """Per-plane curvilinear a-vectors for `func`, one per leg.

    With `USE_H` off this is `[FUNCTIONALS[func]] * n` -- the identical object,
    so every legacy number is reproduced bit for bit.  With it on, plane k gets
    `H_k^T e_i` = row i of H_k, i.e. the curvilinear direction whose projection
    IS the local component the sim residual reports.

    No standardization and no orientation here (unlike `dir_closure.avecs`):
    the closure divides a RAW local residual by sF, so sigma must be the
    predicted width of that same raw local component.  The overall sign is
    irrelevant -- the closure statistic is even in z.
    """
    n = len(legs)
    if not _canonical().USE_H:
        if func not in FUNCTIONALS:
            raise KeyError(
                f"{func!r} has no legacy curvilinear a-vector. locy is "
                f"deliberately absent from FUNCTIONALS (its naive yT vector is "
                f"wrong by sec(lambda)); set hbasis.USE_H = True to get it "
                f"from H instead.")
        return [FUNCTIONALS[func]] * n
    if func not in LOCAL:
        raise KeyError(f"{func!r} is not a local component; known: {LOCAL}")
    B, A = _state()
    key = (id(legs), func)
    hit = A.get(key)
    if hit is not None and len(hit) == n:
        return hit
    b = B.get(id(legs))
    if b is None:
        raise RuntimeError(
            "hbasis.USE_H is on but these legs were never bound. Call "
            "hbasis.bind(legs, model_path, pdg=...) in the parent process "
            "before any scale or closure call.")
    path, mass, bfield = b
    i = LOCAL.index(func)
    out = []
    for k in range(n):
        H, _ = c2l.leg_H(legs, k, bfield=bfield, mass=mass)
        out.append(np.ascontiguousarray(H[i, :]))
    A[key] = out
    return out


def width_ratio(legs, func):
    """sigma(H^T e_i) / sigma(e_i) per plane -- the direct width error the
    legacy basis carries.  Independent of USE_H; it reads both bases."""
    import cf_propagation_test as cpt
    was = _canonical().USE_H
    try:
        set_use_h(True)
        new = avecs(legs, func)
        set_use_h(False)
        old = (avecs(legs, func) if func in FUNCTIONALS else None)
    finally:
        set_use_h(was)
    r = []
    for k in range(len(legs)):
        sn = np.sqrt(cpt.model_variance(legs, k, new[k])[0])
        so = (np.sqrt(cpt.model_variance(legs, k, old[k])[0])
              if old is not None else np.nan)
        r.append(float(sn / so) if old is not None else np.nan)
    return np.array(r)


# ==========================================================================
# the control
# ==========================================================================

def cmd_check(args):
    """locy is the pre-registered control: its -0.0156 must go to zero."""
    import allcorr as ac                                        # noqa: F401
    import hadron_probe as hp
    import radoff_species as rs
    import cf_propagation_test as cpt

    path = rs.mp(args.pdg, True)
    legs = cpt.load_model(path)
    bind(legs, path, pdg=args.pdg)
    print(f"model {os.path.basename(path)}   {len(legs)} planes   "
          f"pdg {args.pdg} (mass {bound(legs)[1]:.4f} GeV)")
    print("\nwidth ratio sigma(H^T e_i)/sigma(e_i), per plane:")
    print(f"{'k':>3}" + "".join(f"{f:>10}" for f in ("locx", "dxdz", "dydz")))
    rr = {f: width_ratio(legs, f) for f in ("locx", "dxdz", "dydz")}
    for k in range(len(legs)):
        print(f"{k:3d}" + "".join(f"{rr[f][k]:10.5f}"
                                 for f in ("locx", "dxdz", "dydz")))
    print("\nexpected, from the geometry alone: locx sec(alpha), dxdz "
          "sec^2(alpha), dydz sec^2(lambda) sec(alpha)")
    print("  alpha = bending-plane incidence, 0.008 -> 0.204 rad; "
          "sec(lambda) = 1.0453 at eta = 0.30")


def cmd_profile(args):
    """THE MEASUREMENT: the per-plane closure profile in both bases.

    Reports u = 1 only -- the profile is what the ladder mean hides (it is a
    mean over nested cumulative values with a sign change in the middle), so
    the per-plane row is the statistic, with the outermost plane called out.

    `locy` is the control and is H-only: it has no legacy curvilinear vector
    (its naive yT direction is wrong by sec(lambda), which is the whole point),
    so `FUNCTIONALS` deliberately omits it.  Its published legacy value is a
    flat -0.0156 and under H it must go to zero.
    """
    import allcorr as ac
    import fisher_norm as fn

    iu = list(ac.UCURVE).index(1.0)
    print(f"seven corrections, radiation ON, u = 1, Fisher normalization, "
          f"nothing fitted")
    print(ac.switch_banner())
    for pdg in args.pdg:
        lab = _label(pdg)
        for func in args.funcs:
            print(f"\n{lab}  {func}")
            for useh in (False, True):
                if func == "locy" and not useh:
                    print(f"  {'legacy':<8}  (no curvilinear a-vector -- "
                          f"FUNCTIONALS omits locy by design)")
                    continue
                r = ac.cell(pdg, True, func, ms=True, useh=useh)
                row = np.asarray(r["rows"])[:, iu]
                err = np.asarray(r["errs"])[:, iu]
                print(f"  {'H' if useh else 'legacy':<8}" +
                      "".join(f"{x:8.4f}" for x in row) +
                      f"   ladder {row.mean():+.4f}  outer {row[-1]:+.4f}"
                      f" +- {err[-1]:.4f}")
    print("\nnote: dEdxlast is 0 on plane 1 of the pT = 3 toy export, so H's "
          "energy-loss term is absent there.  It is a per-mille effect at that "
          "radius, but it is an EXPORT defect, not a physics one.")


def cmd_geom(args):
    """The same both-basis profile on a `geom_closure` geometry.

    This is where the REAL tracker lives.  Its published radial growth is
    +0.0004 -> +0.0676 over 19 planes (HANDOFF_BENDING_PLANE s8, filed there as
    a separate open item) -- an order of magnitude larger than the toy arch
    that turned out to be basis, and never measured in the H basis.
    `curv2local`'s own header already records locx sec(alpha) = 1.000-1.029 and
    qop = 1.000-1.029 on this geometry.
    """
    import geom_closure as gc
    import fisher_norm as fn
    import cf_propagation_test as cpt

    iu = list(fn.UCURVE).index(1.0)
    for g in args.geoms:
        legs = gc.geom_model(g)
        path = gc.GEOMS[g]["model"]
        bind(legs, fn._path(path), pdg=args.pdg)
        print(f"\n=== {g}: {gc.GEOMS[g]['label']}, {len(legs)} planes")
        for func in args.funcs:
            print(f"\n{func}")
            for useh in (False, True):
                if func == "locy" and not useh:
                    print(f"  {'legacy':<8}  (no curvilinear a-vector)")
                    continue
                was = set_use_h(useh)
                fn._SCALE_CACHE.clear()
                cpt._PHI_CACHE.clear()
                try:
                    r = gc.geom_closure(g, func)
                finally:
                    set_use_h(False if was is None else was)
                    fn._SCALE_CACHE.clear()
                    cpt._PHI_CACHE.clear()
                row = np.asarray(r["rows"])[:, iu]
                err = np.asarray(r["errs"])[:, iu]
                print(f"  {'H' if useh else 'legacy':<8}" +
                      "".join(f"{x:8.4f}" for x in row))
                print(f"  {'':8}  ladder {row.mean():+.4f}  "
                      f"outer {row[-1]:+.4f} +- {err[-1]:.4f}")


def _label(pdg):
    import hadron_probe as hp
    return hp.SPECIES[pdg]["label"]


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the width ratios H introduces")
    c.add_argument("--pdg", type=int, default=13)
    c.set_defaults(f=cmd_check)
    q = sub.add_parser("profile", help="the per-plane closure in both bases")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, 2212])
    q.add_argument("--funcs", nargs="+",
                   default=["locy", "qop", "locx", "dxdz", "dydz"])
    q.set_defaults(f=cmd_profile)
    g = sub.add_parser("geom", help="both bases on a geom_closure geometry")
    g.add_argument("--geoms", nargs="+", default=["real"])
    g.add_argument("--pdg", type=int, default=13)
    g.add_argument("--funcs", nargs="+",
                   default=["locy", "qop", "locx", "dxdz", "dydz"])
    g.set_defaults(f=cmd_geom)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
