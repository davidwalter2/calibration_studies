#!/usr/bin/env python3
"""Two probes of what is left of the real-geometry q/p closure (+0.0676).

The candidates left are: stereo modules, phi segmentation / module edges, and
the real-geometry export itself.  Acceptance
(<= 0.3 %), step structure (the real-material toy has it and closes), material
sampling (40 %, measured) and misalignment (the study runs useIdealGeometry) are
all excluded -- NOTES_GEOMCLOSURE s9-s10.

    hrot      is H's q/p row sensitive to the STEREO rotation at all?
    refdrift  does the reference track the simulated MEAN, per plane, in q/p?

usage (from calibration_studies/resolution, after `source ../setup_env.sh`):
    python stereo_probe.py hrot
    python stereo_probe.py refdrift
"""
import argparse

import numpy as np

import curv2local as c2l
import fisher_norm as fn
import geom_closure as gc

STEREO = 0.100          # rad, the CMS TIB/TOB stereo angle


def cmd_hrot(args):
    """Rotate the in-plane axes by the stereo angle and re-derive H.

    A CMS stereo module is rotated ABOUT ITS OWN NORMAL: the strip direction
    turns within the module plane, the plane does not move.  The q/p row of
    curv2localJacobianAltelossD carries the alteloss term -- the extra energy a
    transversely displaced track loses reaching the plane -- and that path is
    set by the NORMAL.  So the row should not see an in-plane rotation.  This
    checks it rather than asserting it, and prints the position rows alongside
    so the null cannot be an inert control.
    """
    legs = gc.geom_model(args.geom)
    c2l.attach_extras(legs, gc.GEOMS[args.geom]["model"])

    def H_rot(k, theta):
        leg = legs[k]
        fr = c2l.frames(leg, sinlam_sign=c2l._sinlam_sign(legs, k),
                        uz_sign=c2l.uz_sign_from_zoff(leg))
        ct, st = np.cos(theta), np.sin(theta)
        J, K = np.array(fr["J1"]), np.array(fr["K1"])
        fr2 = dict(fr, J1=ct * J + st * K, K1=-st * J + ct * K)
        q = np.sign(leg["refqop"]) or 1.0
        Bv = np.array([0.0, 0.0, 3.8 * c2l.K_TESLA_TO_INVGEV])
        return c2l.curv2local(fr2, leg["refqop"], q, leg.get("dEdxlast", 0.0),
                              c2l.MU_MASS, Bv)

    print(f"stereo rotation of {1e3 * STEREO:.0f} mrad about the module normal, "
          f"{args.geom}")
    print(f"{'k':>3} {'r[cm]':>8} {'|dH q/p row|':>14} {'|dH all|':>11} "
          f"{'|dH x,y rows|':>14}  dEdxlast")
    worst = 0.0
    for k in range(len(legs)):
        d = np.abs(H_rot(k, STEREO) - H_rot(k, 0.0))
        worst = max(worst, d[0].max())
        print(f"{k:3d} {legs[k].get('refglobr', np.nan):8.2f} {d[0].max():14.3e} "
              f"{d.max():11.3e} {d[3:5].max():14.3e}  "
              f"{legs[k].get('dEdxlast', 0.0):+.5f}")
    print(f"\nworst change in the q/p row over {len(legs)} planes: {worst:.3e}")
    print("(the position rows move by ~0.1, so the rotation IS applied)")


def cmd_refdrift(args):
    """<sim - ref>/sigma per plane, in q/p.

    A growing q/p closure is what a reference that DRIFTS off the simulated
    ensemble looks like: the residual accumulates with radius.  Both the mean
    and the median are printed, and the difference between them matters -- the
    median offset is the known mean-vs-mode of energy-loss straggling (the
    reference integrates the unrestricted MEAN loss, the typical track loses the
    MODE), which is common to every geometry; the MEAN is tail-driven and is
    where the two geometries part company.
    """
    for g in args.geoms:
        legs, sim = gc.geom_model(g), gc.geom_sim(g)
        sc = fn.plane_scales(legs, "qop", tag=g)
        print(f"\n=== {g}: {len(legs)} planes")
        print(f"{'k':>3} {'r[cm]':>8} {'<sim-ref>/sig':>14} {'med/sig':>10} "
              f"{'sigma':>11} {'dE_ref[MeV]':>12} {'dE_sim[MeV]':>12}")
        p0 = 1.0 / abs(legs[0]["refqop"])
        for k in range(len(legs)):
            ok = sim["valid"][:, k] & np.isfinite(sim["qop"][:, k])
            d = sim["qop"][ok, k] - legs[k]["refqop"]
            s = sc["sigma"][k]
            print(f"{k:3d} {np.nanmedian(sim['globr'][:, k]):8.2f} "
                  f"{d.mean() / s:14.4f} {np.median(d) / s:10.4f} {s:11.3e} "
                  f"{1e3 * (p0 - 1.0 / abs(legs[k]['refqop'])):12.4f} "
                  f"{1e3 * (p0 - np.nanmedian(1.0 / np.abs(sim['qop'][ok, k]))):12.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    q = sub.add_parser("hrot")
    q.add_argument("--geom", default="real")
    q.set_defaults(fn=cmd_hrot)
    q = sub.add_parser("refdrift")
    q.add_argument("--geoms", nargs="+", default=["real", "realmat"])
    q.set_defaults(fn=cmd_refdrift)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
