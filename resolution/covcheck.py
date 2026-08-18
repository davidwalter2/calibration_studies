#!/usr/bin/env python3
"""Model vs sim, element by element, in the LOCAL 5x5.

WHY NOT THE CLOSURE
-------------------
`<e^{-u z^2}>` is a scalar per direction: it says the model is wrong, not WHERE.
Two residuals survive the basis fix -- a +0.005 locx plateau on both geometries,
and the qop degradation H exposes -- and both are statements about a particular
element of the covariance.  The sim gives all five local residuals per event, so
the covariance can be compared directly.

    model:  C_local = H C_curv H^T,   C_curv = sum_j A_j Q_j A_j^T
    sim:    the sample covariance of (sim - ref) in the same five components

CORRELATIONS, NOT VARIANCES, ARE THE ROBUST COMPARISON
------------------------------------------------------
The model's variance is a TRUNCATED second moment: the ionization channel is cut
at alpha = 0.999 and `G4UniversalFluctuationForExtrapolator.hh` records that the
truncated sigma grows ~3x between alpha = 0.99 and 0.999, i.e. the variance is
convention-dependent.  A raw sample covariance from the sim is not truncated at
all and is dominated by the Landau tail, especially in qop.  Comparing those two
numbers directly compares conventions as much as physics -- the same trap
NOTES.md flagged for the per-leg truncation study.

CORRELATIONS largely divide that out: a common scale error on a component
cancels in rho = C_ij/sqrt(C_ii C_jj).  So rho is the headline and the variance
ratio is reported beside it as context, never on its own.  The tail sensitivity
that survives is handled by quoting the sim correlation BOTH raw and on a core
(|z| < ZCUT on every component), so a number that moves between them is flagged
rather than believed.

WHAT TO LOOK FOR
----------------
  rho(qop, locx)   the term H's qop row adds, and the only thing that changed
                   when qop broke under H.  If the model and sim disagree here
                   the qop story is closed.
  rho(locx, dxdz)  the model predicts ~-0.9 (dir_closure); a position/angle
                   correlation error is the natural home for a locx plateau
                   that is flat in radius.

usage:
    python covcheck.py --geom real lay1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_propagation_test as cpt                                # noqa: E402
import curv2local as c2l                                         # noqa: E402
import dir_closure as dc                                         # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402

LOC = ("qop", "dxdz", "dydz", "locx", "locy")
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
ZCUT = 3.0


def model_local(legs, k, mass=0.1056583745, bfield=3.8):
    H, _ = c2l.leg_H(legs, k, mass=mass, bfield=bfield)
    C = dc.cov_curv(legs, k)
    return H @ C @ H.T, H


def sim_local(sim, legs, k, core=None):
    """Sample covariance of the five local residuals on plane k.

    `core` (in units of the MODEL's own predicted sigma) trims every component
    simultaneously -- a per-component trim would sculpt the correlations, which
    are the thing being measured.
    """
    good = sim["valid"][:, k].copy()
    for b in LOC:
        good &= np.isfinite(sim[b][:, k])
    if good.sum() < 100:
        return None, 0
    R = np.stack([sim[b][good, k] - legs[k][cpt.REF_BRANCH[b]]
                  for b in LOC], axis=1)
    if core is not None:
        sd = np.sqrt(np.diag(core))
        keep = np.all(np.abs(R) < ZCUT * sd[None, :], axis=1)
        R = R[keep]
    return np.cov(R, rowvar=False), len(R)


def corr(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def cmd_main(args):
    for g in args.geom:
        legs = gc.geom_model(g)
        path = fn._path(gc.GEOMS[g]["model"])
        c2l.attach_extras(legs, path)
        sim = gc.geom_sim(g)
        nl = min(sim["valid"].shape[1], len(legs))
        planes = args.planes or [0, nl // 2, nl - 1]
        print("=" * 108)
        print(f"{g}: {gc.GEOMS[g]['label']}, {len(legs)} planes")
        print("=" * 108)
        for k in planes:
            Cm, H = model_local(legs, k)
            Cs, n = sim_local(sim, legs, k)
            Cc, nc = sim_local(sim, legs, k, core=Cm)
            if Cs is None:
                continue
            rm, rs, rc = corr(Cm), corr(Cs), corr(Cc)
            print(f"\n--- plane {k}   n = {n}  (core {nc}, "
                  f"{100.*nc/n:.1f} %)   dEdxlast = {legs[k]['dEdxlast']:+.6f}")
            print(f"  {'component':>12}{'sigma model':>14}{'sigma sim':>12}"
                  f"{'ratio':>9}{'core sim':>12}{'ratio':>9}")
            for i, nm in enumerate(LOC):
                sm, ss, sc = (np.sqrt(Cm[i, i]), np.sqrt(Cs[i, i]),
                              np.sqrt(Cc[i, i]))
                print(f"  {nm:>12}{sm:14.6g}{ss:12.6g}{ss/sm:9.4f}"
                      f"{sc:12.6g}{sc/sm:9.4f}")
            print(f"  {'correlation':>12}{'model':>14}{'sim':>12}"
                  f"{'diff':>9}{'core sim':>12}{'diff':>9}")
            for i, j in PAIRS:
                d1, d2 = rs[i, j] - rm[i, j], rc[i, j] - rm[i, j]
                flag = "  <<<" if abs(d2) > 0.02 else ""
                print(f"  {LOC[i] + '/' + LOC[j]:>12}{rm[i, j]:14.4f}"
                      f"{rs[i, j]:12.4f}{d1:9.4f}{rc[i, j]:12.4f}"
                      f"{d2:9.4f}{flag}")

        # the two headline correlations, all planes
        print(f"\n### {g}: the two diagnostic correlations, every plane "
              f"(core estimator)")
        print(f"  {'k':>3}{'rho(qop,locx) mod':>20}{'sim':>9}{'diff':>9}"
              f"   {'rho(locx,dxdz) mod':>20}{'sim':>9}{'diff':>9}")
        for k in range(nl):
            Cm, _ = model_local(legs, k)
            Cc, nc = sim_local(sim, legs, k, core=Cm)
            if Cc is None:
                continue
            rm, rc = corr(Cm), corr(Cc)
            print(f"  {k:3d}{rm[0, 3]:20.4f}{rc[0, 3]:9.4f}"
                  f"{rc[0, 3] - rm[0, 3]:9.4f}   {rm[3, 1]:20.4f}"
                  f"{rc[3, 1]:9.4f}{rc[3, 1] - rm[3, 1]:9.4f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", nargs="+", default=["lay1", "real"])
    p.add_argument("--planes", type=int, nargs="*", default=None)
    cmd_main(p.parse_args())


if __name__ == "__main__":
    main()
