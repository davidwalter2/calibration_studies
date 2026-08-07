"""Which multiple-scattering summary -- Rossi (15 MeV, no log) or Highland
(13.6 MeV, with log) -- actually describes the tracker data, and in what
sense?

Neither is first-principles. The first-principles object is the Moliere /
screened-Rutherford compound-Poisson distribution; Rossi and Highland are
two DIFFERENT SUMMARIES of it:

  * Rossi matches the FULL SECOND MOMENT <theta^2>. Second moments add
    exactly under convolution, which is why the Rossi form is additive over
    steps. But <theta^2> of a 1/theta^4 tail is TAIL-DOMINATED and only
    logarithmically convergent -- it is a property of the rare hard
    scatters, not of the bulk.

  * Highland is an empirical fit (Highland 1975; Lynch & Dahl 1991) to the
    width of the Gaussian describing the CENTRAL 98% of the Moliere
    distribution. It is not a moment, so it does not add under convolution
    -- applying it per step is simply wrong, and PDG says so explicitly.

So the question "which is better" is really "which feature of the true
distribution does the fit's Gaussian Q need to match?" This script measures
both features of the EXACT distribution, for the real per-step material of
our tracks, and compares each formula to the feature it claims to describe.

Method: build the exact Moliere log-CF for the whole traversal (additive
over steps -- compound-Poisson exponents always are), invert it numerically
to p(theta), then read off
    (a) the full RMS                     -> compare to Rossi / the fit's Q
    (b) the RMS of the central 98%       -> compare to Highland
All angles are projected (one plane); the CF is a 2D (space-angle)
transform, so <theta_space^2> = 2 <theta_plane^2> is applied explicitly.

usage:
    python rossi_vs_highland.py --file <globalcor_resclosure_0.root> [--ntracks 400]
"""

import argparse

import numpy as np
import uproot
from scipy.special import j0

from wums import logging

import cf_ms_exact as cm

logger = logging.child_logger(__name__)

# msmoliv record layout (8 floats/step), from ResidualGlobalCorrectionMaker:
#   0 effZ  1 effA  2 x[g/cm^2]  3 p[GeV]  4 beta  5 thp2(fit Q)  6 d/X0  7 grp
C_Z, C_A, C_XG, C_P, C_BETA, C_THP2, C_DX0 = 0, 1, 2, 3, 4, 5, 6


def total_logcf(steps, tau):
    """Exact Moliere log-CF of the summed deflection over all steps.

    Compound-Poisson exponents are additive, so this is exact regardless of
    how Geant4 chose to slice the material -- the property Highland lacks.
    """
    S = np.zeros_like(tau)
    for s in steps:
        chic2, chia2, thff2 = cm.moliere_params(*s[:5])
        if not (chic2 > 0. and chia2 > 0.):
            continue
        y2 = np.logspace(-4., 13., 400)
        dy2 = np.gradient(y2)
        ym2 = thff2 / chia2
        w = dy2 / (1. + y2) ** 2 / (1. + y2 / ym2) ** 2
        arg = np.sqrt(chia2) * tau
        S += (chic2 / chia2) * ((j0(np.outer(arg, np.sqrt(y2))) - 1.) @ w)
    return S


def invert(S, tau, thmax, nth=4000):
    """p(theta_space) from the 2D CF via the Hankel transform."""
    th = np.linspace(1e-9, thmax, nth)
    # p(th) th dth = th dth Int tau J0(tau th) e^{S(tau)} dtau
    p = np.array([np.trapezoid(tau * j0(tau * t) * np.exp(S), tau) for t in th])
    p = np.clip(p, 0., None)
    norm = np.trapezoid(p * 2. * np.pi * th, th)
    return th, p * 2. * np.pi * th / norm       # radial density in th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ntracks", type=int, default=400)
    ap.add_argument("--taumax", type=float, default=4e5)
    a = ap.parse_args()
    logging.setup_logger(__file__, 3, False)

    t = uproot.open(a.file)["tree"]
    arr = t.arrays(["msmoliidx", "msmoliv"], library="np", entry_stop=a.ntracks)

    tau = np.linspace(1e-3, a.taumax, 3000)
    rows = []
    for idx, v in zip(arr["msmoliidx"], arr["msmoliv"]):
        st = np.asarray(v, dtype=float).reshape(-1, 8)
        if len(st) < 3:
            continue
        tot_x0 = st[:, C_DX0].sum()
        q_fit = st[:, C_THP2].sum()                 # Rossi, as the fit uses it
        if not (tot_x0 > 0. and q_fit > 0.):
            continue
        pb = np.median(st[:, C_P] * st[:, C_BETA])
        # Highland on the TOTAL traversal (the only correct way to apply it)
        hl = (0.0136 / pb) ** 2 * tot_x0 * (1. + 0.038 * np.log(tot_x0)) ** 2

        S = total_logcf(st, tau)
        # exact full second moment from the CF curvature at tau->0:
        #   S ~ -(1/4) <th_space^2> tau^2  for the 2D transform
        i = 2
        sp_full = -4. * S[i] / tau[i] ** 2
        th, dens = invert(S, tau, thmax=8. * np.sqrt(max(sp_full, 1e-12)))
        c = np.cumsum(dens) * (th[1] - th[0])
        c /= c[-1]
        k98 = np.searchsorted(c, 0.98)
        m = slice(0, max(k98, 10))
        sp_98 = (np.trapezoid(dens[m] * th[m] ** 2, th[m])
                 / np.trapezoid(dens[m], th[m]))
        # projected (one plane) = space/2
        rows.append((tot_x0, q_fit, hl, sp_full / 2., sp_98 / 2.))

    r = np.array(rows)
    logger.info(f"{len(r)} tracks; median total x/X0 = {np.median(r[:,0]):.3f}")
    q, hl, ex_full, ex_98 = r[:, 1], r[:, 2], r[:, 3], r[:, 4]
    print()
    print("  All quantities are PROJECTED angular variance <theta_plane^2>.")
    print(f"  {'':32s} {'median':>12s}")
    print(f"  {'fit Q (Rossi 15 MeV, no log)':32s} {np.median(q):12.4e}")
    print(f"  {'Highland 13.6 on TOTAL x/X0':32s} {np.median(hl):12.4e}")
    print(f"  {'EXACT Moliere, full variance':32s} {np.median(ex_full):12.4e}")
    print(f"  {'EXACT Moliere, central 98%':32s} {np.median(ex_98):12.4e}")
    print()
    print("  Does each formula reproduce the feature it CLAIMS to describe?")
    print(f"    Rossi/fit-Q  vs exact FULL variance : {np.median(q/ex_full):6.3f}")
    print(f"    Highland     vs exact CENTRAL-98%   : {np.median(hl/ex_98):6.3f}")
    print()
    print("  Cross-comparisons (each formula vs the OTHER feature):")
    print(f"    Rossi/fit-Q  vs exact central-98%   : {np.median(q/ex_98):6.3f}")
    print(f"    Highland     vs exact full variance : {np.median(hl/ex_full):6.3f}")
    print()
    print(f"  How much wider is the full variance than the core?"
          f"  {np.median(ex_full/ex_98):6.3f}x")


if __name__ == "__main__":
    main()
