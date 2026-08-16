#!/usr/bin/env python3
"""Does lateral wander explain the model-vs-Geant4 mode gap?

HYPOTHESIS. The model propagates ONE deterministic reference path and knows only
the material on that path. A simulated ray that scatters away from it traverses
DIFFERENT material -- the "material sampling" limitation, measured at ~6 % of the
energy loss at pT=3. The model's ionization mode sits at +4.295 z against
Geant4's +3.298, a gap of -1.0 z ~ -1.2 MeV, which is the same order as the
1.5 MeV that material sampling has available. If the hypothesis holds, tracks
that stayed NEAR the reference should agree with the model, and the disagreement
should grow with wander.

WHY LOCAL Y, NOT THE FULL DEVIATION. At pT=3 the energy loss itself moves the
track in the BENDING plane: a 0.8 % momentum change shifts the sagitta by
~0.4 mm, comparable to the ~2 mm multiple-scattering wander. Binning on local x
(or on the 2D deviation) would therefore correlate the binning variable with the
q/p residual by construction and manufacture the effect. Local y is the
non-bending direction in the barrel: multiple scattering populates it, energy
loss does not. Multiple scattering is isotropic, so |dy| is an unbiased proxy
for the wander in both directions.

Two independent signatures are reported per wander bin:
  1. median energy loss (pabs[0] - pabs[k]) -- the DIRECT material-sampling
     signature, needing no model at all;
  2. the mode of the standardized q/p residual -- which should approach the
     model's prediction as wander -> 0.

usage: python cleanprop/wander_test.py [--pt 3] [--plane -1] [--nbins 5]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cf_propagation_test import (load_sim, load_model, model_variance,
                                 FUNCTIONALS, SIM_BRANCH, REF_BRANCH)
from cgf_phase0_validate import collect_ioni_steps, empirical_mode, SAMPLES
from cgf_saddlepoint import mode


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", type=int, default=3, choices=sorted(SAMPLES))
    p.add_argument("--plane", type=int, default=-1, help="-1 = outermost")
    p.add_argument("--nbins", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    simpath, modelpath = SAMPLES[args.pt]
    sim = load_sim(simpath, acceptance="perplane")
    legs = load_model(modelpath)

    sd = sim["detid"]
    ld = np.array([l["detid"] for l in legs])
    assert len(sd) == len(ld) and np.all(sd == ld), "sim/model rays differ"

    k = len(legs) - 1 if args.plane < 0 else args.plane
    avec = FUNCTIONALS["qop"]
    var, _, _ = model_variance(legs, k, avec)
    sigma = float(np.sqrt(var))

    good = (sim["valid"][:, k] & sim["valid"][:, 0]
            & np.isfinite(sim["locy"][:, k]) & np.isfinite(sim["pabs"][:, k])
            & np.isfinite(sim["pabs"][:, 0]))
    d = sim[SIM_BRANCH["qop"]][:, k] - legs[k][REF_BRANCH["qop"]]
    z = d / sigma
    # non-bending-plane wander, in microns
    dy = (sim["locy"][:, k] - legs[k]["reflocy"]) * 1e4
    # energy loss between the first and this plane, in MeV
    eloss = (sim["pabs"][:, 0] - sim["pabs"][:, k]) * 1e3

    good &= np.isfinite(z) & np.isfinite(dy) & np.isfinite(eloss)
    w = np.abs(dy)

    steps = collect_ioni_steps(legs, k, avec, sigma)
    spa, _ = mode(steps)

    print(f"pT = {args.pt} GeV, plane {k} (r = "
          f"{np.nanmedian(sim['globr'][:, k]):.1f} cm), {int(good.sum())} rays")
    print(f"model (saddlepoint) mode = {spa:+.3f} z   "
          f"sigma = {sigma:.4g} (qop units)\n")
    print("binned by |local-y deviation| = multiple-scattering wander, which "
          "energy loss does not drive\n")
    print(f"{'bin':>4} {'n':>7} {'<|dy|> um':>10} {'med loss MeV':>13} "
          f"{'G4 mode z':>10} {'med z':>8} | {'model-G4':>9}")
    print("-" * 78)

    qs = np.quantile(w[good], np.linspace(0, 1, args.nbins + 1))
    for i in range(args.nbins):
        sel = good & (w >= qs[i]) & (w < qs[i + 1] if i < args.nbins - 1
                                     else w <= qs[i + 1])
        if sel.sum() < 500:
            continue
        m = empirical_mode(z[sel])
        print(f"{i:4d} {int(sel.sum()):7d} {np.median(w[sel]):10.1f} "
              f"{np.median(eloss[sel]):13.3f} {m:10.3f} "
              f"{np.median(z[sel]):8.3f} | {spa - m:+9.3f}")

    lo = good & (w <= qs[1])
    hi = good & (w >= qs[-2])
    print("-" * 78)
    print(f"least-wandered bin:  mode {empirical_mode(z[lo]):+.3f}, "
          f"median loss {np.median(eloss[lo]):.3f} MeV")
    print(f"most-wandered bin:   mode {empirical_mode(z[hi]):+.3f}, "
          f"median loss {np.median(eloss[hi]):.3f} MeV")
    print(f"model prediction:    mode {spa:+.3f}")
    print("\nIf material sampling drives the gap: loss RISES with wander, and "
          "the least-wandered\nmode approaches the model's.")


if __name__ == "__main__":
    main()
