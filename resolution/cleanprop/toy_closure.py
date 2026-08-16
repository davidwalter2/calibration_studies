#!/usr/bin/env python3
"""First closure number from the toy homogeneous clean-propagation test.

Same statistic as the real test: the bounded average <e^{-u z^2}> measured on
the simulated sample against the model's own value, obtained from the model
characteristic function by the Weierstrass transform. Small u weights the tails,
u ~ 1 the core.

What the toy removes relative to the real geometry:
  * ACCEPTANCE. 100 % of rays cross every surface (2000/2000), against 94.4 % at
    pT=3 in the real tracker where the missing 5.6 % were TAIL-SELECTIVE and
    turned out to be the single largest error in the whole study.
  * MATERIAL SAMPLING. Every path through the same z-extent crosses identical
    material, so <dE(path)> = dE(<path>) exactly. Measured at +11 % of the loss
    across wander quintiles in the real geometry.

What it does NOT remove, and what therefore limits it:
  * The model's ionization peak shape. The wander test showed material sampling
    explains only ~40 % of the model-vs-Geant4 mode gap; ~60 % is a genuine
    model difference and survives here.
  * The cylinder-vs-tangent-plane offset, <= 7.9 um against mm-scale deviations.

IMPORTANT on interpretation: the toy matches the real tracker's MEAN loss
(25.13 vs 24.99 MeV) but NOT its Landau shape (mean-minus-median 13.7 % vs
9.0 %), because the material here is continuous rather than in thin layers and
the mean/median ratio is step-size dependent. These numbers validate the MODEL;
they do not transfer numerically to the real detector.

usage: python cleanprop/toy_closure.py --sim toystates.root --model model_toy.root
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cf_propagation_test import (load_model, model_variance, model_phi,
                                 weier_scalar, FUNCTIONALS, SIM_BRANCH,
                                 REF_BRANCH, TAU)
from toy_loader import load_toy_sim

PLANES = ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
          "Analysis/HitAnalyzer/test/toyPlanes_pt3.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--planes", default=PLANES)
    p.add_argument("--probes", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--funcs", nargs="+", default=["locx", "qop"])
    return p.parse_args()


def main():
    args = parse_args()
    ns = {}
    exec(open(args.planes).read(), ns)
    sim = load_toy_sim(args.sim, ns["origin"], ns["normal"], ns["uaxis"])
    legs = load_model(args.model)

    nlayer = min(sim["valid"].shape[1], len(legs))
    print(f"toy closure: {sim['ntot']} events, {sim['nkept']} complete "
          f"({100.*sim['nkept']/sim['ntot']:.1f} %), {nlayer} planes\n")

    for name in args.funcs:
        avec = FUNCTIONALS[name]
        print(f"=== {name} ===")
        hdr = "  ".join(f"u={u:<6g}" for u in args.probes)
        print(f"{'k':>3} {'r[cm]':>7} {'n':>6} {'sigma':>11}  {hdr}")
        print("-" * (34 + 12 * len(args.probes)))
        allrows = []
        for k in range(nlayer):
            good = sim["valid"][:, k] & np.isfinite(sim[SIM_BRANCH[name]][:, k])
            if good.sum() < 100:
                continue
            d = sim[SIM_BRANCH[name]][good, k] - legs[k][REF_BRANCH[name]]
            var, _, _ = model_variance(legs, k, avec)
            sigma = float(np.sqrt(var))
            z = d / sigma
            phi = model_phi(legs, k, avec, sigma, TAU)
            row = []
            for u in args.probes:
                fm = weier_scalar(phi, u, TAU)
                fd = float(np.mean(np.exp(-u * z ** 2)))
                row.append(fd - fm)
            allrows.append(row)
            cells = "  ".join(f"{v:+9.5f}" for v in row)
            print(f"{k:3d} {np.nanmedian(sim['globr'][good, k]):7.1f} "
                  f"{int(good.sum()):6d} {sigma:11.4e}  {cells}")
        a = np.array(allrows)
        if len(a):
            print("-" * (34 + 12 * len(args.probes)))
            print(f"{'mean':>3} {'':>7} {'':>6} {'':>11}  "
                  + "  ".join(f"{v:+9.5f}" for v in a.mean(axis=0)))
            print(f"{'rms':>3} {'':>7} {'':>6} {'':>11}  "
                  + "  ".join(f"{v:9.5f}" for v in a.std(axis=0)))
        print()

    print("data - model. The plane-to-plane RMS is the number to watch: in the "
          "real\ngeometry it was ~3e-3 and momentum-INDEPENDENT, which no "
          "scattering-strength\nerror can produce, and it was never explained.")


if __name__ == "__main__":
    main()
