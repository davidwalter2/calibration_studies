"""Path-AVERAGED clean-propagation closure.

THE PROBLEM THIS FIXES. The standard closure compares the simulated ensemble
against a model CF built from the material along ONE reference path. But the
tracker material is transversely NON-UNIFORM on the scale the tracks spread
over (measured 2026-08-08: tracks in the most-wandered quartile lose 1.6x more
energy in the outer interval than the least-wandered), so the simulated
ensemble samples material the reference never sees:

    <dE(path)>  !=  dE(<path>)

This shows up as a closure residual that GROWS WITH THE ENSEMBLE SPREAD --
+0.022..+0.029 at pT=3 where sigma reaches 2.1 mm, and absent at pT=40 where
sigma never exceeds 137 um. It is an artifact of the TEST (the CVH fit does
not have it: the fit's reference is iterated onto the hits, so it follows the
actual track and samples the material that track really traversed).

WHAT THIS DOES INSTEAD. The correct object when the material is not
transversely uniform is the model CF averaged over the PATH DISTRIBUTION:

    f_model(u) = < weier(phi_path, u) >_paths

Each phi_path is built from the material along its own ray. Rays are sampled
by perturbing the initial direction, all evaluated on the SAME target surfaces
so the CFs can be averaged plane by plane.

NOTE ON WHAT "PER-EVENT" WOULD MEAN. Conditioning on each simulated event's
own path is NOT well defined here: the closure compares x_sim - x_ref, so if
the reference were the event's own path the residual would be zero by
construction. The fit escapes this only because it has hits. The ensemble
average above is the correct generalisation.

KNOWN APPROXIMATION. A rotated initial ray diverges LINEARLY with radius,
whereas a scattered track random-walks (spread ~ r^1.5). The sampling width is
therefore matched in the OUTER tracker, where the effect is largest, and
OVER-spreads the inner layers. Inner layers are where the within-step
displacement term dominates instead, and that is handled separately
(MS_NSUB in cf_propagation_test).

usage:
    python path_averaged_closure.py --sim '<glob>' --rays '<dir>/ray_*.root' \
        [--single <one model.root>] [--probes 0.1 1.0]
"""

import argparse
import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cf_propagation_test as cp   # noqa: E402

from wums import logging  # noqa: E402

logger = logging.child_logger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", required=True)
    p.add_argument("--rays", required=True, help="glob of perturbed model files")
    p.add_argument("--single", default="", help="the single-path model, for reference")
    p.add_argument("--probes", type=float, nargs="+", default=[0.1, 1.0])
    p.add_argument("--layers", type=int, nargs="+", default=[])
    a = p.parse_args()
    logging.setup_logger(__file__, 3, False)

    sim = cp.load_sim(a.sim)
    avec = cp.FUNCTIONALS["locx"]
    tau = cp.TAU

    rayfiles = sorted(glob.glob(a.rays))
    if not rayfiles:
        raise SystemExit(f"no ray models matched {a.rays}")
    logger.info(f"{len(rayfiles)} sampled rays; sim has {len(sim['locx'])} events")

    models = [cp.load_model(f) for f in rayfiles]
    single = cp.load_model(a.single) if a.single else None

    nlay = min(len(m) for m in models)
    layers = a.layers or list(range(nlay))
    print(f"\n{'lay':>4} {'sigma[um]':>10} {'d-m single':>12} {'d-m averaged':>14} {'shift':>10}")
    for k in layers:
        if k >= nlay:
            continue
        # data side: identical for both, uses the SINGLE-path sigma so the two
        # model variants are compared in the same units
        ref = single if single is not None else models[0]
        var, _, _ = cp.model_variance(ref, k, avec)
        sigma = np.sqrt(var)
        d = avec @ (np.vstack([sim[c][:, k] for c in ("qop", "dxdz", "dydz", "locx", "locy")])
                    - np.array([[ref[k]["ref" + c]] for c in ("qop", "dxdz", "dydz", "locx", "locy")]))
        z = d / sigma
        out = []
        for u in a.probes:
            fd = float(np.mean(np.exp(-u * z ** 2)))
            fm_s = cp.weier_scalar(cp.model_phi(ref, k, avec, sigma, tau), u, tau)
            fm_a = float(np.mean([cp.weier_scalar(cp.model_phi(m, k, avec, sigma, tau), u, tau)
                                  for m in models]))
            out.append((u, fd - fm_s, fd - fm_a))
        for u, ds, da in out:
            print(f"{k:4d} {sigma*1e4:10.1f} {ds:+12.5f} {da:+14.5f} {da-ds:+10.5f}   (u={u})")


if __name__ == "__main__":
    main()
