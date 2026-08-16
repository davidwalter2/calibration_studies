#!/usr/bin/env python3
"""Phase 0 gate: does the saddlepoint mode of the ionization CGF reproduce
Geant4, and how big is the mean-vs-mode shift the fit is currently missing?

Three things are compared per sensor plane, for the q/p residual:

  1. the saddlepoint density from cgf_saddlepoint  (new, no Fourier inversion)
  2. the exact Fourier inversion of the same CF     (existing, validated code)
  3. Geant4 itself                                  (ground truth)

If 1 == 2 the continuation to real argument is right. If 2 == 3 the physics
model is right (already shown in the 2026-08-11 deck). If both hold, the
mode-minus-mean shift printed here is the bias the fit is currently taking,
and Phase 1 is simply to subtract it in the reference.

Uses the pT=3 muon ladder -- the SAME matched sim/model pair as the closure
results. The pair is asserted, not assumed: regen_models.sh mis-targeted the
pt10 model on 2026-08-08 and produced a plausible-looking fake disagreement.

usage: python cgf_phase0_validate.py [--pt 3] [--planes 4,9,14,18]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_propagation_test import (load_sim, load_model, model_phi, model_variance,
                                 step_transports, FUNCTIONALS, SIM_BRANCH,
                                 REF_BRANCH, TAU)
from cgf_saddlepoint import ioni_cgf_derivs, log_density, mode

BASE = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop"
SAMPLES = {
    3:  (f"{BASE}/sim_260808tight_pt3_eta0.30_phi0.70/simstates_*.root",
         f"{BASE}/model/model_mu_pt3_eta0.30.root"),
    40: (f"{BASE}/sim_260808match_pt40_eta0.30_phi0.70/simstates_*.root",
         f"{BASE}/model/model_mu_pt40_eta0.30_phi0.70.root"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", type=int, default=3, choices=sorted(SAMPLES))
    p.add_argument("--planes", default="",
                   help="comma list of plane indices; default = a spread of 5")
    return p.parse_args()


def collect_ioni_steps(legs, k, avec, sigma):
    """All ionization steps up to plane k, with the transport weight folded
    into the qop-per-MeV column -- exactly what model_phi does before calling
    ioni_step_exponent with unit weight."""
    _, A_ioni, _ = step_transports(legs, k)
    out = []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, 10] *= w
        out.append(st)
    return np.concatenate(out) if out else np.zeros((0, 11))


def empirical_mode(z, lo=-8.0, hi=8.0, nb=160):
    """Geant4 mode from the histogram peak, parabolically refined."""
    sel = z[(z > lo) & (z < hi)]
    if len(sel) < 500:
        return np.nan
    h, e = np.histogram(sel, bins=nb, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    i = int(np.argmax(h))
    if 0 < i < len(h) - 1:
        y0, y1, y2 = float(h[i - 1]), float(h[i]), float(h[i + 1])
        den = y0 - 2 * y1 + y2
        if den != 0:
            return float(c[i] - 0.5 * (c[1] - c[0]) * (y2 - y0) / den)
    return float(c[i])


def main():
    args = parse_args()
    simpath, modelpath = SAMPLES[args.pt]
    sim = load_sim(simpath, acceptance="perplane")
    legs = load_model(modelpath)

    sd = sim["detid"]
    ld = np.array([l["detid"] for l in legs])
    assert len(sd) == len(ld) and np.all(sd == ld), (
        f"sim/model module sequences differ ({len(sd)} vs {len(ld)}) -- "
        "different rays, comparison meaningless")

    nl = len(legs)
    if args.planes:
        planes = [int(x) for x in args.planes.split(",")]
    else:
        planes = sorted(set(np.linspace(1, nl - 1, 5).astype(int).tolist()))

    avec = FUNCTIONALS["qop"]
    print(f"pT = {args.pt} GeV muon, {nl} planes, "
          f"{int(sim['valid'][:, 0].sum())} rays\n")
    print("z = (q/p residual) / sigma_model.  mode-mean is what the fit's "
          "reference is currently off by.\n")
    print(f"{'plane':>5} {'r[cm]':>7} {'n':>7} | {'G4 mode':>8} {'SPA mode':>9} "
          f"{'FFT mode':>9} | {'G4 med':>8} {'shift':>8} | {'dp/p':>10}")
    print("-" * 92)

    for k in planes:
        good = sim["valid"][:, k] & np.isfinite(sim[SIM_BRANCH["qop"]][:, k])
        d = sim[SIM_BRANCH["qop"]][:, k][good] - legs[k][REF_BRANCH["qop"]]
        var, _, _ = model_variance(legs, k, avec)
        sigma = float(np.sqrt(var))
        z = d / sigma

        steps = collect_ioni_steps(legs, k, avec, sigma)
        spa_mode, spa_sig = mode(steps)

        # exact Fourier inversion of the same CF, for the ionization only
        phi = model_phi(legs, k, avec, sigma, TAU)
        tu = np.linspace(0.0, TAU[-1], 20000)
        phiu = np.interp(tu, TAU, phi.real) + 1j * np.interp(tu, TAU, phi.imag)
        # FIXED z grid, fine enough to resolve the peak. Do NOT scale it by
        # sqrt(kappa2): the variance is tail-dominated (~366 in z units), so
        # such a grid is ~100x too wide and cannot locate the mode.
        zg = np.linspace(-30.0, 30.0, 2401)
        pz = np.array([np.trapezoid((phiu * np.exp(-1j * tu * zz)).real, tu) / np.pi
                       for zz in zg])
        i = int(np.argmax(pz))
        fft_mode = float(zg[i])
        if 0 < i < len(zg) - 1:
            y0, y1, y2 = pz[i - 1], pz[i], pz[i + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                fft_mode = float(zg[i] - 0.5 * (zg[1] - zg[0]) * (y2 - y0) / den)

        g4_mode = empirical_mode(z)
        # the sample mean of a Landau is unstable and any clip biases it, so
        # quote the robust median; the model's mean is 0 by construction
        # (the CGF is centered, matching the reference's mean-loss subtraction)
        g4_med = float(np.median(z))
        # the model's own mode-minus-mean, converted to a relative momentum
        # shift: dqop = shift * sigma, and dp/p = -dqop/qop
        refqop = legs[k]["refqop"]
        dpp = -spa_mode * sigma / refqop if np.isfinite(spa_mode) else np.nan

        print(f"{k:5d} {np.nanmedian(sim['globr'][:, k]):7.1f} {int(good.sum()):7d} | "
              f"{g4_mode:8.3f} {spa_mode:9.3f} {fft_mode:9.3f} | "
              f"{g4_med:8.3f} {spa_mode:8.3f} | {dpp*1e4:9.2f}e-4")

    print("\nSPA = saddlepoint (new), FFT = exact inversion of the same CF "
          "(existing), G4 = simulation.")
    print("SPA vs FFT tests the real-argument continuation; FFT vs G4 tests "
          "the physics model.")


if __name__ == "__main__":
    main()
