#!/usr/bin/env python3
"""How big is the COMMON-p mismatch, really? A scan over f and sigma/m.

The card convolves every candidate in ONE variable v = Int dm/m^p with
p = 1.264, chosen so the conditioning label k_i = sigma_i/m_i^p carries no
information about the true mass. But the width scaling inside the convolution
is physics with a per-candidate answer, sigma_i(m) ~ m^{1+f_i}, f_i = vgf_i.
With a common p the model mis-scales each candidate's width by
(m/m_i)^{p-(1+f_i)} -- a MASS-DEPENDENT width error, which biases m_Z with a
sign set by the sign of p-(1+f_i).

This maps the size of that bias over the range of f and sigma/m the sample
actually spans, using `proto_vmass`'s quadrature (the same machinery that
produced the sigma-vs-k prediction), so the hypothesis can be compared with
the observed per-band closure rather than argued about.

    python3 gate_pmismatch.py [--p 1.264]
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto_vmass import born, truth_density, v_model, shift_needed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=1.264)
    ap.add_argument("--f", type=float, nargs="*",
                    default=[0.10, 0.15, 0.1925, 0.2104, 0.2696, 0.35, 0.42, 0.50])
    ap.add_argument("--srel", type=float, nargs="*",
                    default=[0.00968, 0.01247, 0.01510, 0.025])
    ap.add_argument("--dm", type=float, default=0.02)
    a = ap.parse_args()

    mgrid = np.arange(40.0, 150.0, a.dm)
    pden = born(mgrid)
    pden = pden / (pden.sum() * a.dm)
    obs = np.arange(60.0, 120.0 + 1e-9, a.dm)
    fmod = a.p - 1.0

    print(f"common p = {a.p}, i.e. the model's f = {fmod:.4f}\n")
    print(f"the m_Z shift the COMMON-p v-model needs, MeV "
          f"(0 = the model is right)\n")
    hdr = f"{'f (vgf)':>9s} {'p-(1+f)':>9s} " + " ".join(
        f"{'s/m=' + f'{s:.4f}':>12s}" for s in a.srel)
    print(hdr)
    print("-" * len(hdr))
    for f in a.f:
        row = []
        for s in a.srel:
            sig91 = s * 91.1876
            k = sig91 / 91.1876 ** (1.0 + f)
            T = truth_density(mgrid, pden, obs, k, f)
            M = v_model(mgrid, pden, obs, k, f, -0.5 * (1.0 + fmod), fmod=fmod)
            row.append(shift_needed(obs, T, M))
        print(f"{f:9.4f} {a.p - (1 + f):+9.4f} "
              + " ".join(f"{v:+12.2f}" for v in row))
    print("\nfor comparison, the CERTIFIED per-band v-form closure is")
    print("  |eta|<0.9  -21.08 +- 3.24 ,  0.9-1.6  +12.39 +- 4.16 , "
          "1.6-3.0  ~+34 (not yet converged)")
    print("and the sample's medians are f = 0.2104 / 0.1925 / 0.2696 with "
          "sigma/m = 0.0097 / 0.0125 / 0.0151.")


if __name__ == "__main__":
    main()
