#!/usr/bin/env python3
"""IS THE PHI MODULATION GENUINELY CHARGE-EVEN AT THE MODULE, OR A ROTATED
CHARGE-ODD ONE? -- the caveat that has to be closed before the n = 8 / n = 10
harmonics can be called a sagitta effect.

THE MIXING. `genphi` is the azimuth of the momentum at the PCA, but the
modules are sampled at the azimuth the track has when it crosses them, and the
two differ by the BENDING, `phi_module = phi_PCA + q delta` with
`delta ~ 0.3 B r / pT` (0.057 rad at pT = 20 GeV, r = 1 m) and a sign that
flips with the charge. So a modulation that is charge-ODD at the module,

    <z> = q A sin(n phi_mod) = q A sin(n phi_PCA + n q delta),

expands into `q A cos(n delta) sin(n phi_PCA)` -- charge-odd, as it started --
PLUS `A sin(n delta) cos(n phi_PCA)`, which is **charge-EVEN**. With
`n delta = 0.57` at n = 10 the leak is 54 %. A charge-odd source therefore
CAN fake a charge-even harmonic, in the OTHER quadrature and with a strong pT
dependence (`sin(n delta) ~ n delta ~ 1/pT`).

TWO DISCRIMINANTS, both computed here:
  (1) the CHARGE-ODD amplitudes themselves. A charge-odd source at the module
      shows up directly as charge-odd at the PCA, only attenuated by
      cos(n delta) ~ 0.84. If the odd amplitudes are null, there is no
      charge-odd source to rotate.
  (2) the pT DEPENDENCE of the charge-even amplitude. A rotated charge-odd
      source scales as sin(n delta) ~ 1/pT and must fall by a factor ~3 across
      20 -> 60 GeV; a genuine charge-even source is attenuated only by
      cos(n delta), i.e. RISES slightly with pT.
"""
import argparse

import numpy as np

import conv_common as cc


def amps(V, g, ph, n):
    """(even, odd, err) for the cos and sin quadratures at harmonic n."""
    x, q, p = V.x[g], V.q[g], ph[g]
    out = {}
    for lab, f in (("cos", np.cos), ("sin", np.sin)):
        a, e = [], []
        for s in (+1, -1):
            m = q == s
            a.append(2. * np.mean(x[m] * f(n * p[m])))
            e.append(2. * np.std(x[m] * f(n * p[m])) / np.sqrt(m.sum()))
        out[lab] = (0.5 * (a[0] + a[1]) * 1e3, 0.5 * (a[0] - a[1]) * 1e3,
                    0.5 * float(np.hypot(*e)) * 1e3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--harm", type=int, nargs="+", default=[8, 10, 17])
    a = ap.parse_args()
    for path in a.npz:
        V = cc.Var(path)
        g0 = V.good & (np.abs(V.x) < 10)
        ph = V.d["genphi"].astype(float)
        gp = V.d["genpt"].astype(float)
        print(f"\n{'='*78}\n=== PHI QUADRATURES -- {V.name}\n{'='*78}")
        print(f"  {'n':>4s}{'EVEN cos':>11s}{'EVEN sin':>11s}"
              f"{'ODD cos':>11s}{'ODD sin':>11s}{'err':>8s}")
        for n in a.harm:
            o = amps(V, g0, ph, n)
            ec, oc, er = o["cos"]
            es, os_, _ = o["sin"]
            print(f"  {n:4d}{ec:+11.1f}{es:+11.1f}{oc:+11.1f}{os_:+11.1f}"
                  f"{er:8.1f}")
        print("\n  (1) if the ODD amplitudes are null there is no charge-odd "
              "source to rotate.")
        print("  (2) pT dependence of the EVEN amplitude. Rotated-odd ~ 1/pT "
              "(falls x3 over the range);")
        print("      genuine-even ~ cos(n delta) (flat, or rising slightly).")
        qs = np.percentile(gp[g0], [0, 25, 50, 75, 100])
        qs[-1] += 1e-9
        for n in a.harm:
            print(f"    n = {n}:")
            for i in range(4):
                g = g0 & (gp >= qs[i]) & (gp < qs[i + 1])
                o = amps(V, g, ph, n)
                print(f"      pT {qs[i]:5.1f}-{qs[i+1]:5.1f}  "
                      f"even cos {o['cos'][0]:+6.1f}  even sin {o['sin'][0]:+6.1f}"
                      f"  odd cos {o['cos'][1]:+6.1f}  odd sin {o['sin'][1]:+6.1f}"
                      f"   +-{o['cos'][2]:4.1f}")


if __name__ == "__main__":
    main()
