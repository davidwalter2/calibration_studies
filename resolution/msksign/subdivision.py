"""Step-subdivision invariance of the accumulated multiple-scattering block.

The propagator accumulates, step by step,

    Q  <-  J Q J^T  +  q_step ,

with J the step's transport (J(4,1) = +L, J(3,2) = +L cos(lam); see
`transport_sign.py`) and q_step the within-step block written by
`Geant4ePropagator::PropagateErrorMSC`:

    S2 = DD = Var(theta),  S1 = DD l^2/3 = Var(offset),  S3 = DD l/2 = Cov,

with DD = 2.25e-4 l (q/p beta)^2 / Xs LINEAR in the step length and with no
logarithmic term, so the scattering power of a slab is EXACTLY additive over
any subdivision of it and the accumulated Q of the slab must be too.

The code writes  res(2,3) = +S3/CLA  (phi, xt: the bending projection) but
res(1,4) = -S3  (lam, yt: the non-bending one). This script shows, for a
straight slab of length L split into k equal steps, what each sign gives.
"""
import numpy as np


def accumulate(L, DDtot, k, sign):
    """2x2 (angle, offset) recursion, k equal sub-steps of a uniform slab."""
    l = L / k
    d = DDtot / k
    Q = np.zeros((2, 2))
    J = np.array([[1.0, 0.0], [l, 1.0]])          # theta' = theta ; y' = y + l theta
    q = np.array([[d, sign * d * l / 2.0], [sign * d * l / 2.0, d * l * l / 3.0]])
    for _ in range(k):
        Q = J @ Q @ J.T + q
    return Q


def main():
    L, DD = 10.0, 1.0
    exact = {"Var(theta)": DD, "Var(offset)": DD * L * L / 3.0, "Cov": DD * L / 2.0}
    print("uniform slab, L = %g, DD_total = %g" % (L, DD))
    print("exact:  Var(theta) = %.6f   Var(offset) = %.6f   Cov = %.6f"
          % (exact["Var(theta)"], exact["Var(offset)"], exact["Cov"]))
    print()
    for sign, name in ((+1.0, "res(1,4) = +S3   (the fix / what res(2,3) already does)"),
                       (-1.0, "res(1,4) = -S3   (the code as written)")):
        print(name)
        print(f"{'k':>5} {'Var(theta)':>13} {'Var(offset)':>13} {'ratio':>9}"
              f" {'Cov':>13} {'ratio':>9}")
        for k in (1, 2, 4, 8, 16, 32, 39, 64, 128):
            Q = accumulate(L, DD, k, sign)
            print(f"{k:5d} {Q[0,0]:13.6f} {Q[1,1]:13.6f} {Q[1,1]/exact['Var(offset)']:9.4f}"
                  f" {Q[0,1]:13.6f} {Q[0,1]/exact['Cov']:9.4f}")
        print()
    print("closed form for k equal steps:")
    print("   +:  Var(offset) = DD L^2/3            Cov = DD L/2            (exact, any k)")
    print("   -:  Var(offset) = DD L^2/3 (1-3/k+3/k^2)   Cov = DD L/2 (1-2/k)")
    print("   => the minus sign is a deficit of 3/k - 3/k^2 in the offset variance")
    print("      and 2/k in the angle-offset term, k = the number of steps over the leg.")


if __name__ == "__main__":
    main()
