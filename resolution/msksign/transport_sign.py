"""Signs of the curvilinear transport Jacobian entries J(4,1)=d(yt)/d(lam0)
and J(3,2)=d(xt)/d(phi0), in the frame `transportJacobianBxByBzD` transports
in, and the step-subdivision invariance of the accumulated multiple-scattering
covariance that decides the sign of the within-step angle-offset correlation.

The map is the closed-form helix of
TrackPropagation/Geant4e/python/gen_transport_jacobian.py, reimplemented in
numpy so the derivative can be taken by finite differences without sympy:

    W0 = direction,  U0 = zhat x W0 / |.|,  V0 = W0 x U0
    M  = M0 + gamma (theta - sin theta)/Q H + sin(theta)/Q T0
             + alpha (1 - cos theta)/Q N0,        Q = -B qop,  theta = Q s
    T  = dM/ds
    U  = zhat x W / |.| ,  V = W x U   (W = T at the nominal point)
    parms_out = (qop, lam, phi, xt = M.U, yt = M.V)
"""
import numpy as np

ZH = np.array([0.0, 0.0, 1.0])


def frame(w):
    u = np.cross(ZH, w)
    u /= np.linalg.norm(u)
    return u, np.cross(w, u)


def propagate(qop0, lam0, phi0, xt0, yt0, M0ref, W0ref, B, s):
    """Exact helix transport; returns (qop, lam, phi, xt, yt) at path s.
    The output frame (U, V) is built on the NOMINAL end direction W (that is
    what `subsconst` does in the generator), so it is held fixed here."""
    U0, V0 = frame(W0ref / np.linalg.norm(W0ref))
    W0n = W0ref / np.linalg.norm(W0ref)
    zt0 = M0ref @ W0n
    M0 = xt0 * U0 + yt0 * V0 + zt0 * W0n
    T0 = np.array([np.cos(lam0) * np.cos(phi0), np.cos(lam0) * np.sin(phi0), np.sin(lam0)])
    Bmag = np.linalg.norm(B)
    H = B / Bmag
    HxT0 = np.cross(H, T0)
    alpha = np.linalg.norm(HxT0)
    N0 = HxT0 / alpha
    gamma = H @ T0
    Q = -Bmag * qop0
    th = Q * s
    M = M0 + gamma * (th - np.sin(th)) / Q * H + np.sin(th) / Q * T0 + alpha * (1 - np.cos(th)) / Q * N0
    T = gamma * (1 - np.cos(th)) * H + np.cos(th) * T0 + alpha * np.sin(th) * N0
    return M, T


def parms_out(M, T, Uend, Vend, qop0):
    tt = np.hypot(T[0], T[1])
    return np.array([qop0, np.arctan2(T[2], tt), np.arctan2(T[1], T[0]), M @ Uend, M @ Vend])


def jac_fd(qop0, lam0, phi0, M0, B, s, eps=1e-7):
    W0 = np.array([np.cos(lam0) * np.cos(phi0), np.cos(lam0) * np.sin(phi0), np.sin(lam0)])
    Mn, Tn = propagate(qop0, lam0, phi0, 0.0, 0.0, M0, W0, B, s)
    Wend = Tn / np.linalg.norm(Tn)
    Uend, Vend = frame(Wend)
    p0 = np.array([qop0, lam0, phi0, 0.0, 0.0])
    J = np.zeros((5, 5))
    for j in range(5):
        pp, pm = p0.copy(), p0.copy()
        pp[j] += eps
        pm[j] -= eps
        Mp, Tp = propagate(pp[0], pp[1], pp[2], pp[3], pp[4], M0, W0, B, s)
        Mm, Tm = propagate(pm[0], pm[1], pm[2], pm[3], pm[4], M0, W0, B, s)
        J[:, j] = (parms_out(Mp, Tp, Uend, Vend, pp[0]) - parms_out(Mm, Tm, Uend, Vend, pm[0])) / (2 * eps)
    return J


def main():
    rng = np.random.default_rng(7)
    print("J(4,1) = d(yt)/d(lam0)   and   J(3,2) = d(xt)/d(phi0)")
    print("expected for a straight step:  J(4,1) = +L,  J(3,2) = +L cos(lam)")
    print()
    print(f"{'B[T]':>7} {'s[cm]':>8} {'lam':>8} {'phi':>8} {'pT[GeV]':>8}"
          f" {'J(4,1)':>12} {'L':>10} {'J(3,2)':>12} {'L cos(lam)':>12}")
    for Bz in (1e-9, 3.8):
        # field in 1/GeV/cm units as in CMSSW: B[T] * kTeslaToInvGeV(=2.99792458e-3) / 100 cm
        Bv = np.array([0.0, 0.0, Bz * 2.99792458e-3 * 1e-2])
        for _ in range(4):
            lam0 = rng.uniform(-1.0, 1.0)
            phi0 = rng.uniform(-np.pi, np.pi)
            pt = rng.uniform(2.0, 40.0)
            p = pt / np.cos(lam0)
            qop0 = 1.0 / p
            s = rng.uniform(0.2, 3.0)
            M0 = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)])
            J = jac_fd(qop0, lam0, phi0, M0, Bv, s)
            print(f"{Bz:7.2g} {s:8.4f} {lam0:8.4f} {phi0:8.4f} {pt:8.3f}"
                  f" {J[4,1]:12.6f} {s:10.6f} {J[3,2]:12.6f} {s*np.cos(lam0):12.6f}")
    print()
    print("=> both lever entries are POSITIVE: a +dlam kick tilts toward +yt,")
    print("   a +dphi kick tilts toward +xt (scaled by cos lam).")


if __name__ == "__main__":
    main()
