"""The deterministic dependence of the q/p change between two planes on the
LOCAL state at the first, in the sim and in the model's transport (the
export's step Jacobians, which carry the energy loss's direction dependence in
layered material: Geant4ePropagator's layer-chord q/p row).

The q/p change y = (q/p_k1 - ref) - (q/p_k0 - ref) against the local state
X = (q/p, dx/dz, dy/dz, x, y) at plane k0, like for like with the model's
first-order MEAN transport:
  partial slopes: sim -- least squares over the events on all five (the q/p
        regressor matters: the loss's own q/p dependence and upstream
        correlations make q/p and dx/dz covary), untrimmed, with
        heteroscedasticity-robust errors (the loss's Landau tail);
        model -- the q/p row of the transport k0 -> k1 in the local frames
        (H T H^-1);
  marginal slopes along one variable: sim -- cov(y, x_i) / var(x_i), and the
        medians of y in 20 equal-count bins (robust to the tail, but a
        different functional of the loss: indicative); model -- its row
        along the sim's covariance of X, c . cov(X, x_i) / var(x_i).
With --ref <model file> the same model slopes from a second export of the
leg (another propagator build) are printed as well.

    python cleanprop/realmat_losscoupling.py <decay> <leg> <k0> <k1> [--ref <model.root>]
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import realmat_ditrack as rd, realmat_closure as rc, cf_propagation_test as cpt
import hbasis as hb, geom_closure as gc

ap = argparse.ArgumentParser()
ap.add_argument("decay")
ap.add_argument("leg")
ap.add_argument("k0", type=int)
ap.add_argument("k1", type=int)
ap.add_argument("--ref", default=None, help="a second export of the same leg to compare")
a = ap.parse_args()
k0, k1 = a.k0, a.k1
rd.configure(a.decay); rc.MODEL_TAG = "_mat"
LOC = ("qop", "dxdz", "dydz", "locx", "locy")
sim = rd.load_sim(a.leg)


def model_coef(path):
    """The q/p row of the transport from plane k0's end to k1, local frames."""
    legs = cpt.load_model(path)
    hb.bind(legs, path, pdg=rd.LEGS[a.leg]["pdg"])
    A_ms, A_io, A_ms0, A_io0 = cpt.step_transports(legs, k1, ioni_start=True)
    # transport from plane k0's END (= start of leg k0+1) to k1: the START
    # transport of leg k0+1's first step
    T = A_ms0[k0 + 1][0] if len(A_ms0[k0 + 1]) else np.eye(5)
    _, mass, bf = hb._state()[0][id(legs)]
    H0, H1 = hb.leg_H(legs, k0, bf, mass)[0], hb.leg_H(legs, k1, bf, mass)[0]
    M = H1 @ T @ np.linalg.inv(H0)          # local_k1 = M local_k0
    return legs, M[0] - np.eye(5)[0]        # d(q/p change) per unit local state


path = rd.model_path(a.leg)
legs, c = model_coef(path)
g = sim["valid"][:, k0] & sim["valid"][:, k1]
X = np.stack([sim[gc.SIM_BRANCH[f]][g, k0] - legs[k0][gc.REF_BRANCH[f]] for f in LOC], axis=1)
y = (sim["qop"][g, k1] - legs[k1]["refqop"]) - X[:, 0]
CX = np.cov(X.T)


def ols(A, b):
    """Least squares with the sandwich (heteroscedasticity-robust) errors."""
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ beta
    Bi = np.linalg.inv(A.T @ A)
    return beta, np.sqrt(np.diag(Bi @ (A.T * r ** 2) @ A @ Bi))


beta, ebeta = ols(np.column_stack([X, np.ones(len(y))]), y)
mean_marg = [ols(np.column_stack([X[:, i], np.ones(len(y))]), y) for i in range(1, 5)]


def median_slope(i, nb=20):
    q = np.quantile(X[:, i], np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(q, X[:, i], side="right") - 1, 0, nb - 1)
    xc = np.array([np.median(X[idx == b, i]) for b in range(nb)])
    yc = np.array([np.median(y[idx == b]) for b in range(nb)])
    return np.polyfit(xc, yc, 1)[0]


def marginal(cc):
    return np.array([cc @ CX[:, i] / CX[i, i] for i in range(1, 5)])


def row(v, e=None):
    if e is None:
        return " ".join(f"{b:+.3e}" for b in v)
    return " ".join(f"{b:+.3e}+-{d:.1e}" for b, d in zip(v, e))


print(f"d(q/p) from plane {k0} to {k1}, {g.sum()} events")
print("  partial  per (q/p, dx/dz, dy/dz, x, y):")
print("    sim   (least squares)   :", row(beta[:5], ebeta[:5]))
print("    model                   :", row(c))
print("  marginal per (dx/dz, dy/dz, x, y):")
print("    sim   (cov / var)       :", row([m[0][0] for m in mean_marg], [m[1][0] for m in mean_marg]))
print("    sim   (bin medians)     :", row([median_slope(i) for i in range(1, 5)]))
print("    model                   :", row(marginal(c)))
if a.ref:
    _, cr = model_coef(a.ref)
    print(f"  {os.path.basename(a.ref)}:")
    print("    model partial           :", row(cr))
    print("    model marginal          :", row(marginal(cr)))
