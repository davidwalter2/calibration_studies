"""The deterministic dependence of the q/p change between two planes on the
LOCAL state at the first, in the sim (least squares over events) and in the
model's transport (with and without the layer-path coupling, CF_LAYER_PATH).

    python cleanprop/realmat_losscoupling.py <decay> <leg> <k0> <k1>
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import realmat_ditrack as rd, realmat_closure as rc, cf_propagation_test as cpt
import hbasis as hb, geom_closure as gc

decay, leg, k0, k1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rd.configure(decay); rc.MODEL_TAG = "_mat"
path = rd.model_path(leg); legs = cpt.load_model(path)
hb.bind(legs, path, pdg=rd.LEGS[leg]["pdg"])
sim = rd.load_sim(leg)
LOC = ("qop", "dxdz", "dydz", "locx", "locy")
g = sim["valid"][:, k0] & sim["valid"][:, k1]
X = np.stack([sim[gc.SIM_BRANCH[f]][g, k0] - legs[k0][gc.REF_BRANCH[f]] for f in LOC], axis=1)
y = (sim["qop"][g, k1] - legs[k1]["refqop"]) - X[:, 0]
# robust: trim the Landau tail of y
lo, hi = np.percentile(y, [2, 98]); m = (y > lo) & (y < hi)
Xd = np.column_stack([X[m, 1:], np.ones(m.sum())])
beta, *_ = np.linalg.lstsq(Xd, y[m], rcond=None)
cov = np.linalg.inv(Xd.T @ Xd) * np.var(y[m] - Xd @ beta)
print(f"sim  d(qop)_{k0}->{k1} per local (dxdz, dydz, locx, locy):",
      " ".join(f"{b:+.3e}+-{np.sqrt(c):.1e}" for b, c in zip(beta[:4], np.diag(cov)[:4])))

def model_coef(on):
    cpt.LAYER_PATH = on
    cpt._LAYER_CACHE.clear()
    A_ms, A_io, A_ms0, A_io0 = cpt.step_transports(legs, k1, ioni_start=True)
    # transport from plane k0's END (= start of leg k0+1) to k1: the START
    # transport of leg k0+1's first step
    T = A_ms0[k0 + 1][0] if len(A_ms0[k0 + 1]) else np.eye(5)
    _, mass, bf = hb._state()[0][id(legs)]
    H0, H1 = hb.leg_H(legs, k0, bf, mass)[0], hb.leg_H(legs, k1, bf, mass)[0]
    # local_k1 = H1 T H0^-1 local_k0
    M = H1 @ T @ np.linalg.inv(H0)
    return M[0, 1:] 
for on in (False, True):
    c = model_coef(on)
    print(f"model{' +layer' if on else '       '}:", " ".join(f"{b:+.3e}" for b in c))
