"""Closure of combined functionals (a_f1/s1 +- a_f2/s2) on one ditrack leg:
isolates the model's covariance between two local parameters."""
import os, sys, numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
os.environ.setdefault("RES_NO_PHI_CACHE", "1")
import realmat_ditrack as rd, realmat_closure as rc, cf_propagation_test as cpt
import hbasis as hb, geom_closure as gc, fisher_norm as fn
decay, leg, f1, f2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
planes = [int(x) for x in sys.argv[5].split(",")]
rd.configure(decay); rc.MODEL_TAG = "_mat"; rd._setup_physics()
path = rd.model_path(leg); legs = cpt.load_model(path)
hb.set_use_h(True); hb.bind(legs, path, pdg=rd.LEGS[leg]["pdg"])
sim = rd.load_sim(leg)
a1s, a2s = hb.avecs(legs, f1), hb.avecs(legs, f2)
tau = fn.closure_tau(10.0)
LOC = ("qop", "dxdz", "dydz", "locx", "locy")
def resid(k):
    g = sim["valid"][:, k]
    X = np.stack([sim[gc.SIM_BRANCH[f]][g, k] - legs[k][gc.REF_BRANCH[f]] for f in LOC], axis=1)
    return X
for k in planes:
    X = resid(k)
    e1, e2 = np.eye(5)[LOC.index(f1)], np.eye(5)[LOC.index(f2)]
    z1, z2 = X @ e1, X @ e2
    s1 = 1.4826 * np.median(np.abs(z1 - np.median(z1))); s2 = 1.4826 * np.median(np.abs(z2 - np.median(z2)))
    for sgn in (+1, -1):
        # the sim's LOCAL residual; the model's curvilinear a-vector (H basis)
        a = np.asarray(a1s[k]) / s1 + sgn * np.asarray(a2s[k]) / s2
        z = X @ (e1 / s1 + sgn * e2 / s2)
        s = 1.4826 * np.median(np.abs(z - np.median(z)))
        e = np.exp(-np.asarray(gc.UCURVE)[:, None] * (z / s)[None, :] ** 2)
        phi = cpt.model_phi(legs, k, a, s, tau)
        m = np.array([gc.weier_scalar(phi, u, tau) for u in gc.UCURVE])
        r = e.mean(axis=1) - m; er = e.std(axis=1) / np.sqrt(len(z))
        print(f"{leg} plane {k:2d} {f1}{'+' if sgn > 0 else '-'}{f2}", " ".join(f"{1e3*x:+6.2f}" for x in r),
              "| pull", " ".join(f"{x/y:+5.1f}" for x, y in zip(r, er)), flush=True)
