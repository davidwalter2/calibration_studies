"""The energy loss of one leg against the TRUE incidence at its first plane,
from the sim's own crossing point and direction: c = d . r_hat at the scored
cylinder (the path through a coaxial shell is t / c).  Pure path-length
physics predicts d<dE>/d ln(1/c) ~ <dE> and no dependence on the crossing
azimuth phi_P (rotational symmetry).

    python cleanprop/realmat_incidence.py <decay> <leg> <k0>-<k1> [<k0>-<k1> ...]
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import realmat_ditrack as rd, realmat_closure as rc
from toy_loader import plane_frames

decay, leg = sys.argv[1], sys.argv[2]
rd.configure(decay); rc.MODEL_TAG = "_mat"
sim = rd.load_sim(leg)
ns = {}
exec(open(rd.planes_path(leg)).read(), ns)
o, R = plane_frames(ns["origin"], ns["normal"], ns["uaxis"])
mass = rd.LEGS[leg]["mass"] if "mass" in rd.LEGS[leg] else None
import hadron_probe as hp
mass = hp.SPECIES[rd.LEGS[leg]["pdg"]]["mass"] * 1e-3

def state(k, m):
    lp = np.stack([sim["locx"][m, k], sim["locy"][m, k], sim["locz"][m, k]], axis=1)
    ld = np.stack([sim["dxdz"][m, k], sim["dydz"][m, k], np.ones(m.sum())], axis=1)
    P = o[k] + lp @ R[k]
    d = ld @ R[k]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rh = P.copy(); rh[:, 2] = 0.0; rh /= np.linalg.norm(rh, axis=1, keepdims=True)
    return P, d, np.abs(np.sum(d * rh, axis=1))

for pair in sys.argv[3:]:
    k0, k1 = (int(x) for x in pair.split("-"))
    m = sim["valid"][:, k0] & sim["valid"][:, k1]
    P, d, c = state(k0, m)
    p0, p1 = sim["pabs"][m, k0], sim["pabs"][m, k1]
    dE = (np.sqrt(p0 ** 2 + mass ** 2) - np.sqrt(p1 ** 2 + mass ** 2)) * 1e3     # MeV
    lnic = -np.log(c)
    phiP = np.arctan2(P[:, 1], P[:, 0]); phiP -= np.median(phiP)
    lam = np.arcsin(np.clip(d[:, 2], -1, 1))
    X = np.column_stack([lnic - lnic.mean(), phiP, (p0 - p0.mean()), np.ones(len(c))])
    b, *_ = np.linalg.lstsq(X, dE, rcond=None)
    res = dE - X @ b
    XtX = np.linalg.inv(X.T @ X)
    err = np.sqrt(np.diag(XtX @ (X.T * res ** 2) @ X @ XtX))
    # medians in bins of ln(1/c)
    q = np.quantile(lnic, np.linspace(0, 1, 21)); mids, meds, means = [], [], []
    for a, bb in zip(q[:-1], q[1:]):
        s = (lnic >= a) & (lnic < bb); mids.append(lnic[s].mean()); meds.append(np.median(dE[s])); means.append(dE[s].mean())
    print(f"leg {k0}->{k1}: r {sim['globr'][m, k0].mean():.1f}->{sim['globr'][m, k1].mean():.1f} cm, "
          f"<dE> {dE.mean():.3f} MeV (median {np.median(dE):.3f}), <1/c> {np.mean(1/c):.4f}, "
          f"incidence {np.degrees(np.arccos(np.median(c))):.1f} deg, rms ln(1/c) {lnic.std():.4f}")
    print(f"   d<dE>/d ln(1/c) = {b[0]:+.3f} +- {err[0]:.3f} MeV  ( / <dE> = {b[0] / dE.mean():+.2f} )   "
          f"median slope {np.polyfit(mids, meds, 1)[0]:+.3f}   d<dE>/d phi_P = {b[1]:+.3f} +- {err[1]:.3f} MeV/rad")
