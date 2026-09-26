"""Elastic energy loss per track: the PER-ELEMENT prediction against the sim.

Prediction: sum over MS steps and over the elements of each step's material
(the `_mat` export's material table) of N * P(dE > X), N = w_i mu_i xg, with
the recoil samples of `nucel_g4driver`.
Sim: the `PrimaryLossCensusWatcher` records of the realmat_closure.py samples,
where hadElastic lands in the census's `other` slot.  For pions `other` also
holds hBrems/hPairProd, so the elastic part is `other(elonly) - other(off)`.

usage (NUCEL_DRIVER / NUCEL_CACHEDIR as realmat_closure.py sets them):
    python realmat_recoil_census.py
"""
import glob, hashlib, os, sys, numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
import cf_nucel_exact as cne, cf_propagation_test as cpt
import tail_probe as tp
M = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/realmat_full_260925/model/"
SIM = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/realmat_full_260925/sim"
thr = np.array([1., 10., 50.]); cache = {}
for lab, pdg in [("p", 2212), ("pbar", -2212), ("pim", -211), ("pip", 211), ("Km", -321), ("Kp", 321)]:
    legs = cpt.load_model(M + f"model_{lab}_pt3_all4_mat.root")
    mass = cne.mass_of(pdg); pexp = np.zeros(3); ntot = meanE = 0.0
    for L in legs:
        ms = np.asarray(L["ms"]); p = float(ms[0, 3]); ek = (np.sqrt(p * p + mass ** 2) - mass) * 1e3
        for s in range(len(ms)):
            for z, a, w in cne.step_composition(L, s):
                key = cne._bucket(pdg, z, a, ek)
                if key not in cache:
                    mu, _ = cne._run_driver(pdg, z, a, ek)
                    tag = hashlib.sha256(repr((key, cne.NUCEL_NSAMP, cne.NUCEL_SEED)).encode()).hexdigest()[:16]
                    dE = np.fromfile(sorted(glob.glob(os.path.join(cne._cachedir(), f"drv_{tag}_*.eloss.bin")))[-1], dtype=np.float64)
                    cache[key] = (mu, np.array([(dE > t).mean() for t in thr]), dE.mean())
                mu, pt, mE = cache[key]
                n = w * mu * float(ms[s, 2])
                pexp += n * pt; ntot += n; meanE += n * mE
    print(f"{lab:4} per-element prediction: N_elastic {ntot:.4f}/track  recoil mean {meanE:.4f} MeV/track  "
          f"P(>1) {pexp[0]:.2e}  P(>10) {pexp[1]:.2e}  P(>50) {pexp[2]:.2e}", flush=True)
    io = tp.I_DEPROC + tp.PROCS.index("other")
    cen = {arm: np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, tp.NREC)
                                for f in sorted(glob.glob(f"{SIM}/{lab}_pt3_{arm}_s1??_census.bin"))])[:, io]
           for arm in ("elonly", "off")}
    m = cen["elonly"].mean() - cen["off"].mean()
    ps = [np.mean(cen["elonly"] > t) - np.mean(cen["off"] > t) for t in thr]
    print(f"{lab:4} sim census (elonly - off): recoil mean {m:.4f} MeV/track  "
          f"P(>1) {ps[0]:.2e}  P(>10) {ps[1]:.2e}  P(>50) {ps[2]:.2e}", flush=True)
