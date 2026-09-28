"""closure rows (data - model, the realmat_closure.cell set-up) on chosen planes."""
import os, sys, numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
os.environ.setdefault("RES_NO_PHI_CACHE", "1")
import realmat_closure as rc, cf_propagation_test as cpt, cf_track_resolution as ctr
import fisher_norm as fn, hbasis as hb, hadron_probe as hp, cf_nucel_exact as cne
import geom_closure as gc
pdg, func = int(sys.argv[1]), sys.argv[2]
planes = [int(x) for x in sys.argv[3].split(",")]
rc.MODEL_TAG = "_mat"
hb.set_use_h(True); cne.NUCEL_CHANNEL = False; cpt.RAD_CHANNEL = True
ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0; ctr.IONI_KOKOULIN_TCUT = 0.0
path = rc.model_path(pdg); legs = cpt.load_model(path); hb.bind(legs, path, pdg=pdg)
sim = rc._sim(pdg)
fn._scale_cache_load = lambda *a, **k: None; fn._scale_cache_store = lambda *a, **k: None
sc = fn.plane_scales(legs, func, tag=path, channels=("ioni", "ms", "rad"))
tau = fn.closure_tau(10.0); av = hb.avecs(legs, func)
for k in planes:
    s = float(sc["sF"][k]); good = sim["valid"][:, k] & np.isfinite(sim[gc.SIM_BRANCH[func]][:, k])
    z = (sim[gc.SIM_BRANCH[func]][good, k] - legs[k][gc.REF_BRANCH[func]]) / s
    e = np.exp(-np.asarray(gc.UCURVE)[:, None] * z[None, :] ** 2)
    phi = cpt.model_phi(legs, k, av[k], s, tau)
    m = np.array([gc.weier_scalar(phi, u, tau) for u in gc.UCURVE])
    r = e.mean(axis=1) - m; er = e.std(axis=1) / np.sqrt(good.sum())
    print(f"{pdg} {func} plane {k:2d} ", " ".join(f"{1e3*x:+6.2f}" for x in r), " | pull", " ".join(f"{x/y:+4.1f}" for x, y in zip(r, er)), flush=True)
