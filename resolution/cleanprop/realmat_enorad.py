"""e+- without radiation: the norad sim arm against an ionisation-only
reference with the radiative channel off -- isolates scattering + Moller."""
import os, sys, numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
os.environ.setdefault("RES_NO_PHI_CACHE", "1")
import realmat_closure as rc
cmd, pdg = sys.argv[1], int(sys.argv[2])
rc.ARM = "norad"
out = os.path.join(rc.OUT, "model", f"model_{rc.lab(pdg)}_pt3_all4_ion.root")
if cmd == "export":
    g = rc.gkey(pdg)
    env = dict(rc.FOUR_ON); env["CVH_IONONLY"] = "1"
    r = rc._run(g, "runToyModel.py", f"pt={rc.PT} eta={rc.ETA} phi={rc.PHI} partId={pdg} "
                f"output={out} toyGeom={rc.toygeom(g)}", out[:-5] + ".log", rc._env(pdg, env))
    print("export rc", r, out)
elif cmd == "closure":
    import cf_propagation_test as cpt, cf_track_resolution as ctr, fisher_norm as fn
    import hbasis as hb, hadron_probe as hp, cf_nucel_exact as cne, geom_closure as gc
    func = sys.argv[3]; planes = [int(x) for x in sys.argv[4].split(",")]
    hb.set_use_h(True); cne.NUCEL_CHANNEL = False; cpt.RAD_CHANNEL = False
    ctr.IONI_KOKOULIN = 0.0; ctr.IONI_KOKOULIN_TCUT = 0.0
    legs = cpt.load_model(out); hb.bind(legs, out, pdg=pdg)
    sim = rc._sim(pdg)
    fn._scale_cache_load = lambda *a, **k: None; fn._scale_cache_store = lambda *a, **k: None
    sc = fn.plane_scales(legs, func, tag=out, channels=("ioni", "ms"))
    tau = fn.closure_tau(10.0); av = hb.avecs(legs, func)
    for k in planes:
        s = float(sc["sF"][k]); good = sim["valid"][:, k] & np.isfinite(sim[gc.SIM_BRANCH[func]][:, k])
        z = (sim[gc.SIM_BRANCH[func]][good, k] - legs[k][gc.REF_BRANCH[func]]) / s
        e = np.exp(-np.asarray(gc.UCURVE)[:, None] * z[None, :] ** 2)
        phi = cpt.model_phi(legs, k, av[k], s, tau)
        m = np.array([gc.weier_scalar(phi, u, tau) for u in gc.UCURVE])
        r = e.mean(axis=1) - m; er = e.std(axis=1) / np.sqrt(good.sum())
        print(f"{pdg} {func} norad plane {k:2d} n={good.sum()} ", " ".join(f"{1e3*x:+6.2f}" for x in r),
              " | pull", " ".join(f"{x/y:+5.1f}" for x, y in zip(r, er)), flush=True)
