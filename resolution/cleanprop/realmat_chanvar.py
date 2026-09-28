"""Channel exponents on one plane: MS (electron term to Tmax vs to the cut) and
the knock-on channel (map / joint / all), Re S at a few tau."""
import os, sys, numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
os.environ.setdefault("RES_NO_PHI_CACHE", "1")
import realmat_closure as rc, cf_propagation_test as cpt, cf_track_resolution as ctr
import fisher_norm as fn, hbasis as hb, hadron_probe as hp, cf_knockon as ck, cf_rows
pdg, func, k = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rc.MODEL_TAG = "_mat"; hb.set_use_h(True)
ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0; ctr.IONI_KOKOULIN_TCUT = 0.0
path = rc.model_path(pdg); legs = cpt.load_model(path); hb.bind(legs, path, pdg=pdg)
fn._scale_cache_load = lambda *a, **kk: None; fn._scale_cache_store = lambda *a, **kk: None
sc = fn.plane_scales(legs, func, tag=path, channels=("ioni", "ms", "rad"))
s = float(sc["sF"][k]); avec = hb.avecs(legs, func)[k]
tau = np.array([0.1, 0.3, 1.0, 3.0])
A_ms, A_io, A_ms0, A_io0 = cpt.step_transports(legs, k, ioni_start=True)
def tot(fn_):
    S = np.zeros(len(tau), dtype=np.complex128)
    for j in range(k + 1):
        S += fn_(legs[j], A_ms[j], A_io[j], A_ms0[j], A_io0[j])
    return S
def ms(leg, Am, Ai, Am0, Ai0):
    rid, _, wb, frac = cf_rows.transport_entries(leg, Am, Am0, avec, s, cpt.MS_NSUB, "length")
    return cf_rows.ms_rows(tau, leg["ms"], rid, wb, frac)
def kn(part):
    def f(leg, Am, Ai, Am0, Ai0):
        if not len(leg["ioni"]): return np.zeros(len(tau), dtype=np.complex128)
        rid, wq, wb, frac = cf_rows.transport_entries(leg, Ai, Ai0, avec, s, ck.KNOCKON_NSUB, "vector")
        return cf_rows.knockon_rows(tau, leg["ioni"], rid, wq, wb, frac, part)
    return f
ck.KNOCKON_JOINT = 0.0; S_ms_tmax = tot(ms)
ck.KNOCKON_JOINT = 1.0; S_ms_cut = tot(ms)
S_map = tot(kn("map")); S_joint = tot(kn("joint"))
pr = lambda n, S: print(f"{n:12s}", " ".join(f"{x.real:+.5f}{x.imag:+.5f}j" for x in S))
print(pdg, func, "plane", k, "sF", s, "tau", tau)
pr("ms_tmax", S_ms_tmax); pr("ms_cut", S_ms_cut); pr("ms cut-tmax", S_ms_cut - S_ms_tmax)
pr("kn map", S_map); pr("kn joint", S_joint)
pr("net change", S_ms_cut - S_ms_tmax + S_joint)
