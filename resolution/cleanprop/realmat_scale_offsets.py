"""Mean momentum offset of the Geant4 ensemble from the deterministic
reference, per plane: <p_sim - p_ref>/p_ref from q/p, with its statistical error, plus the
median. The closure statistic is even in z and blind to this at the 1e-5 level;
this is the direct test of the reference's MEAN energy loss."""
import sys, json, numpy as np
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
import realmat_closure as rc
import cf_propagation_test as cpt
out = {}
for pdg in rc.ORDER8:
    sim = rc._sim(pdg)
    legs = cpt.load_model(rc.model_path(pdg))
    rows = []
    for k in range(len(legs)):
        ok = sim["valid"][:, k]
        q = sim["qop"][ok, k]
        qr = legs[k]["refqop"]
        # exact relative momentum difference (p_sim - p_ref)/p_ref, sign-safe
        dpe = qr / q - 1.0
        n = dpe.size
        rows.append(dict(k=k, n=int(n), mean=float(dpe.mean()),
                         err=float(dpe.std() / np.sqrt(n)),
                         median=float(np.median(dpe)), std=float(dpe.std()),
                         p_ref=float(abs(1.0 / qr))))
    out[rc.lab(pdg)] = rows
    r = rows[-1]
    print(f"{rc.hp.SPECIES[pdg]['label']:<5} outer plane: mean dp/p = {1e5*r['mean']:+7.2f} +- {1e5*r['err']:.2f} e-5   "
          f"median {1e5*r['median']:+8.1f} e-5   std {1e5*r['std']:7.1f} e-5   p_ref {r['p_ref']:.4f} GeV", flush=True)
    print("      per plane mean (e-5): " + " ".join(f"{1e5*x['mean']:+.1f}" for x in rows[::3] + [rows[-1]]), flush=True)
    print("      per plane err  (e-5): " + " ".join(f"{1e5*x['err']:.1f}" for x in rows[::3] + [rows[-1]]), flush=True)
json.dump(out, open("/work/submit/david_w/ZMass/calibration_studies/resolution/runs/realmat_closure_260925/scale_offsets.json", "w"), indent=1)
