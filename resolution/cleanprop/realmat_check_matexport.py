"""Gate for the material-table export: (1) every pre-existing `legs` branch is
bit-identical between the old and the new model file; (2) the per-step material
indices resolve, and their element tables reproduce the exported effZ/effA and
per-element Moliere sum zzp1OverA to float precision; (3) hydrogen mass along
the path."""
import sys, numpy as np, uproot
sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
import cf_propagation_test as cpt
M = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/realmat_full_260925/model/"
for lab in ["mum", "pim", "pip", "Km", "Kp", "pbar", "p"]:
    fo, fn_ = uproot.open(M + f"model_{lab}_pt3_all4.root"), uproot.open(M + f"model_{lab}_pt3_all4_mat.root")
    to = fo[next(k for k in fo.keys() if k.split(";")[0].endswith("/legs"))]
    tn = fn_[next(k for k in fn_.keys() if k.split(";")[0].endswith("/legs"))]
    ko, kn = set(to.keys()), set(tn.keys())
    diff = []
    for b in sorted(ko & kn):
        a, c = to[b].array(library="np"), tn[b].array(library="np")
        same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, c)) if a.dtype == object else np.array_equal(a, c)
        if not same:
            diff.append(b)
    legs = cpt.load_model(M + f"model_{lab}_pt3_all4_mat.root")
    worst = [0.0, 0.0, 0.0]; xh = xt = 0.0; used = {}
    for L in legs:
        ms = np.asarray(L["ms"])
        for s in range(len(ms)):
            m = L["mattab"][int(L["msmat"][s])]
            z, a_, w = m["Z"], m["A"], m["W"]
            ez, ea = float((w * z).sum()), float((w * a_).sum())
            zz = float((w * z * (z + 1) / a_).sum())
            worst[0] = max(worst[0], abs(ez - ms[s, 0])); worst[1] = max(worst[1], abs(ea - ms[s, 1]))
            worst[2] = max(worst[2], abs(zz - ms[s, 7]) / max(abs(ms[s, 7]), 1e-30))
            xt += ms[s, 2]; xh += ms[s, 2] * float(w[z == 1].sum())
            used[m["name"]] = used.get(m["name"], 0.0) + ms[s, 2]
    print(f"{lab:4} new branches {sorted(kn - ko)}  changed pre-existing branches: {diff or 'NONE'}")
    print(f"     effZ max|d| {worst[0]:.2e}  effA max|d| {worst[1]:.2e}  zzp1OverA max rel {worst[2]:.2e}  "
          f"materials used {len(used)}  path {xt:.4f} g/cm2, hydrogen {xh:.4f} g/cm2 ({100*xh/xt:.2f} %)")
