"""Localize the dead module via geometric z-mirror.

Hotspot good muons (eta-1.6) cross -z modules; mirror good muons (eta+1.6)
cross the geometric +z twins at equal rate. For each -z module compute
presence among good-hotspot muons and compare to its nearest +z twin's
presence among good-mirror muons. A DEAD -z module -> presence_hotspot near 0
while the twin presence_mirror is normal (ratio << 1). Live modules -> ratio ~1.
"""
import os
import numpy as np
import uproot

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "nano_hotspot2_gidx.npz")
RUNTREE = os.path.join(HERE, "runtree_map", "effstudy_fix_0.root")
SUBDET = {0: "PXB", 1: "PXF", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}


def load_map():
    rt = uproot.open(RUNTREE)["runtree"].arrays(
        ["iidx", "rawdetid", "subdet", "layer", "rho", "z", "phi"], library="np")
    n = int(rt["iidx"].max()) + 1
    det = np.zeros(n, np.int64)
    det[rt["iidx"]] = rt["rawdetid"]
    # unique modules
    rd, first = np.unique(rt["rawdetid"], return_index=True)
    mod = {int(rt["rawdetid"][i]): dict(
        subdet=int(rt["subdet"][i]), layer=int(rt["layer"][i]),
        rho=float(rt["rho"][i]), z=float(rt["z"][i]), phi=float(rt["phi"][i]))
        for i in first}
    return det, mod


def presence(muon_of_hit, module_of_hit, flag):
    hm = flag[muon_of_hit]
    mu = muon_of_hit[hm]; mod = module_of_hit[hm]
    B = module_of_hit.max() + 1
    ukey = np.unique(mu.astype(np.int64) * B + mod)
    vals, cnts = np.unique(ukey % B, return_counts=True)
    N = int(flag.sum())
    return {int(v): c / N for v, c in zip(vals, cnts)}, N


def main():
    z = np.load(CACHE)
    det, mod = load_map()
    r = z["cvhpt"] / z["pt"]; reg = z["region"]
    gcnt = z["gcnt"].astype(np.int64); gflat = z["gflat"].astype(np.int64)
    Nmu = len(reg)
    muon_of_hit = np.repeat(np.arange(Nmu), gcnt)
    module_of_hit = det[gflat]

    good_h = (reg == 0) & (np.abs(r - 1) < 0.05)
    good_m = (reg == 1) & (np.abs(r - 1) < 0.05)
    ph, nh = presence(muon_of_hit, module_of_hit, good_h)
    pm, nm = presence(muon_of_hit, module_of_hit, good_m)
    print(f"good hotspot {nh}, good mirror {nm}")

    # +z modules as twin candidates (those hit by mirror muons)
    pos = [(m, mod[m]) for m in pm if m in mod and mod[m]["z"] > 0]
    pos_rho = np.array([d["rho"] for _, d in pos])
    pos_z = np.array([d["z"] for _, d in pos])
    pos_phi = np.array([d["phi"] for _, d in pos])
    pos_sd = np.array([d["subdet"] for _, d in pos])
    pos_ll = np.array([abs(d["layer"]) for _, d in pos])
    pos_id = np.array([m for m, _ in pos])

    rows = []
    for m in ph:
        d = mod.get(m)
        if d is None or d["z"] > 0:
            continue  # only -z hotspot modules
        # nearest +z twin: same subdet & |layer|, min dist in (rho,|z|,phi)
        cand = (pos_sd == d["subdet"]) & (pos_ll == abs(d["layer"]))
        if not cand.any():
            continue
        dphi = np.abs(np.angle(np.exp(1j * (pos_phi[cand] - d["phi"]))))
        dist = ((pos_rho[cand] - d["rho"]) / 2) ** 2 + \
               ((np.abs(pos_z[cand]) - abs(d["z"])) / 5) ** 2 + (dphi / 0.05) ** 2
        j = np.argmin(dist)
        tid = int(pos_id[cand][j])
        pmt = pm[tid]
        rows.append((ph[m] / pmt if pmt else np.nan, ph[m], pmt, m, tid, d, np.sqrt(dist[j])))

    # dead candidates: twin well-populated (muons cross it) but hotspot presence low
    rows = [x for x in rows if x[2] > 0.15 and np.isfinite(x[0])]
    rows.sort()
    print("\n=== lowest presence_hotspot/presence_twin (dead-module candidates) ===")
    print(f"{'ratio':>6} {'p_hot':>5} {'p_mir':>5} {'-z detid':>10} {'+z twin':>10}  sub L   rho     z    phi   dist")
    for ratio, phv, pmv, m, tid, d, dd in rows[:20]:
        print(f"{ratio:6.2f} {phv:5.2f} {pmv:5.2f} {m:10d} {tid:10d}  {SUBDET.get(d['subdet'],'?'):>3} "
              f"{d['layer']:>2} {d['rho']:6.1f} {d['z']:7.1f} {d['phi']:6.2f}  {dd:4.2f}")


if __name__ == "__main__":
    main()
