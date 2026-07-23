"""Name the module behind the hotspot (eta~-1.6, phi~-1.05), fully vectorized.

Within the SAME cell, split muons into good-momentum (|cvhPt/pt-1|<0.05) and
bad-momentum (>0.2). Each muon's per-hit global-parameter indices
(Muon_cvhmergedGlobalIdxs) map -> module rawdetid via the runtree. For each
module compute the fraction of muons (good / bad / mirror-good) that have a hit
on it. The dead/dropped module shows a large good->bad presence DEFICIT in the
hotspot while staying alive in the mirror.
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
        ["iidx", "rawdetid", "subdet", "layer", "stereo", "glued",
         "rho", "z", "eta", "phi"], library="np")
    n = int(rt["iidx"].max()) + 1
    det = np.zeros(n, np.int64)
    det[rt["iidx"]] = rt["rawdetid"]
    # per-module info (first occurrence)
    info = {}
    order = np.argsort(rt["iidx"])
    for i in order:
        rd = int(rt["rawdetid"][i])
        if rd not in info:
            info[rd] = dict(subdet=int(rt["subdet"][i]), layer=int(rt["layer"][i]),
                            stereo=int(rt["stereo"][i]), glued=int(rt["glued"][i]),
                            rho=float(rt["rho"][i]), z=float(rt["z"][i]),
                            eta=float(rt["eta"][i]), phi=float(rt["phi"][i]))
    return det, info


def presence_counts(muon_of_hit, module_of_hit, muon_flag):
    """For muons where muon_flag[muon]=True, count #muons touching each module.
    Returns dict module-> count, and N muons in the subset."""
    hm = muon_flag[muon_of_hit]
    mu = muon_of_hit[hm]
    mod = module_of_hit[hm]
    # unique (muon, module) pairs
    key = mu.astype(np.int64) * (module_of_hit.max() + 1) + mod
    ukey = np.unique(key)
    umod = ukey % (module_of_hit.max() + 1)
    vals, cnts = np.unique(umod, return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist())), int(muon_flag.sum())


def main():
    z = np.load(CACHE)
    det, info = load_map()
    r = z["cvhpt"] / z["pt"]
    reg = z["region"]
    gcnt = z["gcnt"].astype(np.int64)
    gflat = z["gflat"].astype(np.int64)

    Nmu = len(reg)
    muon_of_hit = np.repeat(np.arange(Nmu), gcnt)
    module_of_hit = det[gflat]

    good = (reg == 0) & (np.abs(r - 1) < 0.05)
    bad = (reg == 0) & (np.abs(r - 1) > 0.20)
    mir = (reg == 1) & (np.abs(r - 1) < 0.05)
    print(f"hotspot: {int((reg==0).sum())} muons (good {int(good.sum())}, bad {int(bad.sum())})")
    print(f"mirror : {int((reg==1).sum())} muons (good {int(mir.sum())})")

    cg, ng = presence_counts(muon_of_hit, module_of_hit, good)
    cb, nb = presence_counts(muon_of_hit, module_of_hit, bad)
    cm, nm = presence_counts(muon_of_hit, module_of_hit, mir)

    uni = sorted(cg.keys())
    rows = []
    for m in uni:
        pg = cg.get(m, 0) / ng
        pb = cb.get(m, 0) / nb if nb else 0.0
        pmr = cm.get(m, 0) / nm if nm else 0.0
        rows.append((pg - pb, m, pg, pb, pmr, cg.get(m, 0), cb.get(m, 0)))
    rows.sort(reverse=True)

    print("\n=== modules PRESENT-in-good / ABSENT-in-bad in the hotspot (top 25) ===")
    print(f"{'d(g-b)':>7} {'rawdetid':>10} {'g':>4} {'b':>4} {'mir':>4} {'Ng':>4} {'Nb':>3}  sub  L st gl   rho     z    eta    phi")
    for d, m, pg, pb, pmr, ng_, nb_ in rows[:25]:
        di = info.get(m, {})
        sd = SUBDET.get(di.get("subdet", -1), "?")
        print(f"{d:7.3f} {m:10d} {pg:4.2f} {pb:4.2f} {pmr:4.2f} {ng_:4d} {nb_:3d}  "
              f"{sd:>4} {di.get('layer','?'):>1} {di.get('stereo','?'):>2} {di.get('glued','?'):>2} "
              f"{di.get('rho',0):6.1f} {di.get('z',0):7.1f} {di.get('eta',0):6.2f} {di.get('phi',0):6.2f}")

    # summary: mean modules/muon good vs bad (confirms hit loss)
    nmod_good = np.array([len(np.unique(module_of_hit[muon_of_hit == i])) for i in np.where(good)[0][:2000]])
    nmod_bad = np.array([len(np.unique(module_of_hit[muon_of_hit == i])) for i in np.where(bad)[0]])
    print(f"\nmean unique modules/muon: good={nmod_good.mean():.1f}  bad={nmod_bad.mean():.1f}")


if __name__ == "__main__":
    main()
