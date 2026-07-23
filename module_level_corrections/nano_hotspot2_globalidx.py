"""Reload NanoAOD hotspot-region muons WITH per-hit global-parameter indices
(Muon_cvhmergedGlobalIdxs), vectorized with awkward. Keeps only hotspot +
mirror muons, stores the jagged gidx as flat vals + per-muon counts.
"""
import glob
import os
import sys

import awkward as ak
import numpy as np
import uproot

DIRS = {
    "F": "/scratch/submit/cms/wmass/NanoAOD/SingleMuon/NanoV9Run2016FDataPostVFP_TrackFitV722_NanoProdv6",
    "G": "/scratch/submit/cms/wmass/NanoAOD/SingleMuon/NanoV9Run2016GDataPostVFP_TrackFitV722_NanoProdv6",
    "H": "/scratch/submit/cms/wmass/NanoAOD/SingleMuon/NanoV9Run2016HDataPostVFP_TrackFitV722_NanoProdv6",
}
BR = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_cvhPt",
      "Muon_cvhNValidHits", "Muon_isGlobal", "Muon_mediumId",
      "Muon_cvhmergedGlobalIdxs_Counts", "Muon_cvhmergedGlobalIdxs_Vals"]
CACHE = os.path.join(os.path.dirname(__file__), "nano_hotspot2_gidx.npz")

ETA_LO, ETA_HI = -1.75, -1.45
MIR_LO, MIR_HI = 1.45, 1.75
PHI_LO, PHI_HI = -1.30, -0.80


def build(nfiles):
    files = []
    for e in DIRS.values():
        files += sorted(glob.glob(f"{e}/**/*.root", recursive=True))[: nfiles]
    print(f"scanning {len(files)} files ...")
    eta_a, phi_a, pt_a, cvh_a, ch_a, nvh_a, reg_a = ([] for _ in range(7))
    gflat, gcnt = [], []
    for i, f in enumerate(files):
        try:
            a = uproot.open(f)["Events"].arrays(BR, library="ak")
        except Exception as ex:
            print("  skip", ex); continue
        sel = (a["Muon_mediumId"] == 1) & (a["Muon_isGlobal"] == 1) & \
              (a["Muon_pt"] > 25) & (a["Muon_pt"] < 65)
        eta = a["Muon_eta"][sel]; phi = a["Muon_phi"][sel]
        inphi = (phi > PHI_LO) & (phi < PHI_HI)
        hot = inphi & (eta > ETA_LO) & (eta < ETA_HI)
        mir = inphi & (eta > MIR_LO) & (eta < MIR_HI)
        take = hot | mir

        # jagged gidx per selected muon: split flat vals by counts, then apply sel
        cnts_all = a["Muon_cvhmergedGlobalIdxs_Counts"]
        vals_all = a["Muon_cvhmergedGlobalIdxs_Vals"]
        per_muon = ak.unflatten(ak.flatten(vals_all), ak.flatten(cnts_all))  # flat over events
        gidx = ak.unflatten(per_muon, ak.num(cnts_all))                      # regroup into events
        gidx = gidx[sel]

        eta, phi = eta[take], phi[take]
        gidx = gidx[take]
        pt = a["Muon_pt"][sel][take]; cvh = a["Muon_cvhPt"][sel][take]
        ch = a["Muon_charge"][sel][take]; nvh = a["Muon_cvhNValidHits"][sel][take]
        reg = ak.where(hot[take], 0, 1)

        for arr, dst in ((eta, eta_a), (phi, phi_a), (pt, pt_a), (cvh, cvh_a),
                         (ch, ch_a), (nvh, nvh_a), (reg, reg_a)):
            dst.append(ak.to_numpy(ak.flatten(arr)))
        gflat.append(ak.to_numpy(ak.flatten(gidx, axis=None)))
        gcnt.append(ak.to_numpy(ak.flatten(ak.num(gidx, axis=2))))
        if (i + 1) % 100 == 0:
            print(f"  {i+1} files, kept {sum(len(x) for x in eta_a)} muons", flush=True)

    def cat(lst):
        return np.concatenate(lst) if lst else np.array([])
    eta = cat(eta_a); reg = cat(reg_a)
    np.savez(CACHE, eta=eta, phi=cat(phi_a), pt=cat(pt_a), cvhpt=cat(cvh_a),
             charge=cat(ch_a), nvh=cat(nvh_a), region=reg,
             gflat=cat(gflat).astype(np.int32), gcnt=cat(gcnt).astype(np.int32))
    print(f"cached {len(eta)} muons ({int((reg==0).sum())} hotspot, {int((reg==1).sum())} mirror) -> {CACHE}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    build(n)
