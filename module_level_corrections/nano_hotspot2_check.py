"""Verify the second CVH-efficiency hotspot (eta~-1.6, phi~-1.05) directly in
the produced data NanoAOD (SingleMuon Run2016 F/G/H, TrackFitV722 NanoProdv6).

The user's data/MC efficiency study flagged this cell. Refit-level A/B and
mirror-eta occupancy on J/psi ALCARECO showed NO hole -- but that is a different
sample. Here we look at the actual analysis-level NanoAOD: selected reco muons,
their (eta,phi) occupancy, and whether a localized deficit sits at the hotspot.

Signatures examined:
  1) 2D (eta,phi) occupancy of preselected muons -> localized hole?
  2) mirror-eta control: phi-profile at eta=-1.6 divided by eta=+1.6 (kills the
     smooth phi-sector acceptance modulation) -> data-only eta-odd deficit?
  3) same using CVH-corrected position (cvhEta/cvhPhi) since that is what the
     analysis actually bins in.
"""
import glob
import os
import sys

import numpy as np
import uproot

DIRS = {
    "F": "/scratch/submit/cms/wmass/NanoAOD/SingleMuon/NanoV9Run2016FDataPostVFP_TrackFitV722_NanoProdv6",
    "G": "/scratch/submit/cms/wmass/NanoAOD/SingleMuon/NanoV9Run2016GDataPostVFP_TrackFitV722_NanoProdv6",
    "H": "/scratch/submit/cms/wmass/NanoAOD/SingleMuon/NanoV9Run2016HDataPostVFP_TrackFitV722_NanoProdv6",
}
BR = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge",
      "Muon_cvhPt", "Muon_cvhEta", "Muon_cvhPhi", "Muon_cvhNValidHits",
      "Muon_isGlobal", "Muon_mediumId", "Muon_nTrackerLayers"]


def load(nfiles):
    files = []
    for e in DIRS.values():
        files += sorted(glob.glob(f"{e}/**/*.root", recursive=True))[: nfiles]
    print(f"loading {len(files)} files ...")
    cols = {b: [] for b in BR}
    for i, f in enumerate(files):
        try:
            a = uproot.open(f)["Events"].arrays(BR, library="np")
        except Exception as ex:
            print(f"  skip {f}: {ex}"); continue
        for b in BR:
            cols[b].append(np.concatenate(a[b]) if len(a[b]) else np.array([]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1} files")
    d = {b: np.concatenate(cols[b]) for b in BR}
    print(f"loaded {len(d['Muon_pt'])} muons")
    return d


def presel(d):
    return (d["Muon_mediumId"] == 1) & (d["Muon_isGlobal"] == 1) & \
           (d["Muon_pt"] > 25) & (d["Muon_pt"] < 65)


def mirror(eta, phi, tag):
    pe = np.linspace(-1.6, -0.5, 12)
    ctr = 0.5 * (pe[:-1] + pe[1:])
    m1 = np.abs(eta + 1.6) < 0.15
    m2 = np.abs(eta - 1.6) < 0.15
    hs, _ = np.histogram(phi[m1], bins=pe)
    hc, _ = np.histogram(phi[m2], bins=pe)
    r = np.divide(hs, hc, out=np.full(len(hs), np.nan), where=hc > 0)
    rn = r / np.nanmedian(r)
    sel = (ctr >= -1.15) & (ctr <= -0.95)
    print(f"\n[{tag}] mirror-eta  eta(-1.6)/eta(+1.6), norm  (dip at phi~-1.05 => data-only hole)")
    print("  phi ctr :", np.round(ctr, 2))
    print("  N(-1.6) :", hs)
    print("  N(+1.6) :", hc)
    print("  ratio(n):", np.round(rn, 2))
    print(f"  --> at phi~-1.05: {np.nanmean(rn[sel]):.3f}")


def occ2d(eta, phi, tag):
    ee = np.linspace(-2.1, -1.1, 11)
    pe = np.linspace(-1.6, -0.5, 12)
    h, _, _ = np.histogram2d(eta, phi, bins=[ee, pe])
    # normalize each phi column is wrong; instead show deviation from row-median
    med = np.median(h, axis=0, keepdims=True)
    dev = np.divide(h, med, out=np.full_like(h, np.nan), where=med > 0)
    print(f"\n[{tag}] 2D occupancy / column-median  (rows=eta, cols=phi); <0.85 flagged")
    print("      phi:" + "".join(f"{p:+6.2f}" for p in 0.5 * (pe[:-1] + pe[1:])))
    for i in range(len(ee) - 1):
        row = f"  eta {0.5*(ee[i]+ee[i+1]):+.2f}:"
        for j in range(len(pe) - 1):
            v = dev[i, j]
            row += "     ." if (np.isnan(v) or v > 0.85) else f"{v:6.2f}"
        print(row)


def main():
    nfiles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    d = load(nfiles)
    s = presel(d)
    print(f"preselected muons: {s.sum()}")
    eta, phi = d["Muon_eta"][s], d["Muon_phi"][s]
    ceta, cphi = d["Muon_cvhEta"][s], d["Muon_cvhPhi"][s]
    mirror(eta, phi, "reco eta/phi")
    mirror(ceta, cphi, "cvh eta/phi")
    occ2d(eta, phi, "reco eta/phi")
    occ2d(ceta, cphi, "cvh eta/phi")


if __name__ == "__main__":
    main()
