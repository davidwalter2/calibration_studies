"""CVH-quality map in the data NanoAOD around the hotspot (eta~-1.6, phi~-1.05).

Occupancy showed no hole there (mirror-eta ~1.0). So test whether the CVH refit
QUALITY degrades locally: per (eta,phi) cell compute the rate of soft/hard
cvhPt/pt outliers, mean cvhNValidHits, and edmval tails. Caches muon arrays.
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
BR = ["Muon_pt", "Muon_eta", "Muon_phi",
      "Muon_cvhPt", "Muon_cvhEta", "Muon_cvhPhi", "Muon_cvhNValidHits",
      "Muon_cvhEdmval", "Muon_isGlobal", "Muon_mediumId"]
CACHE = os.path.join(os.path.dirname(__file__), "nano_hotspot2_arrays.npz")


def load(nfiles):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {b: z[b] for b in z.files}
    files = []
    for e in DIRS.values():
        files += sorted(glob.glob(f"{e}/**/*.root", recursive=True))[: nfiles]
    print(f"loading {len(files)} files ...")
    cols = {b: [] for b in BR}
    for i, f in enumerate(files):
        try:
            a = uproot.open(f)["Events"].arrays(BR, library="np")
        except Exception as ex:
            print("  skip", ex); continue
        for b in BR:
            cols[b].append(np.concatenate(a[b]) if len(a[b]) else np.array([]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1} files")
    d = {b: np.concatenate(cols[b]) for b in BR}
    np.savez(CACHE, **d)
    print(f"loaded+cached {len(d['Muon_pt'])} muons")
    return d


def cellmap(eta, phi, val, ee, pe, agg):
    """agg(values_in_cell) -> scalar; returns 2D array."""
    out = np.full((len(ee) - 1, len(pe) - 1), np.nan)
    ie = np.digitize(eta, ee) - 1
    ip = np.digitize(phi, pe) - 1
    for i in range(len(ee) - 1):
        for j in range(len(pe) - 1):
            m = (ie == i) & (ip == j)
            if m.sum() >= 20:
                out[i, j] = agg(val[m])
    return out


def show(mat, ee, pe, title, flag=None, fmt="{:6.2f}"):
    print(f"\n[{title}] rows=eta cols=phi" + (f"  (flag {flag})" if flag else ""))
    print("      phi:" + "".join(f"{p:+6.2f}" for p in 0.5 * (pe[:-1] + pe[1:])))
    for i in range(len(ee) - 1):
        row = f"  eta {0.5*(ee[i]+ee[i+1]):+.2f}:"
        for j in range(len(pe) - 1):
            v = mat[i, j]
            if np.isnan(v):
                row += "     ."
            elif flag and flag(v):
                row += ("*" + fmt.format(v).strip()).rjust(6)
            else:
                row += fmt.format(v)
        print(row)


def main():
    nfiles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    d = load(nfiles)
    s = (d["Muon_mediumId"] == 1) & (d["Muon_isGlobal"] == 1) & \
        (d["Muon_pt"] > 25) & (d["Muon_pt"] < 65)
    eta, phi = d["Muon_eta"][s], d["Muon_phi"][s]
    r = (d["Muon_cvhPt"][s] / d["Muon_pt"][s])
    nvh = d["Muon_cvhNValidHits"][s]
    edm = d["Muon_cvhEdmval"][s]
    print(f"presel muons: {s.sum()}")

    ee = np.round(np.arange(-2.0, -1.09, 0.1), 2)
    pe = np.round(np.arange(-1.6, -0.49, 0.1), 2)

    # overall reference rates
    print("\n=== overall (presel) reference rates ===")
    print(f"  soft outlier |cvhpt/pt-1|>0.05 : {np.mean(np.abs(r-1)>0.05):.4f}")
    print(f"  hard outlier |cvhpt/pt-1|>0.20 : {np.mean(np.abs(r-1)>0.20):.4f}")
    print(f"  edmval>100                     : {np.mean(edm>100):.4f}")
    print(f"  mean cvhNValidHits             : {nvh.mean():.2f}")

    m_soft = cellmap(eta, phi, (np.abs(r - 1) > 0.05).astype(float), ee, pe, np.mean)
    m_hard = cellmap(eta, phi, (np.abs(r - 1) > 0.20).astype(float), ee, pe, np.mean)
    m_edm = cellmap(eta, phi, (edm > 100).astype(float), ee, pe, np.mean)
    m_nvh = cellmap(eta, phi, nvh, ee, pe, np.mean)

    show(m_soft, ee, pe, "soft-outlier rate |cvhpt/pt-1|>0.05", flag=lambda v: v > 0.08, fmt="{:6.3f}")
    show(m_hard, ee, pe, "hard-outlier rate |cvhpt/pt-1|>0.20", flag=lambda v: v > 0.01, fmt="{:6.3f}")
    show(m_edm, ee, pe, "edmval>100 rate", flag=lambda v: v > 0.05, fmt="{:6.3f}")
    show(m_nvh, ee, pe, "mean cvhNValidHits", flag=lambda v: v < 12, fmt="{:6.1f}")


if __name__ == "__main__":
    main()
