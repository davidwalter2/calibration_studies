"""Compare the garbage-shift fix policies on the 6-file diag sample:
  old     = pre-fix build          (diag_run)
  drop    = hits excluded          (diag_run_shiftfix2)
  reorder = hits re-inserted       (diag_run_reorder)

Checks: hotspot track counts, recovered-track hit counts, and the q/p pull of
the re-included 798 hit (reorder vs drop on the SAME recovered tracks).
"""
import awkward as ak
import numpy as np
import uproot

B = "/work/submit/david_w/ZMass/calibration_studies/module_level_corrections"
RUNS = {"old": "diag_run", "drop": "diag_run_shiftfix2", "reorder": "diag_run_reorder"}


def load(d):
    return uproot.open(f"{B}/{d}/effstudy_miniaod_fix_0.root")["tree"].arrays(
        ["trackPt", "trackEta", "trackPhi", "nValidHits", "run", "event",
         "refParms", "refCov"], library="ak")


def keys(t):
    return list(zip(ak.to_numpy(t["run"]), ak.to_numpy(t["event"]),
                    np.round(ak.to_numpy(t["trackEta"]), 4),
                    np.round(ak.to_numpy(t["trackPhi"]), 4)))


def main():
    d = {k: load(v) for k, v in RUNS.items()}
    idx = {k: {kk: i for i, kk in enumerate(keys(t))} for k, t in d.items()}

    for k, t in d.items():
        eta = ak.to_numpy(t["trackEta"]); phi = ak.to_numpy(t["trackPhi"])
        hot = (eta > -1.75) & (eta < -1.45) & (phi > -1.30) & (phi < -0.80)
        print(f"{k:8s}: total {len(eta)}, hotspot {hot.sum()}")

    # recovered tracks = in drop AND reorder but NOT in old
    rec = [kk for kk in idx["drop"] if kk not in idx["old"] and kk in idx["reorder"]]
    print(f"\nrecovered tracks present in both drop and reorder: {len(rec)}")
    qd = np.array([ak.to_list(d["drop"]["refParms"][idx["drop"][kk]])[0] for kk in rec])
    qr = np.array([ak.to_list(d["reorder"]["refParms"][idx["reorder"][kk]])[0] for kk in rec])
    ed = np.array([np.sqrt(ak.to_list(d["drop"]["refCov"][idx["drop"][kk]])[0]) for kk in rec])
    nd = np.array([int(d["drop"]["nValidHits"][idx["drop"][kk]]) for kk in rec])
    nr = np.array([int(d["reorder"]["nValidHits"][idx["reorder"][kk]]) for kk in rec])
    print(f"nValidHits: drop mean {nd.mean():.1f}  reorder mean {nr.mean():.1f}  (delta = re-included hits)")
    dq = qr - qd
    pull = dq / ed
    print(f"q/p (reorder - drop) on recovered tracks:")
    print(f"  dq/p: median {np.median(dq):+.2e}  |dq/p| max {np.abs(dq).max():.2e}")
    print(f"  pull (dq/p / sigma_drop): median {np.median(pull):+.3f}  RMS {pull.std():.3f}  max|pull| {np.abs(pull).max():.3f}")

    # surgical check reorder vs drop on ALL common tracks except recovered
    common = [kk for kk in idx["drop"] if kk in idx["reorder"] and kk in idx["old"]]
    qd2 = np.array([ak.to_list(d["drop"]["refParms"][idx["drop"][kk]])[0] for kk in common[:20000]])
    qr2 = np.array([ak.to_list(d["reorder"]["refParms"][idx["reorder"][kk]])[0] for kk in common[:20000]])
    ident = np.mean(qd2 == qr2)
    print(f"\nsurgical: {len(common)} tracks in all three; drop-vs-reorder q/p identical: {ident:.4f} (first 20k)")


if __name__ == "__main__":
    main()
