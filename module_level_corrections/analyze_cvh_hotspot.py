"""Check whether the eb96caef CVH fix affects a second data/MC efficiency
hotspot at eta ~ -1.6, phi ~ -1.05 rad.

The A/B refit (buggy vs fixed, on data alignment) already covers the whole
detector. If the fix repairs something there, N_bug/N_fix < 1 in that cell;
if the A/B is flat there, the fix does NOT explain the data/MC discrepancy
(it would be a different effect the fix doesn't address).

Caches the loaded arrays to .npz for fast re-runs.
"""
import glob
import os

import numpy as np
import uproot

BASE = "/ceph/submit/data/user/d/david_w/effstudy_eb96caef"
ORIG = ["0000", "0001", "0003", "0004", "0006"]
CACHE = os.path.join(os.path.dirname(__file__), "cvh_ab_arrays.npz")


def load_all():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return ({k[4:]: z[k] for k in z if k.startswith("fix_")},
                {k[4:]: z[k] for k in z if k.startswith("bug_")})
    out = {}
    for tag in ("fix", "bug"):
        files = sorted(glob.glob(f"{BASE}/night_{tag}_317c988483d/task_*/effstudy_miniaod_{tag}_0.root"))
        files += [f"{BASE}/{tag}_317c988483d/task_{t}/effstudy_miniaod_{tag}_0.root" for t in ORIG]
        files = [f for f in files if os.path.exists(f)]
        d = uproot.concatenate([f"{f}:tree" for f in files],
                               ["trackPt", "trackEta", "trackPhi", "trackCharge"], library="np")
        out[tag] = d
        print(f"{tag}: {len(files)} files, {len(d['trackPt'])} tracks")
    np.savez(CACHE, **{f"fix_{k}": v for k, v in out["fix"].items()},
             **{f"bug_{k}": v for k, v in out["bug"].items()})
    return out["fix"], out["bug"]


def sf_cell(fix, bug, elo, ehi, plo, phi, ptlo=25, pthi=65):
    def n(d):
        m = (d["trackPt"] >= ptlo) & (d["trackPt"] < pthi) & \
            (d["trackEta"] >= elo) & (d["trackEta"] < ehi) & \
            (d["trackPhi"] >= plo) & (d["trackPhi"] < phi)
        return int(m.sum())
    nf, nb = n(fix), n(bug)
    s = nb / nf if nf else float("nan")
    e = (s * (1 - s) / nf) ** 0.5 if nf else float("nan")
    return nf, nb, s, e


def main():
    fix, bug = load_all()

    print("\n=== SANITY: known module 369141860 cell (eta 0.25-0.55, phi 0.80-1.00) ===")
    nf, nb, s, e = sf_cell(fix, bug, 0.25, 0.55, 0.80, 1.00)
    print(f"  N_fix={nf} N_bug={nb} SF={s:.3f}+/-{e:.3f}  (expect ~0.66)")

    print("\n=== NEW HOTSPOT: eta ~ -1.6, phi ~ -1.05 ===")
    for tag, (elo, ehi, plo, phi) in {
        "tight  (dEta,dPhi=0.2)": (-1.70, -1.50, -1.15, -0.95),
        "wider  (dEta,dPhi=0.4)": (-1.80, -1.40, -1.25, -0.85),
        "wider, ALL pT":         (-1.80, -1.40, -1.25, -0.85),
    }.items():
        pt = (0, 1e9) if "ALL pT" in tag else (25, 65)
        nf, nb, s, e = sf_cell(fix, bug, elo, ehi, plo, phi, *pt)
        print(f"  {tag}: N_fix={nf} N_bug={nb} lost={nf-nb} SF={s:.3f}+/-{e:.3f}")

    print("\n=== 2D scan around the hotspot (SF=N_bug/N_fix, pT 25-65) ===")
    eta_edges = np.round(np.arange(-2.0, -1.19, 0.1), 2)
    phi_edges = np.round(np.arange(-1.6, -0.49, 0.1), 2)
    m_f = (fix["trackPt"] >= 25) & (fix["trackPt"] < 65)
    m_b = (bug["trackPt"] >= 25) & (bug["trackPt"] < 65)
    hf, _, _ = np.histogram2d(fix["trackEta"][m_f], fix["trackPhi"][m_f], bins=[eta_edges, phi_edges])
    hb, _, _ = np.histogram2d(bug["trackEta"][m_b], bug["trackPhi"][m_b], bins=[eta_edges, phi_edges])
    sf = np.divide(hb, hf, out=np.full_like(hf, np.nan), where=hf > 30)
    print("  rows=eta, cols=phi; '.'=SF>0.97, digit=10*(1-SF) rounded, ' '=lowstat")
    print("       phi:" + "".join(f"{p:+5.1f}" for p in phi_edges[:-1]))
    for i, elo in enumerate(eta_edges[:-1]):
        row = f"  eta {elo:+.1f}: "
        for j in range(len(phi_edges) - 1):
            v = sf[i, j]
            if np.isnan(v):
                row += "    ."
            elif v > 0.97:
                row += "    ."
            else:
                row += f"{v:5.2f}"
        print(row)


if __name__ == "__main__":
    main()
