#!/usr/bin/env python3
"""TAIL study: WHICH SUBDETECTOR carries the excess chi2.

The excess chi2 -- the quantity the residual tail is -- rises with |eta| on DY
and FALLS with it on the gun, so whatever it is lives in the forward
subdetectors.  This counts the hits a candidate actually has per subdetector
(the `parmtype == 0` entries of `globalidxv`, one per hit module, against the
`runtree`) and measures the chi2 excess against each count.

usage: python3 tail_subdet.py --npz <npz> --files '<prod>/task_*/*.root'
"""
import argparse, glob, os, sys
import numpy as np
import uproot
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402

SUBDET = {0: "BPix", 1: "FPix", 2: "TIB", 3: "TOB", 4: "TID", 5: "TEC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--need-bs", type=int, default=1)
    a = ap.parse_args()
    files = sorted(glob.glob(a.files))
    rt = uproot.open(files[0] + ":runtree").arrays(
        ["iidx", "parmtype", "subdet", "dx", "dy", "dz",
         "lxx", "lxy", "lxz"], library="np")
    nmax = int(rt["iidx"].max()) + 1
    sub = np.full(nmax, -1, np.int8)
    dlx = np.full(nmax, np.nan)
    isal = np.zeros(nmax, bool)
    m0 = rt["parmtype"] == 0
    sub[rt["iidx"][m0]] = rt["subdet"][m0]
    dr = np.stack([rt["dx"][m0], rt["dy"][m0], rt["dz"][m0]], axis=1)
    lx = np.stack([rt["lxx"][m0], rt["lxy"][m0], rt["lxz"][m0]], axis=1)
    dlx[rt["iidx"][m0]] = (dr * lx).sum(1)
    isal[rt["iidx"][m0]] = True

    rows, dmx = [], []
    for fn in files:
        d = uproot.open(fn + ":tree").arrays(["globalidxv"], library="np")
        for i in range(len(d["globalidxv"])):
            gi = np.asarray(d["globalidxv"][i], np.int64)
            gi = gi[isal[gi]]
            s = sub[gi]
            rows.append([int((s == k).sum()) for k in sorted(SUBDET)])
            v = np.abs(dlx[gi]) * 1e4
            dmx.append([np.nanmax(v[s == k]) if (s == k).any() else 0.0
                        for k in sorted(SUBDET)])
    N = np.array(rows)
    D = np.array(dmx)

    d = dict(np.load(a.npz, allow_pickle=False))
    assert len(d["run"]) == len(N)
    base, _, _ = TC.baseline(d, need_bs=bool(a.need_bs), max_abs_vtxz=0.0,
                             chi2=0.0, verbose=False)
    if "genidx_plus" in d:
        cls, names, _ = GB.classify(d)
        base = base & (cls == names.index("signal"))
    zv = np.asarray(d["vtxz"], float)
    nd = np.asarray(d["ndof"], float)
    s_ = np.asarray(d.get("vtxdchi2", zv ** 2), float)
    ns = 1.0
    if a.need_bs and "bschi2fit" in d:
        s_ = s_ + np.asarray(d["bschi2fit"], float); ns += 3.0
    ndr = np.maximum(nd - ns, 1.0)
    chired = (np.asarray(d["chisq"], float) - s_) / ndr
    u = stats.chi2.sf(chired * ndr, ndr)
    etamax = np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"]))

    print(f"# {int(base.sum())} candidates; median hits per subdetector:")
    print("  " + "  ".join(f"{SUBDET[k]} {np.median(N[base, k]):.1f}"
                           for k in sorted(SUBDET)))
    print("\n=== the chi2 excess against the hit count per subdetector")
    for k in sorted(SUBDET):
        print(f"\n  -- {SUBDET[k]}")
        vals = np.unique(N[base, k])
        edges = np.percentile(N[base, k], [0, 25, 50, 75, 100]) if len(vals) > 6 \
            else np.append(vals, vals[-1] + 1)
        edges = np.unique(edges)
        for i in range(len(edges) - 1):
            b = base & (N[:, k] >= edges[i]) & (N[:, k] < edges[i + 1] if i < len(edges)-2 else N[:, k] <= edges[i+1])
            if b.sum() < 60:
                continue
            print(f"     n = {edges[i]:.0f}..{edges[i+1]:.0f}  N {int(b.sum()):6d}"
                  f"   median chi2/ndof {np.median(chired[b]):.4f}"
                  f"   P(prob<1e-3) {TC.pm(int((u[b]<1e-3).sum()), int(b.sum()))}")
    print("\n=== and at FIXED |eta|, is it the subdetector or the eta?")
    for elo, ehi in ((0., 1.4), (1.4, 1.8), (1.8, 2.4)):
        e = base & (etamax >= elo) & (etamax < ehi)
        for k in (4, 5):
            med = np.median(N[e, k])
            for lab, b in ((f"{SUBDET[k]} <= {med:.0f}", e & (N[:, k] <= med)),
                           (f"{SUBDET[k]} >  {med:.0f}", e & (N[:, k] > med))):
                if b.sum() < 60:
                    continue
                print(f"  |eta| {elo:.1f}-{ehi:.1f}, {lab:<12s} N {int(b.sum()):6d}"
                      f"  median chi2/ndof {np.median(chired[b]):.4f}"
                      f"  P(prob<1e-3) {TC.pm(int((u[b]<1e-3).sum()), int(b.sum()))}")
    print("\n=== the worst TEC / TID misalignment a candidate carries")
    for k in (4, 5):
        q = D[:, k]
        e = np.percentile(q[base & (N[:, k] > 0)], [0, 50, 90, 100])
        for i in range(len(e) - 1):
            b = base & (N[:, k] > 0) & (q >= e[i]) & (q <= e[i + 1])
            if b.sum() < 60:
                continue
            print(f"  {SUBDET[k]} max|dlocx| {e[i]:7.2f}..{e[i+1]:<8.2f} um  "
                  f"N {int(b.sum()):6d}  median chi2/ndof {np.median(chired[b]):.4f}"
                  f"  P(prob<1e-3) {TC.pm(int((u[b]<1e-3).sum()), int(b.sum()))}")


if __name__ == "__main__":
    main()
