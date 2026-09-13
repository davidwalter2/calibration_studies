#!/usr/bin/env python3
"""TAIL study, hypothesis B, the decisive leg: THE IDEAL GEOMETRY.

The same DY MiniAOD events refitted with `useIdealGeometry=True`.  In MC the
Geant4 tracker is the ideal one, so this is the perfectly aligned control: it
removes the UL16 MC alignment from the reconstructed hit positions and
changes nothing else.  If the excess chi2 -- the quantity the residual tail
is -- is the misalignment, it goes away here.

usage: python3 tail_ideal.py --real <npz> --ideal <npz> [--match]
"""
import argparse, os, sys
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402


def prep(fn):
    d = dict(np.load(fn, allow_pickle=False))
    base, _, _ = TC.baseline(d, max_abs_vtxz=0.0, chi2=0.0, verbose=False)
    cls, names, _ = GB.classify(d)
    base = base & (cls == names.index("signal"))
    zv = np.asarray(d["vtxz"], float)
    nd = np.asarray(d["ndof"], float)
    sub = np.asarray(d["vtxdchi2"], float) + np.asarray(d["bschi2fit"], float)
    ndr = np.maximum(nd - 4.0, 1.0)
    d["_chired"] = (np.asarray(d["chisq"], float) - sub) / ndr
    d["_u"] = stats.chi2.sf(d["_chired"] * ndr, ndr)
    d["_sel"] = base
    d["_zv"] = zv
    # THE MATCHING KEY has to be geometry-INDEPENDENT: the reconstructed pT
    # moves by ~1e-4 between the two geometries, so keying on it matches
    # nothing.  (run, lumi, event) plus the two GEN indices is invariant --
    # `genbkg` already uses that pair as the identity of a candidate -- and an
    # occurrence counter breaks the tie for the gen-unmatched ones.
    base = [f"{r}:{l}:{e}:{ip}:{im}" for r, l, e, ip, im in
            zip(d["run"], d["lumi"], d["event"],
                d["genidx_plus"], d["genidx_minus"])]
    seen = {}
    keys = []
    for b in base:
        k = seen.get(b, 0)
        seen[b] = k + 1
        keys.append(f"{b}:{k}")
    d["_key"] = np.array(keys)
    return d


def summarise(tag, d, sel):
    z1, z2, zv = d["bsz"][:, 0], d["bsz"][:, 1], d["_zv"]
    u, c = d["_u"], d["_chired"]
    print(f"\n  {tag}:  N {int(sel.sum())}")
    print(f"    chi2/ndof median {np.median(c[sel]):.4f}  "
          f"p90 {np.percentile(c[sel],90):.4f}  "
          f"P(prob<0.01) {TC.pm(int((u[sel]<1e-2).sum()), int(sel.sum()))}  "
          f"P(prob<1e-3) {TC.pm(int((u[sel]<1e-3).sum()), int(sel.sum()))}")
    for nm, z in (("z_1", z1), ("z_2", z2), ("z_v", zv)):
        print(f"    {nm}: Var {np.var(z[sel]):7.4f}  "
              f"P>3 {TC.pm(int((np.abs(z[sel])>3).sum()), int(sel.sum()))}  "
              f"P>5 {TC.pm(int((np.abs(z[sel])>5).sum()), int(sel.sum()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--ideal", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    R, I = prep(a.real), prep(a.ideal)
    print(f"# real {len(R['run'])} candidates, ideal {len(I['run'])}")
    # SAME-CANDIDATE matching on (run, lumi, event, pT+)
    common, ir, ii = np.intersect1d(R["_key"], I["_key"], return_indices=True)
    print(f"# matched {len(common)}")
    mr = np.zeros(len(R["run"]), bool); mr[ir] = True
    mi = np.zeros(len(I["run"]), bool); mi[ii] = True
    selr = R["_sel"] & mr
    seli = I["_sel"] & mi
    # keep only the pairs where BOTH pass the baseline
    okr = np.zeros(len(common), bool); okr = R["_sel"][ir] & I["_sel"][ii]
    print(f"# both pass the baseline: {int(okr.sum())}")
    sr = np.zeros(len(R["run"]), bool); sr[ir[okr]] = True
    si = np.zeros(len(I["run"]), bool); si[ii[okr]] = True

    print("\n=== THE SAME CANDIDATES, the two geometries")
    summarise("aligned (UL16 MC alignment)", R, sr)
    summarise("IDEAL geometry", I, si)

    cr = R["_chired"][ir[okr]]
    ci = I["_chired"][ii[okr]]
    print(f"\n  per-candidate chi2/ndof(ideal) - chi2/ndof(aligned): "
          f"median {np.median(ci-cr):+.4f}  mean {np.mean(ci-cr):+.4f}")
    print(f"  the candidates with prob < 1e-3 under the ALIGNED geometry: "
          f"{int((R['_u'][ir[okr]]<1e-3).sum())}; of those, still < 1e-3 "
          f"under the IDEAL one: "
          f"{int(((R['_u'][ir[okr]]<1e-3) & (I['_u'][ii[okr]]<1e-3)).sum())}")
    z1r, z1i = R["bsz"][ir[okr], 0], I["bsz"][ii[okr], 0]
    t = np.abs(z1r) > 3
    print(f"  the |z_1| > 3 candidates under the ALIGNED geometry: {int(t.sum())};"
          f" still > 3 under the IDEAL one: {int((t & (np.abs(z1i)>3)).sum())};"
          f" median |z_1| there {np.median(np.abs(z1i[t])):.2f} against "
          f"{np.median(np.abs(z1r[t])):.2f}")
    etam = np.maximum(np.abs(R["eta_plus"][ir[okr]]), np.abs(R["eta_minus"][ir[okr]]))
    print("\n=== the chi2 excess against |eta|, both geometries")
    for lo, hi in ((0., 1.4), (1.4, 1.8), (1.8, 2.4)):
        b = (etam >= lo) & (etam < hi)
        if b.sum() < 40:
            continue
        print(f"  |eta| {lo:.1f}-{hi:.1f}  N {int(b.sum()):6d}   "
              f"aligned {TC.pm(int((R['_u'][ir[okr]][b]<1e-3).sum()), int(b.sum()))}"
              f"   ideal {TC.pm(int((I['_u'][ii[okr]][b]<1e-3).sum()), int(b.sum()))}")


if __name__ == "__main__":
    main()
