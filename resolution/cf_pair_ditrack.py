#!/usr/bin/env python3
"""Paired comparison of two TwoTrack productions on the same events.

Keys candidates by (run, lumi, event) plus the generated J/psi kinematics, so
the same physical candidate is compared across fits. Reports the paired
reconstructed-mass shift (median/mean, robust), sigma_m ratio and resinfcov
ratio, in bins of the harmonic mean of the two gen muon momenta if available.
"""
import argparse, glob, sys
import numpy as np, uproot

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="glob of production A (reference)")
ap.add_argument("--b", required=True, help="glob of production B (new)")
ap.add_argument("--ntasks", type=int, default=100000)
ap.add_argument("--label", default="B-A")
args = ap.parse_args()

def load(pattern):
    files = sorted(glob.glob(pattern))[:args.ntasks]
    t0 = uproot.open(files[0])["tree"]
    keys = set(t0.keys())
    want = ["run", "lumi", "event", "Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass", "resinfcov", "niter", "chisqval"]
    genk = [k for k in ("Jpsigen_pt", "Jpsigen_eta", "Jpsigen_phi", "Muplusgen_pt", "Muminusgen_pt", "Muplusgen_eta", "Muminusgen_eta") if k in keys]
    want += genk
    want = [w for w in want if w in keys]
    out = {w: [] for w in want}
    for f in files:
        a = uproot.open(f)["tree"].arrays(want, library="np")
        for w in want: out[w].append(np.asarray(a[w]))
    d = {w: np.concatenate(v) for w, v in out.items()}
    return d, genk

A, genk = load(args.a); B, _ = load(args.b)
print("gen keys used for matching:", genk, " nA=%d nB=%d" % (len(A["event"]), len(B["event"])))
def key(d):
    cols = [d["run"].astype(np.int64), d["lumi"].astype(np.int64), d["event"].astype(np.int64)]
    for g in genk: cols.append(np.round(d[g].astype(np.float64), 5))
    return list(zip(*cols))
ka, kb = key(A), key(B)
ia = {k: i for i, k in enumerate(ka)}
dup = len(ka) - len(ia)
pairs = [(ia[k], j) for j, k in enumerate(kb) if k in ia]
print("duplicate keys in A: %d ; matched pairs: %d (%.1f%% of B)" % (dup, len(pairs), 100*len(pairs)/len(kb)))
i, j = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
mA, mB = A["Jpsi_mass"][i], B["Jpsi_mass"][j]; sA, sB = A["Jpsi_sigmamass"][i], B["Jpsi_sigmamass"][j]
g = A["Jpsigen_mass"][i]
ok = np.isfinite(sA) & np.isfinite(sB) & (sA > 0) & (sB > 0) & np.isfinite(mA) & np.isfinite(mB)
print("both fits valid: %d (A-only NaN %d, B-only NaN %d)" % (ok.sum(), (~np.isfinite(sA)&np.isfinite(sB)).sum(), (np.isfinite(sA)&~np.isfinite(sB)).sum()))
i, j, mA, mB, sA, sB, g = i[ok], j[ok], mA[ok], mB[ok], sA[ok], sB[ok], g[ok]
dm = mB - mA
def rob(x): 
    q = np.percentile(x, [16, 50, 84]); return q[1], 0.5*(q[2]-q[0])
def summarize(sel, name):
    d = dm[sel]; n = sel.sum()
    med, w = rob(d); trim = np.abs(d - med) < 5*max(w, 1e-6)
    print(f"{name:22s} n={n:7d}  d m(B-A): median {1e3*med:+.4f} MeV, mean(5w trim) {1e3*d[trim].mean():+.4f} +- {1e3*d[trim].std()/np.sqrt(trim.sum()):.4f} MeV, rob width {1e3*w:.3f} MeV, |d|>1 MeV: {100*(np.abs(d)>1e-3).mean():.2f}%"
          f" | sigma_m B/A median {np.median(sB[sel]/sA[sel]):.4f} | m-gen: A {1e3*np.median(mA[sel]-g[sel]):+.3f} B {1e3*np.median(mB[sel]-g[sel]):+.3f} MeV (median), inv-var mean A {1e3*np.sum((mA[sel]-g[sel])/sA[sel]**2)/np.sum(1/sA[sel]**2):+.3f} B {1e3*np.sum((mB[sel]-g[sel])/sB[sel]**2)/np.sum(1/sB[sel]**2):+.3f} MeV")
summarize(np.ones(len(dm), bool), "all")
if "Jpsigen_pt" in A:
    pt = A["Jpsigen_pt"][i]
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        s = (pt >= lo) & (pt < hi)
        if s.sum() > 100: summarize(s, f"Jpsi gen pT {lo}-{hi}")
if "resinfcov" in A and "resinfcov" in B:
    rA, rB = A["resinfcov"][i], B["resinfcov"][j]
    print("resinfcov B/A median %.4f ; niter A %.2f B %.2f ; chisq median A %.2f B %.2f" % (np.median(rB/rA), A["niter"][i].mean(), B["niter"][j].mean(), np.median(A["chisqval"][i]), np.median(B["chisqval"][j])))
