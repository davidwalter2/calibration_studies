#!/usr/bin/env python3
# ############################ RETRACTED ############################
# THIS SCRIPT'S RESULTS ARE WRONG. DO NOT USE IT.
#
# It pairs the convergence variants on (run, lumi, event) alone. The muon
# gun puts TWO muons in every event -- 79 965 tracks in 40 000 unique keys,
# measured 2026-09-08 -- so the searchsorted pairing below matches muon A of
# one variant to muon B of the other as soon as a track is added or dropped
# upstream. That manufactures an O(1 sigma) per-track spread and a fat tail,
# and it is what produced the retracted claims of a +3.4e-3 paired shift,
# "15 % of fits unconverged" and "the shift lives in the unconverged 15 %".
#
# The correct tooling is `extract_conv.py`'s `slot` (index within the event)
# plus `conv_common.py` / `c1_conv.py` / `c2_pair.py`, which pair on
# (run, lumi, event, slot). Their answer is that the three variants land on
# the SAME minimum: paired charge-even shift < 0.004e-3, chi2/ndof equal to
# six digits. See STATE.md, PART 3.
# ###################################################################
"""Which tracks actually MOVE when the convergence is tightened, and is it the
IN population? Paired, on the common (run,lumi,event) set."""
import numpy as np

V = {k: np.load(f"data/conv_{k}.npz") for k in ("base", "tight", "damp")}


def key(d):
    return (d["run"].astype(np.int64) * 10**6 + d["lumi"].astype(np.int64)) \
        * 10**9 + d["event"].astype(np.int64)


K = {k: key(d) for k, d in V.items()}
common = np.intersect1d(np.intersect1d(K["base"], K["tight"]), K["damp"])
SEL = {k: np.argsort(K[k])[np.searchsorted(np.sort(K[k]), common)] for k in V}
Z = {k: V[k]["z"][SEL[k]].astype(np.float64) for k in V}
sig = V["base"]["sigma"][SEL["base"]].astype(np.float64)
q = V["base"]["q"][SEL["base"]].astype(np.float64)
eta = np.abs(V["base"]["eta"][SEL["base"]].astype(np.float64))
band = np.digitize(eta, [0.9, 1.6])
dq = np.abs((V["base"]["qop_ref"] - V["base"]["qop_seed"])[SEL["base"]]
            / np.abs(V["base"]["qop_seed"][SEL["base"]]))
nit = V["base"]["niter"][SEL["base"]]
t90 = np.percentile(dq, 90)

print(f"paired on {len(common)} tracks\n")
for k in ("tight", "damp"):
    d = Z[k] - Z["base"]
    moved = np.abs(d) > 1e-6
    print(f"=== {k} - base ===")
    print(f"  moved at all: {100*moved.mean():.1f}%   "
          f"median |dz| (moved) {np.median(np.abs(d[moved])):.4f}   "
          f"rms {d.std():.4f}   mean {d.mean()*1e3:+.2f}e-3")
    print(f"  charge-even mean shift: "
          f"{0.5*(d[q>0].mean()+d[q<0].mean())*1e3:+.2f}e-3   "
          f"charge-odd {0.5*(d[q>0].mean()-d[q<0].mean())*1e3:+.2f}e-3")
    for lab, m in (("IN  (top 10% |seed step|)", dq > t90),
                   ("OUT (rest)", dq <= t90),
                   ("niter(base)=2", nit == 2), ("niter(base)>=3", nit >= 3)):
        de = 0.5 * (d[m & (q > 0)].mean() + d[m & (q < 0)].mean())
        print(f"    {lab:26s} n={m.sum():6d} moved {100*moved[m].mean():5.1f}% "
              f"charge-even dz {de*1e3:+8.2f}e-3  rms {d[m].std():.4f}")
    print()
