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
"""The UNCONVERGED fraction and what it correlates with."""
import numpy as np

V = {k: np.load(f"data/conv_{k}.npz") for k in ("base", "tight", "damp")}


def key(d):
    return (d["run"].astype(np.int64) * 10**6 + d["lumi"].astype(np.int64)) \
        * 10**9 + d["event"].astype(np.int64)


K = {k: key(d) for k, d in V.items()}
common = np.intersect1d(np.intersect1d(K["base"], K["tight"]), K["damp"])
SEL = {k: np.argsort(K[k])[np.searchsorted(np.sort(K[k]), common)] for k in V}
Z = {k: V[k]["z"][SEL[k]].astype(np.float64) for k in V}
B = {k: V["base"][k][SEL["base"]] for k in
     ("q", "eta", "sigma", "vgf", "pt", "niter", "chi2n", "nvalid", "npixhit",
      "edm", "qop_seed", "qop_ref")}
eta = np.abs(B["eta"].astype(np.float64))
band = np.digitize(eta, [0.9, 1.6])
q = B["q"].astype(np.float64)
sr = B["sigma"].astype(np.float64) * B["pt"].astype(np.float64) * np.cosh(B["eta"])
dq = np.abs((B["qop_ref"] - B["qop_seed"]) / np.abs(B["qop_seed"]))
unc = np.abs(Z["tight"] - Z["base"]) > 0.5
unc_d = np.abs(Z["damp"] - Z["base"]) > 0.5

print("=== UNCONVERGED fraction (|z_tight - z_base| > 0.5 sigma) ===")
print(f"{'band':10s} {'n':>7s} {'tight':>9s} {'damp':>9s}")
for ib, nm in enumerate(("barrel", "middle", "endcap")):
    m = band == ib
    print(f"{nm:10s} {m.sum():7d} {100*unc[m].mean():8.2f}% "
          f"{100*unc_d[m].mean():8.2f}%")
print(f"{'ALL':10s} {len(unc):7d} {100*unc.mean():8.2f}% {100*unc_d.mean():8.2f}%")

print("\n=== what the UNCONVERGED tracks look like (base variables) ===")
print(f"{'':14s} " + "".join(f"{n:>11s}" for n in
                             ("frac", "<niter>", "chi2/ndof", "sigma_rel",
                              "nvalid", "npixhit", "log10 edm", "<z>_even")))
for lab, m in (("unconverged", unc), ("converged", ~unc)):
    ze = 0.5 * (Z["base"][m & (q > 0)].mean() + Z["base"][m & (q < 0)].mean())
    print(f"{lab:14s} " + "".join(f"{v:11.4f}" for v in (
        m.mean(), B["niter"][m].mean(), B["chi2n"][m].mean(), sr[m].mean(),
        B["nvalid"][m].mean(), B["npixhit"][m].mean(),
        np.log10(np.maximum(B["edm"][m], 1e-30)).mean(), ze * 1e3)))

print("\n=== the charge-even mean, converged subset only (1e-3) ===")
print(f"{'band':10s} {'base(conv)':>12s} {'tight(conv)':>12s} "
      f"{'base(unconv)':>13s} {'tight(unconv)':>14s}")
rng = np.random.default_rng(97)


def ev(x, m, nb=200):
    idx = np.where(m)[0]
    v = 0.5 * (x[m & (q > 0)].mean() + x[m & (q < 0)].mean())
    o = np.empty(nb)
    for i in range(nb):
        k = idx[rng.integers(0, len(idx), len(idx))]
        o[i] = 0.5 * (x[k][q[k] > 0].mean() + x[k][q[k] < 0].mean())
    return v * 1e3, o.std() * 1e3


for ib, nm in enumerate(("barrel", "middle", "endcap")):
    m0 = band == ib
    a = ev(Z["base"], m0 & ~unc)
    b = ev(Z["tight"], m0 & ~unc)
    c = ev(Z["base"], m0 & unc)
    d = ev(Z["tight"], m0 & unc)
    print(f"{nm:10s} {a[0]:+7.2f}+-{a[1]:4.2f} {b[0]:+7.2f}+-{b[1]:4.2f} "
          f"{c[0]:+8.2f}+-{c[1]:5.2f} {d[0]:+8.2f}+-{d[1]:5.2f}")
