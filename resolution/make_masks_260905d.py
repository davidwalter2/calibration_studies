#!/usr/bin/env python3
"""Quality masks for the _260905d (step-damping) candidate caches.

Same construction as `make_masks_260904f.py`, so the floor-2.0 (_260903x),
floor-0.25 (_260904f) and step-damped (_260905d) fits use identical selections.

usage: python3 make_masks_260905d.py [tag ...]   (default: all)
"""
import os
import sys

import numpy as np

OUT = "runs/stepdamp260905"
SIGMAX = 0.15
CACHES = {
    "gun_260905d": ("runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz",
                    f"{OUT}/aux_jpsigun_260905d.npz"),
}

os.makedirs(OUT, exist_ok=True)
tags = sys.argv[1:] or list(CACHES)

for tag in tags:
    cache, auxp = CACHES[tag]
    if not (os.path.exists(cache) and os.path.exists(auxp)):
        print(f"[skip] {tag}: missing {cache if not os.path.exists(cache) else auxp}")
        continue
    d = np.load(cache)
    a = np.load(auxp)
    s = d["sigma"].astype(np.float64)
    z = d["z"].astype(np.float64)
    assert np.array_equal(a["z"].astype(np.float64), z), f"{tag}: aux not aligned"
    sane = np.isfinite(s) & (s > 0) & (s < SIGMAX) & np.isfinite(z)
    c2 = a["normchi2"].astype(np.float64)
    it = a["niter"].astype(np.float64)
    fz = a["frozen"].astype(np.float64) > 0.5
    pt = a["ptmin"].astype(np.float64)
    pm = a["pmin"].astype(np.float64)
    print(f"\n== {tag}  n={len(z)} ==")
    print("   chi2/ndof q50/q80/q87/q90/q95/q99/max = "
          + " ".join(f"{np.quantile(c2, q):.4g}" for q in
                     (.5, .8, .87, .9, .95, .99, 1.)))
    print(f"   frac chi2>10 {np.mean(c2 > 10):.5f}   frac chi2>3 {np.mean(c2 > 3):.5f}"
          f"   frozen {fz.mean():.5f}   niter>=10 {np.mean(it >= 10):.5f}")
    print(f"   ptmin<2 {np.mean(pt < 2):.5f}  ptmin<0.5 {np.mean(pt < .5):.5f}"
          f"   pmin<2 {np.mean(pm < 2):.5f}  pmin<0.5 {np.mean(pm < .5):.5f}")
    for nm, sel in (
            (f"{tag}_sane", sane),
            (f"{tag}_q3", sane & (c2 < 3.)),
            (f"{tag}_q3nf", sane & (c2 < 3.) & ~fz & (it < 10)),
            (f"{tag}_nofrozen", sane & ~fz),
            (f"{tag}_pt2", sane & (pt > 2.)),
            (f"{tag}_pt3", sane & (pt > 3.)),
            (f"{tag}_pt3q3", sane & (pt > 3.) & (c2 < 3.)),
    ):
        p = f"{OUT}/mask_{nm}.npz"
        np.savez_compressed(p, mask=sel)
        print(f"   {nm:<24s} {int(sel.sum()):8d} / {len(sel):8d} ({100. * sel.mean():6.3f} %)")
