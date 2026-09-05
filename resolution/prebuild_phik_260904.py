#!/usr/bin/env python3
"""Pre-build the phi_K tables cf_masslik_fit.py will need for the subset fits.

The table is cached under a key that contains tmax_abs = TG[-1]/sigma.min(),
so every subset whose smallest sigma_m differs from the full sample's triggers
a fresh 590 s build INSIDE the fit.  Building them here, in parallel, keeps the
fit chain at ~1 min per fit.
"""
import sys, os, numpy as np
from multiprocessing import Pool

def build(job):
    kern, pairs, tmax = job
    dm = np.load(kern)["dm"]
    cf = os.path.join(os.path.dirname(os.path.abspath(pairs)),
                      "masslikfit_phiKtab_%s_%d_%.12e.npz"
                      % (os.path.basename(kern).replace(".npz", ""), len(dm), tmax))
    if os.path.exists(cf):
        return f"[hit ] {cf}"
    tabs = np.linspace(0.0, tmax, 8192)
    blk = 1024
    tab = np.concatenate([np.mean(np.exp(1j*np.outer(tabs[i:i+blk], dm)), axis=1)
                          for i in range(0, len(tabs), blk)])
    np.savez(cf, tabs=tabs, phiK_tab=tab)
    return f"[made] {cf}"

if __name__ == "__main__":
    GP = "runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz"
    GK = "runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz"
    VP = "runs/cf_masspairs_btojpsix_v3_260903x_m0.npz"
    VK = "runs/cf_masskernel_btojpsix_v3_260903x_m0.npz"
    jobs = []
    for sm in (0.008560, 0.011181, 0.011399, 0.011527):
        jobs.append((GK, GP, 14.0/sm))
    for sm in (0.011007,):
        jobs.append((VK, VP, 14.0/sm))
    with Pool(len(jobs)) as p:
        for r in p.map(build, jobs):
            print(r, flush=True)
