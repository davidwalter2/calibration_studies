#!/usr/bin/env python3
"""Generator-level diagnosis of the kernel ON THE SELECTED CANDIDATES.

The detector-level `m_Z` closes at -11.06 +- 2.27 MeV with an `|eta|` pattern
(-35.8 barrel, +0.1 transition, +4.1 endcap at 300 k).  The detector half is
excluded: with no kernel at all the fit closes at +0.9 +- 2.1 MeV.  This script
asks the kernel side the same question with NO detector at all, by fitting the
selected candidates' own GEN masses with exactly the detector-level chain:

    (a) pre-FSR gen mass  vs  lineshape (x) A (x) K          -- no fold
    (b) post-FSR gen mass vs  lineshape (x) A (x) FSR (x) K  -- the full fold

`(b) - (a)` is the fold, isolated: (a) shares every other ingredient, so if (a)
closes and (b) does not, the FSR kernel and the acceptance are the defect.
Each is run with the kernel/acceptance pair IN USE (built from generator events
in a gen fiducial) and with the pair built from the selected candidates
themselves (`kern_from_selected.py`), inclusively and per `|eta|` band.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_gen as FG                                          # noqa: E402

BANDS = ((0.0, 0.9, "barrel"), (0.9, 1.6, "transition"),
         (1.6, 3.0, "endcap"), (0.0, 9.9, "inclusive"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/gen_selected.npz")
    ap.add_argument("--kernel-a", default="data/kern_loose_band3.3e-4.npz")
    ap.add_argument("--acc-a", default="data/acc_loose_d8.json")
    ap.add_argument("--kernel-b", default="data/kern_selected_band3.3e-4.npz")
    ap.add_argument("--acc-b", default="data/acc_selected_grid.json")
    ap.add_argument("--shape", type=int, default=5)
    ap.add_argument("--nm", type=int, default=32768)
    ap.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0])
    ap.add_argument("--bands", action="store_true",
                    help="also run the three |eta| bands")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    import tensorflow as tf

    g = FG.load_gen(a.gen)
    w = FG.clip_weights(g["weight"].astype(np.float64), 100.0)
    lead = np.abs(g["eta1"])
    print(f"[gen] {len(w)} selected candidates, N_eff/N = "
          f"{w.sum()**2 / np.sum(w**2) / len(w):.3f}", flush=True)

    def acc_of(path):
        if not path:
            return None
        with open(path) as fh:
            return {k: v for k, v in json.load(fh).items()
                    if not k.startswith("_")}

    accs = {"inuse": acc_of(a.acc_a), "selected": acc_of(a.acc_b)}
    kers = {"inuse": a.kernel_a, "selected": a.kernel_b}
    out = {}

    def one(tag, mass, sel, fsr, acc):
        z = FG.make_provider(tf, (50.0, 130.0), a.nm, "fixed",
                             "nnpdf31_nnlo_13tev",
                             ("gamma", "int", "z"), fsr, acc,
                             mz_ref=FG.MZ_FIXED, gz_ref=FG.GZ_FIXED,
                             fsr_mmax=None)
        f = FG.GenFit(z, g[mass][sel], w[sel], a.window, shape=a.shape, tf=tf)
        t0 = time.time()
        r = f.fit(verbose=False)
        i, j = r["params"].index("m_Z"), r["params"].index("Gamma_Z")
        print(f"  {tag:52s} m_Z {r['x'][i]:+8.2f} +- {r['err'][i]:5.2f}   "
              f"Gamma_Z {r['x'][j]:+8.2f} +- {r['err'][j]:5.2f}   "
              f"({f.n_used} ev, {time.time() - t0:.0f} s)", flush=True)
        out[tag] = FG.summarise(tag, r, quiet=True)
        return r["x"][i], r["x"][j]

    for lo, hi, name in (BANDS if a.bands else BANDS[-1:]):
        sel = (lead >= lo) & (lead < hi)
        print(f"\n=== |eta_lead| in [{lo}, {hi})  --  {name}  "
              f"({sel.sum()} candidates) ===", flush=True)
        for which in ("inuse", "selected"):
            ma, _ = one(f"{name}/{which}: (a) pre-FSR, no fold", "m_pre", sel,
                        None, accs[which])
            mb, _ = one(f"{name}/{which}: (b) post-FSR, full fold", "m_post",
                        sel, kers[which], accs[which])
            print(f"  {'--> (b) - (a), the FOLD alone':52s} "
                  f"{mb - ma:+8.2f} MeV", flush=True)

    if a.output:
        with open(a.output, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n-> {a.output}")


if __name__ == "__main__":
    main()
