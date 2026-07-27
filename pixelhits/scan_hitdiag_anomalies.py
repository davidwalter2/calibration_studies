#!/usr/bin/env python
"""Anomaly scan of the pixel hit-class biases over the full 2016 dataset.

Reads the hitdiag trees of the per-file scan production (runCvhJpsi.py with
keepPixelEdgeHits=True pixelMinSizeX=1 deweightPathoHits=True
fillHitDiagnostics=True, golden JSON applied) and accumulates per
(run, LS-block, class) residual statistics in one streaming pass.

Flags runs whose class-mean deviates from the era median by more than
--flag-um AND --flag-nsig; for flagged runs prints the LS-block profile of
the offending class (mask candidates). Modelled on the run-283453 case
(end-of-fill HV/timing scan inside the golden JSON).

Classes monitored (BPix): sizeX1 dx, edge-x dx (both sides pooled),
clean dx (control), edge-y inward dy ((hi-lo)/2 pattern via per-side
means), sizeY1 dy.
"""

import argparse
import glob
from collections import defaultdict

import numpy as np
import uproot

CM2UM = 1e4
LSBLOCK = 25          # lumisections per block
CLIP_CM = 0.10        # accumulate |res| < 1000 um only

CLASSES = {
    # name: (bit mask test, coordinate)
    "clean_dx":   ("clean", "dx"),
    "sizeX1_dx":  ("sizeX1", "dx"),
    "edgeX_dx":   ("edgeX", "dx"),
    "edgeYlo_dy": ("edgeYlo", "dy"),
    "edgeYhi_dy": ("edgeYhi", "dy"),
    "sizeY1_dy":  ("sizeY1", "dy"),
}


def class_masks(cls):
    edgex = (cls & 0x3) > 0
    sizex1 = (cls & 0x10) > 0
    return {
        "clean": cls == 0,
        "sizeX1": sizex1,
        "edgeX": edgex & ~sizex1,
        "edgeYlo": ((cls & 0x4) > 0) & ~edgex & ~sizex1,
        "edgeYhi": ((cls & 0x8) > 0) & ~edgex & ~sizex1,
        "sizeY1": ((cls & 0x20) > 0) & ((cls & 0xF) == 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scandir", required=True, help="production outdir (task_*/)")
    p.add_argument("--flag-um", type=float, default=8.,
                   help="minimum |deviation from era median| to flag (um)")
    p.add_argument("--flag-nsig", type=float, default=5.)
    p.add_argument("--min-hits", type=int, default=200,
                   help="minimum hits per (run, class) to consider")
    args = p.parse_args()

    files = sorted(glob.glob(args.scandir.rstrip("/") + "/task_*/globalcor_*.root"))
    print(f"{len(files)} files")

    # acc[(run, lsblock, class)] = [sum, sumsq, n_in, n_out] where n_out
    # counts hits with CLIP_CM < |res| < 2 cm (mm-scale pathologies that a
    # clipped mean is blind to -- the run-283453 failure mode); lsblock -1
    # = whole run
    acc = defaultdict(lambda: [0.0, 0.0, 0, 0])
    cols = ["run", "lumi", "hitdiag_detid", "hitdiag_class",
            "hitdiag_dx", "hitdiag_dy"]
    for i, fn in enumerate(files):
        if i % 50 == 0:
            print(f"  file {i}/{len(files)}")
        try:
            t = uproot.open(fn)["tree"]
            if t.num_entries == 0:
                continue
            a = t.arrays(cols, library="np")
        except Exception as e:
            print("  skip", fn, e)
            continue
        counts = np.array([len(x) for x in a["hitdiag_detid"]])
        run = np.repeat(a["run"], counts)
        ls = np.repeat(a["lumi"], counts)
        det = np.concatenate(a["hitdiag_detid"]) if len(counts) else np.array([], int)
        cls = np.concatenate(a["hitdiag_class"]).astype(int)
        dx = np.concatenate(a["hitdiag_dx"])
        dy = np.concatenate(a["hitdiag_dy"])
        bpix = ((det.astype(np.int64) >> 25) & 0x7) == 1
        masks = class_masks(cls)
        for cname, (mname, coord) in CLASSES.items():
            res = dx if coord == "dx" else dy
            base = bpix & masks[mname] & (np.abs(res) < 2.0)
            if not base.any():
                continue
            isout = np.abs(res[base]) >= CLIP_CM
            r, l, v = run[base], ls[base] // LSBLOCK, res[base]
            # vectorised accumulation over unique (run, lsblock) pairs
            pairidx = r.astype(np.int64) * 100000 + l
            for pid in np.unique(pairidx):
                m = pairidx == pid
                vin = v[m][~isout[m]]
                s, s2, n = vin.sum(), (vin ** 2).sum(), int(len(vin))
                nout = int(isout[m].sum())
                rr, ll = int(pid // 100000), int(pid % 100000)
                for k in ((rr, ll, cname), (rr, -1, cname)):
                    e = acc[k]
                    e[0] += s
                    e[1] += s2
                    e[2] += n
                    e[3] += nout

    # --- per-run summary and era medians -----------------------------------
    def stats(key):
        s, s2, n, nout = acc[key]
        if n == 0:
            return np.nan, np.nan, 0, 0.
        m = s / n
        e = np.sqrt(max(s2 / n - m * m, 0.) / n)
        return m * CM2UM, e * CM2UM, n, nout / max(n + nout, 1)

    runs = sorted({k[0] for k in acc})
    print(f"\n{len(runs)} runs")
    med = {}
    for cname in CLASSES:
        good = [stats((r, -1, cname)) for r in runs]
        vals = [g[0] for g in good if g[2] >= args.min_hits]
        fouts = [g[3] for g in good if g[2] >= args.min_hits]
        med[cname] = np.nanmedian(vals) if vals else np.nan
        med[cname + "_fout"] = np.nanmedian(fouts) if fouts else 0.
        print(f"era median {cname:>10s}: {med[cname]:+8.2f} um, "
              f"f_out {med[cname + '_fout']*100:.3f}% "
              f"({len(vals)} runs with >= {args.min_hits} hits)")

    print(f"\n=== flagged runs (|dev| > {args.flag_um} um and > {args.flag_nsig} sigma) ===")
    nflag = 0
    for r in runs:
        for cname in CLASSES:
            m, e, n, fout = stats((r, -1, cname))
            if n < args.min_hits or not np.isfinite(m):
                continue
            dev = m - med[cname]
            dfout = fout - med[cname + "_fout"]
            ferr = np.sqrt(max(fout * (1 - fout), 1e-9) / max(n, 1))
            flag_mean = abs(dev) > args.flag_um and abs(dev) > args.flag_nsig * e
            flag_fout = dfout > max(0.01, args.flag_nsig * ferr)
            if flag_mean or flag_fout:
                nflag += 1
                why = []
                if flag_mean: why.append(f"mean dev {dev:+.1f} um")
                if flag_fout: why.append(f"f_out {fout*100:.2f}% (era {med[cname+'_fout']*100:.3f}%)")
                print(f"\nrun {r}  {cname}: {m:+8.2f} ± {e:.2f} um, n={n}  [{', '.join(why)}]")
                # LS profile of the offending class
                blocks = sorted(l for (rr, l, cc) in acc
                                if rr == r and cc == cname and l >= 0)
                for l in blocks:
                    bm, be, bn, bfout = stats((r, l, cname))
                    if bn < 30:
                        continue
                    bad_mean = (abs(bm - med[cname]) > args.flag_um
                                and abs(bm - med[cname]) > 3 * be)
                    bad_fout = bfout - med[cname + "_fout"] > 0.02
                    mark = " <-- OUTLIER" if (bad_mean or bad_fout) else ""
                    print(f"    LS [{l*LSBLOCK:4d},{(l+1)*LSBLOCK-1:4d}] "
                          f"{bm:+8.2f} ± {be:5.2f} (n={bn}, f_out {bfout*100:.2f}%){mark}")
    if nflag == 0:
        print("none")


if __name__ == "__main__":
    main()
