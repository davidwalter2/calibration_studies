#!/usr/bin/env python3
"""Merge the per-file `dump_gen_fsr.py` shards into one compact npz.

Drops the bookkeeping columns and downcasts to float32 (the masses are stored
in MiniAOD at reduced float precision anyway, ~4 keV, so float32's 6 ppm at
91 GeV = 0.5 MeV is *not* enough for the masses -- those stay float64; the
kinematics and the weights go to float32).

    python3 merge_gen.py -i "/ceph/.../zgen/gen_*.npz" -o data/genmerged.npz
"""
import argparse, glob, sys, time
import numpy as np

F64 = ("m_pre", "m_prelep", "m_post", "m_dress")
F32 = ("weight", "pt_pre", "y_pre", "eph",
       "pt1", "eta1", "pt2", "eta2",
       "pt1_pre", "eta1_pre", "pt2_pre", "eta2_pre")
I16 = ("nph", "npre")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--inputs", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()

    files = sorted({f for pat in a.inputs for f in glob.glob(pat)}) or list(a.inputs)
    print(f"[merge_gen] {len(files)} shards", flush=True)
    acc = {k: [] for k in F64 + F32 + I16}
    n = 0
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            d = np.load(f)
        except Exception as ex:                      # a shard still being written
            print(f"  SKIP {f}: {ex}", file=sys.stderr)
            continue
        for k in F64:
            acc[k].append(np.asarray(d[k], np.float64))
        for k in F32:
            acc[k].append(np.asarray(d[k], np.float32))
        for k in I16:
            acc[k].append(np.asarray(d[k], np.int16))
        n += len(d["m_pre"])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)}  {n} events  {time.time()-t0:.0f} s",
                  flush=True)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["nfiles"] = np.array([len(files)])
    np.savez(a.output, **out)
    w = out["weight"].astype(np.float64)
    print(f"[merge_gen] {n} events, sum(w) = {w.sum():.6g}, "
          f"Neff = {w.sum()**2/np.sum(w**2):.6g} -> {a.output} "
          f"({time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
