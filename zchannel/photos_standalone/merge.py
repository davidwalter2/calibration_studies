#!/usr/bin/env python3
"""Merge `photos_gen` band records into one npz with `mcref.npz`'s layout.

Replicas of the same band are summed; the replicas are also split into two
halves so that every derived quantity carries a half-sample error computed the
same way `cmp_fsr.py` computes the sample's.
"""
import argparse
import glob
import os
import struct

import numpy as np

HDR = 8 + 4 * 8 + 4 * 8
NB_MAX = 4096


def read_file(fn):
    with open(fn, "rb") as f:
        blob = f.read()
    if blob[:7] not in (b"PHGEN01", b"PHGEN02"):
        raise ValueError(f"{fn}: bad magic {blob[:8]!r}")
    ne_ = 4 if blob[:7] == b"PHGEN02" else 3
    nb, nfine, ntail, nlog = struct.unpack_from("<qqqq", blob, 8)
    ufine, utail, loglo, loghi = struct.unpack_from("<dddd", blob, 8 + 32)
    rec = 8 + 2 * 8 + 9 * 8 + ne_ * 8 + (3 * nfine + 3 * ntail + nlog) * 8
    out = []
    for i in range(nb):
        o = HDR + i * rec
        idx = struct.unpack_from("<q", blob, o)[0]
        lo, hi = struct.unpack_from("<dd", blob, o + 8)
        s = np.frombuffer(blob, "<f8", 9, o + 24)
        e = np.frombuffer(blob, "<f8", ne_, o + 24 + 72)
        p = o + 24 + 72 + ne_ * 8
        arr = np.frombuffer(blob, "<f8", 3 * nfine + 3 * ntail + nlog, p)
        out.append((idx, lo, hi, s, e, arr, nfine, ntail, nlog))
    return out, (nfine, ntail, nlog, ufine, utail, loglo, loghi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="glob(s) of photos_gen .bin files")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--nband", type=int, default=78)
    ap.add_argument("--meta", default="", help="free-form configuration string")
    a = ap.parse_args()

    files = sorted(sum([glob.glob(p) for p in a.inputs], []))
    if not files:
        raise SystemExit("no input files")
    _, shp = read_file(files[0])
    nfine, ntail, nlog = shp[:3]
    nb = a.nband

    tot = {}
    halves = {"A": {}, "B": {}}

    def blank():
        return dict(
            n=np.zeros(nb), n_nophot=np.zeros(nb), n_norad=np.zeros(nb),
            mpre_s1=np.zeros(nb), over=np.zeros(nb), mom=np.zeros((nb, 4)),
            nphot_s1=np.zeros(nb), npair=np.zeros(nb), nbad=np.zeros(nb),
            n_noemit=np.zeros(nb),
            fine_s0=np.zeros((nb, nfine)), fine_s1=np.zeros((nb, nfine)),
            fine_s2=np.zeros((nb, nfine)), tail_s0=np.zeros((nb, ntail)),
            tail_s1=np.zeros((nb, ntail)), tail_s2=np.zeros((nb, ntail)),
            logu_h=np.zeros((nb, nlog)))

    tot = blank()
    halves = {"A": blank(), "B": blank()}
    lo_all = np.zeros(nb)
    hi_all = np.zeros(nb)
    seen = np.zeros(nb, bool)
    # the half a file belongs to is decided by its own name so that a rerun
    # with more replicas keeps the split reproducible
    for k, fn in enumerate(files):
        recs, _ = read_file(fn)
        h = halves["A" if (k % 2 == 0) else "B"]
        for idx, lo, hi, s, e, arr, nf, nt, nl in recs:
            lo_all[idx], hi_all[idx], seen[idx] = lo, hi, True
            p = 0
            blocks = {}
            for name, n_ in (("fine_s0", nf), ("fine_s1", nf), ("fine_s2", nf),
                             ("tail_s0", nt), ("tail_s1", nt), ("tail_s2", nt),
                             ("logu_h", nl)):
                blocks[name] = arr[p:p + n_]
                p += n_
            for d in (tot, h):
                d["n"][idx] += s[0]
                d["n_nophot"][idx] += s[1]
                d["n_norad"][idx] += s[2]
                d["mpre_s1"][idx] += s[3]
                d["over"][idx] += s[4]
                d["mom"][idx] += s[5:9]
                d["nphot_s1"][idx] += e[0]
                d["npair"][idx] += e[1]
                d["nbad"][idx] += e[2]
                d["n_noemit"][idx] += e[3] if len(e) > 3 else s[1]
                for name, v in blocks.items():
                    d[name][idx] += v

    out = dict(tot)
    out["bands_lo"] = lo_all
    out["bands_hi"] = hi_all
    out["w2"] = tot["n"]                # unweighted generation
    for tag in ("A", "B"):
        for k, v in halves[tag].items():
            out[f"half{tag}_{k}"] = v
    out["meta"] = np.array(a.meta)
    out["nfiles"] = np.array(len(files))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    np.savez_compressed(a.output, **out)
    n = tot["n"].sum()
    print(f"{a.output}: {len(files)} files, {int(seen.sum())}/{nb} bands, "
          f"N = {n:.4e}")
    print(f"  inclusive-over-bands <u> = {tot['mom'][:,1].sum()/n:.6e}, "
          f"P(0 gamma) = {tot['n_nophot'].sum()/n:.6f}, "
          f"bad = {tot['nbad'].sum():.0f}")


if __name__ == "__main__":
    main()
