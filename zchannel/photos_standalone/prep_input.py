#!/usr/bin/env python3
"""Dump the sample's pre-FSR kinematics as binary input for `photos_gen`.

Two products under ``data/photos/``:

``mpre_bands.bin``
    the band definitions and, per band, the *conditional* ``m_pre`` density of
    the sample inside that band on a 1 MeV (1 GeV above 200) grid.  The driver
    samples a band from a fixed allocation and then ``m_pre`` from that band's
    own conditional density, so every band is generated with the sample's own
    within-band mass distribution while the *number* of events per band is free
    (each band's kernel is normalised separately, so the band populations never
    enter).  The sample weight of each band is carried along so that unions of
    bands - the analysis bands of ``00_moments.txt``, or the inclusive kernel -
    can be recombined with the right weights offline.

``boost_sample.bin``
    ``(m_pre, pt_pre, y_pre)`` triples drawn from the sample, for the check that
    the Z boost does not change ``u`` (it cannot: ``u`` is invariant; the check
    is on the numerics of Photos in a boosted frame).

Both are little-endian ``float64``/``int32`` with a self-describing header.
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fit_gen as FG                                            # noqa: E402

#: 2 GeV bands over 50-200 GeV plus three coarse bands to the sample's end.
#: Every edge of the analysis bands of ``00_moments.txt`` (50, 60, 70, 80, 86,
#: 96, 110, 130, 150, 200) is an edge of this set, so those bands are exact
#: unions of these.
def default_bands():
    e = list(np.arange(50.0, 200.0 + 1e-9, 2.0))
    return [(e[i], e[i + 1]) for i in range(len(e) - 1)] + \
           [(200.0, 240.0), (240.0, 320.0), (320.0, 1e5)]


def fine_edges():
    """1 MeV bins to 200 GeV, 1 GeV to 3200, 100 GeV to 100 TeV."""
    e = np.concatenate([50.0 + 0.001 * np.arange(150_000),
                        200.0 + 1.0 * np.arange(3_000),
                        3200.0 + 100.0 * np.arange(969 + 1)])
    return np.round(e, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/genmerged_full.npz")
    ap.add_argument("--out", default="data/photos")
    ap.add_argument("--nboost", type=int, default=4_000_000)
    ap.add_argument("--wclip", type=float, default=100.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = FG.load_gen(a.gen, ["m_pre", "pt_pre", "y_pre", "weight"])
    m = g["m_pre"].astype(np.float64)
    w = FG.clip_weights(g["weight"].astype(np.float64), a.wclip)

    edges = fine_edges()
    hist, _ = np.histogram(m, bins=edges, weights=w)
    bands = default_bands()

    # --- bands file ------------------------------------------------------
    # header: nband(int32)
    # per band: m_lo, m_hi, sumw (float64), i0, i1 (int32 into the edge array)
    # then: nedge(int32), edges(float64), hist(float64)
    blk = struct.pack("<i", len(bands))
    for lo, hi in bands:
        i0 = int(np.argmin(np.abs(edges - lo)))
        i1 = len(edges) - 1 if hi > edges[-1] else int(np.argmin(np.abs(edges - hi)))
        assert abs(edges[i0] - lo) < 1e-7, (lo, edges[i0])
        assert i1 == len(edges) - 1 or abs(edges[i1] - hi) < 1e-7, (hi, edges[i1])
        sw = float(hist[i0:i1].sum())
        blk += struct.pack("<dddqq", lo, min(hi, float(edges[-1])), sw, i0, i1)
    blk += struct.pack("<q", len(edges))
    blk += edges.astype("<f8").tobytes()
    blk += hist.astype("<f8").tobytes()
    fn = os.path.join(a.out, "mpre_bands.bin")
    with open(fn, "wb") as f:
        f.write(blk)
    print(f"{fn}: {len(bands)} bands, {len(edges)-1} fine bins, "
          f"sumw = {hist.sum():.6e} / {w.sum():.6e}")
    for i, (lo, hi) in enumerate(bands):
        i0 = int(np.argmin(np.abs(edges - lo)))
        i1 = len(edges) - 1 if hi > edges[-1] else int(np.argmin(np.abs(edges - hi)))
        if i < 3 or hist[i0:i1].sum() / w.sum() > 0.05 or i >= len(bands) - 3:
            print(f"   band {i:3d} [{lo:7.1f},{hi:9.1f})  "
                  f"frac = {hist[i0:i1].sum()/w.sum():.6f}")

    # --- boost file ------------------------------------------------------
    rng = np.random.default_rng(20260913)
    p = np.abs(w) / np.abs(w).sum()
    idx = rng.choice(len(m), size=min(a.nboost, len(m)), replace=False, p=p)
    tri = np.stack([m[idx], g["pt_pre"][idx].astype(np.float64),
                    g["y_pre"][idx].astype(np.float64)], 1)
    fn = os.path.join(a.out, "boost_sample.bin")
    with open(fn, "wb") as f:
        f.write(struct.pack("<q", len(tri)))
        f.write(np.ascontiguousarray(tri, "<f8").tobytes())
    print(f"{fn}: {len(tri)} (m, pt, y) triples, "
          f"<pt> = {tri[:,1].mean():.3f}, <|y|> = {np.abs(tri[:,2]).mean():.3f}")


if __name__ == "__main__":
    main()
