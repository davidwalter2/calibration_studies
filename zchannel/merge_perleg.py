#!/usr/bin/env python3
"""Merge the per-leg gen dumps and add the derived collinear variables.

``x_q = E'_q / E_q`` is evaluated in the **Z rest frame** (the frame in which
the collinear factorisation ``p' = x p`` is stated); ``xlab_q = p_T'/p_T`` is the
same ratio taken in the lab, which the massless collinear limit makes equal.
"""
import argparse
import glob
import math

import numpy as np

COLS = ("weight", "m_pre", "m_post", "npre", "nph", "pt_pre", "y_pre",
        "pz_pre", "e_pre",
        "ptp", "etap", "phip", "ep", "ptm", "etam", "phim", "em",
        "ptp_pre", "etap_pre", "phip_pre", "ep_pre",
        "ptm_pre", "etam_pre", "phim_pre", "em_pre")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--inputs", required=True)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    files = sorted(glob.glob(a.inputs))
    out = {k: [] for k in COLS}
    for f in files:
        try:
            d = np.load(f)
        except Exception as e:                                # noqa: BLE001
            print(f"  skip {f}: {e}")
            continue
        for k in COLS:
            out[k].append(d[k])
    g = {k: np.concatenate(v) for k, v in out.items()}
    n = len(g["m_pre"])

    # the Z rest frame: boost the two legs with the pre-FSR (Born) Z
    px = (g["ptp_pre"] * np.cos(g["phip_pre"])
          + g["ptm_pre"] * np.cos(g["phim_pre"]))
    py = (g["ptp_pre"] * np.sin(g["phip_pre"])
          + g["ptm_pre"] * np.sin(g["phim_pre"]))
    pz, en = g["pz_pre"], g["e_pre"]
    mm = np.sqrt(np.maximum(en**2 - px**2 - py**2 - pz**2, 1e-12))
    bx, by, bz, gm = px / en, py / en, pz / en, en / mm

    def estar(pt, eta, phi, e):
        ppx, ppy = pt * np.cos(phi), pt * np.sin(phi)
        ppz = pt * np.sinh(eta)
        bp = bx * ppx + by * ppy + bz * ppz
        return gm * (e - bp)

    res = {k: g[k].astype(np.float32) for k in COLS
           if k not in ("npre", "nph")}
    res["npre"] = g["npre"].astype(np.int8)
    res["nph"] = g["nph"].astype(np.int8)
    for q in ("p", "m"):
        e1 = estar(g[f"pt{q}_pre"], g[f"eta{q}_pre"], g[f"phi{q}_pre"],
                   g[f"e{q}_pre"])
        e2 = estar(g[f"pt{q}"], g[f"eta{q}"], g[f"phi{q}"], g[f"e{q}"])
        res[f"x{q}"] = (e2 / e1).astype(np.float64)
        res[f"xlab{q}"] = (g[f"pt{q}"] / g[f"pt{q}_pre"]).astype(np.float64)
    np.savez(a.output, **res)
    print(f"[merge_perleg] {len(files)} files, {n} events -> {a.output}")


if __name__ == "__main__":
    main()
