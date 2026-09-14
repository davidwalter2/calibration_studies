#!/usr/bin/env python3
"""Project the generator sample onto the same band/``u`` binning as `photos_gen`.

Produces ``data/photos/mcref.npz`` with exactly the arrays the C++ driver
writes, so the sample and any standalone run are compared bin for bin and the
same kernel builder runs on either.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fit_gen as FG                                            # noqa: E402
from prep_input import default_bands                            # noqa: E402

U_FINE = 2e-5           # fine bin width, = fit_gen.build_kernel's u_fine
N_FINE = 100_000        # 0 <= u < 2
U_TAIL = 0.05           # coarse bins above u = 2
N_TAIL = 200            # 2 <= u < 12
N_LOG = 160             # log-spaced 1e-6 .. 2 for the density figures
LOG_LO, LOG_HI = 1e-6, 2.0
NORAD_U = 1e-5          # fit_gen.build_kernel's float-noise floor


def log_edges():
    return np.geomspace(LOG_LO, LOG_HI, N_LOG + 1)


def accumulate(m_pre, u, w, nph, bands):
    nb = len(bands)
    lo = np.array([b[0] for b in bands])
    hi = np.array([b[1] for b in bands])
    ib = np.clip(np.searchsorted(lo, m_pre, "right") - 1, 0, nb - 1)
    ib = np.where((m_pre >= lo[ib]) & (m_pre < hi[ib]), ib, -1)
    ok = ib >= 0
    ib, m_pre, u, w, nph = ib[ok], m_pre[ok], u[ok], w[ok], nph[ok]

    out = {}
    out["n"] = np.bincount(ib, w, nb)
    out["n_nophot"] = np.bincount(ib, w * (nph == 0), nb)
    out["n_norad"] = np.bincount(ib, w * (u <= NORAD_U), nb)
    out["mpre_s1"] = np.bincount(ib, w * m_pre, nb)
    x = -np.expm1(-2.0 * u)                                   # 1 - z
    out["mom"] = np.stack([np.bincount(ib, w, nb),
                           np.bincount(ib, w * u, nb),
                           np.bincount(ib, w * u * u, nb),
                           np.bincount(ib, w * x, nb)], 1)
    out["w2"] = np.bincount(ib, w * w, nb)

    def h2(idx, mask, nbin, wts):
        j = ib[mask] * nbin + idx[mask]
        return np.bincount(j, wts[mask], nb * nbin).reshape(nb, nbin)

    jf = np.floor(u / U_FINE).astype(np.int64)
    mf = (jf >= 0) & (jf < N_FINE)
    out["fine_s0"] = h2(jf, mf, N_FINE, w)
    out["fine_s1"] = h2(jf, mf, N_FINE, w * u)
    out["fine_s2"] = h2(jf, mf, N_FINE, w * u * u)

    jt = np.floor((u - 2.0) / U_TAIL).astype(np.int64)
    mt = (u >= 2.0) & (jt < N_TAIL)
    out["tail_s0"] = h2(jt, mt, N_TAIL, w)
    out["tail_s1"] = h2(jt, mt, N_TAIL, w * u)
    out["tail_s2"] = h2(jt, mt, N_TAIL, w * u * u)
    out["over"] = np.bincount(ib[u >= 2.0 + N_TAIL * U_TAIL],
                              w[u >= 2.0 + N_TAIL * U_TAIL], nb)

    e = log_edges()
    jl = np.clip(np.searchsorted(e, u, "right") - 1, -1, N_LOG)
    ml = (u >= e[0]) & (u < e[-1])
    out["logu_h"] = h2(jl, ml, N_LOG, w)

    out["bands_lo"] = lo
    out["bands_hi"] = hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/genmerged_full.npz")
    ap.add_argument("-o", "--output", default="data/photos/mcref.npz")
    ap.add_argument("--wclip", type=float, default=100.0)
    ap.add_argument("--half", action="store_true",
                    help="also store the two half-sample accumulations, for errors")
    a = ap.parse_args()

    g = FG.load_gen(a.gen, ["m_pre", "m_post", "weight", "npre", "nph"])
    m = g["m_pre"].astype(np.float64)
    w = FG.clip_weights(g["weight"].astype(np.float64), a.wclip)
    r = np.minimum(g["m_post"].astype(np.float64) / m, 1.0)
    u = -np.log(r)
    # the sample's own "Photos emitted nothing" flag: no status-746 pre-Photos
    # muon copy.  ``nph`` counts the status-1 photons kept in the record.
    nph = g["npre"].astype(np.int32)

    bands = default_bands()
    out = accumulate(m, u, w, nph, bands)
    if a.half:
        h = np.arange(len(m)) < len(m) // 2
        for tag, sel in (("A", h), ("B", ~h)):
            o = accumulate(m[sel], u[sel], w[sel], nph[sel], bands)
            for k in ("n", "n_nophot", "n_norad", "mom", "mpre_s1",
                      "fine_s0", "fine_s1", "fine_s2", "tail_s0", "over"):
                out[f"half{tag}_{k}"] = o[k]
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    np.savez_compressed(a.output, **out)
    tot = out["n"].sum()
    print(f"{a.output}: {len(bands)} bands, sumw = {tot:.6e}")
    print(f"  inclusive <u> = {out['mom'][:,1].sum()/tot:.6e}, "
          f"P(no photon) = {out['n_nophot'].sum()/tot:.6f}, "
          f"P(u<={NORAD_U:g}) = {out['n_norad'].sum()/tot:.6f}")


if __name__ == "__main__":
    main()
