#!/usr/bin/env python3
"""Does the basis have to be re-derived when the sample / model changes?

Fits the joint (ALL-families) PCA basis on cache A and measures the
reconstruction error on cache B, for every ordered pair, against the
same-cache baseline (train on the first half, test on the second).

A basis that transfers between two samples with different kinematics is a
basis that does not have to be refitted every time the physics model behind
the exponents is retuned; one that does not transfer is a per-production
object that must be shipped with the cache.

Usage
  python3 transfer.py --out transfer.npz
"""
import argparse
import numpy as np
import cfbasis as CB
import spectral as SP

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress")
GROUPS = {"mass": (["jpsigun", "btojpsix"],
                   ["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"], True),
          "track": (["trk_lowpt", "trk_ul16"],
                    ["Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im"],
                    False)}
RL = [8, 16, 32, 48, 64]


def load(tag, fams, n0, n1, phik):
    d = np.load(f"{SCRATCH}/sub_{tag}_n60000_s1234.npz")
    TG = d["tgrid"].astype(np.float64)
    sl = slice(n0, n1)
    M = {f: d[f][sl].astype(np.float64) for f in fams}
    vgf = d["vgf"][sl].astype(np.float64)
    Sre = -0.5 * vgf[:, None] * TG[None, :] ** 2
    for f in ("Sms", "Sio_re", "Srad_re"):
        Sre = Sre + M[f]
    W = np.exp(Sre)
    if phik:
        W = W * SP.phik_abs(tag, TG, d["sigma"][sl].astype(np.float64))
    return TG, np.concatenate([M[f] for f in fams], axis=1), W, len(TG)


def err(X, W, mu, V, r, nfam, nt):
    R = (X - mu) - ((X - mu) @ V[:, :r]) @ V[:, :r].T
    aR = np.abs(R)
    e = np.zeros(len(X))
    for j in range(nfam):
        e = np.maximum(e, (aR[:, j * nt:(j + 1) * nt] * W).max(axis=1))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{SCRATCH}/transfer.npz")
    a = ap.parse_args()
    res = {}
    for gname, (tags, fams, phik) in GROUPS.items():
        tr = {t: load(t, fams, 0, 20000, phik) for t in tags}
        te = {t: load(t, fams, 20000, 40000, phik) for t in tags}
        B = {t: CB.pca_fit(tr[t][1], max(RL))[:2] for t in tags}
        print(f"== {gname} ==")
        print(f"{'train':10s} {'test':10s} " +
              " ".join(f"r={r:<3d}" for r in RL))
        for a_ in tags:
            for b_ in tags:
                mu, V = B[a_]
                TG, X, W, nt = te[b_]
                row = [np.quantile(err(X, W, mu, V, r, len(fams), nt), .999)
                       for r in RL]
                res[f"{gname}/{a_}->{b_}"] = np.array(row)
                mark = "  (same)" if a_ == b_ else ""
                print(f"{a_:10s} {b_:10s} " +
                      " ".join(f"{v:8.2e}" for v in row) + mark)
    np.savez(a.out, **res, ranks=np.array(RL))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
