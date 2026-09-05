#!/usr/bin/env python3
"""Cost model: storage and time per NLL evaluation, full vs compressed.

Measures, on one core, (a) the integrand cost per candidate with the exponents
already materialised, (b) the extra cost of reconstructing them from r
coefficients (a dense GEMM), and extrapolates to 1e7 candidates.
"""
import argparse, time
import numpy as np
import massnll_np as MN
import cfbasis as CB

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/cfcompress")
FAMS = ["Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jpsigun")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ranks", default="8,16,32,48")
    ap.add_argument("--rep", type=int, default=3)
    a = ap.parse_args()
    d = np.load(f"{SCRATCH}/sub_{a.tag}_n60000_s1234.npz")
    TG = d["tgrid"].astype(np.float64)
    n = a.n
    sig = d["sigma"][:n].astype(np.float64)
    mobs = d["z"][:n] * sig + (d["eta"][:n] - MN.MJPSI)
    vgf = d["vgf"][:n].astype(np.float64)
    M = {f: np.ascontiguousarray(d[f][:n]) for f in FAMS}
    pKre, pKim = MN.phik(a.tag, TG, sig)
    m = MN.Model(sig, mobs, vgf, M["Sms"], M["Sio_re"], M["Sio_im"],
                 M["Srad_re"], M["Srad_im"], pKre, pKim, TG, nproc=1)
    th = np.array([0.25, 0.94, 1.01, 0.69])

    def tm(fn, rep):
        t = []
        for _ in range(rep):
            t0 = time.time(); fn(); t.append(time.time() - t0)
        return min(t)

    tv = tm(lambda: m.nll_grad(th, want_grad=False), a.rep)
    tg = tm(lambda: m.nll_grad(th, want_grad=True), a.rep)
    print(f"n={n}  1 core: NLL {tv:.2f} s ({tv/n*1e6:.2f} us/cand), "
          f"NLL+grad {tg:.2f} s ({tg/n*1e6:.2f} us/cand)")
    print(f"  -> 1e7 candidates, 1 core: NLL {tv/n*1e7:.0f} s, "
          f"NLL+grad {tg/n*1e7:.0f} s;  16 cores: {tv/n*1e7/16:.0f} / "
          f"{tg/n*1e7/16:.0f} s")
    full_b = 5 * 448 * 4
    print(f"  storage full: {full_b} B/cand -> {full_b*1e7/1e9:.1f} GB for 1e7")
    X = np.concatenate([M[f].astype(np.float64) for f in FAMS], axis=1)
    mu, V, sv = CB.pca_fit(X, 64)
    C = (X - mu) @ V
    for r in [int(x) for x in a.ranks.split(",")]:
        Cr = np.ascontiguousarray(C[:, :r].astype(np.float32))
        Vr = np.ascontiguousarray(V[:, :r].T)          # (r, 2240)
        trec = tm(lambda: (Cr.astype(np.float64) @ Vr) + mu, a.rep)
        nb = 4 * r
        print(f"  r={r:3d}: {nb:4d} B/cand ({full_b/nb:5.1f}x, "
              f"{nb*1e7/1e9:.2f} GB for 1e7); reconstruction "
              f"{trec:.2f} s ({trec/n*1e6:.2f} us/cand, "
              f"{trec/tv*100:.0f} % of the NLL) -> "
              f"1e7 on 16 cores {trec/n*1e7/16:.0f} s")


if __name__ == "__main__":
    main()
